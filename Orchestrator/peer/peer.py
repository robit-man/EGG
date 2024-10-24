import os
import json
import threading
import time
import socket
import random
import string
import subprocess
import uuid
import traceback
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.backends import default_backend
from stem.control import Controller
import socks

ADDRESS_BOOK_FILE = "address_book.json"
NODE_CONFIG_FILE = "node_config.json"
CONFIG_FILE = "peer.cf"
SOCKS_PORT = 9050
TOR_CONTROL_PORT = 9051

# Default configuration
default_config = {
    'orchestrator_host': 'localhost',
    'orchestrator_ports': '6000-6010',
    'port_range': '6200-6300',
    'node_name': '',  # This will be set during runtime
    'script_uuid': '',
}

config = {}
registered = False
address_book = {}
latency_metrics = {}
cancel_event = threading.Event()  # For graceful shutdown

# UUID and name for the peripheral
script_uuid = None
script_name = 'TOR_Engine'  # Set a unique name for your script

# Ensure hidden service directory ownership and permissions
def ensure_hidden_service_permissions(hidden_service_dir):
    if not os.path.exists(hidden_service_dir):
        os.makedirs(hidden_service_dir, mode=0o700)
    subprocess.run(["sudo", "chown", "-R", "debian-tor:debian-tor", hidden_service_dir])
    subprocess.run(["sudo", "chmod", "0700", hidden_service_dir])

# Setup or load hidden service
def setup_hidden_service(node_name, port=8080):
    hidden_service_dir = f"/var/lib/tor/{node_name}_hidden_service/"
    try:
        with Controller.from_port(port=TOR_CONTROL_PORT) as controller:
            controller.authenticate()
            ensure_hidden_service_permissions(hidden_service_dir)
            if not os.path.exists(os.path.join(hidden_service_dir, "hostname")):
                result = controller.create_hidden_service(hidden_service_dir, port, target_port=port)
                onion_address = result.hostname
                print(f"New persistent hidden service created for {node_name} at {onion_address}")
            else:
                with open(os.path.join(hidden_service_dir, "hostname"), "r") as f:
                    onion_address = f.read().strip()
                print(f"Existing hidden service loaded for {node_name} at {onion_address}")
            return onion_address
    except Exception as e:
        print(f"Error setting up hidden service: {e}")
        return None

# Generate RSA keys
def generate_rsa_keypair():
    private_key = rsa.generate_private_key(
        public_exponent=65537, key_size=2048, backend=default_backend()
    )
    public_key = private_key.public_key()
    return private_key, public_key

# Save RSA keys to file
def save_keypair(private_key, public_key, node_name):
    with open(f'{node_name}_private_key.pem', 'wb') as f:
        f.write(
            private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )
    with open(f'{node_name}_public_key.pem', 'wb') as f:
        f.write(
            public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
        )

# Load private RSA key from file
def load_private_key(node_name):
    with open(f'{node_name}_private_key.pem', 'rb') as f:
        return serialization.load_pem_private_key(f.read(), password=None, backend=default_backend())

# Read configuration from a JSON file
def read_config():
    global config, script_uuid
    config = default_config.copy()
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                config.update(json.load(f))
            print(f"[Info] Configuration loaded from {CONFIG_FILE}")
        except (json.JSONDecodeError, FileNotFoundError):
            print(f"[Error] Could not read {CONFIG_FILE}. Using default configuration.")
    if 'script_uuid' in config and config['script_uuid']:
        script_uuid = config['script_uuid']
    else:
        script_uuid = str(uuid.uuid4())
        config['script_uuid'] = script_uuid
        write_config()  # Save the newly generated UUID

# Write configuration to a JSON file
def write_config():
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"[Info] Configuration saved to {CONFIG_FILE}")
    except Exception as e:
        print(f"[Error] Writing configuration failed: {e}")

# Save the address book to a file
def save_address_book():
    try:
        with open(ADDRESS_BOOK_FILE, 'w') as f:
            json.dump(address_book, f, indent=4)
        print("Address book saved successfully.")
    except Exception as e:
        print(f"Error saving address book: {e}")

# Load the address book from a file
def load_address_book():
    global address_book
    if os.path.exists(ADDRESS_BOOK_FILE):
        try:
            with open(ADDRESS_BOOK_FILE, 'r') as f:
                address_book = json.load(f)
                if not address_book:
                    raise ValueError("Address book is empty.")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Corrupted or invalid address book: {e}. Re-initializing...")
            address_book = {}
            save_address_book()
    else:
        print(f"Address book not found. Creating new one...")
        address_book = {}
        save_address_book()

# Save node name to a configuration file
def save_node_name(node_name):
    try:
        with open(NODE_CONFIG_FILE, 'w') as f:
            json.dump({"node_name": node_name}, f)
        print(f"Node name {node_name} saved to {NODE_CONFIG_FILE}.")
    except Exception as e:
        print(f"Error saving node name: {e}")

# Load node name from the configuration file
def load_node_name():
    if os.path.exists(NODE_CONFIG_FILE):
        try:
            with open(NODE_CONFIG_FILE, 'r') as f:
                node_config = json.load(f)
                return node_config.get("node_name")
        except Exception as e:
            print(f"Error loading node name: {e}")
    return None

# Function to parse port range
def parse_port_range(port_range_str):
    ports = []
    for part in port_range_str.split(','):
        if '-' in part:
            try:
                start, end = map(int, part.split('-'))
                ports.extend(range(start, end + 1))
            except ValueError:
                print(f"[Warning] Invalid port range format: {part}")
        else:
            try:
                ports.append(int(part))
            except ValueError:
                print(f"[Warning] Invalid port number: {part}")
    return ports

# Register with the local orchestrator
def register_with_orchestrator(port):
    global registered
    if registered:
        print("[Info] Already registered with the orchestrator.")
        return True

    host = config.get('orchestrator_host', 'localhost')
    orchestrator_ports_str = config.get('orchestrator_ports', '6000-6005')
    orchestrator_command_ports = parse_port_range(orchestrator_ports_str)

    message = f"/register {script_name} {script_uuid} {port}\n"
    max_retries = 5
    retry_delay = 1  # Start with 1 second delay for retries
    backoff_factor = 2  # Exponential backoff factor

    for attempt in range(max_retries):
        for orch_port in orchestrator_command_ports:
            try:
                print(f"[Attempt {attempt + 1}] Registering with orchestrator at {host}:{orch_port}...")

                # Open a connection to the orchestrator
                with socket.create_connection((host, orch_port), timeout=5) as s:
                    s.sendall(message.encode())
                    print(f"[Info] Sent registration message: {message.strip()}")

                    # Receive acknowledgment
                    data = s.recv(1024)
                    if data:
                        ack_message = data.decode().strip()
                        print(f"[Info] Received acknowledgment: {ack_message}")

                        # Check if the acknowledgment is valid
                        if ack_message.startswith('/ack'):
                            config['data_port'] = ack_message.split()[1]
                            write_config()  # Save updated config with assigned data_port
                            registered = True
                            print(f"[Success] Registered with orchestrator on port {orch_port}.")
                            return True
                    else:
                        print(f"[Warning] No acknowledgment received from orchestrator at {host}:{orch_port}.")

            except socket.timeout:
                print(f"[Error] Timeout while connecting to orchestrator at {host}:{orch_port}.")
            except ConnectionRefusedError:
                print(f"[Error] Connection refused by orchestrator at {host}:{orch_port}.")
            except Exception as e:
                print(f"[Error] Unexpected error during registration: {e}")
                traceback.print_exc()

        # Retry with exponential backoff
        print(f"Retrying registration in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
        time.sleep(retry_delay)
        retry_delay *= backoff_factor  # Exponential backoff

    # If all attempts failed
    print("[Error] Max retries reached. Could not register with orchestrator.")
    return False

# Rolling code mechanism for handshake
def generate_rolling_code():
    return ''.join(random.choices(string.ascii_letters + string.digits, k=32))

# Secure handshake using rolling code and RSA keys
def handshake_with_peer(onion_address, node_name, private_key, retry_interval=10):
    socks.set_default_proxy(socks.SOCKS5, "127.0.0.1", SOCKS_PORT)
    s = None
    try:
        s = socks.socksocket()
        print(f"Attempting to connect to {onion_address}")
        s.connect((onion_address, 8080))

        rolling_code = generate_rolling_code()
        signed_message = private_key.sign(
            rolling_code.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )

        # Send the signed rolling code
        s.sendall(signed_message)

        # Receive the response
        response = s.recv(1024)
        print(f"Handshake response from {onion_address}: {repr(response)}")
        return True

    except Exception as e:
        print(f"Failed to connect to {onion_address}: {e}. Retrying in {retry_interval} seconds...")
        time.sleep(retry_interval)
        return False

    finally:
        if s:
            s.close()

# Listen for incoming handshakes and messages
def listen_for_handshakes_and_messages(private_key, port=8080):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # Instead of binding to 127.0.0.1, bind to all interfaces or the hidden service address
    s.bind(('0.0.0.0', port))  # Binds to all available interfaces
    
    s.listen(5)
    print(f"[Info] Listening for incoming connections on port {port}")
    while not cancel_event.is_set():
        try:
            client_socket, addr = s.accept()
            threading.Thread(
                target=handle_client_connection, args=(client_socket, addr, private_key), daemon=True
            ).start()
        except Exception as e:
            print(f"[Error] Error accepting connections: {e}")
            traceback.print_exc()

    s.close()
    print("[Info] Listener shut down.")

def handle_client_connection(client_socket, addr, private_key):
    print(f"Connection from {addr}")
    try:
        data = client_socket.recv(1024).decode('utf-8')
        print(f"Received raw data: {repr(data)}")

        # Handle remote message (sent by peer) 
        if data.startswith("[remote]"):
            print("Message received from a remote peer.")
            forward_to_orchestrator(data[8:])  # Strip the "[remote]" header and forward to orchestrator
        
        # Handle message from orchestrator (local message)
        else:
            print("Message received from orchestrator.")
            # Forward the message to a peer in the address book, skipping the local onion address
            peer_name = get_peer_name_to_forward()
            if peer_name:
                forward_to_peer(peer_name, f"{data}")  # Add "[remote]" header and forward to peer
            else:
                print("No valid peers to forward the message.")
                
        client_socket.sendall(b'Message received')
    except Exception as e:
        print(f"Error receiving data from {addr}: {e}")
    finally:
        client_socket.close()

# Get a valid peer name to forward, skipping the local onion address
def get_peer_name_to_forward():
    local_onion_address = config.get('onion_address')
    for peer_name, peer_info in address_book.items():
        if peer_info['onion_address'] != local_onion_address:
            return peer_name
    return None

# Forward message to peer
def forward_to_peer(peer_name, message):
    """Forwards the message to a peer, adding the [remote] header."""
    if peer_name in address_book:
        print(f"Forwarding message to peer {peer_name}")
        send_message_to_peer(peer_name, message)
    else:
        print(f"Peer {peer_name} not found in address book.")
        
# Send message to a peer
def send_message_to_peer(peer_name, message):
    if peer_name in address_book:
        peer_onion_address = address_book[peer_name]["onion_address"]
        try:
            # Set up SOCKS5 proxy via Tor
            socks.set_default_proxy(socks.SOCKS5, "127.0.0.1", SOCKS_PORT)
            s = socks.socksocket()

            # Connect to the peer's .onion address
            s.connect((peer_onion_address, 8080))
            print(f"Sending message to {peer_name} at {peer_onion_address}")

            # Send the message
            s.sendall(f"[remote]{message}".encode('utf-8'))

            # Wait for acknowledgment
            response = s.recv(1024)
            print(f"Response from {peer_name}: {response.decode('utf-8')}")

        except Exception as e:
            print(f"Failed to send message to {peer_name}: {e}")

        finally:
            if s:
                s.close()
    else:
        print(f"Peer {peer_name} not found in address book.")

# Forward message to orchestrator
def forward_to_orchestrator(message):
    host = config.get('orchestrator_host', 'localhost')
    port = int(config.get('orchestrator_port', '6000'))  # Use port 6000 for sending data
    try:
        message_with_header = f"/data {script_uuid} {message}\n"  # Adding the required /data header
        with socket.create_connection((host, port), timeout=5) as s:
            s.sendall(message_with_header.encode())
            print(f"Forwarded message to orchestrator: {message_with_header}")
    except Exception as e:
        print(f"Error forwarding message to orchestrator: {e}")

# Command handler for /add, /peers, /metrics, /send
def handle_commands():
    while not cancel_event.is_set():
        command = input("Enter command (/add, /peers, /metrics, /send): ").strip()
        if command.startswith("/add"):
            try:
                _, peer_name, peer_onion = command.split()
                address_book[peer_name] = {
                    "onion_address": peer_onion,
                    "public_key": ""
                }
                save_address_book()
                print(f"Peer {peer_name} with address {peer_onion} added successfully.")
            except ValueError:
                print("Usage: /add <peer_name> <peer_onion_address>")
        elif command == "/peers":
            print("Peers in address book:")
            for peer_name, peer_info in address_book.items():
                print(f"{peer_name}: {peer_info['onion_address']}")
        elif command == "/metrics":
            print("Latency Metrics:")
            for peer, latency in latency_metrics.items():
                print(f"{peer}: {latency:.2f} ms")
        elif command.startswith("/send"):
            try:
                _, peer_name, *payload = command.split()
                message = " ".join(payload)
                send_message_to_peer(peer_name, message)
            except ValueError:
                print("Usage: /send <peer_name> <message>")
        else:
            print("Unknown command. Available commands: /add, /peers, /metrics, /send")

# Start the node service
def start_node_service(node_name):
    load_address_book()
    private_key = load_private_key(node_name)
    onion_address = setup_hidden_service(node_name)
    if onion_address is None:
        print("Failed to set up hidden service. Exiting.")
        return

    # Update configuration with node_name and onion_address
    config['node_name'] = node_name
    config['onion_address'] = onion_address
    write_config()

    address_book[node_name] = {
        "onion_address": onion_address,
        "public_key": private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode('utf-8')
    }
    save_address_book()

    listener_thread = threading.Thread(
        target=listen_for_handshakes_and_messages, args=(private_key,)
    )
    listener_thread.start()

    command_thread = threading.Thread(target=handle_commands)
    command_thread.start()

    # Register with the orchestrator using the correct port
    if not register_with_orchestrator(8080):
        print("Failed to register with the orchestrator.")
        cancel_event.set()  # Ensure the script shuts down gracefully
        return  # Exit if registration fails

    while not cancel_event.is_set():
        for peer_name, peer_info in address_book.items():
            if peer_info['onion_address'] != onion_address:
                handshake_with_peer(peer_info['onion_address'], node_name, private_key)
        time.sleep(60)

    # Wait for threads to finish
    listener_thread.join()
    command_thread.join()

# Entry point
def main():
    try:
        read_config()
        node_name = load_node_name()
        if not node_name:
            node_name = input("Enter your node name (e.g., egg_nodal_0001): ")
            save_node_name(node_name)
        if not os.path.exists(f'{node_name}_private_key.pem'):
            private_key, public_key = generate_rsa_keypair()
            save_keypair(private_key, public_key, node_name)
        start_node_service(node_name)
    except KeyboardInterrupt:
        print("\n[Shutdown] Node service terminated by user.")
        cancel_event.set()
    except Exception as e:
        print(f"[Fatal Error] An unexpected error occurred: {e}")
        traceback.print_exc()
        cancel_event.set()

if __name__ == "__main__":
    main()
