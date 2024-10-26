import socket
import json
import uuid
import os
import threading
import random
import traceback

# Configuration
CONFIG_FILE = 'plug.cf'

# Initial configuration dictionary with default values for the script
config = {
    'orchestrator_host': 'localhost',          # Host address of the orchestrator
    'orchestrator_ports': '6000-6010',         # Range of ports for orchestrator connection attempts
    'port_range': '6200-6300',                 # Range of ports to try for the data listener
    'script_uuid': str(uuid.uuid4()),          # Unique identifier for this script instance
    'script_name': 'PLUG_Engine',              # Name of the script
    'data_port': None,                         # Port for data communication; assigned dynamically
    'forward': True                            # Determines if data should be forwarded
}

script_uuid = None            # Global variable for the script's unique identifier
script_name = 'PLUG_Engine'   # Global variable for the script's name
registered = False            # Tracks registration status with the orchestrator
cancel_event = threading.Event()  # Event to signal thread termination
config_lock = threading.Lock()    # Lock to ensure thread-safe access to config

def read_config():
    """
    Reads configuration from the config file if it exists, updating the global config dictionary.
    If the config file does not exist, writes the current config dictionary to the file.
    """
    global config
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                config.update(json.load(f))
            print(f"[Info] Loaded configuration from {CONFIG_FILE}.")
        except Exception as e:
            print(f"[Error] Could not read {CONFIG_FILE}: {e}")
    # Save the current config to ensure any defaults are written if not present
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)


def parse_port_range(port_range_str):
    """
    Parses a port range string and returns a list of ports.
    Supports single ports and ranges (e.g., '6000-6010').
    """
    ports = []
    for part in port_range_str.split(','):
        if '-' in part:
            start, end = map(int, part.split('-'))
            ports.extend(range(start, end + 1))
        else:
            ports.append(int(part))
    return ports

def find_available_port(port_range, preferred_port=None):
    """
    Attempts to find an available port for the data listener.
    Tries the preferred port first, if specified; otherwise, picks a random port from the range.
    Raises an exception if no available port is found.
    """
    # Try the preferred (stored) port first if provided
    if preferred_port:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            result = sock.connect_ex(('localhost', preferred_port))
            if result != 0:  # Port is available
                return preferred_port
    
    # If the preferred port is unavailable, pick a random available port from the range
    tried_ports = set()
    while len(tried_ports) < len(port_range):
        port = random.choice(port_range)
        if port in tried_ports:
            continue
        tried_ports.add(port)
        
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            result = sock.connect_ex(('localhost', port))
            if result != 0:
                return port

    raise Exception("No available ports found in range.")

def register_with_orchestrator():
    """
    Attempts to register with the orchestrator by connecting to each port in the orchestrator range.
    If successful, assigns a confirmed data port and updates the config file with this information.
    """
    host = config['orchestrator_host']
    orch_ports = parse_port_range(config['orchestrator_ports'])
    port_range = parse_port_range(config['port_range'])
    
    # First, try to use the stored data_port from the config if it exists
    preferred_port = config.get('data_port')
    
    for port in orch_ports:
        try:
            print(f"Attempting registration on {host}:{port}...")
            with socket.create_connection((host, port), timeout=5) as s:
                # Use the stored port as the preferred port, only finding a new one if necessary
                assigned_data_port = find_available_port(port_range, preferred_port=preferred_port)
                config['data_port'] = assigned_data_port  # Update config with the confirmed data port
                
                message = f"/register {config['script_name']} {config['script_uuid']} {assigned_data_port}\n"
                s.sendall(message.encode())
                
                # Wait for acknowledgment
                ack_data = s.recv(1024).decode().strip()
                print(f"[Orchestrator Response] {ack_data}")
                if ack_data.startswith('/ack'):
                    print(f"[Success] Registered with orchestrator. Confirmed data port: {assigned_data_port}")
                    update_config()  # Save the selected port to the config file
                    return True
        except Exception as e:
            print(f"[Error] Registration failed on port {port}: {e}")
    print("[Error] Could not register with any orchestrator ports.")
    return False

def start_data_listener():
    """
    Starts a data listener on the assigned port to handle incoming data connections.
    Automatically attempts to rebind on a new port if the current port is in use.
    """
    host = '0.0.0.0'
    port_range = parse_port_range(config['port_range'])
    data_port = config.get('data_port')

    if not data_port:
        print("[Error] No data port assigned. Unable to start data listener.")
        return

    while not cancel_event.is_set():
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
                try:
                    server_socket.bind((host, data_port))
                    server_socket.listen(5)
                    print(f"[Info] Data listener started on port {data_port}.")
                    
                    while not cancel_event.is_set():
                        try:
                            server_socket.settimeout(1.0)
                            client_socket, addr = server_socket.accept()
                            print(f"[Connection] Data received from orchestrator at {addr}")
                            threading.Thread(target=handle_client_connection, args=(client_socket,), daemon=True).start()
                        except socket.timeout:
                            continue
                        except Exception as e:
                            print(f"[Error] Error in data listener: {e}")
                            break
                    return  # Listener started successfully; exit loop
                except OSError as e:
                    if "Address already in use" in str(e):
                        print(f"[Warning] Port {data_port} already in use. Selecting a new port...")
                        data_port = find_available_port(port_range, preferred_port=None)  # Skip the preferred port
                        config['data_port'] = data_port
                        update_config()  # Save new port to config
                        update_orchestrator_data_port(data_port)
                    else:
                        print(f"[Error] Could not start data listener on port {data_port}: {e}")
                        break
        except Exception as e:
            print(f"[Error] Unexpected error in data listener setup: {e}")

def update_config():
    """Updates the config file with the current data port."""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"[Info] Configuration updated with new data port: {config['data_port']}")


def update_orchestrator_data_port(new_data_port):
    """Informs the orchestrator of the new data port if the old one was in use."""
    host = config['orchestrator_host']
    orch_ports = parse_port_range(config['orchestrator_ports'])

    for port in orch_ports:
        try:
            with socket.create_connection((host, port), timeout=5) as s:
                message = f"/update_data_port {config['script_uuid']} {new_data_port}\n"
                s.sendall(message.encode())
                print(f"[Info] Notified orchestrator of new data port: {new_data_port}")
                return  # Exit after successful notification
        except Exception as e:
            print(f"[Error] Could not notify orchestrator of new data port on port {port}: {e}")

def handle_client_connection(client_socket):
    """
    Handles each incoming client connection, responding to '/info' requests
    and forwarding other data as required by the forward setting.
    """
    with client_socket:
        while not cancel_event.is_set():
            data = client_socket.recv(1024).decode().strip()
            if not data:
                break

            # Process each message as it comes
            if data.lower().startswith('/info'):
                send_info(client_socket)
            elif not data.startswith("Acknowledged:"):
                print(f"[Data Received] {data}")
                # This following action forwards data this module receives, on to the orchestrator, to be routed to the next module.
                forwarding = config['forward']
                if not forwarding:
                    print("Forwarding Disabled")
                else:
                    response = f"{data}"
                    send_data_to_orchestrator(response)  # Send acknowledgment
            else:
                print(f"[Info] Received acknowledgment message: {data}")

def send_info(client_socket):
    """
    Sends configuration and script information to the client socket in a specific format,
    ending with 'EOF' to signal completion.
    """
    try:
        response = f"{script_name}\n{script_uuid}\n"
        with config_lock:
            for key, value in config.items():
                response += f"{key}={value}\n"
        response += 'EOF\n'  # End the response with EOF to signal completion
        client_socket.sendall(response.encode())
        print(f"[Info] Sent configuration info to {client_socket.getpeername()}")
    except Exception as e:
        print(f"[Error] Failed to send info to {client_socket.getpeername()}: {e}")
        traceback.print_exc()


def send_data_to_orchestrator(data):
    """
    Forwards data to the orchestrator, iterating through orchestrator ports until a connection succeeds.
    Retries are attempted in case of connection refusal.
    """
    host = config.get('orchestrator_host', 'localhost')
    orch_ports = parse_port_range(config['orchestrator_ports'])
    message = f"/data {config['script_uuid']} {data}\n"
    
    for orch_port in orch_ports:
        try:
            # Establish a temporary socket connection
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((host, orch_port))
                s.sendall(message.encode())
                print(f"[Sent] Data sent to orchestrator on port {orch_port}: {data}")
            return  # Exit after successful send
        except ConnectionRefusedError:
            print(f"[Error] Connection refused by orchestrator at {host}:{orch_port}. Trying next port...")
        except Exception as e:
            print(f"[Error] Failed to send data to orchestrator on port {orch_port}: {e}")

    print("[Error] All attempts to send data to orchestrator failed.")


def handle_user_input():
    """
    Continuously reads user input from the terminal, forwarding each message to the orchestrator.
    Terminates if 'exit' is entered.
    """
    while not cancel_event.is_set():
        user_input = input("Enter message for orchestrator: ")
        if user_input.lower() == 'exit':
            cancel_event.set()
            break
        send_data_to_orchestrator(user_input)

# Main script execution: Reads config, registers with orchestrator, starts data listener, and handles user input
if __name__ == '__main__':
    read_config()

    if register_with_orchestrator():
        listener_thread = threading.Thread(target=start_data_listener, daemon=True)
        listener_thread.start()
        handle_user_input()
        listener_thread.join()
    else:
        print("[Error] Could not complete registration. Exiting.")
