import socket
import json
import uuid
import os
import threading

# Configuration
CONFIG_FILE = 'plug.cf'

config = {
    'orchestrator_host': 'localhost',
    'orchestrator_ports': '6000-6010',
    'port_range': '6200-6300',
    'script_uuid': str(uuid.uuid4()),
    'script_name': 'PLUG',
    'data_port': None
}

registered = False
cancel_event = threading.Event()

def read_config():
    global config
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                config.update(json.load(f))
        except Exception as e:
            print(f"[Error] Could not read {CONFIG_FILE}: {e}")
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)

def parse_port_range(port_range_str):
    ports = []
    for part in port_range_str.split(','):
        if '-' in part:
            start, end = map(int, part.split('-'))
            ports.extend(range(start, end + 1))
        else:
            ports.append(int(part))
    return ports

def find_available_port(port_range):
    for port in port_range:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            result = sock.connect_ex(('localhost', port))
            if result != 0:
                return port
    raise Exception("No available ports found in range.")

def register_with_orchestrator():
    host = config['orchestrator_host']
    orch_ports = parse_port_range(config['orchestrator_ports'])
    port_range = parse_port_range(config['port_range'])

    for port in orch_ports:
        try:
            print(f"Attempting registration on {host}:{port}...")
            with socket.create_connection((host, port), timeout=5) as s:
                assigned_data_port = find_available_port(port_range)
                config['data_port'] = assigned_data_port
                message = f"/register {config['script_name']} {config['script_uuid']} {assigned_data_port}\n"
                s.sendall(message.encode())
                
                # Wait for acknowledgment
                ack_data = s.recv(1024).decode().strip()
                print(f"[Orchestrator Response] {ack_data}")
                if ack_data.startswith('/ack'):
                    print(f"[Success] Registered with orchestrator. Confirmed data port: {assigned_data_port}")
                    return True
        except Exception as e:
            print(f"[Error] Registration failed on port {port}: {e}")
    print("[Error] Could not register with any orchestrator ports.")
    return False

def start_data_listener():
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
                        data_port = find_available_port(port_range)
                        config['data_port'] = data_port
                        update_orchestrator_data_port(data_port)
                    else:
                        print(f"[Error] Could not start data listener on port {data_port}: {e}")
                        break
        except Exception as e:
            print(f"[Error] Unexpected error in data listener setup: {e}")

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
                response = f"{data}"
                send_data_to_orchestrator(response)  # Send acknowledgment
            else:
                print(f"[Info] Received acknowledgment message: {data}")

def send_info(client_socket):
    info = {
        "script_name": config['script_name'],
        "script_uuid": config['script_uuid'],
        "data_port": config['data_port']
    }
    info_message = json.dumps(info)
    client_socket.sendall(info_message.encode())
    print(f"[Info Sent] Configuration info sent to orchestrator.")

def send_data_to_orchestrator(data):
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
    while not cancel_event.is_set():
        user_input = input("Enter message for orchestrator: ")
        if user_input.lower() == 'exit':
            cancel_event.set()
            break
        send_data_to_orchestrator(user_input)

if __name__ == '__main__':
    read_config()

    if register_with_orchestrator():
        listener_thread = threading.Thread(target=start_data_listener, daemon=True)
        listener_thread.start()
        handle_user_input()
        listener_thread.join()
    else:
        print("[Error] Could not complete registration. Exiting.")
