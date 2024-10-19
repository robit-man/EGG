import threading
import time
import socket
import json
import uuid
import os
import argparse

# Global Variables
CONFIG_FILE = 'asr.cf'
script_uuid = None  # Initialize as None for UUID persistence
script_name = 'Terminal_Input_Engine'  # Updated peripheral name
config = {}

# Default configuration
default_config = {
    'port': '6200',                # Starting port to listen on
    'port_range': '6200-6300',     # Range of ports to try if initial port is unavailable
    'orchestrator_host': 'localhost',  # Orchestrator's host
    'orchestrator_port': '6000',       # Orchestrator's command port
    'script_uuid': '',             # Initialize as empty; will be set in read_config()
}

# Threading and queues
cancel_event = threading.Event()

# Locks for configuration
config_lock = threading.Lock()

def read_config():
    global config, script_uuid
    config = default_config.copy()
    
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                config.update(json.load(f))  # Load the JSON file and update config
        except json.JSONDecodeError:
            print(f"[Error] Could not parse {CONFIG_FILE}. Using default configuration.")
    else:
        # If CONFIG_FILE does not exist, generate a new UUID and create the config file
        script_uuid = str(uuid.uuid4())
        config['script_uuid'] = script_uuid  # Add UUID to config
        write_config()  # Write the initial config with the new UUID
        print(f"[Info] Generated new UUID: {script_uuid} and created {CONFIG_FILE}")
    
    # Ensure UUID is handled
    if 'script_uuid' in config and config['script_uuid']:
        script_uuid = config['script_uuid']
    else:
        # Generate and update UUID if missing
        script_uuid = str(uuid.uuid4())
        config['script_uuid'] = script_uuid
        write_config()
        print(f"[Info] Generated new UUID: {script_uuid} and updated {CONFIG_FILE}")
    
    # Debug: Print the loaded configuration
    print("[Debug] Configuration Loaded:")
    for k, v in config.items():
        if k == 'system_prompt':
            print(f"{k}={'[REDACTED]'}")  # Hide system_prompt in debug
        else:
            print(f"{k}={v}")
    return config

def write_config():
    with config_lock:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)  # Write the configuration as JSON
        print(f"[Info] Configuration written to {CONFIG_FILE}")
        
def parse_args():
    parser = argparse.ArgumentParser(
        description="Terminal Input Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--port",
        type=str,
        default=None,
        help="Starting port number to listen on."
    )
    parser.add_argument(
        "--port-range",
        type=str,
        default=None,
        help="Range of ports to try if the initial port is unavailable (e.g., '6200-6300')."
    )
    parser.add_argument(
        "--orchestrator-host",
        type=str,
        default=None,
        help="Orchestrator's host address."
    )
    parser.add_argument(
        "--orchestrator-port",
        type=str,
        default=None,
        help="Orchestrator's command port."
    )
    args = parser.parse_args()

    # Update config with args if provided
    for key in config.keys():
        arg_name = key.replace('-', '_')
        arg_value = getattr(args, arg_name, None)
        if arg_value is not None:
            config[key] = str(arg_value)
    write_config()
    return args

def send_text_to_orchestrator(recognized_text):
    host = config.get('orchestrator_host', 'localhost')
    port = int(config.get('orchestrator_port', '6000'))
    print(f"Sending to orchestrator: {recognized_text}")
    message = f"/data {script_uuid} {recognized_text}\n"
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((host, port))
        s.sendall(message.encode())
        s.close()
    except Exception as e:
        print(f"Failed to send text to orchestrator at {host}:{port}: {e}")
        # Optionally retry or handle the error

def terminal_input_processing():
    print("Enter text and press Enter to send. Type '/exit' to quit.")
    while not cancel_event.is_set():
        try:
            user_input = input("> ")
            if user_input.strip().lower() == '/exit':
                print("Exiting Terminal Input Engine.")
                send_exit_command()
                cancel_event.set()
                break
            # Send the input as recognized text
            send_text_to_orchestrator(user_input)
        except EOFError:
            # Handle Ctrl+D or end of input
            print("EOF received. Exiting.")
            send_exit_command()
            cancel_event.set()
            break
        except Exception as e:
            print(f"Error reading input: {e}")
            continue

def send_exit_command():
    host = config.get('orchestrator_host', 'localhost')
    port = int(config.get('orchestrator_port', '6000'))
    message = f"/exit {script_uuid}\n"
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((host, port))
        s.sendall(message.encode())
        s.close()
        print(f"Sent exit command to orchestrator at {host}:{port}")
    except Exception as e:
        print(f"Failed to send exit command to orchestrator at {host}:{port}: {e}")

def start_server():
    host = '0.0.0.0'
    port_range = config.get('port_range', '6200-6300')
    port_list = parse_port_range(port_range)
    server_socket = None

    for port in port_list:
        try:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.bind((host, port))
            print(f"Terminal Input Engine listening on port {port}...")
            config['port'] = str(port)  # Update the port in the config
            write_config()
            break
        except OSError:
            print(f"Port {port} is unavailable, trying next port...")
            server_socket.close()
            server_socket = None

    if not server_socket:
        print("Failed to bind to any port in the specified range.")
        return

    # Register with orchestrator
    register_with_orchestrator()

    server_socket.listen(5)
    while not cancel_event.is_set():
        try:
            client_socket, addr = server_socket.accept()
            threading.Thread(target=handle_client_socket, args=(client_socket,), daemon=True).start()
        except socket.error as e:
            if not cancel_event.is_set():
                print(f"Socket error: {e}")
            break
    server_socket.close()
    print("Server socket closed.")

def parse_port_range(port_range_str):
    ports = []
    for part in port_range_str.split(','):
        if '-' in part:
            try:
                start, end = map(int, part.split('-'))
                ports.extend(range(start, end + 1))
            except ValueError:
                print(f"[Error] Invalid port range format: '{part}'. Skipping.")
        else:
            try:
                ports.append(int(part))
            except ValueError:
                print(f"[Error] Invalid port number: '{part}'. Skipping.")
    return ports

def register_with_orchestrator():
    host = config.get('orchestrator_host', 'localhost')
    port = int(config.get('orchestrator_port', '6000'))
    message = f"/register {script_name} {script_uuid} {config['port']}\n"
    max_retries = 5
    retry_delay = 1
    for attempt in range(max_retries):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((host, port))
            s.sendall(message.encode())
            s.close()
            print(f"Registered with orchestrator at {host}:{port}")
            return
        except Exception as e:
            print(f"Failed to register with orchestrator: {e}")
            time.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff
    print("Max retries reached. Could not register with orchestrator.")

def handle_client_socket(client_socket):
    with client_socket:
        client_socket.settimeout(5)  # Increase timeout to 5 seconds
        buffer = ''
        while not cancel_event.is_set():
            try:
                data = client_socket.recv(1024)
                if not data:
                    break
                buffer += data.decode()
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    command = line.strip()
                    if command == '/info':
                        response = f"{script_name}\n{script_uuid}\n"
                        with config_lock:
                            for key, value in config.items():
                                response += f"{key}={value}\n"
                        response += 'EOF\n'
                        client_socket.sendall(response.encode())
                        client_socket.shutdown(socket.SHUT_WR)  # Ensure data is flushed
                    elif command.startswith('/exit'):
                        response = "Exiting Terminal Input Engine.\n"
                        client_socket.sendall(response.encode())
                        cancel_event.set()
                        return
                    else:
                        client_socket.sendall(b"Unknown command.\n")
            except socket.timeout:
                continue
            except Exception as e:
                print(f"Error in handle_client_socket: {e}")
                break

def main():
    read_config()
    parse_args()

    # Start server to handle /info and /exit requests and registration
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()

    # Start terminal input processing
    try:
        terminal_input_processing()
    finally:
        cancel_event.set()
        server_thread.join()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTerminal Input Engine terminated by user.")
