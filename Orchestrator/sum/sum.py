import threading
import time
import socket
import json
import uuid
import os
import re
import argparse
import requests
import traceback

# Configuration file name
CONFIG_FILE = 'sum.cf'

# Default configuration
default_config = {
    'model_name': 'llama3.2:1b',
    'input_mode': 'port',           # Options: 'port', 'terminal', 'route'
    'output_mode': 'port',          # Options: 'port', 'terminal', 'route'
    'input_format': 'chunk',        # Options: 'streaming', 'chunk'
    'output_format': 'chunk',       # Options: 'streaming', 'chunk'
    'port_range': '6200-6300',
    'orchestrator_host': 'localhost',
    'orchestrator_ports': '6000-6010',  # Updated to handle multiple ports
    'route': '/sum',
    'script_uuid': '',              # Initialize as empty; will be set in read_config()
    'system_prompt': "You summarize what you recieve and perform additional analysis",
    'temperature': 0.7,             # Model parameter: temperature
    'top_p': 0.9,                   # Model parameter: top_p
    'max_tokens': 150,              # Model parameter: max tokens
    'repeat_penalty': 1.0,          # Model parameter: repeat penalty
    'inference_timeout': 5,         # Timeout in seconds for inference
    'json_filtering': False,        # Use JSON filtering by default
    'api_endpoint': 'generate'      # New option: 'generate' or 'chat'
}

config = {}

# UUID and name for the peripheral
script_uuid = None
script_name = 'SUM_Engine'

# Event to signal shutdown
cancel_event = threading.Event()

# Ollama API base URL
OLLAMA_URL = "http://localhost:11434/api/"

# Lock for thread-safe operations
config_lock = threading.Lock()

# Registration flag
registered = False

# Function to read configuration from a JSON file
def read_config():
    global config, script_uuid
    config = default_config.copy()  # Start with the default configuration

    # Read from JSON file if it exists
    if os.path.exists(CONFIG_FILE):
        print(f"Reading configuration from {CONFIG_FILE}")
        try:
            with open(CONFIG_FILE, 'r') as f:
                config.update(json.load(f))  # Update default config with the values from the file
        except (json.JSONDecodeError, FileNotFoundError):
            print(f"[Error] Could not read {CONFIG_FILE}. Using default configuration.")

    # Ensure script_uuid is set
    if 'script_uuid' in config and config['script_uuid']:
        script_uuid = config['script_uuid']
    else:
        script_uuid = str(uuid.uuid4())
        config['script_uuid'] = script_uuid
        write_config()  # Save updated config with the new UUID

    # Debug: Print the loaded configuration
    print("[Debug] Configuration Loaded:")
    for k, v in config.items():
        if k == 'system_prompt':
            print(f"{k}={'[REDACTED]'}")  # Hide system_prompt in debug
        else:
            print(f"{k}={v}")

    return config

# Function to write configuration to a JSON file
def write_config(config_to_write=None):
    config_to_write = config_to_write or config
    print(f"Writing configuration to {CONFIG_FILE}")
    with config_lock:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config_to_write, f, indent=4)

# Function to parse command-line arguments and update config
def parse_args():
    parser = argparse.ArgumentParser(description='SUM Engine Peripheral')
    parser.add_argument('--port-range', type=str, help='Port range to use for connections')
    parser.add_argument('--orchestrator-host', type=str, help='Orchestrator host address')
    parser.add_argument('--orchestrator-ports', type=str, help='Comma-separated list or range of orchestrator command ports')
    parser.add_argument('--model-name', type=str, help='Name of the language model to use')
    parser.add_argument('--system-prompt', type=str, help='System prompt for the model')
    parser.add_argument('--temperature', type=float, help='Model parameter: temperature')
    parser.add_argument('--top_p', type=float, help='Model parameter: top_p')
    parser.add_argument('--max_tokens', type=int, help='Model parameter: max tokens')
    parser.add_argument('--repeat_penalty', type=float, help='Model parameter: repeat penalty')
    parser.add_argument('--inference_timeout', type=int, help='Timeout in seconds for inference')
    parser.add_argument('--api-endpoint', type=str, choices=['generate', 'chat'], help='API endpoint to use: generate or chat')

    args = parser.parse_args()

    # Update config with command-line arguments
    if args.port_range:
        config['port_range'] = args.port_range
    if args.orchestrator_host:
        config['orchestrator_host'] = args.orchestrator_host
    if args.orchestrator_ports:
        config['orchestrator_ports'] = args.orchestrator_ports
    if args.model_name:
        config['model_name'] = args.model_name
    if args.system_prompt:
        config['system_prompt'] = args.system_prompt
    if args.temperature is not None:
        config['temperature'] = args.temperature
    if args.top_p is not None:
        config['top_p'] = args.top_p
    if args.max_tokens is not None:
        config['max_tokens'] = args.max_tokens
    if args.repeat_penalty is not None:
        config['repeat_penalty'] = args.repeat_penalty
    if args.inference_timeout is not None:
        config['inference_timeout'] = args.inference_timeout
    if args.api_endpoint:
        config['api_endpoint'] = args.api_endpoint

    write_config()
    return args

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

# Function to check if Ollama API is up
def check_ollama_api():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            print("[Info] Ollama API is up and running.")
            return True
        else:
            print(f"[Warning] Ollama API responded with status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Unable to connect to Ollama API.")
        return False
    except requests.exceptions.Timeout:
        print("⏰ Ollama API connection timed out.")
        return False
    except Exception as e:
        print(f"[Error] Unexpected error while checking Ollama API: {e}")
        return False

# Function to register with the orchestrator
def register_with_orchestrator(port):
    global registered
    if registered:
        print("[Info] Already registered with the orchestrator. Skipping registration.")
        return True
    host = config.get('orchestrator_host', 'localhost')
    orchestrator_ports_str = config.get('orchestrator_ports', '6000-6005')
    orchestrator_command_ports = []

    # Parse orchestrator_ports which can be a range like '6000-6005' or a list '6000,6001,6002'
    for port_entry in orchestrator_ports_str.split(','):
        port_entry = port_entry.strip()
        if '-' in port_entry:
            try:
                start_port, end_port = map(int, port_entry.split('-'))
                orchestrator_command_ports.extend(range(start_port, end_port + 1))
            except ValueError:
                print(f"Invalid orchestrator port range: {port_entry}")
        elif port_entry.isdigit():
            orchestrator_command_ports.append(int(port_entry))
        else:
            print(f"Invalid orchestrator port entry: {port_entry}")

    if not orchestrator_command_ports:
        print("[Error] No valid orchestrator command ports found.")
        return False

    message = f"/register {script_name} {script_uuid} {port}\n"
    max_retries = 5
    retry_delay = 1
    for attempt in range(max_retries):
        for orch_port in orchestrator_command_ports:
            try:
                print(f"[Attempt {attempt + 1}] Registering with orchestrator at {host}:{orch_port}...")
                with socket.create_connection((host, orch_port), timeout=5) as s:
                    s.sendall(message.encode())
                    print(f"[Info] Sent registration message: {message.strip()}")
                    # Receive acknowledgment
                    data = s.recv(1024)
                    if data:
                        ack_message = data.decode().strip()
                        print(f"[Info] Received acknowledgment: {ack_message}")
                        if ack_message.startswith('/ack'):
                            config['data_port'] = ack_message.split()[1]
                            write_config(config)
                            registered = True
                            return True
                    else:
                        print(f"[Warning] No acknowledgment received from orchestrator at {host}:{orch_port}.")
            except (socket.timeout, ConnectionRefusedError) as e:
                print(f"[Error] Connection to orchestrator at {host}:{orch_port} failed: {e}")
            except Exception as e:
                print(f"[Error] Unexpected error during registration: {e}")
                traceback.print_exc()
        print(f"Retrying registration in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
        time.sleep(retry_delay)
        retry_delay *= 2  # Exponential backoff

    print("[Error] Max retries reached. Could not register with orchestrator.")
    cancel_event.set()
    exit(1)  # Exit the script if registration fails


# Start server to handle incoming connections
def start_server():
    host = '0.0.0.0'
    port_list = parse_port_range(config.get('port_range', '6200-6300'))  # Parse port range from config
    
    try:
        # Dynamically find an available port in the specified range
        available_port = find_available_port(port_list)
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((host, available_port))
        print(f"[Info] SUM Engine listening on port {available_port}...")
        config['port'] = str(available_port)  # Update the port in the config
        write_config(config)  # Save the updated configuration with the selected port
    except Exception as e:
        print(f"[Error] Could not bind to any available port: {e}")
        cancel_event.set()  # Ensure the script shuts down gracefully
        return

    # Register with the orchestrator
    registration_successful = register_with_orchestrator(available_port)

    if not registration_successful and config['input_mode'] == 'terminal':
        # Fallback to terminal mode if orchestrator registration fails
        terminal_input()
        return
    elif not registration_successful:
        # Exit if not in terminal mode and registration fails
        return

    server_socket.listen(5)
    print(f"[Info] Server started on port {config['port']}. Waiting for connections...")

    while not cancel_event.is_set():
        try:
            server_socket.settimeout(1.0)  # Set a timeout to handle periodic cancellation
            client_socket, addr = server_socket.accept()
            print(f"[Connection] Connection from {addr}")
            threading.Thread(target=handle_client_socket, args=(client_socket,), daemon=True).start()
        except socket.timeout:
            continue  # Continue looping until a client connects or cancel_event is set
        except Exception as e:
            if not cancel_event.is_set():
                print(f"[Error] Error accepting connections: {e}")
                traceback.print_exc()

    server_socket.close()
    print("[Info] Server shut down.")


# Function to find an available port within the specified range
def find_available_port(port_range):
    """
    Scans the provided port range and returns the first available port.
    Raises an exception if no available ports are found.
    """
    for port in port_range:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            result = sock.connect_ex(('localhost', port))
            if result != 0:  # Port is available (no connection)
                return port
    raise Exception("No available ports found in range.")



# Handle incoming connections
def handle_client_socket(client_socket):
    with client_socket:
        client_socket.settimeout(1.0)  # Set lower timeout for data reception
        addr = client_socket.getpeername()
        print(f"[Connection] Handling data from {addr}")

        # Initialize timer variables
        data_lock = threading.Lock()
        data_buffer = []
        timer = None

        def inference_trigger():
            nonlocal data_buffer, timer
            with data_lock:
                if data_buffer:
                    user_input = ' '.join(data_buffer).strip()
                    if user_input:
                        print(f"[Inference] Timeout reached. Triggering inference for accumulated input: {user_input}")
                        # Run perform_inference in a separate thread
                        threading.Thread(target=perform_inference, args=(user_input,), daemon=True).start()
                        data_buffer.clear()
                timer = None

        while not cancel_event.is_set():
            try:
                data = client_socket.recv(1024)
                if not data:
                    print(f"[Connection] No data received. Closing connection from {addr}.")
                    break
                incoming = data.decode().strip()
                print(f"[Data Received Raw] {incoming}")  # Log raw data

                # Check if the incoming data is a command
                if incoming.startswith('/'):
                    handle_command(incoming, client_socket)
                else:
                    with data_lock:
                        data_buffer.append(incoming)
                        # Reset the timer
                        if timer:
                            timer.cancel()
                            print("[Timer] Existing timer canceled.")
                        timeout_seconds = float(config.get('inference_timeout', '5'))
                        timer = threading.Timer(timeout_seconds, inference_trigger)
                        timer.start()
                        print(f"[Timeout] Inference will be triggered in {timeout_seconds} seconds if no more data is received.")
            except socket.timeout:
                continue
            except Exception as e:
                print(f"[Error] Error in handle_client_socket: {e}")
                traceback.print_exc()
                break

        # Cleanup timer if connection is closing
        with data_lock:
            if timer:
                timer.cancel()
                print("[Timer] Existing timer canceled during cleanup.")
            if data_buffer:
                print(f"[Inference] Connection closing. Triggering final inference for remaining input.")
                # Run perform_inference in a separate thread
                threading.Thread(target=perform_inference, args=(' '.join(data_buffer).strip(),), daemon=True).start()
                data_buffer.clear()

        print(f"[Connection] Closing connection from {addr}")

# Handle special commands like /info and /exit
def handle_command(command, client_socket):
    addr = client_socket.getpeername()
    print(f"[Command Received] {command} from {addr}")
    if command.startswith('/info'):
        send_info(client_socket)
    elif command.startswith('/exit'):
        send_exit(client_socket)
    else:
        print(f"[Warning] Unknown command received: {command}")

# Send configuration info to the orchestrator over the existing socket
def send_info(client_socket):
    try:
        response = {
            "script_name": script_name,
            "script_uuid": script_uuid,
            "configuration": config
        }
        response_json = json.dumps(response, indent=2)
        response_message = f"{response_json}\n"
        client_socket.sendall(response_message.encode())
        print(f"[Info] Sent configuration info to {client_socket.getpeername()}")
    except Exception as e:
        print(f"[Error] Failed to send info to {client_socket.getpeername()}: {e}")
        traceback.print_exc()

# Send exit acknowledgment and shutdown
def send_exit(client_socket):
    try:
        exit_message = "Exiting SUM Engine.\n"
        client_socket.sendall(exit_message.encode())
        print(f"[Shutdown] Received /exit command from {client_socket.getpeername()}. Shutting down.")
        cancel_event.set()
    except Exception as e:
        print(f"[Error] Failed to send exit acknowledgment to {client_socket.getpeername()}: {e}")
        traceback.print_exc()

# Lock for controlling concurrency of inference requests
inference_lock = threading.Lock()

# Function to perform inference using Ollama API
def perform_inference(user_input):
    with inference_lock:  # Ensure only one inference happens at a time
        model_name = config.get('model_name', 'llama3.2:1b')
        system_prompt = config.get('system_prompt', '')
        json_filtering = config.get('json_filtering', True)  # Use it as a boolean
        api_endpoint = config.get('api_endpoint', 'generate')

        # Model parameters
        try:
            temperature = float(config.get('temperature', 0.7))
            top_p = float(config.get('top_p', 0.9))
            max_tokens = int(config.get('max_tokens', 150))
            repeat_penalty = float(config.get('repeat_penalty', 1.0))
        except ValueError as ve:
            print(f"[Error] Invalid model parameter in config: {ve}")
            send_error_response(f"Error: Invalid model parameter in config: {ve}")
            return

        if not system_prompt:
            print("❌ System prompt is not configured.")
            send_error_response("Error: System prompt is not configured.")
            return

        # Build the prompt
        prompt = f"{system_prompt}\n\nUser Input: {user_input}\nOutput:"

        # Set format to "json" if json_filtering is enabled, otherwise no specific format
        payload = {
            "model": model_name,
            "prompt": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "repeat_penalty": repeat_penalty,
            "max_tokens": max_tokens,
            "format": "json" if json_filtering else "",  # Handle JSON filtering as a boolean
            "stream": False
        }

        headers = {
            "Content-Type": "application/json"
        }

        # Check if Ollama API is up before making the request
        if not check_ollama_api():
            send_error_response("Error: Ollama API is not available.")
            return

        try:
            api_url = f"{OLLAMA_URL}{api_endpoint}"
            print(f"[Performing Inference] Sending request to Ollama API at {api_url} with payload: {json.dumps(payload, indent=2)}")
            response = requests.post(api_url, json=payload, headers=headers, timeout=60)
            print(f"[Ollama API Response Status] {response.status_code}")
            
            if response.status_code == 200:
                response_json = response.json()
                model_response = response_json.get("response", "")

                if not model_response:
                    send_error_response("Error: Empty response from model.")
                    return

                # Apply JSON filtering only if json_filtering is enabled
                if json_filtering:
                    try:
                        parsed_json = json.loads(model_response)
                        filtered_response = extract_values(parsed_json)
                        print(f"[Inference Complete] Model Output (Filtered):\n{filtered_response}\n")
                        send_response(filtered_response)
                    except json.JSONDecodeError as e:
                        print(f"[Error] Parsing model response failed: {e}")
                        send_error_response(f"Error parsing model response: {e}\nResponse Content: {model_response}")
                else:
                    # Send raw response if JSON filtering is disabled
                    print(f"[Inference Complete] Model Output (Raw):\n{model_response}\n")
                    send_response(model_response)

            else:
                print(f"[Error] Ollama API responded with status code: {response.status_code}")
                send_error_response(f"Error: {response.status_code} - {response.text}")
        except requests.exceptions.Timeout:
            print("[Error] Request to Ollama API timed out.")
            send_error_response("Error: Request to Ollama API timed out.")
        except requests.exceptions.ConnectionError:
            print("[Error] Connection error occurred while contacting Ollama API.")
            send_error_response("Error: Connection error with Ollama API.")
        except Exception as e:
            print(f"[Error] Exception during model inference: {e}")
            traceback.print_exc()
            send_error_response(f"Error during model inference: {e}")


def extract_values(obj, key=None):
    """Recursively extract values from JSON object."""
    values = []

    def extract(obj, values, key):
        """Recursively search for values of key in JSON tree."""
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == key or key is None:
                    values.append(v)
                if isinstance(v, (dict, list)):
                    extract(v, values, key)
        elif isinstance(obj, list):
            for item in obj:
                extract(item, values, key)
        return values

    return extract(obj, values, key)


# Function to send response back to orchestrator
def send_response(output_data):
    # Connect to the orchestrator's data port
    host = config.get('orchestrator_host', 'localhost')
    data_port = config.get('data_port', '6001')

    # Validate data_port
    try:
        data_port = int(data_port)
    except ValueError:
        print(f"[Error] Invalid data port: {data_port}")
        return

    try:
        with socket.create_connection((host, data_port), timeout=5) as s:
            # Send script UUID first
            s.sendall(f"{script_uuid}\n".encode())
            # Send the output data
            s.sendall(f"{output_data}\n".encode())
            print(f"[Sent] Chunk sent to orchestrator: {output_data}")
    except ConnectionRefusedError:
        print(f"[Error] Connection refused by orchestrator at {host}:{data_port}.")
    except socket.timeout:
        print(f"[Error] Connection to orchestrator at {host}:{data_port} timed out.")
    except Exception as e:
        print(f"[Error] Failed to send output to orchestrator: {e}")
        traceback.print_exc()


# Function to send error response back to orchestrator
def send_error_response(error_message):
    error_json = {
        "error": error_message
    }
    formatted_error = json.dumps(error_json, indent=2)
    print(f"[Error Response] {formatted_error}")
    send_response(formatted_error)


# Main function
def main():
    # Read configuration from file
    global config
    config = read_config()

    # Parse command-line arguments
    parse_args()

    # Write updated config to file
    write_config(config)

    # Check if Ollama API is up before starting
    if not check_ollama_api():
        print("🛑 Ollama is not running. Please start the Ollama API and try again.")
    else:
        print("[Info] Ollama API is available.")

    # Start the server to handle incoming connections
    start_server()

# Function to list available models (optional utility)
def list_available_models():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            print("📦 Available Models:")
            for model in models:
                print(f" - {model['name']}")
        else:
            print(f"⚠️ Failed to retrieve models. Status Code: {response.status_code}")
    except Exception as e:
        print(f"[Error] Could not retrieve models: {e}")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Shutdown] SUM Engine terminated by user.")
    except Exception as e:
        print(f"[Fatal Error] An unexpected error occurred: {e}")
        traceback.print_exc()
