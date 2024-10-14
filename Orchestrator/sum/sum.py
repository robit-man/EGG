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
    'input_format': 'streaming',    # Options: 'streaming', 'chunk'
    'output_format': 'streaming',   # Options: 'streaming', 'chunk'
    'port_range': '6200-6300',
    'orchestrator_host': 'localhost',
    'orchestrator_ports': '6000-6005',  # Updated to handle multiple ports
    'route': '/slm',
    'script_uuid': '',              # Initialize as empty; will be set in read_config()
    'system_prompt': "You Respond Conversationally",
    'temperature': '0.6',           # Model parameter: temperature
    'top_p': '0.9',                 # Model parameter: top_p
    'max_tokens': '70',            # Model parameter: max tokens
    'repeat_penalty': '1.0',        # Model parameter: repeat penalty
    'inference_timeout': '5',       # Timeout in seconds for inference
    'json_filtering': 'true',
}
config = {}

# UUID and name for the peripheral
# Initialize script_uuid as None; it will be set in read_config()
script_uuid = None
script_name = 'SUM_Engine'

# Event to signal shutdown
cancel_event = threading.Event()

# Ollama API endpoint
OLLAMA_URL = "http://localhost:11434/api/generate"

# Lock for thread-safe operations
config_lock = threading.Lock()

# Registration flag
registered = False

# Function to read configuration from file
def read_config():
    global config, script_uuid
    config = default_config.copy()
    if os.path.exists(CONFIG_FILE):
        print(f"Reading configuration from {CONFIG_FILE}")
        with open(CONFIG_FILE, 'r') as f:
            content = f.read()
            # Use regex to parse key="value" pairs with support for escaped characters
            matches = re.findall(r'(\w+)\s*=\s*("(?:\\.|[^"])*"|[^\n]+)', content)
            for key, value in matches:
                key = key.strip()
                value = value.strip()
                if value.startswith('"') and value.endswith('"'):
                    # Remove surrounding quotes and unescape characters
                    try:
                        value = bytes(value[1:-1], "utf-8").decode("unicode_escape")
                    except UnicodeDecodeError:
                        print(f"[Error] Decoding error for key '{key}'. Using raw value.")
                        value = value[1:-1]
                if key in config:
                    config[key] = value
                else:
                    print(f"Unknown configuration key: {key}")
    else:
        # If CONFIG_FILE does not exist, generate a new UUID and create the config file
        script_uuid = str(uuid.uuid4())
        config['script_uuid'] = script_uuid  # Add UUID to config
        write_config(config)
        print(f"[Info] Generated new UUID: {script_uuid} and created {CONFIG_FILE}")
    
    # After reading config, check for 'script_uuid'
    if 'script_uuid' in config and config['script_uuid']:
        script_uuid = config['script_uuid']
    else:
        # If 'script_uuid' is missing or empty, generate it and update config
        script_uuid = str(uuid.uuid4())
        config['script_uuid'] = script_uuid
        write_config(config)
        print(f"[Info] Generated new UUID: {script_uuid} and updated {CONFIG_FILE}")
    
    # Debug: Print configuration after reading
    print("[Debug] Configuration Loaded:")
    for k, v in config.items():
        if k == 'system_prompt':
            print(f"{k}={'[REDACTED]'}")  # Hide system_prompt in debug
        elif k == 'script_uuid':
            print(f"{k}={v}")  # Display script_uuid
        else:
            print(f"{k}={v}")
    return config

# Function to write configuration to file
def write_config(config_to_write=None):
    config_to_write = config_to_write or config
    print(f"Writing configuration to {CONFIG_FILE}")
    with config_lock:
        with open(CONFIG_FILE, 'w') as f:
            for key, value in config_to_write.items():
                value = str(value)  # Ensure all values are strings
                if any(c in value for c in ' \n"\\'):
                    # If value contains special characters, enclose it in quotes and escape
                    escaped_value = value.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
                    f.write(f'{key}="{escaped_value}"\n')
                else:
                    f.write(f"{key}={value}\n")

# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='SLM Engine Peripheral')
    parser.add_argument('--port-range', type=str, help='Port range to use for connections')
    parser.add_argument('--orchestrator-host', type=str, help='Orchestrator host address')
    parser.add_argument('--orchestrator-ports', type=str, help='Comma-separated list or range of orchestrator command ports (e.g., 6000,6001,6002 or 6000-6005)')
    parser.add_argument('--model-name', type=str, help='Name of the language model to use')
    parser.add_argument('--system-prompt', type=str, help='System prompt for the model')
    parser.add_argument('--temperature', type=str, help='Model parameter: temperature')
    parser.add_argument('--top_p', type=str, help='Model parameter: top_p')
    parser.add_argument('--max_tokens', type=str, help='Model parameter: max tokens')
    parser.add_argument('--repeat_penalty', type=str, help='Model parameter: repeat penalty')
    parser.add_argument('--inference_timeout', type=str, help='Timeout in seconds for inference')
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
    if args.temperature:
        config['temperature'] = args.temperature
    if args.top_p:
        config['top_p'] = args.top_p
    if args.max_tokens:
        config['max_tokens'] = args.max_tokens
    if args.repeat_penalty:
        config['repeat_penalty'] = args.repeat_penalty
    if args.inference_timeout:
        config['inference_timeout'] = args.inference_timeout

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
                        if isinstance(ack_message, str) and ack_message.startswith('/ack'):
                            tokens = ack_message.split()
                            if len(tokens) == 2 and tokens[0] == '/ack':
                                try:
                                    data_port = tokens[1]  # Keep as string
                                    config['data_port'] = data_port
                                    write_config(config)
                                    print(f"[Info] Registered successfully. Data port: {data_port}")
                                    registered = True
                                    return True
                                except ValueError:
                                    print(f"[Error] Invalid data port received in acknowledgment: {tokens[1]}")
                            else:
                                print(f"[Error] Unexpected acknowledgment format: {ack_message}")
                        else:
                            print(f"[Error] Invalid acknowledgment type: {type(ack_message)}")
                    else:
                        print(f"[Warning] No acknowledgment received from orchestrator at {host}:{orch_port}.")
            except socket.timeout:
                print(f"[Error] Connection to orchestrator at {host}:{orch_port} timed out.")
            except ConnectionRefusedError:
                print(f"[Error] Connection refused by orchestrator at {host}:{orch_port}.")
            except Exception as e:
                print(f"[Error] Exception during registration with orchestrator at {host}:{orch_port}: {e}")
                traceback.print_exc()
        print(f"Retrying registration in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
        time.sleep(retry_delay)
        retry_delay *= 2  # Exponential backoff

    print("[Error] Max retries reached. Could not register with orchestrator.")
    if config['input_mode'] == 'terminal':
        print("Failed to register with the orchestrator after multiple attempts. Entering terminal mode.")
        return False  # Allow fallback to terminal mode
    else:
        print("Failed to register with the orchestrator after multiple attempts. Exiting...")
        cancel_event.set()
        exit(1)  # Exit the script as registration is critical

# Start server to handle incoming connections
def start_server():
    host = '0.0.0.0'
    port_range = config.get('port_range', '6200-6300')
    port_list = parse_port_range(port_range)
    server_socket = None

    for port in port_list:
        try:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.bind((host, port))
            print(f"[Info] SLM Engine listening on port {port}...")
            config['port'] = str(port)  # Update the port in the config
            write_config(config)
            break
        except OSError:
            print(f"[Warning] Port {port} is unavailable, trying next port...")
            if server_socket:
                server_socket.close()
            server_socket = None

    if not server_socket:
        print("[Error] Failed to bind to any port in the specified range.")
        return

    # Register with orchestrator
    registration_successful = register_with_orchestrator(port)

    if not registration_successful and config['input_mode'] == 'terminal':
        # Fallback to terminal mode
        terminal_input()
        return
    elif not registration_successful:
        # If not in terminal mode, exit
        return

    server_socket.listen(5)
    print(f"[Info] Server started on port {config['port']}. Waiting for connections...")

    while not cancel_event.is_set():
        try:
            server_socket.settimeout(1.0)
            client_socket, addr = server_socket.accept()
            print(f"[Connection] Connection from {addr}")
            threading.Thread(target=handle_client_socket, args=(client_socket,), daemon=True).start()
        except socket.timeout:
            continue
        except Exception as e:
            if not cancel_event.is_set():
                print(f"[Error] Error accepting connections: {e}")
                traceback.print_exc()

    server_socket.close()
    print("[Info] Server shut down.")

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
        exit_message = "Exiting SLM Engine.\n"
        client_socket.sendall(exit_message.encode())
        print(f"[Shutdown] Received /exit command from {client_socket.getpeername()}. Shutting down.")
        cancel_event.set()
    except Exception as e:
        print(f"[Error] Failed to send exit acknowledgment to {client_socket.getpeername()}: {e}")
        traceback.print_exc()

# Function to perform inference using Ollama API
def perform_inference(user_input):
    model_name = config.get('model_name', 'llama3.2:1b')
    system_prompt = config.get('system_prompt', '')
    json_filtering = config.get('json_filtering', 'true').lower() == 'true'  # Enable JSON filtering based on config

    # Model parameters
    try:
        temperature = float(config.get('temperature', '0.7'))
        top_p = float(config.get('top_p', '0.9'))
        max_tokens = int(config.get('max_tokens', '150'))  # Set to 150 as per requirement
        repeat_penalty = float(config.get('repeat_penalty', '1.0'))
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
        "format": "json" if json_filtering else "",
        "stream": False  # Disable streaming for simplified handling
    }

    headers = {
        "Content-Type": "application/json"
    }

    # Check if Ollama API is up before making the request
    if not check_ollama_api():
        send_error_response("Error: Ollama API is not available.")
        return

    try:
        print(f"[Performing Inference] Sending request to Ollama API with payload: {json.dumps(payload, indent=2)}")
        response = requests.post(OLLAMA_URL, json=payload, headers=headers, timeout=60)
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

                    # Extract only the values from the JSON object
                    def extract_values(data):
                        if isinstance(data, dict):
                            return ' '.join(extract_values(v) for v in data.values())
                        elif isinstance(data, list):
                            return ' '.join(extract_values(v) for v in data)
                        else:
                            return str(data)

                    filtered_response = extract_values(parsed_json)
                    print(f"[Inference Complete] Model Output (Filtered):\n{filtered_response}\n")
                    send_response(filtered_response)
                except json.JSONDecodeError as e:
                    print(f"[Error] Parsing model response failed: {e}")
                    send_error_response(f"Error parsing model response: {e}\nResponse Content: {model_response}")
            else:
                # If json_filtering is disabled, send raw response
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
        # Optionally, you can retry or exit
        # Here, we'll proceed to allow the user to handle it
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
        print("\n[Shutdown] SLM Engine terminated by user.")
    except Exception as e:
        print(f"[Fatal Error] An unexpected error occurred: {e}")
        traceback.print_exc()
