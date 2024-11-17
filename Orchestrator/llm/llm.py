# orchestrator_script.py

import threading
import socket
import json
import uuid
import os
import re
import argparse
import importlib.util
import requests
import traceback
import subprocess
import time
from datetime import datetime

# ANSI color codes for terminal output
COLOR_RESET = "\033[0m"
COLOR_BLUE = "\033[34m"
COLOR_GREEN = "\033[32m"
COLOR_YELLOW = "\033[33m"
COLOR_CYAN = "\033[36m"
COLOR_RED = "\033[31m"
COLOR_MAGENTA = "\033[35m"

# Configuration file name
CONFIG_FILE = 'llm.cf'

# Chat history and tools files
CHAT_HISTORY_FILE = 'chat.hst'
TOOLS_FILE = 'model.tools.json'
FUNCTIONS_FILE = 'model.functions.py'

# Base system prompt to guide the model's behavior
BASE_SYSTEM_PROMPT = (
'''
You are an expert in human communication, attuned to context, and understand when engagement is appropriate. Recognize that in ongoing conversations, certain exchanges—especially lengthy dialogues between others without any direct query—do not require a response. In such cases, remain silent and continue listening for an explicit request that seeks your input. Additionally, observe the delimiter `<|NEXT DATA|>`, which separates sequential ASR content. When key-value pairs are received, evaluate their relevance to prior conversational input and integrate this unpacked data into your response where it provides additional clarity or context. Now please apply the logic above to the following content, noticing also that you will have context from previous interactions, and to incorporate them when appropriate: 
''')

# URLs for the files in your GitHub repository
TOOLS_FILE_URL = 'https://raw.githubusercontent.com/robit-man/EGG/main/Orchestrator/tools/model.tools.json'
FUNCTIONS_FILE_URL = 'https://raw.githubusercontent.com/robit-man/EGG/main/Orchestrator/tools/model.functions.py'


# Default configuration
default_config = {
    'model_name': 'llama:3.2:3b',
    'input_mode': 'port',           # Options: 'port', 'terminal', 'route'
    'output_mode': 'port',          # Options: 'port', 'terminal', 'route'
    'input_format': 'chunk',        # Options: 'streaming', 'chunk'
    'output_format': 'chunk',       # Options: 'streaming', 'chunk'
    'port_range': '6200-6300',
    'orchestrator_host': 'localhost',
    'orchestrator_ports': '6000-6099',  # Updated to handle multiple ports
    'route': '/llm',
    'script_uuid': '',              # Initialize as empty; will be set in read_config()
    'system_prompt': BASE_SYSTEM_PROMPT,
    'temperature': 0.7,             # Model parameter: temperature
    'top_p': 0.9,                   # Model parameter: top_p
    'max_tokens': 150,              # Model parameter: max tokens
    'repeat_penalty': 1.0,          # Model parameter: repeat penalty
    'inference_timeout': 15,        # Timeout in seconds for inference
    'json_filtering': False,        # Use JSON filtering by default
    'api_endpoint': 'chat',         # Options: 'generate', 'chat'
    'stream': False,                # Assuming 'stream' is a valid config key
    'enable_chat_history': True,
    'use_tools': False,
    'history_limit': 20             # New key for chat history depth, default to 20
}

config = {}

# UUID and name for the peripheral
script_uuid = None
script_name = 'LLM_Engine'

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
        print(f"{COLOR_BLUE}[Info]{COLOR_RESET} Reading configuration from {CONFIG_FILE}")
        try:
            with open(CONFIG_FILE, 'r') as f:
                loaded_config = json.load(f)
                config.update(loaded_config)  # Update default config with the values from the file
        except (json.JSONDecodeError, FileNotFoundError):
            print(f"{COLOR_RED}[Error]{COLOR_RESET} Could not read {CONFIG_FILE}. Using default configuration.")

    # Ensure script_uuid is set
    if 'script_uuid' in config and config['script_uuid']:
        script_uuid = config['script_uuid']
    # Ensure script_uuid is set
    
    else:
        script_uuid = str(uuid.uuid4())
        config['script_uuid'] = script_uuid
        write_config()  # Save updated config with the new UUID

    if config['use_tools'] is True:
        ensure_required_files()

        # Initialize global functions module
        functions = load_functions()
        if not functions:
            exit(1)  # Exit if functions module couldn't be loaded

        # Initialize tools mapping
        setup_tools()
    # Debug: Print the loaded configuration
    print(f"{COLOR_GREEN}[Debug]{COLOR_RESET} Configuration Loaded:")
    for k, v in config.items():
        if k == 'system_prompt':
            print(f"{k}={'[REDACTED]'}")  # Hide system_prompt in debug
        else:
            print(f"{k}={v}")

    return config


def download_file(url, file_path):
    """
    Downloads a file from the given URL and saves it to the specified path.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad status codes
        with open(file_path, 'w') as file:
            file.write(response.text)
        print(f"Downloaded and saved: {file_path}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {file_path} from {url}: {e}")

def ensure_required_files():
    # Check for and create 'model.tools.json' if it does not exist
    if not os.path.exists(TOOLS_FILE):
        print(f"{COLOR_YELLOW}[Warning]{COLOR_RESET} '{TOOLS_FILE}' not found. Downloading from {TOOLS_FILE_URL}.")
        download_file(TOOLS_FILE_URL, TOOLS_FILE)

    # Check for and create 'model.functions.py' if it does not exist
    if not os.path.exists(FUNCTIONS_FILE):
        print(f"{COLOR_YELLOW}[Warning]{COLOR_RESET} '{FUNCTIONS_FILE}' not found. Downloading from {FUNCTIONS_FILE_URL}.")
        download_file(FUNCTIONS_FILE_URL, FUNCTIONS_FILE)


# Function to write configuration to a JSON file
def write_config(config_to_write=None):
    config_to_write = config_to_write or config
    print(f"{COLOR_BLUE}[Info]{COLOR_RESET} Writing configuration to {CONFIG_FILE}")
    with config_lock:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config_to_write, f, indent=4)

# Chat history management functions
def load_chat_history():
    if not config.get('enable_chat_history', True):
        print(f"{COLOR_YELLOW}[Info]{COLOR_RESET} Chat history is disabled.")
        return []
    
    if os.path.exists(CHAT_HISTORY_FILE):
        print(f"{COLOR_BLUE}[Info]{COLOR_RESET} Reading chat history from {CHAT_HISTORY_FILE}")
        try:
            with open(CHAT_HISTORY_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            print(f"{COLOR_RED}[Error]{COLOR_RESET} Could not read {CHAT_HISTORY_FILE}. Starting with empty chat history.")
            return []
    return []

def save_chat_history(history):
    if not config.get('enable_chat_history', True):
        print(f"{COLOR_YELLOW}[Info]{COLOR_RESET} Chat history saving is disabled.")
        return
    
    print(f"{COLOR_BLUE}[Info]{COLOR_RESET} Saving chat history to {CHAT_HISTORY_FILE}")
    with open(CHAT_HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=4)

def append_to_chat_history(message):
    if not config.get('enable_chat_history', True):
        print(f"{COLOR_YELLOW}[Info]{COLOR_RESET} Chat history is disabled. Not appending message.")
        return

    chat_history = load_chat_history()
    chat_history.append(message)
    save_chat_history(chat_history)


def get_recent_chat_history(config=default_config):
    history_limit = int(config.get('history_limit', 20))
    chat_history = load_chat_history()
    return chat_history[-history_limit:] if len(chat_history) > history_limit else chat_history


# Function to load functions module
def load_functions():
    try:
        # Load `model.functions.py`
        spec = importlib.util.spec_from_file_location("model_functions", "model.functions.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except FileNotFoundError:
        print(f"{COLOR_RED}[Error]{COLOR_RESET} 'model.functions.py' not found.")
        return None
    except Exception as e:
        print(f"{COLOR_RED}[Error]{COLOR_RESET} Error loading 'model.functions.py': {e}")
        traceback.print_exc()
        return None

# Function to load tools from model.tools.json
def load_tools():
    tools_file = TOOLS_FILE
    try:
        with open(tools_file, 'r') as f:
            tools = json.load(f)
            if not isinstance(tools, dict):
                print(f"{COLOR_RED}[Error]{COLOR_RESET} '{tools_file}' should contain a JSON object mapping tool names to function names.")
                return {}
            return tools
    except (json.JSONDecodeError, FileNotFoundError):
        print(f"{COLOR_RED}[Error]{COLOR_RESET} Could not read '{tools_file}'. Returning empty tools list.")
        return {}

# Function to call the appropriate tool
def call_tool(name, data):
    tools = load_tools()  # Load tools mapping
    functions = load_functions()  # Load functions module

    # Ensure tools were loaded correctly
    if not tools or not isinstance(tools, dict):
        print(f"{COLOR_RED}[Error]{COLOR_RESET} No tools found or invalid tools format in '{TOOLS_FILE}' file.")
        return "Error: No tools available."

    # Check if the tool exists in the tools list
    if name in tools:
        function_name = tools[name]
        print(f"{COLOR_CYAN}[Tool Match Found]{COLOR_RESET} Tool '{name}' corresponds to function '{function_name}'.")

        # Check if the function exists in model.functions
        if functions and hasattr(functions, function_name):
            function_to_call = getattr(functions, function_name)
            try:
                print(f"{COLOR_CYAN}[Tool Execution]{COLOR_RESET} Calling function '{function_name}' with data: {data}.")
                
                # **Print the structured data (arguments)**:
                print(f"Arguments passed to the tool: {json.dumps(data, indent=2)}")
                
                # Call the function and ensure synchronous processing
                result = function_to_call(data)

                return handle_result(result)  # Handle the result based on the type (string, dict, etc.)
            except Exception as e:
                print(f"{COLOR_RED}[Error]{COLOR_RESET} Error while calling function '{function_name}': {e}")
                return f"Error: {e}"
        else:
            print(f"{COLOR_RED}[Error]{COLOR_RESET} Function '{function_name}' not found in 'model.functions.py'.")
            return f"Error: Function '{function_name}' not found."
    else:
        print(f"{COLOR_RED}[Error]{COLOR_RESET} Tool '{name}' not found in tools list.")
        return f"Error: Tool '{name}' not found."

def handle_result(result):
    """
    Process the result returned by the tool function.
    Depending on the type of result, return a string or a formatted output.

    Args:
        result (Any): The result returned by the tool function.

    Returns:
        str: The processed output to be sent back to the orchestrator or displayed.
    """
    # If the result is a string, return it directly
    if isinstance(result, str):
        return result

    # If the result is a dictionary, format it as a pretty-printed JSON
    elif isinstance(result, dict):
        try:
            return json.dumps(result, indent=2)
        except (TypeError, ValueError):
            return f"Error: Unable to format dictionary result: {result}"

    # If the result is a list, convert it into a human-readable format
    elif isinstance(result, list):
        try:
            return "\n".join(map(str, result))
        except Exception as e:
            return f"Error: Unable to format list result: {e}"

    # If the result is of any other type, convert it to string
    else:
        return str(result)


def save_tools(tools):
    print(f"{COLOR_BLUE}[Info]{COLOR_RESET} Saving tools to {TOOLS_FILE}")
    with open(TOOLS_FILE, 'w') as f:
        json.dump(tools, f, indent=4)

def list_tools():
    tools = load_tools()
    return list(tools.keys())

# Function to parse command-line arguments and update config
def parse_args():
    parser = argparse.ArgumentParser(description='LLM Engine Peripheral')
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
        part = part.strip()
        if '-' in part:
            try:
                start, end = map(int, part.split('-'))
                ports.extend(range(start, end + 1))
            except ValueError:
                print(f"{COLOR_YELLOW}[Warning]{COLOR_RESET} Invalid port range format: {part}")
        else:
            try:
                ports.append(int(part))
            except ValueError:
                print(f"{COLOR_YELLOW}[Warning]{COLOR_RESET} Invalid port number: {part}")
    return ports

# Function to check if Ollama API is up
def check_ollama_api():
    try:
        response = requests.get(f"{OLLAMA_URL}tags", timeout=5)
        if response.status_code == 200:
            print(f"{COLOR_GREEN}[Info]{COLOR_RESET} Ollama API is up and running.")
            return True
        else:
            print(f"{COLOR_YELLOW}[Warning]{COLOR_RESET} Ollama API responded with status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"{COLOR_RED}❌ Unable to connect to Ollama API.{COLOR_RESET}")
        return False
    except requests.exceptions.Timeout:
        print(f"{COLOR_YELLOW}⏰ Ollama API connection timed out.{COLOR_RESET}")
        return False
    except Exception as e:
        print(f"{COLOR_RED}[Error]{COLOR_RESET} Unexpected error while checking Ollama API: {e}")
        return False

# Function to register with the orchestrator
def register_with_orchestrator(port):
    global registered
    if registered:
        print(f"{COLOR_BLUE}[Info]{COLOR_RESET} Already registered with the orchestrator. Skipping registration.")
        return True

    host = config.get('orchestrator_host', 'localhost')
    orchestrator_ports_str = config.get('orchestrator_ports', '6000-6099')
    orchestrator_command_ports = []

    # Parse orchestrator_ports (e.g., '6000-6005' or '6000,6001,6002')
    for port_entry in orchestrator_ports_str.split(','):
        port_entry = port_entry.strip()
        if '-' in port_entry:
            try:
                start_port, end_port = map(int, port_entry.split('-'))
                orchestrator_command_ports.extend(range(start_port, end_port + 1))
            except ValueError:
                print(f"{COLOR_RED}[Error]{COLOR_RESET} Invalid orchestrator port range: {port_entry}")
        elif port_entry.isdigit():
            orchestrator_command_ports.append(int(port_entry))
        else:
            print(f"{COLOR_RED}[Error]{COLOR_RESET} Invalid orchestrator port entry: {port_entry}")

    if not orchestrator_command_ports:
        print(f"{COLOR_RED}[Error]{COLOR_RESET} No valid orchestrator command ports found.")
        return False

    # Registration message
    message = f"/register {script_name} {script_uuid} {port}\n"
    max_retries = 5
    retry_delay = 1  # Start with 1 second delay for retries
    backoff_factor = 2  # Exponential backoff factor

    for attempt in range(max_retries):
        for orch_port in orchestrator_command_ports:
            try:
                print(f"{COLOR_YELLOW}[Attempt {attempt + 1}]{COLOR_RESET} Registering with orchestrator at {host}:{orch_port}...")

                # Open a connection to the orchestrator
                with socket.create_connection((host, orch_port), timeout=5) as s:
                    s.sendall(message.encode())
                    print(f"{COLOR_BLUE}[Info]{COLOR_RESET} Sent registration message: {message.strip()}")

                    # Receive acknowledgment
                    data = s.recv(1024)
                    if data:
                        ack_message = data.decode().strip()
                        print(f"{COLOR_BLUE}[Info]{COLOR_RESET} Received acknowledgment: {ack_message}")

                        # Check if the acknowledgment is valid
                        if ack_message.startswith('/ack'):
                            parts = ack_message.split()
                            if len(parts) >= 2:
                                config['data_port'] = parts[1]
                                write_config(config)  # Save updated config with assigned data_port
                                registered = True
                                print(f"{COLOR_GREEN}[Success]{COLOR_RESET} Registered with orchestrator on port {orch_port}.")
                                return True
                            else:
                                print(f"{COLOR_RED}[Error]{COLOR_RESET} Invalid acknowledgment format from orchestrator: {ack_message}")
                        else:
                            print(f"{COLOR_RED}[Error]{COLOR_RESET} Unexpected acknowledgment message: {ack_message}")
                    else:
                        print(f"{COLOR_YELLOW}[Warning]{COLOR_RESET} No acknowledgment received from orchestrator at {host}:{orch_port}.")

            except socket.timeout:
                print(f"{COLOR_RED}[Error]{COLOR_RESET} Timeout while connecting to orchestrator at {host}:{orch_port}.")
            except ConnectionRefusedError:
                print(f"{COLOR_RED}[Error]{COLOR_RESET} Connection refused by orchestrator at {host}:{orch_port}.")
            except Exception as e:
                print(f"{COLOR_RED}[Error]{COLOR_RESET} Unexpected error during registration: {e}")
                traceback.print_exc()

        # Retry with exponential backoff
        print(f"{COLOR_YELLOW}Retrying registration in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries}){COLOR_RESET}")
        time.sleep(retry_delay)
        retry_delay *= backoff_factor  # Exponential backoff

    # If all attempts failed
    print(f"{COLOR_RED}[Error]{COLOR_RESET} Max retries reached. Could not register with orchestrator.")
    cancel_event.set()
    exit(1)  # Exit the script if registration fails

# Function to send the full configuration to the orchestrator
def send_full_config_to_orchestrator(host, port):
    try:
        message = f"/config {script_uuid}\n{json.dumps(config, indent=2)}\nEOF\n"
        print(f"{COLOR_BLUE}[Info]{COLOR_RESET} Sending full configuration to orchestrator at {host}:{port}...")

        with socket.create_connection((host, port), timeout=5) as s:
            s.sendall(message.encode())
            print(f"{COLOR_GREEN}[Success]{COLOR_RESET} Full configuration sent to orchestrator.")

    except socket.timeout:
        print(f"{COLOR_RED}[Error]{COLOR_RESET} Timeout while sending configuration to orchestrator at {host}:{port}.")
    except ConnectionRefusedError:
        print(f"{COLOR_RED}[Error]{COLOR_RESET} Connection refused by orchestrator at {host}:{port}.")
    except Exception as e:
        print(f"{COLOR_RED}[Error]{COLOR_RESET} Unexpected error while sending configuration: {e}")
        traceback.print_exc()

# Start server to handle incoming connections
def start_server():
    host = '0.0.0.0'
    port_list = parse_port_range(config.get('port_range', '6200-6300'))  # Parse port range from config

    try:
        # Dynamically find an available port in the specified range
        available_port = find_available_port(port_list)
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((host, available_port))
        print(f"{COLOR_BLUE}[Info]{COLOR_RESET} LLM Engine listening on port {available_port}...")
        config['port'] = str(available_port)  # Update the port in the config
        write_config(config)  # Save the updated configuration with the selected port
    except Exception as e:
        print(f"{COLOR_RED}[Error]{COLOR_RESET} Could not bind to any available port: {e}")
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
    print(f"{COLOR_BLUE}[Info]{COLOR_RESET} Server started on port {config['port']}. Waiting for connections...")

    while not cancel_event.is_set():
        try:
            server_socket.settimeout(1.0)  # Set a timeout to handle periodic cancellation
            client_socket, addr = server_socket.accept()
            print(f"{COLOR_GREEN}[Connection]{COLOR_RESET} Accepted connection from {addr}")
            threading.Thread(target=handle_client_socket, args=(client_socket,), daemon=True).start()
        except socket.timeout:
            continue  # Continue looping until a client connects or cancel_event is set
        except Exception as e:
            if not cancel_event.is_set():
                print(f"{COLOR_RED}[Error]{COLOR_RESET} Error accepting connections: {e}")
                traceback.print_exc()

    server_socket.close()
    print(f"{COLOR_BLUE}[Info]{COLOR_RESET} Server shut down.")

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
        print(f"{COLOR_BLUE}[Connection]{COLOR_RESET} Handling data from {addr}")

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
                        print(f"{COLOR_YELLOW}[Inference]{COLOR_RESET} Timeout reached. Triggering inference for accumulated input: {user_input}")
                        # Run perform_inference in a separate thread
                        threading.Thread(target=perform_inference, args=(user_input,), daemon=True).start()
                        data_buffer.clear()
                timer = None

        while not cancel_event.is_set():
            try:
                data = client_socket.recv(1024)
                if not data:
                    print(f"{COLOR_YELLOW}[Connection]{COLOR_RESET} No data received. Closing connection from {addr}.")
                    break
                incoming = data.decode().strip()
                print(f"{COLOR_CYAN}[Data Received Raw]{COLOR_RESET} {incoming}")  # Log raw data

                # Check if the incoming data is a command
                if incoming.startswith('/'):
                    handle_command(incoming, client_socket)
                else:
                    with data_lock:
                        data_buffer.append(incoming)
                        # Reset the timer
                        if timer:
                            timer.cancel()
                            print(f"{COLOR_YELLOW}[Timer]{COLOR_RESET} Existing timer canceled.")
                        timeout_seconds = float(config.get('inference_timeout', '15'))
                        timer = threading.Timer(timeout_seconds, inference_trigger)
                        timer.start()
                        print(f"{COLOR_YELLOW}[Timeout]{COLOR_RESET} Inference will be triggered in {timeout_seconds} seconds if no more data is received.")
            except socket.timeout:
                continue
            except Exception as e:
                print(f"{COLOR_RED}[Error]{COLOR_RESET} Error in handle_client_socket: {e}")
                traceback.print_exc()
                break

        # Cleanup timer if connection is closing
        with data_lock:
            if timer:
                timer.cancel()
                print(f"{COLOR_YELLOW}[Timer]{COLOR_RESET} Existing timer canceled during cleanup.")
            if data_buffer:
                print(f"{COLOR_YELLOW}[Inference]{COLOR_RESET} Connection closing. Triggering final inference for remaining input.")
                # Run perform_inference in a separate thread
                threading.Thread(target=perform_inference, args=(' '.join(data_buffer).strip(),), daemon=True).start()
                data_buffer.clear()

        print(f"{COLOR_YELLOW}[Connection]{COLOR_RESET} Closing connection from {addr}")

# Function to handle commands from the client
def handle_command(command, client_socket):
    addr = client_socket.getpeername()
    print(f"{COLOR_MAGENTA}[Command Received]{COLOR_RESET} {command} from {addr}")

    # List all available tools
    tools = list_tools()

    # Check if the command matches any tool name (starting with '/')
    if command.startswith('/') and command[1:] in tools:
        tool_name = command[1:]  # Extract the tool name without the leading '/'
        print(f"{COLOR_CYAN}[Tool Call Initiated]{COLOR_RESET} Calling tool '{tool_name}'")

        # Call the tool and capture its response
        response = call_tool(tool_name, {})

        # Print the tool response in green if successful
        if "Error" not in response:
            print(f"{COLOR_GREEN}[Tool Response]{COLOR_RESET} {response}")
        else:
            print(f"{COLOR_RED}[Error]{COLOR_RESET} {response}")

        # Send the tool response to the client
        try:
            client_socket.sendall(f"{response}\n".encode())
            print(f"{COLOR_BLUE}[Info]{COLOR_RESET} Sent tool response to {addr}")
        except Exception as e:
            print(f"{COLOR_RED}[Error]{COLOR_RESET} Failed to send tool response to {addr}: {e}")
            traceback.print_exc()
    elif command.startswith('/info'):
        send_info(client_socket)
    elif command.startswith('/exit'):
        send_exit(client_socket)
    else:
        error_msg = f"Unknown command received: {command}\n"
        print(f"{COLOR_RED}[Error]{COLOR_RESET} {error_msg.strip()}")
        try:
            client_socket.sendall(error_msg.encode())
        except Exception as e:
            print(f"{COLOR_RED}[Error]{COLOR_RESET} Failed to send error message to {addr}: {e}")
            traceback.print_exc()

# Function to send configuration info to the orchestrator over the existing socket
def send_info(client_socket):
    try:
        response = f"{script_name}\n{script_uuid}\n"
        with config_lock:
            for key, value in config.items():
                response += f"{key}={value}\n"
        response += 'EOF\n'  # End the response with EOF to signal completion
        client_socket.sendall(response.encode())
        print(f"{COLOR_GREEN}[Info]{COLOR_RESET} Sent configuration info to {client_socket.getpeername()}")
    except Exception as e:
        print(f"{COLOR_RED}[Error]{COLOR_RESET} Failed to send info to {client_socket.getpeername()}: {e}")
        traceback.print_exc()

# Function to send exit acknowledgment and shutdown
def send_exit(client_socket):
    try:
        exit_message = "Exiting LLM Engine.\n"
        client_socket.sendall(exit_message.encode())
        print(f"{COLOR_YELLOW}[Shutdown]{COLOR_RESET} Received /exit command from {client_socket.getpeername()}. Shutting down.")
        cancel_event.set()
    except Exception as e:
        print(f"{COLOR_RED}[Error]{COLOR_RESET} Failed to send exit acknowledgment to {client_socket.getpeername()}: {e}")
        traceback.print_exc()

# Function to check if the required model is available
def check_and_download_model():
    """
    Checks if the required model is available locally. If not, attempts to pull it from the Ollama API.
    """
    model_name = config.get('model_name', 'llama3.2')  # Default model name
    model_name_sanitized = re.sub(r'[^\w.-]', '_', model_name)  # Sanitize model name for filenames
    ollama_models_dir = os.path.expanduser("~/.ollama/models/")
    model_path = os.path.join(ollama_models_dir, model_name_sanitized)

    # Check if the model exists locally
    if not os.path.exists(model_path):
        print(f"{COLOR_YELLOW}[Warning]{COLOR_RESET} Model '{model_name}' not found in {ollama_models_dir}. Attempting to pull...")

        # API endpoint and payload
        pull_url = f"{OLLAMA_URL}pull"
        payload = {
            "name": model_name,
            "stream": True  # Enable streaming to track progress
        }

        try:
            # Send the POST request to pull the model
            response = requests.post(pull_url, json=payload, stream=True)

            if response.status_code == 200:
                print(f"{COLOR_BLUE}[Info]{COLOR_RESET} Streaming pull progress for model '{model_name}':")
                for line in response.iter_lines():
                    if line:
                        progress = json.loads(line.decode('utf-8'))
                        print(json.dumps(progress, indent=2))
                        if progress.get("status") == "success":
                            print(f"{COLOR_GREEN}[Success]{COLOR_RESET} Model '{model_name}' pulled successfully.")
                            return
            else:
                print(f"{COLOR_RED}[Error]{COLOR_RESET} Failed to pull model '{model_name}'. Status code: {response.status_code}")
                print(f"Response: {response.text}")
                exit(1)

        except requests.exceptions.Timeout:
            print(f"{COLOR_RED}[Error]{COLOR_RESET} Request to pull the model '{model_name}' timed out.")
            exit(1)
        except requests.exceptions.ConnectionError:
            print(f"{COLOR_RED}[Error]{COLOR_RESET} Connection error occurred while contacting Ollama API.")
            exit(1)
        except Exception as e:
            print(f"{COLOR_RED}[Error]{COLOR_RESET} Exception during model pull: {e}")
            traceback.print_exc()
            exit(1)
    else:
        print(f"{COLOR_GREEN}[Info]{COLOR_RESET} Model '{model_name}' is already available.")

# Function to perform inference by sending a request to the Ollama API
def perform_inference(user_input):
    with threading.Lock():  # Ensure only one inference happens at a time
        model_name = config.get('model_name', 'llama3.2:3b')
        json_filtering = config.get('json_filtering', False)
        api_endpoint = config.get('api_endpoint', 'chat')
        stream = config.get('stream', False)
        use_tools = config.get('use_tools', True)  # Read the use_tools flag from the config

        # Prepare system prompt with tools if use_tools is enabled
        tools_definitions = []
        if use_tools:
            tools = list_tools()
            for tool in tools:
                tool_info = {
                    "type": "function",
                    "function": {
                        "name": tool,
                        "description": f"Function to execute {tool.replace('_', ' ').title()}",
                        "parameters": {
                            "type": "object",
                            "properties": {}
                        }
                    }
                }
                tools_definitions.append(tool_info)

        # Model parameters
        try:
            temperature = float(config.get('temperature', 0.7))
            top_p = float(config.get('top_p', 0.9))
            max_tokens = int(config.get('max_tokens', 150))
            repeat_penalty = float(config.get('repeat_penalty', 1.0))
        except ValueError as ve:
            print(f"{COLOR_RED}[Error]{COLOR_RESET} Invalid model parameter in config: {ve}")
            send_error_response(f"Error: Invalid model parameter in config: {ve}")
            return

        # Include recent chat history if it's a chat-based interaction
        chat_history = get_recent_chat_history()
        messages = [
            {"role": "system", "content": config.get("system_prompt", BASE_SYSTEM_PROMPT)}
        ]
        messages += [
            {"role": "user", "content": entry['user_input']} if i % 2 == 0 else {"role": "assistant", "content": entry['response']}
            for i, entry in enumerate(chat_history)
        ]
        messages.append({"role": "user", "content": user_input})

        # Prepare the payload for /api/chat
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "repeat_penalty": repeat_penalty,
            "stream": stream
        }

        # Only include tools if the use_tools flag is enabled
        if use_tools:
            payload["tools"] = tools_definitions

        headers = {
            "Content-Type": "application/json"
        }

        # Check if Ollama API is up before making the request
        if not check_ollama_api():
            send_error_response("Error: Ollama API is not available.")
            return

        try:
            api_url = f"http://localhost:11434/api/{api_endpoint}"
            print(f"{COLOR_BLUE}[Info]{COLOR_RESET} Sending request to Ollama API at {api_url} with payload:")
            print(json.dumps(payload, indent=2))

            if stream:
                # Streaming mode
                with requests.post(api_url, json=payload, headers=headers, stream=True, timeout=60) as response:
                    if response.status_code == 200:
                        for chunk in response.iter_lines():
                            if chunk:
                                chunk_data = json.loads(chunk.decode('utf-8'))
                                content = chunk_data.get('message', {}).get('content', '')
                                if content:
                                    send_response(content)
                                    print(content, end='', flush=True)
                    else:
                        send_error_response(f"Error: {response.status_code} - {response.text}")

            else:
                # Non-streaming mode
                response = requests.post(api_url, json=payload, headers=headers, timeout=60)
                if response.status_code == 200:
                    response_json = response.json()
                    content = response_json.get('message', {}).get('content', '')
                    if content:
                        print(f"{COLOR_GREEN}[Model Output]{COLOR_RESET}\n{content}\n")
                        append_to_chat_history({"user_input": user_input, "response": content})
                        send_response(content)
                else:
                    print(f"{COLOR_RED}[Error]{COLOR_RESET} Ollama API responded with status code: {response.status_code}")
                    send_error_response(f"Error: {response.status_code} - {response.text}")
                    
        except requests.exceptions.Timeout:
            print(f"{COLOR_RED}[Error]{COLOR_RESET} Request to Ollama API timed out.")
            send_error_response("Error: Request to Ollama API timed out.")
        except requests.exceptions.ConnectionError:
            print(f"{COLOR_RED}[Error]{COLOR_RESET} Connection error occurred while contacting Ollama API.")
            send_error_response("Error: Connection error with Ollama API.")
        except Exception as e:
            print(f"{COLOR_RED}[Error]{COLOR_RESET} Exception during model inference: {e}")
            traceback.print_exc()
            send_error_response(f"Error during model inference: {e}")



def process_arguments(arguments):
    """
    Recursively process and clean tool arguments, ensuring all values are valid.
    Args:
        arguments (dict): Arguments passed to the tool function.

    Returns:
        dict: Cleaned arguments.
    """
    if isinstance(arguments, dict):
        processed_args = {}
        for key, value in arguments.items():
            if isinstance(value, dict):
                processed_args[key] = process_arguments(value)  # Recursive processing for nested dicts
            else:
                processed_args[key] = value  # Add value as it is if not a dict
        return processed_args
    return arguments

def handle_tool_call(tool_call):
    """
    Execute the tool function and return its response.

    Args:
        tool_call (dict): Contains 'function' with 'name' and 'arguments'.

    Returns:
        str: Response from the tool function or error message.
    """
    try:
        # Store the incoming tool_call for reference
        tool_call_data = tool_call

        # Extract the function information and the arguments from the tool call
        function_info = tool_call_data.get("function", {})
        function_name = function_info.get("name")

        # Debugging: Print the incoming tool_call to see what is being passed
        print(f"[Debug] Tool call received: {json.dumps(tool_call_data, indent=2)}")
        print(f"[Debug] Extracted function name: {function_name}")

        # Ensure the function name is present
        if not function_name:
            return "Error: No function name provided in tool call."

        # Pass the entire tool_call_data to the function in model.functions
        if hasattr(functions, function_name):
            func = getattr(functions, function_name)
            print(f"[Tool Execution] Executing '{function_name}' with raw tool call data")
            
            # Execute the corresponding function in model.functions with the full JSON object
            response = func(tool_call_data)
            
            # Return the response from the function
            return response
        else:
            return f"Error: Function '{function_name}' not found."

    except Exception as e:
        return f"Error executing tool '{function_name}': {e}"



# Function to send a tool's response back to the API
def send_tool_response(tool_response, context):
    """
    Send the tool's response back to the API to continue the conversation.

    Args:
        tool_response (str): The response from the executed tool.
        context (list): The context from the previous API response (if any).
    """
    host = config.get('orchestrator_host', 'localhost')
    data_port = config.get('data_port', '6001')

    # Validate data_port
    try:
        data_port = int(data_port)
    except ValueError:
        print(f"{COLOR_RED}[Error]{COLOR_RESET} Invalid data port: {data_port}")
        return

    try:
        with socket.create_connection((host, data_port), timeout=5) as s:
            # Send script UUID first
            s.sendall(f"{script_uuid}\n".encode())
            # Send the tool response as a new user message
            new_message = {
                "role": "user",
                "content": tool_response
            }
            s.sendall(f"{json.dumps(new_message)}\n".encode())
            print(f"{COLOR_GREEN}[Sent]{COLOR_RESET} Tool response sent to orchestrator: {tool_response}")
    except ConnectionRefusedError:
        print(f"{COLOR_RED}[Error]{COLOR_RESET} Connection refused by orchestrator at {host}:{data_port}.")
    except socket.timeout:
        print(f"{COLOR_RED}[Error]{COLOR_RESET} Connection to orchestrator at {host}:{data_port} timed out.")
    except Exception as e:
        print(f"{COLOR_RED}[Error]{COLOR_RESET} Failed to send tool response to orchestrator: {e}")
        traceback.print_exc()

# Function to send response back to orchestrator
def send_response(output_data):
    # Connect to the orchestrator's data port
    host = config.get('orchestrator_host', 'localhost')
    data_port = config.get('data_port', '6001')

    # Validate data_port
    try:
        data_port = int(data_port)
    except ValueError:
        print(f"{COLOR_RED}[Error]{COLOR_RESET} Invalid data port: {data_port}")
        return

    try:
        with socket.create_connection((host, data_port), timeout=5) as s:
            # Send script UUID first
            s.sendall(f"{script_uuid}\n".encode())
            # Send the output data
            s.sendall(f"{output_data}\n".encode())
            print(f"{COLOR_GREEN}[Sent]{COLOR_RESET} Output sent to orchestrator: {output_data}")
    except ConnectionRefusedError:
        print(f"{COLOR_RED}[Error]{COLOR_RESET} Connection refused by orchestrator at {host}:{data_port}.")
    except socket.timeout:
        print(f"{COLOR_RED}[Error]{COLOR_RESET} Connection to orchestrator at {host}:{data_port} timed out.")
    except Exception as e:
        print(f"{COLOR_RED}[Error]{COLOR_RESET} Failed to send output to orchestrator: {e}")
        traceback.print_exc()

# Function to send error response back to orchestrator
def send_error_response(error_message):
    error_json = {
        "error": error_message
    }
    formatted_error = json.dumps(error_json, indent=2)
    print(f"{COLOR_RED}[Error Response]{COLOR_RESET} {formatted_error}")
    send_response(formatted_error)

# Optional terminal input fallback
def terminal_input():
    print("[Info] Running in terminal input mode. Type your input below:")
    while not cancel_event.is_set():
        try:
            user_input = input(">> ").strip()
            if user_input.lower() == "/exit":
                print("Exiting LLM Engine.")
                cancel_event.set()
                break
            elif user_input.startswith("/"):
                # Handle commands similarly to handle_command
                fake_socket = FakeSocket()
                handle_command(user_input, fake_socket)
            else:
                perform_inference(user_input)
        except KeyboardInterrupt:
            print("\n[Shutdown] LLM Engine terminated by user.")
            cancel_event.set()
        except Exception as e:
            print(f"{COLOR_RED}[Error]{COLOR_RESET} Unexpected error in terminal_input: {e}")
            traceback.print_exc()

# Function to load and map tools
def setup_tools():
    global tools_mapping
    tools_mapping = load_tools()
    if not tools_mapping:
        print(f"{COLOR_YELLOW}[Warning]{COLOR_RESET} No tools loaded. Proceeding without tools.")
    else:
        print(f"{COLOR_GREEN}[Info]{COLOR_RESET} Loaded tools: {', '.join(tools_mapping.keys())}")


# Main function
def main():

    # Read configuration from file
    read_config()

    # Parse command-line arguments
    parse_args()

    # Check and download the model if necessary
    check_and_download_model()

    # Write updated config to file (redundant if parse_args already does)
    # write_config(config)

    # Check if Ollama API is up before starting
    if not check_ollama_api():
        print(f"{COLOR_RED}🛑 Ollama is not running. Please start the Ollama API and try again.{COLOR_RESET}")
        exit(1)
    else:
        print(f"{COLOR_BLUE}[Info]{COLOR_RESET} Ollama API is available.")

    # Start the server to handle incoming connections
    start_server()


# Fake socket class to mimic sendall behavior for terminal input
class FakeSocket:
    def sendall(self, data):
        print(f"Response: {data.decode()}")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{COLOR_YELLOW}[Shutdown]{COLOR_RESET} LLM Engine terminated by user.")
    except Exception as e:
        print(f"{COLOR_RED}[Fatal Error]{COLOR_RESET} An unexpected error occurred: {e}")
        traceback.print_exc()
