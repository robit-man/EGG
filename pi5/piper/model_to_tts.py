#!/usr/bin/env python3
import os
import sys
import subprocess
import socket
import re
import json
import argparse
import threading
from queue import Queue, Empty
import shutil
import time
import logging
import psutil  # For CPU usage monitoring

#############################################
# Step 2: Ensure we're running inside a venv #
#############################################

VENV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv")
NEEDED_PACKAGES = ["requests", "num2words", "pyalsaaudio", "psutil"]

def in_venv():
    return (hasattr(sys, 'real_prefix') or
            (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix))

def setup_venv():
    # Create venv if it doesn't exist
    if not os.path.isdir(VENV_DIR):
        logging.info("Creating virtual environment...")
        subprocess.check_call([sys.executable, '-m', 'venv', VENV_DIR])

    pip_path = os.path.join(VENV_DIR, 'bin', 'pip')
    subprocess.check_call([pip_path, 'install'] + NEEDED_PACKAGES)

def relaunch_in_venv():
    # Relaunch inside venv python
    python_path = os.path.join(VENV_DIR, 'bin', 'python')
    os.execv(python_path, [python_path] + sys.argv)

if not in_venv():
    # Setup VENV and install packages
    setup_venv()
    # Relaunch the script within the VENV
    relaunch_in_venv()
else:
    #############################################
    # Step 3: Imports after venv set up          #
    #############################################
    
    import requests
    from num2words import num2words
    import alsaaudio  # For ALSA audio playback
    import psutil     # For CPU usage monitoring

    #############################################
    # Step 1: Setup Logging
    #############################################
    
    # Configure logging to include thread name and timestamp
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(threadName)s] [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            # You can add FileHandler here to log to a file if needed
        ]
    )
    
    #############################################
    # Step 4: Config Defaults & File
    #############################################
    
    DEFAULT_CONFIG = {
        "model": "llama3.2:3b",
        "stream": False,
        "format": None,
        "system": None,
        "raw": False,
        "history": None,
        "images": [],
        "tools": None,
        "options": {},
        "host": "0.0.0.0",
        "port": 6545,
        "tts_url": "http://localhost:6434",
        "ollama_url": "http://localhost:11434/api/chat"
    }
    CONFIG_PATH = "config.json"
    
    def load_config():
        if not os.path.exists(CONFIG_PATH):
            logging.info("No config.json found. Creating default config.json...")
            with open(CONFIG_PATH, 'w') as f:
                json.dump(DEFAULT_CONFIG, f, indent=2)
            return dict(DEFAULT_CONFIG)
        else:
            try:
                with open(CONFIG_PATH, 'r') as f:
                    cfg = json.load(f)
                # Merge with DEFAULT_CONFIG
                for key, value in DEFAULT_CONFIG.items():
                    if key not in cfg:
                        cfg[key] = value
                return cfg
            except Exception as e:
                logging.error(f"Error loading config.json: {e}. Using default settings.")
                return dict(DEFAULT_CONFIG)
    
    CONFIG = load_config()
    
    #############################################
    # Step 5: Parse Command-Line Arguments       #
    #############################################
    
    parser = argparse.ArgumentParser(description="Ollama Chat Server with TTS and advanced features.")
    
    parser.add_argument("--model", type=str, help="Model name to use.")
    parser.add_argument("--stream", action="store_true", help="Enable streaming responses from the model.")
    parser.add_argument("--format", type=str, help="Structured output format: 'json' or path to JSON schema file.")
    parser.add_argument("--system", type=str, help="System message override.")
    parser.add_argument("--raw", action="store_true", help="If set, use raw mode (no template).")
    parser.add_argument(
        "--history",
        type=str,
        nargs='?',
        const="chat.json",
        help="Path to a JSON file containing conversation history messages. Defaults to 'chat.json' if no path is provided."
    )
    parser.add_argument("--images", type=str, nargs='*', help="List of base64-encoded image files.")
    parser.add_argument("--tools", type=str, help="Path to a JSON file defining tools.")
    parser.add_argument("--option", action="append", help="Additional model parameters (e.g. --option temperature=0.7)")
    
    args = parser.parse_args()
    
    def merge_config_and_args(config, args):
        if args.model:
            config["model"] = args.model
        if args.stream:
            config["stream"] = True
        if args.format is not None:
            config["format"] = args.format
        if args.system is not None:
            config["system"] = args.system
        if args.raw:
            config["raw"] = True
        if args.history is not None:
            config["history"] = args.history
        if args.images is not None:
            config["images"] = args.images
        if args.tools is not None:
            config["tools"] = args.tools
        if args.option:
            for opt in args.option:
                if '=' in opt:
                    k, v = opt.split('=', 1)
                    k = k.strip()
                    v = v.strip()
                    if v.isdigit():
                        v = int(v)
                    else:
                        try:
                            v = float(v)
                        except ValueError:
                            pass
                    config["options"][k] = v
        return config
    
    CONFIG = merge_config_and_args(CONFIG, args)
    
    #############################################
    # Step 6: Load Optional Configurations       #
    #############################################
    
    def safe_load_json_file(path, default):
        if not path:
            return default
        if not os.path.exists(path):
            logging.warning(f"File '{path}' not found. Using default {default}.")
            if path == CONFIG["history"] and default == []:
                # Create empty history file
                with open(path, 'w') as f:
                    json.dump([], f)
            return default
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"Could not load '{path}': {e}. Using default {default}.")
            return default
    
    def load_format_schema(fmt):
        if not fmt:
            return None
        if fmt.lower() == "json":
            return "json"
        if os.path.exists(fmt):
            try:
                with open(fmt, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logging.warning(f"Could not load format schema from '{fmt}': {e}. Ignoring format.")
                return None
        else:
            logging.warning(f"Format file '{fmt}' not found. Ignoring format.")
            return None
    
    history_messages = safe_load_json_file(CONFIG["history"], [])
    tools_data = safe_load_json_file(CONFIG["tools"], None)
    format_schema = load_format_schema(CONFIG["format"])
    
    #############################################
    # Step 7: Ensure Ollama and Model are Installed #
    #############################################
    
    MAX_PULL_ATTEMPTS = 3  # Maximum number of attempts to pull the model
    
    def check_ollama_installed():
        """Check if Ollama is installed by verifying if the 'ollama' command is available."""
        ollama_path = shutil.which('ollama')
        return ollama_path is not None
    
    def install_ollama():
        """Install Ollama using the official installation script for Linux."""
        logging.info("Ollama not found. Attempting to install using the official installation script...")
        try:
            # The installation script might require interactive shell; using shell=True to handle the pipe
            subprocess.check_call('curl -fsSL https://ollama.com/install.sh | sh', shell=True, executable='/bin/bash')
            logging.info("Ollama installation initiated.")
        except subprocess.CalledProcessError as e:
            logging.error(f"Error installing Ollama: {e}")
            sys.exit(1)
    
    def wait_for_ollama():
        """Wait until Ollama service is up and running by checking GET /api/tags."""
        ollama_tags_url = "http://localhost:11434/api/tags"
        max_retries = 10
        for attempt in range(max_retries):
            try:
                response = requests.get(ollama_tags_url)
                if response.status_code == 200:
                    logging.info("Ollama service is up and running.")
                    return
            except requests.exceptions.RequestException:
                pass
            logging.info(f"Waiting for Ollama service to start... ({attempt + 1}/{max_retries})")
            time.sleep(2)
        logging.error("Ollama service did not start in time. Please check the Ollama installation.")
        sys.exit(1)
    
    def get_available_models():
        """Retrieve the list of available models from Ollama via GET /api/tags."""
        ollama_tags_url = "http://localhost:11434/api/tags"
        try:
            response = requests.get(ollama_tags_url)
            if response.status_code == 200:
                data = response.json()
                # Assuming the response has a 'models' field which is a list of model dictionaries
                available_models = data.get('models', [])
                logging.info("\nAvailable Models:")
                for model in available_models:
                    logging.info(f" - {model.get('name')}")
                # Extract model names for further processing
                model_names = [model.get('name') for model in available_models if 'name' in model]
                return model_names
            else:
                logging.error(f"Failed to retrieve models from Ollama: Status code {response.status_code}")
                return []
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching models from Ollama: {e}")
            return []
    
    def check_model_exists_in_tags(model_name):
        """Check if the specified model exists in Ollama's available models.
        Returns the actual model name (with suffix) if found, else None.
        """
        available_models = get_available_models()
        # Direct match
        if model_name in available_models:
            logging.info(f"\nModel '{model_name}' is available in Ollama's tags.")
            return model_name
        # Check for ':latest' suffix
        model_latest = f"{model_name}:latest"
        if model_latest in available_models:
            logging.info(f"\nModel '{model_latest}' is available in Ollama's tags.")
            return model_latest
        # Otherwise, not found
        logging.error(f"\nModel '{model_name}' does not exist in Ollama's available tags.")
        return None
    
    def check_model_installed(model_name):
        """Check if the specified model is already installed.
        Returns True if installed, False otherwise.
        """
        try:
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, check=True)
            models = result.stdout.splitlines()
            # Normalize model names by stripping whitespace
            models = [model.strip() for model in models]
            # Check for exact match
            if model_name in models:
                return True
            # If model_name ends with ':latest', check without the suffix
            if model_name.endswith(':latest'):
                base_model = model_name.rsplit(':', 1)[0]
                if base_model in models:
                    return True
            return False
        except subprocess.CalledProcessError as e:
            logging.error(f"Error checking installed models: {e}")
            sys.exit(1)
    
    def pull_model(model_name):
        """Pull the specified model using Ollama with retry logic."""
        logging.info(f"\nModel '{model_name}' not found. Attempting to pull the model...")
        attempts = 0
        while attempts < MAX_PULL_ATTEMPTS:
            try:
                subprocess.check_call(['ollama', 'pull', model_name])
                logging.info(f"Model '{model_name}' has been successfully pulled.")
                return True
            except subprocess.CalledProcessError as e:
                attempts += 1
                logging.error(f"Attempt {attempts}: Error pulling model '{model_name}': {e}")
                if attempts < MAX_PULL_ATTEMPTS:
                    logging.info("Retrying to pull the model...")
                    time.sleep(2)  # Wait before retrying
                else:
                    logging.error(f"Failed to pull model '{model_name}' after {MAX_PULL_ATTEMPTS} attempts.")
                    return False
    
    def ensure_ollama_and_model():
        """Ensure that Ollama is installed and the specified model is available."""
        if not check_ollama_installed():
            install_ollama()
            # After installation, ensure the 'ollama' command is available
            if not check_ollama_installed():
                logging.error("Ollama installation failed or 'ollama' command is not in PATH.")
                sys.exit(1)
        else:
            logging.info("Ollama is already installed.")
    
        # Wait for Ollama service to be ready by checking GET /api/tags
        wait_for_ollama()
    
        # Check if the model exists in tags
        model_name = CONFIG["model"]
        model_actual_name = check_model_exists_in_tags(model_name)
        if not model_actual_name:
            logging.error(f"Model '{model_name}' does not exist in Ollama's available tags. Cannot proceed.")
            sys.exit(1)
    
        # Check if the model is installed
        if not check_model_installed(model_actual_name):
            # Attempt to pull the model with retry logic
            pull_successful = pull_model(model_actual_name)
            if not pull_successful:
                logging.warning(f"Proceeding as if model '{model_actual_name}' is installed despite failed pull attempts.")
        else:
            logging.info(f"Model '{model_actual_name}' is already installed.")
    
    # Call the function to ensure Ollama and the model are installed
    ensure_ollama_and_model()
    
    #############################################
    # Step 8: Ollama chat interaction
    #############################################
    
    OLLAMA_CHAT_URL = CONFIG["ollama_url"]
    
    def convert_numbers_to_words(text):
        """
        Convert all standalone numbers in text to their word equivalents.
        """
        def replace_num(match):
            number_str = match.group(0)
            try:
                number_int = int(number_str)
                return num2words(number_int)
            except ValueError:
                return number_str
        return re.sub(r'\b\d+\b', replace_num, text)
    
    def build_payload(user_message):
        messages = []
        if CONFIG["system"]:
            messages.append({"role": "system", "content": CONFIG["system"]})
        messages.extend(history_messages)
        messages.append({"role": "user", "content": user_message})
    
        payload = {
            "model": CONFIG["model"],
            "messages": messages,
            "stream": CONFIG["stream"]
        }
    
        if format_schema:
            payload["format"] = format_schema
        if CONFIG["raw"]:
            payload["raw"] = True
        if CONFIG["images"]:
            if payload["messages"] and payload["messages"][-1]["role"] == "user":
                payload["messages"][-1]["images"] = CONFIG["images"]
        if tools_data:
            payload["tools"] = tools_data
        if CONFIG["options"]:
            payload["options"] = CONFIG["options"]
    
        return payload
    
    #############################################
    # Step 9: Dedicated Worker Threads for Ollama and TTS
    #############################################
    
    # Initialize Queues for inter-thread communication
    ollama_queue = Queue()
    tts_queue = Queue()
    
    # Dictionary to map request IDs to response queues
    response_dict = {}
    response_dict_lock = threading.Lock()
    request_id_counter = 0
    request_id_lock = threading.Lock()
    
    def ollama_worker():
        """
        Worker thread that processes messages from the Ollama queue.
        """
        while True:
            try:
                request_id, user_message = ollama_queue.get(timeout=1)  # Wait for 1 second
                if request_id is None and user_message is None:
                    logging.info("Ollama Worker: Received shutdown signal.")
                    break
                logging.info(f"Ollama Worker: Processing message: {user_message}")
                response_content = ""
                payload = build_payload(user_message)
                if CONFIG["stream"]:
                    try:
                        with requests.post(OLLAMA_CHAT_URL, json=payload, headers={"Content-Type": "application/json"}, stream=True) as r:
                            r.raise_for_status()
                            buffer = ""
                            sentence_endings = re.compile(r'[.?!]+')
                            for line in r.iter_lines():
                                if line:
                                    obj = json.loads(line.decode('utf-8'))
                                    msg = obj.get("message", {})
                                    content = msg.get("content", "")
                                    done = obj.get("done", False)
                                    response_content += content
                                    buffer += content
                                    # Check for sentence endings
                                    while True:
                                        match = sentence_endings.search(buffer)
                                        if not match:
                                            break
                                        end_index = match.end()
                                        sentence = buffer[:end_index].strip()
                                        buffer = buffer[end_index:].strip()
                                        if sentence:
                                            tts_queue.put(sentence)
                                    if done:
                                        break
                    except Exception as e:
                        logging.error(f"Error during streaming inference: {e}")
                        response_content = ""
                else:
                    try:
                        r = requests.post(OLLAMA_CHAT_URL, json=payload, headers={"Content-Type": "application/json"})
                        r.raise_for_status()
                        data = r.json()
                        response_content = data.get("message", {}).get("content", "")
                        # Split into sentences
                        sentence_endings = re.compile(r'[.?!]+')
                        buffer = response_content
                        while True:
                            match = sentence_endings.search(buffer)
                            if not match:
                                break
                            end_index = match.end()
                            sentence = buffer[:end_index].strip()
                            buffer = buffer[end_index:].strip()
                            if sentence:
                                tts_queue.put(sentence)
                        # Handle leftover
                        leftover = buffer.strip()
                        if leftover:
                            tts_queue.put(leftover)
                    except Exception as e:
                        logging.error(f"Error during non-stream inference: {e}")
                        response_content = ""
    
                # Send back the response to the client via the response queue
                with response_dict_lock:
                    if request_id in response_dict:
                        response_queue = response_dict.pop(request_id)
                        response_queue.put(response_content)
    
                # Update history
                update_history(user_message, response_content)
    
            except Empty:
                continue
            except Exception as e:
                logging.error(f"Ollama Worker: Unexpected error: {e}")
    
    def tts_worker():
        """
        Worker thread that processes sentences from the TTS queue.
        """
        while True:
            try:
                sentence = tts_queue.get(timeout=1)  # Wait for 1 second
                if sentence is None:
                    logging.info("TTS Worker: Received shutdown signal.")
                    break
                logging.info(f"TTS Worker: Processing sentence: {sentence}")
                synthesize_and_play(sentence)
            except Empty:
                continue
            except Exception as e:
                logging.error(f"TTS Worker: Error processing sentence: {e}")
    
    def start_ollama_thread():
        """
        Start the Ollama worker thread.
        """
        ollama_thread = threading.Thread(target=ollama_worker, daemon=True, name="OllamaWorker")
        ollama_thread.start()
        logging.info("Ollama Worker: Started.")
    
    def start_tts_thread():
        """
        Start the TTS worker thread.
        """
        tts_thread = threading.Thread(target=tts_worker, daemon=True, name="TTSWorker")
        tts_thread.start()
        logging.info("TTS Worker: Started.")
    
    def synthesize_and_play(sentence):
        """
        Send the sentence to the TTS engine.
        Playback is handled elsewhere.
        """
        sentence = sentence.strip()
        if not sentence:
            return
        try:
            payload = {"prompt": sentence}
            logging.info(f"Sending TTS request with prompt: {sentence}")
            with requests.post(CONFIG["tts_url"], json=payload, stream=True, timeout=10) as response:
                if response.status_code != 200:
                    logging.warning(f"TTS received status code {response.status_code}")
                    try:
                        error_msg = response.json().get('error', 'No error message provided.')
                        logging.warning(f"TTS error: {error_msg}")
                    except:
                        logging.warning("No JSON error message provided for TTS.")
                    return
        
                # Forward the audio data to the TTS engine
                # Assuming the TTS engine handles playback internally
                for chunk in response.iter_content(chunk_size=4096):
                    if chunk:
                        # If needed, process or log the audio data here
                        pass  # No action since playback is handled elsewhere
            logging.info("TTS request completed successfully.")
        except requests.exceptions.Timeout:
            logging.error("TTS request timed out.")
        except requests.exceptions.ConnectionError as ce:
            logging.error(f"Connection error during TTS request: {ce}")
        except Exception as e:
            logging.error(f"Unexpected error during TTS: {e}")
    
    def update_history(user_message, assistant_message):
        if not CONFIG["history"]:
            return
        current_history = safe_load_json_file(CONFIG["history"], [])
        current_history.append({"role": "user", "content": user_message})
        current_history.append({"role": "assistant", "content": assistant_message})
        try:
            with open(CONFIG["history"], 'w') as f:
                json.dump(current_history, f, indent=2)
        except Exception as e:
            logging.warning(f"Could not write to history file {CONFIG['history']}: {e}")
    
    #############################################
    # Step 10: Monitoring CPU Usage
    #############################################
    
    def monitor_cpu_usage(interval=5):
        """
        Monitor CPU usage at regular intervals and log it.
        Runs in a separate daemon thread.
        """
        while True:
            cpu_percent = psutil.cpu_percent(interval=interval)
            logging.info(f"CPU Usage: {cpu_percent}%")
            time.sleep(interval)
    
    #############################################
    # Step 11: Server Handling without Threads  #
    #############################################
    
    HOST = CONFIG["host"]
    PORT = CONFIG["port"]
    
    def start_server():
        # Start TTS and Ollama worker threads
        logging.info("Starting TTS Worker...")
        start_tts_thread()
        logging.info("Starting Ollama Worker...")
        start_ollama_thread()
    
        # Start CPU usage monitoring
        cpu_monitor_thread = threading.Thread(target=monitor_cpu_usage, daemon=True, name="CPUMonitor")
        cpu_monitor_thread.start()
        logging.info("CPU Usage Monitor: Started.")
    
        # Start the main server socket
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            server.bind((HOST, PORT))
        except Exception as e:
            logging.error(f"Error binding to {HOST}:{PORT} - {e}. Using defaults: 0.0.0.0:64162")
            HOST_D = '0.0.0.0'
            PORT_D = 64162
            try:
                server.bind((HOST_D, PORT_D))
                logging.info(f"Server bound to {HOST_D}:{PORT_D}")
            except Exception as e:
                logging.error(f"Failed to bind to default host and port: {e}")
                sys.exit(1)
    
        server.listen(5)
        logging.info(f"Listening for incoming connections on {HOST}:{PORT}...")
    
        # Retrieve the actual model name to use
        model_actual_name = CONFIG["model"]
    
        try:
            while True:
                try:
                    client_sock, addr = server.accept()
                    logging.info(f"Accepted connection from {addr}")
                    
                    # Handle client connection sequentially
                    data = client_sock.recv(65536)
                    if not data:
                        logging.info(f"No data from {addr}, closing connection.")
                        client_sock.close()
                        continue
                    user_message = data.decode('utf-8').strip()
                    if not user_message:
                        logging.info(f"Empty prompt from {addr}, ignoring.")
                        client_sock.close()
                        continue
                    logging.info(f"Received prompt from {addr}: {user_message}")
    
                    # Generate a unique request ID
                    with request_id_lock:
                        global request_id_counter
                        request_id_counter += 1
                        request_id = request_id_counter
    
                    # Create a response queue for this request
                    response_queue = Queue()
                    with response_dict_lock:
                        response_dict[request_id] = response_queue
    
                    # Enqueue the message to the Ollama worker
                    ollama_queue.put((request_id, user_message))
    
                    # Wait for the response from the Ollama worker
                    try:
                        response_content = response_queue.get(timeout=60)  # Wait up to 60 seconds
                    except Empty:
                        logging.error("Timeout waiting for Ollama response.")
                        response_content = "I'm sorry, I couldn't process your request at this time."
    
                    # Send back the response to the client
                    client_sock.sendall(response_content.encode('utf-8'))
    
                    # **Removed: Enqueuing sentences to tts_queue from the main thread**
    
                    # Update history
                    update_history(user_message, response_content)
    
                    # Close the client socket
                    client_sock.close()
                    logging.info(f"Connection with {addr} closed.")
    
                except KeyboardInterrupt:
                    logging.info("\nInterrupt received, shutting down server.")
                    break
                except Exception as e:
                    logging.error(f"Error handling client connection: {e}")
        finally:
            # Stop accepting new connections
            server.close()
            logging.info("Server socket closed.")
    
            # Stop worker threads
            logging.info("Stopping TTS Worker...")
            tts_queue.put(None)  # Signal TTS worker to exit
            logging.info("Stopping Ollama Worker...")
            ollama_queue.put((None, None))  # Signal Ollama worker to exit
    
            # Wait a moment to allow workers to shutdown
            time.sleep(2)
    
            logging.info("Shutting down complete.")
    
    if __name__ == "__main__":
        start_server()
