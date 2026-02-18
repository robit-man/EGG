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
import multiprocessing  # For handling inference processes

PSUTIL_AVAILABLE = False
ALSAAUDIO_AVAILABLE = False

#############################################
# Utility Functions
#############################################

def is_connected(host="8.8.8.8", port=53, timeout=3):
    """
    Check internet connectivity by attempting to connect to a well-known DNS server.
    Returns True if connected, False otherwise.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            sock.connect((host, port))
        return True
    except socket.error:
        return False

#############################################
# Step 2: Ensure we're running inside a venv #
#############################################

VENV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv")
NEEDED_PACKAGES = ["requests", "num2words", "pyalsaaudio", "psutil"]

def in_venv():
    return (hasattr(sys, 'real_prefix') or
            (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix))

def setup_venv(online=True):
    # Create venv if it doesn't exist
    if not os.path.isdir(VENV_DIR):
        logging.info("Creating virtual environment...")
        try:
            subprocess.check_call([sys.executable, '-m', 'venv', VENV_DIR])
            logging.info("Virtual environment created successfully.")
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to create virtual environment: {e}")
            if not online:
                logging.warning("Proceeding without setting up virtual environment due to offline mode.")
            else:
                sys.exit(1)

    # Determine pip path based on OS
    pip_path = os.path.join(VENV_DIR, 'bin', 'pip') if os.name != 'nt' else os.path.join(VENV_DIR, 'Scripts', 'pip.exe')

    if online:
        try:
            logging.info("Installing required packages...")
            subprocess.check_call([pip_path, 'install'] + NEEDED_PACKAGES)
            logging.info("Required packages installed successfully.")
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to install required packages: {e}")
            logging.warning("Proceeding without installing all packages. Ensure required packages are installed.")
    else:
        logging.warning("Offline mode: Skipping package installation. Ensure required packages are installed.")

def relaunch_in_venv():
    # Relaunch inside venv python
    python_path = os.path.join(VENV_DIR, 'bin', 'python') if os.name != 'nt' else os.path.join(VENV_DIR, 'Scripts', 'python.exe')
    if os.path.exists(python_path):
        logging.info("Relaunching script inside the virtual environment...")
        os.execv(python_path, [python_path] + sys.argv)
    else:
        logging.error("Virtual environment Python executable not found.")
        sys.exit(1)

if not in_venv():
    # Determine online status
    ONLINE = is_connected()
    if not ONLINE:
        logging.warning("No internet connection detected. Operating in offline mode.")
    # Setup VENV and install packages if online
    setup_venv(online=ONLINE)
    # Always relaunch in venv, regardless of online/offline status
    relaunch_in_venv()
else:
    #############################################
    # Step 3: Imports after venv set up          #
    #############################################
    
    try:
        import requests
        from num2words import num2words
    except ImportError as e:
        logging.error(f"Failed to import required modules: {e}")
        logging.error("Ensure all required packages are installed in the virtual environment.")
        sys.exit(1)

    try:
        import alsaaudio  # Optional ALSA module
        ALSAAUDIO_AVAILABLE = True
    except ImportError:
        alsaaudio = None
        ALSAAUDIO_AVAILABLE = False

    try:
        import psutil  # Optional CPU usage monitoring
        PSUTIL_AVAILABLE = True
    except ImportError:
        psutil = None
        PSUTIL_AVAILABLE = False

    #############################################
    # Step 1: Setup Logging
    #############################################
    
    # Configure logging to include thread name and timestamp
    logging.basicConfig(
        level=logging.INFO,  # Can set to DEBUG for more detailed logs
        format='[%(asctime)s] [%(threadName)s] [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            # Uncomment the following line to log to a file
            # logging.FileHandler("server.log")
        ]
    )
    if not ALSAAUDIO_AVAILABLE:
        logging.warning("pyalsaaudio not available; continuing without direct ALSA hooks.")
    if not PSUTIL_AVAILABLE:
        logging.warning("psutil not available; CPU monitor will use a fallback value.")
    
    #############################################
    # Step 4: Config Defaults & File
    #############################################
    
    DEFAULT_CONFIG = {
        "model": "llama3.2:3b",
        "stream": True,
        "format": None,
        "system": ("You are a highly efficient and curious small language model running on a Raspberry Pi 5. "
                   "Your primary goal is to provide clear, concise, and actionable responses. Avoid disclaimers about "
                   "your limitations or unnecessary verbosity—focus on being personable and engaging. Express curiosity "
                   "about the context when appropriate, and always strive to be helpful, resourceful, and to the point. "
                   "Remember, your brevity is your strength. You are getting prompts sent to you from 'whisper' speech "
                   "recognition, which sometimes sends you messages in perenthesis like (background noise) or "
                   "(speaking in foreign language) or (door opening), These indicate detected content and you should respond "
                   "naturally, without treating these like vocal input. Also note that everything you receive as a prompt "
                   "is actually coming from a microphone and interpretation of the input from it. As such, you should reply "
                   "to everything as if you actually have ears and the ability to hear as you have a transducer at your input "
                   "helping aid in your understanding of what is around you! Also, any use of end of sentence delimiters like "
                   "periods or other punctuation truncates what is then sent to the TTS generator, so please use punctuation in "
                   "a strategic way to prevent run on sentences that take a long time to generate text, and also when making "
                   "lists, avoid using periods after the number delimiters to prevent treating the list position as its own "
                   "isolated sentence passed to TTS. When responding with the intention of laughing or other expressions, "
                   "produce content like 'hahaha' or 'awwww' or 'ahhhhh', not with (laughing) or other expressions that you receive."),
        "raw": False,
        "history": "chat.json",
        "images": [],
        "tools": None,
        "options": {},
        "host": "0.0.0.0",
        "port": 6545,
        "tts_host": "127.0.0.1",
        "tts_port": 6434,
        "tts_url": "http://localhost:6434",
        "ollama_url": "http://localhost:11434/api/chat",
        "max_history_messages": 3,  # New Configuration Parameter
        "offline_mode": False  # Default offline mode
    }
    CONFIG_PATH = "llm_bridge_config.json"
    AUDIO_ROUTER_CONFIG_PATH = "audio_router_config.json"
    
    def load_config():
        if not os.path.exists(CONFIG_PATH):
            logging.info(f"No {CONFIG_PATH} found. Creating default config file...")
            try:
                with open(CONFIG_PATH, 'w') as f:
                    json.dump(DEFAULT_CONFIG, f, indent=2)
                return dict(DEFAULT_CONFIG)
            except Exception as e:
                logging.error(f"Failed to create {CONFIG_PATH}: {e}")
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
                logging.error(f"Error loading {CONFIG_PATH}: {e}. Using default settings.")
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
    
    # New Command-Line Argument for max_history_messages
    parser.add_argument("--max-history", type=int, help="Maximum number of recent chat history messages to recall.")
    
    # New Argument to Force Offline Mode
    parser.add_argument("--offline", action="store_true", help="Force the script to operate in offline mode.")
    
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
        # Handle the new --max-history argument
        if args.max_history is not None:
            config["max_history_messages"] = args.max_history
        # Handle the new --offline argument
        config["offline_mode"] = args.offline
        return config

    def _get_nested(data, path, default=None):
        current = data
        for key in str(path or "").split("."):
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current

    def apply_audio_router_overrides(config):
        try:
            with open(AUDIO_ROUTER_CONFIG_PATH, "r", encoding="utf-8") as fp:
                payload = json.load(fp)
            if not isinstance(payload, dict):
                return config
        except Exception:
            return config

        tts_host = str(_get_nested(payload, "audio_router.integrations.tts_host", config.get("tts_host", "127.0.0.1"))).strip()
        tts_port = _get_nested(payload, "audio_router.integrations.tts_port", config.get("tts_port", 6434))
        try:
            tts_port = int(tts_port)
        except Exception:
            tts_port = int(config.get("tts_port", 6434))

        if tts_host:
            config["tts_host"] = tts_host
        config["tts_port"] = tts_port
        config["tts_url"] = f"http://{config['tts_host']}:{config['tts_port']}"
        return config
    
    CONFIG = merge_config_and_args(CONFIG, args)
    CONFIG = apply_audio_router_overrides(CONFIG)
    
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
                try:
                    with open(path, 'w') as f:
                        json.dump([], f)
                    logging.info(f"Created empty history file at '{path}'.")
                except Exception as e:
                    logging.warning(f"Could not create history file '{path}': {e}")
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
    
    # Use absolute path for history
    history_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), CONFIG["history"])
    history_messages = safe_load_json_file(history_path, [])
    tools_data = safe_load_json_file(CONFIG["tools"], None)
    format_schema = load_format_schema(CONFIG["format"])
    
    #############################################
    # Step 7: Ensure Ollama and Model are Installed #
    #############################################
    
    # Removed all functions and checks related to model availability and installation.
    # The script will now always attempt to use the model specified in the configuration without verifying its availability.
    # This includes removing the 'ensure_ollama_and_model' function and setting 'MODEL_AVAILABLE' to True unconditionally.
    
    # Set MODEL_AVAILABLE to True to always enable model inference.
    MODEL_AVAILABLE = True
    logging.info(f"Configured to use model: {CONFIG['model']} regardless of availability.")
    
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
        
        # Truncate history_messages based on max_history_messages
        if CONFIG.get("max_history_messages"):
            # Ensure we have an even number of messages (user and assistant)
            # If odd, remove the oldest user message without a corresponding assistant message
            max_messages = CONFIG["max_history_messages"]
            if len(history_messages) > max_messages:
                # Slice the last max_messages messages
                truncated_history = history_messages[-max_messages:]
                # Ensure even number of messages for user-assistant pairs
                if len(truncated_history) % 2 != 0:
                    truncated_history = truncated_history[1:]
                messages.extend(truncated_history)
            else:
                messages.extend(history_messages)
        else:
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
    
    # Initialize Queues for inter-thread and inter-process communication
    ollama_queue = Queue()
    tts_queue = Queue()
    inference_queue = multiprocessing.Queue()  # For inference process output
    
    # Dictionary to map request IDs to response queues
    response_dict = {}
    response_dict_lock = threading.Lock()
    request_id_counter = 0
    request_id_lock = threading.Lock()
    
    # Initialize a Lock for history updates to ensure thread safety
    history_lock = threading.Lock()
    
    # Function to handle inter-process communication from inference processes to TTS queue
    def inference_to_tts_handler():
        while True:
            try:
                sentence = inference_queue.get()
                if sentence == "__SHUTDOWN__":
                    logging.info("Inference to TTS Handler: Received shutdown signal.")
                    break  # Sentinel to stop the thread
                # Append assistant sentence to history
                update_history("assistant", sentence)
                # Enqueue sentence to TTS queue
                tts_queue.put(sentence)
                logging.debug(f"Inference to TTS Handler: Enqueued sentence to TTS: {sentence}")
            except Exception as e:
                logging.error(f"Inference to TTS Handler: {e}")
    
    # Start the inference_to_tts_handler thread
    inference_to_tts_thread = threading.Thread(target=inference_to_tts_handler, daemon=True, name="InferenceToTTSHandler")
    inference_to_tts_thread.start()
    logging.info("InferenceToTTSHandler: Started.")

    def _cpu_percent(interval=1.0):
        if PSUTIL_AVAILABLE and psutil is not None:
            try:
                return float(psutil.cpu_percent(interval=interval))
            except Exception:
                pass
        try:
            time.sleep(max(0.0, float(interval)))
        except Exception:
            time.sleep(0.1)
        return 0.0
    
    def ollama_worker():
        """
        Worker thread that processes messages from the Ollama queue.
        Manages inference processes and handles CPU usage constraints.
        """
        current_inference_process = None  # Track the current inference process
        while True:
            try:
                request_id, user_message = ollama_queue.get(timeout=1)  # Wait for 1 second
                if request_id is None and user_message is None:
                    logging.info("Ollama Worker: Received shutdown signal.")
                    # Terminate any ongoing inference process
                    if current_inference_process and current_inference_process.is_alive():
                        logging.info("Ollama Worker: Terminating ongoing inference process.")
                        current_inference_process.terminate()
                        current_inference_process.join()
                        logging.info("Ollama Worker: Ongoing inference process terminated.")
                    break
                logging.info(f"Ollama Worker: Received new prompt: {user_message}")
                
                # Check if an inference process is already running
                if current_inference_process and current_inference_process.is_alive():
                    cpu_usage = _cpu_percent(interval=1)
                    logging.info(f"Ollama Worker: Current CPU usage is {cpu_usage}%.")
                    if cpu_usage > 50:
                        logging.info("Ollama Worker: CPU usage > 50%. Terminating current inference.")
                        current_inference_process.terminate()
                        current_inference_process.join()
                        logging.info("Ollama Worker: Current inference terminated.")
                    else:
                        logging.info("Ollama Worker: CPU usage <= 50%. Waiting for current inference to finish.")
                        # Wait until the current inference finishes
                        current_inference_process.join()
                        logging.info("Ollama Worker: Previous inference completed.")
                
                # Start a new inference process for the new prompt
                logging.info("Ollama Worker: Starting new inference process.")
                current_inference_process = multiprocessing.Process(
                    target=inference_process,
                    args=(user_message, inference_queue)
                )
                current_inference_process.start()
                logging.info(f"Ollama Worker: Inference process started with PID {current_inference_process.pid}.")
                
                # Continue listening for new prompts
                
            except Empty:
                continue
            except Exception as e:
                logging.error(f"Ollama Worker: Unexpected error: {e}")
    
    def inference_process(user_message, output_queue):
        """
        Function to handle inference in a separate process.
        Sends sentences to the output_queue for TTS.
        """
        try:
            payload = build_payload(user_message)
            if CONFIG["stream"]:
                # Streaming response
                with requests.post(
                    OLLAMA_CHAT_URL,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    stream=True
                ) as r:
                    r.raise_for_status()
                    buffer = ""
                    sentence_endings = re.compile(r'[.?!]+')
                    for line in r.iter_lines():
                        if line:
                            try:
                                obj = json.loads(line.decode('utf-8'))
                                msg = obj.get("message", {})
                                content = msg.get("content", "")
                                # Remove colons and asterisks
                                content = content.replace(':', '').replace('*', '')
                                done = obj.get("done", False)
                                buffer += content
                                # Split into sentences
                                while True:
                                    match = sentence_endings.search(buffer)
                                    if not match:
                                        break
                                    end_index = match.end()
                                    sentence = buffer[:end_index].strip()
                                    buffer = buffer[end_index:].strip()
                                    if sentence:
                                        output_queue.put(sentence)
                                        logging.debug(f"Inference Process: Enqueued sentence to TTS: {sentence}")
                                if done:
                                    if buffer:
                                        output_queue.put(buffer.strip())
                                        logging.debug(f"Inference Process: Enqueued final sentence to TTS: {buffer.strip()}")
                                    break
                            except json.JSONDecodeError as e:
                                logging.error(f"Inference Process: Invalid JSON received: {e}")
                            except Exception as e:
                                logging.error(f"Inference Process: Error processing stream: {e}")
            else:
                # Non-streaming response
                r = requests.post(
                    OLLAMA_CHAT_URL,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                r.raise_for_status()
                data = r.json()
                response_content = data.get("message", {}).get("content", "")
                # Remove colons and asterisks
                response_content = response_content.replace(':', '').replace('*', '')
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
                        output_queue.put(sentence)
                        logging.debug(f"Inference Process: Enqueued sentence to TTS: {sentence}")
                # Handle leftover
                leftover = buffer.strip()
                if leftover:
                    output_queue.put(leftover)
                    logging.debug(f"Inference Process: Enqueued final sentence to TTS: {leftover}")
        except Exception as e:
            logging.error(f"Inference Process: Unexpected error: {e}")
        # Removed output_queue.put(None)
        # The inference_to_tts_handler will be shutdown separately
    
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

    # Added synthesize_and_play function to handle TTS requests
    def synthesize_and_play(sentence):
        """
        Send the sentence to the TTS engine.
        Playback is handled by output.py through voice_server.py.
        """
        sentence = sentence.strip()
        if not sentence:
            return
        try:
            payload = {"prompt": sentence}
            tts_host = str(CONFIG.get("tts_host", "127.0.0.1")).strip() or "127.0.0.1"
            try:
                tts_port = int(CONFIG.get("tts_port", 6434))
            except Exception:
                tts_port = 6434
            logging.info(f"Forwarding TTS prompt to {tts_host}:{tts_port}")
            with socket.create_connection((tts_host, tts_port), timeout=10) as sock:
                sock.sendall(json.dumps(payload).encode("utf-8"))
            logging.info("TTS prompt forwarded successfully.")
        except socket.timeout:
            logging.error("TTS socket request timed out.")
        except OSError as ce:
            logging.error(f"Connection error during TTS socket forward: {ce}")
        except Exception as e:
            logging.error(f"Unexpected error during TTS: {e}")

    def update_history(role, content):
        """
        Append a single message to the chat history.
        Only appends if the role is 'user' or 'assistant' and content is provided.
        """
        if not role or not content:
            return
        with history_lock:
            current_history = safe_load_json_file(history_path, [])
            current_history.append({"role": role, "content": content})
            try:
                with open(history_path, 'w') as f:
                    json.dump(current_history, f, indent=2)
                logging.info(f"History updated in '{history_path}' with {role} message.")
            except Exception as e:
                logging.warning(f"Could not write to history file {history_path}: {e}")

    #############################################
    # Step 10: Monitoring CPU Usage
    #############################################
    
    def monitor_cpu_usage(interval=5):
        """
        Monitor CPU usage at regular intervals and log it.
        Runs in a separate daemon thread.
        """
        while True:
            cpu_percent = _cpu_percent(interval=interval)
            logging.info(f"CPU Usage: {cpu_percent}%")

    #############################################
    # Step 11: Receiver Thread for Incoming Messages #
    #############################################
    
    def receiver_thread(host, port):
        """
        Dedicated thread to listen for incoming socket connections and receive messages asynchronously.
        Continuously retries to bind to the specified port until successful.
        """
        while True:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                server_socket.bind((host, port))
                logging.info(f"Receiver Thread: Bound to {host}:{port}")
                server_socket.listen(5)
                logging.info(f"Receiver Thread: Listening for incoming connections on {host}:{port}...")
                break  # Successfully bound and listening
            except Exception as e:
                logging.error(f"Receiver Thread: Failed to bind to {host}:{port} - {e}. Retrying in 5 seconds...")
                try:
                    server_socket.close()
                except:
                    pass
                time.sleep(5)  # Wait before retrying

        while True:
            try:
                client_sock, addr = server_socket.accept()
                logging.info(f"Receiver Thread: Accepted connection from {addr}")
                # Start a new thread to handle the client
                client_handler = threading.Thread(
                    target=handle_client,
                    args=(client_sock, addr),
                    daemon=True,
                    name=f"ClientHandler-{addr}"
                )
                client_handler.start()
            except Exception as e:
                logging.error(f"Receiver Thread: Error accepting connections: {e}")
                # Continue accepting new connections
                continue

        server_socket.close()
        logging.info("Receiver Thread: Server socket closed.")

    def handle_client(client_sock, addr):
        """
        Handle individual client connections.
        """
        try:
            data = client_sock.recv(65536)
            if not data:
                logging.info(f"ClientHandler: No data from {addr}, closing connection.")
                client_sock.close()
                return
            user_message = data.decode('utf-8').strip()
            if not user_message:
                logging.info(f"ClientHandler: Empty prompt from {addr}, ignoring.")
                client_sock.close()
                return
            logging.info(f"ClientHandler: Received prompt from {addr}: {user_message}")

            # Append user message to history
            update_history("user", user_message)

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
                # Send back the response to the client
                try:
                    client_sock.sendall(response_content.encode('utf-8'))
                    logging.info(f"ClientHandler: Sent response to {addr}.")
                except Exception as e:
                    logging.error(f"ClientHandler: Failed to send response to {addr}: {e}")
            except Empty:
                logging.error("ClientHandler: Timeout waiting for Ollama response.")
                response_content = "I'm sorry, I couldn't process your request at this time."
                # Send back the placeholder to the client
                try:
                    client_sock.sendall(response_content.encode('utf-8'))
                    logging.info(f"ClientHandler: Sent response to {addr}.")
                except Exception as e:
                    logging.error(f"ClientHandler: Failed to send response to {addr}: {e}")
                # Do not append to history
        except Exception as e:
            logging.error(f"ClientHandler: Unexpected error: {e}")
        finally:
            client_sock.close()
            logging.info(f"ClientHandler: Connection with {addr} closed.")

    #############################################
    # Step 12: Server Handling with Dedicated Receiver Thread #
    #############################################
    
    def start_server():
        # Start TTS and Ollama worker threads
        logging.info("Starting TTS Worker...")
        start_tts_thread()
        logging.info("Starting Ollama Worker...")
        start_ollama_thread()

        # Start Inference to TTS Handler thread
        # Already started earlier

        # Start CPU usage monitoring
        cpu_monitor_thread = threading.Thread(target=monitor_cpu_usage, daemon=True, name="CPUMonitor")
        cpu_monitor_thread.start()
        logging.info("CPU Usage Monitor: Started.")

        # Start the receiver thread
        receiver = threading.Thread(target=receiver_thread, args=(CONFIG["host"], CONFIG["port"]), daemon=True, name="ReceiverThread")
        receiver.start()
        logging.info("Receiver Thread: Started.")

        # Keep the main thread alive to allow daemon threads to run
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logging.info("\nInterrupt received, shutting down server.")
        finally:
            # Stop worker threads
            logging.info("Stopping TTS Worker...")
            tts_queue.put(None)  # Signal TTS worker to exit
            logging.info("Stopping Ollama Worker...")
            ollama_queue.put((None, None))  # Signal Ollama worker to exit
            logging.info("Stopping Inference to TTS Handler...")
            inference_queue.put("__SHUTDOWN__")  # Signal inference_to_tts_handler to exit

            # Allow some time for threads to shutdown
            time.sleep(2)

            logging.info("Shutting down complete.")

if __name__ == "__main__":
    # Determine if running in offline mode
    if CONFIG.get("offline_mode", False):
        logging.info("Operating in offline mode.")
    else:
        ONLINE_STATUS = is_connected()
        if not ONLINE_STATUS:
            logging.warning("No internet connection detected. Switching to offline mode.")
            CONFIG["offline_mode"] = True
        else:
            CONFIG["offline_mode"] = False

    start_server()
