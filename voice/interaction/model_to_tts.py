#!/usr/bin/env python3
import os
import sys
import subprocess
import socket
import re
import json
import argparse
import threading
from queue import Queue
import shutil
import time
from contextlib import redirect_stdout

#############################################
# Step 1: Ensure we're running inside a venv #
#############################################
print("[VENV] Checking if running inside a virtual environment...")
VENV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv")
NEEDED_PACKAGES = ["requests", "num2words", "ollama"]

def in_venv():
    is_in = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    print(f"[VENV] in_venv: {is_in}")
    return is_in

def setup_venv():
    if not os.path.isdir(VENV_DIR):
        print("[VENV] Creating virtual environment...")
        subprocess.check_call([sys.executable, '-m', 'venv', VENV_DIR])
    pip_path = os.path.join(VENV_DIR, 'bin', 'pip')
    print("[VENV] Installing required packages:", NEEDED_PACKAGES)
    subprocess.check_call([pip_path, 'install'] + NEEDED_PACKAGES)

def relaunch_in_venv():
    python_path = os.path.join(VENV_DIR, 'bin', 'python')
    print("[VENV] Relaunching inside virtual environment...")
    os.execv(python_path, [python_path] + sys.argv)

if not in_venv():
    setup_venv()
    relaunch_in_venv()

#############################################
# Step 2: Imports after venv set up          #
#############################################
print("[Imports] Importing external modules...")
import requests
from num2words import num2words
from ollama import chat  # Use Ollama Python library for inference

#############################################
# Additional: Short Tone/Beep Utilities      #
#############################################
def beep(freq=1200, duration=0.01):
    print(f"[Beep] Playing beep at {freq}Hz for {duration}s.")
    try:
        subprocess.run(
            ["play", "-nq", "-t", "alsa", "synth", str(duration), "sine", str(freq)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
        print("[Beep] Beep played.")
    except Exception:
        print("[Beep] Beep failed (or 'play' not available).")

#############################################
# Step 3: Config Defaults & File             #
#############################################
print("[Config] Loading configuration...")
DEFAULT_CONFIG = {
    "model": "gemma3:12b",
    "stream": True,
    "format": None,
    "system": None,
    "raw": False,
    "history": "chat.json",
    "history_depth": 40,
    "images": [],
    "tools": None,
    "options": {},
    "host": "0.0.0.0",
    "port": 64162,
    "tts_url": "http://localhost:61637/synthesize",
    "ollama_url": "http://localhost:11434/api/chat"
}
CONFIG_PATH = "config.json"

def load_config():
    if not os.path.exists(CONFIG_PATH):
        print("[Config] No config.json found. Creating default config file.")
        with open(CONFIG_PATH, 'w') as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
        return dict(DEFAULT_CONFIG)
    else:
        try:
            with open(CONFIG_PATH, 'r') as f:
                cfg = json.load(f)
            for key, value in DEFAULT_CONFIG.items():
                if key not in cfg:
                    cfg[key] = value
            print("[Config] Loaded configuration:")
            print(json.dumps(cfg, indent=2))
            return cfg
        except Exception as e:
            print(f"[Config] Error loading config.json: {e}. Using default settings.")
            return dict(DEFAULT_CONFIG)

CONFIG = load_config()

def update_config_file():
    try:
        with open(CONFIG_PATH, 'w') as f:
            json.dump(CONFIG, f, indent=2)
        print("[Config] Configuration file updated.")
    except Exception as e:
        print(f"[Config] Error updating config file: {e}")

#############################################
# Step 4: Parse Command-Line Arguments       #
#############################################
print("[Args] Parsing command-line arguments...")
parser = argparse.ArgumentParser(description="Ollama Chat Server with TTS and advanced features.")
parser.add_argument("--model", type=str, help="Model name to use.")
parser.add_argument("--stream", action="store_true", help="Enable streaming responses from the model.")
parser.add_argument("--format", type=str, help="Structured output format: 'json' or path to JSON schema file.")
parser.add_argument("--system", type=str, help="System message override.")
parser.add_argument("--raw", action="store_true", help="If set, use raw mode (no template).")
parser.add_argument("--history", type=str, nargs='?', const="chat.json",
                    help="Path to a JSON file containing conversation history messages.")
parser.add_argument("--images", type=str, nargs='*', help="List of base64-encoded image files.")
parser.add_argument("--tools", type=str, help="Path to a JSON file defining tools.")
parser.add_argument("--option", action="append", help="Additional model parameters (e.g. --option temperature=0.7)")
args = parser.parse_args()

def merge_config_and_args(config, args):
    print("[Args] Merging command-line arguments into configuration...")
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
    print("[Args] Configuration after merge:")
    print(json.dumps(config, indent=2))
    return config

CONFIG = merge_config_and_args(CONFIG, args)

#############################################
# Step 5: Load Optional Configurations       #
#############################################
print("[Config] Loading optional JSON configurations...")
def safe_load_json_file(path, default):
    if not path:
        return default
    if not os.path.exists(path):
        print(f"[Config] Warning: File '{path}' not found. Using default {default}.")
        if path == CONFIG["history"] and default == []:
            with open(path, 'w') as f:
                json.dump([], f)
        return default
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"[Config] Warning: Could not load '{path}': {e}. Using default {default}.")
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
            print(f"[Config] Warning: Could not load format schema from '{fmt}': {e}. Ignoring format.")
            return None
    print(f"[Config] Warning: Format file '{fmt}' not found. Ignoring format.")
    return None

history_messages = safe_load_json_file(CONFIG["history"], [])
tools_data = safe_load_json_file(CONFIG["tools"], None)
format_schema = load_format_schema(CONFIG["format"])
print("[Config] Loaded history messages:")
print(history_messages)

#############################################
# Step 5.1: Ensure Ollama and Model are Installed #
#############################################
print("[Ollama] Checking Ollama installation and model availability...")

def check_ollama_installed():
    ollama_path = shutil.which('ollama')
    print(f"[Ollama] ollama command path: {ollama_path}")
    return ollama_path is not None

def install_ollama():
    print("[Ollama] Installing Ollama via official installation script...")
    try:
        subprocess.check_call('curl -fsSL https://ollama.com/install.sh | sh', shell=True, executable='/bin/bash')
        print("[Ollama] Installation initiated.")
    except subprocess.CalledProcessError as e:
        print(f"[Ollama] Error installing Ollama: {e}")
        sys.exit(1)

def wait_for_ollama():
    ollama_tags_url = "http://localhost:11434/api/tags"
    max_retries = 10
    for attempt in range(max_retries):
        try:
            response = requests.get(ollama_tags_url)
            if response.status_code == 200:
                print("[Ollama] Service is up and running.")
                return
        except requests.exceptions.RequestException:
            pass
        print(f"[Ollama] Waiting for service to start... ({attempt+1}/{max_retries})")
        time.sleep(2)
    print("[Ollama] Service did not start in time. Exiting.")
    sys.exit(1)

def get_available_models():
    ollama_tags_url = "http://localhost:11434/api/tags"
    try:
        response = requests.get(ollama_tags_url)
        if response.status_code == 200:
            data = response.json()
            available_models = data.get('models', [])
            print("\n[Ollama] Available Models:")
            for model in available_models:
                print(f" - {model.get('name')}")
            return [m.get('name') for m in available_models if 'name' in m]
        else:
            print(f"[Ollama] Failed to retrieve models: Status code {response.status_code}")
            return []
    except requests.exceptions.RequestException as e:
        print(f"[Ollama] Error fetching models: {e}")
        return []

def check_model_exists_in_tags(model_name):
    available_models = get_available_models()
    if model_name in available_models:
        print(f"[Ollama] Model '{model_name}' found in tags.")
        return model_name
    model_latest = f"{model_name}:latest"
    if model_latest in available_models:
        print(f"[Ollama] Model '{model_latest}' found in tags.")
        return model_latest
    print(f"[Ollama] Model '{model_name}' not found in available tags.")
    return None

def check_model_installed(model_name):
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, check=True)
        models = [m.strip() for m in result.stdout.splitlines()]
        print("[Ollama] Installed models:", models)
        if model_name in models:
            return True
        if model_name.endswith(':latest'):
            base_model = model_name.rsplit(':', 1)[0]
            if base_model in models:
                return True
        return False
    except subprocess.CalledProcessError as e:
        print(f"[Ollama] Error checking installed models: {e}")
        sys.exit(1)

def pull_model(model_name):
    print(f"[Ollama] Pulling model '{model_name}'...")
    try:
        subprocess.check_call(['ollama', 'pull', model_name])
        print(f"[Ollama] Model '{model_name}' pulled successfully.")
    except subprocess.CalledProcessError as e:
        print(f"[Ollama] Error pulling model '{model_name}': {e}")
        sys.exit(1)

def ensure_ollama_and_model():
    if not check_ollama_installed():
        install_ollama()
        if not check_ollama_installed():
            print("[Ollama] Installation failed or 'ollama' command not in PATH.")
            sys.exit(1)
    else:
        print("[Ollama] Ollama is already installed.")
    wait_for_ollama()
    model_name = CONFIG["model"]
    model_actual_name = check_model_exists_in_tags(model_name)
    if not model_actual_name:
        print(f"[Ollama] Model '{model_name}' not available. Exiting.")
        sys.exit(1)
    print(f"[Ollama] Using model: {model_actual_name}")
    CONFIG["model"] = model_actual_name

ensure_ollama_and_model()

#############################################
# Step 6: Ollama Chat Interaction
#############################################
print("[Inference] Preparing Ollama chat interaction using the Ollama Python library...")
OLLAMA_CHAT_URL = CONFIG["ollama_url"]  # For legacy logging; not used with the library

def convert_numbers_to_words(text):
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
    # Truncate history to the last CONFIG["history_depth"] messages
    history_depth = CONFIG.get("history_depth", 40)
    if len(history_messages) > history_depth:
        print(f"[Payload] Truncating history messages to last {history_depth} (was {len(history_messages)})")
        messages.extend(history_messages[-history_depth:])
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
    print("[Payload] Built payload:")
    print(json.dumps(payload, indent=2))
    return payload

stop_flag = False
thread_lock = threading.Lock()

#############################################
# Step 7: TTS Playback with Queue and Thread
#############################################
tts_queue = None
tts_stop_flag = False
tts_thread = None
tts_thread_lock = threading.Lock()

def synthesize_and_play(prompt):
    prompt = re.sub(r'[\*#]', '', prompt).strip()
    if not prompt:
        print("[TTS] Empty prompt; skipping TTS.")
        return
    print(f"[TTS] Starting TTS for prompt: {prompt}")
    start_wait_beeps()
    try:
        payload = {"prompt": prompt}
        with requests.post(CONFIG["tts_url"], json=payload, stream=True) as response:
            if response.status_code != 200:
                print(f"[TTS] Warning: Received status code {response.status_code}")
                try:
                    error_msg = response.json().get('error', 'No error message provided.')
                    print(f"[TTS] Error message: {error_msg}")
                except:
                    print("[TTS] No JSON error message provided for TTS.")
                stop_wait_beeps()
                return
            print("[TTS] TTS request successful. Beginning audio stream...")
            aplay = subprocess.Popen(['aplay', '-r', '22050', '-f', 'S16_LE', '-t', 'raw'],
                                     stdin=subprocess.PIPE)
            try:
                for chunk in response.iter_content(chunk_size=4096):
                    stop_wait_beeps()
                    if tts_stop_flag:
                        print("[TTS] Stop flag detected. Terminating TTS playback.")
                        break
                    if chunk:
                        aplay.stdin.write(chunk)
            except BrokenPipeError:
                print("[TTS] Warning: aplay subprocess terminated unexpectedly.")
            finally:
                aplay.stdin.close()
                aplay.wait()
                print("[TTS] TTS playback finished.")
    except Exception as e:
        print(f"[TTS] Unexpected error during TTS: {e}")
        stop_wait_beeps()
    else:
        stop_wait_beeps()

def tts_worker():
    global tts_stop_flag
    print("[TTS] TTS worker started.")
    while not tts_stop_flag:
        try:
            sentence = tts_queue.get(timeout=0.1)
            print(f"[TTS] Retrieved sentence from queue: {sentence}")
        except Exception:
            if tts_stop_flag:
                break
            continue
        if tts_stop_flag:
            break
        synthesize_and_play(sentence)
    print("[TTS] TTS worker terminating.")

def start_tts_thread():
    global tts_queue, tts_thread, tts_stop_flag
    with tts_thread_lock:
        if tts_thread and tts_thread.is_alive():
            print("[TTS] TTS thread already running.")
            return
        print("[TTS] Starting TTS thread...")
        tts_stop_flag = False
        tts_queue = Queue()
        tts_thread = threading.Thread(target=tts_worker, daemon=True)
        tts_thread.start()

def stop_tts_thread():
    global tts_stop_flag, tts_thread, tts_queue
    with tts_thread_lock:
        if tts_thread and tts_thread.is_alive():
            print("[TTS] Stopping TTS thread...")
            tts_stop_flag = True
            with tts_queue.mutex:
                tts_queue.queue.clear()
            tts_thread.join()
            print("[TTS] TTS thread stopped.")
        tts_stop_flag = False
        tts_queue = None
        tts_thread = None

def enqueue_sentence_for_tts(sentence):
    if tts_queue and not tts_stop_flag:
        print(f"[TTS] Enqueuing sentence for TTS: {sentence}")
        tts_queue.put(sentence)

#############################################
# Step 7.1: Non-blocking beep while waiting #
#############################################
wait_beeps_thread = None
wait_beeps_flag = False
wait_beeps_lock = threading.Lock()

def wait_beeps_worker():
    while True:
        with wait_beeps_lock:
            if not wait_beeps_flag:
                break
        beep(1480, 0.03)
        beep(8130, 0.02)
        beep(1480, 0.03)


def start_wait_beeps():
    global wait_beeps_thread, wait_beeps_flag
    with wait_beeps_lock:
        if wait_beeps_thread and wait_beeps_thread.is_alive():
            return
        wait_beeps_flag = True
    wait_beeps_thread = threading.Thread(target=wait_beeps_worker, daemon=True)
    wait_beeps_thread.start()

def stop_wait_beeps():
    global wait_beeps_thread, wait_beeps_flag
    with wait_beeps_lock:
        wait_beeps_flag = False
    if wait_beeps_thread and wait_beeps_thread.is_alive():
        wait_beeps_thread.join()
    wait_beeps_thread = None

#############################################
# Step 8: Streaming the Output via Ollama Library
#############################################
def chat_completion_stream(user_message):
    payload = build_payload(user_message)
    print("[Inference] Sending streaming request via ollama.chat()...")
    try:
        stream = chat(model=CONFIG["model"], messages=payload["messages"], stream=CONFIG["stream"])
        print(f"[Inference] Streaming response object type: {type(stream)}")
        for part in stream:
            print("[Inference] Raw stream part:", part)
            content = part["message"]["content"]
            done = part.get("done", False)
            print(f"[Inference] Received chunk: {content!r}, done: {done}")
            yield content, done
            if done:
                break
    except Exception as e:
        print("[Inference] Error during streaming inference:", e)
        yield "", True

def chat_completion_nonstream(user_message):
    payload = build_payload(user_message)
    print("[Inference] Sending non-streaming request via ollama.chat()...")
    try:
        response = chat(model=CONFIG["model"], messages=payload["messages"], stream=False)
        result = response["message"]["content"]
        print("[Inference] Non-streamed result received:")
        print(result)
        return result
    except Exception as e:
        print("[Inference] Error during non-stream inference:", e)
        return ""

#############################################
# Step 9: Processing the Model Output
#############################################
def process_text(text):
    global stop_flag
    processed_text = convert_numbers_to_words(text)
    print("[Process] Processed text:", processed_text)
    sentence_endings = re.compile(r'[.?!]+')
    if CONFIG["stream"]:
        buffer = ""
        sentences = []
        for content, done in chat_completion_stream(processed_text):
            if stop_flag:
                break
            print(f"[Process] Received chunk: {content!r}")
            buffer += content
            print(f"[Process] Buffer now: {buffer!r}")
            while True:
                match = sentence_endings.search(buffer)
                if not match:
                    break
                end_index = match.end()
                sentence = buffer[:end_index].strip()
                buffer = buffer[end_index:].lstrip()
                if sentence:
                    print(f"[Process] Detected complete sentence: {sentence!r}")
                    sentences.append(sentence)
                    threading.Thread(target=enqueue_sentence_for_tts, args=(sentence,), daemon=True).start()
            if done:
                print("[Process] Inference indicated completion.")
                break
        print("[Process] Stream ended. Final buffer:", buffer)
        if buffer.strip():
            leftover = buffer.strip()
            print(f"[Process] Final leftover sentence: {leftover!r}")
            sentences.append(leftover)
            threading.Thread(target=enqueue_sentence_for_tts, args=(leftover,), daemon=True).start()
        full_text = "".join(sentences)
        print("[Process] Full assembled response:")
        print(full_text)
        return full_text
    else:
        result = chat_completion_nonstream(processed_text)
        print("[Process] Non-streamed response:")
        print(result)
        sentences = []
        buffer = result
        sentence_endings = re.compile(r'[.?!]+')
        while True:
            match = sentence_endings.search(buffer)
            if not match:
                break
            end_index = match.end()
            sentence = buffer[:end_index].strip()
            buffer = buffer[end_index:].lstrip()
            if sentence:
                sentences.append(sentence)
                enqueue_sentence_for_tts(sentence)
        if buffer.strip():
            leftover = buffer.strip()
            sentences.append(leftover)
            enqueue_sentence_for_tts(leftover)
        full_text = "".join(sentences)
        print("[Process] Full assembled non-streamed response:")
        print(full_text)
        return full_text

#############################################
# Step 10: Update History File with New Messages
#############################################
def update_history(user_message, assistant_message):
    if not CONFIG["history"]:
        return
    current_history = safe_load_json_file(CONFIG["history"], [])
    current_history.append({"role": "user", "content": user_message})
    current_history.append({"role": "assistant", "content": assistant_message})
    try:
        with open(CONFIG["history"], 'w') as f:
            json.dump(current_history, f, indent=2)
        print("[History] History file updated.")
    except Exception as e:
        print(f"[History] Warning: Could not write to history file {CONFIG['history']}: {e}")

#############################################
# Step 11: Handling Concurrent Requests and Cancellation
#############################################
stop_flag = False
current_thread = None
inference_lock = threading.Lock()

def inference_thread(user_message, result_holder, model_actual_name):
    global stop_flag
    stop_flag = False
    print("[Inference Thread] Starting inference for message:")
    print(user_message)
    result = process_text(user_message)
    print("[Inference Thread] Inference result obtained.")
    result_holder.append(result)

def new_request(user_message, model_actual_name):
    global stop_flag, current_thread
    print("[Request] New inference request received.")
    beep(1520, 0.03)
    with inference_lock:
        if current_thread and current_thread.is_alive():
            print("[Request] Interrupting current inference...")
            beep(1480, 0.03)
            stop_flag = True
            current_thread.join()
            stop_flag = False
        print("[Request] Stopping current TTS thread...")
        beep(1680, 0.02)
        stop_tts_thread()
        print("[Request] Restarting TTS thread...")
        beep(1200, 0.02)
        start_tts_thread()
        result_holder = []
        current_thread = threading.Thread(
            target=inference_thread,
            args=(user_message, result_holder, model_actual_name)
        )
        current_thread.start()
    current_thread.join()
    result = result_holder[0] if result_holder else ""
    print("[Request] Inference complete. Returning result.")
    return result

#############################################
# Step 12: Interactive Command-Line Interface
#############################################
def interactive_loop():
    print("\n[Interactive] Enter commands (/send, /model, /history_depth, /quit):")
    while True:
        try:
            cmd = input("[Interactive] > ").strip()
        except EOFError:
            break
        if not cmd:
            continue
        if cmd.startswith("/send "):
            message = cmd[len("/send "):].strip()
            print(f"[Interactive] Sending message: {message}")
            result = new_request(message, CONFIG["model"])
            print("[Interactive] Inference result:")
            print(result)
        elif cmd.startswith("/model "):
            new_model = cmd[len("/model "):].strip()
            print(f"[Interactive] Changing model to: {new_model}")
            CONFIG["model"] = new_model
            ensure_ollama_and_model()
            update_config_file()
            print(f"[Interactive] Model updated to: {CONFIG['model']}")
        elif cmd.startswith("/history_depth "):
            try:
                depth = int(cmd[len("/history_depth "):].strip())
                print(f"[Interactive] Setting history depth to: {depth}")
                CONFIG["history_depth"] = depth
                update_config_file()
            except ValueError:
                print("[Interactive] Invalid history depth value. Please enter an integer.")
        elif cmd == "/quit":
            print("[Interactive] Quitting interactive mode.")
            break
        else:
            print("[Interactive] Unknown command. Commands are: /send, /model, /history_depth, /quit.")

#############################################
# Step 13: Start Server with Enhanced Interrupt Handling
#############################################
HOST = CONFIG["host"]
PORT = CONFIG["port"]

client_threads = []
client_threads_lock = threading.Lock()

def handle_client_connection(client_socket, address, model_actual_name):
    global stop_flag, current_thread
    print(f"\n[Server] Accepted connection from {address}")
    beep(1320, 0.03)
    try:
        data = client_socket.recv(65536)
        if not data:
            print(f"[Server] No data from {address}, closing connection.")
            return
        user_message = data.decode('utf-8').strip()
        if not user_message:
            print(f"[Server] Empty prompt from {address}, ignoring.")
            return
        print(f"[Server] Received prompt from {address}: {user_message}")
        beep(1520, 0.03)
        result = new_request(user_message, model_actual_name)
        client_socket.sendall(result.encode('utf-8'))
        update_history(user_message, result)
    except Exception as e:
        print(f"[Server] Error handling client {address}: {e}")
    finally:
        client_socket.close()
        print(f"[Server] Connection with {address} closed.")

def start_server():
    global client_threads
    print("\n[Server] Starting TTS thread...")
    start_tts_thread()
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server.bind((HOST, PORT))
        print(f"[Server] Bound to {HOST}:{PORT}")
    except Exception as e:
        print(f"[Server] Error binding to {HOST}:{PORT}: {e}. Using default 0.0.0.0:64162")
        server.bind(('0.0.0.0', 64162))
    server.listen(5)
    print(f"[Server] Listening for incoming connections on {HOST}:{PORT}...")
    model_actual_name = CONFIG["model"]
    try:
        while True:
            try:
                client_sock, addr = server.accept()
                print(f"[Server] New connection accepted from {addr}")
                client_thread = threading.Thread(
                    target=handle_client_connection,
                    args=(client_sock, addr, model_actual_name)
                )
                client_thread.start()
                with client_threads_lock:
                    client_threads.append(client_thread)
            except KeyboardInterrupt:
                print("\n[Server] Keyboard interrupt received. Shutting down server.")
                break
            except Exception as e:
                print(f"[Server] Error accepting connections: {e}")
    finally:
        server.close()
        print("\n[Server] Server socket closed.")
        print("[Server] Stopping TTS thread...")
        stop_tts_thread()
        print("[Server] Waiting for client threads to finish...")
        with client_threads_lock:
            for t in client_threads:
                t.join()
        print("[Server] All client threads have been terminated.")
        print("[Server] Shutting down complete.")

#############################################
# Step 14: Interactive Command-Line Interface Thread
#############################################
def start_interactive_thread():
    thread = threading.Thread(target=interactive_loop, daemon=True)
    thread.start()
    print("[Interactive] Interactive command thread started.")

#############################################
# Main
#############################################
if __name__ == "__main__":
    start_interactive_thread()
    start_server()
