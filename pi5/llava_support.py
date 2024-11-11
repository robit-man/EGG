#!/usr/bin/env python3

import os
import sys
import subprocess
import curses
import curses.textpad  # Required for text input within curses
import json
import requests
import threading
import textwrap
from pathlib import Path
from queue import Queue, Empty
import logging
import time
import re
from datetime import datetime, timedelta
import zipfile
import shutil
import base64
import tempfile

# ======================= Configuration Constants =======================
VENV_DIR = "voice_venv"
CONFIG_FILE = "config.json"
CHAT_HISTORY_FILE = "chat_history.json"  # Path to chat history file
VOSK_MODEL_PATH = "models/vosk-model-small-en-us-0.15"  # Path to Vosk model
VOSK_MODEL_ZIP_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"  # URL to download the Vosk model
FRAMES_DIR = "frames"  # Directory to save inference frames

# User-provided configuration
DEFAULT_CONFIG = {
    "system_prompt": "make a clear and conscious decision which carefully and appropriately responds to inputs",
    "ollama_chat_history": True,
    "history_limit": 9,
    "temperature": 0.6,
    "tts_enabled": True,
    "tts_voice": "slt",
    "top_k": 40,
    "top_p": 0.9,
    "seed": 42,
    "stop": [
        "user:",
        "assistant:"
    ],
    "stream": True,
    "format": "json",               # Ensuring format is set to 'json'
    "keep_alive": "5m",
    "raw": False,
    "penalize_newline": True,
    "num_keep": 24,
    "num_predict": 100,
    "mirostat": 1,
    "mirostat_tau": 0.8,
    "mirostat_eta": 0.6,
    "presence_penalty": 1.5,
    "frequency_penalty": 1.0,
    "repeat_penalty": 1.2,
    "tools": [],
    "model_list_refresh_interval": 300,
    "model": "llava",  # Updated to 'llava'
    "generate_api_url": "http://127.0.0.1:11434/api/generate",
    "chat_api_url": "http://127.0.0.1:11434/api/chat",
    "default_endpoint": "chat"
}

# ======================= Logging Configuration ==========================
logging.basicConfig(
    filename='app.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ======================= Suppress ALSA and JACK Warnings ==================
def redirect_stderr():
    """Redirects the OS-level stderr to suppress ALSA and JACK errors."""
    sys.stderr.flush()
    devnull = open(os.devnull, 'w')
    os.dup2(devnull.fileno(), sys.stderr.fileno())

redirect_stderr()

# ======================= Subprocess stderr suppression ====================
SUBPROCESS_STDERR = subprocess.DEVNULL

# ======================= Configuration Management Functions ============
def load_config():
    """Load configuration from CONFIG_FILE or create it with DEFAULT_CONFIG."""
    if not os.path.exists(CONFIG_FILE):
        save_config(DEFAULT_CONFIG)
        logging.info("Config file not found. Created default config.")
        return DEFAULT_CONFIG.copy()
    else:
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
            # Ensure all default keys exist
            updated = False
            for key, value in DEFAULT_CONFIG.items():
                if key not in config:
                    config[key] = value
                    updated = True
                elif key == "history_limit":
                    try:
                        config[key] = int(config[key])
                    except ValueError:
                        config[key] = DEFAULT_CONFIG[key]
                        updated = True
            if updated:
                save_config(config)
                logging.info("Updated config with missing or invalid default keys.")
            return config
        except (json.JSONDecodeError, IOError) as e:
            logging.error(f"Error loading config: {e}. Recreating default config.")
            save_config(DEFAULT_CONFIG)
            return DEFAULT_CONFIG.copy()

def save_config(config):
    """Save configuration to CONFIG_FILE."""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)
        logging.info("Configuration saved successfully.")
    except IOError as e:
        logging.error(f"Failed to save configuration: {e}")
        print(f"Failed to save configuration: {e}")

# ======================= Virtual Environment Setup ======================
def is_venv():
    """Check if the script is running inside a virtual environment."""
    return (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) or \
           (hasattr(sys, 'real_prefix') and sys.real_prefix != sys.prefix)

def create_venv():
    """Create a virtual environment if it doesn't exist."""
    if not os.path.exists(VENV_DIR):
        logging.info("Creating virtual environment.")
        result = subprocess.run([sys.executable, "-m", "venv", VENV_DIR], stderr=SUBPROCESS_STDERR)
        if result.returncode != 0:
            logging.error("Failed to create virtual environment.")
            sys.exit("Error: Failed to create virtual environment.")
        logging.info("Virtual environment created.")
        print("Virtual environment created.")
    else:
        logging.info("Virtual environment already exists.")
        print("Virtual environment already exists.")

def install_dependencies():
    """Install required Python packages in the virtual environment."""
    logging.info("Installing dependencies.")
    # Determine the pip executable path based on OS
    if os.name == 'nt':
        pip_executable = os.path.join(VENV_DIR, "Scripts", "pip.exe")
    else:
        pip_executable = os.path.join(VENV_DIR, "bin", "pip")
    
    # Define required package versions to prevent multiple version attempts
    required_packages = [
        "SpeechRecognition",
        "pyaudio",
        "vosk",
        "requests",
        "numpy",
        "transformers==4.33.0",  # Specific version to prevent multiple downloads
        "umap-learn==0.5.6"      # Specific version to prevent multiple downloads
    ]
    
    try:
        # Upgrade pip
        subprocess.check_call([pip_executable, "install", "--upgrade", "pip"], stderr=SUBPROCESS_STDERR)
        # Install required packages with specified versions
        subprocess.check_call([pip_executable, "install"] + required_packages, stderr=SUBPROCESS_STDERR)
        logging.info("Dependencies installed successfully.")
        print("Dependencies installed.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Dependency installation failed: {e}")
        sys.exit("Error: Failed to install dependencies. Check app.log for details.")

def activate_venv():
    """Activate virtual environment by updating sys.path."""
    if os.name == 'nt':
        venv_site_packages = Path(VENV_DIR) / "Lib" / "site-packages"
    else:
        venv_site_packages = Path(VENV_DIR) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    sys.path.insert(0, str(venv_site_packages))
    logging.info("Virtual environment activated.")
    print("Virtual environment activated.")

# ======================= Vosk Model Downloader ============================
def download_vosk_model():
    """Download and extract the Vosk model if not already present."""
    if os.path.exists(VOSK_MODEL_PATH):
        logging.info(f"Vosk model already exists at {VOSK_MODEL_PATH}.")
        return
    os.makedirs(os.path.dirname(VOSK_MODEL_PATH), exist_ok=True)
    model_zip_path = VOSK_MODEL_PATH + ".zip"
    try:
        logging.info(f"Downloading Vosk model from {VOSK_MODEL_ZIP_URL}...")
        print("Downloading Vosk model. This may take a few minutes...")
        response = requests.get(VOSK_MODEL_ZIP_URL, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        with open(model_zip_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    done = int(50 * downloaded / total_size)
                    sys.stdout.write('\r[{}{}]'.format('█' * done, '.' * (50-done)))
                    sys.stdout.flush()
        print("\nDownload completed.")
        logging.info("Vosk model downloaded successfully.")
        
        # Extract the zip file
        logging.info(f"Extracting Vosk model to {VOSK_MODEL_PATH}...")
        print("Extracting Vosk model...")
        with zipfile.ZipFile(model_zip_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(VOSK_MODEL_PATH))
        logging.info("Vosk model extracted successfully.")
        print("Vosk model extracted successfully.")
        
        # Remove the zip file
        os.remove(model_zip_path)
    except requests.RequestException as e:
        logging.error(f"Failed to download Vosk model: {e}")
        print(f"Error: Failed to download Vosk model: {e}")
        sys.exit("Error: Failed to download Vosk model.")
    except zipfile.BadZipFile as e:
        logging.error(f"Failed to extract Vosk model: {e}")
        print(f"Error: Failed to extract Vosk model: {e}")
        sys.exit("Error: Failed to extract Vosk model.")

# ======================= Model Checker and Downloader ====================
def check_and_download_model(model_name, progress_queue):
    """
    Check if the specified Ollama model is installed.
    If not, download it using 'ollama pull' and send progress updates to the queue.
    
    :param model_name: Name of the model to check/download.
    :param progress_queue: Queue to send progress updates.
    :return: True if model is ready, False otherwise.
    """
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, check=True)
        installed_models = result.stdout.splitlines()
        if model_name in installed_models:
            logging.info(f"Model '{model_name}' is already installed.")
            progress_queue.put(("status", f"Model '{model_name}' is already installed."))
            return True
        else:
            logging.info(f"Model '{model_name}' not found. Pulling model...")
            progress_queue.put(("status", f"Model '{model_name}' not found. Downloading..."))
            
            # Start the 'ollama pull' subprocess
            process = subprocess.Popen(['ollama', 'pull', model_name], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            
            # Read the output line by line and send progress updates
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    # Parse the output for progress information
                    # This parsing depends on the actual output format of 'ollama pull'
                    # Adjust the regex as per the actual output
                    progress_queue.put(("progress", output.strip()))
            return_code = process.poll()
            if return_code == 0:
                logging.info(f"Model '{model_name}' pulled successfully.")
                progress_queue.put(("status", f"Model '{model_name}' downloaded successfully."))
                return True
            else:
                error_msg = f"Failed to pull model '{model_name}'."
                logging.error(error_msg)
                progress_queue.put(("error", error_msg))
                return False
    except subprocess.CalledProcessError as e:
        error_msg = f"Failed to list Ollama models: {e}"
        logging.error(error_msg)
        progress_queue.put(("error", error_msg))
        return False
    except Exception as e:
        error_msg = f"Unexpected error during model check/download: {e}"
        logging.error(error_msg)
        progress_queue.put(("error", error_msg))
        return False

# ======================= Text-to-Speech Handler Using flite ==============
def clear_queue(q):
    """Drain all items from a queue."""
    try:
        while True:
            q.get_nowait()
    except Empty:
        pass

class TTSHandler:
    """Handles Text-to-Speech operations using flite."""
    def __init__(self, voice='slt'):
        self.voice = voice  # Default voice
        self.queue = Queue()
        self.thread = threading.Thread(target=self.run, name="TTSHandlerThread")
        self.stop_event = threading.Event()
        self.thread.start()
        logging.info("TTSHandler initialized successfully.")
    
    def run(self):
        """Continuously processes the TTS queue and speaks sentences."""
        while not self.stop_event.is_set():
            try:
                sentence = self.queue.get(timeout=0.1)
                if sentence:
                    logging.info(f"TTSHandler is speaking: {sentence}")
                    subprocess.run(['flite', '-voice', self.voice, '-t', sentence], check=True, stderr=subprocess.PIPE)
                    logging.info(f"TTSHandler finished speaking: {sentence}")
            except Empty:
                continue
            except subprocess.CalledProcessError as e:
                error_output = e.stderr.decode().strip()
                logging.error(f"flite failed to speak '{sentence}': {error_output}")
            except Exception as e:
                logging.exception(f"Unexpected error in TTSHandler: {e}")
    
    def speak(self, sentence):
        """Enqueue a sentence to be spoken."""
        if self.queue.qsize() < 50:  # Example max size
            logging.debug(f"Enqueuing sentence for TTS: {sentence}")
            self.queue.put(sentence)
        else:
            logging.warning("TTS queue is full. Dropping sentence.")
    
    def clear_queue(self):
        """Clear the TTS queue."""
        clear_queue(self.queue)
        logging.debug("TTSHandler queue cleared.")
    
    def stop(self):
        """Stop the TTSHandler thread."""
        logging.info("Stopping TTSHandler...")
        self.stop_event.set()
        self.thread.join()
        logging.info("TTSHandler stopped.")

# ======================= Chat History Management ==========================
def load_chat_history(config):
    """Load chat history from CHAT_HISTORY_FILE if enabled."""
    if config.get("ollama_chat_history", False):
        if os.path.exists(CHAT_HISTORY_FILE):
            try:
                with open(CHAT_HISTORY_FILE, 'r') as f:
                    chat_history = json.load(f)
                logging.info("Chat history loaded successfully.")
                # Ensure system prompt is the first message
                if not chat_history or chat_history[0].get("role") != "system":
                    chat_history.insert(0, {"role": "system", "content": config.get("system_prompt", "You are a helpful assistant.")})
                return chat_history
            except (json.JSONDecodeError, IOError) as e:
                logging.error(f"Error loading chat history: {e}. Starting with empty history.")
                return [{"role": "system", "content": config.get("system_prompt", "You are a helpful assistant.")}]
        else:
            logging.info("Chat history file not found. Starting with system prompt.")
            return [{"role": "system", "content": config.get("system_prompt", "You are a helpful assistant.")}]
    else:
        return []

def save_chat_history(chat_history):
    """Save chat history to CHAT_HISTORY_FILE."""
    try:
        with open(CHAT_HISTORY_FILE, 'w') as f:
            json.dump(chat_history, f, indent=4)
        logging.info("Chat history saved successfully.")
    except IOError as e:
        logging.error(f"Failed to save chat history: {e}")

def append_to_chat_history(chat_history, role, content, config):
    """Append a new message to chat history and enforce history limit."""
    if not config.get("ollama_chat_history", False):
        logging.info("Chat history is disabled. Skipping append.")
        return chat_history
    chat_history.append({"role": role, "content": content})
    # Enforce history limit
    history_limit = config.get("history_limit", 20)
    if len(chat_history) > history_limit:
        removed = len(chat_history) - history_limit
        for _ in range(removed):
            removed_entry = chat_history.pop(0)
            logging.debug(f"Removed from chat history: {removed_entry}")
    save_chat_history(chat_history)
    return chat_history

def get_recent_chat_history(chat_history, history_limit):
    """Return the last `history_limit` messages from chat history."""
    if len(chat_history) > history_limit:
        return chat_history[-history_limit:]
    return chat_history

# ======================= Speech Recognition Thread with Vosk ============
def speech_recognition_thread(text_queue, stop_event):
    """Thread function for speech recognition using Vosk."""
    from vosk import Model, KaldiRecognizer, SetLogLevel
    import pyaudio

    SetLogLevel(-1)  # Suppress Vosk logs

    if not os.path.exists(VOSK_MODEL_PATH):
        error_msg = f"Vosk model not found at {VOSK_MODEL_PATH}. Please download and place it accordingly."
        text_queue.put(error_msg)
        logging.error(error_msg)
        return

    try:
        model = Model(VOSK_MODEL_PATH)
        recognizer = KaldiRecognizer(model, 16000)
    except Exception as e:
        error_msg = f"Failed to initialize Vosk model: {e}"
        text_queue.put(error_msg)
        logging.error(error_msg)
        return

    p = pyaudio.PyAudio()

    try:
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=16000,
                        input=True,
                        frames_per_buffer=4000)
        stream.start_stream()
    except Exception as e:
        error_msg = f"Microphone error: {e}"
        text_queue.put(error_msg)
        logging.error(error_msg)
        return

    logging.info("Speech recognition thread started.")

    while not stop_event.is_set():
        try:
            data = stream.read(4000, exception_on_overflow=False)
            if recognizer.AcceptWaveform(data):
                result = recognizer.Result()
                result_json = json.loads(result)
                text = result_json.get("text", "")
                if text:
                    text_queue.put(text)
                    logging.info(f"Recognized speech: {text}")
            else:
                partial = recognizer.PartialResult()
                # Optionally handle partial results
        except Exception as e:
            error_msg = f"Speech Recognition error: {e}"
            text_queue.put(error_msg)
            logging.error(error_msg)
            break

    stream.stop_stream()
    stream.close()
    p.terminate()
    logging.info("Speech recognition thread terminated.")

# ======================= Ollama API Streaming ============================
def send_to_ollama_api_stream(endpoint, data, response_queue, stop_event, config, chat_history, image_path=None):
    """
    Thread function to send user input to Ollama API and stream responses.

    :param endpoint: API endpoint to use ('generate' or 'chat')
    :param data: Dictionary containing the request payload
    :param response_queue: Queue to put the responses
    :param stop_event: Event to signal thread to stop
    :param config: Configuration dictionary
    :param chat_history: Current chat history
    :param image_path: Path to the image to send for inference (if any)
    """
    url = config.get("generate_api_url") if endpoint == "generate" else config.get("chat_api_url")
    headers = {"Content-Type": "application/json"}

    # If image_path is provided, encode the image in base64 and add to data
    if image_path and os.path.exists(image_path):
        try:
            with open(image_path, "rb") as img_file:
                encoded_string = base64.b64encode(img_file.read()).decode('utf-8')
            data["image"] = encoded_string
            logging.info(f"Image '{image_path}' encoded and added to the payload.")
        except Exception as e:
            error_msg = f"Failed to encode image '{image_path}': {e}"
            response_queue.put(("error", error_msg))
            logging.error(error_msg)
            return
    else:
        if image_path:
            error_msg = f"Image file '{image_path}' does not exist."
            response_queue.put(("error", error_msg))
            logging.error(error_msg)
            return

    logging.debug(f"Sending data to Ollama API at {url}: {json.dumps(data, indent=2)}")

    try:
        with requests.post(url, headers=headers, json=data, stream=True, timeout=120) as response:
            response.raise_for_status()
            logging.info(f"Ollama API ({endpoint}) request sent successfully.")
            buffer = ""
            for chunk in response.iter_content(chunk_size=1024):
                if stop_event.is_set():
                    logging.info("Stop event set. Exiting Ollama API stream.")
                    break
                if chunk:
                    decoded_chunk = chunk.decode('utf-8')
                    buffer += decoded_chunk
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            json_obj = json.loads(line)
                            if config.get("format") == "json":
                                if endpoint == "generate":
                                    content = json_obj.get("response", "")
                                elif endpoint == "chat":
                                    message = json_obj.get("message", {})
                                    content = message.get("content", "")
                                else:
                                    content = ""
                            else:  # raw format
                                content = line
                            
                            if content:
                                response_queue.put(("progress", content))
                                logging.debug(f"Received content: {content}")
                        except json.JSONDecodeError:
                            logging.warning(f"Malformed JSON line received: {line}")
                            continue  # Ignore malformed JSON lines
    except requests.RequestException as e:
        error_msg = f"Failed to send to Ollama API: {e}"
        response_queue.put(("error", error_msg))
        logging.error(error_msg)
    finally:
        # Signal that the API streaming is done
        response_queue.put(("complete", None))
        logging.info("Ollama API streaming ended.")

# ======================= Curses Menu System ==============================
class Menu:
    """Handles the configuration menu within the curses interface."""
    def __init__(self, stdscr, config, tts_handler, chat_history, progress_queue=None):
        self.stdscr = stdscr
        self.config = config
        self.tts_handler = tts_handler
        self.chat_history = chat_history
        self.menu_items = [
            "Set System Prompt",
            "Toggle Ollama Chat History",
            "Set History Limit",
            "Set Temperature",
            "Set Response Format",
            "Toggle Text-to-Speech",
            "Set TTS Voice",
            "Clear Chat History",
            "View Chat History",
            "Check and Download Models",
            "Back to Main"
        ]
        self.current_selection = 0
        self.progress_queue = progress_queue  # Queue for progress updates

    def display_menu(self):
        """Display the configuration menu."""
        self.stdscr.clear()
        h, w = self.stdscr.getmaxyx()
        title = "Configuration Menu (Use Arrow Keys & Enter)"
        try:
            self.stdscr.addstr(1, w//2 - len(title)//2, title, curses.A_BOLD | curses.A_UNDERLINE)
        except curses.error:
            pass  # Handle window too small

        for idx, item in enumerate(self.menu_items):
            x = w//2 - 30
            y = 3 + idx
            if y >= h - 1:
                break  # Prevent writing beyond the window
            if idx == self.current_selection:
                self.stdscr.attron(curses.color_pair(1))
                self.stdscr.addstr(y, x, f"> {item}")
                self.stdscr.attroff(curses.color_pair(1))
            else:
                self.stdscr.addstr(y, x, f"  {item}")
        self.stdscr.refresh()

    def navigate(self, key):
        """Navigate through the menu items."""
        if key == curses.KEY_UP and self.current_selection > 0:
            self.current_selection -= 1
        elif key == curses.KEY_DOWN and self.current_selection < len(self.menu_items) - 1:
            self.current_selection += 1

    def run(self):
        """Run the menu loop."""
        curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_WHITE)
        while True:
            self.display_menu()
            key = self.stdscr.getch()
            if key in [curses.KEY_UP, curses.KEY_DOWN]:
                self.navigate(key)
            elif key in [curses.KEY_ENTER, 10, 13]:
                selected_item = self.menu_items[self.current_selection]
                if selected_item == "Set System Prompt":
                    self.set_system_prompt()
                elif selected_item == "Toggle Ollama Chat History":
                    self.toggle_chat_history()
                elif selected_item == "Set History Limit":
                    self.set_history_limit()
                elif selected_item == "Set Temperature":
                    self.set_temperature()
                elif selected_item == "Set Response Format":
                    self.set_response_format()
                elif selected_item == "Toggle Text-to-Speech":
                    self.toggle_tts()
                elif selected_item == "Set TTS Voice":
                    self.set_tts_voice()
                elif selected_item == "Clear Chat History":
                    self.clear_chat_history()
                elif selected_item == "View Chat History":
                    self.view_chat_history()
                elif selected_item == "Check and Download Models":
                    self.check_and_download_models()
                elif selected_item == "Back to Main":
                    break

    def set_system_prompt(self):
        """Set a new system prompt."""
        prompt = self.get_input("Enter new system prompt:", self.config["system_prompt"])
        if prompt is not None:
            self.config["system_prompt"] = prompt
            save_config(self.config)
            self.show_message("System prompt updated successfully.")
            if self.tts_handler and self.config.get("tts_enabled", False):
                self.tts_handler.speak("System prompt updated successfully.")

    def toggle_chat_history(self):
        """Toggle the Ollama chat history feature."""
        self.config["ollama_chat_history"] = not self.config.get("ollama_chat_history", False)
        save_config(self.config)
        status = "enabled" if self.config["ollama_chat_history"] else "disabled"
        self.show_message(f"Ollama Chat History {status}.")
        if self.tts_handler and self.config.get("tts_enabled", False):
            self.tts_handler.speak(f"Ollama Chat History {status}.")

    def set_history_limit(self):
        """Set the history limit for chat history."""
        while True:
            limit = self.get_input("Set history limit (number of messages):", str(self.config.get("history_limit", 20)))
            if limit is None:
                break  # User cancelled
            try:
                limit_val = int(limit)
                if limit_val > 0:
                    self.config["history_limit"] = limit_val
                    save_config(self.config)
                    self.show_message("History limit updated successfully.")
                    if self.tts_handler and self.config.get("tts_enabled", False):
                        self.tts_handler.speak("History limit updated successfully.")
                    break
                else:
                    self.show_message("Please enter a positive integer.")
            except ValueError:
                self.show_message("Invalid input. Please enter a numerical value.")

    def set_temperature(self):
        """Set the temperature parameter for the API."""
        while True:
            temp = self.get_input("Set temperature (0.0 to 1.0):", str(self.config.get("temperature", 0.7)))
            if temp is None:
                break  # User cancelled
            try:
                temp_val = float(temp)
                if 0.0 <= temp_val <= 1.0:
                    self.config["temperature"] = temp_val
                    save_config(self.config)
                    self.show_message("Temperature updated successfully.")
                    if self.tts_handler and self.config.get("tts_enabled", False):
                        self.tts_handler.speak("Temperature updated successfully.")
                    break
                else:
                    self.show_message("Please enter a value between 0.0 and 1.0.")
            except ValueError:
                self.show_message("Invalid input. Please enter a numerical value.")

    def set_response_format(self):
        """Set the response format for the API."""
        formats = ["json", "raw"]
        current_format = self.config.get("format", "json")
        format_input = self.get_input(f"Set response format ({', '.join(formats)}):", current_format)
        if format_input in formats:
            self.config["format"] = format_input
            save_config(self.config)
            self.show_message(f"Response format set to {format_input}.")
            if self.tts_handler and self.config.get("tts_enabled", False):
                self.tts_handler.speak(f"Response format set to {format_input}.")
        else:
            self.show_message(f"Invalid format. Available formats: {', '.join(formats)}.")

    def toggle_tts(self):
        """Toggle the Text-to-Speech feature."""
        self.config["tts_enabled"] = not self.config.get("tts_enabled", False)
        save_config(self.config)
        status = "enabled" if self.config["tts_enabled"] else "disabled"
        self.show_message(f"Text-to-Speech {status}.")
        if status == "enabled":
            if self.tts_handler is None:
                try:
                    self.tts_handler = TTSHandler(voice=self.config.get("tts_voice", "slt"))
                    self.tts_handler.speak("Text-to-speech enabled.")
                except Exception as e:
                    logging.error(f"Failed to initialize TTSHandler: {e}")
                    self.show_message(f"Failed to initialize TTS: {e}")
        else:
            if self.tts_handler:
                self.tts_handler.stop()
                self.tts_handler = None

    def set_tts_voice(self):
        """Set the voice for Text-to-Speech."""
        available_voices = self.get_available_flite_voices()
        if not available_voices:
            self.show_message("No available flite voices found.")
            return
        voice = self.get_input(f"Enter TTS Voice ({', '.join(available_voices)}):", self.config.get("tts_voice", "slt"))
        if voice in available_voices:
            self.config["tts_voice"] = voice
            save_config(self.config)
            if self.tts_handler:
                self.tts_handler.voice = voice  # Update the voice in TTSHandler
            self.show_message(f"TTS Voice set to {voice}.")
            if self.tts_handler and self.config.get("tts_enabled", False):
                self.tts_handler.speak(f"TTS voice set to {voice}.")
        else:
            self.show_message("Invalid voice selected.")

    def clear_chat_history(self):
        """Clear the chat history."""
        if not self.config.get("ollama_chat_history", False):
            self.show_message("Chat history is disabled.")
            return
        confirm = self.get_input("Are you sure you want to clear chat history? (y/n):", "n")
        if confirm.lower() == 'y':
            try:
                with open(CHAT_HISTORY_FILE, 'w') as f:
                    json.dump([], f, indent=4)
                self.chat_history.clear()
                self.show_message("Chat history cleared successfully.")
                if self.tts_handler and self.config.get("tts_enabled", False):
                    self.tts_handler.speak("Chat history cleared successfully.")
            except IOError as e:
                logging.error(f"Failed to clear chat history: {e}")
                self.show_message(f"Failed to clear chat history: {e}")
        else:
            self.show_message("Chat history not cleared.")

    def view_chat_history(self):
        """View the current chat history."""
        if not self.config.get("ollama_chat_history", False):
            self.show_message("Chat history is disabled.")
            return
        if not self.chat_history:
            self.show_message("No chat history available.")
            return
        # Display the last `history_limit` entries
        history_to_display = self.chat_history[-self.config.get("history_limit", 20):]
        display_text = "\n".join([f"{entry['role'].capitalize()}: {entry['content']}" for entry in history_to_display])
        self.show_message(f"Chat History:\n{display_text}")

    def check_and_download_models(self):
        """Check and download necessary models, displaying progress."""
        model_name = self.config.get("model", "llava")
        progress_queue = Queue()
        download_thread = threading.Thread(
            target=check_and_download_model,
            args=(model_name, progress_queue),
            name="ModelDownloadThread",
            daemon=True
        )
        download_thread.start()

        # Display progress in a separate window
        self.display_download_progress(progress_queue, model_name)
    
    def display_download_progress(self, progress_queue, model_name):
        """Display the download progress in the curses interface."""
        self.stdscr.clear()
        h, w = self.stdscr.getmaxyx()
        title = f"Downloading Model '{model_name}' (Press 'c' to cancel)"
        try:
            self.stdscr.addstr(1, w//2 - len(title)//2, title, curses.A_BOLD | curses.A_UNDERLINE)
        except curses.error:
            pass  # Handle window too small
        
        progress_window = curses.newwin(h-4, w-2, 3, 1)
        progress_window.box()
        progress_window.refresh()

        download_complete = False
        while not download_complete:
            try:
                msg_type, message = progress_queue.get(timeout=0.1)
                if msg_type == "status":
                    progress_window.addstr(1, 2, message[:w-6])
                elif msg_type == "progress":
                    # Display progress line
                    progress_window.addstr(3, 2, message[:w-6])
                elif msg_type == "error":
                    progress_window.addstr(5, 2, f"Error: {message}"[:w-6], curses.A_BOLD | curses.A_BLINK)
                elif msg_type == "complete":
                    download_complete = True
                progress_window.refresh()
            except Empty:
                pass
            # Check for user input to cancel
            key = self.stdscr.getch()
            if key in [ord('c'), ord('C')]:
                # Attempt to terminate the download thread
                logging.info("User requested to cancel model download.")
                # Note: Python does not provide a direct way to terminate threads.
                # As a workaround, you can set the stop_event or implement a flag.
                # Here, we'll just inform the user.
                self.show_message("Cancellation not supported at this time.")
        
        self.show_message(f"Model '{model_name}' download process completed.")
        if self.tts_handler and self.config.get("tts_enabled", False):
            self.tts_handler.speak(f"Model {model_name} download process completed.")

    def get_available_flite_voices(self):
        """Retrieve available flite voices."""
        try:
            result = subprocess.run(['flite', '-lv'], capture_output=True, text=True, check=True)
            voices = [line.strip() for line in result.stdout.split('\n') if line.strip()]
            return voices
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to retrieve flite voices: {e.stderr.strip()}")
            return []
        except FileNotFoundError:
            logging.error("flite is not installed or not found in PATH.")
            return []

    def get_input(self, prompt, default=""):
        """Get user input within the curses interface."""
        self.stdscr.clear()
        h, w = self.stdscr.getmaxyx()
        prompt_str = f"{prompt}"
        try:
            self.stdscr.addstr(h//2 - 1, w//2 - len(prompt_str)//2, prompt_str)
            self.stdscr.addstr(h//2, w//2 - 10, "> ")
        except curses.error:
            pass  # Handle window too small
        self.stdscr.refresh()
        # Create a window for input
        input_width = min(len(default) + 20, w - (w//2 - 10) - 4)
        if input_width <= 0:
            logging.error("Window too small for input.")
            return default
        input_win = curses.newwin(1, input_width, h//2, w//2 - 8)
        box = curses.textpad.Textbox(input_win)
        try:
            # Disable echoing for the window since Textbox handles it
            user_input = box.edit().strip()
        except curses.error:
            user_input = ""
        return user_input if user_input else default

    def show_message(self, message):
        """Display a message within the curses interface."""
        self.stdscr.clear()
        h, w = self.stdscr.getmaxyx()
        try:
            # Split message into lines if it's too long
            lines = message.split('\n')
            for idx, line in enumerate(lines):
                if idx >= h - 4:
                    break  # Prevent writing beyond the window
                self.stdscr.addstr(h//2 - len(lines)//2 + idx, w//2 - len(line)//2, line[:w-1])
            self.stdscr.addstr(h//2 + 2, w//2 - 10, "Press any key to continue.")
            self.stdscr.refresh()
            self.stdscr.getch()
        except curses.error:
            pass  # Ignore if the string is too long for the window

# ======================= Main curses display loop with Menu Integration =========================
def main(stdscr):
    """Main function to handle the curses interface and voice assistant operations."""
    # Initialize curses settings
    curses.curs_set(0)  # Hide cursor
    stdscr.clear()
    stdscr.nodelay(True)  # Make getch() non-blocking
    stdscr.timeout(100)   # Set timeout for screen refresh rate

    # Load or create configuration
    config = load_config()

    # Ensure frames directory exists
    if not os.path.exists(FRAMES_DIR):
        os.makedirs(FRAMES_DIR)
        logging.info(f"Created frames directory at '{FRAMES_DIR}'.")

    # Check and download the required model ('llava')
    # This will automatically download if not present and display progress
    # Initialize a queue to receive progress updates
    progress_queue = Queue()
    progress_thread = threading.Thread(
        target=check_and_download_model,
        args=(config.get("model", "llava"), progress_queue),
        name="ModelDownloadThread",
        daemon=True
    )
    progress_thread.start()

    # Initialize a temporary directory for captured images
    temp_dir = tempfile.mkdtemp(prefix="voice_assistant_")

    # Load chat history
    chat_history = load_chat_history(config)

    # Set up virtual environment and dependencies
    create_venv()
    install_dependencies()
    activate_venv()

    # Download Vosk model if not present
    download_vosk_model()

    # Import modules within the activated virtual environment
    try:
        from vosk import Model, KaldiRecognizer
    except ImportError as e:
        stdscr.addstr(0, 0, f"Failed to import Vosk: {e}")
        stdscr.refresh()
        stdscr.getch()
        return

    # Initialize Queues for inter-thread communication
    text_queue = Queue()
    response_queue = Queue()

    # Event to signal threads to stop
    stop_event = threading.Event()

    # Initialize TTS Handler if enabled
    tts_handler = None
    if config.get("tts_enabled", False):
        try:
            tts_handler = TTSHandler(voice=config.get("tts_voice", "slt"))
            # Enqueue "System started." message
            tts_handler.speak("System started.")
        except Exception as e:
            logging.error(f"Failed to initialize TTSHandler: {e}")
            display_error(stdscr, f"Failed to initialize TTS: {e}", tts_handler, config)
            return  # Exit the main loop

    # Start speech recognition thread
    speech_thread = threading.Thread(
        target=speech_recognition_thread,
        args=(text_queue, stop_event),
        name="SpeechRecognitionThread",
        daemon=True
    )
    speech_thread.start()

    # Variables to handle API responses
    api_thread = None
    ollama_response = ""
    last_speech_time = None
    accumulated_text = ""

    # Initialize conversation log with system prompt
    conversation_log = [{"role": "system", "content": config.get("system_prompt", "You are a helpful assistant.")}]
    MAX_CONVERSATION_LINES = 100  # Adjust as needed

    # Define a regex pattern to detect sentence endings
    sentence_end_pattern = re.compile(r'([.!?])')  # Matches ., !, or ?

    # Initialize TTS buffer for sentence chunking
    tts_buffer = ""

    try:
        while True:
            # Handle model download progress
            try:
                while True:
                    msg_type, message = progress_queue.get_nowait()
                    if msg_type == "status":
                        # Display status messages
                        display_download_status(stdscr, message)
                        if tts_handler and config.get("tts_enabled", False):
                            tts_handler.speak(message)
                    elif msg_type == "progress":
                        # Display progress messages
                        display_download_progress(stdscr, message)
                    elif msg_type == "error":
                        display_error(stdscr, message, tts_handler, config)
                    elif msg_type == "complete":
                        # Download complete
                        display_download_complete(stdscr, config.get("model", "llava"))
            except Empty:
                pass

            # Check for user input
            try:
                key = stdscr.getch()
            except:
                key = -1

            if key != -1:
                if key in [ord('q'), ord('Q')]:
                    break
                elif key in [ord('m'), ord('M')]:
                    # Open Menu
                    menu = Menu(stdscr, config, tts_handler, chat_history)
                    menu.run()
                    # Reload config in case it was changed
                    config = load_config()
                    # Reload chat history in case it was cleared
                    chat_history = load_chat_history(config)
                    # Handle TTS Handler based on updated config
                    if config.get("tts_enabled", False) and not tts_handler:
                        try:
                            tts_handler = TTSHandler(voice=config.get("tts_voice", "slt"))
                            tts_handler.speak("Text-to-speech enabled.")
                        except Exception as e:
                            logging.error(f"Failed to initialize TTSHandler: {e}")
                            display_error(stdscr, f"Failed to initialize TTS: {e}", tts_handler, config)
                    elif not config.get("tts_enabled", False) and tts_handler:
                        tts_handler.stop()
                        tts_handler = None
                    continue

            # Check if any new recognized text is available
            try:
                text = text_queue.get_nowait()
                if text.startswith("Speech Recognition error:"):
                    conversation_log.append({"role": "error", "content": text})
                else:
                    if text:
                        accumulated_text += " " + text
                        last_speech_time = datetime.now()
                        conversation_log.append({"role": "user", "content": text})
                        # Append user input to chat history
                        chat_history = append_to_chat_history(chat_history, "user", text, config)
            except Empty:
                pass

            # Check if 2 seconds have passed since last speech
            if accumulated_text and last_speech_time:
                if datetime.now() - last_speech_time > timedelta(seconds=2):
                    # ==== Clearing Queues and Buffers Before New Generation ====
                    # 1. Stop any existing API thread
                    if api_thread and api_thread.is_alive():
                        logging.info("Stopping existing API thread.")
                        stop_event.set()
                        api_thread.join()
                        stop_event.clear()

                    # 2. Clear the response_queue
                    logging.debug("Clearing response_queue.")
                    clear_queue(response_queue)

                    # 3. Clear the TTSHandler's queue
                    if tts_handler:
                        logging.debug("Clearing TTSHandler's queue.")
                        tts_handler.clear_queue()

                    # 4. Reset the tts_buffer
                    logging.debug("Resetting tts_buffer.")
                    tts_buffer = ""

                    # ==== End of Clearing ====

                    # Determine which endpoint to use based on configuration
                    endpoint = config.get("default_endpoint", "chat")

                    # Prepare the data payload based on endpoint
                    if endpoint == "generate":
                        data = {
                            "model": config.get("model", "llava"),
                            "prompt": accumulated_text.strip(),
                            "stream": config.get("stream", True),
                            "format": config.get("format", "json"),
                            "options": {
                                "temperature": config.get("temperature", 0.7),
                                "top_k": config.get("top_k", 40),
                                "top_p": config.get("top_p", 0.9),
                                "seed": config.get("seed", 42),
                                "stop": config.get("stop", ["\n", "user:", "assistant:"]),
                                "keep_alive": config.get("keep_alive", "5m"),
                                "mirostat": config.get("mirostat", 1),
                                "mirostat_tau": config.get("mirostat_tau", 0.8),
                                "mirostat_eta": config.get("mirostat_eta", 0.6),
                                "presence_penalty": config.get("presence_penalty", 1.5),
                                "frequency_penalty": config.get("frequency_penalty", 1.0),
                                "repeat_penalty": config.get("repeat_penalty", 1.2),
                                "penalize_newline": config.get("penalize_newline", True),
                                "raw": config.get("raw", False)
                            }
                        }
                        # If 'tools' are specified in config, include them (applicable for 'chat' endpoint)
                        if config.get("tools"):
                            data["tools"] = config["tools"]
                    elif endpoint == "chat":
                        data = {
                            "model": config.get("model", "llava"),
                            "messages": chat_history,
                            "stream": config.get("stream", True),
                            "format": config.get("format", "json"),
                            "options": {
                                "temperature": config.get("temperature", 0.7),
                                "top_k": config.get("top_k", 40),
                                "top_p": config.get("top_p", 0.9),
                                "seed": config.get("seed", 42),
                                "stop": config.get("stop", ["\n", "user:", "assistant:"]),
                                "keep_alive": config.get("keep_alive", "5m"),
                                "mirostat": config.get("mirostat", 1),
                                "mirostat_tau": config.get("mirostat_tau", 0.8),
                                "mirostat_eta": config.get("mirostat_eta", 0.6),
                                "presence_penalty": config.get("presence_penalty", 1.5),
                                "frequency_penalty": config.get("frequency_penalty", 1.0),
                                "repeat_penalty": config.get("repeat_penalty", 1.2),
                                "penalize_newline": config.get("penalize_newline", True),
                                "raw": config.get("raw", False)
                            }
                        }
                        # If 'tools' are specified in config, include them
                        if config.get("tools"):
                            data["tools"] = config["tools"]
                    else:
                        logging.error(f"Unsupported endpoint selected: {endpoint}")
                        conversation_log.append({"role": "error", "content": f"Unsupported endpoint selected: {endpoint}"})
                        accumulated_text = ""
                        last_speech_time = None
                        continue

                    # Capture camera frame for inference and save to 'frames' directory
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    image_filename = f"inference_frame_{timestamp}.jpg"
                    image_path = os.path.join(temp_dir, image_filename)
                    logging.info(f"Capturing camera frame to '{image_path}'...")
                    try:
                        # Execute libcamera-still to capture a frame
                        subprocess.run(['libcamera-still', '-o', image_path, '--width', '640', '--height', '480', '-t', '1'], check=True, stdout=SUBPROCESS_STDERR, stderr=SUBPROCESS_STDERR)
                        logging.info(f"Captured frame saved to '{image_path}'.")
                        
                        # Save the image to 'frames' directory with timestamped name
                        frames_image_path = os.path.join(FRAMES_DIR, f"{timestamp}.jpg")
                        shutil.copy(image_path, frames_image_path)
                        logging.info(f"Image saved to frames directory at '{frames_image_path}'.")
                        
                        if tts_handler and config.get("tts_enabled", False):
                            tts_handler.speak(f"Captured a frame for inference at {timestamp}.")
                        # Display a message about the captured frame
                        conversation_log.append({"role": "assistant", "content": f"Captured a frame for inference at {timestamp}."})
                    except subprocess.CalledProcessError as e:
                        error_msg = f"Failed to capture camera frame: {e}"
                        conversation_log.append({"role": "error", "content": error_msg})
                        logging.error(error_msg)
                        accumulated_text = ""
                        last_speech_time = None
                        continue

                    # Send accumulated_text and image_path to Ollama API
                    api_thread = threading.Thread(
                        target=send_to_ollama_api_stream,
                        args=(endpoint, data, response_queue, stop_event, config, chat_history, image_path),
                        name="OllamaAPIThread",
                        daemon=True
                    )
                    api_thread.start()
                    accumulated_text = ""
                    last_speech_time = None

            # Check if any new API response tokens are available
            try:
                while True:
                    msg_type, message = response_queue.get_nowait()
                    if msg_type == "progress":
                        ollama_response += message
                        logging.debug(f"Received token: {message}")
                        # Append token to the assistant's last message
                        if 'assistant_entry_index' not in locals():
                            conversation_log.append({"role": "assistant", "content": ""})
                            assistant_entry_index = len(conversation_log) - 1
                        conversation_log[assistant_entry_index]['content'] += message

                        # ==== Sentence Chunking and TTS Playback ====
                        # Append token to tts_buffer
                        tts_buffer += message
                        # Check for sentence-ending punctuation
                        sentences = []
                        while True:
                            match = sentence_end_pattern.search(tts_buffer)
                            if match:
                                end = match.end()
                                sentence = tts_buffer[:end]
                                sentences.append(sentence.strip())
                                tts_buffer = tts_buffer[end:].strip()
                            else:
                                break
                        # Send complete sentences to TTSHandler
                        for sentence in sentences:
                            if sentence:
                                tts_handler.speak(sentence)
                        # ==== End of TTS Playback ====
                    elif msg_type == "error":
                        conversation_log.append({"role": "error", "content": message})
                    elif msg_type == "complete":
                        # API response is complete
                        break
            except Empty:
                pass

            # Additionally, check if API response is complete and send any remaining text to TTS
            if api_thread and not api_thread.is_alive() and tts_buffer:
                tts_handler.speak(tts_buffer.strip())
                tts_buffer = ""

            # Enforce MAX_CONVERSATION_LINES
            while len(conversation_log) > MAX_CONVERSATION_LINES:
                removed_entry = conversation_log.pop(0)
                logging.debug(f"Removed from conversation_log: {removed_entry}")

            # Get terminal dimensions
            height, width = stdscr.getmaxyx()

            # Clear the screen
            stdscr.erase()

            # Display conversation_log entries with wrapping
            display_text = ""
            for entry in conversation_log:
                if entry['role'] == 'system':
                    prefix = "System: "
                elif entry['role'] == 'user':
                    prefix = "User: "
                elif entry['role'] == 'assistant':
                    prefix = "Ollama: "
                elif entry['role'] == 'error':
                    prefix = "Error: "
                else:
                    prefix = ""
                
                if entry['role'] != 'separator':
                    wrapped_content = textwrap.fill(f"{prefix}{entry['content']}", width=width-1)
                else:
                    wrapped_content = f"{prefix}"
                
                display_text += wrapped_content + "\n"

            # Now split display_text into lines and display only the last 'max_display_lines' lines
            display_lines = display_text.split('\n')
            max_display_lines = height - 2  # Reserve last line for instructions
            if len(display_lines) > max_display_lines:
                display_lines = display_lines[-max_display_lines:]

            # Display the lines
            for idx, line in enumerate(display_lines):
                try:
                    stdscr.addstr(idx, 0, line[:width-1])
                except curses.error:
                    pass  # Ignore if the string is too long for the window

            # Instructions at the bottom
            instruction = "Press 'M' for Menu | 'Q' to Quit."
            try:
                stdscr.addstr(height-1, 0, instruction[:width-1], curses.A_BOLD)
            except curses.error:
                pass  # Ignore if the string is too long for the window

            # Refresh the screen
            stdscr.refresh()

    except Exception as e:
        # In case of unexpected errors, display them before exiting
        stdscr.clear()
        h, w = stdscr.getmaxyx()
        error_message = f"An error occurred: {str(e)}"
        try:
            stdscr.addstr(h//2, w//2 - len(error_message)//2, error_message, curses.A_BOLD)
            stdscr.refresh()
            stdscr.getch()
        except curses.error:
            pass  # If even this fails, just exit
        logging.error(f"Unhandled exception: {e}")
    finally:
        # Signal threads to stop and wait for them to finish
        stop_event.set()
        speech_thread.join()
        if api_thread:
            api_thread.join()
        if tts_handler:
            tts_handler.stop()
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)
        logging.info("Cleaned up temporary files.")

def display_error(stdscr, message, tts_handler=None, config=None):
    """Display an error message in the curses interface and optionally speak it."""
    stdscr.clear()
    h, w = stdscr.getmaxyx()
    try:
        # Split message into lines if it's too long
        lines = message.split('\n')
        for idx, line in enumerate(lines):
            if idx >= h - 4:
                break  # Prevent writing beyond the window
            stdscr.addstr(h//2 - len(lines)//2 + idx, w//2 - len(line)//2, line[:w-1])
        stdscr.addstr(h//2 + 2, w//2 - 10, "Press any key to exit.")
        stdscr.refresh()
        stdscr.getch()
        # Speak the error message if TTS is enabled
        if tts_handler and config.get("tts_enabled", False):
            tts_handler.speak("An error occurred. Please check the logs for details.")
    except curses.error:
        pass  # If even this fails, just exit

def display_download_status(stdscr, message):
    """Display a status message in the curses interface."""
    h, w = stdscr.getmaxyx()
    status_window = curses.newwin(3, w-2, h//2 - 1, 1)
    status_window.box()
    try:
        status_window.addstr(1, 2, message[:w-6])
    except curses.error:
        pass  # Handle window too small
    status_window.refresh()

def display_download_progress(stdscr, message):
    """Display a progress message in the curses interface."""
    h, w = stdscr.getmaxyx()
    progress_window = curses.newwin(3, w-2, h//2 + 2, 1)
    progress_window.box()
    try:
        progress_window.addstr(1, 2, message[:w-6])
    except curses.error:
        pass  # Handle window too small
    progress_window.refresh()

def display_download_complete(stdscr, model_name):
    """Display a completion message in the curses interface."""
    h, w = stdscr.getmaxyx()
    complete_window = curses.newwin(3, w-2, h//2 + 6, 1)
    complete_window.box()
    try:
        complete_window.addstr(1, 2, f"Model '{model_name}' download completed.")
    except curses.error:
        pass  # Handle window too small
    complete_window.refresh()

# ======================= Entry Point ======================================
if __name__ == "__main__":
    if not is_venv():
        print("Setting up the virtual environment and installing dependencies. This may take a few minutes...")
        create_venv()
        install_dependencies()
        activate_venv()
        # Download Vosk model
        download_vosk_model()
        # Relaunch the script within the virtual environment
        if os.name == 'nt':
            python_executable = os.path.join(VENV_DIR, "Scripts", "python.exe")
        else:
            python_executable = os.path.join(VENV_DIR, "bin", "python")
        if not os.path.exists(python_executable):
            logging.error(f"Pip executable not found at {python_executable}")
            sys.exit("Error: Pip executable not found. Check the virtual environment setup.")
        logging.info("Relaunching the script within the virtual environment.")
        try:
            subprocess.check_call([python_executable] + sys.argv, stderr=SUBPROCESS_STDERR)
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to relaunch the script within the virtual environment: {e}")
            sys.exit("Error: Failed to relaunch the script within the virtual environment.")
        sys.exit()
    else:
        # If already in virtual environment, proceed to run the main function within curses
        curses.wrapper(main)
