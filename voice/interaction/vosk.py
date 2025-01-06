#!/usr/bin/env python3

import os
import sys
import subprocess
import threading
import socket
import json
import requests
import zipfile
import shutil
import time
from queue import Queue, Empty
import logging
from pathlib import Path

# ======================= Configuration Constants =======================
VENV_DIR = "vosk_venv"
VOSK_MODEL_PATH = "models/vosk-model-small-en-us-0.15"  # Path to Vosk model
VOSK_MODEL_ZIP_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"  # URL to download the Vosk model

HOST = 'localhost'  # or the hostname/IP where the server runs
PORT = 64162        # must match the port in the provided server script

# ======================= Logging Configuration ==========================
logging.basicConfig(
    filename='client_app.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ======================= Virtual Environment Management =================

def is_venv():
    """
    Check if the script is running inside a virtual environment.
    """
    return (
        (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) or
        (hasattr(sys, 'real_prefix') and sys.real_prefix != sys.prefix)
    )

def create_venv():
    """
    Create a virtual environment in VENV_DIR if it doesn't exist.
    """
    if not os.path.exists(VENV_DIR):
        logging.info("Creating virtual environment.")
        print("Creating virtual environment...")
        try:
            subprocess.check_call([sys.executable, "-m", "venv", VENV_DIR])
            logging.info("Virtual environment created successfully.")
            print("Virtual environment created successfully.")
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to create virtual environment: {e}")
            sys.exit("Error: Failed to create virtual environment.")
    else:
        logging.info("Virtual environment already exists.")
        print("Virtual environment already exists.")

def install_dependencies():
    """
    Install required Python packages in the virtual environment.
    """
    logging.info("Installing dependencies in the virtual environment.")
    print("Installing dependencies in the virtual environment...")
    try:
        if os.name == 'nt':
            pip_executable = os.path.join(VENV_DIR, "Scripts", "pip.exe")
        else:
            pip_executable = os.path.join(VENV_DIR, "bin", "pip")
        
        # Upgrade pip
        subprocess.check_call([pip_executable, "install", "--upgrade", "pip"], stderr=subprocess.DEVNULL)
        
        # Install required packages
        required_packages = [
            "vosk",
            "pyaudio",
            "requests"
        ]
        subprocess.check_call([pip_executable, "install"] + required_packages, stderr=subprocess.DEVNULL)
        logging.info("Dependencies installed successfully.")
        print("Dependencies installed successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to install dependencies: {e}")
        sys.exit("Error: Failed to install dependencies in the virtual environment.")

def activate_venv():
    """
    Activate the virtual environment by modifying sys.path.
    """
    if os.name == 'nt':
        venv_site_packages = Path(VENV_DIR) / "Lib" / "site-packages"
    else:
        venv_site_packages = Path(VENV_DIR) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    
    sys.path.insert(0, str(venv_site_packages))
    logging.info("Virtual environment activated.")
    print("Virtual environment activated.")

def relaunch_in_venv():
    """
    Relaunch the current script within the virtual environment.
    """
    if os.name == 'nt':
        python_executable = os.path.join(VENV_DIR, "Scripts", "python.exe")
    else:
        python_executable = os.path.join(VENV_DIR, "bin", "python")
    
    if not os.path.exists(python_executable):
        logging.error(f"Pip executable not found at {python_executable}")
        sys.exit("Error: Python executable not found in the virtual environment.")
    
    logging.info("Relaunching the script within the virtual environment.")
    print("Relaunching the script within the virtual environment...")
    try:
        subprocess.check_call([python_executable] + sys.argv, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to relaunch the script within the virtual environment: {e}")
        sys.exit("Error: Failed to relaunch the script within the virtual environment.")
    sys.exit()

def setup_virtual_environment():
    """
    Ensure that the virtual environment is set up and dependencies are installed.
    If not running inside the virtual environment, set it up and relaunch the script within it.
    """
    if not is_venv():
        create_venv()
        install_dependencies()
        relaunch_in_venv()
    else:
        activate_venv()

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

# ======================= Speech Recognition Thread with Vosk ============
def speech_recognition_thread(text_queue, stop_event):
    """Thread function for speech recognition using Vosk."""
    from vosk import Model, KaldiRecognizer, SetLogLevel
    import pyaudio

    SetLogLevel(-1)  # Suppress Vosk logs

    while not stop_event.is_set():
        try:
            if not os.path.exists(VOSK_MODEL_PATH):
                error_msg = f"Vosk model not found at {VOSK_MODEL_PATH}. Please download and place it accordingly."
                text_queue.put(error_msg)
                logging.error(error_msg)
                return

            model = Model(VOSK_MODEL_PATH)
            recognizer = KaldiRecognizer(model, 16000)

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
                p.terminate()
                time.sleep(5)  # Wait before retrying
                continue  # Retry initializing the microphone

            logging.info("Speech recognition thread started.")
            print("Listening... Speak into the microphone.")

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
                    break  # Exit the inner loop to restart recognition

            stream.stop_stream()
            stream.close()
            p.terminate()
            logging.info("Speech recognition stream closed.")

        except Exception as e:
            error_msg = f"Speech Recognition initialization error: {e}"
            text_queue.put(error_msg)
            logging.error(error_msg)
        
        # Wait before retrying to prevent rapid restarts
        time.sleep(5)
    
    logging.info("Speech recognition thread terminated.")

# ======================= Socket Communication Functions ===================
def send_and_receive(prompt):
    """
    Handles sending the prompt to the server and receiving the response.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            s.sendall(prompt.encode('utf-8'))
            # Receive the response from the server
            response = b""
            while True:
                part = s.recv(4096)
                if not part:
                    break
                response += part
            if response:
                print("\nServer Response:")
                print(response.decode('utf-8'))
            else:
                print("\nNo response received from the server.")
    except ConnectionRefusedError:
        print("\nError: Unable to connect to the server. Ensure that the server is running.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

# ======================= Main Speech Recognition and Communication Loop ==
def main():
    # Setup virtual environment
    setup_virtual_environment()

    # After setting up and activating the virtual environment, ensure that Vosk and pyaudio are importable
    try:
        from vosk import Model, KaldiRecognizer
        import pyaudio
    except ImportError as e:
        logging.error(f"Failed to import required modules after venv activation: {e}")
        sys.exit(f"Error: Required modules not found. {e}")

    # Download and prepare the Vosk model
    download_vosk_model()

    # Initialize Queues for inter-thread communication
    text_queue = Queue()

    # Event to signal threads to stop
    stop_event = threading.Event()

    # Start speech recognition thread
    speech_thread = threading.Thread(
        target=speech_recognition_thread,
        args=(text_queue, stop_event),
        name="SpeechRecognitionThread",
        daemon=True
    )
    speech_thread.start()

    print("Voice Client Started. Speak into the microphone.")
    print("Say 'exit' or 'quit' to terminate the client.\n")

    try:
        while not stop_event.is_set():
            try:
                # Wait for recognized text with a timeout
                text = text_queue.get(timeout=0.1)
                if isinstance(text, str):
                    if text.lower() in ['exit', 'quit']:
                        print("Exiting Voice Client.")
                        stop_event.set()
                        break
                    elif text.startswith("Speech Recognition error:"):
                        print(f"Error: {text}")
                        continue
                    else:
                        print(f"\nYou: {text}")
                        send_and_receive(text)
            except Empty:
                continue
    except KeyboardInterrupt:
        print("\nKeyboard Interrupt received. Exiting Voice Client.")
    finally:
        # Signal threads to stop and wait for them to finish
        stop_event.set()
        speech_thread.join()
        logging.info("Client terminated gracefully.")

# ======================= Entry Point ======================================
if __name__ == "__main__":
    main()
