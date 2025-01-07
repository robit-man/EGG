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
import math
import struct

# ======================= Configuration Constants =======================
VENV_DIR = "vosk_venv"
VOSK_MODEL_PATH = "models/vosk-model-en-us-0.42-gigaspeech"  # Path to Vosk model
VOSK_MODEL_ZIP_URL = "https://alphacephei.com/vosk/models/vosk-model-en-us-0.42-gigaspeech.zip"  # URL to download the Vosk model

HOST = 'localhost'  # or the hostname/IP where the server runs
PORT = 64162        # must match the port in the provided server script

# Volume Threshold for Filtering Recognition Results
VOLUME_THRESHOLD = 400  # Adjust based on microphone sensitivity

# Timeouts
SENTENCE_TIMEOUT = 2.0  # Seconds after which to send the recognized sentence
PARTIAL_TIMEOUT = 3.0    # Seconds after which to send the partial utterance
RESET_TIMEOUT = 5.0      # Seconds of total inactivity to reset the recognition buffer

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
        logging.error(f"Python executable not found at {python_executable}")
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

# ======================= Recognition Reset Function ======================
def reset_recognition(partial_queue, final_queue, reset_event):
    """
    Clears both partial and final queues and signals the recognizer to reset.
    """
    clear_queue(partial_queue)
    clear_queue(final_queue)
    reset_event.set()
    logging.info("Recognition reset triggered.")
    print("Recognition has been reset.")

# ======================= Speech Recognition Thread ============================
def speech_recognition_thread(partial_queue, final_queue, stop_event, reset_event):
    """Thread function for continuous speech recognition using Vosk."""
    from vosk import Model, KaldiRecognizer, SetLogLevel
    import pyaudio

    SetLogLevel(-1)  # Suppress Vosk logs

    # Constants for audio stream
    CHUNK = 1024  # Number of audio frames per buffer
    FORMAT = pyaudio.paInt16  # 16-bit int sampling
    CHANNELS = 1  # Mono audio
    RATE = 16000  # Sampling rate

    p = pyaudio.PyAudio()

    try:
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
        stream.start_stream()
        logging.info("Microphone stream started.")
        print("Microphone stream started.")
    except Exception as e:
        error_msg = f"Microphone error: {e}"
        print(f"Error: {error_msg}")
        logging.error(error_msg)
        p.terminate()
        return

    model = Model(VOSK_MODEL_PATH)
    recognizer = KaldiRecognizer(model, RATE)

    try:
        logging.info("Speech recognition thread started.")
        print("Listening for speech...")

        while not stop_event.is_set():
            # Check if a reset has been signaled
            if reset_event.is_set():
                logging.info("Reset event detected. Re-instantiating recognizer.")
                print("Re-instantiating recognizer...")
                recognizer = KaldiRecognizer(model, RATE)
                reset_event.clear()
                logging.info("Recognizer re-instantiated successfully.")

            try:
                data = stream.read(CHUNK, exception_on_overflow=False)

                # Calculate RMS (Root Mean Square) amplitude
                rms = calculate_rms(data)

                if rms < VOLUME_THRESHOLD:
                    # Considered as feedback or background noise; discard
                    continue
                else:
                    # Process the audio chunk
                    if recognizer.AcceptWaveform(data):
                        result = recognizer.Result()
                        result_json = json.loads(result)
                        text = result_json.get("text", "")
                        if text:
                            # Extract confidence scores
                            words = result_json.get("result", [])
                            if words:
                                confidences = [word.get("conf", 0) for word in words]
                                avg_confidence = sum(confidences) / len(confidences)
                            else:
                                avg_confidence = 0

                            final_queue.put((text, avg_confidence))
                            logging.info(f"Recognized final sentence: {text} (Avg Confidence: {avg_confidence})")
                    else:
                        partial = recognizer.PartialResult()
                        partial_json = json.loads(partial)
                        partial_text = partial_json.get("partial", "")
                        if partial_text:
                            partial_queue.put((partial_text, rms))
                            logging.debug(f"Recognized partial text: {partial_text} (RMS: {rms})")
            except Exception as e:
                error_msg = f"Speech Recognition error: {e}"
                print(f"Error: {error_msg}")
                logging.error(error_msg)
                break  # Exit the loop to restart recognition if needed

    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        logging.info("Microphone stream closed.")
        print("Microphone stream closed.")

    logging.info("Speech recognition thread terminated.")

def calculate_rms(audio_data):
    """
    Calculate the Root Mean Square (RMS) amplitude of the audio data.
    """
    # Unpack the binary data to integers
    count = len(audio_data) // 2  # 2 bytes per sample for paInt16
    format = "<" + "h" * count  # Little endian, signed short
    samples = struct.unpack(format, audio_data)

    # Compute RMS
    sum_squares = sum(sample**2 for sample in samples)
    rms = math.sqrt(sum_squares / count) if count > 0 else 0
    return rms

# ======================= Assembler Thread ================================
def assembler_thread(partial_queue, final_queue, stop_event, send_function, reset_event):
    """
    Thread function to assemble sentences and send to downstream API.
    Handles real-time partial updates and sends only final sentences after a timeout.
    Implements a 2-second inactivity timeout to send the latest recognized sentence.
    Additionally, sends partial utterances after a 3-second delay.
    Resets the recognition buffer if a timeout is reached with no new words.
    """
    last_final_time = None
    last_sentence = None

    last_partial_time = None
    last_partial_text = None

    while not stop_event.is_set():
        try:
            # Handle partial results
            try:
                partial_text, rms = partial_queue.get_nowait()
                if partial_text:
                    last_partial_text = partial_text
                    last_partial_time = time.time()
                    # Clear the terminal
                    clear_terminal()

                    # Print the partial sentence with RMS
                    print(f"Recognizing: {partial_text} (RMS: {rms})")
            except Empty:
                pass

            # Handle final results
            try:
                final, avg_confidence = final_queue.get_nowait()
                if final:
                    last_sentence = (final, avg_confidence)
                    last_final_time = time.time()
                    # Clear the terminal
                    clear_terminal()

                    # Print the final sentence
                    print(f"Recognized Sentence:\n{final}\n(Avg Confidence: {avg_confidence:.2f})")
                    logging.debug(f"Recognized sentence displayed: {final} (Avg Confidence: {avg_confidence})")
            except Empty:
                pass

            current_time = time.time()

            # Check if 2-second timeout has passed since the last final recognition
            if last_final_time and (current_time - last_final_time) > SENTENCE_TIMEOUT:
                if last_sentence:
                    # Send the last recognized sentence to the LLM
                    send_function(last_sentence[0])
                    logging.info(f"Sent sentence to downstream API: {last_sentence[0]} (Avg Confidence: {last_sentence[1]})")

                    # Reset recognition after sending
                    reset_recognition(partial_queue, final_queue, reset_event)

                    # Reset last_sentence and last_final_time
                    last_sentence = None
                    last_final_time = None

            # Check if 3-second timeout has passed since the last partial recognition
            if last_partial_time and (current_time - last_partial_time) > PARTIAL_TIMEOUT:
                if last_partial_text:
                    # Send the partial utterance to the LLM
                    send_function(last_partial_text)
                    logging.info(f"Sent partial utterance to downstream API: {last_partial_text}")

                    # Reset recognition after sending
                    reset_recognition(partial_queue, final_queue, reset_event)

                    # Reset last_partial_text and last_partial_time
                    last_partial_text = None
                    last_partial_time = None

            # Check for overall inactivity to reset recognition buffer
            # Reset only if timeout is reached with no new words
            if (last_final_time or last_partial_time) and (current_time - max(filter(None, [last_final_time, last_partial_time])) ) > RESET_TIMEOUT:
                # Reset the recognition buffer
                clear_terminal()
                print("No new speech detected for 5 seconds. Resetting recognition buffer.")
                logging.info("No new speech detected for 5 seconds. Resetting recognition buffer.")

                # Reset recognition after timeout
                reset_recognition(partial_queue, final_queue, reset_event)

                # Reset variables
                last_sentence = None
                last_final_time = None
                last_partial_text = None
                last_partial_time = None

            time.sleep(0.1)  # Small delay to prevent high CPU usage

        except Exception as e:
            logging.error(f"Assembler Thread error: {e}")

    # After stop_event is set, process any remaining final sentences
    while not final_queue.empty():
        try:
            final, avg_confidence = final_queue.get_nowait()
            if final:
                send_function(final)
                logging.info(f"Sent sentence to downstream API: {final} (Avg Confidence: {avg_confidence})")
        except Empty:
            break

    # Process any remaining partial utterances
    while not partial_queue.empty():
        try:
            partial_text, rms = partial_queue.get_nowait()
            if partial_text:
                send_function(partial_text)
                logging.info(f"Sent partial utterance to downstream API: {partial_text}")
        except Empty:
            break

def clear_terminal():
    """
    Clears the terminal screen.
    """
    if os.name == 'nt':
        os.system('cls')
    else:
        os.system('clear')

def clear_queue(q):
    """
    Clears all items from the given queue.
    """
    while not q.empty():
        try:
            q.get_nowait()
        except Empty:
            break

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

# ======================= Recognition Reset Function ======================
def reset_recognition(partial_queue, final_queue, reset_event):
    """
    Clears both partial and final queues and signals the recognizer to reset.
    """
    clear_queue(partial_queue)
    clear_queue(final_queue)
    reset_event.set()
    logging.info("Recognition reset triggered.")
    print("Recognition has been reset.")

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
    partial_queue = Queue()
    final_queue = Queue()

    # Event to signal threads to stop
    stop_event = threading.Event()

    # Event to signal a recognizer reset
    reset_event = threading.Event()

    # Start speech recognition thread with reset_event
    speech_thread = threading.Thread(
        target=speech_recognition_thread,
        args=(partial_queue, final_queue, stop_event, reset_event),
        name="SpeechRecognitionThread",
        daemon=True
    )
    speech_thread.start()

    # Start assembler thread to assemble sentences and send
    assembler = threading.Thread(
        target=assembler_thread,
        args=(partial_queue, final_queue, stop_event, send_and_receive, reset_event),
        name="AssemblerThread",
        daemon=True
    )
    assembler.start()

    print("Voice Client Started. Speak into the microphone.")
    print("Recognizing speech in real-time with volume filtering.")
    print("Press Ctrl+C to terminate the client.\n")

    try:
        while not stop_event.is_set():
            # Keep the main thread alive to listen for termination commands
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nKeyboard Interrupt received. Exiting Voice Client.")
    finally:
        # Signal threads to stop and wait for them to finish
        stop_event.set()
        speech_thread.join()
        assembler.join()
        logging.info("Client terminated gracefully.")

# ======================= Entry Point ======================================
if __name__ == "__main__":
    main()
