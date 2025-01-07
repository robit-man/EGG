#!/usr/bin/env python3

import os
import sys
import subprocess
import logging
from pathlib import Path
import time

# ======================= Configuration Constants =======================
VENV_DIR = "audio_venv"
REQUIRED_PACKAGES = [
    "PyAudio",
    "requests"
]

HOST = 'localhost'  # or the hostname/IP where the server runs
PORT = 64167        # Updated port where the server listens for audio

LOG_FILE = 'client_app.log'

# Volume Threshold for Filtering Noise
VOLUME_THRESHOLD = 300  # Adjust based on microphone sensitivity

# Audio Chunk Size
CHUNK_DURATION = 5        # seconds
CHUNK_SIZE = 16000 * CHUNK_DURATION  # 16kHz * seconds

# ======================= Logging Configuration ==========================
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
        logger.info("Creating virtual environment.")
        print("Creating virtual environment...")
        try:
            subprocess.check_call([sys.executable, "-m", "venv", VENV_DIR])
            logger.info("Virtual environment created successfully.")
            print("Virtual environment created successfully.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create virtual environment: {e}")
            sys.exit("Error: Failed to create virtual environment.")
    else:
        logger.info("Virtual environment already exists.")
        print("Virtual environment already exists.")

def install_dependencies():
    """
    Install required Python packages in the virtual environment.
    """
    logger.info("Installing dependencies in the virtual environment.")
    print("Installing dependencies in the virtual environment...")
    try:
        if os.name == 'nt':
            pip_executable = os.path.join(VENV_DIR, "Scripts", "pip.exe")
        else:
            pip_executable = os.path.join(VENV_DIR, "bin", "pip")
        
        # Upgrade pip
        subprocess.check_call([pip_executable, "install", "--upgrade", "pip"], stderr=subprocess.DEVNULL)
        
        # Install required packages
        subprocess.check_call([pip_executable, "install"] + REQUIRED_PACKAGES, stderr=subprocess.DEVNULL)
        logger.info("Dependencies installed successfully.")
        print("Dependencies installed successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        sys.exit("Error: Failed to install dependencies in the virtual environment.")

def activate_venv():
    """
    Activate the virtual environment by modifying sys.path.
    """
    if os.name == 'nt':
        venv_site_packages = Path(VENV_DIR) / "Lib" / "site-packages"
    else:
        venv_site_packages = Path(VENV_DIR) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    
    if venv_site_packages.exists():
        sys.path.insert(0, str(venv_site_packages))
        logger.info("Virtual environment activated.")
        print("Virtual environment activated.")
    else:
        logger.error(f"Site-packages directory not found in virtual environment at {venv_site_packages}.")
        sys.exit("Error: Virtual environment site-packages directory not found.")

def relaunch_in_venv():
    """
    Relaunch the current script within the virtual environment.
    """
    if os.name == 'nt':
        python_executable = os.path.join(VENV_DIR, "Scripts", "python.exe")
    else:
        python_executable = os.path.join(VENV_DIR, "bin", "python")
    
    if not os.path.exists(python_executable):
        logger.error(f"Python executable not found at {python_executable}")
        sys.exit("Error: Python executable not found in the virtual environment.")
    
    logger.info("Relaunching the script within the virtual environment.")
    print("Relaunching the script within the virtual environment...")
    try:
        subprocess.check_call([python_executable] + sys.argv)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to relaunch the script within the virtual environment: {e}")
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

# ======================= Audio Streaming Function =========================

def send_audio_stream():
    """
    Captures audio from the microphone using PyAudio, filters out noise, and streams it to the server.
    """
    import pyaudio  # Import after activating venv
    import requests  # Import after activating venv
    import struct
    import math

    logger.info("Starting audio streaming.")
    print("Starting audio streaming. Press Ctrl+C to stop.")
    
    # Configuration
    FORMAT = pyaudio.paInt16  # 16-bit PCM
    CHANNELS = 1              # Mono
    RATE = 16000              # 16kHz
    # CHUNK_DURATION and CHUNK_SIZE are already defined globally
    
    def calculate_rms(frames):
        """
        Calculate the Root Mean Square (RMS) amplitude of the audio frames.
        
        Parameters:
            frames (bytes): The raw audio data.
        
        Returns:
            float: The RMS value.
        """
        # Convert bytes to integers
        count = len(frames) // 2  # 2 bytes per sample for paInt16
        format = "<" + "h" * count  # little endian, signed short
        try:
            samples = struct.unpack(format, frames)
        except struct.error as e:
            logger.error(f"Struct unpacking failed: {e}")
            return 0.0
        sum_squares = sum(sample**2 for sample in samples)
        rms = math.sqrt(sum_squares / count) if count > 0 else 0.0
        return rms

    try:
        p = pyaudio.PyAudio()
    except Exception as e:
        logger.error(f"Failed to initialize PyAudio: {e}")
        sys.exit(f"Error: Failed to initialize PyAudio: {e}")

    try:
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK_SIZE)
        logger.info("Microphone stream opened.")
        print("Microphone stream opened.")
    except Exception as e:
        logger.error(f"Failed to open microphone stream: {e}")
        p.terminate()
        sys.exit(f"Error: Failed to open microphone stream: {e}")

    try:
        while True:
            try:
                frames = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                if not frames:
                    logger.warning("No audio frames received.")
                    continue
                
                # Compute RMS to determine if audio contains speech
                rms = calculate_rms(frames)
                logger.debug(f"RMS: {rms}")
                
                if rms < VOLUME_THRESHOLD:
                    logger.info("Audio chunk deemed as noise. Skipping transmission.")
                    print("Noise detected. Skipping audio transmission.")
                    continue  # Skip sending this chunk
                
                # Send the audio data via POST request
                response = requests.post(SERVER_URL, data=frames)
                
                if response.status_code == 200:
                    transcribed_text = response.json().get('transcribed_text', '')
                    print(f"Transcribed Text: {transcribed_text}")
                else:
                    error = response.json().get('error', 'Unknown error')
                    print(f"Error: {error}")
            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt received. Stopping audio streaming.")
                print("\nStopping audio streaming.")
                break
            except Exception as e:
                logger.error(f"Error during audio streaming: {e}")
                print(f"Error during audio streaming: {e}")
                time.sleep(1)  # Brief pause before retrying
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        logger.info("Microphone stream closed.")
        print("Microphone stream closed.")

# ======================= Socket Communication Functions ===================
def send_and_receive(prompt):
    """
    Handles sending the prompt to the server and receiving the response.
    """
    try:
        logger.debug(f"Attempting to send prompt: {prompt}")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            s.sendall(prompt.encode('utf-8'))
            logger.debug("Prompt sent successfully. Awaiting response...")
            # Receive the response from the server
            response = b""
            while True:
                part = s.recv(4096)
                if not part:
                    break
                response += part
            if response:
                response_text = response.decode('utf-8')
                print("\nServer Response:")
                print(response_text)
                logger.debug(f"Received response: {response_text}")
            else:
                print("\nNo response received from the server.")
                logger.warning("No response received from the server.")
    except ConnectionRefusedError:
        error_msg = "Unable to connect to the server. Ensure that the server is running."
        print(f"\nError: {error_msg}")
        logger.error(error_msg)
    except Exception as e:
        error_msg = f"An unexpected error occurred: {e}"
        print(f"\nError: {error_msg}")
        logger.error(error_msg)

# ======================= Main Function ======================================

def main():
    """
    Main function to set up virtual environment and start audio streaming.
    """
    setup_virtual_environment()
    
    # After setting up and activating the virtual environment, ensure that PyAudio and requests are importable
    try:
        import pyaudio
        import requests
    except ImportError as e:
        logger.error(f"Failed to import required modules after venv activation: {e}")
        sys.exit(f"Error: Required modules not found. {e}")
    
    # Define the server URL after ensuring it's imported
    global SERVER_URL
    SERVER_URL = f"http://{HOST}:{PORT}"
    
    send_audio_stream()

# ======================= Entry Point ======================================
if __name__ == "__main__":
    main()
