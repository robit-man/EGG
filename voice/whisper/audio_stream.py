#!/usr/bin/env python3
import os
import sys
import subprocess

# ----- Virtual Environment Bootstrapping (run BEFORE any external imports) -----
VENV_DIR = "audio_venv"
REQUIRED_PACKAGES = ["pyusb", "PyAudio", "requests"]

if sys.prefix == sys.base_prefix:
    # Not running inside a virtual environment; set it up.
    if not os.path.exists(VENV_DIR):
        print("Creating virtual environment...")
        subprocess.check_call([sys.executable, "-m", "venv", VENV_DIR])
    # Determine the correct Python executable for the venv.
    if os.name == "nt":
        python_executable = os.path.join(VENV_DIR, "Scripts", "python.exe")
    else:
        python_executable = os.path.join(VENV_DIR, "bin", "python")
    # Upgrade pip and install required packages.
    subprocess.check_call([python_executable, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([python_executable, "-m", "pip", "install"] + REQUIRED_PACKAGES)
    # Relaunch this script inside the virtual environment.
    os.execv(python_executable, [python_executable] + sys.argv)
# ----- End of Virtual Environment Bootstrapping -----


import logging
from pathlib import Path
import time
import threading
import struct
import math
import curses

# ======================= Configuration Constants =======================
HOST = 'localhost'  # or the hostname/IP where the server runs
PORT = 64167        # Updated port where the server listens for audio

LOG_FILE = 'client_app.log'

# Audio configuration
RATE = 16000              # 16kHz
CHANNELS = 1              # Mono
FORMAT = None             # Will be set after PyAudio import
CHUNK_DURATION = 5        # seconds
# We'll capture audio in smaller blocks for UI updates (200ms per block)
BLOCK_DURATION = 0.2      # seconds
BLOCK_SIZE = int(RATE * BLOCK_DURATION)  # samples per block
NUM_BLOCKS = int(CHUNK_DURATION / BLOCK_DURATION)  # number of blocks per chunk

# ======================= Global Variables for Audio UI =================
current_rms = 0.0
speech_detected_current = False
last_sent_message = ""
running = True  # Global flag to stop audio thread and UI

# ======================= Logging Configuration ==========================
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ======================= USB Tuning Interface (for SPEECHDETECTED) =========
import usb.core
import usb.util

# For this script we only need the SPEECHDETECTED parameter.
PARAMETERS = {
    'SPEECHDETECTED': (19, 22, 'int', 1, 0, 'ro', 'Speech detection status.')
}

class Tuning:
    TIMEOUT = 100000

    def __init__(self, dev):
        self.dev = dev

    def write(self, name, value):
        try:
            data = PARAMETERS[name]
        except KeyError:
            return
        if data[5] == 'ro':
            raise ValueError('{} is read-only'.format(name))
        param_id = data[0]
        if data[2] == 'int':
            payload = struct.pack(b'iii', data[1], int(value), 1)
        else:
            payload = struct.pack(b'ifi', data[1], float(value), 0)
        self.dev.ctrl_transfer(
            usb.util.CTRL_OUT | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
            0, 0, param_id, payload, self.TIMEOUT)

    def read(self, name):
        try:
            data = PARAMETERS[name]
        except KeyError:
            return None
        param_id = data[0]
        cmd = 0x80 | data[1]
        if data[2] == 'int':
            cmd |= 0x40
        length = 8
        response = self.dev.ctrl_transfer(
            usb.util.CTRL_IN | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
            0, cmd, param_id, length, self.TIMEOUT)
        response = struct.unpack(b'ii', response.tobytes())
        if data[2] == 'int':
            return response[0]
        else:
            return response[0] * (2.**response[1])

    def close(self):
        usb.util.dispose_resources(self.dev)

def find_tuning_device(vid=0x2886, pid=0x0018):
    dev = usb.core.find(idVendor=vid, idProduct=pid)
    if not dev:
        return None
    return Tuning(dev)

# ======================= Audio Streaming and Curses UI =====================
def calculate_rms(frames):
    """
    Calculate the Root Mean Square (RMS) amplitude of the audio frames.
    """
    count = len(frames) // 2  # 2 bytes per sample for paInt16
    if count == 0:
        return 0.0
    fmt = "<" + "h" * count
    try:
        samples = struct.unpack(fmt, frames)
    except struct.error as e:
        logger.error(f"Struct unpacking failed: {e}")
        return 0.0
    sum_squares = sum(sample**2 for sample in samples)
    rms = math.sqrt(sum_squares / count)
    return rms

def audio_capture_thread(dev):
    """
    Captures audio from the microphone in blocks, monitors the SPEECHDETECTED
    parameter via the USB device, and sends audio chunks downstream.
    
    If SPEECHDETECTED becomes 1 at any point within the chunk, the captured
    chunk is sent; otherwise, no audio is sent.
    """
    global current_rms, speech_detected_current, last_sent_message, running, SERVER_URL

    try:
        import pyaudio
        import requests
        import wave
    except ImportError as e:
        logger.error(f"Failed to import required modules in audio thread: {e}")
        running = False
        return

    p = pyaudio.PyAudio()
    global FORMAT
    FORMAT = pyaudio.paInt16  # 16-bit PCM

    try:
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=BLOCK_SIZE)
        logger.info("Microphone stream opened.")
    except Exception as e:
        logger.error(f"Failed to open microphone stream: {e}")
        p.terminate()
        running = False
        return

    # Optionally save noise for debugging.
    DEBUG_SAVE_NOISE = True
    DEBUG_AUDIO_DIR = "debug_noise"
    os.makedirs(DEBUG_AUDIO_DIR, exist_ok=True)

    while running:
        chunk_blocks = []
        speech_detected_current = False  # Reset flag for this chunk
        for i in range(NUM_BLOCKS):
            if not running:
                break
            try:
                block_data = stream.read(BLOCK_SIZE, exception_on_overflow=False)
            except Exception as e:
                logger.error(f"Error reading audio block: {e}")
                continue
            current_rms = calculate_rms(block_data)
            try:
                # Poll the USB tuning device for speech detection state.
                speech_state = dev.read("SPEECHDETECTED")
                if speech_state == 1:
                    speech_detected_current = True
            except Exception as e:
                logger.error(f"Error reading SPEECHDETECTED: {e}")
            chunk_blocks.append(block_data)
            time.sleep(BLOCK_DURATION * 0.1)  # slight pause for UI update

        chunk_data = b"".join(chunk_blocks)
        if speech_detected_current:
            to_send = chunk_data
            last_sent_message = "Sent chunk with speech."
            logger.info("Speech detected in chunk. Sending audio chunk.")
            try:
                response = requests.post(SERVER_URL, data=to_send)
                if response.status_code == 200:
                    last_sent_message += " Downstream acknowledged."
                else:
                    last_sent_message += f" Downstream error: {response.status_code}"
            except Exception as e:
                last_sent_message = f"Error sending chunk: {e}"
                logger.error(last_sent_message)
        else:
            # No speech detected; do not send any audio.
            last_sent_message = "No speech detected. Chunk not sent."
            logger.info("No speech detected in chunk. Not sending audio.")

    try:
        stream.stop_stream()
        stream.close()
    except Exception:
        pass
    p.terminate()
    logger.info("Microphone stream closed.")

def curses_main(stdscr):
    """
    Curses interface that displays a continuously updating audio volume bar,
    the current speech detection state, and the last sending status.
    
    Press 'q' to quit.
    """
    global current_rms, speech_detected_current, last_sent_message, running
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(200)  # refresh every 200ms

    while running:
        stdscr.clear()
        height, width = stdscr.getmaxyx()
        title = "Audio Streaming Interface - Press 'q' to quit"
        stdscr.addstr(0, 0, title[:width-1])
        max_bar_length = width - 20
        rms_for_display = min(current_rms, 3000)
        bar_length = int((rms_for_display / 3000) * max_bar_length)
        volume_bar = "[" + "#" * bar_length + "-" * (max_bar_length - bar_length) + "]"
        stdscr.addstr(2, 0, f"Volume: {volume_bar} {current_rms:6.1f}")
        speech_text = "Yes" if speech_detected_current else "No"
        stdscr.addstr(4, 0, f"Speech Detected: {speech_text}")
        stdscr.addstr(6, 0, f"Last Sent: {last_sent_message}")
        stdscr.refresh()
        try:
            key = stdscr.getch()
            if key == ord('q'):
                running = False
                break
        except Exception:
            pass
        time.sleep(0.1)
    # End of curses interface

# ======================= Legacy Socket Communication Function ============
def send_and_receive(prompt):
    """
    Handles sending the prompt to the server and receiving the response.
    """
    import socket
    try:
        logger.debug(f"Attempting to send prompt: {prompt}")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            s.sendall(prompt.encode('utf-8'))
            logger.debug("Prompt sent successfully. Awaiting response...")
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
    Main function to set up the virtual environment, initialize USB tuning,
    start the audio capture thread, and run the curses interface.
    """
    global SERVER_URL
    SERVER_URL = f"http://{HOST}:{PORT}"

    try:
        import pyaudio
        import requests
    except ImportError as e:
        logger.error(f"Failed to import required modules after venv activation: {e}")
        sys.exit(f"Error: Required modules not found. {e}")
    
    tuning_dev = find_tuning_device()
    if tuning_dev is None:
        logger.error("No USB tuning device found. Exiting.")
        sys.exit("Error: No USB tuning device found.")
    else:
        logger.info("USB tuning device connected.")
    
    audio_thread = threading.Thread(target=audio_capture_thread, args=(tuning_dev,), daemon=True)
    audio_thread.start()
    
    try:
        curses.wrapper(curses_main)
    except Exception as e:
        logger.error(f"Error in curses interface: {e}")
        print("Error in curses interface:", e)
    finally:
        global running
        running = False
        audio_thread.join(timeout=5)
        tuning_dev.close()
        logger.info("Exiting main.")

if __name__ == "__main__":
    main()
