#!/usr/bin/env python3
import os
import sys
import subprocess
import shutil

# ----- Virtual Environment Bootstrapping (run BEFORE any external imports) -----
VENV_DIR = "audio_venv"
REQUIRED_PACKAGES = ["pyusb", "PyAudio", "requests"]

if sys.prefix == sys.base_prefix:
    if not os.path.exists(VENV_DIR):
        print("Creating virtual environment...")
        subprocess.check_call([sys.executable, "-m", "venv", VENV_DIR])
    if os.name == "nt":
        python_executable = os.path.join(VENV_DIR, "Scripts", "python.exe")
    else:
        python_executable = os.path.join(VENV_DIR, "bin", "python")
    subprocess.check_call([python_executable, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([python_executable, "-m", "pip", "install"] + REQUIRED_PACKAGES)
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
HOST = 'localhost'
PORT = 64167

LOG_FILE = 'client_app.log'

# Audio configuration
RATE = 16000              # 16kHz
CHANNELS = 1              # Mono
FORMAT = None             # Will be set after PyAudio import
CHUNK_DURATION = 5        # seconds (used for fixed-length chunks, if needed)
BLOCK_DURATION = 0.2      # seconds per block
BLOCK_SIZE = int(RATE * BLOCK_DURATION)  # samples per block
# For the silence buffer: 3 seconds = 3/0.2 = 15 blocks
SILENCE_THRESHOLD_BLOCKS = 5

# ======================= Global Variables for Audio UI =================
current_rms = 0.0
speech_detected_current = False
current_doa = 0         # DOA angle readout
last_sent_message = ""
running = True          # Global flag to stop audio thread and UI

# ======================= Logging Configuration ==========================
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ======================= USB Tuning Interface (AEC and DOA) =========
import usb.core
import usb.util

PARAMETERS = {
    'SPEECHDETECTED':     (19, 22, 'int',   1,   0, 'ro', 'Speech detection status.'),
    'DOAANGLE':           (21, 0,  'int', 359,   0, 'ro', 'DOA angle.'),
    'AECFREEZEONOFF':     (18, 7,  'int',   1,   0, 'rw', 'Adaptive Echo Canceler updates inhibit. 0=Adapt on, 1=Freeze'),
    'AECSILENCELEVEL':    (18, 30, 'float', 1, 1e-09, 'rw', 'Threshold for signal detection in AEC'),
    'ECHOONOFF':          (19, 14, 'int',   1,   0, 'rw', 'Echo suppression. 0=OFF, 1=ON')
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

# ======================= System Volume Helpers =====================
def fade_out_volume(duration=0.1, steps=10):
    """
    Gradually reduce system volume from 100% to 0% over 'duration' seconds.
    """
    interval = duration / steps
    for i in range(steps):
        vol = 100 - int((100/steps) * (i+1))
        try:
            if shutil.which("pactl"):
                subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{vol}%"], check=True)
            elif shutil.which("amixer"):
                subprocess.run(["amixer", "set", "Master", f"{vol}%"], check=True)
            else:
                logger.warning("No suitable volume control tool found for fade-out.")
        except Exception as e:
            logger.warning(f"Fade-out step failed: {e}")
        time.sleep(interval)

def fade_in_volume(duration=0.1, steps=10):
    """
    Gradually restore system volume from 0% to 100% over 'duration' seconds.
    """
    interval = duration / steps
    for i in range(steps):
        vol = int((80/steps) * (i+1))
        try:
            if shutil.which("pactl"):
                subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{vol}%"], check=True)
            elif shutil.which("amixer"):
                subprocess.run(["amixer", "set", "Master", f"{vol}%"], check=True)
            else:
                logger.warning("No suitable volume control tool found for fade-in.")
        except Exception as e:
            logger.warning(f"Fade-in step failed: {e}")
        time.sleep(interval)

# ======================= Muffled Noise Generation (Not used when no speech) =====================
def generate_muffled_noise():
    """
    Generate a muffled noise chunk (low-amplitude white noise)
    of length CHUNK_DURATION seconds as 16-bit little-endian PCM samples.
    """
    num_samples = int(RATE * CHUNK_DURATION)
    import random
    amplitude = 100  # Low amplitude for muffled noise
    samples = [int(random.gauss(0, amplitude)) for _ in range(num_samples)]
    return struct.pack("<" + "h" * num_samples, *samples)

# ======================= Audio Streaming and Curses UI =====================
def calculate_rms(frames):
    """
    Calculate the Root Mean Square (RMS) amplitude of the audio frames.
    """
    count = len(frames) // 2
    if count == 0:
        return 0.0
    fmt = "<" + "h" * count
    try:
        samples = struct.unpack(fmt, frames)
    except struct.error as e:
        logger.error(f"Struct unpacking failed: {e}")
        return 0.0
    sum_squares = sum(sample**2 for sample in samples)
    return math.sqrt(sum_squares / count)

def audio_capture_thread(dev):
    """
    Continuous, voice-activated capture:
      - Continuously read audio blocks.
      - When speech is detected (via SPEECHDETECTED), if not already capturing,
        fade out the system volume over 1 second and begin capturing audio blocks.
      - While capturing, reset a silence counter for each block that has speech.
      - If silence lasts for at least 3 seconds (15 blocks) with no speech,
        finalize the chunk, send it downstream, fade the volume back in over 1 second,
        and then reset capturing state.
      - If no speech is detected and not capturing, nothing is sent.
    """
    global current_rms, speech_detected_current, current_doa, last_sent_message, running, SERVER_URL

    try:
        import pyaudio
        import requests
    except ImportError as e:
        logger.error(f"Failed to import required modules in audio thread: {e}")
        running = False
        return

    p = pyaudio.PyAudio()
    global FORMAT
    FORMAT = pyaudio.paInt16

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

    capturing = False
    current_chunk = []
    silence_count = 0

    while running:
        try:
            block_data = stream.read(BLOCK_SIZE, exception_on_overflow=False)
        except Exception as e:
            logger.error(f"Error reading audio block: {e}")
            time.sleep(0.1)
            continue
        current_rms = calculate_rms(block_data)
        try:
            speech_val = dev.read("SPEECHDETECTED") or 0
        except Exception as e:
            logger.error(f"Error reading SPEECHDETECTED: {e}")
            speech_val = 0
        try:
            current_doa = dev.read("DOAANGLE") or 0
        except Exception as e:
            logger.error(f"Error reading DOAANGLE: {e}")
            current_doa = 0

        if speech_val == 1:
            speech_detected_current = True
            silence_count = 0
            if not capturing:
                # Speech just started – fade out volume and start capturing.
                # (Reducing output volume helps prevent self-heard audio.)
                fade_out_volume(duration=1.0, steps=10)
                capturing = True
                current_chunk = []
            current_chunk.append(block_data)
        else:
            speech_detected_current = False
            if capturing:
                silence_count += 1
                current_chunk.append(block_data)
                # If silence lasts for at least 3 seconds (15 blocks), finalize the chunk.
                if silence_count >= SILENCE_THRESHOLD_BLOCKS:
                    chunk_data = b"".join(current_chunk)
                    last_sent_message = "Sent captured chunk with speech."
                    logger.info("No speech detected for 3 seconds. Sending captured audio chunk.")
                    try:
                        response = requests.post(SERVER_URL, data=chunk_data)
                        if response.status_code == 200:
                            last_sent_message += " Downstream acknowledged."
                        else:
                            last_sent_message += f" Downstream error: {response.status_code}"
                    except Exception as e:
                        last_sent_message = f"Error sending chunk: {e}"
                        logger.error(last_sent_message)
                    # Fade the volume back in and reset capturing state
                    fade_in_volume(duration=1.0, steps=10)
                    capturing = False
                    current_chunk = []
                    silence_count = 0
        time.sleep(BLOCK_DURATION * 0.1)
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
    the current speech detection state, the current DOA angle, and the last sending status.
    Press 'q' to quit.
    """
    global current_rms, speech_detected_current, current_doa, last_sent_message, running
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(200)
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
        stdscr.addstr(5, 0, f"DOA Angle: {current_doa}")
        stdscr.addstr(7, 0, f"Last Sent: {last_sent_message}")
        stdscr.refresh()
        try:
            key = stdscr.getch()
            if key == ord('q'):
                running = False
                break
        except Exception:
            pass
        time.sleep(0.1)

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

def main():
    """
    Main function to set up the virtual environment, initialize USB tuning,
    set recommended AEC parameters (including adjusted AECSILENCELEVEL to help mask self-generated audio),
    start the voice-activated audio capture thread, and run the curses interface.
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
    try:
        tuning_dev.write('AECFREEZEONOFF', 0)
        tuning_dev.write('ECHOONOFF', 1)
        # Adjust AECSILENCELEVEL to a higher threshold (from 1e-5 to 1e-4)
        # to help filter out low-level self-generated (feedback) audio.
        tuning_dev.write('AECSILENCELEVEL', 1e-4)
        logger.info("AEC settings updated: AECFREEZEONOFF=0, ECHOONOFF=1, AECSILENCELEVEL=1e-4")
    except Exception as e:
        logger.warning(f"Could not set recommended AEC parameters: {e}")
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
