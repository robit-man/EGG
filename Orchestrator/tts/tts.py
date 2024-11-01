import socket
import json
import uuid
import threading
import logging
import queue
import grpc
import os
import time
import pyaudio
import numpy as np
from riva.client.proto import riva_tts_pb2, riva_tts_pb2_grpc
import riva.client
import random
import re
import textwrap
from num2words import num2words
import subprocess  # For calling pactl to mute/unmute mic
import requests
import hashlib

# Configuration and Constants
CONFIG_FILE = 'tts.cf'
script_uuid = str(uuid.uuid4())
script_name = 'TTS_Engine'
tts_queue = queue.Queue(maxsize=20)
cancel_event = threading.Event()
config_lock = threading.Lock()
sample_rate = 22050  # Default sample rate for Riva TTS
output_channels = 1  # Mono audio for speaker output
pyaudio_instance = pyaudio.PyAudio()
stream = None
selected_voice = None
asr_muted = False  # Flag to mute ASR during playback
tts_state = "IDLE"  # Can be RECEIVING, CHUNKING, INFERENCE, PLAYBACK
is_muted = False  # Track if the microphone is currently muted

recent_audio_patterns = []

# Constants
MAX_SENTENCES = 5
MAX_CHARACTERS = 350
# Configuration defaults
config = {
    'orchestrator_host': 'localhost',
    'orchestrator_ports': '6000-6099',
    'port_range': '6200-6300',
    'data_port': None,
    'output_mode': 'speaker',
    'server': 'localhost:50051',
    'mute_during_playback': True,
    'voice': None,  # Voice will be selected upon startup
    'language_code': 'en-US'  # Default language code
}

# Logger setup
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

# Regex patterns
SPECIAL_CHAR_PATTERN = re.compile(r'[!"#$%&()*+,\-./:;<=>?@[\\\]^_`{|}~]')
NUMBER_PATTERN = re.compile(r'-?\d+(?:,\d{3})*(?:\.\d+)?')

# Configuration Load/Save
def read_config():
    global config, script_uuid
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            config.update(json.load(f))
        # Retrieve or generate UUID if it doesn’t exist
        script_uuid = config.get('script_uuid')
        if not script_uuid:
            script_uuid = str(uuid.uuid4())
            config['script_uuid'] = script_uuid
            write_config()
    else:
        # Generate and save a new UUID
        script_uuid = str(uuid.uuid4())
        config['script_uuid'] = script_uuid
        write_config()

def write_config():
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)

# Port Parsing and Selection
def parse_port_range(port_range_str):
    ports = []
    for part in port_range_str.split(','):
        if '-' in part:
            start, end = map(int, part.split('-'))
            ports.extend(range(start, end + 1))
        else:
            ports.append(int(part))
    return ports

def find_available_port(port_range):
    tried_ports = set()
    while len(tried_ports) < len(port_range):
        port = random.choice(port_range)
        tried_ports.add(port)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(('localhost', port)) != 0:
                return port
    raise Exception("No available ports found in range")

# Orchestrator Registration
def register_with_orchestrator():
    host = config['orchestrator_host']
    orch_ports = parse_port_range(config['orchestrator_ports'])
    port_range = parse_port_range(config['port_range'])
    data_port = config.get('data_port')

    if not data_port or port_in_use(data_port):
        data_port = find_available_port(port_range)
        config['data_port'] = data_port
        write_config()

    for port in orch_ports:
        try:
            logger.info(f"Attempting registration on {host}:{port}...")
            with socket.create_connection((host, port), timeout=5) as s:
                message = f"/register {script_name} {script_uuid} {data_port}\n"
                s.sendall(message.encode())
                ack = s.recv(1024).decode().strip()
                if ack.startswith('/ack'):
                    logger.info(f"Registered with orchestrator. Confirmed data port: {data_port}")
                    return True
        except Exception as e:
            logger.warning(f"Registration failed on {host}:{port} - {e}")

    logger.error("Failed to register with orchestrator.")
    return False

def port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(('localhost', port)) == 0

# Start Data Listener
def start_data_listener():
    host = '0.0.0.0'
    port_range = parse_port_range(config['port_range'])
    data_port = config['data_port']

    if not data_port:
        logger.error("No data port assigned. Exiting data listener.")
        return

    while not cancel_event.is_set():
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
                server_socket.bind((host, data_port))
                server_socket.listen(5)
                logger.info(f"Data listener started on port {data_port}.")

                while not cancel_event.is_set():
                    client_socket, addr = server_socket.accept()
                    threading.Thread(target=handle_client_connection, args=(client_socket,), daemon=True).start()
                return
        except OSError as e:
            if "Address already in use" in str(e):
                logger.warning(f"Port {data_port} in use. Selecting new port...")
                data_port = find_available_port(port_range)
                config['data_port'] = data_port
                write_config()
                update_orchestrator_data_port(data_port)
            else:
                logger.error(f"Failed to bind data listener on port {data_port}: {e}")
                break

# Fetch Available Voices from Riva and Set Selected Voice
def fetch_and_set_voice(tts_stub):
    global selected_voice
    unmute_microphone() 
    try:
        response = tts_stub.GetRivaSynthesisConfig(riva_tts_pb2.RivaSynthesisConfigRequest())
        available_voices = [
            (model.parameters["voice_name"], model.parameters.get("language_code", "en-US"))
            for model in response.model_config if "voice_name" in model.parameters
        ]

        if available_voices:
            # Attempt to match the configured voice and language code
            selected_voice = next(
                (voice for voice in available_voices if voice[0] == config['voice']), 
                None
            )
            if not selected_voice:
                # Default to the first available voice if not specified
                selected_voice = available_voices[0]
                config['voice'] = selected_voice[0]
                config['language_code'] = selected_voice[1]
                write_config()

            logger.info(f"Using voice: {selected_voice[0]} with language: {selected_voice[1]}")
        else:
            logger.error("No voices available on Riva TTS server.")
            exit(1)
    except grpc.RpcError as e:
        logger.error(f"Failed to retrieve voices: {e.details()}")
        exit(1)

def calculate_hash(audio_data):
    """Calculate a hash for a segment of audio data to detect repeats."""
    return hashlib.md5(audio_data).hexdigest()

def detect_playback_error(audio_data, recent_audio_patterns):
    """Detects repeated patterns in audio data to identify playback errors."""
    current_hash = calculate_hash(audio_data[:1024])  # Hash of the first segment for comparison

    # Check if the new audio pattern is similar to any recent patterns
    for previous_hash in recent_audio_patterns:
        if current_hash == previous_hash:
            return True  # Repeated pattern detected, likely a playback error

    # Append the new hash to recent patterns and limit size of history
    recent_audio_patterns.append(current_hash)
    if len(recent_audio_patterns) > 5:  # Limit history to 5 segments
        recent_audio_patterns.pop(0)
        
    return False

def reset_playback():
    """Resets the playback stream to handle potential underrun or looping errors."""
    global stream  # Ensure stream is recognized as a global variable
    if stream and stream.is_active():
        fade_out_audio()  # Smoothly reduce volume if currently playing
        stream.stop_stream()
        stream.close()
    
    # Reopen the stream to reset it
    stream = pyaudio_instance.open(
        format=pyaudio.paInt16,
        channels=output_channels,
        rate=sample_rate,
        output=True,
        frames_per_buffer=1024
    )
    logger.info("Playback stream reset to handle error.")


# Process Incoming Text
def handle_client_connection(client_socket):
    with client_socket:
        while not cancel_event.is_set():
            data = client_socket.recv(1024).decode().strip()
            if not data:
                break
            if data.lower() == '/info':
                send_info(client_socket)
            else:
                processed_text = preprocess_text(data)
                tts_queue.put(processed_text)

def preprocess_text(text):
    # Step 1: Add spaces around non-word or sentence-oriented symbols, but ignore apostrophes
    text = SPECIAL_CHAR_PATTERN.sub(lambda m: ' ' + m.group(0) + ' ', text)

    # Step 2: Replace numbers (including negative and decimal) with their word equivalents
    def replace_number(match):
        num_str = match.group(0).replace(',', '')  # Remove commas for conversion
        try:
            number = float(num_str) if '.' in num_str else int(num_str)
            # Handle negative numbers and decimals
            return ' ' + num2words(number) + ' '
        except ValueError:
            return num_str  # Return the original string if conversion fails

    # Apply the number replacement
    text = NUMBER_PATTERN.sub(replace_number, text)

    # Step 3: Return the cleaned-up text
    return ' '.join(text.split())  # Remove extra spaces

def send_info(client_socket):
    response = f"{script_name}\n{script_uuid}\n"
    for key, value in config.items():
        response += f"{key}={value}\n"
    response += "EOF\n"
    client_socket.sendall(response.encode())

def update_orchestrator_data_port(new_data_port):
    host = config['orchestrator_host']
    orch_ports = parse_port_range(config['orchestrator_ports'])
    message = f"/update_data_port {script_uuid} {new_data_port}\n"
    
    for port in orch_ports:
        try:
            with socket.create_connection((host, port), timeout=5) as s:
                s.sendall(message.encode())
                logger.info(f"Notified orchestrator of new data port: {new_data_port}")
                return
        except Exception as e:
            logger.warning(f"Could not update orchestrator on port {port} - {e}")


# TTS Synthesis and Playback with Rapid Failure Recovery

def synthesize_and_play(tts_stub):
    global tts_state
    server_health_url = "http://localhost:8000/v2/health/ready"
    max_retries = 2
    failure_count = 0

    while not cancel_event.is_set():
        try:
            # Wait for new text, allowing cancellation mid-processing
            tts_state = "RECEIVING"
            text = tts_queue.get(timeout=0.5)
            
            # Start chunking text
            tts_state = "CHUNKING"
            sentences = re.split(r'(?<=[.!?])\s+', text)
            chunk, sentence_count = "", 0
            mute_microphone()  # Mute once at the start of playback

            for sentence in sentences:
                # Check for cancellation before processing each sentence
                if cancel_event.is_set():
                    logger.info("Cancelled during chunking.")
                    break
                
                if sentence_count < MAX_SENTENCES and len(chunk) + len(sentence) <= MAX_CHARACTERS:
                    chunk += sentence + " "
                    sentence_count += 1
                else:
                    if chunk:
                        success = process_chunk(chunk.strip(), tts_stub, max_retries)
                        if not success:
                            failure_count += 1
                            if failure_count > 3:
                                logger.warning("Multiple consecutive failures.")
                                time.sleep(1)
                        else:
                            failure_count = 0
                    chunk, sentence_count = sentence + " ", 1

            if chunk.strip() and not cancel_event.is_set():
                process_chunk(chunk.strip(), tts_stub, max_retries)

            unmute_microphone()  # Unmute once at the end of playback
            tts_state = "IDLE"

        except queue.Empty:
            if is_muted:
                unmute_microphone()
            continue
        except Exception as e:
            logger.error(f"Unexpected error in synthesis loop: {e}")
            if is_muted:
                unmute_microphone()
            time.sleep(1)

def detect_internal_repetition(audio_data, segment_size=256, repetition_threshold=3):
    """Detects internal repetition in audio data by checking for repeating segments."""
    segment_hashes = []
    repeated_segments = 0

    # Process the audio data in segments
    for i in range(0, len(audio_data) - segment_size, segment_size):
        segment = audio_data[i:i + segment_size]
        segment_hash = calculate_hash(segment)

        # Check for a repeating pattern in recent segments
        if segment_hash in segment_hashes:
            repeated_segments += 1
            if repeated_segments >= repetition_threshold:
                logger.warning("Internal repetition detected in synthesized audio. Requesting re-synthesis.")
                return True  # Repetition threshold exceeded, indicating looping
        else:
            repeated_segments = 0  # Reset if no repetition in this segment

        segment_hashes.append(segment_hash)

        # Limit segment history to avoid excessive memory usage
        if len(segment_hashes) > 10:
            segment_hashes.pop(0)

    return False

def process_chunk(chunk, tts_stub, max_retries):
    global tts_state, recent_audio_patterns
    tts_state = "INFERENCE"
    success = False
    failure_count = 0

    for attempt in range(max_retries):
        if cancel_event.is_set():
            logger.info("Cancelled during inference.")
            break
        try:
            req = riva_tts_pb2.SynthesizeSpeechRequest(
                text=chunk,
                language_code=config['language_code'],
                encoding=riva.client.AudioEncoding.LINEAR_PCM,
                sample_rate_hz=sample_rate,
                voice_name=selected_voice[0]
            )
            resp = tts_stub.Synthesize(req)

            # Check if response audio is empty or invalid, indicating a failed inference
            if not resp.audio:
                failure_count += 1
                logger.warning(f"Synthesis attempt {attempt + 1} failed: Empty or invalid response.")
                if failure_count >= max_retries:
                    logger.error("Max retries reached. Skipping this chunk.")
                    break
                continue  # Retry immediately

            audio_data = np.frombuffer(resp.audio, dtype=np.int16)

            # Detect internal repetition and playback errors (e.g., repeated audio patterns)
            if detect_internal_repetition(audio_data) or \
               detect_playback_error(audio_data.tobytes(), recent_audio_patterns):
                logger.warning("Playback error detected (repeated audio or invalid synthesis). Resetting playback.")
                reset_playback()
                failure_count += 1
                if failure_count >= max_retries:
                    logger.error("Max retries reached due to repeated or erroneous audio. Skipping this chunk.")
                    break
                continue  # Retry with next attempt

            # Proceed to playback only if synthesis was successful and audio is valid
            if not cancel_event.is_set():
                tts_state = "PLAYBACK"
                stream.write(audio_data.tobytes())
                logger.info(f"Played synthesized speech for chunk: '{chunk}'")
                success = True
            break  # Exit loop on success
        except grpc.RpcError as e:
            logger.error(f"Synthesis failed on attempt {attempt + 1}: {e.details()}")
            failure_count += 1
            if failure_count >= max_retries:
                logger.error("Max retries reached. Skipping this chunk.")
                break
            time.sleep(0.5)  # Wait before retrying
        except Exception as e:
            logger.error(f"Unexpected error during playback: {e}")
            break

    tts_state = "IDLE"
    return success




def detect_playback_error(audio_data, recent_audio_patterns):
    """Detects repeated patterns in audio data to identify playback errors."""
    for previous_data in recent_audio_patterns:
        if np.array_equal(audio_data[:100], previous_data):
            return True  # Repeated pattern detected, likely a playback error
    return False


def reset_playback():
    """Resets the playback stream to handle potential underrun or looping errors."""
    global stream  # Ensure stream is recognized as a global variable
    if stream and stream.is_active():
        fade_out_audio() 


def cancel_current_tts():
    global tts_state
    if tts_state in ["INFERENCE", "PLAYBACK", "CHUNKING"]:
        cancel_event.set()  # Set the cancel event

        # Fade out and stop stream if playing
        if stream and stream.is_active():
            fade_out_audio()
            stream.stop_stream()

        # Clear any pending chunks in the queue
        with tts_queue.mutex:
            tts_queue.queue.clear()

        tts_state = "IDLE"
        logger.info("Canceled current TTS synthesis and playback with fade-out.")
        cancel_event.clear()

def fade_out_audio(duration=0.3):
    """Gradually reduces audio volume to zero over the specified duration (in seconds)."""
    try:
        steps = 10
        step_duration = duration / steps
        volume_reduction = 0.1  # Each step reduces volume by 10%
        
        for _ in range(steps):
            if stream.is_active():
                # Reduce the playback volume on each step
                current_volume = stream._volume - volume_reduction
                stream._volume = max(0, current_volume)
                time.sleep(step_duration)
    except Exception as e:
        logger.error(f"Failed to fade out audio: {e}")


# Function to mute the microphone
def mute_microphone():
    global is_muted
    if config['mute_during_playback'] and not is_muted:
        try:
            default_source = subprocess.check_output(
                ["pactl", "get-default-source"], universal_newlines=True
            ).strip()
            subprocess.run(["pactl", "set-source-mute", default_source, "1"], check=True)
            is_muted = True
            logger.info(f"Microphone {default_source} muted.")
        except Exception as e:
            logger.error(f"Failed to mute the microphone: {e}")

# Function to unmute the microphone
def unmute_microphone():
    global is_muted
    if is_muted:  # Only unmute if currently muted
        try:
            default_source = subprocess.check_output(
                ["pactl", "get-default-source"], universal_newlines=True
            ).strip()
            subprocess.run(["pactl", "set-source-mute", default_source, "0"], check=True)
            is_muted = False
            logger.info(f"Microphone {default_source} unmuted.")
        except Exception as e:
            logger.error(f"Failed to unmute the microphone: {e}")


def start_tts_client():
    global stream
    try:
        if config.get('use_ssl') and config.get('ssl_cert'):
            with open(config['ssl_cert'], 'rb') as f:
                creds = grpc.ssl_channel_credentials(f.read())
            channel = grpc.secure_channel(config['server'], creds)
        else:
            channel = grpc.insecure_channel(config['server'])

        tts_stub = riva_tts_pb2_grpc.RivaSpeechSynthesisStub(channel)
        fetch_and_set_voice(tts_stub)
        stream = pyaudio_instance.open(
            format=pyaudio.paInt16,
            channels=output_channels,
            rate=sample_rate,
            output=True,
            frames_per_buffer=1024
        )
        tts_thread = threading.Thread(target=synthesize_and_play, args=(tts_stub,), daemon=True)
        tts_thread.start()
        tts_thread.join()
    except Exception as e:
        logger.error(f"Failed to start TTS client: {e}")

# Main function to initialize and run the TTS engine
if __name__ == '__main__':
    read_config()  # Load the configuration from file
    if register_with_orchestrator():  # Register with the orchestrator if available
        listener_thread = threading.Thread(target=start_data_listener, daemon=True)
        listener_thread.start()  # Start data listener to handle incoming messages
        start_tts_client()  # Start the TTS client for processing and playback
        listener_thread.join()  # Wait for listener thread to finish before exiting
    else:
        logger.error("Could not complete registration. Exiting.")  # Exit if registration fails
