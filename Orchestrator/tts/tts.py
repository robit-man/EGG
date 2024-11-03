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
from num2words import num2words
import subprocess  # For calling pactl to mute/unmute mic
import hashlib

# ============================
# Configuration and Constants
# ============================
CONFIG_FILE = 'tts.cf'
script_uuid = str(uuid.uuid4())
script_name = 'TTS_Engine'
tts_queue = queue.Queue(maxsize=20)         # Queue for incoming TTS requests
playback_queue = queue.Queue()              # Queue for audio playback
cancel_event = threading.Event()            # Event to signal cancellation
reset_playback_event = threading.Event()    # Event to signal playback stream reset
config_lock = threading.Lock()              # Lock for thread-safe config access
sample_rate = 22050                         # Default sample rate for Riva TTS
output_channels = 1                         # Mono audio for speaker output
pyaudio_instance = pyaudio.PyAudio()        # PyAudio instance
stream = None                               # PyAudio stream
selected_voice = None                       # Selected voice for TTS
asr_muted = False                           # Flag to mute ASR during playback
tts_state = "IDLE"                          # Current state: RECEIVING, CHUNKING, INFERENCE, PLAYBACK
is_muted = False                            # Track if the microphone is currently muted

recent_audio_patterns = []                  # List to store hashes of recent audio patterns

# Constants
MAX_SENTENCES = 5                           # Max sentences per chunk
MAX_CHARACTERS = 350                        # Max characters per chunk

# Configuration defaults
config = {
    'orchestrator_host': 'localhost',
    'orchestrator_ports': '6000-6099',
    'port_range': '6200-6300',
    'data_port': None,
    'output_mode': 'speaker',
    'server': 'localhost:50051',
    'mute_during_playback': True,
    'voice': None,                           # Voice will be selected upon startup
    'language_code': 'en-US'                 # Default language code
}

# Logger setup
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

# Regex patterns
SPECIAL_CHAR_PATTERN = re.compile(r'[!"#$%&()*+,\-./:;<=>?@[\\\]^_`{|}~]')
NUMBER_PATTERN = re.compile(r'-?\d+(?:,\d{3})*(?:\.\d+)?')

# ============================
# Configuration Load/Save
# ============================
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

# ============================
# Port Parsing and Selection
# ============================
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
        if port in tried_ports:
            continue
        tried_ports.add(port)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(('localhost', port)) != 0:
                return port
    raise Exception("No available ports found in range")

# ============================
# Orchestrator Registration
# ============================
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

# ============================
# Start Data Listener
# ============================
def start_data_listener():
    host = '0.0.0.0'
    port_range = parse_port_range(config['port_range'])
    data_port = config['data_port']

    if not data_port:
        logger.error("No data port assigned. Exiting data listener.")
        return

    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
                server_socket.bind((host, data_port))
                server_socket.listen(5)
                server_socket.settimeout(1.0)  # Set timeout to allow periodic check
                logger.info(f"Data listener started on port {data_port}.")

                while True:
                    try:
                        client_socket, addr = server_socket.accept()
                        logger.info(f"Accepted connection from {addr}.")
                        threading.Thread(target=handle_client_connection, args=(client_socket,), daemon=True).start()
                    except socket.timeout:
                        continue  # Timeout occurred, loop back and check cancel_event
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

# ============================
# Fetch and Set Voice
# ============================
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

# ============================
# Hash Calculation
# ============================
def calculate_hash(audio_data):
    """Calculate a hash for a segment of audio data to detect repeats."""
    return hashlib.md5(audio_data).hexdigest()

# ============================
# Detect Playback Error
# ============================
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

# ============================
# Playback Thread Function
# ============================
def playback_thread_func():
    global stream
    while True:
        try:
            # Wait for either a playback chunk or a reset signal
            if reset_playback_event.is_set():
                logger.info("Reset playback event detected. Resetting playback stream.")
                # Reset the playback stream
                if stream and stream.is_active():
                    stream.stop_stream()
                    stream.close()
                # Reinitialize the stream
                stream = pyaudio_instance.open(
                    format=pyaudio.paInt16,
                    channels=output_channels,
                    rate=sample_rate,
                    output=True,
                    frames_per_buffer=1024
                )
                logger.info("Playback stream has been reset.")
                reset_playback_event.clear()

            # Attempt to get a chunk from the playback queue
            chunk = playback_queue.get(timeout=0.5)
            if cancel_event.is_set():
                logger.info("Playback cancellation detected. Clearing playback queue.")
                # Clear the playback queue safely
                while not playback_queue.empty():
                    try:
                        playback_queue.get_nowait()
                    except queue.Empty:
                        break
                # Stop and close the stream
                if stream and stream.is_active():
                    logger.info("Stopping playback stream due to cancellation.")
                    stream.stop_stream()
                    stream.close()
                # Reopen the stream for future playback
                stream = pyaudio_instance.open(
                    format=pyaudio.paInt16,
                    channels=output_channels,
                    rate=sample_rate,
                    output=True,
                    frames_per_buffer=1024
                )
                # Now clear the cancel event after handling
                cancel_event.clear()
                continue  # Skip writing this chunk

            if chunk:
                # Before writing, check if cancellation was requested
                if cancel_event.is_set():
                    logger.info("Cancellation requested. Skipping current playback.")
                    cancel_event.clear()
                    continue
                stream.write(chunk)
        except queue.Empty:
            continue  # No audio to play, continue listening
        except Exception as e:
            logger.error(f"Error during playback: {e}")
            continue  # Continue listening despite errors


# ============================
# Handle Client Connection
# ============================
def handle_client_connection(client_socket):
    with client_socket:
        while True:
            try:
                client_socket.settimeout(1.0)  # Set timeout to allow periodic checks
                data = client_socket.recv(1024).decode().strip()
                if not data:
                    break
                if data.lower() == '/info':
                    threading.Thread(target=send_info, args=(client_socket,), daemon=True).start()
                else:
                    processed_text = preprocess_text(data)
                    logger.info("Data received and enqueued for TTS processing.")
                    # Cancel current TTS processing
                    cancel_current_tts()
                    # Wait briefly to ensure cancellation has been processed
                    time.sleep(0.1)
                    # Enqueue new data
                    tts_queue.put(processed_text)
            except socket.timeout:
                continue  # Timeout occurred, loop back
            except Exception as e:
                logger.error(f"Error handling client connection: {e}")
                break

# ============================
# Preprocess Text
# ============================
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

# ============================
# Send Info to Client
# ============================
def send_info(client_socket):
    try:
        response = f"{script_name}\n{script_uuid}\n"
        with config_lock:
            for key, value in config.items():
                response += f"{key}={value}\n"
        response += "EOF\n"
        client_socket.sendall(response.encode())
    except Exception as e:
        logger.error(f"Error sending info to client: {e}")

# ============================
# Update Orchestrator Data Port
# ============================
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

# ============================
# Synthesize and Play Function
# ============================
def synthesize_and_play(tts_stub):
    global tts_state
    max_retries = 2
    failure_count = 0

    while True:
        try:
            if cancel_event.is_set():
                logger.info("Synthesis cancellation detected. Waiting for cancellation to be processed.")
                # Clear the cancel event to prevent infinite loop
                cancel_event.clear()
                continue

            tts_state = "RECEIVING"
            try:
                text = tts_queue.get(timeout=0.5)
            except queue.Empty:
                continue  # No data to process

            # Start chunking text
            tts_state = "CHUNKING"
            sentences = re.split(r'(?<=[.!?])\s+', text)
            chunk, sentence_count = "", 0

            mute_microphone()  # Mute once at the start of playback

            for sentence in sentences:
                if cancel_event.is_set():
                    logger.info("Cancelled during chunking.")
                    cancel_event.clear()  # Clear the event here
                    break

                if sentence_count < MAX_SENTENCES and len(chunk) + len(sentence) <= MAX_CHARACTERS:
                    chunk += sentence + " "
                    sentence_count += 1
                else:
                    if chunk and not cancel_event.is_set():
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

            # Clear the cancel event after processing
            if cancel_event.is_set():
                cancel_event.clear()

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
                continue  # Retry immediately

            audio_data = np.frombuffer(resp.audio, dtype=np.int16)

            # Detect internal repetition and playback errors
            if detect_internal_repetition(audio_data) or \
               detect_playback_error(audio_data.tobytes(), recent_audio_patterns):
                logger.warning("Playback error detected. Requesting re-synthesis.")
                # Signal playback thread to reset the stream
                reset_playback_event.set()
                failure_count += 1
                if failure_count >= max_retries:
                    logger.error("Max retries reached due to repeated or erroneous audio. Skipping this chunk.")
                continue  # Retry with next attempt

            # Enqueue audio data for playback
            if not cancel_event.is_set():
                tts_state = "PLAYBACK"
                enqueue_audio_for_playback(audio_data)
                logger.info(f"Enqueued synthesized speech for chunk: '{chunk}'")
                success = True
            break  # Exit loop on success
        except grpc.RpcError as e:
            logger.error(f"Synthesis failed on attempt {attempt + 1}: {e.details()}")
            failure_count += 1
            if failure_count >= max_retries:
                logger.error("Max retries reached. Skipping this chunk.")
            time.sleep(0.5)  # Wait before retrying
        except Exception as e:
            logger.error(f"Unexpected error during synthesis: {e}")
            break

    tts_state = "IDLE"
    return success

def enqueue_audio_for_playback(audio_data):
    """Enqueues audio data into the playback queue in manageable chunks."""
    chunk_size = 1024  # Bytes
    audio_bytes = audio_data.tobytes()
    total_length = len(audio_bytes)
    current_position = 0

    while current_position < total_length:
        if cancel_event.is_set():
            logger.info("Playback enqueuing interrupted by cancellation.")
            break
        end_position = min(current_position + chunk_size, total_length)
        chunk = audio_bytes[current_position:end_position]
        playback_queue.put(chunk)
        current_position = end_position

# ============================
# Cancel Current TTS Function
# ============================
def cancel_current_tts():
    global tts_state
    if tts_state in ["INFERENCE", "PLAYBACK", "CHUNKING", "RECEIVING"]:
        logger.info("Cancelling current TTS synthesis and playback.")
        cancel_event.set()  # Set the cancel event

        # Clear the playback queue
        while not playback_queue.empty():
            try:
                playback_queue.get_nowait()
            except queue.Empty:
                break

        # Clear any pending chunks in the tts queue
        while not tts_queue.empty():
            try:
                tts_queue.get_nowait()
            except queue.Empty:
                break

        tts_state = "IDLE"
        logger.info("Canceled current TTS synthesis and playback.")
        # Do not clear the cancel_event here

# ============================
# Mute and Unmute Microphone
# ============================
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

# ============================
# Start TTS Client Function
# ============================
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

        # Initialize PyAudio stream
        stream = pyaudio_instance.open(
            format=pyaudio.paInt16,
            channels=output_channels,
            rate=sample_rate,
            output=True,
            frames_per_buffer=1024
        )
        logger.info("PyAudio stream initialized.")

        # Start playback thread
        playback_thread = threading.Thread(target=playback_thread_func, daemon=True)
        playback_thread.start()
        logger.info("Playback thread started.")

        # Start synthesis thread
        synth_thread = threading.Thread(target=synthesize_and_play, args=(tts_stub,), daemon=True)
        synth_thread.start()
        logger.info("Synthesis thread started.")

        # No need to join threads here

    except Exception as e:
        logger.error(f"Failed to start TTS client: {e}")

# ============================
# Main Function
# ============================
if __name__ == '__main__':
    read_config()  # Load the configuration from file
    if register_with_orchestrator():  # Register with the orchestrator if available
        listener_thread = threading.Thread(target=start_data_listener, daemon=True)
        listener_thread.start()  # Start data listener to handle incoming messages
        start_tts_client()  # Start the TTS client for processing and playback
        try:
            while True:
                time.sleep(1)  # Keep the main thread alive
        except KeyboardInterrupt:
            logger.info("Shutdown signal received. Exiting...")
            cancel_current_tts()
            # Signal the playback thread to reset the stream if needed
            reset_playback_event.set()
            if stream:
                stream.stop_stream()
                stream.close()
            pyaudio_instance.terminate()
    else:
        logger.error("Could not complete registration. Exiting.")  # Exit if registration fails
