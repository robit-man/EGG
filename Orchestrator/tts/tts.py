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
from concurrent.futures import ThreadPoolExecutor, as_completed

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
is_active = False  # Add this at the global level
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
    global is_active
    with client_socket:
        while not cancel_event.is_set():
            data = client_socket.recv(1024).decode().strip()
            if not data:
                break
            if data.lower() == '/info':
                send_info(client_socket)
            else:
                # Cancel ongoing TTS if synthesis or playback is active
                if is_active:
                    cancel_current_tts()
                    # Clear the queue only if we canceled an active TTS
                    with tts_queue.mutex:
                        tts_queue.queue.clear()

                # Process the new incoming text and add it to the queue
                processed_text = preprocess_text(data)
                tts_queue.put(processed_text)
                cancel_event.clear()  # Ensure we're ready for new processing


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
    global tts_state, stream, is_active
    max_retries = 2

    while not cancel_event.is_set():
        try:
            # Receive text from the queue
            tts_state = "RECEIVING"
            text = tts_queue.get(timeout=0.5)
            if cancel_event.is_set():
                break

            is_active = True  # Set to True when starting processing

            # Split into sentences for chunking
            tts_state = "CHUNKING"
            sentences = re.split(r'(?<=[.!?])\s+', text)
            mute_microphone()

            # Chunk processing setup
            chunk_list = []
            chunk = ""
            sentence_count = 0

            # Accumulate sentences until reaching the max sentence or character limits
            for sentence in sentences:
                if cancel_event.is_set():
                    break
                if sentence_count < MAX_SENTENCES and len(chunk) + len(sentence) <= MAX_CHARACTERS:
                    chunk += sentence + " "
                    sentence_count += 1
                else:
                    # Add completed chunk and start a new one
                    chunk_list.append(chunk.strip())
                    chunk = sentence + " "
                    sentence_count = 1

            # Add any remaining text in chunk to the list
            if chunk.strip():
                chunk_list.append(chunk.strip())

            # Process each chunk in separate threads
            with ThreadPoolExecutor(max_workers=3) as executor:
                future_to_chunk = {executor.submit(process_chunk, chunk, tts_stub, max_retries): chunk for chunk in chunk_list}
                for future in as_completed(future_to_chunk):
                    if cancel_event.is_set():
                        break
                    chunk = future_to_chunk[future]
                    try:
                        success = future.result(timeout=60)  # Set a timeout for each chunk processing
                        if not success:
                            logger.warning(f"Failed to synthesize chunk: '{chunk}'")
                    except Exception as e:
                        logger.error(f"Chunk '{chunk}' generated an exception: {e}")

            unmute_microphone()
            tts_state = "IDLE"
            is_active = False  # Set to False when processing is done

        except queue.Empty:
            if is_muted:
                unmute_microphone()
            continue
        except Exception as e:
            logger.error(f"Unexpected error in synthesis loop: {e}")
            if is_muted:
                unmute_microphone()
            time.sleep(1)
            is_active = False  # Ensure it's set to False in case of exception




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
    global tts_state, recent_audio_patterns, stream
    tts_state = "INFERENCE"
    success = False
    failure_count = 0
    synthesis_timeout = 10  # Timeout in seconds for the synthesis call
    chunk_timeout = 30  # Maximum time to spend on a single chunk
    chunk_start_time = time.time()

    while failure_count < max_retries:
        if cancel_event.is_set():
            logger.info("Cancelled during inference.")
            break
        if time.time() - chunk_start_time > chunk_timeout:
            logger.error("Chunk processing time exceeded maximum duration. Skipping chunk.")
            break
        try:
            logger.info(f"[Thread-{threading.get_ident()}] Starting synthesis for chunk: '{chunk}' (Attempt {failure_count + 1})")
            req = riva_tts_pb2.SynthesizeSpeechRequest(
                text=chunk,
                language_code=config['language_code'],
                encoding=riva.client.AudioEncoding.LINEAR_PCM,
                sample_rate_hz=sample_rate,
                voice_name=selected_voice[0]
            )

            start_time = time.time()
            resp = tts_stub.Synthesize(req, timeout=synthesis_timeout)
            end_time = time.time()
            logger.info(f"[Thread-{threading.get_ident()}] Synthesis completed in {end_time - start_time:.2f} seconds")

            if not resp.audio or cancel_event.is_set():
                failure_count += 1
                logger.warning(f"Synthesis attempt {failure_count} failed or canceled.")
                continue

            audio_data = np.frombuffer(resp.audio, dtype=np.int16)
            if detect_internal_repetition(audio_data) or detect_playback_error(audio_data.tobytes(), recent_audio_patterns):
                logger.warning("Playback error detected. Resetting playback.")
                reset_playback()
                failure_count += 1
                continue

            if not cancel_event.is_set():
                tts_state = "PLAYBACK"
                # Ensure the stream is open before writing
                if stream is None or not stream.is_active():
                    stream = pyaudio_instance.open(
                        format=pyaudio.paInt16,
                        channels=output_channels,
                        rate=sample_rate,
                        output=True,
                        frames_per_buffer=1024
                    )
                    logger.info("Playback stream reopened in process_chunk.")

                logger.info(f"[Thread-{threading.get_ident()}] Starting playback for chunk.")
                stream.write(audio_data.tobytes())
                logger.info(f"[Thread-{threading.get_ident()}] Finished playback for chunk: '{chunk}'")
                success = True
            break
        except grpc.RpcError as e:
            failure_count += 1
            if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                logger.error(f"Synthesis timed out on attempt {failure_count}: {e.details()}")
            else:
                logger.error(f"Synthesis failed on attempt {failure_count}: {e.details()}")
            time.sleep(0.5)
        except Exception as e:
            logger.error(f"Unexpected error during playback: {e}")
            failure_count += 1
            time.sleep(0.5)

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
    global stream
    if stream and stream.is_active():
        stream.stop_stream()
        stream.close()
        logger.info("Playback stream stopped and closed.")

    # Reopen the stream to reset it
    stream = pyaudio_instance.open(
        format=pyaudio.paInt16,
        channels=output_channels,
        rate=sample_rate,
        output=True,
        frames_per_buffer=1024
    )
    logger.info("Playback stream reset and reopened.")


def cancel_current_tts():
    global tts_state, is_active, stream
    if is_active and tts_state in ["INFERENCE", "PLAYBACK", "CHUNKING"]:
        cancel_event.set()  # Set the cancel event

        # Stop and close stream if playing
        if stream and stream.is_active():
            stream.stop_stream()
            stream.close()
            logger.info("Playback stream stopped and closed in cancel_current_tts.")

        # Reopen the stream to ensure it's ready for the next playback
        stream = pyaudio_instance.open(
            format=pyaudio.paInt16,
            channels=output_channels,
            rate=sample_rate,
            output=True,
            frames_per_buffer=1024
        )
        logger.info("Playback stream reopened in cancel_current_tts.")

        # Clear any pending chunks in the queue
        with tts_queue.mutex:
            tts_queue.queue.clear()

        tts_state = "IDLE"
        is_active = False  # Reset active state
        logger.info("Canceled current TTS synthesis and playback.")
        # Wait briefly to ensure processing halt before clearing
        time.sleep(0.1)
    cancel_event.clear()




def fade_out_audio():
    """Stops the audio playback stream."""
    try:
        if stream and stream.is_active():
            stream.stop_stream()
            logger.info("Playback stream stopped.")
    except Exception as e:
        logger.error(f"Failed to stop audio: {e}")



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
        # Keepalive settings
        keepalive_options = [
            ('grpc.keepalive_time_ms', 10000),  # Send keepalive ping every 10 seconds
            ('grpc.keepalive_timeout_ms', 5000),  # Timeout for keepalive ping
            ('grpc.keepalive_permit_without_calls', True),
            ('grpc.http2.max_pings_without_data', 0),
            ('grpc.http2.min_time_between_pings_ms', 10000),
            ('grpc.http2.min_ping_interval_without_data_ms', 5000),
        ]

        if config.get('use_ssl') and config.get('ssl_cert'):
            with open(config['ssl_cert'], 'rb') as f:
                creds = grpc.ssl_channel_credentials(f.read())
            channel = grpc.secure_channel(config['server'], creds, options=keepalive_options)
        else:
            channel = grpc.insecure_channel(config['server'], options=keepalive_options)

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
