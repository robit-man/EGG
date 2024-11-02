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
selected_voice = None
tts_state = "IDLE"  # Can be RECEIVING, CHUNKING, INFERENCE, PLAYBACK
is_muted = False  # Track if the microphone is currently muted
is_active = threading.Event()
audio_queue = queue.Queue(maxsize=10)
tts_state_lock = threading.Lock()
is_muted_lock = threading.Lock()
stream_lock = threading.Lock()

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

# Fetch Available Voices from Riva and Set Selected Voice
def fetch_and_set_voice(tts_stub):
    global selected_voice
    unmute_microphone()
    max_retries = 5
    retry_delay = 5  # seconds
    attempt = 0

    while attempt < max_retries or max_retries == 0:
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
                break
            else:
                logger.error("No voices available on Riva TTS server.")
                time.sleep(retry_delay)
                attempt += 1
        except grpc.RpcError as e:
            logger.error(f"Failed to retrieve voices: {e.details()}")
            logger.info(f"Retrying to fetch voices in {retry_delay} seconds...")
            time.sleep(retry_delay)
            attempt += 1
        except Exception as e:
            logger.error(f"Unexpected error while fetching voices: {e}")
            logger.info(f"Retrying to fetch voices in {retry_delay} seconds...")
            time.sleep(retry_delay)
            attempt += 1

    if selected_voice is None:
        raise RuntimeError("Failed to retrieve voices from the TTS server after multiple attempts.")

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
                # Cancel ongoing TTS if synthesis or playback is active
                if is_active.is_set():
                    cancel_current_tts()
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

# TTS Synthesis and Playback with Improved Time-Sensitive Handling
def synthesize_and_play(tts_stub):
    max_retries = 5  # Increase retries for aggressive generation

    while not cancel_event.is_set():
        try:
            update_tts_state("RECEIVING")
            text = tts_queue.get(timeout=0.5)
            if cancel_event.is_set():
                break

            is_active.set()  # Set to True when starting processing

            # Split into sentences for chunking
            update_tts_state("CHUNKING")
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
                    if chunk.strip():
                        chunk_list.append(chunk.strip())
                    chunk = sentence + " "
                    sentence_count = 1

            # Add any remaining text in chunk to the list
            if chunk.strip():
                chunk_list.append(chunk.strip())

            # Process chunks using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=5) as executor:
                future_to_index = {}
                futures = []
                for idx, chunk_text in enumerate(chunk_list):
                    if cancel_event.is_set():
                        break
                    future = executor.submit(process_chunk, idx, chunk_text, tts_stub, max_retries)
                    future_to_index[future] = idx
                    futures.append(future)

                # Collect results as they complete
                results = {}
                failed_chunks = set()
                next_index_to_play = 0

                for future in as_completed(futures):
                    if cancel_event.is_set():
                        break
                    idx = future_to_index[future]
                    try:
                        audio_data = future.result()
                        if audio_data is not None:
                            results[idx] = audio_data
                            logger.info(f"Chunk index {idx} synthesized successfully.")
                        else:
                            logger.warning(f"Failed to synthesize chunk at index {idx}.")
                            failed_chunks.add(idx)
                    except Exception as e:
                        logger.error(f"Exception in chunk synthesis at index {idx}: {e}")
                        failed_chunks.add(idx)

                    # Enqueue any ready chunks in order
                    while next_index_to_play in results or next_index_to_play in failed_chunks:
                        if next_index_to_play in results:
                            audio_queue.put(results[next_index_to_play])
                            del results[next_index_to_play]
                        elif next_index_to_play in failed_chunks:
                            logger.warning(f"Skipping failed chunk index {next_index_to_play}")
                        next_index_to_play += 1

            unmute_microphone()
            update_tts_state("IDLE")
            is_active.clear()  # Set to False when processing is done

        except queue.Empty:
            with is_muted_lock:
                if is_muted:
                    unmute_microphone()
            continue
        except Exception as e:
            logger.error(f"Unexpected error in synthesis loop: {e}")
            with is_muted_lock:
                if is_muted:
                    unmute_microphone()
            time.sleep(1)
            is_active.clear()  # Ensure it's set to False in case of exception


def process_chunk(idx, chunk, tts_stub, max_retries):
    failure_count = 0
    synthesis_timeout = 5  # Reduce timeout to fail faster
    chunk_timeout = 15  # Reduce maximum time to spend on a single chunk
    chunk_start_time = time.time()
    backoff_delay = 0.5  # Initial delay for exponential backoff

    # Count the number of words (tokens) in the chunk
    num_tokens = len(chunk.split())

    while failure_count < max_retries and not cancel_event.is_set():
        if time.time() - chunk_start_time > chunk_timeout:
            logger.error(f"Chunk processing time exceeded maximum duration for chunk index {idx}. Skipping chunk.")
            break
        try:
            logger.info(f"Starting synthesis for chunk index {idx} with length {len(chunk)} characters. (Attempt {failure_count + 1})")
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
            synthesis_time = end_time - start_time

            # Calculate tokens per second
            tokens_per_second = num_tokens / synthesis_time if synthesis_time > 0 else 0
            logger.info(f"Synthesis for chunk index {idx} completed in {synthesis_time:.2f} seconds. Tokens per second: {tokens_per_second:.2f}")

            # Check if the response contains valid audio data
            if not resp.audio:
                failure_count += 1
                logger.warning(f"Synthesis attempt {failure_count} for chunk index {idx} failed: No audio data received.")
                continue

            # Additional checks to validate the audio data
            audio_data = resp.audio
            if len(audio_data) == 0:
                failure_count += 1
                logger.warning(f"Synthesis attempt {failure_count} for chunk index {idx} failed: Empty audio data.")
                continue

            logger.info(f"Synthesized chunk index {idx} successfully.")
            return audio_data  # Return the audio data instead of queuing it

        except grpc.RpcError as e:
            failure_count += 1
            logger.error(f"Synthesis failed on attempt {failure_count} for chunk index {idx}: {e.code().name} - {e.details()}")
            # Exponential backoff with jitter
            sleep_time = backoff_delay + random.uniform(0, 0.5)
            logger.info(f"Retrying chunk index {idx} after {sleep_time:.2f} seconds...")
            time.sleep(sleep_time)
            backoff_delay *= 2  # Exponential increase
            continue
        except Exception as e:
            failure_count += 1
            logger.error(f"Unexpected error during synthesis attempt {failure_count} for chunk index {idx}: {e}")
            # Exponential backoff with jitter
            sleep_time = backoff_delay + random.uniform(0, 0.5)
            logger.info(f"Retrying chunk index {idx} after {sleep_time:.2f} seconds...")
            time.sleep(sleep_time)
            backoff_delay *= 2  # Exponential increase
            continue

    logger.error(f"Failed to synthesize chunk index {idx} after {max_retries} attempts.")
    return None  # Return None to indicate failure

def update_tts_state(new_state):
    global tts_state
    with tts_state_lock:
        tts_state = new_state

def cancel_current_tts():
    if is_active.is_set():
        cancel_event.set()  # Set the cancel event

        # Wait for threads to acknowledge the cancel event
        time.sleep(0.5)

        # Clear any pending chunks in the TTS queue
        with tts_queue.mutex:
            tts_queue.queue.clear()

        # Clear any pending audio in the playback queue
        with audio_queue.mutex:
            audio_queue.queue.clear()

        update_tts_state("IDLE")
        is_active.clear()  # Reset active state
        logger.info("Canceled current TTS synthesis and playback.")
        # Do not clear cancel_event here; let threads manage it appropriately

def mute_microphone():
    global is_muted
    with is_muted_lock:
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
    with is_muted_lock:
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

def playback_thread():
    global stream
    while True:
        try:
            audio_data = audio_queue.get()
            if audio_data is None:
                break  # Exit signal received
            with stream_lock:
                if stream is not None:
                    stream.write(audio_data)
            audio_queue.task_done()
        except Exception as e:
            logger.error(f"Error during playback: {e}")
            break

def start_tts_client():
    global stream
    # Adjusted keepalive settings
    keepalive_options = [
        ('grpc.keepalive_time_ms', 7200000),  # Send keepalive ping every 2 hours
        ('grpc.keepalive_timeout_ms', 20000),  # Timeout for keepalive ping
        ('grpc.keepalive_permit_without_calls', False),
        ('grpc.http2.max_pings_without_data', 0),
        ('grpc.http2.min_time_between_pings_ms', 7200000),  # Minimum time between pings without data
        ('grpc.http2.max_ping_strikes', 0),
    ]

    channel = None
    tts_stub = None
    retry_delay = 5  # seconds

    while not cancel_event.is_set():
        try:
            if config.get('use_ssl') and config.get('ssl_cert'):
                with open(config['ssl_cert'], 'rb') as f:
                    creds = grpc.ssl_channel_credentials(f.read())
                channel = grpc.secure_channel(config['server'], creds, options=keepalive_options)
            else:
                channel = grpc.insecure_channel(config['server'], options=keepalive_options)

            tts_stub = riva_tts_pb2_grpc.RivaSpeechSynthesisStub(channel)
            logger.info("Successfully connected to Riva TTS server.")
            break  # Exit loop if connection is successful
        except Exception as e:
            logger.error(f"Failed to connect to Riva TTS server: {e}")
            logger.info(f"Retrying to connect in {retry_delay} seconds...")
            time.sleep(retry_delay)

    try:
        # Fetch and set voice with retry logic
        fetch_and_set_voice(tts_stub)

        # Initialize audio stream
        with stream_lock:
            stream = pyaudio_instance.open(
                format=pyaudio.paInt16,
                channels=output_channels,
                rate=sample_rate,
                output=True,
                frames_per_buffer=1024
            )

        # Start playback thread
        playback_thread_instance = threading.Thread(target=playback_thread)
        playback_thread_instance.start()

        # Start synthesis and play thread
        tts_thread = threading.Thread(target=synthesize_and_play, args=(tts_stub,))
        tts_thread.start()
        tts_thread.join()

        # Signal playback thread to exit
        audio_queue.put(None)
        playback_thread_instance.join()
    except Exception as e:
        logger.error(f"Failed during TTS client operation: {e}")
    finally:
        with stream_lock:
            if stream is not None:
                stream.stop_stream()
                stream.close()
                stream = None
            pyaudio_instance.terminate()
        if channel:
            channel.close()

# Main function to initialize and run the TTS engine
if __name__ == '__main__':
    read_config()  # Load the configuration from file
    if register_with_orchestrator():  # Register with the orchestrator if available
        listener_thread = threading.Thread(target=start_data_listener)
        listener_thread.start()  # Start data listener to handle incoming messages
        start_tts_client()  # Start the TTS client for processing and playback
        listener_thread.join()  # Wait for listener thread to finish before exiting
    else:
        logger.error("Could not complete registration. Exiting.")  # Exit if registration fails
