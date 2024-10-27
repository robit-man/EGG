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

# Configuration defaults
config = {
    'orchestrator_host': 'localhost',
    'orchestrator_ports': '6000-6010',
    'port_range': '6200-6300',
    'data_port': None,
    'output_mode': 'speaker',
    'server': 'localhost:50051',
    'voice': GLaDOS,  # Voice will be selected upon startup
    'language_code': 'en-US'  # Default language code
}

# Logger setup
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

# Regex patterns
SPECIAL_CHAR_PATTERN = re.compile(r'[!\"#$%&\'()*+,\-./:;<=>?@\[\\\]^_`{|}~]')
NUMBER_PATTERN = re.compile(r'\b\d+\b')

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
    text = SPECIAL_CHAR_PATTERN.sub(lambda m: ' ' + m.group(0) + ' ', text)
    text = NUMBER_PATTERN.sub(lambda m: ' ' + num2words(int(m.group(0))) + ' ', text)
    return text.strip()

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

# TTS Synthesis and Playback
def synthesize_and_play(tts_stub):
    while not cancel_event.is_set():
        try:
            text = tts_queue.get(timeout=1)
            chunks = textwrap.wrap(text, width=400)
            mute_microphone()  # Mute the microphone during playback

            for chunk in chunks:
                req = riva_tts_pb2.SynthesizeSpeechRequest(
                    text=chunk,
                    language_code=config['language_code'],
                    encoding=riva.client.AudioEncoding.LINEAR_PCM,
                    sample_rate_hz=sample_rate,
                    voice_name=selected_voice[0]
                )
                resp = tts_stub.Synthesize(req)
                audio_data = np.frombuffer(resp.audio, dtype=np.int16)
                stream.write(audio_data.tobytes())
                logger.info(f"Played synthesized speech for text: '{chunk}'")
            unmute_microphone()  # Mute the microphone during playback
        except queue.Empty:
            continue
            unmute_microphone()  # Mute the microphone during playback
        except grpc.RpcError as e:
            logger.error(f"TTS synthesis failed: {e.details()}")
            unmute_microphone()  # Mute the microphone during playback
        except Exception as e:
            logger.error(f"Unexpected error in TTS synthesis and playback: {e}")
            unmute_microphone()  # Mute the microphone during playback

# Function to mute the microphone
def mute_microphone():
    try:
        # Use pactl to get the default input source
        default_source = subprocess.check_output(
            ["pactl", "get-default-source"], universal_newlines=True
        ).strip()

        # Mute the default input source
        subprocess.run(["pactl", "set-source-mute", default_source, "1"], check=True)
        logger.info(f"Microphone {default_source} muted.")
    except Exception as e:
        logger.error(f"Failed to mute the microphone: {e}")

# Function to unmute the microphone
def unmute_microphone():
    try:
        # Use pactl to get the default input source
        default_source = subprocess.check_output(
            ["pactl", "get-default-source"], universal_newlines=True
        ).strip()

        # Unmute the default input source
        subprocess.run(["pactl", "set-source-mute", default_source, "0"], check=True)
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
