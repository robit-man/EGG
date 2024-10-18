import threading
import time
import socket
import json
import uuid
import os
import queue
import argparse
import grpc
import numpy as np
import sounddevice as sd
import riva.client
from copy import deepcopy
from riva.client.argparse_utils import add_connection_argparse_parameters

# Global Variables
CONFIG_FILE = 'asr.cf'
script_uuid = None  # Initialize as None for UUID persistence
script_name = 'ASR_Engine'  # Peripheral name
config = {}

# Default configuration
default_config = {
    'port': '6200',                # Starting port to listen on
    'port_range': '6200-6300',     # Range of ports to try if initial port is unavailable
    'orchestrator_host': 'localhost',  # Orchestrator's host
    'orchestrator_port': '6000',       # Orchestrator's command port
    'language_code': 'en-US',
    'use_ssl': 'False',
    'ssl_cert': '',
    'server': 'localhost:50051',
    'input_device': '',
    'sample_rate': '16000',
    'chunk_size': '1600',          # Number of samples per chunk (e.g., 100ms chunks at 16kHz)
    'script_uuid': '',             # Initialize as empty; will be set in read_config()
}

# Threading and queues
cancel_event = threading.Event()
asr_queue = queue.Queue()

# Locks for configuration
config_lock = threading.Lock()

def read_config():
    global config, script_uuid
    config = default_config.copy()
    
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                config.update(json.load(f))  # Load the JSON file and update config
        except json.JSONDecodeError:
            print(f"[Error] Could not parse {CONFIG_FILE}. Using default configuration.")
    else:
        # If CONFIG_FILE does not exist, generate a new UUID and create the config file
        script_uuid = str(uuid.uuid4())
        config['script_uuid'] = script_uuid  # Add UUID to config
        write_config()  # Write the initial config with the new UUID
        print(f"[Info] Generated new UUID: {script_uuid} and created {CONFIG_FILE}")
    
    # Ensure UUID is handled
    if 'script_uuid' in config and config['script_uuid']:
        script_uuid = config['script_uuid']
    else:
        # Generate and update UUID if missing
        script_uuid = str(uuid.uuid4())
        config['script_uuid'] = script_uuid
        write_config()
        print(f"[Info] Generated new UUID: {script_uuid} and updated {CONFIG_FILE}")
    
    # Debug: Print the loaded configuration
    print("[Debug] Configuration Loaded:")
    for k, v in config.items():
        if k == 'system_prompt':
            print(f"{k}={'[REDACTED]'}")  # Hide system_prompt in debug
        else:
            print(f"{k}={v}")
    return config

def write_config():
    with config_lock:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)  # Write the configuration as JSON
        print(f"[Info] Configuration written to {CONFIG_FILE}")
        
def parse_args():
    parser = argparse.ArgumentParser(
        description="ASR Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser = add_connection_argparse_parameters(parser)
    parser.add_argument(
        "--port",
        type=str,
        default=None,
        help="Starting port number to listen on."
    )
    parser.add_argument(
        "--port-range",
        type=str,
        default=None,
        help="Range of ports to try if the initial port is unavailable (e.g., '6200-6300')."
    )
    parser.add_argument(
        "--orchestrator-host",
        type=str,
        default=None,
        help="Orchestrator's host address."
    )
    parser.add_argument(
        "--orchestrator-port",
        type=str,
        default=None,
        help="Orchestrator's command port."
    )
    parser.add_argument(
        "--input-device",
        type=str,
        default=None,
        help="Input audio device to use."
    )
    parser.add_argument(
        "--sample-rate",
        type=str,
        default=None,
        help="Sample rate for audio input."
    )
    parser.add_argument(
        "--chunk-size",
        type=str,
        default=None,
        help="Chunk size (number of samples) for audio input."
    )
    args = parser.parse_args()

    # Update config with args if provided
    for key in config.keys():
        arg_name = key.replace('-', '_')
        arg_value = getattr(args, arg_name, None)
        if arg_value is not None:
            config[key] = str(arg_value)
    write_config()
    return args

def asr_processing():
    # Load configuration
    sample_rate = int(config.get('sample_rate', '16000'))
    chunk_size = int(config.get('chunk_size', '1600'))  # 100ms chunks at 16kHz
    input_device = config.get('input_device', None)
    language_code = config.get('language_code', 'en-US')

    # Initialize Riva ASR service
    use_ssl = config.get('use_ssl', 'False') == 'True'
    server = config.get('server', 'localhost:50051')

    if use_ssl:
        ssl_cert = config.get('ssl_cert', '')
        auth = riva.client.Auth(uri=server, use_ssl=True, ssl_cert=ssl_cert)
    else:
        auth = riva.client.Auth(uri=server, use_ssl=False)

    try:
        asr_service = riva.client.ASRService(auth)
        print("ASR Service initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize ASR Service: {e}")
        return

    # Configure ASR Recognition
    offline_config = riva.client.RecognitionConfig(
        encoding=riva.client.AudioEncoding.LINEAR_PCM,
        sample_rate_hertz=sample_rate,
        max_alternatives=3,
        enable_automatic_punctuation=True,  # Ensure punctuation is automatically added
        verbatim_transcripts=True,          # Ensure transcripts reflect exactly what's said
        language_code=language_code
    )

    # Simplified StreamingRecognitionConfig without manual endpointing
    streaming_config = riva.client.StreamingRecognitionConfig(
        config=deepcopy(offline_config),
        interim_results=False               # Only deliver final results
        # We rely on Riva's default endpointing behavior to handle sentence completion
    )

    # Set up audio input stream
    try:
        input_device_index = None
        if input_device:
            devices = sd.query_devices()
            for idx, device in enumerate(devices):
                if input_device in device['name']:
                    input_device_index = idx
                    break
            if input_device_index is None:
                print(f"Input device '{input_device}' not found.")
                return
        else:
            input_device_index = None  # Use default input device

        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype='int16',
            device=input_device_index,
            blocksize=chunk_size
        ) as stream:
            print("Microphone input stream opened successfully.")
            audio_chunk_iterator = iter(
                lambda: stream.read(chunk_size)[0].tobytes(), b''
            )

            # ASR processing using Riva streaming response generator
            response_generator = asr_service.streaming_response_generator(
                audio_chunks=audio_chunk_iterator,
                streaming_config=streaming_config,
            )

            send_text_to_orchestrator(response_generator)

    except Exception as e:
        print(f"Error during ASR streaming: {e}")



def send_text_to_orchestrator(response_generator):
    host = config.get('orchestrator_host', 'localhost')
    port = int(config.get('orchestrator_port', '6000'))
    for response in response_generator:
        for result in response.results:
            if result.is_final:
                recognized_text = result.alternatives[0].transcript
                print(f"Recognized (final): {recognized_text}")
                # Send recognized text to orchestrator
                message = f"/data {script_uuid} {recognized_text}\n"
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.connect((host, port))
                    s.sendall(message.encode())
                    s.close()
                except Exception as e:
                    print(f"Failed to send text to orchestrator at {host}:{port}: {e}")
                    # Optionally retry or handle the error

def start_server():
    host = '0.0.0.0'
    port_range = config.get('port_range', '6200-6300')
    port_list = parse_port_range(port_range)
    server_socket = None

    for port in port_list:
        try:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.bind((host, port))
            print(f"ASR Engine listening on port {port}...")
            config['port'] = str(port)  # Update the port in the config
            write_config()
            break
        except OSError:
            print(f"Port {port} is unavailable, trying next port...")
            server_socket.close()
            server_socket = None

    if not server_socket:
        print("Failed to bind to any port in the specified range.")
        return

    # Register with orchestrator
    register_with_orchestrator()

    server_socket.listen(5)
    while True:
        client_socket, addr = server_socket.accept()
        threading.Thread(target=handle_client_socket, args=(client_socket,), daemon=True).start()

def parse_port_range(port_range_str):
    ports = []
    for part in port_range_str.split(','):
        if '-' in part:
            start, end = map(int, part.split('-'))
            ports.extend(range(start, end + 1))
        else:
            ports.append(int(part))
    return ports

def register_with_orchestrator():
    host = config.get('orchestrator_host', 'localhost')
    port = int(config.get('orchestrator_port', '6000'))
    message = f"/register {script_name} {script_uuid} {config['port']}\n"
    max_retries = 5
    retry_delay = 1
    for attempt in range(max_retries):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((host, port))
            s.sendall(message.encode())
            s.close()
            print(f"Registered with orchestrator at {host}:{port}")
            return
        except Exception as e:
            print(f"Failed to register with orchestrator: {e}")
            time.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff
    print("Max retries reached. Could not register with orchestrator.")

def handle_client_socket(client_socket):
    with client_socket:
        client_socket.settimeout(5)  # Increase timeout to 5 seconds
        buffer = ''
        while True:
            try:
                data = client_socket.recv(1024)
                if not data:
                    break
                buffer += data.decode()
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    command = line.strip()
                    if command == '/info':
                        response = f"{script_name}\n{script_uuid}\n"
                        with config_lock:
                            for key, value in config.items():
                                response += f"{key}={value}\n"
                        response += 'EOF\n'
                        client_socket.sendall(response.encode())
                        client_socket.shutdown(socket.SHUT_WR)  # Ensure data is flushed
                    elif command == '/exit':
                        client_socket.sendall(b"Exiting ASR Engine.\n")
                        cancel_event.set()
                        return
                    else:
                        client_socket.sendall(b"Unknown command.\n")
            except socket.timeout:
                continue
            except Exception as e:
                print(f"Error in handle_client_socket: {e}")
                break

def main():
    read_config()
    parse_args()

    # Start server to handle /info requests and registration
    threading.Thread(target=start_server, daemon=True).start()

    # Start ASR processing
    asr_processing()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("ASR Engine terminated by user.")
