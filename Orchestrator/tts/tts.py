import argparse
import threading
import queue
import pyaudio
import numpy as np
import grpc
import socket
import curses
import time
import json
import textwrap
import re  # Import regular expressions module
import os  # For file operations
from riva.client.proto import riva_tts_pb2, riva_tts_pb2_grpc
import riva.client
import uuid
import logging  # Import logging module
import traceback  # For detailed exception traces

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Initialize UUID and script name
script_uuid = None  # Initialize as None for UUID persistence
script_name = 'TTS_Engine'

# Arguments setup for Riva
def parse_args():
    parser = argparse.ArgumentParser(
        description="Riva TTS Application",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--port", type=int, default=6100, help="Port to listen on. Defaults to 6100."
    )
    parser.add_argument(
        "-o", "--output", type=str, help="Output .wav file to write synthesized audio."
    )
    parser.add_argument(
        "--voice",
        type=str,
        default=None,
        help="A voice name to use for TTS."
    )
    parser.add_argument(
        "--language-code",
        type=str,
        default="en-US",
        help="Language code for TTS. Default is 'en-US'."
    )
    parser.add_argument(
        "--use-ssl",
        action='store_true',
        help="Use SSL/TLS authentication."
    )
    parser.add_argument(
        "--ssl-cert",
        type=str,
        help="Path to SSL/TLS certificate."
    )
    parser.add_argument(
        "--server",
        type=str,
        default="localhost:50051",
        help="Riva server URI and port."
    )
    parser.add_argument(
        "--orchestrator-host", type=str, default="localhost", help="Orchestrator host address."
    )
    parser.add_argument(
        "--orchestrator-ports", type=str, default="6000-6005", help="Comma-separated list or range of orchestrator command ports (e.g., 6000,6001,6002 or 6000-6005)."
    )
    parser.add_argument(
        "--input-mode",
        type=str,
        default="port",
        choices=["terminal", "port", "route"],
        help="Input mode for TTS. 'terminal' for terminal input, 'port' to listen on port, 'route' for routed input."
    )
    return parser.parse_args()

args = parse_args()

# Global Variables
tts_queue = queue.Queue()
cancel_event = threading.Event()
current_tts_thread = None

CONFIG_FILE = 'tts.cf'  # Configuration file name

# Default configuration dictionary
default_config = {
    'input_mode': 'port',   # Options: 'terminal', 'port', 'route'
    'input_format': 'chunk',    # Options: 'streaming', 'chunk' (currently only 'chunk' is implemented)
    'output_mode': 'speaker',   # Options: 'speaker', 'file', 'stream'
    'port': str(args.port),     # Store as string for consistency in config file
    'route': '/tts_route',
    'voice': '',                # Voice name; empty string means not set
    'script_uuid': '',          # UUID for the TTS script
    'orchestrator_host': args.orchestrator_host,
    'orchestrator_ports': args.orchestrator_ports,  # New config key
}

config = {}  # Will be populated by reading config file or using defaults

config_changed = False  # Flag to indicate if config has changed

# Set the sample rate and channels for TTS
sample_rate = 22050  # Riva TTS default sample rate (adjusted to 22050Hz)
output_channels = 1  # Mono audio for TTS output

# Mapping of special characters to their spelled-out versions
SPECIAL_CHAR_MAP = {
    '%': 'percent',
    '&': 'and',
    '+': 'plus',
    '=': 'equals',
    '@': 'at',
}

# Regular expression patterns
SPECIAL_CHAR_PATTERN = re.compile(r'[!\"#$%&\'()*+,\-./:;<=>?@\[\\\]^_`{|}~]')
NUMBER_PATTERN = re.compile(r'\b\d+\b')  # Matches integers in the text

# Function to replace numbers with their spelled-out versions
def number_to_words(n):
    # Handle zero explicitly
    if n == 0:
        return 'zero'

    # Define words for numbers
    ones = ['', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    teens = ['ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen',
             'sixteen', 'seventeen', 'eighteen', 'nineteen']
    tens = ['', '', 'twenty', 'thirty', 'forty', 'fifty',
            'sixty', 'seventy', 'eighty', 'ninety']
    # Expanded list to include very large numbers
    thousands = [
        '', 'thousand', 'million', 'billion', 'trillion', 'quadrillion',
        'quintillion', 'sextillion', 'septillion', 'octillion', 'nonillion',
        'decillion', 'undecillion', 'duodecillion', 'tredecillion',
        'quattuordecillion', 'quindecillion', 'sexdecillion', 'septendecillion',
        'octodecillion', 'novemdecillion', 'vigintillion',
        'unvigintillion', 'duovigintillion', 'trevigintillion', 'quattuorvigintillion',
        'quinvigintillion', 'sexvigintillion', 'septenvigintillion', 'octovigintillion',
        'novemvigintillion', 'trigintillion', 'untrigintillion', 'duotrigintillion',
        'tretrigintillion', 'quattuortrigintillion', 'quintrigintillion',
        'sextrigintillion', 'septentrigintillion', 'octotrigintillion', 'novemtrigintillion',
        'quadragintillion', 'unquadragintillion', 'duoquadragintillion', 'trequadragintillion',
        'quattuorquadragintillion', 'quinquadragintillion', 'sexquadragintillion',
        'septenquadragintillion', 'octoquadragintillion', 'novemquadragintillion',
        'quinquagintillion', 'unquinquagintillion', 'duoquinquagintillion',
        'trequinquagintillion', 'quattuorquinquagintillion', 'quinquinquagintillion',
        'sexquinquagintillion', 'septenquinquagintillion', 'octoquinquagintillion',
        'novemquinquagintillion', 'sexagintillion', 'unsexagintillion',
        'duosexagintillion', 'tresexagintillion', 'quattuorsexagintillion',
        'quinsexagintillion', 'sexsexagintillion', 'septensexagintillion',
        'octosexagintillion', 'novemsexagintillion', 'septuagintillion',
        'unseptuagintillion', 'duoseptuagintillion', 'treseptuagintillion',
        'quattuorseptuagintillion', 'quinseptuagintillion', 'sexseptuagintillion',
        'septenseptuagintillion', 'octoseptuagintillion', 'novemseptuagintillion',
        'octogintillion', 'unoctogintillion', 'duooctogintillion', 'treoctogintillion',
        'quattuoroctogintillion', 'quinoctogintillion', 'sexoctogintillion',
        'septenoctogintillion', 'octooctogintillion', 'novemoctogintillion',
        'nonagintillion', 'unnonagintillion', 'duononagintillion', 'trenonagintillion',
        'quattuornonagintillion', 'quinnonagintillion', 'sexnonagintillion',
        'septennonagintillion', 'octononagintillion', 'novemnonagintillion',
        'centillion'
    ]
    words = []
    num_str = str(n)
    num_length = len(num_str)
    groups = []

    # Split number into groups of three digits from the end
    for i in range(0, num_length, 3):
        start = max(0, num_length - i - 3)
        end = num_length - i
        groups.insert(0, int(num_str[start:end]))

    group_count = len(groups)
    if group_count > len(thousands):
        return 'number too large to convert'

    for idx, group in enumerate(groups):
        if group == 0:
            continue

        group_words = []
        hundreds_digit = group // 100
        tens_digit = (group % 100) // 10
        ones_digit = group % 10

        if hundreds_digit > 0:
            group_words.append(ones[hundreds_digit] + ' hundred')

        remainder = group % 100
        if 10 <= remainder < 20:
            group_words.append(teens[remainder - 10])
        else:
            if tens_digit > 1:
                group_words.append(tens[tens_digit])
                if ones_digit > 0:
                    group_words.append(ones[ones_digit])
            else:
                if ones_digit > 0:
                    group_words.append(ones[ones_digit])

        if group_count - idx - 1 > 0:
            group_words.append(thousands[group_count - idx - 1])

        words.extend(group_words)

    return ' '.join(words)

# Function to replace special characters and numbers with their spelled-out versions
def replace_special_characters_and_numbers(text):
    # Replace special characters except for specific ones where spaces are not wanted
    def special_char_replacer(match):
        char = match.group(0)
        # Only add spaces around certain characters
        if char in SPECIAL_CHAR_MAP:
            return ' ' + SPECIAL_CHAR_MAP.get(char, char) + ' '
        # Keep characters like ' and ? without spaces
        return char

    # Use regular expression to substitute special characters
    text = SPECIAL_CHAR_PATTERN.sub(special_char_replacer, text)

    # Replace numbers
    def number_replacer(match):
        num = int(match.group(0))
        return ' ' + number_to_words(num) + ' '

    text = NUMBER_PATTERN.sub(number_replacer, text)

    return text

# Function to split text into sentences
def split_text_into_sentences(text):
    """Split text into sentences based on common sentence-ending punctuation."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())  # Split on sentence boundaries
    return sentences

# Function to read configuration from a JSON file
def read_config():
    global config, script_uuid
    config = default_config.copy()
    
    if os.path.exists(CONFIG_FILE):
        logger.info(f"Reading configuration from {CONFIG_FILE}")
        try:
            # Read the JSON file
            with open(CONFIG_FILE, 'r') as f:
                config.update(json.load(f))  # Update default config with the values from the file
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.error(f"Could not read {CONFIG_FILE}: {e}. Using default configuration.")
    else:
        logger.info(f"No configuration file found. Creating default {CONFIG_FILE}")
        script_uuid = str(uuid.uuid4())
        config['script_uuid'] = script_uuid
        write_config(config)
        logger.info(f"[Info] Generated new UUID: {script_uuid} and created {CONFIG_FILE}")

    if 'script_uuid' in config and config['script_uuid']:
        script_uuid = config['script_uuid']
    else:
        script_uuid = str(uuid.uuid4())
        config['script_uuid'] = script_uuid
        config['voice'] = 'GLaDOS'  # Just to set an example voice
        write_config(config)
        logger.info(f"[Info] Generated new UUID: {script_uuid} and updated {CONFIG_FILE}")
    
    logger.debug("Configuration Loaded:")
    for k, v in config.items():
        logger.debug(f"{k}={v}")
    
    return config


# Function to write configuration to a JSON file
def write_config(config_data):
    logger.info(f"Writing configuration to {CONFIG_FILE}")
    with config_lock:
        # Save the config in proper JSON format
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config_data, f, indent=4)


# Lock for thread-safe operations
config_lock = threading.Lock()

# Function to clear the tts_queue
def clear_queue(q):
    """Clears all items from the given queue."""
    with q.mutex:
        q.queue.clear()
    logger.info("TTS queue has been cleared.")

# Function to split text into manageable chunks
def split_text_into_chunks(text, max_chunk_size=200):
    # Use textwrap to split text at word boundaries
    return textwrap.wrap(text, width=max_chunk_size, break_long_words=False)

# Get Available TTS Voices
def get_available_voices(tts_stub):
    logger.info("Retrieving available voices from Riva TTS...")

    # Create the request to get synthesis configuration
    request = riva_tts_pb2.RivaSynthesisConfigRequest()

    # Call the GetRivaSynthesisConfig RPC
    try:
        response = tts_stub.GetRivaSynthesisConfig(request)
    except Exception as e:
        logger.error(f"Failed to get TTS synthesis config: {e}")
        return []

    # Parse the response to find available voices
    available_voices = []
    if response.model_config:
        logger.info("Available TTS Models:")
        for model in response.model_config:
            voice_name = model.parameters.get("voice_name", "")
            language_code = model.parameters.get("language_code", "")

            if voice_name:
                available_voices.append({"name": voice_name, "language": language_code})
                logger.info(f"Voice: {voice_name}, Language: {language_code}")

    if not available_voices:
        logger.warning("No available voices found in the response.")

    return available_voices

# Configuration Menu using curses
def run_config_menu(config, available_voices):
    import curses

    # Initialize curses
    stdscr = curses.initscr()
    curses.noecho()
    curses.cbreak()
    stdscr.keypad(True)
    curses.start_color()
    curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_WHITE)
    try:
        # Define menu items and their options
        menu = [
            {'label': 'Input Mode', 'options': ['terminal', 'port', 'route'], 'current': config['input_mode']},
            {'label': 'Input Format', 'options': ['streaming', 'chunk'], 'current': config['input_format']},
            {'label': 'Output Mode', 'options': ['speaker', 'file', 'stream'], 'current': config['output_mode']},
            {'label': 'Voice', 'options': [v['name'] for v in available_voices], 'current': config.get('voice', '')},
        ]

        current_row = 0

        while True:
            stdscr.clear()
            # Display menu
            for idx, item in enumerate(menu):
                if idx == current_row:
                    stdscr.attron(curses.color_pair(1))
                    stdscr.addstr(idx, 0, f"{item['label']}: {item['current']}")
                    stdscr.attroff(curses.color_pair(1))
                else:
                    stdscr.addstr(idx, 0, f"{item['label']}: {item['current']}")
            stdscr.addstr(len(menu) + 1, 0, "Use arrow keys to navigate, Enter to save, Esc to cancel.")
            stdscr.refresh()

            key = stdscr.getch()
            if key == curses.KEY_UP and current_row > 0:
                current_row -= 1
            elif key == curses.KEY_DOWN and current_row < len(menu) - 1:
                current_row += 1
            elif key == curses.KEY_LEFT:
                # Change option to previous
                try:
                    current_option_idx = menu[current_row]['options'].index(menu[current_row]['current'])
                except ValueError:
                    current_option_idx = 0
                if current_option_idx > 0:
                    current_option_idx -= 1
                menu[current_row]['current'] = menu[current_row]['options'][current_option_idx]
            elif key == curses.KEY_RIGHT:
                # Change option to next
                try:
                    current_option_idx = menu[current_row]['options'].index(menu[current_row]['current'])
                except ValueError:
                    current_option_idx = -1
                if current_option_idx < len(menu[current_row]['options']) - 1:
                    current_option_idx += 1
                menu[current_row]['current'] = menu[current_row]['options'][current_option_idx]
            elif key == ord('\n'):
                # Enter key pressed
                # Update config with current selections and exit menu
                config['input_mode'] = menu[0]['current']
                config['input_format'] = menu[1]['current']
                config['output_mode'] = menu[2]['current']
                config['voice'] = menu[3]['current']
                break
            elif key == 27:  # Escape key
                # Exit without saving
                break
    finally:
        # Clean up curses
        stdscr.keypad(False)
        curses.echo()
        curses.nocbreak()
        curses.endwin()

# Function to register with the orchestrator
def register_with_orchestrator():
    orchestrator_host = config.get('orchestrator_host', 'localhost')
    orchestrator_ports_str = config.get('orchestrator_ports', '6000-6005')
    orchestrator_command_ports = []

    # Parse orchestrator_ports which can be a range like '6000-6005' or a list '6000,6001,6002'
    for port_entry in orchestrator_ports_str.split(','):
        port_entry = port_entry.strip()
        if '-' in port_entry:
            try:
                start_port, end_port = map(int, port_entry.split('-'))
                orchestrator_command_ports.extend(range(start_port, end_port + 1))
            except ValueError:
                logger.warning(f"Invalid orchestrator port range: {port_entry}")
        elif port_entry.isdigit():
            orchestrator_command_ports.append(int(port_entry))
        else:
            logger.warning(f"Invalid orchestrator port entry: {port_entry}")

    if not orchestrator_command_ports:
        logger.error("No valid orchestrator command ports found.")
        return False

    message = f"/register {script_name} {script_uuid} {config['port']}\n"
    max_retries = 5
    retry_delay = 1
    for attempt in range(max_retries):
        for orch_port in orchestrator_command_ports:
            try:
                logger.info(f"[Attempt {attempt + 1}] Registering with orchestrator at {orchestrator_host}:{orch_port}...")
                with socket.create_connection((orchestrator_host, orch_port), timeout=5) as s:
                    s.sendall(message.encode())
                    logger.info(f"Sent registration message: {message.strip()}")

                    # Receive acknowledgment
                    data = s.recv(1024)
                    if data:
                        ack_message = data.decode().strip()
                        logger.info(f"Received acknowledgment: {ack_message}")
                        if isinstance(ack_message, str) and ack_message.startswith('/ack'):
                            tokens = ack_message.split()
                            if len(tokens) == 2 and tokens[0] == '/ack':
                                try:
                                    data_port = tokens[1]  # Keep as string
                                    config['data_port'] = data_port
                                    write_config(config)
                                    logger.info(f"Registered successfully. Data port: {data_port}")
                                    return True
                                except ValueError:
                                    logger.error(f"Invalid data port received in acknowledgment: {tokens[1]}")
                            else:
                                logger.error(f"Unexpected acknowledgment format: {ack_message}")
                        else:
                            logger.error(f"Invalid acknowledgment type: {type(ack_message)}")
                    else:
                        logger.warning(f"No acknowledgment received from orchestrator at {orchestrator_host}:{orch_port}.")
            except socket.timeout:
                logger.error(f"Connection to orchestrator at {orchestrator_host}:{orch_port} timed out.")
            except ConnectionRefusedError:
                logger.error(f"Connection refused by orchestrator at {orchestrator_host}:{orch_port}.")
            except Exception as e:
                logger.error(f"Exception during registration with orchestrator at {orchestrator_host}:{orch_port}: {e}")
                logger.debug(traceback.format_exc())
        logger.info(f"Retrying registration in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
        time.sleep(retry_delay)
        retry_delay *= 2  # Exponential backoff

    logger.error("Max retries reached. Could not register with orchestrator.")
    if config['input_mode'] == 'terminal':
        logger.info("Failed to register with the orchestrator after multiple attempts. Entering terminal mode.")
        return False  # Allow fallback to terminal mode
    else:
        logger.error("Failed to register with the orchestrator after multiple attempts. Exiting...")
        cancel_event.set()
        exit(1)  # Exit the script as registration is critical

# Function to perform a single warm-up call
def perform_warmup(tts_stub, selected_voice):
    dummy_req = riva_tts_pb2.SynthesizeSpeechRequest(
        text="Warm up call",
        language_code=selected_voice['language'],
        encoding=riva.client.AudioEncoding.LINEAR_PCM,
        sample_rate_hz=sample_rate,
        voice_name=selected_voice['name']
    )
    try:
        logger.info("Performing warm-up inference...")
        tts_stub.Synthesize(dummy_req, timeout=15)
        logger.info("Warm-up completed successfully.")
    except grpc.RpcError as e:
        logger.warning(f"Warm-up failed: {e.details()}")
    except Exception as e:
        logger.error(f"Unexpected error during warm-up: {e}")
        logger.debug(traceback.format_exc())

# Generate TTS and Playback
def tts_generation_and_playback(tts_stub, selected_voice):
    output_mode = config['output_mode']

    # Initialize audio output stream
    if output_mode == 'speaker':
        try:
            p = pyaudio.PyAudio()
            stream = p.open(
                format=pyaudio.paInt16,
                channels=output_channels,
                rate=sample_rate,
                output=True,
                frames_per_buffer=2048  # Increased buffer size
            )
            logger.info("Audio output stream opened successfully.")
        except Exception as e:
            logger.error(f"Failed to open audio output stream: {e}")
            return
    elif output_mode == 'file':
        try:
            output_file_path = args.output or 'output.wav'
            output_file = open(output_file_path, 'wb')
            logger.info(f"Opened output file: {output_file_path}")
        except Exception as e:
            logger.error(f"Failed to open output file: {e}")
            return
    elif output_mode == 'stream':
        # Implement stream output if needed
        logger.info("Stream output mode is not yet implemented.")
        pass
    else:
        logger.error("Invalid output mode.")
        return

    # Process TTS requests from the queue
    while not cancel_event.is_set():
        try:
            text = tts_queue.get(timeout=1)
        except queue.Empty:
            continue

        if text:
            logger.info(f"Original text: {text}")
            # Replace special characters and numbers
            processed_text = replace_special_characters_and_numbers(text)
            logger.info(f"Processed text: {processed_text}")

            # Split the text into sentences before processing to avoid overloading the model
            sentences = split_text_into_sentences(processed_text)

            for sentence in sentences:
                # Further split sentences if they exceed 400 characters
                if len(sentence) > 400:
                    text_chunks = split_text_into_chunks(sentence, max_chunk_size=200)
                else:
                    text_chunks = [sentence]

                for chunk in text_chunks:
                    req = riva_tts_pb2.SynthesizeSpeechRequest(
                        text=chunk,
                        language_code=selected_voice['language'],
                        encoding=riva.client.AudioEncoding.LINEAR_PCM,
                        sample_rate_hz=sample_rate,
                        voice_name=selected_voice['name']
                    )

                    retries = 3
                    for attempt in range(retries):
                        try:
                            # Send the TTS request
                            resp = tts_stub.Synthesize(req, timeout=15)
                            audio_samples = np.frombuffer(resp.audio, dtype=np.int16)

                            if output_mode == 'speaker':
                                stream.write(audio_samples.tobytes())
                            elif output_mode == 'file':
                                output_file.write(resp.audio)
                            elif output_mode == 'stream':
                                # Implement streaming output
                                logger.info("Streaming output is not yet implemented.")
                                pass

                            # Small sleep to prevent buffer overruns
                            time.sleep(0.05)
                            break  # Exit retry loop if successful
                        except grpc.RpcError as e:
                            if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                                logger.warning(f"Attempt {attempt + 1}/{retries}: TTS deadline exceeded. Retrying...")
                                time.sleep(1)  # Small delay before retrying
                            else:
                                logger.error(f"TTS generation failed: {e.details()}")
                                break  # Exit if it's not a deadline error
                        except Exception as e:
                            logger.error(f"An unexpected error occurred during TTS generation: {e}")
                            logger.debug(traceback.format_exc())
                            break  # Exit retry loop for unexpected errors

    # Cleanup resources upon cancellation
    if output_mode == 'speaker':
        stream.stop_stream()
        stream.close()
        p.terminate()
        logger.info("Audio output stream closed.")
    elif output_mode == 'file':
        output_file.close()
        logger.info("Output file closed.")

# Start TTS Thread
def start_tts_thread(tts_stub, selected_voice):
    global current_tts_thread
    if current_tts_thread and current_tts_thread.is_alive():
        logger.info("TTS thread is already running. Attempting to terminate existing thread...")
        cancel_event.set()
        current_tts_thread.join()
        logger.info("Existing TTS thread terminated.")

    cancel_event.clear()
    current_tts_thread = threading.Thread(target=tts_generation_and_playback, args=(tts_stub, selected_voice), daemon=True)
    current_tts_thread.start()
    logger.info("TTS thread started successfully.")

# Handle incoming text via port
def handle_port_client(client_socket, addr):
    with client_socket:
        try:
            data = client_socket.recv(1024).decode().strip()
            if data == '/info':
                response = f"{script_name}\n{script_uuid}\n"
                # Include config
                for key, value in config.items():
                    response += f"{key}={value}\n"
                response += 'EOF\n'
                client_socket.sendall(response.encode())
                logger.info(f"Sent configuration info to {addr}")
            elif data == '/config':
                client_socket.send("Entering configuration mode not supported over port.\n".encode())
            elif data == '/cancel':
                clear_queue(tts_queue)
                client_socket.sendall(b"TTS queue has been cleared.\n")
                logger.info(f"Received /cancel command from {addr}. Cleared TTS queue.")
            elif data == '/exit':
                client_socket.sendall(b"Exiting TTS Engine.\n")
                logger.info(f"Received /exit command from {addr}. Shutting down.")
                cancel_event.set()
            else:
                logger.info(f"Received from {addr}: {data}")
                tts_queue.put(data)
        except Exception as e:
            logger.error(f"Error handling client {addr}: {e}")

# Port Input Mode
def port_input():
    host = '0.0.0.0'
    port = int(config['port'])  # Convert port to integer
    logger.info(f"Listening for text input on port {port}...")
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        server_socket.bind((host, port))
        server_socket.listen(5)
        logger.info(f"Server started on port {port}. Waiting for connections...")
    except Exception as e:
        logger.error(f"Failed to bind to port {port}: {e}")
        if config['input_mode'] == 'terminal':
            logger.info("Switching to terminal mode due to port binding failure.")
            config['input_mode'] = 'terminal'
            write_config(config)
            terminal_input()
            return
        else:
            cancel_event.set()
            exit(1)

    while not cancel_event.is_set():
        try:
            client_socket, addr = server_socket.accept()
            threading.Thread(target=handle_port_client, args=(client_socket, addr), daemon=True).start()
        except Exception as e:
            if not cancel_event.is_set():
                logger.error(f"Error accepting connection: {e}")

    server_socket.close()
    logger.info("Port input server shut down.")

# Route Input Mode
def route_input():
    from http.server import BaseHTTPRequestHandler, HTTPServer
    import threading
    host = '0.0.0.0'
    port = int(config['port'])  # Convert port to integer
    route = config['route']

    class RequestHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path == route:
                content_length = int(self.headers.get('Content-Length', 0))
                post_data = self.rfile.read(content_length).decode().strip()
                if post_data == '/info':
                    response = f"{script_name}\n{script_uuid}\n"
                    # Include config
                    for key, value in config.items():
                        response += f"{key}={value}\n"
                    response += 'EOF\n'
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/plain')
                    self.end_headers()
                    self.wfile.write(response.encode())
                elif post_data == '/cancel':
                    clear_queue(tts_queue)
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b"TTS queue has been cleared.\n")
                    logger.info("Received /cancel command via HTTP POST. Cleared TTS queue.")
                elif post_data == '/exit':
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b"Exiting TTS Engine.\n")
                    logger.info("Received /exit command via HTTP POST. Shutting down.")
                    cancel_event.set()
                else:
                    logger.info(f"Received text via HTTP POST: {post_data}")
                    tts_queue.put(post_data)
                    self.send_response(200)
                    self.end_headers()
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format, *args):
            return  # Suppress default logging

    server_address = (host, port)
    try:
        httpd = HTTPServer(server_address, RequestHandler)
        logger.info(f"HTTP server started on port {port}, route {route}")
    except Exception as e:
        logger.error(f"Failed to start HTTP server on port {port}: {e}")
        if config['input_mode'] == 'terminal':
            logger.info("Switching to terminal mode due to HTTP server failure.")
            config['input_mode'] = 'terminal'
            write_config(config)
            terminal_input()
            return
        else:
            cancel_event.set()
            exit(1)
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()

    try:
        while not cancel_event.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        httpd.shutdown()
        httpd.server_close()
        server_thread.join()
        logger.info("HTTP input server shut down.")

# Terminal Input Mode
def terminal_input():
    global config_changed
    print("Enter text to synthesize (type /config to enter configuration mode, /cancel to clear the queue, /exit to quit):")
    while True:
        try:
            user_input = input("> ")
        except EOFError:
            # Handle Ctrl+D
            user_input = '/exit'
        if user_input.strip() == '/config':
            run_config_menu(config, available_voices)
            print("Configuration updated.")
            logger.info("Configuration updated via terminal.")
            print("Current config:")
            for key, value in config.items():
                print(f"{key}={value}")
            config_changed = True
            # Write updated configuration to file
            write_config(config)
            # Restart TTS thread if necessary
            if current_tts_thread and current_tts_thread.is_alive():
                logger.info("Restarting TTS thread due to configuration change.")
                start_tts_thread(tts_stub, selected_voice)
        elif user_input.strip() == '/cancel':
            clear_queue(tts_queue)
            print("TTS queue has been cleared.")
            logger.info("Cleared TTS queue via terminal command.")
        elif user_input.strip() == '/exit':
            cancel_event.set()
            logger.info("Received /exit command via terminal. Shutting down.")
            break
        else:
            tts_queue.put(user_input.strip())
            logger.info(f"Queued text for TTS: {user_input.strip()}")

# Input Handler
def input_handler():
    if config['input_mode'] == 'terminal':
        terminal_input()
    elif config['input_mode'] == 'port':
        port_input()
    elif config['input_mode'] == 'route':
        route_input()
    else:
        logger.error("Invalid input mode.")
        return

def main():
    global config_changed
    global available_voices  # To access in terminal_input
    global tts_stub
    global selected_voice

    # Read configuration from file
    read_config()

    # Register with the orchestrator
    registration_successful = register_with_orchestrator()

    if not registration_successful and config['input_mode'] == 'terminal':
        # Proceed to terminal input mode
        pass
    else:
        # Initialize TTS stub
        try:
            if args.use_ssl and args.ssl_cert:
                with open(args.ssl_cert, 'rb') as f:
                    creds = grpc.ssl_channel_credentials(f.read())
                channel = grpc.secure_channel(args.server, creds)
            else:
                channel = grpc.insecure_channel(args.server)

            tts_stub = riva_tts_pb2_grpc.RivaSpeechSynthesisStub(channel)
            logger.info("TTS stub initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize TTS stub: {e}")
            logger.debug(traceback.format_exc())
            if config['input_mode'] == 'terminal':
                logger.info("Entering terminal mode due to TTS stub initialization failure.")
                config['input_mode'] = 'terminal'
                write_config(config)
                terminal_input()
                return
            else:
                cancel_event.set()
                exit(1)

        # Get available voices
        available_voices = get_available_voices(tts_stub)
        if not available_voices:
            logger.error("No available voices.")
            if config['input_mode'] == 'terminal':
                logger.info("Entering terminal mode due to no available voices.")
                config['input_mode'] = 'terminal'
                write_config(config)
                terminal_input()
                return
            else:
                cancel_event.set()
                exit(1)

        # Check if voice is specified in config
        selected_voice = None
        if config.get('voice'):
            # Check if the voice exists in the available voices
            matching_voices = [v for v in available_voices if v['name'] == config['voice']]
            if matching_voices:
                selected_voice = matching_voices[0]
                logger.info(f"Using voice from config: {selected_voice['name']} (Language: {selected_voice['language']})")
            else:
                logger.warning(f"Voice '{config['voice']}' specified in config not found.")
                config['voice'] = ''  # Reset the voice in config

        # If voice is not specified or not found, prompt user to select
        if not selected_voice:
            logger.info("\nAvailable voices:")
            for idx, voice in enumerate(available_voices):
                logger.info(f"{idx + 1}: {voice['name']} (Language: {voice['language']})")

            while not selected_voice:
                user_input = input("Select a voice by number or name: ").strip()
                # Try to interpret input as a number
                try:
                    idx = int(user_input) - 1
                    if 0 <= idx < len(available_voices):
                        selected_voice = available_voices[idx]
                    else:
                        logger.warning("Invalid selection. Please try again.")
                except ValueError:
                    # Not a number, treat as name
                    matching_voices = [v for v in available_voices if v['name'].lower() == user_input.lower()]
                    if matching_voices:
                        selected_voice = matching_voices[0]
                    else:
                        logger.warning("Voice not found. Please try again.")

            logger.info(f"Selected Voice: {selected_voice['name']} (Language: {selected_voice['language']})")
            # Update config with selected voice and write to file
            config['voice'] = selected_voice['name']
            write_config(config)

        # Perform warm-up once
        perform_warmup(tts_stub, selected_voice)

        # Start TTS thread with selected voice
        start_tts_thread(tts_stub, selected_voice)

    # Start input handler in a separate thread
    input_thread = threading.Thread(target=input_handler, daemon=True)
    input_thread.start()

    # Keep the main thread alive until shutdown is signaled
    try:
        while not cancel_event.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Program terminated by user.")
        cancel_event.set()
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        logger.debug(traceback.format_exc())
        cancel_event.set()

    # Wait for threads to finish
    if current_tts_thread and current_tts_thread.is_alive():
        current_tts_thread.join()

    logger.info("Program terminated.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Clean up curses if needed
        try:
            curses.endwin()
        except Exception:
            pass
        logger.info("Program terminated by user.")
    except Exception as e:
        # Handle any other exceptions
        logger.error(f"An unexpected error occurred: {e}")
        logger.debug(traceback.format_exc())
        try:
            curses.endwin()
        except Exception:
            pass
