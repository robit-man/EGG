import argparse
import threading
import queue
import pyaudio
import numpy as np
import grpc
import socket
import curses
import time
import textwrap
import re  # Import regular expressions module
import os  # For file operations
from riva.client.proto import riva_tts_pb2, riva_tts_pb2_grpc
import riva.client
import uuid

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
}

config = {}  # Will be populated by reading config file or using defaults

config_changed = False  # Flag to indicate if config has changed

# Set the sample rate and channels for TTS
sample_rate = 22050  # Riva TTS default sample rate (adjusted to 22050Hz)
output_channels = 1  # Mono audio for TTS output

# Mapping of special characters to their spelled-out versions
SPECIAL_CHAR_MAP = {
#    '!': 'exclamation mark',
#    '"': 'double quote',
#    '#': 'hash',
#    '$': 'dollar',
    '%': 'percent',
    '&': 'and',
#    "'": 'apostrophe',
#    '(': 'left parenthesis',
#    ')': 'right parenthesis',
#    '*': 'asterisk',
    '+': 'plus',
#    ',': 'comma',
#    '-': 'dash',
#    '.': 'dot',
#    '/': 'slash',
#    ':': 'colon',
#    ';': 'semicolon',
#    '<': 'less than',
    '=': 'equals',
#    '>': 'greater than',
#    '?': 'question mark',
    '@': 'at',
#    '[': 'left bracket',
#    '\\': 'backslash',
#    ']': 'right bracket',
#    '^': 'caret',
#    '_': 'underscore',
#    '`': 'backtick',
#    '{': 'left brace',
#    '|': 'pipe',
#    '}': 'right brace',
#    '~': 'tilde',
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
    ones = [
        '', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight',
        'nine'
    ]
    teens = [
        'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen',
        'sixteen', 'seventeen', 'eighteen', 'nineteen'
    ]
    tens = [
        '', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy',
        'eighty', 'ninety'
    ]
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
        groups.insert(0, int(num_str[max(0, num_length - i - 3):num_length - i]))

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
        if remainder >= 10 and remainder < 20:
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


# Function to read configuration from file
def read_config():
    global config, script_uuid
    config = default_config.copy()
    if os.path.exists(CONFIG_FILE):
        print(f"Reading configuration from {CONFIG_FILE}")
        with open(CONFIG_FILE, 'r') as f:
            for line in f:
                line = line.strip()
                if '=' in line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")  # Remove potential quotes
                    if key in config:
                        config[key] = value
                    else:
                        print(f"Unknown configuration key: {key}")
    else:
        print(f"No configuration file found. Creating default {CONFIG_FILE}")
        # Generate UUID and set it
        script_uuid = str(uuid.uuid4())
        config['script_uuid'] = script_uuid
        write_config(config)
        print(f"[Info] Generated new UUID: {script_uuid} and created {CONFIG_FILE}")

    # After reading config, check for 'script_uuid'
    if 'script_uuid' in config and config['script_uuid']:
        script_uuid = config['script_uuid']
    else:
        script_uuid = str(uuid.uuid4())
        config['script_uuid'] = script_uuid
        write_config(config)
        print(f"[Info] Generated new UUID: {script_uuid} and updated {CONFIG_FILE}")

    # Debug: Print configuration after reading
    print("[Debug] Configuration Loaded:")
    for k, v in config.items():
        if k == 'script_uuid':
            print(f"{k}={v}")  # Display script_uuid
        else:
            print(f"{k}={v}")
    return config

# Function to write configuration to file
def write_config(config):
    print(f"Writing configuration to {CONFIG_FILE}")
    with config_lock:
        with open(CONFIG_FILE, 'w') as f:
            for key, value in config.items():
                value = str(value)  # Ensure all values are strings
                if any(c in value for c in ' \n"\\'):
                    # If value contains special characters, enclose it in quotes and escape
                    escaped_value = value.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
                    f.write(f'{key}="{escaped_value}"\n')
                else:
                    f.write(f"{key}={value}\n")

# Lock for thread-safe operations
config_lock = threading.Lock()

# Generate TTS and Playback
def tts_generation_and_playback(tts_stub, selected_voice):
    output_mode = config['output_mode']

    if output_mode == 'speaker':
        # Set up audio output stream with increased buffer size to prevent underruns
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=output_channels,
            rate=sample_rate,
            output=True,
            frames_per_buffer=2048  # Increased buffer size
        )
    elif output_mode == 'file':
        # Open file for writing
        output_file = open(args.output or 'output.wav', 'wb')
    elif output_mode == 'stream':
        # Implement stream output if needed
        pass
    else:
        print("Invalid output mode.")
        return

    while not cancel_event.is_set():
        try:
            text = tts_queue.get(timeout=1)
        except queue.Empty:
            continue

        if text:
            print(f"Original text: {text}")
            # Replace special characters and numbers
            processed_text = replace_special_characters_and_numbers(text)
            print(f"Processed text: {processed_text}")

            # Split long text into smaller chunks to prevent server timeouts
            text_chunks = split_text_into_chunks(processed_text, max_chunk_size=200)  # Adjust chunk size as needed

            for chunk in text_chunks:
                req = riva_tts_pb2.SynthesizeSpeechRequest(
                    text=chunk,
                    language_code=selected_voice['language'],
                    encoding=riva.client.AudioEncoding.LINEAR_PCM,
                    sample_rate_hz=sample_rate,
                    voice_name=selected_voice['name']
                )

                # Retry mechanism
                retries = 3
                for attempt in range(retries):
                    try:
                        # Increased timeout to 15 seconds to avoid premature deadline exceeded errors
                        resp = tts_stub.Synthesize(req, timeout=15)  # Adjust the timeout as needed
                        audio_samples = np.frombuffer(resp.audio, dtype=np.int16)

                        if output_mode == 'speaker':
                            stream.write(audio_samples.tobytes())
                        elif output_mode == 'file':
                            output_file.write(resp.audio)
                        elif output_mode == 'stream':
                            # Implement streaming output
                            pass

                        # Small sleep to prevent buffer overruns
                        time.sleep(0.05)
                        break  # Exit retry loop if successful
                    except grpc.RpcError as e:
                        if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                            print(f"Attempt {attempt + 1}/{retries}: TTS deadline exceeded. Retrying...")
                            time.sleep(1)  # Small delay before retrying
                        else:
                            print(f"TTS generation failed: {e.details()}")
                            break  # Exit if it's not a deadline error
                    except Exception as e:
                        print(f"An unexpected error occurred: {e}")
                        break  # Exit retry loop for unexpected errors

    if output_mode == 'speaker':
        stream.stop_stream()
        stream.close()
        p.terminate()
    elif output_mode == 'file':
        output_file.close()


# Function to split text into manageable chunks
def split_text_into_chunks(text, max_chunk_size=200):
    # Use textwrap to split text at word boundaries
    return textwrap.wrap(text, width=max_chunk_size, break_long_words=False)

# Start TTS Thread
def start_tts_thread(tts_stub, selected_voice):
    global current_tts_thread
    if current_tts_thread and current_tts_thread.is_alive():
        cancel_event.set()
        current_tts_thread.join()

    cancel_event.clear()
    current_tts_thread = threading.Thread(target=tts_generation_and_playback, args=(tts_stub, selected_voice))
    current_tts_thread.start()

# Get Available TTS Voices
def get_available_voices(tts_stub):
    print("Retrieving available voices from Riva TTS...")

    # Create the request to get synthesis configuration
    request = riva_tts_pb2.RivaSynthesisConfigRequest()

    # Call the GetRivaSynthesisConfig RPC
    try:
        response = tts_stub.GetRivaSynthesisConfig(request)
    except Exception as e:
        print(f"Failed to get TTS synthesis config: {e}")
        return []

    # Parse the response to find available voices
    available_voices = []
    if response.model_config:
        print("Available TTS Models:")
        for model in response.model_config:
            voice_name = model.parameters.get("voice_name", "")
            language_code = model.parameters.get("language_code", "")

            if voice_name:
                available_voices.append({"name": voice_name, "language": language_code})
                print(f"Voice: {voice_name}, Language: {language_code}")

    if not available_voices:
        print("No available voices found in the response.")

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
                current_option_idx = menu[current_row]['options'].index(menu[current_row]['current']) if menu[current_row]['current'] in menu[current_row]['options'] else 0
                if current_option_idx > 0:
                    current_option_idx -= 1
                menu[current_row]['current'] = menu[current_row]['options'][current_option_idx]
            elif key == curses.KEY_RIGHT:
                # Change option to next
                current_option_idx = menu[current_row]['options'].index(menu[current_row]['current']) if menu[current_row]['current'] in menu[current_row]['options'] else 0
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
    orchestrator_host = 'localhost'
    orchestrator_command_port = 6000  # As defined in orchestrator's orch.py

    registration_successful = False
    attempt = 0
    max_attempts = 12  # Try for up to 1 minute (12 * 5 seconds)

    while not registration_successful and attempt < max_attempts and not cancel_event.is_set():
        try:
            with socket.create_connection((orchestrator_host, orchestrator_command_port), timeout=5) as sock:
                register_command = f"/register {script_name} {script_uuid} {config['port']}\n"
                sock.sendall(register_command.encode())

                # Wait for acknowledgment
                response = sock.recv(1024).decode().strip()
                if response.startswith("/ack"):
                    ack_data = response.split(' ', 1)[1] if ' ' in response else ''
                    print(f"Successfully registered with orchestrator. Ack Data: {ack_data}")
                    registration_successful = True
                else:
                    print(f"Registration failed. Response: {response}")
        except ConnectionRefusedError:
            print(f"Orchestrator not available at {orchestrator_host}:{orchestrator_command_port}. Retrying in 5 seconds...")
        except Exception as e:
            print(f"Error during registration attempt {attempt + 1}: {e}")

        if not registration_successful:
            attempt += 1
            time.sleep(5)  # Wait before retrying

    if not registration_successful:
        print("Failed to register with the orchestrator after multiple attempts. Exiting...")
        cancel_event.set()
        exit(1)  # Exit the script as registration is critical

# Input Handler
def input_handler():
    if config['input_mode'] == 'terminal':
        terminal_input()
    elif config['input_mode'] == 'port':
        port_input()
    elif config['input_mode'] == 'route':
        route_input()
    else:
        print("Invalid input mode.")
        return

# Terminal Input Mode
def terminal_input():
    global config_changed
    print("Enter text to synthesize (type /config to enter configuration mode, /exit to quit):")
    while True:
        user_input = input("> ")
        if user_input.strip() == '/config':
            run_config_menu(config, available_voices)
            print("Configuration updated.")
            print("Current config:")
            print(config)
            config_changed = True
            # Write updated configuration to file
            write_config(config)
            break
        elif user_input.strip() == '/exit':
            cancel_event.set()
            break
        else:
            tts_queue.put(user_input.strip())

# Port Input Mode
def port_input():
    host = '0.0.0.0'
    port = int(config['port'])  # Convert port to integer
    print(f"Listening for text input on port {port}...")
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(5)
    print(f"Server started on port {port}. Waiting for connections...")

    while True:
        client_socket, addr = server_socket.accept()
        threading.Thread(target=handle_port_client, args=(client_socket, addr), daemon=True).start()

def handle_port_client(client_socket, addr):
    with client_socket:
        data = client_socket.recv(1024).decode().strip()
        if data == '/info':
            response = f"{script_name}\n{script_uuid}\n"
            # Include config
            for key, value in config.items():
                response += f"{key}={value}\n"
            response += 'EOF\n'
            client_socket.sendall(response.encode())
        elif data == '/config':
            client_socket.send("Entering configuration mode not supported over port.\n".encode())
        elif data == '/exit':
            client_socket.sendall(b"Exiting TTS Engine.\n")
            cancel_event.set()
        else:
            print(f"Received: {data}")
            tts_queue.put(data)
        client_socket.close()

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
                content_length = int(self.headers['Content-Length'])
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
                else:
                    print(f"Received text via HTTP POST: {post_data}")
                    tts_queue.put(post_data)
                    self.send_response(200)
                    self.end_headers()
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format, *args):
            return

    server_address = (host, port)
    httpd = HTTPServer(server_address, RequestHandler)
    print(f"HTTP server started on port {port}, route {route}")
    server_thread = threading.Thread(target=httpd.serve_forever)
    server_thread.start()

    try:
        while not cancel_event.is_set():
            pass
    except KeyboardInterrupt:
        pass
    finally:
        httpd.shutdown()
        httpd.server_close()
        server_thread.join()

def main():
    global config_changed
    global available_voices  # Add this to access available_voices in terminal_input

    # Read configuration from file
    global config
    config = read_config()

    # Register with the orchestrator
    register_with_orchestrator()

    # Initialize TTS stub
    try:
        if args.use_ssl and args.ssl_cert:
            with open(args.ssl_cert, 'rb') as f:
                creds = grpc.ssl_channel_credentials(f.read())
            channel = grpc.secure_channel(args.server, creds)
        else:
            channel = grpc.insecure_channel(args.server)

        tts_stub = riva_tts_pb2_grpc.RivaSpeechSynthesisStub(channel)
        print("TTS stub initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize TTS stub: {e}")
        return

    # Get available voices
    available_voices = get_available_voices(tts_stub)
    if not available_voices:
        print("No available voices.")
        return

    # Check if voice is specified in config
    selected_voice = None
    if config.get('voice'):
        # Check if the voice exists in the available voices
        matching_voices = [v for v in available_voices if v['name'] == config['voice']]
        if matching_voices:
            selected_voice = matching_voices[0]
            print(f"Using voice from config: {selected_voice['name']} (Language: {selected_voice['language']})")
        else:
            print(f"Voice '{config['voice']}' specified in config not found.")
            config['voice'] = ''  # Reset the voice in config

    # If voice is not specified or not found, prompt user to select
    if not selected_voice:
        print("\nAvailable voices:")
        for idx, voice in enumerate(available_voices):
            print(f"{idx + 1}: {voice['name']} (Language: {voice['language']})")

        while not selected_voice:
            user_input = input("Select a voice by number or name: ").strip()
            # Try to interpret input as a number
            try:
                idx = int(user_input) - 1
                if 0 <= idx < len(available_voices):
                    selected_voice = available_voices[idx]
                else:
                    print("Invalid selection. Please try again.")
            except ValueError:
                # Not a number, treat as name
                matching_voices = [v for v in available_voices if v['name'] == user_input]
                if matching_voices:
                    selected_voice = matching_voices[0]
                else:
                    print("Voice not found. Please try again.")

        print(f"Selected Voice: {selected_voice['name']} (Language: {selected_voice['language']})")
        # Update config with selected voice and write to file
        config['voice'] = selected_voice['name']
        write_config(config)

    # Start TTS thread with selected voice
    start_tts_thread(tts_stub, selected_voice)

    # Input handler loop
    while True:
        config_changed = False
        input_handler()
        if config_changed:
            print("Configuration has changed. Restarting input handler...")
            # Restart TTS thread if output mode has changed
            if 'output_mode' in config:
                start_tts_thread(tts_stub, selected_voice)
            continue
        else:
            break

    cancel_event.set()
    if current_tts_thread and current_tts_thread.is_alive():
        current_tts_thread.join()

    print("Program terminated.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Program terminated by user.")
