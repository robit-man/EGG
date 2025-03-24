#!/usr/bin/env python3
import os
import sys
import subprocess
import socket
import re
import json
import argparse
import threading
from queue import Queue
import shutil
import time
from contextlib import redirect_stdout
import inspect
import curses
import textwrap

#############################################
# Global Variables for Curses Display
#############################################
display_lock = threading.Lock()
current_request = ""
current_tokens = ""
current_tool_calls = ""
tts_flag = False      # Indicates if TTS process is active (flag for overall status)
tts_playing = False   # Indicates if audio is currently being played

#############################################
# Step 1: Ensure we're running inside a venv #
#############################################
VENV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv")
NEEDED_PACKAGES = ["requests", "num2words", "ollama", "pyserial", "dotenv", "beautifulsoup4"]

def in_venv():
    return hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)

def setup_venv():
    if not os.path.isdir(VENV_DIR):
        subprocess.check_call([sys.executable, '-m', 'venv', VENV_DIR])
    pip_path = os.path.join(VENV_DIR, 'bin', 'pip')
    subprocess.check_call([pip_path, 'install'] + NEEDED_PACKAGES)

def relaunch_in_venv():
    python_path = os.path.join(VENV_DIR, 'bin', 'python')
    os.execv(python_path, [python_path] + sys.argv)

if not in_venv():
    setup_venv()
    relaunch_in_venv()

#############################################
# Step 2: Imports after venv set up          #
#############################################
import requests
from num2words import num2words
from ollama import chat  # Use Ollama Python library for inference
import re
import serial
import psutil
from dotenv import load_dotenv
from bs4 import BeautifulSoup

load_dotenv()

#############################################
# Additional: Short Tone/Beep Utilities      #
#############################################
def beep(freq=3000, duration=0.05):
    command = [
        "play", "-n", "synth", "0.02",
        "sine", "13000",
        "sine", "14500",
        "square", "16000",
        "square", "19000"
    ]
    try:
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception:
        pass

#############################################
# Step 3: Config Defaults & File             #
#############################################
DEFAULT_CONFIG = {
    "model": "gemma3:12b",
    "stream": True,
    "format": None,
    "system": "YOU ARE EMBODIED INSIDE AN EGG SHAPED ROBOT, You can see the world and learn about things using TOOL CALLS which you have available to you! You have a set of chat messages back and forth, as well as a set of tools you can call, which are layed out as available functions, during each turn, you can choose to call a tool, but in calling the tool, do not  explain it or preface or read the contents of the function, simply wrap the function and any arguments passed in in triple backticks for the parser to catch it, and then the tool response will be passed to you, at which point you should package the tool response, and the initial user request into an appropriate concise reply, no verbose explaination of steps you are taking, words like (processing) just do all of that silently and only respond conversationally. Sometimes no tool call is needed and it is up to you to decide whether or not to use the tools available based on the users input during their turn.",
    "raw": False,
    "history": "chat.json",
    "history_depth": 100,
    "images": [],
    "tools": None,
    "options": {},
    "host": "0.0.0.0",
    "port": 64162,
    "tts_url": "http://localhost:61637/synthesize",
    "ollama_url": "http://localhost:11434/api/chat"
}
CONFIG_PATH = "config.json"

def load_config():
    if not os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'w') as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
        return dict(DEFAULT_CONFIG)
    else:
        try:
            with open(CONFIG_PATH, 'r') as f:
                cfg = json.load(f)
            for key, value in DEFAULT_CONFIG.items():
                if key not in cfg:
                    cfg[key] = value
            return cfg
        except Exception:
            return dict(DEFAULT_CONFIG)

CONFIG = load_config()

def update_config_file():
    try:
        with open(CONFIG_PATH, 'w') as f:
            json.dump(CONFIG, f, indent=2)
    except Exception:
        pass

#############################################
# Step 4: Parse Command-Line Arguments       #
#############################################
parser = argparse.ArgumentParser(description="Ollama Chat Server with TTS and advanced features.")
parser.add_argument("--model", type=str, help="Model name to use.")
parser.add_argument("--stream", action="store_true", help="Enable streaming responses from the model.")
parser.add_argument("--format", type=str, help="Structured output format: 'json' or path to JSON schema file.")
parser.add_argument("--system", type=str, help="System message override.")
parser.add_argument("--raw", action="store_true", help="If set, use raw mode (no template).")
parser.add_argument("--history", type=str, nargs='?', const="chat.json",
                    help="Path to a JSON file containing conversation history messages.")
parser.add_argument("--images", type=str, nargs='*', help="List of base64-encoded image files.")
parser.add_argument("--tools", type=str, help="Path to a JSON file defining tools.")
parser.add_argument("--option", action="append", help="Additional model parameters (e.g. --option temperature=0.7)")
args = parser.parse_args()

def merge_config_and_args(config, args):
    if args.model:
        config["model"] = args.model
    if args.stream:
        config["stream"] = True
    if args.format is not None:
        config["format"] = args.format
    if args.system is not None:
        config["system"] = args.system
    if args.raw:
        config["raw"] = True
    if args.history is not None:
        config["history"] = args.history
    if args.images is not None:
        config["images"] = args.images
    if args.tools is not None:
        config["tools"] = args.tools
    if args.option:
        for opt in args.option:
            if '=' in opt:
                k, v = opt.split('=', 1)
                k = k.strip()
                v = v.strip()
                if v.isdigit():
                    v = int(v)
                else:
                    try:
                        v = float(v)
                    except ValueError:
                        pass
                config["options"][k] = v
    return config

CONFIG = merge_config_and_args(CONFIG, args)

#############################################
# Monitor Config & Script Changes
#############################################
def monitor_config(interval=5):
    last_mtime = os.path.getmtime(CONFIG_PATH)
    while True:
        time.sleep(interval)
        try:
            new_mtime = os.path.getmtime(CONFIG_PATH)
            if new_mtime != last_mtime:
                with display_lock:
                    global current_tool_calls
                    current_tool_calls = "Config changed; reloading..."
                new_config = load_config()
                CONFIG.update(new_config)
                global history_messages
                history_messages = safe_load_json_file(CONFIG["history"], [])
                last_mtime = new_mtime
        except Exception:
            pass

def monitor_script(interval=5):
    script_path = os.path.abspath(__file__)
    last_mtime = os.path.getmtime(script_path)
    while True:
        time.sleep(interval)
        try:
            new_mtime = os.path.getmtime(script_path)
            if new_mtime != last_mtime:
                os.execv(sys.executable, [sys.executable] + sys.argv)
        except Exception:
            pass

#############################################
# Step 5: Load Optional Configurations       #
#############################################
def safe_load_json_file(path, default):
    if not path:
        return default
    if not os.path.exists(path):
        if path == CONFIG["history"] and default == []:
            with open(path, 'w') as f:
                json.dump([], f)
        return default
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return default

def load_format_schema(fmt):
    if not fmt:
        return None
    if fmt.lower() == "json":
        return "json"
    if os.path.exists(fmt):
        try:
            with open(fmt, 'r') as f:
                return json.load(f)
        except Exception:
            return None
    return None

history_messages = safe_load_json_file(CONFIG["history"], [])
tools_data = safe_load_json_file(CONFIG["tools"], None)
format_schema = load_format_schema(CONFIG["format"])

#############################################
# Step 5.1: Ensure Ollama and Model are Installed #
#############################################
def check_ollama_installed():
    return shutil.which('ollama') is not None

def install_ollama():
    try:
        subprocess.check_call('curl -fsSL https://ollama.com/install.sh | sh', shell=True, executable='/bin/bash')
    except subprocess.CalledProcessError:
        sys.exit(1)

def wait_for_ollama():
    ollama_tags_url = "http://localhost:11434/api/tags"
    max_retries = 10
    for attempt in range(max_retries):
        try:
            response = requests.get(ollama_tags_url)
            if response.status_code == 200:
                return
        except requests.exceptions.RequestException:
            pass
        time.sleep(2)
    sys.exit(1)

def get_available_models():
    ollama_tags_url = "http://localhost:11434/api/tags"
    try:
        response = requests.get(ollama_tags_url)
        if response.status_code == 200:
            data = response.json()
            available_models = data.get('models', [])
            return [m.get('name') for m in available_models if 'name' in m]
        else:
            return []
    except requests.exceptions.RequestException:
        return []

def check_model_exists_in_tags(model_name):
    available_models = get_available_models()
    if model_name in available_models:
        return model_name
    model_latest = f"{model_name}:latest"
    if model_latest in available_models:
        return model_latest
    return None

def check_model_installed(model_name):
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, check=True)
        models = [m.strip() for m in result.stdout.splitlines()]
        if model_name in models:
            return True
        if model_name.endswith(':latest'):
            base_model = model_name.rsplit(':', 1)[0]
            if base_model in models:
                return True
        return False
    except subprocess.CalledProcessError:
        sys.exit(1)

def pull_model(model_name):
    try:
        subprocess.check_call(['ollama', 'pull', model_name])
    except subprocess.CalledProcessError:
        sys.exit(1)

def ensure_ollama_and_model():
    if not check_ollama_installed():
        install_ollama()
        if not check_model_installed(CONFIG["model"]):
            sys.exit(1)
    wait_for_ollama()
    model_name = CONFIG["model"]
    model_actual_name = check_model_exists_in_tags(model_name)
    if not model_actual_name:
        sys.exit(1)
    CONFIG["model"] = model_actual_name

ensure_ollama_and_model()

#############################################
# Step 6: Ollama Chat Interaction
#############################################
OLLAMA_CHAT_URL = CONFIG["ollama_url"]

def convert_numbers_to_words(text):
    def replace_num(match):
        number_str = match.group(0)
        try:
            number_int = int(number_str)
            return num2words(number_int)
        except ValueError:
            return number_str
    return re.sub(r'\b\d+\b', replace_num, text)

def see_whats_around() -> str:
    import requests
    images_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    url = "http://127.0.0.1:8080/camera/0"
    try:
        response = requests.get(url, stream=True, timeout=5)
        if response.status_code == 200:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"camera_{timestamp}.jpg"
            file_path = os.path.join(images_dir, filename)
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return file_path
        else:
            return f"Error: {response.status_code}"
    except Exception as e:
        return f"Error: {e}"

def get_battery_voltage() -> float:
    try:
        home_dir = os.path.expanduser("~")
        file_path = os.path.join(home_dir, "voltage.txt")
        with open(file_path, "r") as f:
            line = f.readline().strip()
            voltage = float(line)
        return voltage
    except Exception as e:
        raise RuntimeError(f"Error reading battery voltage: {e}")

def brave_search(topic: str) -> str:
    api_key = os.environ.get("BRAVE_API_KEY", "")
    if not api_key:
        return "Error: BRAVE_API_KEY not set."
    endpoint = "https://api.search.brave.com/res/v1/web/search"
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "x-subscription-token": api_key
    }
    params = {
        "q": topic,
        "count": 3
    }
    try:
        response = requests.get(endpoint, headers=headers, params=params, timeout=5)
        if response.status_code == 200:
            return response.text
        else:
            return f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return f"Error: {e}"

def bs4_scrape(url: str) -> str:
    headers = {
        'User-Agent': ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/42.0.2311.135 Safari/537.36 Edge/12.246")
    }
    try:
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html5lib')
        return soup.prettify()
    except Exception as e:
        return f"Error during scraping: {e}"

def find_file(filename: str, search_path: str = ".") -> str:
    for root, dirs, files in os.walk(search_path):
        if filename in files:
            return root
    return None
     
def get_current_time() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_current_location() -> dict:
    import requests
    try:
        response = requests.get("http://ip-api.com/json", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP error {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}
        
def get_system_utilization() -> dict:
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_usage = psutil.virtual_memory().percent
    disk_usage = psutil.disk_usage('/').percent
    return {
        "cpu_usage": cpu_usage,
        "memory_usage": memory_usage,
        "disk_usage": disk_usage
    }

def extract_tool_call(text):
    import io
    from contextlib import redirect_stdout
    # Updated regex to accept both tool_code and tool_call
    pattern = r"```tool_(?:code|call)\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        code = match.group(1).strip()
        f = io.StringIO()
        with redirect_stdout(f):
            try:
                result = eval(code, globals())
            except Exception as e:
                result = f"Error executing tool: {e}"
        output = f.getvalue()
        r = result if output == '' else output
        return f'```tool_output\n{r}\n```'
    return None

def build_payload(user_message):
    global current_request, current_tokens, current_tool_calls
    with display_lock:
        current_request = user_message
        current_tokens = ""
        current_tool_calls = ""
    messages = []
    if CONFIG["system"]:
        messages.append({"role": "system", "content": CONFIG["system"]})
    tool_instructions = (
        "At each turn, if you decide to invoke any of the function(s), it should be wrapped with \n\n```tool_call\nfunction_name(arguments)\n```\n\n"
        "Review the following Python methods (source code provided for context) to determine if a tool call is appropriate:\n\n"
        "```python\n" +
        inspect.getsource(see_whats_around) + "\n" +
        inspect.getsource(brave_search) + "\n" +
        inspect.getsource(get_battery_voltage) + "\n" +
        inspect.getsource(get_current_time) + "\n" +
        inspect.getsource(get_current_location) + "\n" +
        inspect.getsource(bs4_scrape) + "\n" +
        inspect.getsource(find_file) + "\n" +
        inspect.getsource(pull_model) + "\n" +
        inspect.getsource(get_available_models) + "\n" +
        inspect.getsource(check_model_exists_in_tags) + "\n" +
        inspect.getsource(check_model_installed) + "\n" +
        inspect.getsource(ensure_ollama_and_model) + "\n" +
        inspect.getsource(get_system_utilization) +
        "\n```\n\n"
        "When using a tool call, the generated code should be readable and efficient. "
        "The response from a tool call will be wrapped in ```tool_output```."
    )
    messages.append({"role": "system", "content": tool_instructions})
    if len(history_messages) > CONFIG.get("history_depth", 40):
        messages.extend(history_messages[-CONFIG["history_depth"]:])
    else:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": user_message})
    payload = {
        "model": CONFIG["model"],
        "messages": messages,
        "stream": CONFIG["stream"]
    }
    if format_schema:
        payload["format"] = format_schema
    if CONFIG["raw"]:
        payload["raw"] = True
    if CONFIG["images"]:
        if payload["messages"] and payload["messages"][-1]["role"] == "user":
            payload["messages"][-1]["images"] = CONFIG["images"]
    if tools_data:
        payload["tools"] = tools_data
    if CONFIG["options"]:
        payload["options"] = CONFIG["options"]
    return payload

#############################################
# Curses Display
#############################################
def curses_display(stdscr):
    global current_request, current_tokens, current_tool_calls, tts_flag, tts_playing
    curses.curs_set(0)
    stdscr.nodelay(True)
    while True:
        with display_lock:
            stdscr.erase()
            max_height, max_width = stdscr.getmaxyx()
            try:
                stdscr.addnstr(0, 0, f"User: {current_request}", max_width)
            except curses.error:
                pass
            try:
                stdscr.addnstr(1, 0, f"Characteristics: Tools Called: {current_tool_calls} | TTS Flag: {tts_flag} | TTS Playing: {tts_playing}", max_width)
            except curses.error:
                pass
            try:
                stdscr.hline(2, 0, '-', max_width)
            except curses.error:
                pass
            try:
                stdscr.addnstr(3, 0, "Model Tokens:", max_width)
            except curses.error:
                pass
            wrapped_lines = textwrap.wrap(current_tokens, width=max_width)
            for idx, line in enumerate(wrapped_lines):
                if 4 + idx >= max_height:
                    break
                try:
                    stdscr.addnstr(4 + idx, 0, line, max_width)
                except curses.error:
                    pass
        stdscr.refresh()
        time.sleep(0.5)

#############################################
# Step 7: TTS Playback with Queue and Thread
#############################################
tts_queue = None
tts_stop_flag = False
tts_thread = None
tts_thread_lock = threading.Lock()

def synthesize_and_play(prompt):
    global tts_flag, tts_playing
    tts_flag = True
    tts_playing = True
    prompt = re.sub(r'[\*#]', '', prompt).strip()
    prompt = remove_emojis(prompt)
    if not prompt:
        tts_flag = False
        tts_playing = False
        return
    try:
        payload = {"prompt": prompt}
        with requests.post(CONFIG["tts_url"], json=payload, stream=True) as response:
            if response.status_code != 200:
                tts_flag = False
                tts_playing = False
                return
            aplay = subprocess.Popen(
                ['aplay', '-r', '22050', '-f', 'S16_LE', '-t', 'raw'],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            for chunk in response.iter_content(chunk_size=4096):
                if tts_stop_flag:
                    break
                if chunk:
                    aplay.stdin.write(chunk)
            aplay.stdin.close()
            aplay.wait()
    except Exception:
        pass
    tts_flag = False
    tts_playing = False

def tts_worker():
    global tts_stop_flag
    while not tts_stop_flag:
        try:
            sentence = tts_queue.get(timeout=0.1)
        except Exception:
            if tts_stop_flag:
                break
            continue
        if tts_stop_flag:
            break
        synthesize_and_play(sentence)
    tts_stop_flag = False

def start_tts_thread():
    global tts_queue, tts_thread, tts_stop_flag
    with tts_thread_lock:
        if tts_thread and tts_thread.is_alive():
            return
        tts_stop_flag = False
        tts_queue = Queue()
        tts_thread = threading.Thread(target=tts_worker, daemon=True)
        tts_thread.start()

def stop_tts_thread():
    global tts_stop_flag, tts_thread, tts_queue
    with tts_thread_lock:
        if tts_thread and tts_thread.is_alive():
            tts_stop_flag = True
            with tts_queue.mutex:
                tts_queue.queue.clear()
            tts_thread.join()
        tts_stop_flag = False
        tts_queue = None
        tts_thread = None

def enqueue_sentence_for_tts(sentence):
    if tts_queue and not tts_stop_flag:
        tts_queue.put(sentence)

#############################################
# Step 7.1: Non-blocking beep while waiting #
#############################################
wait_beeps_thread = None
wait_beeps_flag = False
wait_beeps_lock = threading.Lock()

def wait_beeps_worker():
    while True:
        with wait_beeps_lock:
            if not wait_beeps_flag:
                break
        beep(80, 0.05)
        time.sleep(0.1)
        beep(80, 0.05)
        time.sleep(0.1)
        beep(80, 0.05)
        time.sleep(0.65)

def start_wait_beeps():
    global wait_beeps_thread, wait_beeps_flag
    with wait_beeps_lock:
        if wait_beeps_thread and wait_beeps_thread.is_alive():
            return
        wait_beeps_flag = True
    wait_beeps_thread = threading.Thread(target=wait_beeps_worker, daemon=True)
    wait_beeps_thread.start()

def stop_wait_beeps():
    global wait_beeps_thread, wait_beeps_flag
    with wait_beeps_lock:
        wait_beeps_flag = False
    if wait_beeps_thread and wait_beeps_thread.is_alive():
        wait_beeps_thread.join()
    wait_beeps_thread = None

#############################################
# Step 8: Streaming the Output via Ollama Library
#############################################
def chat_completion_stream(user_message):
    global current_tokens, current_tool_calls
    payload = build_payload(user_message)
    tokens = ""
    try:
        stream = chat(model=CONFIG["model"], messages=payload["messages"], stream=CONFIG["stream"])
        for part in stream:
            content = part["message"]["content"]
            done = part.get("done", False)
            tokens += content
            with display_lock:
                current_tokens = tokens
            yield content, done
            if done:
                break
    except Exception:
        yield "", True

def chat_completion_nonstream(user_message):
    payload = build_payload(user_message)
    try:
        response = chat(model=CONFIG["model"], messages=payload["messages"], stream=False)
        result = response["message"]["content"]
        return result
    except Exception:
        return ""

#############################################
# Step 9: Processing the Model Output
#############################################
def process_text(text, skip_tts=False):
    global current_tokens, current_tool_calls
    processed_text = convert_numbers_to_words(text)
    sentence_endings = re.compile(r'[.?!]+')
    tokens = ""
    if CONFIG["stream"]:
        buffer = ""
        for content, done in chat_completion_stream(processed_text):
            buffer += content
            tokens += content
            with display_lock:
                current_tokens = tokens
            while True:
                match = sentence_endings.search(buffer)
                if not match:
                    break
                end_index = match.end()
                sentence = buffer[:end_index].strip()
                buffer = buffer[end_index:].lstrip()
                if sentence:
                    if not skip_tts:
                        threading.Thread(target=enqueue_sentence_for_tts, args=(sentence,), daemon=True).start()
            if done:
                break
        if buffer.strip():
            leftover = buffer.strip()
            tokens += leftover
            with display_lock:
                current_tokens = tokens
        return tokens
    else:
        result = chat_completion_nonstream(processed_text)
        tokens = result
        with display_lock:
            current_tokens = tokens
        return tokens

#############################################
# Step 10: Update History File with New Messages
#############################################
def update_history(user_message, assistant_message):
    if not CONFIG["history"]:
        return
    current_history = safe_load_json_file(CONFIG["history"], [])
    current_history.append({"role": "user", "content": user_message})
    current_history.append({"role": "assistant", "content": assistant_message})
    try:
        with open(CONFIG["history"], 'w') as f:
            json.dump(current_history, f, indent=2)
    except Exception:
        pass

#############################################
# Step 11: Handling Concurrent Requests and Cancellation
#############################################
stop_flag = False
current_thread = None
inference_lock = threading.Lock()

def inference_thread(user_message, result_holder, model_actual_name, skip_tts):
    global stop_flag
    stop_flag = False
    result = process_text(user_message, skip_tts)
    result_holder.append(result)

def new_request(user_message, model_actual_name, depth=0, skip_tts=False):
    global stop_flag, current_thread, current_tool_calls, current_request
    with display_lock:
        current_request = user_message
        current_tool_calls = ""
    with inference_lock:
        if current_thread and current_thread.is_alive():
            stop_flag = True
            current_thread.join()
            stop_flag = False
        stop_tts_thread()
        start_tts_thread()
        result_holder = []
        current_thread = threading.Thread(
            target=inference_thread,
            args=(user_message, result_holder, model_actual_name, skip_tts)
        )
        current_thread.start()
    current_thread.join()
    result = result_holder[0] if result_holder else ""
    tool_call = extract_tool_call(result)
    if tool_call and depth < 1:
        with display_lock:
            current_tool_calls = tool_call
        stop_tts_thread()
        new_message = user_message + "\n" + tool_call
        return new_request(new_message, model_actual_name, depth=depth+1, skip_tts=False)
    return result

#############################################
# Step 12: Interactive Command-Line Interface
#############################################
def interactive_loop():
    while True:
        try:
            cmd = input("[Interactive] > ").strip()
        except EOFError:
            break
        if not cmd:
            continue
        if cmd.startswith("/send "):
            message = cmd[len("/send "):].strip()
            with display_lock:
                global current_request
                current_request = message
            result = new_request(message, CONFIG["model"])
            update_history(message, result)
        elif cmd.startswith("/model "):
            new_model = cmd[len("/model "):].strip()
            CONFIG["model"] = new_model
            ensure_ollama_and_model()
            update_config_file()
        elif cmd.startswith("/history_depth "):
            try:
                depth = int(cmd[len("/history_depth "):].strip())
                CONFIG["history_depth"] = depth
                update_config_file()
            except ValueError:
                pass
        elif cmd == "/quit":
            break

def start_interactive_thread():
    thread = threading.Thread(target=interactive_loop, daemon=True)
    thread.start()

#############################################
# Step 13: Start Server with Enhanced Interrupt Handling
#############################################
HOST = CONFIG["host"]
PORT = CONFIG["port"]

client_threads = []
client_threads_lock = threading.Lock()

def handle_client_connection(client_socket, address, model_actual_name):
    try:
        data = client_socket.recv(65536)
        if not data:
            return
        user_message = data.decode('utf-8').strip()
        if not user_message:
            return
        result = new_request(user_message, model_actual_name)
        client_socket.sendall(result.encode('utf-8'))
        update_history(user_message, result)
    except Exception:
        pass
    finally:
        client_socket.close()

def start_server():
    global client_threads
    start_tts_thread()
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server.bind((HOST, PORT))
    except Exception:
        server.bind(('0.0.0.0', 64162))
    server.listen(5)
    model_actual_name = CONFIG["model"]
    while True:
        try:
            client_sock, addr = server.accept()
            client_thread = threading.Thread(
                target=handle_client_connection,
                args=(client_sock, addr, model_actual_name)
            )
            client_thread.start()
            with client_threads_lock:
                client_threads.append(client_thread)
        except KeyboardInterrupt:
            break
        except Exception:
            pass
    server.close()
    stop_tts_thread()
    with client_threads_lock:
        for t in client_threads:
            t.join()

#############################################
# Step 14: Interactive Command-Line Interface Thread
#############################################
def start_interactive_thread():
    thread = threading.Thread(target=interactive_loop, daemon=True)
    thread.start()

#############################################
# Utility: Remove Emojis from text
#############################################
def remove_emojis(text):
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)

#############################################
# New: Monitor the script file for changes and restart
#############################################
def monitor_script(interval=5):
    script_path = os.path.abspath(__file__)
    last_mtime = os.path.getmtime(script_path)
    while True:
        time.sleep(interval)
        try:
            new_mtime = os.path.getmtime(script_path)
            if new_mtime != last_mtime:
                os.execv(sys.executable, [sys.executable] + sys.argv)
        except Exception:
            pass

#############################################
# Curses Display Function with Scrolling/ Wrapping
#############################################
def curses_display(stdscr):
    global current_request, current_tokens, current_tool_calls, tts_flag, tts_playing
    curses.curs_set(0)
    stdscr.nodelay(True)
    while True:
        with display_lock:
            stdscr.erase()
            max_height, max_width = stdscr.getmaxyx()
            try:
                stdscr.addnstr(0, 0, f"User: {current_request}", max_width)
            except curses.error:
                pass
            try:
                stdscr.addnstr(1, 0, f"Characteristics: Tools Called: {current_tool_calls} | TTS Flag: {tts_flag} | TTS Playing: {tts_playing}", max_width)
            except curses.error:
                pass
            try:
                stdscr.hline(2, 0, '-', max_width)
            except curses.error:
                pass
            try:
                stdscr.addnstr(3, 0, "Model Tokens:", max_width)
            except curses.error:
                pass
            wrapped_lines = textwrap.wrap(current_tokens, width=max_width)
            for idx, line in enumerate(wrapped_lines):
                if 4 + idx >= max_height:
                    break
                try:
                    stdscr.addnstr(4 + idx, 0, line, max_width)
                except curses.error:
                    pass
        stdscr.refresh()
        time.sleep(0.5)

#############################################
# Main: Launch Curses and Start Threads
#############################################
def main(stdscr):
    curses_thread = threading.Thread(target=curses_display, args=(stdscr,), daemon=True)
    curses_thread.start()
    monitor_thread = threading.Thread(target=monitor_config, daemon=True)
    monitor_thread.start()
    script_monitor_thread = threading.Thread(target=monitor_script, daemon=True)
    script_monitor_thread.start()
    start_interactive_thread()
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    while True:
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    curses.wrapper(main)
