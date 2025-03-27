#!/usr/bin/env python3
import os
import sys
import subprocess
import socket
import re
import json
import argparse
import threading
import time
import inspect
import shutil
import curses
import textwrap
from queue import Queue
from contextlib import redirect_stdout
from datetime import datetime
import sqlite3
import numpy as np

#############################################
# Environment Setup (Venv)
#############################################
class EnvironmentManager:
    VENV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv")
    NEEDED_PACKAGES = [
        "requests", "num2words", "ollama", "pyserial", "dotenv",
        "beautifulsoup4", "pywifi", "numpy"
    ]

    @staticmethod
    def in_venv():
        return hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)

    @staticmethod
    def setup_venv():
        if not os.path.isdir(EnvironmentManager.VENV_DIR):
            subprocess.check_call([sys.executable, '-m', 'venv', EnvironmentManager.VENV_DIR])
        pip_path = os.path.join(EnvironmentManager.VENV_DIR, 'bin', 'pip')
        subprocess.check_call([pip_path, 'install'] + EnvironmentManager.NEEDED_PACKAGES)

    @staticmethod
    def relaunch_in_venv():
        python_path = os.path.join(EnvironmentManager.VENV_DIR, 'bin', 'python')
        os.execv(python_path, [python_path] + sys.argv)

if not EnvironmentManager.in_venv():
    EnvironmentManager.setup_venv()
    EnvironmentManager.relaunch_in_venv()

#############################################
# Imports after venv set up
#############################################
import requests
from num2words import num2words
from ollama import chat  # Ollama Python library for inference
import serial
import psutil
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import pywifi
from pywifi import const
load_dotenv()

#############################################
# Utility Functions (non-tool helpers)
#############################################
class Utils:
    @staticmethod
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

    @staticmethod
    def convert_numbers_to_words(text):
        def replace_num(match):
            number_str = match.group(0)
            try:
                return num2words(int(number_str))
            except ValueError:
                return number_str
        return re.sub(r'\b\d+\b', replace_num, text)

    @staticmethod
    def get_current_time():
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    @staticmethod
    def cosine_similarity(vec1, vec2):
        # Compute cosine similarity between two numpy arrays
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

    @staticmethod
    def safe_load_json_file(path, default):
        if not path:
            return default
        if not os.path.exists(path):
            if default == []:
                with open(path, 'w') as f:
                    json.dump([], f)
            return default
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception:
            return default

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def embed_text(text):
        # Use Ollama's nomic-embed-text model to generate an embedding.
        try:
            # The embedding result is assumed to be a JSON string containing a list of floats.
            response = chat(model="nomic-embed-text", messages=[{"role": "user", "content": text}], stream=False)
            # For example, response["message"]["content"] may be " [0.1, 0.2, 0.3, ...] "
            embedding = json.loads(response["message"]["content"])
            vec = np.array(embedding, dtype=float)
            # Normalize the vector
            norm = np.linalg.norm(vec)
            if norm == 0:
                return vec
            return vec / norm
        except Exception as e:
            return np.zeros(768)  # Fallback vector (size adjust as needed)

#############################################
# Tools Class (isolated tool functions)
#############################################
class Tools:
    @staticmethod
    def parse_tool_call(text):
        # Extracts a function call wrapped in triple backticks with label 'tool_code'
        pattern = r"```tool_(?:code|call)\s*(.*?)\s*```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    # Additional utility methods (e.g. see_whats_around, battery voltage, etc.)
    @staticmethod
    def see_whats_around():
        """
        Fetch image from camera URL and save locally, returning the file path.
        This should be used any time you want to gather more visual context.
        """
        images_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)
        url = "http://127.0.0.1:8080/camera/0"
        try:
            response = requests.get(url, stream=True, timeout=5)
            if response.status_code == 200:
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

    @staticmethod
    def get_battery_voltage():
        """
        Reads the battery voltage from a file located in the user's home directory.
    
        The file is expected to be at:
            ~/voltage.txt
        and contain a single line with the battery voltage as a floating point number.
    
        Returns:
           A float representing the battery voltage.
        
        Raises:
           RuntimeError if the file cannot be read or its content cannot be converted to float.
        """
        try:
            home_dir = os.path.expanduser("~")
            file_path = os.path.join(home_dir, "voltage.txt")
            with open(file_path, "r") as f:
                return float(f.readline().strip())
        except Exception as e:
            raise RuntimeError(f"Error reading battery voltage: {e}")

    @staticmethod
    def brave_search(topic):
        """
        Search Brave Web Search API for the specified topic.
    
        Args:
            topic (str): The search query.
    
        Returns:
            A string representing the JSON search results if successful,
            or an error message if the search fails.
        """
        api_key = os.environ.get("BRAVE_API_KEY", "")
        if not api_key:
            return "Error: BRAVE_API_KEY not set."
        endpoint = "https://api.search.brave.com/res/v1/web/search"
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "x-subscription-token": api_key
        }
        params = {"q": topic, "count": 3}
        try:
            response = requests.get(endpoint, headers=headers, params=params, timeout=5)
            return response.text if response.status_code == 200 else f"Error {response.status_code}: {response.text}"
        except Exception as e:
            return f"Error: {e}"

    @staticmethod
    def bs4_scrape(url):
        """
        Scrape the provided website URL using BeautifulSoup and return the prettified HTML.
    
        Args:
            url (str): The URL of the website to scrape.
        
        Returns:
            A string containing the prettified HTML of the page if successful,
            or an error message if the scraping fails.
        """
        headers = {
            'User-Agent': ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/42.0.2311.135 Safari/537.36 Edge/12.246")
        }
        try:
            response = requests.get(url, headers=headers, timeout=5)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html5lib')
            return soup.prettify()
        except Exception as e:
            return f"Error during scraping: {e}"

    @staticmethod
    def find_file(filename, search_path="."):
        """
        Search recursively for a file with the given filename starting from the specified search path.
    
        Args:
            filename (str): The name of the file to search for.
            search_path (str): The directory to start the search from. Defaults to the current directory.
    
        Returns:
            str or None: The directory path where the file was found, or None if the file is not found.
        """
        for root, dirs, files in os.walk(search_path):
            if filename in files:
                return root
        return None

    @staticmethod
    def get_current_location():
        """
        Retrieves the current location based on your IP address by querying an IP geolocation API.
    
        Returns:
            dict: A dictionary with location information if successful.
        """
        try:
            response = requests.get("http://ip-api.com/json", timeout=5)
            return response.json() if response.status_code == 200 else {"error": f"HTTP error {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def get_system_utilization():
        """
        Return system utilization metrics as a dictionary.
        """
        return {
            "cpu_usage": psutil.cpu_percent(interval=1),
            "memory_usage": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent
        }

    @staticmethod
    def secondary_agent_tool(prompt: str) -> str:
        # Use llama3.2:3b for tool calling (instead of gemma)
        secondary_model = "llama3.2:3b"
        payload = {
            "model": secondary_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }
        try:
            response = chat(model=secondary_model, messages=payload["messages"], stream=False)
            return response["message"]["content"]
        except Exception as e:
            return f"Error in secondary agent: {e}"

#############################################
# Memory Manager (Persistent vector storage using SQLite3)
#############################################
class MemoryManager:
    def __init__(self, db_path="memory.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.create_tables()

    def create_tables(self):
        cursor = self.conn.cursor()
        # Table for raw chat memory with embeddings stored as JSON text.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT,
                role TEXT,
                timestamp TEXT,
                content TEXT,
                embedding TEXT
            )
        """)
        # Table for summary narrative (stores condensed summaries and narrative states)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS summary_narrative (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT,
                summary_text TEXT,
                narrative_state TEXT,
                timestamp TEXT
            )
        """)
        self.conn.commit()

    def store_message(self, conversation_id, role, content, embedding):
        timestamp = Utils.get_current_time()
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO chat_memory (conversation_id, role, timestamp, content, embedding)
            VALUES (?, ?, ?, ?, ?)
        """, (conversation_id, role, timestamp, content, json.dumps(embedding.tolist())))
        self.conn.commit()

    def retrieve_similar(self, conversation_id, query_embedding, top_n=5, mode="conversational"):
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, timestamp, role, content, embedding FROM chat_memory WHERE conversation_id=?", (conversation_id,))
        rows = cursor.fetchall()
        results = []
        now = datetime.now()
        for row in rows:
            msg_id, ts, role, content, emb_text = row
            try:
                emb = np.array(json.loads(emb_text), dtype=float)
            except Exception:
                continue
            sim = Utils.cosine_similarity(query_embedding, emb)
            # Optionally adjust similarity based on recency in conversational mode.
            if mode == "conversational":
                try:
                    msg_time = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
                    age = (now - msg_time).total_seconds()
                    decay = np.exp(-age / 3600)  # decay over 1 hour
                    sim *= decay
                except Exception:
                    pass
            results.append((sim, msg_id, ts, role, content))
        results.sort(key=lambda x: x[0], reverse=True)
        return results[:top_n]

    def store_summary(self, conversation_id, summary_text, narrative_state):
        timestamp = Utils.get_current_time()
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO summary_narrative (conversation_id, summary_text, narrative_state, timestamp)
            VALUES (?, ?, ?, ?)
        """, (conversation_id, summary_text, narrative_state, timestamp))
        self.conn.commit()

    def retrieve_latest_summary(self, conversation_id):
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT summary_text, narrative_state, timestamp FROM summary_narrative
            WHERE conversation_id=?
            ORDER BY id DESC LIMIT 1
        """, (conversation_id,))
        return cursor.fetchone()

#############################################
# Mode Manager (Automatic mode switching using llama3.2:3b)
#############################################
class ModeManager:
    def __init__(self, model_id="llama3.2:3b"):
        self.model_id = model_id
        self.prompt_template = (
            "You are a conversation mode classifier. Analyze the following conversation context and determine the appropriate mode. "
            "The possible modes are: 'conversational', 'research', 'meditation'. Output a JSON object with key 'mode'.\n"
            "Conversation context:\n{context}\n"
            "Output only a JSON, for example: {{\"mode\": \"research\"}}."
        )

    def detect_mode(self, conversation_history):
        context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history[-5:]])
        prompt = self.prompt_template.format(context=context)
        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }
        try:
            response = chat(model=self.model_id, messages=payload["messages"], stream=False)
            mode_info = json.loads(response["message"]["content"])
            return mode_info.get("mode", "conversational")
        except Exception:
            return "conversational"


#############################################
# Configuration Manager
#############################################
class ConfigManager:
    DEFAULT_CONFIG = {
        "model": "gemma3:12b",
        "stream": True,
        "format": None,
        "system": "YOU ARE EMBODIED INSIDE AN EGG SHAPED ROBOT. You have a set of chat messages back and forth; use the available tools as needed without verbose explanation. To use a function, wrap the function call in triple backticks with the label `tool_code`. The Python methods described below are imported and available – you can only use these defined methods. The generated code should be readable and efficient. The response to a method will be wrapped in triple backticks with the label `tool_output` – use this to call additional functions or generate a helpful, friendly response. When using a function call, think step by step about why and how it should be used.\n\nThe following Python methods are available:\n\n```python\n" + "\n".join([inspect.getsource(getattr(Tools, attr)) for attr in dir(Tools) if not attr.startswith("_") and callable(getattr(Tools, attr))]) + "\n```",
        "raw": False,
        "history": "chat555.json",
        "history_depth": 100,
        "images": [],
        "tools": None,
        "options": {},
        "host": "0.0.0.0",
        "port": 64162,
        "tts_url": "http://localhost:61637/synthesize",
        "ollama_url": "http://localhost:11434/api/chat",
        "conversation_id": "default_convo"
    }

    def __init__(self, config_path="config.json"):
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self):
        if not os.path.exists(self.config_path):
            with open(self.config_path, 'w') as f:
                json.dump(ConfigManager.DEFAULT_CONFIG, f, indent=2)
            return dict(ConfigManager.DEFAULT_CONFIG)
        try:
            with open(self.config_path, 'r') as f:
                cfg = json.load(f)
            for key, value in ConfigManager.DEFAULT_CONFIG.items():
                if key not in cfg:
                    cfg[key] = value
            return cfg
        except Exception:
            return dict(ConfigManager.DEFAULT_CONFIG)

    def update_config(self):
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception:
            pass

    def merge_args(self, args):
        if args.model:
            self.config["model"] = args.model
        if args.stream:
            self.config["stream"] = True
        if args.format is not None:
            self.config["format"] = args.format
        if args.system is not None:
            self.config["system"] = args.system
        if args.raw:
            self.config["raw"] = True
        if args.history is not None:
            self.config["history"] = args.history
        if args.images is not None:
            self.config["images"] = args.images
        if args.tools is not None:
            self.config["tools"] = args.tools
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
                    self.config["options"][k] = v

    def monitor_config(self, interval=5):
        last_mtime = os.path.getmtime(self.config_path)
        while True:
            time.sleep(interval)
            try:
                new_mtime = os.path.getmtime(self.config_path)
                if new_mtime != last_mtime:
                    with display_state.lock:
                        display_state.current_tool_calls = "Config changed; reloading..."
                    new_config = self.load_config()
                    self.config.update(new_config)
                    last_mtime = new_mtime
            except Exception:
                pass

#############################################
# History Manager (for chat history in JSON)
#############################################
class HistoryManager:
    def __init__(self, history_path):
        self.history_path = history_path
        self.history = Utils.safe_load_json_file(history_path, [])

    def add_entry(self, role, content):
        timestamp = Utils.get_current_time()
        entry = {"role": role, "content": content, "timestamp": timestamp}
        with display_state.lock:
            display_state.chat_history_state = "Writing"
        self.history.append(entry)
        try:
            with open(self.history_path, 'w') as f:
                json.dump(self.history, f, indent=2)
        except Exception:
            pass
        with display_state.lock:
            display_state.chat_history_state = "Reading"

#############################################
# Model Manager (Ollama and Model Checks)
#############################################
class ModelManager:
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager

    def check_ollama_installed(self):
        return shutil.which('ollama') is not None

    def install_ollama(self):
        try:
            subprocess.check_call('curl -fsSL https://ollama.com/install.sh | sh', shell=True, executable='/bin/bash')
        except subprocess.CalledProcessError:
            sys.exit(1)

    def wait_for_ollama(self):
        ollama_tags_url = self.config_manager.config["ollama_url"].replace("/api/chat", "/api/tags")
        max_retries = 10
        for _ in range(max_retries):
            try:
                response = requests.get(ollama_tags_url)
                if response.status_code == 200:
                    return
            except requests.exceptions.RequestException:
                pass
            time.sleep(2)
        sys.exit(1)

    def get_available_models(self):
        ollama_tags_url = self.config_manager.config["ollama_url"].replace("/api/chat", "/api/tags")
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

    def check_model_exists_in_tags(self, model_name):
        available_models = self.get_available_models()
        if model_name in available_models:
            return model_name
        model_latest = f"{model_name}:latest"
        if model_latest in available_models:
            return model_latest
        return None

    def check_model_installed(self, model_name):
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

    def pull_model(self, model_name):
        try:
            subprocess.check_call(['ollama', 'pull', model_name])
        except subprocess.CalledProcessError:
            sys.exit(1)

    def ensure_ollama_and_model(self):
        if not self.check_ollama_installed():
            self.install_ollama()
            if not self.check_model_installed(self.config_manager.config["model"]):
                sys.exit(1)
        self.wait_for_ollama()
        model_name = self.config_manager.config["model"]
        model_actual_name = self.check_model_exists_in_tags(model_name)
        if not model_actual_name:
            sys.exit(1)
        self.config_manager.config["model"] = model_actual_name

#############################################
# TTS Manager
#############################################
class TTSManager:
    def __init__(self, tts_url):
        self.tts_url = tts_url
        self.queue = Queue()
        self.thread = None
        self.stop_flag = False
        self.lock = threading.Lock()
        self.tts_flag = False
        self.tts_playing = False

    def synthesize_and_play(self, prompt):
        self.tts_flag = True
        self.tts_playing = True
        prompt = re.sub(r'[\*#]', '', prompt).strip()
        prompt = Utils.remove_emojis(prompt)
        if not prompt:
            self.tts_flag = False
            self.tts_playing = False
            return
        try:
            payload = {"prompt": prompt}
            with requests.post(self.tts_url, json=payload, stream=True) as response:
                if response.status_code != 200:
                    self.tts_flag = False
                    self.tts_playing = False
                    return
                aplay = subprocess.Popen(
                    ['aplay', '-r', '22050', '-f', 'S16_LE', '-t', 'raw'],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                for chunk in response.iter_content(chunk_size=4096):
                    if self.stop_flag:
                        break
                    if chunk:
                        aplay.stdin.write(chunk)
                aplay.stdin.close()
                aplay.wait()
        except Exception:
            pass
        self.tts_flag = False
        self.tts_playing = False

    def tts_worker(self):
        while not self.stop_flag:
            try:
                sentence = self.queue.get(timeout=0.1)
            except Exception:
                if self.stop_flag:
                    break
                continue
            if self.stop_flag:
                break
            self.synthesize_and_play(sentence)
        self.stop_flag = False

    def start(self):
        with self.lock:
            if self.thread and self.thread.is_alive():
                return
            self.stop_flag = False
            self.thread = threading.Thread(target=self.tts_worker, daemon=True)
            self.thread.start()

    def stop(self):
        with self.lock:
            if self.thread and self.thread.is_alive():
                self.stop_flag = True
                with self.queue.mutex:
                    self.queue.queue.clear()
                self.thread.join()
            self.stop_flag = False
            self.queue = Queue()
            self.thread = None

    def enqueue(self, sentence):
        if self.queue and not self.stop_flag:
            self.queue.put(sentence)

#############################################
# Wait Beep Manager (for non-blocking beep)
#############################################
class WaitBeepManager:
    def __init__(self):
        self.thread = None
        self.flag = False
        self.lock = threading.Lock()

    def wait_beeps_worker(self):
        while True:
            with self.lock:
                if not self.flag:
                    break
            Utils.beep(80, 0.05)
            time.sleep(0.1)
            Utils.beep(80, 0.05)
            time.sleep(0.1)
            Utils.beep(80, 0.05)
            time.sleep(0.65)

    def start(self):
        with self.lock:
            if self.thread and self.thread.is_alive():
                return
            self.flag = True
        self.thread = threading.Thread(target=self.wait_beeps_worker, daemon=True)
        self.thread.start()

    def stop(self):
        with self.lock:
            self.flag = False
        if self.thread and self.thread.is_alive():
            self.thread.join()
        self.thread = None

#############################################
# Chat Manager (Conversation, Inference, Tool Calling & Memory Integration)
#############################################
class ChatManager:
    def __init__(self, config_manager: ConfigManager, history_manager: HistoryManager,
                 tts_manager: TTSManager, tools_data, format_schema,
                 memory_manager: MemoryManager, mode_manager: ModeManager):
        self.config_manager = config_manager
        self.history_manager = history_manager
        self.tts_manager = tts_manager
        self.tools_data = tools_data
        self.format_schema = format_schema
        self.memory_manager = memory_manager
        self.mode_manager = mode_manager
        self.current_tokens = ""
        self.current_tool_calls = ""
        self.stop_flag = False
        self.inference_lock = threading.Lock()
        self.current_thread = None
        # We also track the current mode and memory retrieval results in display_state.
        self.conversation_id = self.config_manager.config.get("conversation_id", "default_convo")

    def build_payload(self):
        cfg = self.config_manager.config
        system_prompt = cfg.get("system", "")
        tools_source = ""
        for attr in dir(Tools):
            if not attr.startswith("_"):
                method = getattr(Tools, attr)
                if callable(method):
                    try:
                        tools_source += "\n" + inspect.getsource(method)
                    except Exception:
                        pass
        tool_instructions = (
            "At each turn, if you decide to invoke any of the function(s), wrap the function call in triple backticks with the label `tool_code`.\n\n"
            "Review the following Python methods (source provided for context) to determine if a function call is appropriate:\n\n"
            "```python\n" + tools_source + "\n```\n\n"
            "When a function call is executed, its response will be wrapped in triple backticks with the label `tool_output`."
        )
        system_message = {"role": "system", "content": system_prompt + "\n\n" + tool_instructions}
        
        # Retrieve memory context based on the current query embedding
        # For this payload, we assume that the last user message (if available) is our query
        if self.history_manager.history:
            last_user_msg = next((msg["content"] for msg in reversed(self.history_manager.history) if msg["role"] == "user"), "")
            query_embedding = Utils.embed_text(last_user_msg)
            current_mode = self.mode_manager.detect_mode(self.history_manager.history)
            # Update display state with current mode
            with display_state.lock:
                display_state.current_mode = current_mode
            memory_items = self.memory_manager.retrieve_similar(self.conversation_id, query_embedding, top_n=3, mode=current_mode)
            mem_context = "\n".join([f"[{ts}] {role}: {content}" for (_, _, ts, role, content) in memory_items])
        else:
            current_mode = "conversational"
            mem_context = ""
        
        # Also retrieve the latest summary narrative (if any)
        summary = self.memory_manager.retrieve_latest_summary(self.conversation_id)
        if summary:
            summary_text, narrative_state, sum_ts = summary
        else:
            summary_text = ""
        
        memory_context = f"Memory Context:\n{mem_context}\n\nSummary Narrative:\n{summary_text}\n"
        memory_message = {"role": "system", "content": memory_context}
        
        messages = [system_message, memory_message] + self.history_manager.history
        payload = {
            "model": cfg["model"],
            "messages": messages,
            "stream": cfg["stream"]
        }
        if self.format_schema:
            payload["format"] = self.format_schema
        if cfg["raw"]:
            payload["raw"] = True
        if cfg["images"]:
            if self.history_manager.history and self.history_manager.history[-1].get("role") == "user":
                self.history_manager.history[-1]["images"] = cfg["images"]
        if self.tools_data:
            payload["tools"] = self.tools_data
        if cfg["options"]:
            payload["options"] = cfg["options"]
        return payload

    def chat_completion_stream(self, processed_text):
        payload = self.build_payload()
        tokens = ""
        try:
            stream = chat(model=self.config_manager.config["model"],
                          messages=payload["messages"],
                          stream=self.config_manager.config["stream"])
            for part in stream:
                if self.stop_flag:
                    yield "", True
                    return
                content = part["message"]["content"]
                done = part.get("done", False)
                tokens += content
                with display_state.lock:
                    display_state.current_tokens = tokens
                yield content, done
                if done:
                    break
        except Exception:
            yield "", True

    def chat_completion_nonstream(self, processed_text):
        payload = self.build_payload()
        try:
            response = chat(model=self.config_manager.config["model"],
                            messages=payload["messages"],
                            stream=False)
            return response["message"]["content"]
        except Exception:
            return ""

    def process_text(self, text, skip_tts=False):
        processed_text = Utils.convert_numbers_to_words(text)
        sentence_endings = re.compile(r'[.?!]+')
        tokens = ""
        if self.config_manager.config["stream"]:
            buffer = ""
            for content, done in self.chat_completion_stream(processed_text):
                buffer += content
                tokens += content
                with display_state.lock:
                    display_state.current_tokens = tokens
                while True:
                    match = sentence_endings.search(buffer)
                    if not match:
                        break
                    end_index = match.end()
                    sentence = buffer[:end_index].strip()
                    buffer = buffer[end_index:].lstrip()
                    if sentence and not skip_tts:
                        threading.Thread(target=self.tts_manager.enqueue, args=(sentence,), daemon=True).start()
                if done:
                    break
            if buffer.strip():
                tokens += buffer.strip()
                with display_state.lock:
                    display_state.current_tokens = tokens
            return tokens
        else:
            result = self.chat_completion_nonstream(processed_text)
            tokens = result
            with display_state.lock:
                display_state.current_tokens = tokens
            return tokens

    def inference_thread(self, user_message, result_holder, skip_tts):
        result = self.process_text(user_message, skip_tts)
        result_holder.append(result)

    def run_inference(self, prompt, skip_tts=False):
        result_holder = []
        with self.inference_lock:
            if self.current_thread and self.current_thread.is_alive():
                self.stop_flag = True
                self.current_thread.join()
                self.stop_flag = False
            self.tts_manager.stop()
            self.tts_manager.start()
            self.current_thread = threading.Thread(
                target=self.inference_thread,
                args=(prompt, result_holder, skip_tts)
            )
            self.current_thread.start()
        self.current_thread.join()
        return result_holder[0] if result_holder else ""

    def run_tool(self, tool_code):
        allowed_tools = {}
        for attr in dir(Tools):
            if not attr.startswith("_"):
                method = getattr(Tools, attr)
                if callable(method):
                    allowed_tools[attr] = method
        try:
            result = eval(tool_code, {"__builtins__": {}}, allowed_tools)
            return str(result)
        except Exception as e:
            return f"Error executing tool: {e}"

    def new_request(self, user_message, skip_tts=False):
        # Add user message to history and store it in memory (with embedding)
        self.history_manager.add_entry("user", user_message)
        user_embedding = Utils.embed_text(user_message)
        self.memory_manager.store_message(self.config_manager.config["conversation_id"], "user", user_message, user_embedding)
        with display_state.lock:
            display_state.current_request = user_message
            display_state.current_tool_calls = ""
        result = self.run_inference(user_message, skip_tts)
        tool_code = Tools.parse_tool_call(result)
        if tool_code:
            tool_output = self.run_tool(tool_code)
            formatted_output = f"```tool_output\n{tool_output}\n```"
            combined_prompt = f"{user_message}\n{formatted_output}"
            self.history_manager.add_entry("user", combined_prompt)
            # Store the combined prompt in memory as well
            combined_embedding = Utils.embed_text(combined_prompt)
            self.memory_manager.store_message(self.config_manager.config["conversation_id"], "user", combined_prompt, combined_embedding)
            final_result = self.new_request(combined_prompt, skip_tts=False)
            return final_result
        else:
            self.history_manager.add_entry("assistant", result)
            assistant_embedding = Utils.embed_text(result)
            self.memory_manager.store_message(self.config_manager.config["conversation_id"], "assistant", result, assistant_embedding)
            return result

#############################################
# Display State and Manager (using curses)
#############################################
class DisplayState:
    def __init__(self):
        self.current_request = ""
        self.current_tokens = ""
        self.current_tool_calls = ""
        self.tts_flag = False
        self.tts_playing = False
        self.chat_history_state = "Idle"
        self.current_mode = "conversational"
        self.memory_list = []  # List of retrieved memory items
        self.lock = threading.Lock()

class DisplayManager:
    def __init__(self, display_state: DisplayState, history_manager: HistoryManager, memory_manager: MemoryManager):
        self.display_state = display_state
        self.history_manager = history_manager
        self.memory_manager = memory_manager

    def curses_display(self, stdscr):
        curses.curs_set(0)
        stdscr.nodelay(True)
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_YELLOW, -1)  # Function call output
        curses.init_pair(2, curses.COLOR_CYAN, -1)    # Assistant response
        curses.init_pair(3, curses.COLOR_WHITE, -1)   # User input
        curses.init_pair(4, curses.COLOR_MAGENTA, -1) # Memory info / mode

        while True:
            stdscr.erase()
            max_height, max_width = stdscr.getmaxyx()
            with self.display_state.lock:
                current_request = self.display_state.current_request
                current_tokens = self.display_state.current_tokens
                current_tool_calls = self.display_state.current_tool_calls
                current_mode = self.display_state.current_mode
                memory_list = self.display_state.memory_list

            # Header: show current mode
            header = f"[Mode: {current_mode}]"
            try:
                stdscr.addnstr(0, 0, header, max_width, curses.color_pair(4) | curses.A_BOLD)
            except curses.error:
                pass

            # Display user input
            top_line = 2
            wrapped_input = textwrap.wrap(current_request, width=max_width)
            for line in wrapped_input:
                try:
                    stdscr.addnstr(top_line, 0, f"User: {line}", max_width, curses.color_pair(3))
                except curses.error:
                    pass
                top_line += 1
            top_line += 1

            # Display function/tool output if any.
            if current_tool_calls:
                try:
                    stdscr.addnstr(top_line, 0, "Function Call:", max_width, curses.color_pair(1) | curses.A_BOLD)
                except curses.error:
                    pass
                top_line += 1
                wrapped_tool = textwrap.wrap(current_tool_calls, width=max_width)
                for line in wrapped_tool:
                    if top_line >= max_height:
                        break
                    try:
                        stdscr.addnstr(top_line, 0, line, max_width, curses.color_pair(1))
                    except curses.error:
                        pass
                    top_line += 1
                top_line += 1

            # Display assistant response.
            try:
                stdscr.addnstr(top_line, 0, "Response:", max_width, curses.color_pair(2) | curses.A_BOLD)
            except curses.error:
                pass
            top_line += 1
            wrapped_response = textwrap.wrap(current_tokens, width=max_width)
            for line in wrapped_response:
                if top_line >= max_height:
                    break
                try:
                    stdscr.addnstr(top_line, 0, line, max_width, curses.color_pair(2))
                except curses.error:
                    pass
                top_line += 1

            # Display retrieved memory context
            mem_top = top_line + 1
            try:
                stdscr.addnstr(mem_top, 0, "Retrieved Memory:", max_width, curses.color_pair(4) | curses.A_BOLD)
            except curses.error:
                pass
            mem_top += 1
            for mem in memory_list:
                # Each memory: (similarity, id, timestamp, role, content)
                sim, msg_id, ts, role, content = mem
                mem_str = f"[{ts}] ({role}) (score:{sim:.2f}): {content[:80]}..."
                try:
                    stdscr.addnstr(mem_top, 0, mem_str, max_width, curses.color_pair(4))
                except curses.error:
                    pass
                mem_top += 1
                if mem_top >= max_height:
                    break

            stdscr.refresh()
            time.sleep(0.5)

#############################################
# Interactive CLI
#############################################
class InteractiveCLI:
    def __init__(self, chat_manager: ChatManager, config_manager: ConfigManager):
        self.chat_manager = chat_manager
        self.config_manager = config_manager

    def interactive_loop(self):
        while True:
            try:
                cmd = input("[Interactive] > ").strip()
            except EOFError:
                break
            if not cmd:
                continue
            if cmd.startswith("/send "):
                message = cmd[len("/send "):].strip()
                self.chat_manager.new_request(message)
            elif cmd.startswith("/model "):
                new_model = cmd[len("/model "):].strip()
                self.config_manager.config["model"] = new_model
                ModelManager(self.config_manager).ensure_ollama_and_model()
                self.config_manager.update_config()
            elif cmd.startswith("/history_depth "):
                try:
                    depth = int(cmd[len("/history_depth "):].strip())
                    self.config_manager.config["history_depth"] = depth
                    self.config_manager.update_config()
                except ValueError:
                    pass
            elif cmd == "/quit":
                break

    def start(self):
        thread = threading.Thread(target=self.interactive_loop, daemon=True)
        thread.start()

#############################################
# Server Manager (TCP Server)
#############################################
class ServerManager:
    def __init__(self, host, port, chat_manager: ChatManager):
        self.host = host
        self.port = port
        self.chat_manager = chat_manager
        self.client_threads = []
        self.client_threads_lock = threading.Lock()

    def handle_client_connection(self, client_socket, address):
        try:
            data = client_socket.recv(65536)
            if not data:
                return
            user_message = data.decode('utf-8').strip()
            if not user_message:
                return
            result = self.chat_manager.new_request(user_message)
            client_socket.sendall(result.encode('utf-8'))
        except Exception:
            pass
        finally:
            client_socket.close()

    def start_server(self):
        self.chat_manager.tts_manager.start()
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            server.bind((self.host, self.port))
        except Exception:
            server.bind(('0.0.0.0', 64162))
        server.listen(5)
        while True:
            try:
                client_sock, addr = server.accept()
                client_thread = threading.Thread(
                    target=self.handle_client_connection,
                    args=(client_sock, addr),
                    daemon=True
                )
                client_thread.start()
                with self.client_threads_lock:
                    self.client_threads.append(client_thread)
            except KeyboardInterrupt:
                break
            except Exception:
                pass
        server.close()
        self.chat_manager.tts_manager.stop()
        with self.client_threads_lock:
            for t in self.client_threads:
                t.join()

#############################################
# Main Application Class
#############################################
class MainApp:
    def __init__(self):
        parser = argparse.ArgumentParser(description="Ollama Chat Server with TTS, Memory, Mode Switching, and advanced features.")
        parser.add_argument("--model", type=str, help="Model name to use.")
        parser.add_argument("--stream", action="store_true", help="Enable streaming responses from the model.")
        parser.add_argument("--format", type=str, help="Structured output format: 'json' or path to JSON schema file.")
        parser.add_argument("--system", type=str, help="System message override.")
        parser.add_argument("--raw", action="store_true", help="If set, use raw mode (no template).")
        parser.add_argument("--history", type=str, nargs='?', const="chat404.json",
                            help="Path to a JSON file containing conversation history messages.")
        parser.add_argument("--images", type=str, nargs='*', help="List of base64-encoded image files.")
        parser.add_argument("--tools", type=str, help="Path to a JSON file defining tools.")
        parser.add_argument("--option", action="append", help="Additional model parameters (e.g. --option temperature=0.7)")
        self.args = parser.parse_args()

        self.config_manager = ConfigManager()
        self.config_manager.merge_args(self.args)
        self.history_manager = HistoryManager(self.config_manager.config.get("history", "chat404.json"))
        self.tools_data = Utils.safe_load_json_file(self.config_manager.config.get("tools"), None)
        self.format_schema = Utils.load_format_schema(self.config_manager.config.get("format"))
        self.tts_manager = TTSManager(self.config_manager.config.get("tts_url"))
        # Initialize MemoryManager and ModeManager
        self.memory_manager = MemoryManager(db_path="memory.db")
        self.mode_manager = ModeManager(model_id="llama3.2:3b")
        self.chat_manager = ChatManager(self.config_manager, self.history_manager,
                                        self.tts_manager, self.tools_data, self.format_schema,
                                        self.memory_manager, self.mode_manager)
        global display_state
        display_state = DisplayState()
        self.display_manager = DisplayManager(display_state, self.history_manager, self.memory_manager)
        self.server_manager = ServerManager(
            self.config_manager.config.get("host", "0.0.0.0"),
            self.config_manager.config.get("port", 64162),
            self.chat_manager
        )
        self.interactive_cli = InteractiveCLI(self.chat_manager, self.config_manager)
        ModelManager(self.config_manager).ensure_ollama_and_model()
        self.config_monitor_thread = threading.Thread(target=self.config_manager.monitor_config, daemon=True)
        self.script_monitor_thread = threading.Thread(target=Utils.monitor_script, daemon=True)

    def run(self, stdscr):
        curses_thread = threading.Thread(target=self.display_manager.curses_display, args=(stdscr,), daemon=True)
        curses_thread.start()
        self.config_monitor_thread.start()
        self.script_monitor_thread.start()
        self.interactive_cli.start()
        server_thread = threading.Thread(target=self.server_manager.start_server, daemon=True)
        server_thread.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass

#############################################
# Main Execution
#############################################
if __name__ == "__main__":
    curses.wrapper(MainApp().run)
