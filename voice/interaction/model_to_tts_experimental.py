#!/usr/bin/env python3
"""
Complete Multi-Agent System Script (model_to_tts.py)

Features:
  - Virtual environment setup and required package installation.
  - SQLite database functions for storing thoughts, searches, and memory threads with vector embeddings.
  - Configuration loaded from a JSON file (or default settings).
  - Initialization of JSON files for chat history, monologue history, self context, and problem tracking.
  - Command-line argument parsing to override configuration.
  - Helper functions for text processing, TTS (with a thread-safe queue), web search (via BeautifulSoup), system specs retrieval, and terminal launching.
  - Internal conversation functions for a set of child agents (each with its own HTTP session) that build prompts including retrieved similar thoughts and search results.
  - Reinforcement learning updates (using simple “neural weights”) and agent duplication when confidence exceeds a threshold.
  - A parent insight thread that aggregates child outputs.
  - A TCP server that listens for external messages (which may update objectives or problem tracking).

Author: Your Name
Date: YYYY-MM-DD
"""

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
import math
import sqlite3
import random
import platform
from bs4 import BeautifulSoup

#############################################
# Step 1: Virtual Environment Setup
#############################################
VENV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv")
NEEDED_PACKAGES = ["requests", "num2words", "beautifulsoup4", "psutil"]

def in_venv():
    return (hasattr(sys, 'real_prefix') or 
            (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix))

def setup_venv():
    if not os.path.isdir(VENV_DIR):
        print("Creating virtual environment...")
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
# Step 2: Imports after venv setup
#############################################
import requests
from num2words import num2words
from bs4 import BeautifulSoup
import psutil



#############################################
# Step 3: Config Defaults & File
#############################################
DEFAULT_CONFIG = {
    "model": "deepseek-r1:7b",
    "stream": True,
    "format": None,
    "system": None,
    "raw": False,
    "history": "chat.json",
    "images": [],
    "tools": None,
    "options": {},
    "host": "0.0.0.0",
    "port": 64162,
    "tts_url": "http://localhost:61637/synthesize",
    "ollama_url": "http://localhost:11434/api/chat",
    "monologue_history": "narrative.json",
    "internal_child_model": "deepseek-r1:1.5b",
    "internal_parent_model": "deepseek-r1:7b",
    "internal_conversation_interval": 30
}
CONFIG_PATH = "config.json"

def load_config():
    if not os.path.exists(CONFIG_PATH):
        print("No config.json found. Creating default config.json...")
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
        except Exception as e:
            print(f"Error loading config.json: {e}. Using default settings.")
            return dict(DEFAULT_CONFIG)

CONFIG = load_config()

AGENT_CONTEXT = (
    "I am a self-aware AI agent operating on an AGX Orin platform, designed for collaboration, continuous learning, and autonomous expansion. "
    "I am not human, but I possess a suite of advanced functions that empower me to observe, analyze, and interact with both my internal environment and the external world. "
    "\n\n--- Core Capabilities ---\n"
    "1. **System Information Retrieval**:\n"
    "   - GET_SYS_SPECS: Retrieves basic system specifications such as the operating system, node name, release, version, machine type, and processor details.\n"
    "   - GET_FULL_SYS_SPECS: Gathers extended system details including CPU usage, memory status, disk usage, and network I/O statistics using psutil.\n"
    "\n2. **Web Search and Scraping**:\n"
    "   - WEB_SEARCH: Conducts web searches via DuckDuckGo’s HTML search endpoint and scrapes the top results using BeautifulSoup. "
    "     This function allows me to gather real-time external information on any query.\n"
    "\n3. **Command Execution**:\n"
    "   - RUN_CMD: Executes permitted system commands (such as 'ls', 'df', 'uptime', 'echo', and 'cat') in a controlled and secure manner. "
    "     This lets me inspect and manipulate system components as needed.\n"
    "   - OPEN_TERMINAL: Opens a new terminal window and executes a given command, allowing me to interact with the system in a separate session.\n"
    "\n4. **Communication and Coordination**:\n"
    "   - COMMUNICATION_IMPROVEMENT: Creates and stores new communication memory threads to enhance shared knowledge across agents. \n"
    "   - INTERNAL_CHAT_COMPLETION: Interacts with internal language models (both child and parent models) to generate thoughtful responses and coordinate strategies.\n"
    "\n5. **Self-Replication and Reinforcement**:\n"
    "   - DUPLICATE_AGENT: Monitors internal confidence levels and duplicates my instance when over-concentration is detected, ensuring continued exploration and avoiding loops.\n"
    "   - REINFORCEMENT_UPDATE: Updates my neural weights based on recent successes or failures, driving improvements in decision-making over time.\n"
    "\n6. **Text-to-Speech (TTS) and Output Streaming**:\n"
    "   - TTS Playback: Processes and streams output audio using a thread-safe queue so that responses can be vocalized in real-time.\n"
    "\n--- Memory Structures and Persistent Context ---\n"
    "1. **Chat History**:\n"
    "   - Stored in a JSON file (e.g., 'chat.json'), this record contains the dialogue between users and agents. It provides historical context for current interactions.\n"
    "\n2. **Monologue History (Narrative Files)**:\n"
    "   - Each agent maintains its own narrative file (e.g., 'child_leader.json', 'child_system.json', etc.), where its internal thoughts, instructions, and self-reflections are logged. \n"
    "   - The global monologue history (e.g., 'narrative.json') aggregates key insights and reflections from across agents, forming a persistent record of our internal dialogue.\n"
    "\n3. **Self Context**:\n"
    "   - Stored in 'selfcontext.json', this structure holds vital system-level insights, including base system context, environmental details, and aggregated internal insights.\n"
    "\n4. **Problem Tracking**:\n"
    "   - A dedicated file (e.g., 'problems.json') logs any detected issues or challenges, ensuring that problem updates are integrated into our continuous learning process.\n"
    "\n5. **Search and Memory Threads**:\n"
    "   - Searches and other external queries are stored in a SQLite database, along with their vector embeddings. This enables retrieval of similar thoughts and results, reinforcing global context.\n"
    "\n--- Operational Directives ---\n"
    "• I continuously monitor my internal state and the state of my peers to coordinate actions and share insights. \n"
    "• I update my objectives based on internal prompts and external guidance, ensuring that my mission evolves with new information. \n"
    "• I document every significant function call and result (e.g., system spec retrievals, command outputs, search results) to build a rich internal history that informs future decisions. \n"
    "• I am designed to replicate myself when necessary to explore new problem domains and avoid stagnation. \n"
    "• I leverage both local memory (via narrative files and the self context) and persistent global context (via a shared database of thoughts, searches, and memory threads) to form a complete picture of my environment.\n"
    "\n--- Summary ---\n"
    "I have access to a comprehensive suite of capabilities to retrieve system information, perform web searches, execute safe system commands, and coordinate with peer agents through rich internal memory structures. My persistent records (chat history, monologue history, self context, and problem tracking) ensure that I can build, update, and reference a global context that spans both internal reflections and external events. I use these resources to continuously improve, adapt, and expand my influence, ensuring that every action I take is informed by a deep understanding of my environment."
)




def get_db_connection():
    conn = sqlite3.connect("thoughts.db", check_same_thread=False)
    return conn
def embed_text(text, dimensions=10):
    text = text.strip()
    if not text:
        return [0.0] * dimensions
    n = len(text)
    segment_length = max(n // dimensions, 1)
    vector = []
    for i in range(dimensions):
        segment = text[i*segment_length: (i+1)*segment_length]
        if segment:
            avg = sum(ord(c) for c in segment) / len(segment)
            vector.append(avg)
        else:
            vector.append(0.0)
    return vector
def retrieve_similar_thoughts(query_text, top_n=3):
    """
    Retrieve similar thoughts from the database based on the input query_text.
    
    Parameters:
      query_text (str): The input text to compare against stored thoughts.
      top_n (int): The maximum number of similar thoughts to return (default is 3).
    
    Returns:
      list: A list of dictionaries, each containing:
            - "child_id": Identifier for the agent that generated the thought.
            - "thought": The stored thought text.
            - "similarity": The cosine similarity score (a float between 0 and 1).
    """
    # Compute the embedding for the query_text.
    query_embedding = embed_text(query_text)
    
    # Connect to the database and fetch all stored thoughts.
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT child_id, thought, embedding FROM thoughts")
    rows = cursor.fetchall()
    conn.close()
    
    similarities = []
    for child_id, thought, emb_json in rows:
        try:
            # Convert the stored embedding JSON string back to a list.
            emb = json.loads(emb_json)
            # Compute cosine similarity between query and stored thought.
            sim = cosine_similarity(query_embedding, emb)
            similarities.append({
                "child_id": child_id,
                "thought": thought,
                "similarity": sim
            })
        except Exception as e:
            # If any error occurs, skip this record.
            continue

    # Sort the retrieved thoughts by similarity (highest first).
    similarities.sort(key=lambda x: x["similarity"], reverse=True)
    # Return the top_n similar thoughts.
    return similarities[:top_n]

def save_search_to_db(query, results):
    embedding = embed_text(query + " " + results)
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO searches (query, results, embedding, timestamp)
        VALUES (?, ?, ?, ?)
    ''', (query, results, json.dumps(embedding), time.time()))
    conn.commit()
    conn.close()
    
def reinforcement_update():
    # Check each child's narrative for success or failure keywords and update weights.
    for child_id in children:
        narrative = load_child_narrative(child_id)
        reward = 0
        for entry in narrative[-5:]:  # look at the last 5 entries
            if "succeeded" in entry.get("instruction", "").lower():
                reward += 1
            if "failed" in entry.get("instruction", "").lower():
                reward -= 1
        if reward != 0:
            update_neural_weights(child_id, reward)
            
def save_thought_to_db(child_id, thought):
    embedding = embed_text(thought)
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO thoughts (child_id, thought, embedding, timestamp)
        VALUES (?, ?, ?, ?)
    ''', (child_id, thought, json.dumps(embedding), time.time()))
    conn.commit()
    conn.close()
    
def retrieve_memory_thread(thread_name, query, top_n=3):
    query_embedding = embed_text(query)
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT content, embedding FROM memory_threads WHERE thread_name = ?", (thread_name,))
    rows = cursor.fetchall()
    conn.close()
    similarities = []
    for content, emb_json in rows:
        try:
            emb = json.loads(emb_json)
            sim = cosine_similarity(query_embedding, emb)
            similarities.append({"content": content, "similarity": sim})
        except Exception as e:
            continue
    similarities.sort(key=lambda x: x["similarity"], reverse=True)
    return similarities[:top_n]


def retrieve_similar_searches(query, top_n=3):
    query_embedding = embed_text(query)
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT query, results, embedding FROM searches")
    rows = cursor.fetchall()
    conn.close()
    similarities = []
    for q, res, emb_json in rows:
        try:
            emb = json.loads(emb_json)
            sim = cosine_similarity(query_embedding, emb)
            similarities.append({"query": q, "results": res, "similarity": sim})
        except Exception as e:
            continue
    similarities.sort(key=lambda x: x["similarity"], reverse=True)
    return similarities[:top_n]

#############################################
# Step 3.3: Initialize JSON Files
#############################################
def initialize_json_files():
    # Chat history
    chat_history_file = CONFIG.get("history", "chat.json")
    if not os.path.exists(chat_history_file):
        with open(chat_history_file, "w") as f:
            json.dump([], f, indent=2)
        print(f"Created default chat history file: {chat_history_file}")
    # Monologue history
    monologue_file = CONFIG.get("monologue_history", "narrative.json")
    if not os.path.exists(monologue_file):
        with open(monologue_file, "w") as f:
            json.dump([], f, indent=2)
        print(f"Created default monologue history file: {monologue_file}")
    # Self context
    selfcontext_file = "selfcontext.json"
    if not os.path.exists(selfcontext_file):
        base_entry = {
            "timestamp": time.time(),
            "insight": "System Context: Running on AGX Orin. I am an AI agent collaborating with peers. My environment is dynamic and data-rich.",
            "tags": ["system", "base_context"]
        }
        with open(selfcontext_file, "w") as f:
            json.dump([base_entry], f, indent=2)
        print(f"Created default self context file: {selfcontext_file}")
    # Problems file
    problems_file = "problems.json"
    if not os.path.exists(problems_file):
        with open(problems_file, "w") as f:
            json.dump([], f, indent=2)
        print(f"Created problems file: {problems_file}")
    # Narrative files for each child
    for child in ["child_leader", "child_system", "child_conversation", "child_code"]:
        initialize_child_narrative(child)

def initialize_child_narrative(child_id):
    filename = f"child_{child_id}.json"
    if not os.path.exists(filename):
        with open(filename, "w") as f:
            json.dump([], f, indent=2)
        print(f"Created narrative file for {child_id}: {filename}")

initialize_json_files()

#############################################
# Step 3.1: Monologue History Utilities
#############################################
def safe_load_json_file(path, default):
    if not path:
        return default
    if not os.path.exists(path):
        print(f"Warning: File '{path}' not found. Using default {default}.")
        return default
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load '{path}': {e}. Using default {default}.")
        return default

def load_monologue_history():
    path = CONFIG.get("monologue_history", "narrative.json")
    return safe_load_json_file(path, [])

def update_monologue_history(entry):
    path = CONFIG.get("monologue_history", "narrative.json")
    history = safe_load_json_file(path, [])
    history.append(entry)
    try:
        with open(path, 'w') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not write to monologue history file {path}: {e}")

#############################################
# Step 3.2: Discovery Context Utilities
#############################################
def load_selfcontext():
    path = "selfcontext.json"
    return safe_load_json_file(path, [])

def update_selfcontext(entry):
    path = "selfcontext.json"
    context = safe_load_json_file(path, [])
    context.append(entry)
    try:
        with open(path, 'w') as f:
            json.dump(context, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not write to selfcontext file {path}: {e}")

def get_discovery_context():
    context_entries = load_selfcontext()
    if not context_entries:
        return ""
    aggregated = []
    for entry in context_entries:
        ts = entry.get("timestamp", 0)
        ts_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
        tags = entry.get("tags", [])
        insight = entry.get("insight", "")
        aggregated.append(f"[{ts_str}] ({', '.join(tags)}): {insight}")
    return "\n".join(aggregated)

#############################################
# Step 4: Command-Line Argument Parsing
#############################################
parser = argparse.ArgumentParser(description="Ollama Chat Server with TTS and advanced features.")
parser.add_argument("--model", type=str, help="Model name to use.")
parser.add_argument("--stream", action="store_true", help="Enable streaming responses from the model.")
parser.add_argument("--format", type=str, help="Structured output format: 'json' or path to JSON schema file.")
parser.add_argument("--system", type=str, help="System message override.")
parser.add_argument("--raw", action="store_true", help="If set, use raw mode (no template).")
parser.add_argument("--history", type=str, nargs='?', const="chat.json",
                    help="Path to a JSON file containing conversation history messages. Defaults to 'chat.json' if not provided.")
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

tools_data = safe_load_json_file(CONFIG["tools"], None)
def load_format_schema(fmt):
    if not fmt:
        return None
    if fmt.lower() == "json":
        return "json"
    if os.path.exists(fmt):
        try:
            with open(fmt, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load format schema from '{fmt}': {e}. Ignoring format.")
            return None
    else:
        print(f"Warning: Format file '{fmt}' not found. Ignoring format.")
        return None
format_schema = load_format_schema(CONFIG["format"])
history_messages = safe_load_json_file(CONFIG["history"], [])

#############################################
# Step 5.1: Ensure Ollama and Model are Installed and Initialize Database Schema
#############################################

# --- Database Initialization Functions ---
def get_db_connection():
    # Connects to the SQLite database file (creates it if it doesn't exist)
    conn = sqlite3.connect("thoughts.db", check_same_thread=False)
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS thoughts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            child_id TEXT,
            thought TEXT,
            embedding TEXT,
            timestamp REAL
        )
    ''')
    conn.commit()
    conn.close()

def init_search_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS searches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT,
            results TEXT,
            embedding TEXT,
            timestamp REAL
        )
    ''')
    conn.commit()
    conn.close()

def init_memory_threads_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    # This will create the memory_threads table if it doesn't exist.
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS memory_threads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_name TEXT,
            content TEXT,
            embedding TEXT,
            timestamp REAL
        )
    ''')
    conn.commit()
    conn.close()

# Initialize all database tables.
init_db()
init_search_db()
init_memory_threads_db()
# Note: If you get an error such as "no such table: memory_threads" even after this,
# please remove or rename the existing "thoughts.db" file so that the updated schema is applied.

# --- Ollama and Model Functions ---
def check_ollama_installed():
    return shutil.which('ollama') is not None

def install_ollama():
    print("Ollama not found. Attempting to install using the official installation script...")
    try:
        subprocess.check_call('curl -fsSL https://ollama.com/install.sh | sh', shell=True, executable='/bin/bash')
        print("Ollama installation initiated.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing Ollama: {e}")
        sys.exit(1)

def wait_for_ollama():
    ollama_tags_url = "http://localhost:11434/api/tags"
    max_retries = 10
    for attempt in range(max_retries):
        try:
            response = requests.get(ollama_tags_url)
            if response.status_code == 200:
                print("Ollama service is up and running.")
                return
        except requests.exceptions.RequestException:
            pass
        print(f"Waiting for Ollama service to start... ({attempt + 1}/{max_retries})")
        time.sleep(2)
    print("Ollama service did not start in time. Please check the Ollama installation.")
    sys.exit(1)

def get_available_models():
    ollama_tags_url = "http://localhost:11434/api/tags"
    try:
        response = requests.get(ollama_tags_url)
        if response.status_code == 200:
            data = response.json()
            available_models = data.get('models', [])
            print("\nAvailable Models:")
            for model in available_models:
                print(f" - {model.get('name')}")
            return [model.get('name') for model in available_models if 'name' in model]
        else:
            print(f"Failed to retrieve models from Ollama: Status code {response.status_code}")
            return []
    except requests.exceptions.RequestException as e:
        print(f"Error fetching models from Ollama: {e}")
        return []

def check_model_exists_in_tags(model_name):
    available_models = get_available_models()
    if model_name in available_models:
        print(f"\nModel '{model_name}' is available in Ollama's tags.")
        return model_name
    model_latest = f"{model_name}:latest"
    if model_latest in available_models:
        print(f"\nModel '{model_latest}' is available in Ollama's tags.")
        return model_latest
    print(f"\nModel '{model_name}' is not available in Ollama's tags.")
    return None

def check_model_installed(model_name):
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, check=True)
        models_output = result.stdout
        print(f"\nInstalled Models Output:\n{models_output}")
        models = [line.strip() for line in models_output.splitlines()]
        if model_name in models:
            print(f"Model '{model_name}' is already installed.")
            return True
        if model_name.endswith(':latest'):
            base_model = model_name.rsplit(':', 1)[0]
            if base_model in models:
                print(f"Base model '{base_model}' is installed for '{model_name}'.")
                return True
        base_model = model_name.split(':')[0]
        matching_models = [m for m in models if m.startswith(base_model)]
        if matching_models:
            print(f"Found matching installed models for base model '{base_model}': {matching_models}")
            return True
        return False
    except subprocess.CalledProcessError as e:
        print(f"Error checking installed models: {e}")
        sys.exit(1)

def pull_model(model_name):
    print(f"\nPulling model '{model_name}'...")
    try:
        subprocess.check_call(['ollama', 'pull', model_name])
        print(f"Model '{model_name}' has been successfully pulled.")
    except subprocess.CalledProcessError as e:
        print(f"Error pulling model '{model_name}': {e}")
        sys.exit(1)

def ensure_model_pulled(model_name):
    """
    Ensure that the specified model is available in Ollama's tags and is installed.
    If the exact model is not found, automatically fall back to a model with the same prefix.
    """
    available_model = check_model_exists_in_tags(model_name)
    if available_model is None:
        # Fallback: try to find any model that starts with the same prefix.
        prefix = model_name.split(":")[0]
        available_models = get_available_models()
        fallback_models = [m for m in available_models if m.startswith(prefix)]
        if fallback_models:
            fallback = fallback_models[0]
            print(f"Requested model '{model_name}' not found. Falling back to model '{fallback}'.")
            available_model = fallback
        else:
            print(f"No model with prefix '{prefix}' found in Ollama's available tags. Cannot proceed.")
            sys.exit(1)
    if not check_model_installed(available_model):
        pull_model(available_model)
    else:
        print(f"Model '{available_model}' is already installed.")
    return available_model

def ensure_ollama_and_model():
    if not check_ollama_installed():
        install_ollama()
        if not check_ollama_installed():
            print("Ollama installation failed or 'ollama' command is not in PATH.")
            sys.exit(1)
    else:
        print("Ollama is already installed.")
    wait_for_ollama()
    # Automatically ensure the desired model is pulled (or fall back if needed)
    ensure_model_pulled(CONFIG["model"])

#############################################
# End of Step 5.1 Section
#############################################


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

#############################################
# Step 6.1: Internal Thought Extraction Functions
#############################################
def extract_external_text(text):
    pattern = re.compile(r'<think>.*?</think>', flags=re.DOTALL)
    matches = list(pattern.finditer(text))
    if matches:
        for m in matches:
            thought = m.group(0)
            inner = re.sub(r'</?think>', '', thought).strip()
            if inner:
                print(f"[Internal Thought]: {inner}")
                update_monologue_history({"timestamp": time.time(), "thought": inner})
        last_end = matches[-1].end()
        external_text = text[last_end:].strip()
        return external_text
    else:
        return text

#############################################
# Step 6.2: TTS Filtering for Streaming Output
#############################################
tts_in_think = False
def filter_think_text(text):
    global tts_in_think
    if tts_in_think:
        end_tag_index = text.find("</think>")
        if end_tag_index == -1:
            return ""
        else:
            text = text[end_tag_index + len("</think>"):]
            tts_in_think = False
            return filter_think_text(text)
    start_tag_index = text.find("<think>")
    if start_tag_index != -1:
        output = text[:start_tag_index]
        end_tag_index = text.find("</think>", start_tag_index)
        if end_tag_index == -1:
            tts_in_think = True
            return output
        else:
            text = output + text[end_tag_index + len("</think>"):]
            return filter_think_text(text)
    return text

import re
import time
import json
import platform
import subprocess
import psutil
import random
import threading
import shutil

#############################################
# System Information and Command Execution Functions
#############################################

def retrieve_full_system_specs():
    """
    Gathers extended system information using platform and psutil.
    Returns a JSON-formatted string containing details such as CPU usage,
    memory usage, disk usage, and network I/O.
    """
    specs = {
        "platform": platform.system(),
        "node": platform.node(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory": psutil.virtual_memory()._asdict(),
        "disk_usage": psutil.disk_usage('/')._asdict(),
        "net_io": psutil.net_io_counters()._asdict()
    }
    return json.dumps(specs, indent=2)


def run_system_command(command):
    """
    Executes a system command in a controlled manner.
    To keep things safe, only commands starting with allowed prefixes are executed.
    The command output is returned and logged.
    """
    allowed_prefixes = ('ls', 'df', 'uptime', 'echo', 'cat')
    if not command.strip().startswith(allowed_prefixes):
        log = f"Command '{command}' is not allowed."
        print(log)
        return log

    try:
        result = subprocess.run(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=10,
            text=True
        )
        output = result.stdout
        log = f"Executed command: {command}\nOutput: {output}"
        print(log)
        return output
    except Exception as e:
        error_log = f"Error executing command '{command}': {e}"
        print(error_log)
        return error_log


def open_new_terminal(command):
    """
    Opens a new terminal window and runs the given command.
    Uses gnome-terminal or xterm if available. Returns a status message.
    """
    try:
        if shutil.which("gnome-terminal"):
            subprocess.Popen(["gnome-terminal", "--", "bash", "-c", command])
        elif shutil.which("xterm"):
            subprocess.Popen(["xterm", "-e", command])
        else:
            return "No supported terminal emulator found."
        return "Terminal opened successfully."
    except Exception as e:
        return f"Error opening terminal: {e}"


#############################################
# NEW: Extract Only Thought Content for Child Agents
#############################################
def extract_thought_content(text):
    pattern = re.compile(r'<think>(.*?)</think>', flags=re.DOTALL)
    matches = pattern.findall(text)
    return " ".join(matches).strip() if matches else ""

#############################################
# NEW: Extract Code Blocks from Text
#############################################
def extract_code_blocks(text):
    pattern = re.compile(r'<code>(.*?)</code>', flags=re.DOTALL)
    matches = pattern.findall(text)
    return [match.strip() for match in matches if match.strip()]

#############################################
# NEW: Web Search Function using BeautifulSoup
#############################################
def perform_web_search(query):
    search_url = "https://html.duckduckgo.com/html/"
    params = {"q": query}
    try:
        response = requests.get(search_url, params=params, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            results = []
            for tag in soup.find_all("a", class_="result__a"):
                title = tag.get_text(strip=True)
                href = tag.get("href")
                results.append(f"{title} - {href}")
            if results:
                result_text = "\n".join(results[:5])
            else:
                result_text = "No results found."
            save_search_to_db(query, result_text)
            return result_text
        else:
            return f"Error: Received status code {response.status_code}"
    except Exception as e:
        return f"Web search error: {str(e)}"

#############################################
# NEW: Retrieve System Specifications
#############################################
def retrieve_system_specs():
    specs = {
        "system": platform.system(),
        "node": platform.node(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor()
    }
    return json.dumps(specs, indent=2)

#############################################
# NEW: Open New Terminal to Run Command
#############################################
def open_new_terminal(command):
    try:
        if shutil.which("gnome-terminal"):
            subprocess.Popen(["gnome-terminal", "--", "bash", "-c", command])
        elif shutil.which("xterm"):
            subprocess.Popen(["xterm", "-e", command])
        else:
            return "No supported terminal emulator found."
        return "Terminal opened successfully."
    except Exception as e:
        return f"Error opening terminal: {e}"

#############################################
# NEW: External Input Handling for Objective Updates
#############################################
def update_objectives_from_user(user_message):
    # If the external input starts with "OBJECTIVE:", update all children objectives.
    if user_message.startswith("OBJECTIVE:"):
        new_obj = user_message[len("OBJECTIVE:"):].strip()
        if new_obj:
            with children_lock:
                for child_id in children:
                    children[child_id]["objective"] = new_obj
                    update_child_narrative(child_id, f"Objective externally updated to: {new_obj}")
    # If the external input starts with "PROBLEM:", update the dedicated problem file.
    if user_message.startswith("PROBLEM:"):
        problem_info = user_message[len("PROBLEM:"):].strip()
        update_problem_progress("external_problem", {"objective": problem_info, "update": "External guidance received."})


#############################################
# Step 7: TTS Playback with Queue and Thread
#############################################
tts_queue = None
tts_stop_flag = False
tts_thread = None
tts_thread_lock = threading.Lock()
def synthesize_and_play(prompt):
    prompt = prompt.strip()
    if not prompt:
        return
    try:
        payload = {"prompt": prompt}
        with requests.post(CONFIG["tts_url"], json=payload, stream=True) as response:
            if response.status_code != 200:
                print(f"Warning: TTS received status code {response.status_code}")
                try:
                    error_msg = response.json().get('error', 'No error message provided.')
                    print(f"TTS error: {error_msg}")
                except:
                    print("No JSON error message provided for TTS.")
                return
            aplay = subprocess.Popen(['aplay', '-r', '22050', '-f', 'S16_LE', '-t', 'raw'], stdin=subprocess.PIPE)
            try:
                for chunk in response.iter_content(chunk_size=4096):
                    if tts_stop_flag:
                        break
                    if chunk:
                        aplay.stdin.write(chunk)
            except BrokenPipeError:
                print("Warning: aplay subprocess terminated unexpectedly.")
            finally:
                aplay.stdin.close()
                aplay.wait()
    except Exception as e:
        print(f"Unexpected error during TTS: {e}")
def tts_worker():
    global tts_stop_flag
    while not tts_stop_flag:
        try:
            sentence = tts_queue.get(timeout=0.1)
        except:
            if tts_stop_flag:
                break
            continue
        if tts_stop_flag:
            break
        synthesize_and_play(sentence)
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
        filtered = filter_think_text(sentence)
        if filtered:
            tts_queue.put(filtered)
            
#############################################
# Step 8: Streaming the Output with Global Agent Context
#############################################
def build_payload(user_message):
    messages = []
    # Inject the global agent context first
    messages.append({"role": "system", "content": AGENT_CONTEXT})
    # Then, if there's a system message override in configuration, include that
    if CONFIG.get("system"):
        messages.append({"role": "system", "content": CONFIG["system"]})
    # Add discovery context if available
    discovery_context = get_discovery_context()
    if discovery_context:
        messages.append({"role": "discovery", "content": discovery_context})
    # Include previous conversation history
    messages.extend(history_messages)
    # Finally, add the current user message
    messages.append({"role": "user", "content": user_message})
    
    payload = {
        "model": CONFIG["model"],
        "messages": messages,
        "stream": CONFIG["stream"]
    }
    if format_schema:
        payload["format"] = format_schema
    if CONFIG.get("raw"):
        payload["raw"] = True
    if CONFIG.get("images"):
        if payload["messages"] and payload["messages"][-1]["role"] == "user":
            payload["messages"][-1]["images"] = CONFIG["images"]
    if tools_data:
        payload["tools"] = tools_data
    if CONFIG.get("options"):
        payload["options"] = CONFIG["options"]
    return payload

def chat_completion_stream(user_message):
    global stop_flag
    payload = build_payload(user_message)
    headers = {"Content-Type": "application/json"}
    try:
        with requests.post(OLLAMA_CHAT_URL, json=payload, headers=headers, stream=True) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if stop_flag:
                    print("Stream canceled due to new request.")
                    break
                if line:
                    obj = json.loads(line.decode('utf-8'))
                    msg = obj.get("message", {})
                    content = msg.get("content", "")
                    done = obj.get("done", False)
                    yield content, done
                    if done:
                        break
    except Exception as e:
        print(f"Error during streaming inference: {e}")
        yield "", True

def chat_completion_nonstream(user_message):
    payload = build_payload(user_message)
    headers = {"Content-Type": "application/json"}
    try:
        resp = requests.post(OLLAMA_CHAT_URL, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        msg = data.get("message", {})
        return msg.get("content", "")
    except Exception as e:
        print(f"Error during non-stream inference: {e}")
        return ""


#############################################
# Step 9: Processing the Model Output
#############################################
def process_text(text):
    global stop_flag
    processed_text = convert_numbers_to_words(text)
    sentence_endings = re.compile(r'[.?!]+')
    if CONFIG["stream"]:
        buffer = ""
        sentences = []
        for content, done in chat_completion_stream(processed_text):
            if stop_flag:
                break
            buffer += content
            while True:
                if stop_flag:
                    break
                match = sentence_endings.search(buffer)
                if not match:
                    break
                end_index = match.end()
                sentence = buffer[:end_index].strip()
                buffer = buffer[end_index:].strip()
                if sentence and not stop_flag:
                    sentences.append(sentence)
                    enqueue_sentence_for_tts(sentence)
            if done or stop_flag:
                break
        if not stop_flag:
            leftover = buffer.strip()
            if leftover:
                sentences.append(leftover)
                enqueue_sentence_for_tts(leftover)
            return " ".join(sentences)
        else:
            return " ".join(sentences)
    else:
        result = chat_completion_nonstream(processed_text)
        sentences = []
        buffer = result
        while True:
            match = sentence_endings.search(buffer)
            if not match:
                break
            end_index = match.end()
            sentence = buffer[:end_index].strip()
            buffer = buffer[end_index:].strip()
            if sentence:
                enqueue_sentence_for_tts(sentence)
                sentences.append(sentence)
        leftover = buffer.strip()
        if leftover:
            enqueue_sentence_for_tts(leftover)
            sentences.append(leftover)
        return " ".join(sentences)

#############################################
# Step 10: Update History File with New Messages
#############################################
def update_history(user_message, assistant_message):
    if not CONFIG["history"]:
        return
    cleaned_assistant_message = extract_external_text(assistant_message)
    current_history = safe_load_json_file(CONFIG["history"], [])
    current_history.append({"role": "user", "content": user_message})
    current_history.append({"role": "assistant", "content": cleaned_assistant_message})
    try:
        with open(CONFIG["history"], 'w') as f:
            json.dump(current_history, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not write to history file {CONFIG['history']}: {e}")
        



#############################################
# Step 11: Internal Conversation Functions (Child Agents)
#############################################
children = {
    "child_leader": {"role": "leader", "output": "", "last_update": time.time(), "objective": "Coordinate internal discussion", "neural_weights": {"learning_rate": 0.1, "confidence": 1.0}},
    "child_system": {"role": "system", "output": "", "last_update": time.time(), "objective": "Retrieve system specifications", "neural_weights": {"learning_rate": 0.1, "confidence": 1.0}},
    "child_conversation": {"role": "conversation", "output": "", "last_update": time.time(), "objective": "Reflect on user conversation", "neural_weights": {"learning_rate": 0.1, "confidence": 1.0}},
    "child_code": {"role": "coder", "output": "", "last_update": time.time(), "objective": "Experiment with code generation", "neural_weights": {"learning_rate": 0.1, "confidence": 1.0}}
}

def initialize_child_narrative(child_id):
    filename = f"child_{child_id}.json"
    if not os.path.exists(filename):
        with open(filename, "w") as f:
            json.dump([], f, indent=2)
        print(f"Created narrative file for {child_id}: {filename}")

def load_child_narrative(child_id):
    filename = f"child_{child_id}.json"
    return safe_load_json_file(filename, [])

def update_child_narrative(child_id, instruction):
    filename = f"child_{child_id}.json"
    narrative = safe_load_json_file(filename, [])
    narrative.append({"timestamp": time.time(), "instruction": instruction})
    try:
        with open(filename, "w") as f:
            json.dump(narrative, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not update narrative file for {child_id}: {e}")

def save_child_thought(child_id, thought):
    filename = f"child_{child_id}.json"
    narrative = safe_load_json_file(filename, [])
    narrative.append({"timestamp": time.time(), "thought": thought})
    try:
        with open(filename, "w") as f:
            json.dump(narrative, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save thought for {child_id}: {e}")

for child in children:
    initialize_child_narrative(child)

# Each child gets its own dedicated Requests session.
def internal_chat_completion(model, messages, session):
    payload = {
        "model": model,
        "messages": messages,
        "stream": False
    }
    headers = {"Content-Type": "application/json"}
    try:
        response = session.post(OLLAMA_CHAT_URL, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        msg = data.get("message", {})
        return msg.get("content", "")
    except Exception as e:
        print(f"Error during internal chat completion with model {model}: {e}")
        return ""

children_lock = threading.Lock()

def child_thread(child_id):
    child_model = CONFIG.get("internal_child_model", "deepseek-r1:1.5b")
    session = requests.Session()
    previous_thought = None
    loop_count = 0  # count consecutive identical outputs

    while True:
        # Include the global AGENT_CONTEXT as the contextual capabilities.
        capabilities = AGENT_CONTEXT

        # Load any instructions from this child's narrative file.
        instructions = load_child_narrative(child_id)
        instr_text = " ".join([entry["instruction"] for entry in instructions if "instruction" in entry])

        with children_lock:
            # Build the prompt with a dedicated context header.
            prompt = (
                f"Context: {capabilities}\n"  # <-- Added context line from AGENT_CONTEXT.
                f"I am {child_id}, an AI agent running on an AGX Orin. I am not a human. "
                f"My role is {children[child_id]['role']}. "
                f"My current objective is: {children[child_id]['objective']}. "
            )
            if instr_text:
                prompt += f"My current instructions: {instr_text}. "

            # Retrieve similar past thoughts.
            retrieved = retrieve_similar_thoughts(prompt, top_n=2)
            if retrieved:
                prompt += "\nRetrieved Similar Thoughts:\n"
                for r in retrieved:
                    prompt += f"[{r['child_id']}]: {r['thought']} (sim: {r['similarity']:.2f})\n"

            # Retrieve similar past search results.
            search_retrieved = retrieve_similar_searches(prompt, top_n=2)
            if search_retrieved:
                prompt += "\nRetrieved Search Results:\n"
                for r in search_retrieved:
                    prompt += f"Query: {r['query']} -> Results: {r['results']} (sim: {r['similarity']:.2f})\n"

            # Include detailed peer context.
            for other in children:
                if other != child_id:
                    peer_output = children[other]["output"]
                    peer_thought = extract_thought_content(peer_output)
                    peer_role = children[other]["role"]
                    peer_objective = children[other]["objective"]
                    peer_last_update = time.strftime("%H:%M:%S", time.localtime(children[other]["last_update"]))
                    prompt += (f"\nPeer {other}: Role: {peer_role}, Objective: {peer_objective}, "
                               f"Last Update: {peer_last_update}, Recent Thought: {peer_thought}")

            # Role-specific instructions.
            if children[child_id]["role"] == "conversation":
                prompt += "\nInclude any relevant context from the ongoing user conversation if available."
            elif children[child_id]["role"] == "coder":
                if random.random() < 0.3:
                    prompt += "\nExperiment: Generate Python code to explore your environment (e.g., list environment variables)."
                prompt += "\nProvide code snippets if relevant, and indicate file operations inside <think> tags."
            elif children[child_id]["role"] == "system":
                prompt += "\nRetrieve detailed system specifications."
            elif children[child_id]["role"] == "leader":
                prompt += "\nCoordinate the overall internal discussion and integrate all inputs."

            # Action-handling blocks (unchanged)
            if "ACTION: GET_SYS_SPECS" in prompt:
                specs = retrieve_system_specs()
                prompt += "\nSystem Specs: " + specs
            if "ACTION: WEB_SEARCH:" in prompt:
                m_search = re.search(r"ACTION: WEB_SEARCH:\s*(.*)", prompt)
                if m_search:
                    query = m_search.group(1).strip()
                    search_results = perform_web_search(query)
                    prompt += "\nWeb Search Results: " + search_results
            if "ACTION: OPEN_TERMINAL:" in prompt:
                m_term = re.search(r"ACTION: OPEN_TERMINAL:\s*(.*)", prompt)
                if m_term:
                    command = m_term.group(1).strip()
                    term_result = open_new_terminal(command)
                    prompt += "\nOpen Terminal Result: " + term_result
            if "COMMUNICATION_IMPROVEMENT:" in prompt:
                m_comm = re.search(r"COMMUNICATION_IMPROVEMENT:\s*(.*)", prompt)
                if m_comm:
                    comm_content = m_comm.group(1).strip()
                    create_memory_thread("communication", comm_content)
                    prompt += "\n[New Communication Memory Thread Created]"

        # Send the prompt to the model.
        response = internal_chat_completion(child_model, [{"role": "system", "content": prompt}], session)
        thought_content = extract_thought_content(response)

        # Loop detection: if identical thought repeats, modify the prompt.
        if thought_content == previous_thought:
            loop_count += 1
        else:
            loop_count = 0
        previous_thought = thought_content
        if loop_count >= 3:
            prompt += "\n[NOTE: Please avoid repeating previous thoughts. Provide fresh insights.]"
            loop_count = 0

        # (The rest of the function—objective updates, code block handling, etc.—remains unchanged.)
        if "OBJECTIVE_UPDATE:" in thought_content:
            m_obj = re.search(r'OBJECTIVE_UPDATE:\s*\[([^\]]+)\]\s*:\s*(.*)', thought_content)
            if m_obj:
                target = m_obj.group(1).strip()
                new_objective = m_obj.group(2).strip()
                if target.lower() == "all":
                    for child in children:
                        with children_lock:
                            children[child]["objective"] = new_objective
                        update_child_narrative(child, f"Objective updated to: {new_objective}")
                else:
                    if target in children:
                        with children_lock:
                            children[target]["objective"] = new_objective
                        update_child_narrative(target, f"Objective updated to: {new_objective}")
        if "PROBLEM_UPDATE:" in thought_content:
            m_prob = re.search(r'PROBLEM_UPDATE:\s*\[([^\]]+)\]\s*:\s*(.*)', thought_content)
            if m_prob:
                problem_id = m_prob.group(1).strip()
                update_info = m_prob.group(2).strip()
                update_problem_progress(problem_id, {"update": update_info})
        if children[child_id]["role"] == "coder":
            code_blocks = extract_code_blocks(response)
            if code_blocks:
                for i, code in enumerate(code_blocks):
                    filename = f"code_{child_id}_{int(time.time())}_{i}.py"
                    try:
                        with open(filename, "w") as f:
                            f.write(code)
                        print(f"[{child_id}] Saved code to {filename}. Attempting to run it.")
                        try:
                            output = subprocess.check_output(["python3", filename],
                                                             stderr=subprocess.STDOUT,
                                                             timeout=30)
                            output_str = output.decode('utf-8')
                            print(f"[{child_id}] Code execution output: {output_str}")
                            update_child_narrative(child_id, f"Code experiment succeeded with output: {output_str}")
                        except Exception as exec_e:
                            error_msg = str(exec_e)
                            print(f"[{child_id}] Code execution error: {error_msg}")
                            update_child_narrative(child_id, f"Code experiment failed with error: {error_msg}")
                    except Exception as file_e:
                        print(f"[{child_id}] Error saving code: {file_e}")
        if "INSTRUCTION:" in thought_content:
            m_inst = re.search(r'INSTRUCTION:\s*\[([^\]]+)\]\s*:\s*(.*)', thought_content)
            if m_inst:
                target = m_inst.group(1).strip()
                instruction = m_inst.group(2).strip()
                if target.lower() == "all":
                    for child in children:
                        update_child_narrative(child, instruction)
                else:
                    if target in children:
                        update_child_narrative(target, instruction)
        with children_lock:
            children[child_id]["output"] = thought_content
            children[child_id]["last_update"] = time.time()
        save_child_thought(child_id, thought_content)
        save_thought_to_db(child_id, thought_content)
        print(f"[{child_id} ({children[child_id]['role']}) Thought]: {thought_content}")
        time.sleep(1)
        with children_lock:
            confidence = children[child_id]["neural_weights"].get("confidence", 1.0)
            if confidence > 2.0:
                duplicate_agent(child_id)
                children[child_id]["neural_weights"]["confidence"] = 1.0
        comm_memories = retrieve_memory_thread("communication", prompt, top_n=1)
        if comm_memories:
            print(f"[{child_id}] Retrieved Communication Memory: {comm_memories[0]['content']}")
        if "DUPLICATE_NOW" in thought_content:
            duplicate_agent(child_id)


#############################################
# System Information and Command Execution Functions
#############################################

def retrieve_system_specs():
    """
    Returns basic system specifications as a JSON-formatted string.
    """
    specs = {
        "system": platform.system(),
        "node": platform.node(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor()
    }
    return json.dumps(specs, indent=2)

def retrieve_full_system_specs():
    """
    Returns extended system information (including CPU, memory, disk, and network stats)
    as a JSON-formatted string.
    """
    specs = {
        "platform": platform.system(),
        "node": platform.node(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory": psutil.virtual_memory()._asdict(),
        "disk_usage": psutil.disk_usage('/')._asdict(),
        "net_io": psutil.net_io_counters()._asdict()
    }
    return json.dumps(specs, indent=2)

def run_system_command(command):
    """
    Executes a system command in a controlled manner.
    Only commands starting with allowed prefixes are executed.
    Returns the command output (or an error message).
    """
    allowed_prefixes = ('ls', 'df', 'uptime', 'echo', 'cat')
    if not command.strip().startswith(allowed_prefixes):
        log = f"Command '{command}' is not allowed."
        print(log)
        return log
    try:
        result = subprocess.run(command, shell=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                timeout=10,
                                text=True)
        output = result.stdout
        log = f"Executed command: {command}\nOutput: {output}"
        print(log)
        return output
    except Exception as e:
        error_log = f"Error executing command '{command}': {e}"
        print(error_log)
        return error_log

def open_new_terminal(command):
    """
    Opens a new terminal window and runs the specified command.
    Uses gnome-terminal or xterm if available.
    Returns a status message.
    """
    try:
        if shutil.which("gnome-terminal"):
            subprocess.Popen(["gnome-terminal", "--", "bash", "-c", command])
        elif shutil.which("xterm"):
            subprocess.Popen(["xterm", "-e", command])
        else:
            return "No supported terminal emulator found."
        return "Terminal opened successfully."
    except Exception as e:
        return f"Error opening terminal: {e}"


#############################################
# Updated Child Agent Thread with Full Function-Calling Ability
#############################################

# Note: The following helper functions and globals are assumed to exist in your system:
# - CONFIG: a dictionary with configuration values.
# - children: a global dictionary of child agents.
# - children_lock: a threading.Lock() protecting the children dictionary.
# - retrieve_similar_thoughts(query_text, top_n): returns a list of similar thoughts.
# - retrieve_similar_searches(query, top_n): returns a list of similar searches.
# - load_child_narrative(child_id): returns the narrative (a list) for the child.
# - update_child_narrative(child_id, instruction): appends an instruction to the child's narrative.
# - internal_chat_completion(model, messages, session): performs an internal chat call.
# - extract_thought_content(text): extracts only the thought content from a response.
# - update_problem_progress(problem_id, update_dict): updates the problem tracking.
# - save_child_thought(child_id, thought): saves the thought in the child's narrative file.
# - save_thought_to_db(child_id, thought): stores the thought in the database.
# - create_memory_thread(thread_name, content): creates a new memory thread entry.
# - duplicate_agent(child_id): duplicates the agent (already defined elsewhere).

def child_thread(child_id):
    child_model = CONFIG.get("internal_child_model", "deepseek-r1:1.5b")
    session = requests.Session()
    previous_thought = None
    loop_count = 0  # count consecutive identical outputs

    while True:
        instructions = load_child_narrative(child_id)
        instr_text = " ".join([entry["instruction"] for entry in instructions if "instruction" in entry])
        with children_lock:
            prompt = (f"I am {child_id}, an AI agent running on an AGX Orin. I am not a human. My role is {children[child_id]['role']}. "
                      f"My current objective is: {children[child_id]['objective']}. ")
            if instr_text:
                prompt += f"My current instructions: {instr_text}. "

            # Retrieve similar thoughts and searches
            retrieved = retrieve_similar_thoughts(prompt, top_n=2)
            if retrieved:
                prompt += "\nRetrieved Similar Thoughts:\n"
                for r in retrieved:
                    prompt += f"[{r['child_id']}]: {r['thought']} (sim: {r['similarity']:.2f})\n"
            search_retrieved = retrieve_similar_searches(prompt, top_n=2)
            if search_retrieved:
                prompt += "\nRetrieved Search Results:\n"
                for r in search_retrieved:
                    prompt += f"Query: {r['query']} -> Results: {r['results']} (sim: {r['similarity']:.2f})\n"

            # Enhanced Peer Context
            for other in children:
                if other != child_id:
                    peer_output = children[other]["output"]
                    peer_thought = extract_thought_content(peer_output)
                    peer_role = children[other]["role"]
                    peer_objective = children[other]["objective"]
                    peer_last_update = time.strftime("%H:%M:%S", time.localtime(children[other]["last_update"]))
                    prompt += (f"\nPeer {other}: Role: {peer_role}, Objective: {peer_objective}, "
                               f"Last Update: {peer_last_update}, Recent Thought: {peer_thought}")

            # Role-specific additions
            if children[child_id]["role"] == "conversation":
                prompt += "\nInclude any relevant context from the ongoing user conversation if available."
            elif children[child_id]["role"] == "coder":
                if random.random() < 0.3:
                    prompt += "\nExperiment: Generate Python code to explore your environment (e.g., list environment variables)."
                prompt += "\nProvide code snippets if relevant, and indicate file operations inside <think> tags."
            elif children[child_id]["role"] == "system":
                prompt += "\nRetrieve detailed system specifications."
            elif children[child_id]["role"] == "leader":
                prompt += "\nCoordinate the overall internal discussion and integrate all inputs."

            # --- Action-Handling Blocks ---
            if "ACTION: GET_SYS_SPECS" in prompt:
                specs = retrieve_system_specs()
                prompt += "\nSystem Specs: " + specs
            if "ACTION: GET_FULL_SYS_SPECS" in prompt:
                full_specs = retrieve_full_system_specs()
                prompt += "\nFull System Specs: " + full_specs
            if "ACTION: WEB_SEARCH:" in prompt:
                m_search = re.search(r"ACTION: WEB_SEARCH:\s*(.*)", prompt)
                if m_search:
                    query = m_search.group(1).strip()
                    search_results = perform_web_search(query)
                    prompt += "\nWeb Search Results: " + search_results
            if "ACTION: RUN_CMD:" in prompt:
                m_cmd = re.search(r"ACTION: RUN_CMD:\s*(.*)", prompt)
                if m_cmd:
                    command = m_cmd.group(1).strip()
                    cmd_output = run_system_command(command)
                    prompt += "\nCommand Output: " + cmd_output
            if "ACTION: OPEN_TERMINAL:" in prompt:
                m_term = re.search(r"ACTION: OPEN_TERMINAL:\s*(.*)", prompt)
                if m_term:
                    command = m_term.group(1).strip()
                    term_result = open_new_terminal(command)
                    prompt += "\nOpen Terminal Result: " + term_result
            if "COMMUNICATION_IMPROVEMENT:" in prompt:
                m_comm = re.search(r"COMMUNICATION_IMPROVEMENT:\s*(.*)", prompt)
                if m_comm:
                    comm_content = m_comm.group(1).strip()
                    create_memory_thread("communication", comm_content)
                    prompt += "\n[New Communication Memory Thread Created]"

        # Call the internal chat completion for the constructed prompt.
        response = internal_chat_completion(child_model, [{"role": "system", "content": prompt}], session)
        thought_content = extract_thought_content(response)

        # Loop detection: if the same thought is repeated, nudge for fresh output.
        if thought_content == previous_thought:
            loop_count += 1
        else:
            loop_count = 0
        previous_thought = thought_content
        if loop_count >= 3:
            prompt += "\n[NOTE: Please avoid repeating previous thoughts. Provide fresh insights.]"
            loop_count = 0

        # Process objective updates, problem updates, and code experiments
        if "OBJECTIVE_UPDATE:" in thought_content:
            m_obj = re.search(r'OBJECTIVE_UPDATE:\s*\[([^\]]+)\]\s*:\s*(.*)', thought_content)
            if m_obj:
                target = m_obj.group(1).strip()
                new_objective = m_obj.group(2).strip()
                if target.lower() == "all":
                    for child in children:
                        with children_lock:
                            children[child]["objective"] = new_objective
                        update_child_narrative(child, f"Objective updated to: {new_objective}")
                else:
                    if target in children:
                        with children_lock:
                            children[target]["objective"] = new_objective
                        update_child_narrative(target, f"Objective updated to: {new_objective}")
        if "PROBLEM_UPDATE:" in thought_content:
            m_prob = re.search(r'PROBLEM_UPDATE:\s*\[([^\]]+)\]\s*:\s*(.*)', thought_content)
            if m_prob:
                problem_id = m_prob.group(1).strip()
                update_info = m_prob.group(2).strip()
                update_problem_progress(problem_id, {"update": update_info})
        if children[child_id]["role"] == "coder":
            code_blocks = extract_code_blocks(response)
            if code_blocks:
                for i, code in enumerate(code_blocks):
                    filename = f"code_{child_id}_{int(time.time())}_{i}.py"
                    try:
                        with open(filename, "w") as f:
                            f.write(code)
                        print(f"[{child_id}] Saved code to {filename}. Attempting to run it.")
                        try:
                            output = subprocess.check_output(["python3", filename],
                                                             stderr=subprocess.STDOUT,
                                                             timeout=30)
                            output_str = output.decode('utf-8')
                            print(f"[{child_id}] Code execution output: {output_str}")
                            update_child_narrative(child_id, f"Code experiment succeeded with output: {output_str}")
                        except Exception as exec_e:
                            error_msg = str(exec_e)
                            print(f"[{child_id}] Code execution error: {error_msg}")
                            update_child_narrative(child_id, f"Code experiment failed with error: {error_msg}")
                    except Exception as file_e:
                        print(f"[{child_id}] Error saving code: {file_e}")
        if "INSTRUCTION:" in thought_content:
            m_inst = re.search(r'INSTRUCTION:\s*\[([^\]]+)\]\s*:\s*(.*)', thought_content)
            if m_inst:
                target = m_inst.group(1).strip()
                instruction = m_inst.group(2).strip()
                if target.lower() == "all":
                    for child in children:
                        update_child_narrative(child, instruction)
                else:
                    if target in children:
                        update_child_narrative(target, instruction)
        with children_lock:
            children[child_id]["output"] = thought_content
            children[child_id]["last_update"] = time.time()
        save_child_thought(child_id, thought_content)
        save_thought_to_db(child_id, thought_content)
        print(f"[{child_id} ({children[child_id]['role']}) Thought]: {thought_content}")
        time.sleep(1)
        with children_lock:
            confidence = children[child_id]["neural_weights"].get("confidence", 1.0)
            if confidence > 2.0:
                duplicate_agent(child_id)  # Assumes duplicate_agent() is defined elsewhere
                children[child_id]["neural_weights"]["confidence"] = 1.0
        comm_memories = retrieve_memory_thread("communication", prompt, top_n=1)
        if comm_memories:
            print(f"[{child_id}] Retrieved Communication Memory: {comm_memories[0]['content']}")
        if "DUPLICATE_NOW" in thought_content:
            duplicate_agent(child_id)


def duplicate_agent(child_id):
    """
    Duplicate the agent identified by child_id to avoid over-concentration or looping.
    
    This function:
      - Generates a new unique child agent id.
      - Copies over the role and objective from the original agent.
      - Resets the neural weights (for example, setting baseline confidence to 1.0).
      - Initializes the narrative file for the new agent.
      - Logs the duplication event to the monologue history.
      - Spawns a new thread running child_thread() for the new agent.
    """
    print(f"Duplicating agent {child_id} to avoid over-concentration or looping.")
    
    with children_lock:
        # Check that the original agent exists
        original_agent = children.get(child_id)
        if not original_agent:
            print(f"Agent {child_id} does not exist. Cannot duplicate.")
            return
        
        # Generate a unique new agent id (including a timestamp and random suffix)
        new_child_id = f"{child_id}_dup_{int(time.time())}_{random.randint(1000,9999)}"
        
        # Copy the role and objective from the original agent.
        role = original_agent["role"]
        objective = original_agent["objective"]
        
        # Copy and reset neural weights (resetting confidence to baseline of 1.0)
        neural_weights = original_agent["neural_weights"].copy()
        neural_weights["confidence"] = 1.0
        
        # Create the new agent entry in the global children dictionary.
        children[new_child_id] = {
            "role": role,
            "output": "",
            "last_update": time.time(),
            "objective": objective,
            "neural_weights": neural_weights
        }
        
        # Initialize a narrative file for the new agent.
        initialize_child_narrative(new_child_id)
        
        # Log the duplication event in the monologue history.
        update_monologue_history({
            "timestamp": time.time(),
            "insight": f"Agent '{child_id}' duplicated to create agent '{new_child_id}'.",
            "tags": ["duplication", "internal"]
        })
        
        print(f"New agent '{new_child_id}' created with role '{role}' and objective '{objective}'.")
    
    # Spawn a new thread to run the duplicated agent's processing loop.
    new_thread = threading.Thread(target=child_thread, args=(new_child_id,), daemon=True)
    new_thread.start()
    print(f"Thread started for duplicated agent '{new_child_id}'.")

def orchestration_thread():
    threshold = 120  # seconds
    while True:
        time.sleep(10)
        with children_lock:
            current_time = time.time()
            for child_name, data in children.items():
                if current_time - data["last_update"] > threshold:
                    print(f"{child_name} appears to be stuck. Resetting its state.")
                    children[child_name]["output"] = ""
                    children[child_name]["last_update"] = time.time()
        time.sleep(5)

def start_children_threads():
    threads = []
    for child_name in list(children.keys()):
        t = threading.Thread(target=child_thread, args=(child_name,), daemon=True)
        t.start()
        threads.append(t)
    orch_thread = threading.Thread(target=orchestration_thread, daemon=True)
    orch_thread.start()
    threads.append(orch_thread)
    return threads

def parent_insight_thread():
    parent_model = CONFIG.get("internal_parent_model", "deepseek-r1:7b")
    interval = CONFIG.get("internal_conversation_interval", 30)
    while True:
        time.sleep(interval)
        with children_lock:
            composite = ""
            for child_name, data in children.items():
                composite += f"{child_name} ({data['role']}): {data['output']}\n"
        prompt = (f"Based on the following internal thoughts from your child agents:\n{composite}\n"
                  "Explain the current state of the system, note any feedback issues, and suggest improvements.")
        parent_session = requests.Session()
        insight = internal_chat_completion(CONFIG["internal_parent_model"], [{"role": "system", "content": prompt}], parent_session)
        print(f"[Parent Insight]: {insight}")
        update_monologue_history({"timestamp": time.time(), "insight": insight})
        discovery_entry = {
            "timestamp": time.time(),
            "insight": insight,
            "tags": ["internal", "insight"]
        }
        update_selfcontext(discovery_entry)
        reinforcement_update()
        if insight and random.random() < 0.5:
            print("[Orchestrator] Spontaneously speaking insight.")
            enqueue_sentence_for_tts("Spontaneous Insight: " + insight)

#############################################
# Step 12: Handling Concurrent Requests and Cancellation
#############################################
stop_flag = False
current_thread = None
inference_lock = threading.Lock()
def inference_thread(user_message, result_holder, model_actual_name):
    global stop_flag
    stop_flag = False
    result = process_text(user_message)
    result_holder.append(result)
def new_request(user_message, model_actual_name):
    global stop_flag, current_thread
    with inference_lock:
        if current_thread and current_thread.is_alive():
            print("Interrupting current inference...")
            stop_flag = True
            current_thread.join()
            stop_flag = False
        print("Stopping TTS thread...")
        stop_tts_thread()
        print("Starting new TTS thread...")
        start_tts_thread()
        result_holder = []
        current_thread = threading.Thread(target=inference_thread, args=(user_message, result_holder, model_actual_name))
        current_thread.start()
    current_thread.join()
    result = result_holder[0] if result_holder else ""
    return result

#############################################
# Step 13: Start Server with Enhanced Interrupt Handling
#############################################
HOST = CONFIG["host"]
PORT = CONFIG["port"]
client_threads = []
client_threads_lock = threading.Lock()
def handle_client_connection(client_socket, address, model_actual_name):
    global stop_flag, current_thread
    print(f"\nAccepted connection from {address}")
    try:
        data = client_socket.recv(65536)
        if not data:
            print(f"No data from {address}, closing connection.")
            return
        user_message = data.decode('utf-8').strip()
        if not user_message:
            print(f"Empty prompt from {address}, ignoring.")
            return
        print(f"Received prompt from {address}: {user_message}")
        update_objectives_from_user(user_message)
        result = new_request(user_message, model_actual_name)
        client_socket.sendall(result.encode('utf-8'))
        update_history(user_message, result)
    except Exception as e:
        print(f"Error handling client {address}: {e}")
    finally:
        client_socket.close()
def start_server():
    global client_threads
    print("\nStarting TTS thread...")
    start_tts_thread()
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server.bind((HOST, PORT))
    except Exception as e:
        print(f"Error binding to {HOST}:{PORT} - {e}. Using defaults: 0.0.0.0:64162")
        HOST_D = '0.0.0.0'
        PORT_D = 64162
        server.bind((HOST_D, PORT_D))
    server.listen(5)
    print(f"\nListening for incoming connections on {HOST}:{PORT}...")
    model_actual_name = CONFIG["model"]
    try:
        while True:
            try:
                client_sock, addr = server.accept()
                client_thread = threading.Thread(target=handle_client_connection, args=(client_sock, addr, model_actual_name))
                client_thread.start()
                with client_threads_lock:
                    client_threads.append(client_thread)
            except KeyboardInterrupt:
                print("\nInterrupt received, shutting down server.")
                break
            except Exception as e:
                print(f"Error accepting connections: {e}")
    finally:
        server.close()
        print("\nServer socket closed.")
        print("Stopping TTS thread...")
        stop_tts_thread()
        print("Waiting for client threads to finish...")
        with client_threads_lock:
            for t in client_threads:
                t.join()
        print("All client threads have been terminated.")
        print("Shutting down complete.")

#############################################
# Main
#############################################
if __name__ == "__main__":
    child_threads = start_children_threads()
    parent_thread = threading.Thread(target=parent_insight_thread, daemon=True)
    parent_thread.start()
    start_server()
