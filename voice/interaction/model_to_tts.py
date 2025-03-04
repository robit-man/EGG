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
import sqlite3
import pickle
import numpy as np
from datetime import datetime

#############################################
# Step 1: Ensure we're running inside a venv #
#############################################

VENV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv")
NEEDED_PACKAGES = ["requests", "num2words", "ollama"]

def in_venv():
    return (
        hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    )

def setup_venv():
    # Create venv if it doesn't exist
    if not os.path.isdir(VENV_DIR):
        print("Creating virtual environment...")
        subprocess.check_call([sys.executable, '-m', 'venv', VENV_DIR])
    pip_path = os.path.join(VENV_DIR, 'bin', 'pip')
    subprocess.check_call([pip_path, 'install'] + NEEDED_PACKAGES)

def relaunch_in_venv():
    # Relaunch inside venv python
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
from ollama import embed  # <--- For memory embedding

#############################################
# Additional: Short Tone/Beep Utilities      #
#############################################

def beep(freq=120, duration=0.05):
    """
    Play a short beep tone (if 'play' is installed) at the specified frequency (Hz)
    and duration (seconds). Captures output to avoid clutter.
    """
    try:
        subprocess.run(
            ["play", "-nq", "-t", "alsa", "synth", str(duration), "sine", str(freq)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
    except Exception:
        # If 'play' is not available or any other error occurs, just ignore.
        pass

#############################################
# Step 3: Config Defaults & File
#############################################

DEFAULT_CONFIG = {
    "model": "llama3.2-vision",
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
    "database_path": "embeddings.db"  # <--- For storing memory embeddings
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
            # Merge with DEFAULT_CONFIG
            for key, value in DEFAULT_CONFIG.items():
                if key not in cfg:
                    cfg[key] = value
            return cfg
        except Exception as e:
            print(f"Error loading config.json: {e}. Using default settings.")
            return dict(DEFAULT_CONFIG)

CONFIG = load_config()

#############################################
# Step 4: Parse Command-Line Arguments       #
#############################################

parser = argparse.ArgumentParser(description="Ollama Chat Server with TTS and advanced features.")

parser.add_argument("--model", type=str, help="Model name to use.")
parser.add_argument("--stream", action="store_true", help="Enable streaming responses from the model.")
parser.add_argument("--format", type=str, help="Structured output format: 'json' or path to JSON schema file.")
parser.add_argument("--system", type=str, help="System message override.")
parser.add_argument("--raw", action="store_true", help="If set, use raw mode (no template).")
parser.add_argument(
    "--history",
    type=str,
    nargs='?',
    const="chat.json",
    help="Path to a JSON file containing conversation history messages. Defaults to 'chat.json' if no path is provided."
)
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
                # Attempt to convert numeric
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
# Step 5: Load Optional Configurations       #
#############################################

def safe_load_json_file(path, default):
    if not path:
        return default
    if not os.path.exists(path):
        print(f"Warning: File '{path}' not found. Using default {default}.")
        if path == CONFIG["history"] and default == []:
            # Create empty history file
            with open(path, 'w') as f:
                json.dump([], f)
        return default
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load '{path}': {e}. Using default {default}.")
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
        except Exception as e:
            print(f"Warning: Could not load format schema from '{fmt}': {e}. Ignoring format.")
            return None
    else:
        print(f"Warning: Format file '{fmt}' not found. Ignoring format.")
        return None

history_messages = safe_load_json_file(CONFIG["history"], [])
tools_data = safe_load_json_file(CONFIG["tools"], None)
format_schema = load_format_schema(CONFIG["format"])

# Filter out invalid items from history
if not isinstance(history_messages, list):
    history_messages = []
else:
    history_messages = [
        m for m in history_messages
        if isinstance(m, dict) and "role" in m and "content" in m
    ]

#############################################
# Step 5.1: Database for Memory Embeddings
#############################################

DB_PATH = CONFIG.get("database_path", "embeddings.db")
import threading

def initialize_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT,
            embedding BLOB,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()
    print(f"[Database] Initialized at {DB_PATH}")

db_lock = threading.Lock()

def generate_embedding(text):
    """
    Uses the 'nomic-embed-text' model to embed 'text'.
    Returns a Python list of floats, or None if fails.
    """
    with db_lock:
        try:
            resp = embed(model='nomic-embed-text', input=text)
        except Exception as e:
            print(f"Memory embed call failed for '{text}': {e}")
            return None

        if not isinstance(resp, dict):
            print("Memory embed returned non-dict:", resp)
            return None
        embeddings = resp.get("embeddings")
        if not embeddings or not isinstance(embeddings, list):
            print("Empty or invalid embeddings array from 'nomic-embed-text':", resp)
            return None

        # If embeddings[0] is itself a list of floats, we use that
        # Otherwise assume embeddings is a single vector
        if isinstance(embeddings[0], list):
            return embeddings[0]
        return embeddings

import pickle
import numpy as np

def store_memory(text):
    """
    Generate embedding for 'text' and store in DB if not empty.
    """
    emb = generate_embedding(text)
    if not emb or len(emb)==0:
        print("No embedding to store for:", text)
        return
    blob=pickle.dumps(emb)
    with db_lock:
        conn=sqlite3.connect(DB_PATH)
        c=conn.cursor()
        c.execute("INSERT INTO embeddings (text,embedding,timestamp) VALUES (?,?,?)",
                  (text, blob, datetime.utcnow().isoformat()))
        conn.commit()
        conn.close()
    print("[Memory] Stored embedding for:", text)

def get_memory_matches(query_text, top_k=3):
    """
    Embed 'query_text', retrieve top K similar from DB.
    Return a list of textual matches.
    """
    q_emb = generate_embedding(query_text)
    if not q_emb:
        return []
    q_arr = np.array(q_emb)

    with db_lock:
        conn=sqlite3.connect(DB_PATH)
        c=conn.cursor()
        c.execute("SELECT text,embedding,timestamp FROM embeddings")
        rows=c.fetchall()
        conn.close()

    results=[]
    for t, eblob, ts in rows:
        try:
            arr=pickle.loads(eblob)
        except:
            continue
        if not arr:
            continue
        arr_np=np.array(arr)
        denom = np.linalg.norm(q_arr)*np.linalg.norm(arr_np)
        if denom==0:
            sim=0.0
        else:
            sim = float(np.dot(q_arr, arr_np)/denom)
        results.append((t, sim))

    results.sort(key=lambda x: x[1], reverse=True)
    top=results[:top_k]
    matches=[x[0] for x in top]
    return matches

#############################################
# Step 6: Ollama chat interaction
#############################################

OLLAMA_CHAT_URL = CONFIG["ollama_url"]

def convert_numbers_to_words(text):
    """
    Convert all standalone numbers in text to their word equivalents.
    """
    def replace_num(match):
        number_str = match.group(0)
        try:
            number_int = int(number_str)
            return num2words(number_int)
        except ValueError:
            return number_str
    return re.sub(r'\b\d+\b', replace_num, text)

def build_payload(user_message):
    """
    Build the messages array with:
    1) system message (if any)
    2) existing history
    3) memory retrieval (top 3)
    4) user message
    """
    messages=[]
    # Add system if present
    if CONFIG["system"]:
        messages.append({"role":"system","content":CONFIG["system"]})

    # Add prior history
    messages.extend(history_messages)

    # memory retrieval
    mems = get_memory_matches(user_message, top_k=3)
    if mems:
        content_str = "\n".join(mems)
        messages.append({"role":"system","content":"Relevant memories:\n"+content_str})

    # user last
    messages.append({"role":"user","content":user_message})

    payload={
        "model": CONFIG["model"],
        "messages": messages,
        "stream": CONFIG["stream"]
    }
    if format_schema:
        payload["format"]=format_schema
    if CONFIG["raw"]:
        payload["raw"]=True
    if CONFIG["images"]:
        if payload["messages"] and payload["messages"][-1]["role"]=="user":
            payload["messages"][-1]["images"]=CONFIG["images"]
    if tools_data:
        payload["tools"]=tools_data
    if CONFIG["options"]:
        payload["options"]=CONFIG["options"]
    return payload

stop_flag = False
thread_lock = threading.Lock()

#############################################
# Step 7: TTS Playback with Queue and Thread
#############################################

tts_queue = None
tts_stop_flag = False
tts_thread = None
tts_thread_lock = threading.Lock()

def synthesize_and_play(prompt):
    # Filter out asterisks (*) and hashtags (#):
    prompt = re.sub(r'[\*#]', '', prompt).strip()
    if not prompt:
        return
    start_wait_beeps()
    try:
        payload={"prompt":prompt}
        with requests.post(CONFIG["tts_url"], json=payload, stream=True) as r:
            if r.status_code!=200:
                print(f"Warning: TTS code {r.status_code}")
                stop_wait_beeps()
                return
            aplay=subprocess.Popen(['aplay','-r','22050','-f','S16_LE','-t','raw'],stdin=subprocess.PIPE)
            try:
                for chunk in r.iter_content(chunk_size=4096):
                    stop_wait_beeps()
                    if tts_stop_flag:
                        break
                    aplay.stdin.write(chunk)
            except BrokenPipeError:
                print("Warning: aplay ended.")
            finally:
                aplay.stdin.close()
                aplay.wait()
    except Exception as e:
        print("TTS error:", e)
        stop_wait_beeps()
    else:
        stop_wait_beeps()

def tts_worker():
    global tts_stop_flag
    while not tts_stop_flag:
        try:
            line=tts_queue.get(timeout=0.1)
        except:
            if tts_stop_flag:
                break
            continue
        if tts_stop_flag:
            break
        synthesize_and_play(line)

def start_tts_thread():
    global tts_queue, tts_thread, tts_stop_flag
    with tts_thread_lock:
        if tts_thread and tts_thread.is_alive():
            return
        tts_stop_flag=False
        tts_queue=Queue()
        tts_thread=threading.Thread(target=tts_worker, daemon=True)
        tts_thread.start()

def stop_tts_thread():
    global tts_stop_flag, tts_thread, tts_queue
    with tts_thread_lock:
        if tts_thread and tts_thread.is_alive():
            tts_stop_flag=True
            with tts_queue.mutex:
                tts_queue.queue.clear()
            tts_thread.join()
        tts_stop_flag=False
        tts_queue=None
        tts_thread=None

def enqueue_sentence_for_tts(sentence):
    if tts_queue and not tts_stop_flag:
        tts_queue.put(sentence)

wait_beeps_thread=None
wait_beeps_flag=False
wait_beeps_lock=threading.Lock()

def wait_beeps_worker():
    while True:
        with wait_beeps_lock:
            if not wait_beeps_flag:
                break
        beep(80,0.05)
        with wait_beeps_lock:
            if not wait_beeps_flag:
                break
        time.sleep(0.1)
        beep(80,0.05)
        with wait_beeps_lock:
            if not wait_beeps_flag:
                break
        time.sleep(0.1)
        beep(80,0.05)
        with wait_beeps_lock:
            if not wait_beeps_flag:
                break
        time.sleep(0.65)

def start_wait_beeps():
    global wait_beeps_thread, wait_beeps_flag
    with wait_beeps_lock:
        if wait_beeps_thread and wait_beeps_thread.is_alive():
            return
        wait_beeps_flag=True
    wait_beeps_thread=threading.Thread(target=wait_beeps_worker, daemon=True)
    wait_beeps_thread.start()

def stop_wait_beeps():
    global wait_beeps_thread, wait_beeps_flag
    with wait_beeps_lock:
        wait_beeps_flag=False
    if wait_beeps_thread and wait_beeps_thread.is_alive():
        wait_beeps_thread.join()
    wait_beeps_thread=None

#############################################
# Step 7.1: Actual Inference
#############################################

def chat_completion_stream(user_message):
    global stop_flag
    payload=build_payload(user_message)
    headers={"Content-Type":"application/json"}
    try:
        with requests.post(CONFIG["ollama_url"], json=payload, headers=headers, stream=True) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if stop_flag:
                    break
                if line:
                    obj=json.loads(line.decode('utf-8'))
                    msg=obj.get("message",{})
                    content=msg.get("content","")
                    done=obj.get("done",False)
                    yield content, done
                    if done:
                        break
    except Exception as e:
        print("Error streaming inference:", e)
        yield "", True

def chat_completion_nonstream(user_message):
    payload=build_payload(user_message)
    headers={"Content-Type":"application/json"}
    try:
        resp=requests.post(CONFIG["ollama_url"],json=payload,headers=headers)
        resp.raise_for_status()
        data=resp.json()
        msg=data.get("message",{})
        return msg.get("content","")
    except Exception as e:
        print("Error non-stream inference:", e)
        return ""

def process_text(user_message):
    """
    Build text, pass to Ollama (stream or not),
    do TTS sentence by sentence in streaming mode.
    """
    global stop_flag
    if CONFIG["stream"]:
        # streaming
        buffer=""
        sentences=[]
        for chunk, done in chat_completion_stream(user_message):
            if stop_flag:
                break
            print(chunk, end='', flush=True)
            buffer+=chunk
            sentence_endings=re.compile(r'[.?!]+')
            while True:
                if stop_flag:
                    break
                match=sentence_endings.search(buffer)
                if not match:
                    break
                end_idx=match.end()
                sentence=buffer[:end_idx].strip()
                buffer=buffer[end_idx:].strip()
                if sentence:
                    sentences.append(sentence)
                    enqueue_sentence_for_tts(sentence)
            if done or stop_flag:
                break
        print()
        leftover=buffer.strip()
        if leftover:
            sentences.append(leftover)
            enqueue_sentence_for_tts(leftover)
        return " ".join(sentences)
    else:
        # non-stream
        text=chat_completion_nonstream(user_message)
        print(text)
        # TTS entire text in sentences
        sentences=[]
        buffer=text
        sentence_endings=re.compile(r'[.?!]+')
        while True:
            match=sentence_endings.search(buffer)
            if not match:
                break
            end_idx=match.end()
            sentence=buffer[:end_idx].strip()
            buffer=buffer[end_idx:].strip()
            if sentence:
                enqueue_sentence_for_tts(sentence)
                sentences.append(sentence)
        leftover=buffer.strip()
        if leftover:
            enqueue_sentence_for_tts(leftover)
            sentences.append(leftover)
        return " ".join(sentences)

#############################################
# Step 10: Update History
#############################################

def update_history_file(user_message, assistant_message):
    if not CONFIG["history"]:
        return
    current_history = safe_load_json_file(CONFIG["history"], [])
    if not isinstance(current_history, list):
        current_history=[]
    current_history.append({"role":"user","content":user_message})
    current_history.append({"role":"assistant","content":assistant_message})
    try:
        with open(CONFIG["history"], 'w') as f:
            json.dump(current_history,f,indent=2)
    except Exception as e:
        print("Warning: can't write to history file:", e)

#############################################
# Step 11: Embedding user & assistant messages
#############################################

def remember_messages(user_message, assistant_message):
    # embed user
    store_memory(user_message)
    # embed assistant
    store_memory(assistant_message)

#############################################
# Step 11: Handling concurrency
#############################################

stop_flag=False
current_thread=None
inference_lock=threading.Lock()

def inference_thread(user_message, result_list):
    global stop_flag
    stop_flag=False
    result=process_text(user_message)
    result_list.append(result)

def new_request(user_message):
    global stop_flag, current_thread
    beep(120,0.05)

    with inference_lock:
        if current_thread and current_thread.is_alive():
            print("Interrupting current inference..")
            beep(80,0.05)
            stop_flag=True
            current_thread.join()
            stop_flag=False

        print("Stopping TTS..")
        beep(80,0.05)
        stop_tts_thread()

        print("Starting new TTS..")
        beep(120,0.05)
        start_tts_thread()

        holder=[]
        current_thread=threading.Thread(target=inference_thread,args=(user_message,holder))
        current_thread.start()

    current_thread.join()
    final=holder[0] if holder else ""
    return final

#############################################
# Step 12: Server
#############################################

HOST=CONFIG["host"]
PORT=CONFIG["port"]

client_threads=[]
client_threads_lock=threading.Lock()

def handle_client_connection(client_socket, addr):
    print("\nAccepted connection from", addr)
    beep(120,0.05)
    try:
        data=client_socket.recv(65536)
        if not data:
            print("No data from", addr)
            return
        user_message=data.decode('utf-8').strip()
        if not user_message:
            print("Empty prompt from", addr)
            return
        print("Received prompt from",addr,":",user_message)
        beep(120,0.05)
        result=new_request(user_message)

        client_socket.sendall(result.encode('utf-8'))
        # update history
        update_history_file(user_message, result)
        # embed user+assistant
        remember_messages(user_message, result)
    except Exception as e:
        print("Error handling client", addr, ":", e)
    finally:
        client_socket.close()

def start_server():
    initialize_db()
    print("Starting TTS thread..")
    start_tts_thread()

    server=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR,1)
    try:
        server.bind((HOST, int(PORT)))
    except Exception as e:
        print(f"Error binding to {HOST}:{PORT}", e,"Using defaults 0.0.0.0:64162")
        server.bind(('0.0.0.0',64162))

    server.listen(5)
    print(f"\nListening for incoming connections on {HOST}:{PORT}...")

    try:
        while True:
            client_sock, client_addr=server.accept()
            t=threading.Thread(target=handle_client_connection,args=(client_sock,client_addr))
            t.start()
            with client_threads_lock:
                client_threads.append(t)
    except KeyboardInterrupt:
        print("\nKeyboard interrupt, shutting down..")
    finally:
        server.close()
        print("Server closed.")
        print("Stopping TTS..")
        stop_tts_thread()
        print("Waiting for client threads..")
        with client_threads_lock:
            for thr in client_threads:
                thr.join()
        print("All client threads done. Shutdown complete.")

if __name__=="__main__":
    start_server()
