#!/usr/bin/env python3
import os
import sys
import subprocess
import socket
import re
import json
import argparse
import threading
from queue import Queue, Empty
import shutil
import time
import logging
import multiprocessing  # For handling inference processes
import math
import difflib
from urllib.parse import urlsplit

PSUTIL_AVAILABLE = False
ALSAAUDIO_AVAILABLE = False

#############################################
# Utility Functions
#############################################

def is_connected(host="8.8.8.8", port=53, timeout=3):
    """
    Check internet connectivity by attempting to connect to a well-known DNS server.
    Returns True if connected, False otherwise.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            sock.connect((host, port))
        return True
    except socket.error:
        return False

#############################################
# Step 2: Ensure we're running inside a venv #
#############################################

VENV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv")
NEEDED_PACKAGES = ["requests", "num2words", "pyalsaaudio", "psutil"]

def in_venv():
    expected = os.path.normcase(os.path.abspath(VENV_DIR))
    current = os.path.normcase(os.path.abspath(sys.prefix))
    return current == expected

def setup_venv(online=True):
    # Create venv if it doesn't exist
    if not os.path.isdir(VENV_DIR):
        logging.info("Creating virtual environment...")
        try:
            subprocess.check_call([sys.executable, '-m', 'venv', '--system-site-packages', VENV_DIR])
            logging.info("Virtual environment created successfully.")
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to create virtual environment: {e}")
            if not online:
                logging.warning("Proceeding without setting up virtual environment due to offline mode.")
            else:
                sys.exit(1)

    # Determine pip path based on OS
    pip_path = os.path.join(VENV_DIR, 'bin', 'pip') if os.name != 'nt' else os.path.join(VENV_DIR, 'Scripts', 'pip.exe')

    try:
        logging.info("Installing required packages...")
        subprocess.check_call([pip_path, 'install', '--upgrade', 'pip'])
        subprocess.check_call([pip_path, 'install'] + NEEDED_PACKAGES)
        logging.info("Required packages installed successfully.")
    except subprocess.CalledProcessError as e:
        if not online:
            logging.warning(f"Offline mode package install failed: {e}")
        else:
            logging.error(f"Failed to install required packages: {e}")
        logging.warning("Proceeding without installing all packages. Ensure required packages are installed.")


def ensure_runtime_packages():
    pip_path = os.path.join(VENV_DIR, 'bin', 'pip') if os.name != 'nt' else os.path.join(VENV_DIR, 'Scripts', 'pip.exe')
    if not os.path.exists(pip_path):
        try:
            setup_venv(online=is_connected())
        except Exception:
            return False
    try:
        subprocess.check_call([pip_path, 'install'] + NEEDED_PACKAGES)
        return True
    except subprocess.CalledProcessError:
        return False

def relaunch_in_venv():
    # Relaunch inside venv python
    python_path = os.path.join(VENV_DIR, 'bin', 'python') if os.name != 'nt' else os.path.join(VENV_DIR, 'Scripts', 'python.exe')
    if os.path.exists(python_path):
        logging.info("Relaunching script inside the virtual environment...")
        os.execv(python_path, [python_path] + sys.argv)
    else:
        logging.error("Virtual environment Python executable not found.")
        sys.exit(1)

if not in_venv():
    # Determine online status
    ONLINE = is_connected()
    if not ONLINE:
        logging.warning("No internet connection detected. Operating in offline mode.")
    # Setup VENV and install packages if online
    setup_venv(online=ONLINE)
    # Always relaunch in venv, regardless of online/offline status
    relaunch_in_venv()
else:
    #############################################
    # Step 3: Imports after venv set up          #
    #############################################
    
    try:
        import requests
        from num2words import num2words
    except ImportError as e:
        logging.warning(f"Initial import failed: {e}. Attempting runtime package install...")
        ensure_runtime_packages()
        try:
            import requests
            from num2words import num2words
        except ImportError as inner_exc:
            logging.error(f"Failed to import required modules: {inner_exc}")
            logging.error("Ensure all required packages are installed in the virtual environment.")
            sys.exit(1)

    try:
        import alsaaudio  # Optional ALSA module
        ALSAAUDIO_AVAILABLE = True
    except ImportError:
        alsaaudio = None
        ALSAAUDIO_AVAILABLE = False

    try:
        import psutil  # Optional CPU usage monitoring
        PSUTIL_AVAILABLE = True
    except ImportError:
        psutil = None
        PSUTIL_AVAILABLE = False

    #############################################
    # Step 1: Setup Logging
    #############################################
    
    # Configure logging to include thread name and timestamp
    logging.basicConfig(
        level=logging.INFO,  # Can set to DEBUG for more detailed logs
        format='[%(asctime)s] [%(threadName)s] [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            # Uncomment the following line to log to a file
            # logging.FileHandler("server.log")
        ]
    )
    if not ALSAAUDIO_AVAILABLE:
        logging.warning("pyalsaaudio not available; continuing without direct ALSA hooks.")
    if not PSUTIL_AVAILABLE:
        logging.warning("psutil not available; CPU monitor will use a fallback value.")
    
    #############################################
    # Step 4: Config Defaults & File
    #############################################
    
    DEFAULT_CONFIG = {
        "model": "qwen3:0.6b",
        "stream": True,
        "thinking_enabled": False,
        "format": None,
        "system": ("You are a highly efficient and curious small language model running on a Raspberry Pi 5. "
                   "Your primary goal is to provide clear, concise, and actionable responses. Avoid disclaimers about "
                   "your limitations or unnecessary verbosity—focus on being personable and engaging. Express curiosity "
                   "about the context when appropriate, and always strive to be helpful, resourceful, and to the point. "
                   "Remember, your brevity is your strength. You are getting prompts sent to you from 'whisper' speech "
                   "recognition, which sometimes sends you messages in perenthesis like (background noise) or "
                   "(speaking in foreign language) or (door opening), These indicate detected content and you should respond "
                   "naturally, without treating these like vocal input. Also note that everything you receive as a prompt "
                   "is actually coming from a microphone and interpretation of the input from it. As such, you should reply "
                   "to everything as if you actually have ears and the ability to hear as you have a transducer at your input "
                   "helping aid in your understanding of what is around you! Also, any use of end of sentence delimiters like "
                   "periods or other punctuation truncates what is then sent to the TTS generator, so please use punctuation in "
                   "a strategic way to prevent run on sentences that take a long time to generate text, and also when making "
                   "lists, avoid using periods after the number delimiters to prevent treating the list position as its own "
                   "isolated sentence passed to TTS. When responding with the intention of laughing or other expressions, "
                   "produce content like 'hahaha' or 'awwww' or 'ahhhhh', not with (laughing) or other expressions that you receive."),
        "raw": False,
        "history": "chat.json",
        "images": [],
        "tools": None,
        "options": {},
        "host": "0.0.0.0",
        "port": 6545,
        "tts_host": "127.0.0.1",
        "tts_port": 6434,
        "tts_url": "http://localhost:6434",
        "audio_out_host": "127.0.0.1",
        "audio_out_port": 6353,
        "audio_out_sample_rate": 22050,
        "audio_out_channels": 1,
        "pipeline_event_url": "http://127.0.0.1:6590/pipeline/event",
        "ollama_url": "http://localhost:11434/api/chat",
        "max_history_messages": 3,  # New Configuration Parameter
        "interrupt_on_new_prompt": False,
        "offline_mode": False  # Default offline mode
    }
    CONFIG_PATH = "llm_bridge_config.json"
    AUDIO_ROUTER_CONFIG_PATH = "audio_router_config.json"
    PIPELINE_API_CONFIG_PATH = "pipeline_api_config.json"
    
    def load_config():
        if not os.path.exists(CONFIG_PATH):
            logging.info(f"No {CONFIG_PATH} found. Creating default config file...")
            try:
                with open(CONFIG_PATH, 'w') as f:
                    json.dump(DEFAULT_CONFIG, f, indent=2)
                return dict(DEFAULT_CONFIG)
            except Exception as e:
                logging.error(f"Failed to create {CONFIG_PATH}: {e}")
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
                logging.error(f"Error loading {CONFIG_PATH}: {e}. Using default settings.")
                return dict(DEFAULT_CONFIG)
    
    CONFIG = load_config()
    
    #############################################
    # Step 5: Parse Command-Line Arguments       #
    #############################################
    
    parser = argparse.ArgumentParser(description="Ollama Chat Server with TTS and advanced features.")
    
    parser.add_argument("--model", type=str, help="Model name to use.")
    parser.add_argument("--stream", action="store_true", help="Enable streaming responses from the model.")
    parser.add_argument("--thinking", dest="thinking_enabled", action="store_true", help="Enable model thinking mode.")
    parser.add_argument("--thinking-off", dest="thinking_enabled", action="store_false", help="Disable model thinking mode.")
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
    
    # New Command-Line Argument for max_history_messages
    parser.add_argument("--max-history", type=int, help="Maximum number of recent chat history messages to recall.")
    
    # New Argument to Force Offline Mode
    parser.add_argument("--offline", action="store_true", help="Force the script to operate in offline mode.")
    parser.set_defaults(thinking_enabled=None)
    
    args = parser.parse_args()
    
    def merge_config_and_args(config, args):
        if args.model:
            config["model"] = args.model
        if args.stream:
            config["stream"] = True
        if args.thinking_enabled is not None:
            config["thinking_enabled"] = bool(args.thinking_enabled)
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
        # Handle the new --max-history argument
        if args.max_history is not None:
            config["max_history_messages"] = args.max_history
        # Handle the new --offline argument
        config["offline_mode"] = args.offline
        return config

    def _get_nested(data, path, default=None):
        current = data
        for key in str(path or "").split("."):
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current

    def apply_audio_router_overrides(config):
        try:
            with open(AUDIO_ROUTER_CONFIG_PATH, "r", encoding="utf-8") as fp:
                payload = json.load(fp)
            if not isinstance(payload, dict):
                return config
        except Exception:
            return config

        tts_host = str(_get_nested(payload, "audio_router.integrations.tts_host", config.get("tts_host", "127.0.0.1"))).strip()
        tts_port = _get_nested(payload, "audio_router.integrations.tts_port", config.get("tts_port", 6434))
        try:
            tts_port = int(tts_port)
        except Exception:
            tts_port = int(config.get("tts_port", 6434))

        if tts_host:
            config["tts_host"] = tts_host
        config["tts_port"] = tts_port
        config["tts_url"] = f"http://{config['tts_host']}:{config['tts_port']}"
        audio_out_host = str(
            _get_nested(payload, "audio_router.integrations.audio_out_host", config.get("audio_out_host", "127.0.0.1"))
        ).strip()
        audio_out_port = _get_nested(payload, "audio_router.integrations.audio_out_port", config.get("audio_out_port", 6353))
        audio_out_rate = _get_nested(
            payload, "audio_router.audio.output_sample_rate", config.get("audio_out_sample_rate", 22050)
        )
        audio_out_channels = _get_nested(
            payload, "audio_router.audio.output_channels", config.get("audio_out_channels", 1)
        )
        try:
            audio_out_port = int(audio_out_port)
        except Exception:
            audio_out_port = int(config.get("audio_out_port", 6353))
        try:
            audio_out_rate = int(audio_out_rate)
        except Exception:
            audio_out_rate = int(config.get("audio_out_sample_rate", 22050))
        try:
            audio_out_channels = int(audio_out_channels)
        except Exception:
            audio_out_channels = int(config.get("audio_out_channels", 1))

        config["audio_out_host"] = audio_out_host or "127.0.0.1"
        config["audio_out_port"] = audio_out_port
        config["audio_out_sample_rate"] = max(8000, audio_out_rate)
        config["audio_out_channels"] = max(1, min(2, audio_out_channels))
        return config

    def apply_pipeline_api_overrides(config):
        try:
            with open(PIPELINE_API_CONFIG_PATH, "r", encoding="utf-8") as fp:
                payload = json.load(fp)
            if not isinstance(payload, dict):
                return config
        except Exception:
            return config

        host = str(_get_nested(payload, "pipeline_api.network.listen_host", "127.0.0.1")).strip() or "127.0.0.1"
        if host in ("0.0.0.0", "::"):
            host = "127.0.0.1"
        port = _get_nested(payload, "pipeline_api.network.listen_port", 6590)
        try:
            port = int(port)
        except Exception:
            port = 6590
        config["pipeline_event_url"] = f"http://{host}:{port}/pipeline/event"
        return config
    
    CONFIG = merge_config_and_args(CONFIG, args)
    CONFIG = apply_audio_router_overrides(CONFIG)
    CONFIG = apply_pipeline_api_overrides(CONFIG)
    CONFIG_LOCK = threading.Lock()
    CONFIG_FILE_ABS = os.path.join(os.path.dirname(os.path.abspath(__file__)), CONFIG_PATH)
    _config_mtime = 0.0

    def _as_bool(value, default=False):
        if isinstance(value, bool):
            return value
        if value is None:
            return bool(default)
        return str(value).strip().lower() in ("1", "true", "yes", "on")

    def _persist_config():
        try:
            with CONFIG_LOCK:
                with open(CONFIG_FILE_ABS, "w", encoding="utf-8") as fp:
                    json.dump(CONFIG, fp, indent=2)
            return True
        except Exception as exc:
            logging.error(f"Failed to persist {CONFIG_PATH}: {exc}")
            return False

    def _resolve_ollama_base_url(chat_url: str) -> str:
        raw = str(chat_url or "").strip() or "http://127.0.0.1:11434/api/chat"
        candidate = raw if "://" in raw else f"http://{raw.lstrip('/')}"
        parsed = urlsplit(candidate)
        if parsed.netloc:
            scheme = parsed.scheme or "http"
            return f"{scheme}://{parsed.netloc}".rstrip("/")
        return "http://127.0.0.1:11434"

    def _ollama_model_installed(model_name: str) -> bool:
        model = str(model_name or "").strip()
        if not model:
            return False
        base_url = _resolve_ollama_base_url(CONFIG.get("ollama_url", ""))
        tags_url = f"{base_url}/api/tags"
        try:
            response = requests.get(tags_url, timeout=10)
            response.raise_for_status()
            payload = response.json()
            models = payload.get("models", []) if isinstance(payload, dict) else []
            for item in models:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name", "")).strip()
                if name == model:
                    return True
        except Exception as exc:
            logging.warning(f"Model check failed for '{model}' via {tags_url}: {exc}")
        return False

    def _pull_ollama_model(model_name: str) -> bool:
        model = str(model_name or "").strip()
        if not model:
            return False
        base_url = _resolve_ollama_base_url(CONFIG.get("ollama_url", ""))
        pull_url = f"{base_url}/api/pull"
        logging.info(f"Pulling missing model '{model}' from {pull_url}...")
        try:
            with requests.post(
                pull_url,
                json={"model": model, "stream": True},
                headers={"Content-Type": "application/json"},
                stream=True,
                timeout=(10, 3600),
            ) as response:
                response.raise_for_status()
                last_status = ""
                for line in response.iter_lines(decode_unicode=True):
                    text = str(line or "").strip()
                    if not text:
                        continue
                    status_line = text
                    try:
                        payload = json.loads(text)
                        if isinstance(payload, dict):
                            status_line = str(payload.get("status") or text).strip() or text
                            if payload.get("error"):
                                raise RuntimeError(str(payload.get("error")))
                    except json.JSONDecodeError:
                        pass
                    if status_line != last_status:
                        logging.info(f"[MODEL PULL] {status_line}")
                        last_status = status_line
            return True
        except Exception as exc:
            logging.error(f"Failed to pull model '{model}': {exc}")
            return False

    def _ensure_ollama_model_ready(model_name: str) -> bool:
        model = str(model_name or "").strip()
        if not model:
            return False
        if _ollama_model_installed(model):
            return True
        logging.warning(f"Configured model '{model}' is missing; auto-pull requested.")
        ok = _pull_ollama_model(model)
        if not ok:
            return False
        return _ollama_model_installed(model)

    def _reload_config_if_updated():
        global _config_mtime
        try:
            current_mtime = os.path.getmtime(CONFIG_FILE_ABS)
        except Exception:
            return
        if current_mtime <= (_config_mtime or 0.0):
            return
        try:
            loaded = load_config()
            loaded = apply_audio_router_overrides(loaded)
            loaded = apply_pipeline_api_overrides(loaded)
            previous_model = str(CONFIG.get("model", "")).strip()
            with CONFIG_LOCK:
                CONFIG.clear()
                CONFIG.update(loaded)
            _config_mtime = float(current_mtime)
            next_model = str(CONFIG.get("model", "")).strip()
            if next_model and next_model != previous_model:
                _emit_pipeline_event(
                    "llm",
                    "model_switching",
                    detail=f"source=config_hot_reload from={previous_model or 'unknown'} to={next_model}",
                )
                try:
                    _cancel_active_inference(
                        reason=f"config_model_switch:{previous_model or 'unknown'}->{next_model}"
                    )
                    dropped, notified = _discard_pending_prompts(reason="config model switch")
                    if dropped > 0:
                        _emit_pipeline_event(
                            "llm",
                            "cancelled",
                            detail=f"config switch dropped_pending={dropped} notified={notified}",
                            level="warn",
                        )
                except Exception:
                    pass
                if _ensure_ollama_model_ready(next_model):
                    _emit_pipeline_event(
                        "llm",
                        "model_switched",
                        detail=f"source=config_hot_reload from={previous_model or 'unknown'} to={next_model}",
                    )
                else:
                    _emit_pipeline_event(
                        "llm",
                        "error",
                        detail=f"source=config_hot_reload model_switch_failed target={next_model}",
                        level="error",
                    )
            logging.info(
                "Reloaded llm bridge config: "
                f"model={CONFIG.get('model')} thinking={'on' if _as_bool(CONFIG.get('thinking_enabled'), False) else 'off'}"
            )
        except Exception as exc:
            logging.warning(f"Config hot-reload failed: {exc}")

    def _set_thinking_enabled(enabled: bool):
        target = bool(enabled)
        changed = False
        with CONFIG_LOCK:
            current = _as_bool(CONFIG.get("thinking_enabled", False), False)
            if current != target:
                CONFIG["thinking_enabled"] = target
                changed = True
        if changed:
            _persist_config()
        return changed

    def _parse_thinking_toggle_command(text: str):
        cleaned = re.sub(r"\s+", " ", str(text or "").strip().lower())
        if not cleaned:
            return None
        cleaned = re.sub(r"[.!?]+$", "", cleaned).strip()
        if re.search(r"\bturn(?: the)? thinking(?: mode)? on\b", cleaned):
            return True
        if re.search(r"\bturn(?: the)? thinking(?: mode)? off\b", cleaned):
            return False
        if re.search(r"\bthinking(?: mode)? on\b", cleaned):
            return True
        if re.search(r"\bthinking(?: mode)? off\b", cleaned):
            return False
        return None

    MODEL_SWITCH_TIMEOUT_SECONDS = 45.0
    CONFIG_WATCH_INTERVAL_SECONDS = 0.75
    _model_switch_lock = threading.Lock()
    _pending_model_switch = {
        "active": False,
        "query": "",
        "options": [],
        "created_at": 0.0,
        "expires_at": 0.0,
    }

    def _list_ollama_models():
        base_url = _resolve_ollama_base_url(CONFIG.get("ollama_url", ""))
        tags_url = f"{base_url}/api/tags"
        names = []
        try:
            response = requests.get(tags_url, timeout=10)
            response.raise_for_status()
            payload = response.json()
            models = payload.get("models", []) if isinstance(payload, dict) else []
            seen = set()
            for item in models:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name", "")).strip()
                if name and name not in seen:
                    seen.add(name)
                    names.append(name)
        except Exception as exc:
            logging.warning(f"Model list fetch failed via {tags_url}: {exc}")
            return []
        names.sort(key=lambda value: value.lower())
        return names

    def _normalize_model_key(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())

    def _match_model_candidates(query: str, models, limit: int = 4):
        needle = str(query or "").strip()
        if not needle:
            return list(models[: max(1, int(limit))])
        needle_l = needle.lower()
        needle_n = _normalize_model_key(needle)
        ranked = []
        for name in models:
            model = str(name or "").strip()
            if not model:
                continue
            model_l = model.lower()
            model_n = _normalize_model_key(model)
            score = 0.0
            if model_l == needle_l:
                score = 1000.0
            elif model_n == needle_n and needle_n:
                score = 960.0
            elif model_l.startswith(needle_l):
                score = 840.0
            elif needle_l in model_l:
                score = 760.0
            elif needle_n and needle_n in model_n:
                score = 700.0
            else:
                ratio = difflib.SequenceMatcher(None, needle_l, model_l).ratio()
                score = ratio * 600.0
            ranked.append((score, len(model), model))
        ranked.sort(key=lambda item: (-item[0], item[1], item[2].lower()))
        top = []
        seen = set()
        for score, _, model in ranked:
            if model in seen:
                continue
            if score < 280.0:
                continue
            top.append(model)
            seen.add(model)
            if len(top) >= max(1, int(limit)):
                break
        return top

    def _parse_model_switch_request(text: str):
        cleaned = re.sub(r"\s+", " ", str(text or "").strip().lower())
        if not cleaned:
            return None
        cleaned = re.sub(r"[.!?]+$", "", cleaned).strip()
        patterns = (
            r"^(?:please\s+)?(?:switch|change|set|use)\s+(?:the\s+)?model(?:\s+(?:to|as))?\s*(.*)$",
            r"^(?:please\s+)?model\s+(?:to|as)\s+(.*)$",
        )
        for pattern in patterns:
            match = re.match(pattern, cleaned)
            if not match:
                continue
            target = str(match.group(1) or "").strip()
            target = re.sub(r"^(?:to|as)\s+", "", target).strip()
            return target
        return None

    def _set_pending_model_switch(query_text: str, options):
        now = time.time()
        with _model_switch_lock:
            _pending_model_switch["active"] = True
            _pending_model_switch["query"] = str(query_text or "").strip()
            _pending_model_switch["options"] = list(options or [])
            _pending_model_switch["created_at"] = now
            _pending_model_switch["expires_at"] = now + MODEL_SWITCH_TIMEOUT_SECONDS

    def _clear_pending_model_switch():
        with _model_switch_lock:
            _pending_model_switch["active"] = False
            _pending_model_switch["query"] = ""
            _pending_model_switch["options"] = []
            _pending_model_switch["created_at"] = 0.0
            _pending_model_switch["expires_at"] = 0.0

    def _get_pending_model_switch():
        with _model_switch_lock:
            if not bool(_pending_model_switch.get("active")):
                return None
            if float(_pending_model_switch.get("expires_at") or 0.0) <= time.time():
                _pending_model_switch["active"] = False
                _pending_model_switch["query"] = ""
                _pending_model_switch["options"] = []
                _pending_model_switch["created_at"] = 0.0
                _pending_model_switch["expires_at"] = 0.0
                return None
            return {
                "query": str(_pending_model_switch.get("query") or ""),
                "options": list(_pending_model_switch.get("options") or []),
                "expires_at": float(_pending_model_switch.get("expires_at") or 0.0),
            }

    def _parse_model_choice_index(text: str, option_count: int):
        cleaned = re.sub(r"\s+", " ", str(text or "").strip().lower())
        if not cleaned:
            return None
        cleaned = re.sub(r"[.!?]+$", "", cleaned).strip()
        if cleaned in ("cancel", "stop", "never mind", "nevermind", "forget it", "abort"):
            return -1

        number_words = {
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "first": 1,
            "second": 2,
            "third": 3,
            "fourth": 4,
            "fifth": 5,
        }
        match = re.search(r"\b(\d+)\b", cleaned)
        if match:
            try:
                value = int(match.group(1))
                if 1 <= value <= option_count:
                    return value - 1
            except Exception:
                pass
        for word, number in number_words.items():
            if re.search(rf"\b{re.escape(word)}\b", cleaned):
                if 1 <= number <= option_count:
                    return number - 1
        return None

    def _switch_model_to(target_model: str):
        target = str(target_model or "").strip()
        if not target:
            return False, "Missing model name."
        with CONFIG_LOCK:
            current = str(CONFIG.get("model", "")).strip()
        if current == target:
            return True, f"Model already set to {target}."

        _emit_pipeline_event("llm", "model_switching", detail=f"from={current or 'unknown'} to={target}")
        try:
            _cancel_active_inference(reason=f"voice_model_switch:{current or 'unknown'}->{target}")
            dropped, notified = _discard_pending_prompts(reason="voice model switch")
            if dropped > 0:
                _emit_pipeline_event(
                    "llm",
                    "cancelled",
                    detail=f"voice switch dropped_pending={dropped} notified={notified}",
                    level="warn",
                )
        except Exception:
            pass
        if not _ensure_ollama_model_ready(target):
            _emit_pipeline_event(
                "llm",
                "error",
                detail=f"model switch failed target={target} reason=prepare_failed",
                level="error",
            )
            return False, f"I could not prepare model {target}."

        with CONFIG_LOCK:
            CONFIG["model"] = target
        if not _persist_config():
            return False, f"I prepared {target}, but failed to save config."
        _emit_pipeline_event("llm", "model_switched", detail=f"from={current or 'unknown'} to={target}")
        return True, f"Switching model to {target}."

    def _build_model_choice_prompt(query_text: str, options):
        display = []
        for idx, name in enumerate(options, start=1):
            display.append(f"{idx}. {name}")
        options_text = " ".join(display)
        query = str(query_text or "").strip()
        if query:
            return f"I found these model matches for {query}. {options_text} Say 1, 2, 3, or 4."
        return f"Pick a model. {options_text} Say 1, 2, 3, or 4."

    def _emit_processing_blip(frequency_hz: float = 6000.0, duration_seconds: float = 0.2):
        """
        Emit a short PCM cue directly to the raw audio output socket.
        This avoids TTS inference overhead for immediate auditory feedback.
        """
        try:
            host = str(CONFIG.get("audio_out_host", "127.0.0.1")).strip() or "127.0.0.1"
            try:
                port = int(CONFIG.get("audio_out_port", 6353))
            except Exception:
                port = 6353
            try:
                sample_rate = int(CONFIG.get("audio_out_sample_rate", 22050))
            except Exception:
                sample_rate = 22050
            sample_rate = max(8000, sample_rate)
            try:
                channels = int(CONFIG.get("audio_out_channels", 1))
            except Exception:
                channels = 1
            channels = max(1, min(2, channels))

            # Keep cue stable if configured sample-rate cannot represent requested tone.
            safe_max_hz = (sample_rate * 0.5) - 120.0
            if safe_max_hz < 200.0:
                safe_max_hz = 200.0
            freq = min(float(frequency_hz), safe_max_hz)
            duration = max(0.02, float(duration_seconds))
            frame_count = max(1, int(sample_rate * duration))
            amplitude = 0.18
            omega = (2.0 * math.pi * freq) / float(sample_rate)

            pcm = bytearray(frame_count * channels * 2)
            pos = 0
            for i in range(frame_count):
                sample = int(32767.0 * amplitude * math.sin(omega * i))
                if sample > 32767:
                    sample = 32767
                if sample < -32768:
                    sample = -32768
                lo = sample & 0xFF
                hi = (sample >> 8) & 0xFF
                for _ in range(channels):
                    pcm[pos] = lo
                    pcm[pos + 1] = hi
                    pos += 2
            with socket.create_connection((host, port), timeout=0.25) as sock:
                sock.sendall(pcm)
        except Exception:
            # Cue tone is best-effort only; never block request handling.
            pass

    def _emit_pipeline_event(
        stage: str,
        state: str,
        text: str = "",
        detail: str = "",
        source: str = "llm_bridge",
        level: str = "info",
    ):
        try:
            event_url = str(CONFIG.get("pipeline_event_url", "")).strip()
            if not event_url:
                return
            payload = {
                "stage": str(stage or "").strip().lower(),
                "state": str(state or "").strip().lower(),
                "source": str(source or "llm_bridge").strip(),
                "text": str(text or "")[:320],
                "detail": str(detail or "")[:240],
                "level": str(level or "info").strip().lower() or "info",
            }
            requests.post(event_url, json=payload, timeout=0.45)
        except Exception:
            pass

    _TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
    _PUNCT_BOUNDARY_RE = re.compile(r"[.!?,;:…\n\r\u3002\uff01\uff1f\uff0c\uff1b\uff1a]+")
    _LLM_STREAM_EVENT_INTERVAL_SECONDS = 0.22

    def _estimate_token_count(text: str) -> int:
        raw = str(text or "")
        if not raw:
            return 0
        return len(_TOKEN_RE.findall(raw))

    def _extract_tts_chunks(buffer_text: str):
        """
        Split buffered model output on punctuation so TTS can start playback quickly.
        Returns (ready_chunks, remaining_tail_without_terminal_punctuation).
        """
        text = str(buffer_text or "")
        if not text:
            return [], ""
        chunks = []
        last = 0
        for match in _PUNCT_BOUNDARY_RE.finditer(text):
            end = match.end()
            candidate = text[last:end].strip()
            if candidate:
                chunks.append(candidate)
            last = end
        return chunks, text[last:]
    
    #############################################
    # Step 6: Load Optional Configurations       #
    #############################################
    
    def safe_load_json_file(path, default):
        if not path:
            return default
        if not os.path.exists(path):
            logging.warning(f"File '{path}' not found. Using default {default}.")
            if path == CONFIG["history"] and default == []:
                # Create empty history file
                try:
                    with open(path, 'w') as f:
                        json.dump([], f)
                    logging.info(f"Created empty history file at '{path}'.")
                except Exception as e:
                    logging.warning(f"Could not create history file '{path}': {e}")
            return default
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"Could not load '{path}': {e}. Using default {default}.")
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
                logging.warning(f"Could not load format schema from '{fmt}': {e}. Ignoring format.")
                return None
        else:
            logging.warning(f"Format file '{fmt}' not found. Ignoring format.")
            return None
    
    # Use absolute path for history
    history_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), CONFIG["history"])
    history_messages = safe_load_json_file(history_path, [])
    tools_data = safe_load_json_file(CONFIG["tools"], None)
    format_schema = load_format_schema(CONFIG["format"])
    
    #############################################
    # Step 7: Ensure Ollama and Model are Installed #
    #############################################

    try:
        _config_mtime = float(os.path.getmtime(CONFIG_FILE_ABS))
    except Exception:
        _config_mtime = 0.0

    MODEL_AVAILABLE = _ensure_ollama_model_ready(str(CONFIG.get("model", "")))
    if MODEL_AVAILABLE:
        logging.info(
            f"Configured model ready: {CONFIG.get('model')} "
            f"(thinking {'on' if _as_bool(CONFIG.get('thinking_enabled', False), False) else 'off'})"
        )
    else:
        logging.warning(
            f"Configured model '{CONFIG.get('model')}' is unavailable; requests may fail until pull completes."
        )
    
    #############################################
    # Step 8: Ollama chat interaction
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
        messages = []
        if CONFIG["system"]:
            messages.append({"role": "system", "content": CONFIG["system"]})
        
        # Truncate history_messages based on max_history_messages
        if CONFIG.get("max_history_messages"):
            # Ensure we have an even number of messages (user and assistant)
            # If odd, remove the oldest user message without a corresponding assistant message
            max_messages = CONFIG["max_history_messages"]
            if len(history_messages) > max_messages:
                # Slice the last max_messages messages
                truncated_history = history_messages[-max_messages:]
                # Ensure even number of messages for user-assistant pairs
                if len(truncated_history) % 2 != 0:
                    truncated_history = truncated_history[1:]
                messages.extend(truncated_history)
            else:
                messages.extend(history_messages)
        else:
            messages.extend(history_messages)
        
        messages.append({"role": "user", "content": user_message})
    
        payload = {
            "model": CONFIG["model"],
            "messages": messages,
            "stream": CONFIG["stream"],
            "think": _as_bool(CONFIG.get("thinking_enabled", False), False),
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
    # Step 9: Dedicated Worker Threads for Ollama and TTS
    #############################################
    
    # Initialize Queues for inter-thread and inter-process communication
    ollama_queue = Queue()
    tts_queue = Queue()
    inference_queue = multiprocessing.Queue()  # For inference process output
    
    # Dictionary to map request IDs to response queues
    response_dict = {}
    response_dict_lock = threading.Lock()
    request_id_counter = 0
    request_id_lock = threading.Lock()
    
    # Initialize a Lock for history updates to ensure thread safety
    history_lock = threading.Lock()
    active_inference_lock = threading.Lock()
    active_inference = {
        "process": None,
        "request_id": 0,
        "model": "",
        "started_at": 0.0,
    }

    def _set_active_inference(process, request_id: int, model_name: str):
        with active_inference_lock:
            active_inference["process"] = process
            active_inference["request_id"] = int(request_id or 0)
            active_inference["model"] = str(model_name or "").strip()
            active_inference["started_at"] = time.time()

    def _clear_active_inference(expected_process=None):
        with active_inference_lock:
            current = active_inference.get("process")
            if expected_process is not None and current is not expected_process:
                return
            active_inference["process"] = None
            active_inference["request_id"] = 0
            active_inference["model"] = ""
            active_inference["started_at"] = 0.0

    def _get_active_inference_snapshot():
        with active_inference_lock:
            process = active_inference.get("process")
            return {
                "process": process,
                "request_id": int(active_inference.get("request_id") or 0),
                "model": str(active_inference.get("model") or ""),
                "started_at": float(active_inference.get("started_at") or 0.0),
            }

    def _cancel_active_inference(reason: str = "model switch"):
        snapshot = _get_active_inference_snapshot()
        process = snapshot.get("process")
        if process is None:
            return False
        request_id = int(snapshot.get("request_id") or 0)
        model_name = str(snapshot.get("model") or "").strip() or "unknown"
        if process.is_alive():
            logging.warning(
                f"Cancelling active inference pid={process.pid} request_id={request_id} model={model_name} reason={reason}"
            )
            _emit_pipeline_event(
                "llm",
                "cancelled",
                detail=f"reason={reason} pid={process.pid} request_id={request_id} model={model_name}",
                level="warn",
            )
            try:
                process.terminate()
            except Exception:
                pass
            process.join(timeout=3.0)
            if process.is_alive() and hasattr(process, "kill"):
                try:
                    process.kill()
                except Exception:
                    pass
                process.join(timeout=2.0)
        _clear_active_inference(expected_process=process)
        return True

    def _discard_pending_prompts(reason: str = "model switch"):
        dropped = 0
        notified = 0
        while True:
            try:
                item = ollama_queue.get_nowait()
            except Empty:
                break
            except Exception:
                break
            if not isinstance(item, tuple) or len(item) != 2:
                continue
            req_id, _ = item
            if req_id is None:
                # Preserve shutdown sentinel.
                try:
                    ollama_queue.put(item)
                except Exception:
                    pass
                break
            dropped += 1
            response_queue = None
            with response_dict_lock:
                response_queue = response_dict.pop(req_id, None)
            if response_queue is not None:
                try:
                    response_queue.put("Prompt cancelled due to model switch.")
                    notified += 1
                except Exception:
                    pass
        if dropped > 0:
            logging.info(f"Dropped {dropped} pending prompt(s) due to {reason}; notified={notified}")
        return dropped, notified
    
    # Function to handle inter-process communication from inference processes to TTS queue
    def inference_to_tts_handler():
        while True:
            try:
                sentence = inference_queue.get()
                if sentence == "__SHUTDOWN__":
                    logging.info("Inference to TTS Handler: Received shutdown signal.")
                    break  # Sentinel to stop the thread
                # Append assistant sentence to history
                update_history("assistant", sentence)
                # Enqueue sentence to TTS queue
                tts_queue.put(sentence)
                _emit_pipeline_event("tts", "queued", text=sentence, detail="inference sentence enqueued")
                logging.debug(f"Inference to TTS Handler: Enqueued sentence to TTS: {sentence}")
            except Exception as e:
                logging.error(f"Inference to TTS Handler: {e}")
    
    # Start the inference_to_tts_handler thread
    inference_to_tts_thread = threading.Thread(target=inference_to_tts_handler, daemon=True, name="InferenceToTTSHandler")
    inference_to_tts_thread.start()
    logging.info("InferenceToTTSHandler: Started.")

    def _cpu_percent(interval=1.0):
        if PSUTIL_AVAILABLE and psutil is not None:
            try:
                return float(psutil.cpu_percent(interval=interval))
            except Exception:
                pass
        try:
            time.sleep(max(0.0, float(interval)))
        except Exception:
            time.sleep(0.1)
        return 0.0
    
    def ollama_worker():
        """
        Worker thread that processes messages from the Ollama queue.
        Processes one inference at a time to avoid interrupting active speech output.
        """
        while True:
            try:
                request_id, user_message = ollama_queue.get(timeout=1)  # Wait for 1 second
                if request_id is None and user_message is None:
                    logging.info("Ollama Worker: Received shutdown signal.")
                    _cancel_active_inference(reason="shutdown")
                    break
                logging.info(f"Ollama Worker: Received new prompt: {user_message}")
                current_model = str(CONFIG.get("model", "")).strip() or "unknown"
                thinking_state = "on" if _as_bool(CONFIG.get("thinking_enabled", False), False) else "off"
                _emit_pipeline_event(
                    "llm",
                    "queued",
                    text=user_message,
                    detail=f"model={current_model} thinking={thinking_state} request_id={request_id}",
                )

                response_queue = None
                with response_dict_lock:
                    response_queue = response_dict.pop(request_id, None)

                # Wait for any active inference to finish before starting a new prompt.
                snapshot = _get_active_inference_snapshot()
                active_process = snapshot.get("process")
                if active_process and active_process.is_alive():
                    logging.info("Ollama Worker: Waiting for active inference to complete before next prompt.")
                    active_process.join()
                    logging.info("Ollama Worker: Previous inference completed.")
                    _clear_active_inference(expected_process=active_process)

                # Start a new inference process for the new prompt
                logging.info("Ollama Worker: Starting new inference process.")
                current_inference_process = multiprocessing.Process(
                    target=inference_process,
                    args=(user_message, inference_queue)
                )
                current_inference_process.start()
                _set_active_inference(current_inference_process, request_id=request_id, model_name=current_model)
                _emit_pipeline_event(
                    "llm",
                    "processing",
                    text=user_message,
                    detail=f"model={current_model} thinking={thinking_state} pid={current_inference_process.pid}",
                )
                logging.info(f"Ollama Worker: Inference process started with PID {current_inference_process.pid}.")
                if response_queue is not None:
                    try:
                        response_queue.put("Prompt accepted.")
                    except Exception:
                        pass
                current_inference_process.join()
                exit_code = int(current_inference_process.exitcode or 0)
                _clear_active_inference(expected_process=current_inference_process)
                if exit_code != 0:
                    _emit_pipeline_event(
                        "llm",
                        "error",
                        detail=f"model={current_model} request_id={request_id} exit={exit_code}",
                        level="error",
                    )
                logging.info("Ollama Worker: Inference process finished.")

            except Empty:
                continue
            except Exception as e:
                _emit_pipeline_event("llm", "error", detail=f"worker: {e}", level="error")
                logging.error(f"Ollama Worker: Unexpected error: {e}")
    
    def inference_process(user_message, output_queue):
        """
        Function to handle inference in a separate process.
        Sends sentences to the output_queue for TTS.
        """
        model_name = str(CONFIG.get("model", "")).strip() or "unknown"
        stream_mode = _as_bool(CONFIG.get("stream", True), True)
        request_started_at = time.time()
        try:
            ollama_chat_url = str(CONFIG.get("ollama_url", OLLAMA_CHAT_URL)).strip() or OLLAMA_CHAT_URL
            payload = build_payload(user_message)
            model_name = str(payload.get("model") or model_name).strip() or "unknown"
            stream_mode = _as_bool(payload.get("stream", stream_mode), stream_mode)
            thinking_state = "on" if _as_bool(payload.get("think", False), False) else "off"
            _emit_pipeline_event(
                "llm",
                "processing",
                text=user_message,
                detail=f"model={model_name} thinking={thinking_state} stream={'on' if stream_mode else 'off'}",
            )

            if stream_mode:
                # Streaming response
                with requests.post(
                    ollama_chat_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    stream=True
                ) as r:
                    r.raise_for_status()
                    buffer = ""
                    full_response = ""
                    stream_started_at = 0.0
                    last_stream_event_at = 0.0
                    for line in r.iter_lines():
                        if not line:
                            continue
                        try:
                            obj = json.loads(line.decode('utf-8'))
                            if not isinstance(obj, dict):
                                continue
                            msg = obj.get("message", {})
                            content = str(msg.get("content", "") if isinstance(msg, dict) else "")
                            # Remove markdown emphasis markers that produce awkward speech.
                            content = content.replace('*', '')
                            done = bool(obj.get("done", False))

                            if content:
                                full_response += content
                                buffer += content
                                if stream_started_at <= 0.0:
                                    stream_started_at = time.time()

                                now = time.time()
                                token_count = _estimate_token_count(full_response)
                                elapsed = max(0.001, now - stream_started_at)
                                tps = float(token_count) / elapsed
                                preview = content.strip()
                                if preview and ((now - last_stream_event_at) >= _LLM_STREAM_EVENT_INTERVAL_SECONDS):
                                    _emit_pipeline_event(
                                        "llm",
                                        "streaming",
                                        text=preview,
                                        detail=f"model={model_name} tok={token_count} tps={tps:.2f}",
                                    )
                                    last_stream_event_at = now

                                ready_chunks, buffer = _extract_tts_chunks(buffer)
                                for sentence in ready_chunks:
                                    output_queue.put(sentence)
                                    logging.debug(f"Inference Process: Enqueued sentence to TTS: {sentence}")

                            if done:
                                leftover = buffer.strip()
                                if leftover:
                                    output_queue.put(leftover)
                                    logging.debug(f"Inference Process: Enqueued final sentence to TTS: {leftover}")
                                total_tokens = _estimate_token_count(full_response)
                                total_elapsed = max(0.001, time.time() - request_started_at)
                                total_tps = float(total_tokens) / total_elapsed
                                _emit_pipeline_event(
                                    "llm",
                                    "completed",
                                    text=full_response[-160:],
                                    detail=f"model={model_name} tok={total_tokens} tps={total_tps:.2f} stream=on",
                                )
                                break
                        except json.JSONDecodeError as e:
                            logging.error(f"Inference Process: Invalid JSON received: {e}")
                        except Exception as e:
                            logging.error(f"Inference Process: Error processing stream: {e}")
            else:
                # Non-streaming response
                r = requests.post(
                    ollama_chat_url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                r.raise_for_status()
                data = r.json()
                response_content = data.get("message", {}).get("content", "")
                # Remove markdown emphasis markers that produce awkward speech.
                response_content = str(response_content or "").replace('*', '')
                ready_chunks, buffer = _extract_tts_chunks(response_content)
                for sentence in ready_chunks:
                    output_queue.put(sentence)
                    logging.debug(f"Inference Process: Enqueued sentence to TTS: {sentence}")
                leftover = buffer.strip()
                if leftover:
                    output_queue.put(leftover)
                    logging.debug(f"Inference Process: Enqueued final sentence to TTS: {leftover}")
                total_tokens = _estimate_token_count(response_content)
                total_elapsed = max(0.001, time.time() - request_started_at)
                total_tps = float(total_tokens) / total_elapsed
                _emit_pipeline_event(
                    "llm",
                    "completed",
                    text=response_content[-160:],
                    detail=f"model={model_name} tok={total_tokens} tps={total_tps:.2f} stream=off",
                )
        except Exception as e:
            logging.error(f"Inference Process: Unexpected error: {e}")
            _emit_pipeline_event(
                "llm",
                "error",
                text=user_message,
                detail=f"model={model_name} stream={'on' if stream_mode else 'off'} err={e}",
                level="error",
            )
        # Removed output_queue.put(None)
        # The inference_to_tts_handler will be shutdown separately
    
    def tts_worker():
        """
        Worker thread that processes sentences from the TTS queue.
        """
        while True:
            try:
                sentence = tts_queue.get(timeout=1)  # Wait for 1 second
                if sentence is None:
                    logging.info("TTS Worker: Received shutdown signal.")
                    break
                logging.info(f"TTS Worker: Processing sentence: {sentence}")
                synthesize_and_play(sentence)
            except Empty:
                continue
            except Exception as e:
                logging.error(f"TTS Worker: Error processing sentence: {e}")
    
    def start_ollama_thread():
        """
        Start the Ollama worker thread.
        """
        ollama_thread = threading.Thread(target=ollama_worker, daemon=True, name="OllamaWorker")
        ollama_thread.start()
        logging.info("Ollama Worker: Started.")

    def start_tts_thread():
        """
        Start the TTS worker thread.
        """
        tts_thread = threading.Thread(target=tts_worker, daemon=True, name="TTSWorker")
        tts_thread.start()
        logging.info("TTS Worker: Started.")

    # Added synthesize_and_play function to handle TTS requests
    def synthesize_and_play(sentence):
        """
        Send the sentence to the TTS engine.
        Playback is handled by output.py through voice_server.py.
        """
        sentence = sentence.strip()
        if not sentence:
            return
        try:
            payload = {"prompt": sentence}
            tts_host = str(CONFIG.get("tts_host", "127.0.0.1")).strip() or "127.0.0.1"
            try:
                tts_port = int(CONFIG.get("tts_port", 6434))
            except Exception:
                tts_port = 6434
            logging.info(f"Forwarding TTS prompt to {tts_host}:{tts_port}")
            with socket.create_connection((tts_host, tts_port), timeout=10) as sock:
                sock.sendall(json.dumps(payload).encode("utf-8"))
            logging.info("TTS prompt forwarded successfully.")
            _emit_pipeline_event("tts", "forwarded", text=sentence, detail=f"tts_host={tts_host}:{tts_port}")
        except socket.timeout:
            logging.error("TTS socket request timed out.")
            _emit_pipeline_event("tts", "error", text=sentence, detail="tts socket timeout", level="error")
        except OSError as ce:
            logging.error(f"Connection error during TTS socket forward: {ce}")
            _emit_pipeline_event("tts", "error", text=sentence, detail=f"tts connection error: {ce}", level="error")
        except Exception as e:
            logging.error(f"Unexpected error during TTS: {e}")
            _emit_pipeline_event("tts", "error", text=sentence, detail=f"tts forward exception: {e}", level="error")

    def update_history(role, content):
        """
        Append a single message to the chat history.
        Only appends if the role is 'user' or 'assistant' and content is provided.
        """
        if not role or not content:
            return
        with history_lock:
            current_history = safe_load_json_file(history_path, [])
            current_history.append({"role": role, "content": content})
            try:
                with open(history_path, 'w') as f:
                    json.dump(current_history, f, indent=2)
                logging.info(f"History updated in '{history_path}' with {role} message.")
            except Exception as e:
                logging.warning(f"Could not write to history file {history_path}: {e}")

    #############################################
    # Step 10: Monitoring CPU Usage
    #############################################
    
    def monitor_cpu_usage(interval=5):
        """
        Monitor CPU usage at regular intervals and log it.
        Runs in a separate daemon thread.
        """
        while True:
            cpu_percent = _cpu_percent(interval=interval)
            logging.info(f"CPU Usage: {cpu_percent}%")

    #############################################
    # Step 11: Receiver Thread for Incoming Messages #
    #############################################
    
    def receiver_thread(host, port):
        """
        Dedicated thread to listen for incoming socket connections and receive messages asynchronously.
        Continuously retries to bind to the specified port until successful.
        """
        while True:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                server_socket.bind((host, port))
                logging.info(f"Receiver Thread: Bound to {host}:{port}")
                server_socket.listen(5)
                logging.info(f"Receiver Thread: Listening for incoming connections on {host}:{port}...")
                break  # Successfully bound and listening
            except Exception as e:
                logging.error(f"Receiver Thread: Failed to bind to {host}:{port} - {e}. Retrying in 5 seconds...")
                try:
                    server_socket.close()
                except:
                    pass
                time.sleep(5)  # Wait before retrying

        while True:
            try:
                client_sock, addr = server_socket.accept()
                logging.debug(f"Receiver Thread: Accepted connection from {addr}")
                # Start a new thread to handle the client
                client_handler = threading.Thread(
                    target=handle_client,
                    args=(client_sock, addr),
                    daemon=True,
                    name=f"ClientHandler-{addr}"
                )
                client_handler.start()
            except Exception as e:
                logging.error(f"Receiver Thread: Error accepting connections: {e}")
                # Continue accepting new connections
                continue

        server_socket.close()
        logging.info("Receiver Thread: Server socket closed.")

    def config_watcher_thread(interval_seconds=CONFIG_WATCH_INTERVAL_SECONDS):
        sleep_for = max(0.25, float(interval_seconds or CONFIG_WATCH_INTERVAL_SECONDS))
        while True:
            try:
                _reload_config_if_updated()
            except Exception as exc:
                logging.debug(f"Config watcher error: {exc}")
            time.sleep(sleep_for)

    def handle_client(client_sock, addr):
        """
        Handle individual client connections.
        """
        try:
            _reload_config_if_updated()
            data = client_sock.recv(65536)
            if not data:
                logging.debug(f"ClientHandler: No data from {addr}, closing connection.")
                client_sock.close()
                return
            user_message = data.decode('utf-8').strip()
            if not user_message:
                logging.debug(f"ClientHandler: Empty prompt from {addr}, ignoring.")
                client_sock.close()
                return
            logging.info(f"ClientHandler: Received prompt from {addr}: {user_message}")
            _emit_pipeline_event("asr", "captured", text=user_message, detail=f"source={addr[0]}")

            thinking_toggle = _parse_thinking_toggle_command(user_message)
            if thinking_toggle is not None:
                changed = _set_thinking_enabled(thinking_toggle)
                state_text = "On" if thinking_toggle else "Off"
                response_content = f"Turned Thinking {state_text}"
                if not changed:
                    response_content = f"Thinking already {state_text}"
                update_history("user", user_message)
                update_history("assistant", response_content)
                tts_queue.put(response_content)
                _emit_pipeline_event("llm", "toggle", text=user_message, detail=response_content)
                _emit_pipeline_event("tts", "queued", text=response_content, detail="thinking toggle feedback")
                try:
                    client_sock.sendall(response_content.encode("utf-8"))
                except Exception as send_exc:
                    logging.error(f"ClientHandler: Failed to send thinking toggle response: {send_exc}")
                return

            switch_request = _parse_model_switch_request(user_message)
            if switch_request is not None:
                available_models = _list_ollama_models()
                if not available_models:
                    response_content = "I cannot fetch model list right now. Please try again."
                    update_history("user", user_message)
                    update_history("assistant", response_content)
                    tts_queue.put(response_content)
                    _emit_pipeline_event("llm", "error", text=user_message, detail="model switch list unavailable", level="error")
                    _emit_pipeline_event("tts", "queued", text=response_content, detail="model switch failure feedback")
                    try:
                        client_sock.sendall(response_content.encode("utf-8"))
                    except Exception as send_exc:
                        logging.error(f"ClientHandler: Failed to send model switch response: {send_exc}")
                    return

                query_text = str(switch_request or "").strip()
                if not query_text:
                    with CONFIG_LOCK:
                        current_model = str(CONFIG.get("model", "")).strip() or "unknown"
                    preview = available_models[:4]
                    _set_pending_model_switch("", preview)
                    response_content = (
                        f"Current model is {current_model}. "
                        + _build_model_choice_prompt("", preview)
                    )
                    update_history("user", user_message)
                    update_history("assistant", response_content)
                    tts_queue.put(response_content)
                    _emit_pipeline_event("llm", "model_switch_prompt", detail=f"current={current_model} options={len(preview)}")
                    _emit_pipeline_event("tts", "queued", text=response_content, detail="model switch prompt")
                    try:
                        client_sock.sendall(response_content.encode("utf-8"))
                    except Exception as send_exc:
                        logging.error(f"ClientHandler: Failed to send model switch response: {send_exc}")
                    return

                candidates = _match_model_candidates(query_text, available_models, limit=4)
                if not candidates:
                    response_content = f"I could not find a close model match for {query_text}."
                    update_history("user", user_message)
                    update_history("assistant", response_content)
                    tts_queue.put(response_content)
                    _emit_pipeline_event("llm", "error", text=user_message, detail=f"model switch no match query={query_text}", level="error")
                    _emit_pipeline_event("tts", "queued", text=response_content, detail="model switch no-match feedback")
                    try:
                        client_sock.sendall(response_content.encode("utf-8"))
                    except Exception as send_exc:
                        logging.error(f"ClientHandler: Failed to send model switch response: {send_exc}")
                    return

                if len(candidates) == 1:
                    ok, response_content = _switch_model_to(candidates[0])
                    update_history("user", user_message)
                    update_history("assistant", response_content)
                    tts_queue.put(response_content)
                    if ok:
                        _clear_pending_model_switch()
                        _emit_pipeline_event("llm", "model_switch", detail=response_content)
                    else:
                        _emit_pipeline_event("llm", "error", detail=response_content, level="error")
                    _emit_pipeline_event("tts", "queued", text=response_content, detail="model switch feedback")
                    try:
                        client_sock.sendall(response_content.encode("utf-8"))
                    except Exception as send_exc:
                        logging.error(f"ClientHandler: Failed to send model switch response: {send_exc}")
                    return

                _set_pending_model_switch(query_text, candidates)
                response_content = _build_model_choice_prompt(query_text, candidates)
                update_history("user", user_message)
                update_history("assistant", response_content)
                tts_queue.put(response_content)
                _emit_pipeline_event(
                    "llm",
                    "model_switch_prompt",
                    text=user_message,
                    detail=f"query={query_text} candidates={','.join(candidates)}",
                )
                _emit_pipeline_event("tts", "queued", text=response_content, detail="model switch prompt")
                try:
                    client_sock.sendall(response_content.encode("utf-8"))
                except Exception as send_exc:
                    logging.error(f"ClientHandler: Failed to send model switch response: {send_exc}")
                return

            pending_model_switch = _get_pending_model_switch()
            if pending_model_switch:
                options = list(pending_model_switch.get("options") or [])
                choice_index = _parse_model_choice_index(user_message, len(options))
                if choice_index == -1:
                    _clear_pending_model_switch()
                    response_content = "Model switch cancelled."
                    update_history("user", user_message)
                    update_history("assistant", response_content)
                    tts_queue.put(response_content)
                    _emit_pipeline_event("llm", "model_switch_cancelled", text=user_message, detail="user cancelled")
                    _emit_pipeline_event("tts", "queued", text=response_content, detail="model switch cancel feedback")
                    try:
                        client_sock.sendall(response_content.encode("utf-8"))
                    except Exception as send_exc:
                        logging.error(f"ClientHandler: Failed to send model switch response: {send_exc}")
                    return
                if choice_index is None:
                    response_content = _build_model_choice_prompt(str(pending_model_switch.get("query") or ""), options)
                    update_history("user", user_message)
                    update_history("assistant", response_content)
                    tts_queue.put(response_content)
                    _emit_pipeline_event("llm", "model_switch_prompt", text=user_message, detail="awaiting numeric selection")
                    _emit_pipeline_event("tts", "queued", text=response_content, detail="model switch reprompt")
                    try:
                        client_sock.sendall(response_content.encode("utf-8"))
                    except Exception as send_exc:
                        logging.error(f"ClientHandler: Failed to send model switch response: {send_exc}")
                    return
                if not (0 <= choice_index < len(options)):
                    response_content = "Please choose one of the listed model numbers."
                    update_history("user", user_message)
                    update_history("assistant", response_content)
                    tts_queue.put(response_content)
                    _emit_pipeline_event("llm", "model_switch_prompt", text=user_message, detail="invalid numeric choice")
                    _emit_pipeline_event("tts", "queued", text=response_content, detail="model switch invalid choice")
                    try:
                        client_sock.sendall(response_content.encode("utf-8"))
                    except Exception as send_exc:
                        logging.error(f"ClientHandler: Failed to send model switch response: {send_exc}")
                    return

                selected_model = str(options[choice_index]).strip()
                ok, response_content = _switch_model_to(selected_model)
                if ok:
                    _clear_pending_model_switch()
                update_history("user", user_message)
                update_history("assistant", response_content)
                tts_queue.put(response_content)
                if ok:
                    _emit_pipeline_event("llm", "model_switch", detail=response_content)
                else:
                    _emit_pipeline_event("llm", "error", detail=response_content, level="error")
                _emit_pipeline_event("tts", "queued", text=response_content, detail="model switch selection feedback")
                try:
                    client_sock.sendall(response_content.encode("utf-8"))
                except Exception as send_exc:
                    logging.error(f"ClientHandler: Failed to send model switch response: {send_exc}")
                return

            # Append user message to history
            update_history("user", user_message)
            current_model = str(CONFIG.get("model", "")).strip()
            thinking_state = "on" if _as_bool(CONFIG.get("thinking_enabled", False), False) else "off"
            if current_model and not _ollama_model_installed(current_model):
                _ensure_ollama_model_ready(current_model)
            _emit_processing_blip(frequency_hz=6000.0, duration_seconds=0.2)
            _emit_pipeline_event(
                "llm",
                "processing",
                text=user_message,
                detail=f"model={current_model or 'unknown'} thinking={thinking_state} accepted",
            )

            # Generate a unique request ID
            with request_id_lock:
                global request_id_counter
                request_id_counter += 1
                request_id = request_id_counter

            # Create a response queue for this request
            response_queue = Queue()
            with response_dict_lock:
                response_dict[request_id] = response_queue

            # Enqueue the message to the Ollama worker
            ollama_queue.put((request_id, user_message))

            # Wait for the response from the Ollama worker
            try:
                response_content = response_queue.get(timeout=60)  # Wait up to 60 seconds
                # Send back the response to the client
                try:
                    client_sock.sendall(response_content.encode('utf-8'))
                    logging.info(f"ClientHandler: Sent response to {addr}.")
                except Exception as e:
                    logging.error(f"ClientHandler: Failed to send response to {addr}: {e}")
            except Empty:
                logging.error("ClientHandler: Timeout waiting for Ollama response.")
                response_content = "I'm sorry, I couldn't process your request at this time."
                # Send back the placeholder to the client
                try:
                    client_sock.sendall(response_content.encode('utf-8'))
                    logging.info(f"ClientHandler: Sent response to {addr}.")
                except Exception as e:
                    logging.error(f"ClientHandler: Failed to send response to {addr}: {e}")
                # Do not append to history
        except Exception as e:
            logging.error(f"ClientHandler: Unexpected error: {e}")
        finally:
            client_sock.close()
            logging.debug(f"ClientHandler: Connection with {addr} closed.")

    #############################################
    # Step 12: Server Handling with Dedicated Receiver Thread #
    #############################################
    
    def start_server():
        # Start TTS and Ollama worker threads
        logging.info("Starting TTS Worker...")
        start_tts_thread()
        logging.info("Starting Ollama Worker...")
        start_ollama_thread()
        watcher = threading.Thread(
            target=config_watcher_thread,
            args=(CONFIG_WATCH_INTERVAL_SECONDS,),
            daemon=True,
            name="ConfigWatcher",
        )
        watcher.start()
        logging.info("Config Watcher: Started.")

        # Start Inference to TTS Handler thread
        # Already started earlier

        # Start CPU usage monitoring
        cpu_monitor_thread = threading.Thread(target=monitor_cpu_usage, daemon=True, name="CPUMonitor")
        cpu_monitor_thread.start()
        logging.info("CPU Usage Monitor: Started.")

        # Start the receiver thread
        receiver = threading.Thread(target=receiver_thread, args=(CONFIG["host"], CONFIG["port"]), daemon=True, name="ReceiverThread")
        receiver.start()
        logging.info("Receiver Thread: Started.")

        # Keep the main thread alive to allow daemon threads to run
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logging.info("\nInterrupt received, shutting down server.")
        finally:
            # Stop worker threads
            logging.info("Stopping TTS Worker...")
            tts_queue.put(None)  # Signal TTS worker to exit
            logging.info("Stopping Ollama Worker...")
            ollama_queue.put((None, None))  # Signal Ollama worker to exit
            logging.info("Stopping Inference to TTS Handler...")
            inference_queue.put("__SHUTDOWN__")  # Signal inference_to_tts_handler to exit

            # Allow some time for threads to shutdown
            time.sleep(2)

            logging.info("Shutting down complete.")

if __name__ == "__main__":
    # Determine if running in offline mode
    if CONFIG.get("offline_mode", False):
        logging.info("Operating in offline mode.")
    else:
        ONLINE_STATUS = is_connected()
        if not ONLINE_STATUS:
            logging.warning("No internet connection detected. Switching to offline mode.")
            CONFIG["offline_mode"] = True
        else:
            CONFIG["offline_mode"] = False

    start_server()
