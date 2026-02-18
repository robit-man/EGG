#!/usr/bin/env python3
"""
HTTP bridge for Pi5 speech stack:
- Exposes health for ASR/TTS/LLM chain.
- Provides HTTP entrypoints for LLM prompt and direct TTS synthesis.
- Supplies router discovery payloads (/router_info, /tunnel_info).
"""

import json
import os
import platform
import random
import re
import secrets
import socket
import subprocess
import sys
import threading
import time
from functools import wraps
from typing import Optional, Tuple


PIPELINE_VENV_DIR_NAME = "pipeline_api_venv"
CONFIG_PATH = "pipeline_api_config.json"


def ensure_venv() -> None:
    script_dir = os.path.abspath(os.path.dirname(__file__))
    venv_dir = os.path.join(script_dir, PIPELINE_VENV_DIR_NAME)
    if os.path.normcase(os.path.abspath(sys.prefix)) == os.path.normcase(os.path.abspath(venv_dir)):
        return

    if os.name == "nt":
        pip_path = os.path.join(venv_dir, "Scripts", "pip.exe")
        python_path = os.path.join(venv_dir, "Scripts", "python.exe")
    else:
        pip_path = os.path.join(venv_dir, "bin", "pip")
        python_path = os.path.join(venv_dir, "bin", "python")

    required = ["Flask", "Flask-CORS", "requests"]
    if not os.path.exists(venv_dir):
        import venv

        print(f"[PIPELINE] Creating virtual environment in '{PIPELINE_VENV_DIR_NAME}'...", flush=True)
        venv.create(venv_dir, with_pip=True)
        subprocess.check_call([pip_path, "install", *required])
    else:
        try:
            check = subprocess.run(
                [python_path, "-c", "import flask, flask_cors, requests"],
                capture_output=True,
                timeout=5,
            )
            if check.returncode != 0:
                subprocess.check_call([pip_path, "install", *required])
        except Exception:
            subprocess.check_call([pip_path, "install", *required])

    print("[PIPELINE] Re-launching from venv...", flush=True)
    os.execv(python_path, [python_path] + sys.argv)


ensure_venv()

import requests
from flask import Flask, jsonify, request
from flask_cors import CORS

UI_AVAILABLE = False
TerminalUI = None
ConfigSpec = None
CategorySpec = None
SettingSpec = None
ui = None

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
try:
    from terminal_ui import CategorySpec, ConfigSpec, SettingSpec, TerminalUI

    UI_AVAILABLE = True
except Exception:
    UI_AVAILABLE = False

DEFAULT_PASSWORD = "egg"
DEFAULT_SESSION_TIMEOUT = 300
DEFAULT_REQUIRE_AUTH = True
DEFAULT_ENABLE_TUNNEL = True
DEFAULT_AUTO_INSTALL_CLOUDFLARED = True
DEFAULT_TUNNEL_RESTART_DELAY_SECONDS = 3.0
DEFAULT_TUNNEL_RATE_LIMIT_DELAY_SECONDS = 45.0
MAX_TUNNEL_RESTART_DELAY_SECONDS = 300.0
PIPELINE_CLOUDFLARED_BASENAME = "pipeline_api_cloudflared"

DEFAULT_CONFIG = {
    "pipeline_api": {
        "network": {
            "listen_host": "0.0.0.0",
            "listen_port": 6590,
        },
        "endpoints": {
            "llm_host": "127.0.0.1",
            "llm_port": 6545,
            "tts_host": "127.0.0.1",
            "tts_port": 6434,
            "audio_out_host": "127.0.0.1",
            "audio_out_port": 6353,
            "ollama_health_url": "http://127.0.0.1:11434/api/tags",
        },
        "limits": {
            "prompt_max_chars": 2000,
            "socket_timeout_seconds": 8.0,
        },
        "security": {
            "password": DEFAULT_PASSWORD,
            "session_timeout": DEFAULT_SESSION_TIMEOUT,
            "require_auth": DEFAULT_REQUIRE_AUTH,
        },
        "tunnel": {
            "enable": DEFAULT_ENABLE_TUNNEL,
            "auto_install_cloudflared": DEFAULT_AUTO_INSTALL_CLOUDFLARED,
        },
    }
}


def _get_nested(data: dict, path: str, default=None):
    cur = data
    for key in path.split("."):
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur


def _set_nested(data: dict, path: str, value) -> None:
    cur = data
    keys = path.split(".")
    for key in keys[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[keys[-1]] = value


def _merge_defaults(config: dict, defaults: dict) -> dict:
    merged = json.loads(json.dumps(config))

    def walk(prefix: str, value):
        if isinstance(value, dict):
            for k, v in value.items():
                next_prefix = f"{prefix}.{k}" if prefix else k
                walk(next_prefix, v)
        else:
            if _get_nested(merged, prefix, None) is None:
                _set_nested(merged, prefix, value)

    walk("", defaults)
    return merged


def _as_bool(value, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _as_int(value, default: int, minimum: Optional[int] = None, maximum: Optional[int] = None) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = int(default)
    if minimum is not None:
        parsed = max(int(minimum), parsed)
    if maximum is not None:
        parsed = min(int(maximum), parsed)
    return parsed


def _as_float(value, default: float, minimum: Optional[float] = None, maximum: Optional[float] = None) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = float(default)
    if minimum is not None:
        parsed = max(float(minimum), parsed)
    if maximum is not None:
        parsed = min(float(maximum), parsed)
    return parsed


def load_config() -> dict:
    if not os.path.exists(CONFIG_PATH):
        save_config(DEFAULT_CONFIG)
        return json.loads(json.dumps(DEFAULT_CONFIG))
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as fp:
            loaded = json.load(fp)
        if not isinstance(loaded, dict):
            loaded = {}
    except Exception:
        loaded = {}
    merged = _merge_defaults(loaded, DEFAULT_CONFIG)
    host_value = str(_get_nested(merged, "pipeline_api.network.listen_host", "0.0.0.0")).strip().lower()
    if host_value in ("127.0.0.1", "localhost", "::1", ""):
        _set_nested(merged, "pipeline_api.network.listen_host", "0.0.0.0")
    save_config(merged)
    return merged


def save_config(cfg: dict) -> None:
    with open(CONFIG_PATH, "w", encoding="utf-8") as fp:
        json.dump(cfg, fp, indent=2)


cfg = load_config()
listen_host = str(_get_nested(cfg, "pipeline_api.network.listen_host", "0.0.0.0")).strip() or "0.0.0.0"
listen_port = int(_get_nested(cfg, "pipeline_api.network.listen_port", 6590) or 6590)
llm_host = str(_get_nested(cfg, "pipeline_api.endpoints.llm_host", "127.0.0.1")).strip() or "127.0.0.1"
llm_port = int(_get_nested(cfg, "pipeline_api.endpoints.llm_port", 6545) or 6545)
tts_host = str(_get_nested(cfg, "pipeline_api.endpoints.tts_host", "127.0.0.1")).strip() or "127.0.0.1"
tts_port = int(_get_nested(cfg, "pipeline_api.endpoints.tts_port", 6434) or 6434)
audio_out_host = str(_get_nested(cfg, "pipeline_api.endpoints.audio_out_host", "127.0.0.1")).strip() or "127.0.0.1"
audio_out_port = int(_get_nested(cfg, "pipeline_api.endpoints.audio_out_port", 6353) or 6353)
ollama_health_url = (
    str(_get_nested(cfg, "pipeline_api.endpoints.ollama_health_url", "http://127.0.0.1:11434/api/tags")).strip()
    or "http://127.0.0.1:11434/api/tags"
)
prompt_max_chars = int(_get_nested(cfg, "pipeline_api.limits.prompt_max_chars", 2000) or 2000)
socket_timeout = float(_get_nested(cfg, "pipeline_api.limits.socket_timeout_seconds", 8.0) or 8.0)
SESSION_TIMEOUT = _as_int(
    _get_nested(cfg, "pipeline_api.security.session_timeout", DEFAULT_SESSION_TIMEOUT),
    DEFAULT_SESSION_TIMEOUT,
    minimum=30,
    maximum=86400,
)
runtime_security = {
    "password": str(_get_nested(cfg, "pipeline_api.security.password", DEFAULT_PASSWORD)).strip() or DEFAULT_PASSWORD,
    "require_auth": _as_bool(
        _get_nested(cfg, "pipeline_api.security.require_auth", DEFAULT_REQUIRE_AUTH),
        default=DEFAULT_REQUIRE_AUTH,
    ),
}
sessions = {}
sessions_lock = threading.Lock()
tunnel_enabled = _as_bool(
    _get_nested(cfg, "pipeline_api.tunnel.enable", DEFAULT_ENABLE_TUNNEL),
    default=DEFAULT_ENABLE_TUNNEL,
)
auto_install_cloudflared = _as_bool(
    _get_nested(cfg, "pipeline_api.tunnel.auto_install_cloudflared", DEFAULT_AUTO_INSTALL_CLOUDFLARED),
    default=DEFAULT_AUTO_INSTALL_CLOUDFLARED,
)
tunnel_process = None
tunnel_url = None
tunnel_last_error = ""
tunnel_desired = False
tunnel_url_lock = threading.Lock()
tunnel_restart_lock = threading.Lock()
tunnel_restart_failures = 0
service_running = threading.Event()
startup_time = time.time()


def _tcp_ok(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def _http_ok(url: str, timeout: float = 1.5) -> bool:
    try:
        resp = requests.get(url, timeout=timeout)
        return 200 <= resp.status_code < 300
    except Exception:
        return False


def _stack_status() -> dict:
    llm_ready = _tcp_ok(llm_host, llm_port)
    tts_ready = _tcp_ok(tts_host, tts_port)
    audio_ready = _tcp_ok(audio_out_host, audio_out_port)
    ollama_ready = _http_ok(ollama_health_url)
    return {
        "llm_bridge": {
            "ready": llm_ready,
            "host": llm_host,
            "port": llm_port,
        },
        "tts_server": {
            "ready": tts_ready,
            "host": tts_host,
            "port": tts_port,
        },
        "audio_output": {
            "ready": audio_ready,
            "host": audio_out_host,
            "port": audio_out_port,
        },
        "ollama": {
            "ready": ollama_ready,
            "health_url": ollama_health_url,
        },
    }


def _send_llm_prompt(prompt: str) -> None:
    payload = prompt.encode("utf-8", errors="replace")
    with socket.create_connection((llm_host, llm_port), timeout=socket_timeout) as sock:
        sock.sendall(payload)


def _send_tts_prompt(prompt: str) -> None:
    payload = json.dumps({"prompt": prompt}).encode("utf-8")
    with socket.create_connection((tts_host, tts_port), timeout=socket_timeout) as sock:
        sock.sendall(payload)


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


def _resolve_lan_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("8.8.8.8", 80))
            candidate = str(probe.getsockname()[0] or "").strip()
            if candidate and not candidate.startswith("127."):
                return candidate
    except Exception:
        pass
    try:
        candidate = str(socket.gethostbyname(socket.gethostname()) or "").strip()
        if candidate and not candidate.startswith("127."):
            return candidate
    except Exception:
        pass
    return ""


def _endpoint_bases():
    local_base = f"http://127.0.0.1:{listen_port}"
    lan_ip = _resolve_lan_ip()
    lan_base = f"http://{lan_ip}:{listen_port}" if lan_ip else ""
    publish_base = lan_base or local_base
    return local_base, lan_base, publish_base


def _prune_expired_sessions(now: Optional[float] = None) -> int:
    current = float(now if now is not None else time.time())
    removed = 0
    with sessions_lock:
        expired = [key for key, entry in sessions.items() if current - float(entry.get("last_used", 0.0)) > SESSION_TIMEOUT]
        for key in expired:
            sessions.pop(key, None)
            removed += 1
    return removed


def _create_session() -> str:
    now = time.time()
    _prune_expired_sessions(now)
    key = secrets.token_urlsafe(32)
    with sessions_lock:
        sessions[key] = {"created_at": now, "last_used": now}
    return key


def _rotate_sessions() -> Tuple[str, int]:
    now = time.time()
    key = secrets.token_urlsafe(32)
    with sessions_lock:
        invalidated = len(sessions)
        sessions.clear()
        sessions[key] = {"created_at": now, "last_used": now}
    return key, invalidated


def _validate_session(session_key: str) -> bool:
    key = str(session_key or "").strip()
    if not key:
        return False
    now = time.time()
    with sessions_lock:
        entry = sessions.get(key)
        if not entry:
            return False
        last_used = float(entry.get("last_used", 0.0))
        if now - last_used > SESSION_TIMEOUT:
            sessions.pop(key, None)
            return False
        entry["last_used"] = now
    return True


def _get_session_key_from_request() -> str:
    key = str(request.args.get("session_key", "")).strip()
    if key:
        return key
    header_key = str(request.headers.get("X-Session-Key", "")).strip()
    if header_key:
        return header_key
    auth_header = str(request.headers.get("Authorization", "")).strip()
    if auth_header.lower().startswith("bearer "):
        candidate = auth_header[7:].strip()
        if candidate:
            return candidate
    data = request.get_json(silent=True)
    if isinstance(data, dict):
        body_key = str(data.get("session_key", "")).strip()
        if body_key:
            return body_key
    return ""


def _auth_required(handler):
    @wraps(handler)
    def wrapper(*args, **kwargs):
        if request.method == "OPTIONS":
            return ("", 204)
        if not runtime_security["require_auth"]:
            return handler(*args, **kwargs)
        session_key = _get_session_key_from_request()
        if not _validate_session(session_key):
            return jsonify({"status": "error", "message": "Unauthorized"}), 401
        return handler(*args, **kwargs)

    return wrapper


def _security_payload(base_url: str) -> dict:
    return {
        "require_auth": bool(runtime_security["require_auth"]),
        "session_timeout": int(SESSION_TIMEOUT),
        "auth_url": f"{base_url}/auth",
        "session_rotate_url": f"{base_url}/session/rotate",
    }


def _next_tunnel_restart_delay(rate_limited: bool = False) -> float:
    global tunnel_restart_failures
    tunnel_restart_failures = min(int(tunnel_restart_failures) + 1, 8)
    base_delay = (
        DEFAULT_TUNNEL_RATE_LIMIT_DELAY_SECONDS
        if rate_limited
        else DEFAULT_TUNNEL_RESTART_DELAY_SECONDS
    )
    delay = float(base_delay) * (2 ** max(0, int(tunnel_restart_failures) - 1))
    jitter = random.uniform(0.0, min(6.0, max(1.0, delay * 0.15)))
    return min(delay + jitter, MAX_TUNNEL_RESTART_DELAY_SECONDS)


def _get_cloudflared_path() -> str:
    if os.name == "nt":
        return os.path.join(SCRIPT_DIR, f"{PIPELINE_CLOUDFLARED_BASENAME}.exe")
    return os.path.join(SCRIPT_DIR, PIPELINE_CLOUDFLARED_BASENAME)


def _is_cloudflared_installed() -> bool:
    if os.path.exists(_get_cloudflared_path()):
        return True
    try:
        subprocess.run(["cloudflared", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _install_cloudflared() -> bool:
    cloudflared_path = _get_cloudflared_path()
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "windows":
        if "amd64" in machine or "x86_64" in machine:
            url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
        else:
            url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-386.exe"
    elif system == "linux":
        if "aarch64" in machine or "arm64" in machine:
            url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64"
        elif "arm" in machine:
            url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm"
        else:
            url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
    elif system == "darwin":
        if "arm" in machine:
            url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-arm64.tgz"
        else:
            url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-amd64.tgz"
    else:
        if ui:
            ui.log(f"[ERROR] Unsupported platform for cloudflared: {system} {machine}")
        return False
    try:
        import urllib.request

        if ui:
            ui.log(f"Downloading cloudflared from {url}")
        urllib.request.urlretrieve(url, cloudflared_path)
        if os.name != "nt":
            os.chmod(cloudflared_path, 0o755)
        if ui:
            ui.log("Installed cloudflared successfully")
        return True
    except Exception as exc:
        if ui:
            ui.log(f"[ERROR] Failed to install cloudflared: {exc}")
        return False


def _stop_cloudflared_tunnel() -> None:
    global tunnel_process, tunnel_last_error, tunnel_url, tunnel_desired, tunnel_restart_failures
    tunnel_desired = False
    tunnel_restart_failures = 0
    process = tunnel_process
    if process is None:
        with tunnel_url_lock:
            tunnel_url = None
        return
    try:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
    except Exception:
        pass
    finally:
        tunnel_process = None
        with tunnel_url_lock:
            tunnel_url = None
        tunnel_last_error = "Tunnel stopped"


def _start_cloudflared_tunnel(local_port: int) -> bool:
    global tunnel_url, tunnel_process, tunnel_last_error, tunnel_desired
    with tunnel_restart_lock:
        if tunnel_process is not None and tunnel_process.poll() is None:
            return True
        tunnel_desired = True
    cloudflared_path = _get_cloudflared_path()
    if not os.path.exists(cloudflared_path):
        cloudflared_path = "cloudflared"
    with tunnel_url_lock:
        tunnel_url = None
    tunnel_last_error = ""
    cmd = [
        cloudflared_path,
        "tunnel",
        "--protocol",
        "http2",
        "--url",
        f"http://localhost:{int(local_port)}",
    ]
    if ui:
        ui.log(f"[START] Launching cloudflared: {' '.join(cmd)}")
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        tunnel_process = process
    except Exception as exc:
        tunnel_last_error = str(exc)
        if ui:
            ui.log(f"[ERROR] Failed to start cloudflared tunnel: {exc}")
        return False

    def monitor_output() -> None:
        global tunnel_process, tunnel_url, tunnel_last_error, tunnel_restart_failures
        found_url = False
        captured_url = ""
        rate_limited = False
        for raw_line in iter(process.stdout.readline, ""):
            line = raw_line.strip()
            if not line:
                continue
            lowered = line.lower()
            if "429 too many requests" in lowered or "error code: 1015" in lowered:
                rate_limited = True
            if "trycloudflare.com" in line:
                match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
                if not match:
                    match = re.search(r"https://[^\s]+trycloudflare\.com[^\s]*", line)
                if match:
                    with tunnel_url_lock:
                        if tunnel_url is None:
                            captured_url = match.group(0)
                            tunnel_url = captured_url
                            found_url = True
                            tunnel_last_error = ""
                            tunnel_restart_failures = 0
                            if ui:
                                ui.log(f"[TUNNEL] Pipeline API URL: {tunnel_url}")
        return_code = process.poll()
        with tunnel_restart_lock:
            if tunnel_process is process:
                tunnel_process = None
        if captured_url:
            with tunnel_url_lock:
                if tunnel_url == captured_url:
                    tunnel_url = None
        if return_code is not None:
            if found_url:
                tunnel_restart_failures = 0
                tunnel_last_error = f"cloudflared exited (code {return_code}); tunnel URL expired"
            else:
                if rate_limited:
                    tunnel_last_error = f"cloudflared rate-limited (429/1015) before URL (code {return_code})"
                else:
                    tunnel_last_error = f"cloudflared exited before URL (code {return_code})"
            if ui:
                ui.log(f"[WARN] {tunnel_last_error}")
            if tunnel_desired and service_running.is_set():
                delay = _next_tunnel_restart_delay(rate_limited=rate_limited and not found_url)
                if ui:
                    ui.log(f"[WARN] Restarting cloudflared in {delay:.1f}s...")
                time.sleep(delay)
                if tunnel_desired and service_running.is_set():
                    _start_cloudflared_tunnel(local_port)

    threading.Thread(target=monitor_output, daemon=True).start()
    return True


def _tunnel_payload() -> dict:
    process_running = tunnel_process is not None and tunnel_process.poll() is None
    with tunnel_url_lock:
        current_tunnel = str(tunnel_url or "").strip() if process_running else ""
        stale_tunnel = str(tunnel_url or "").strip() if (tunnel_url and not process_running) else ""
        current_error = str(tunnel_last_error or "").strip()
    state = "active" if (process_running and current_tunnel) else ("starting" if process_running else "inactive")
    if stale_tunnel and not process_running:
        state = "stale"
    if current_error and not process_running and not current_tunnel and not stale_tunnel:
        state = "error"
    return {
        "state": state,
        "tunnel_url": current_tunnel,
        "stale_tunnel_url": stale_tunnel,
        "error": current_error,
        "running": bool(process_running),
        "enabled": bool(tunnel_enabled),
    }


@app.route("/", methods=["GET"])
def index():
    return jsonify(
        {
            "status": "ok",
            "service": "pipeline_api",
            "routes": {
                "health": "/health",
                "list": "/list",
                "auth": "/auth",
                "session_rotate": "/session/rotate",
                "router_info": "/router_info",
                "tunnel_info": "/tunnel_info",
                "llm_prompt": "/llm/prompt",
                "tts_speak": "/tts/speak",
            },
        }
    )


@app.route("/health", methods=["GET"])
def health():
    stack = _stack_status()
    local_base, lan_base, publish_base = _endpoint_bases()
    tunnel = _tunnel_payload()
    return jsonify(
        {
            "status": "ok",
            "service": "pipeline_api",
            "uptime_seconds": round(time.time() - startup_time, 2),
            "ready": bool(stack["llm_bridge"]["ready"] and stack["tts_server"]["ready"] and stack["audio_output"]["ready"]),
            "security": _security_payload(publish_base),
            "local_security": _security_payload(local_base),
            "lan_security": _security_payload(lan_base) if lan_base else {},
            "tunnel": tunnel,
            "stack": stack,
        }
    )


@app.route("/list", methods=["GET"])
@_auth_required
def list_routes():
    local_base, lan_base, publish_base = _endpoint_bases()
    tunnel = _tunnel_payload()
    tunnel_base = str(tunnel.get("tunnel_url") or "").strip()
    return jsonify(
        {
            "status": "success",
            "service": "pipeline_api",
            "base_url": publish_base,
            "local_base_url": local_base,
            "lan_base_url": lan_base,
            "security": _security_payload(publish_base),
            "local_security": _security_payload(local_base),
            "lan_security": _security_payload(lan_base) if lan_base else {},
            "routes": {
                "auth": "/auth",
                "session_rotate": "/session/rotate",
                "llm_prompt": "/llm/prompt",
                "tts_speak": "/tts/speak",
            },
            "endpoints": {
                "health_url": f"{publish_base}/health",
                "auth_url": f"{publish_base}/auth",
                "session_rotate_url": f"{publish_base}/session/rotate",
                "llm_prompt_url": f"{publish_base}/llm/prompt",
                "tts_speak_url": f"{publish_base}/tts/speak",
                "router_info_url": f"{publish_base}/router_info",
                "local_health_url": f"{local_base}/health",
                "local_auth_url": f"{local_base}/auth",
                "local_session_rotate_url": f"{local_base}/session/rotate",
                "local_llm_prompt_url": f"{local_base}/llm/prompt",
                "local_tts_speak_url": f"{local_base}/tts/speak",
                "lan_health_url": f"{lan_base}/health" if lan_base else "",
                "lan_auth_url": f"{lan_base}/auth" if lan_base else "",
                "lan_session_rotate_url": f"{lan_base}/session/rotate" if lan_base else "",
                "lan_llm_prompt_url": f"{lan_base}/llm/prompt" if lan_base else "",
                "lan_tts_speak_url": f"{lan_base}/tts/speak" if lan_base else "",
                "tunnel_health_url": f"{tunnel_base}/health" if tunnel_base else "",
                "tunnel_auth_url": f"{tunnel_base}/auth" if tunnel_base else "",
                "tunnel_session_rotate_url": f"{tunnel_base}/session/rotate" if tunnel_base else "",
                "tunnel_llm_prompt_url": f"{tunnel_base}/llm/prompt" if tunnel_base else "",
                "tunnel_tts_speak_url": f"{tunnel_base}/tts/speak" if tunnel_base else "",
            },
            "tunnel": tunnel,
        }
    )


@app.route("/llm/prompt", methods=["POST"])
@_auth_required
def llm_prompt():
    data = request.get_json(silent=True) or {}
    prompt = str(data.get("prompt", "")).strip()
    if not prompt:
        return jsonify({"status": "error", "message": "Missing prompt"}), 400
    if len(prompt) > prompt_max_chars:
        return jsonify({"status": "error", "message": "Prompt too long"}), 400
    try:
        _send_llm_prompt(prompt)
    except Exception as exc:
        return jsonify({"status": "error", "message": f"Failed to forward prompt: {exc}"}), 502
    return jsonify({"status": "success", "message": "Prompt forwarded"})


@app.route("/tts/speak", methods=["POST"])
@_auth_required
def tts_speak():
    data = request.get_json(silent=True) or {}
    prompt = str(data.get("prompt", "")).strip()
    if not prompt:
        return jsonify({"status": "error", "message": "Missing prompt"}), 400
    if len(prompt) > prompt_max_chars:
        return jsonify({"status": "error", "message": "Prompt too long"}), 400
    try:
        _send_tts_prompt(prompt)
    except Exception as exc:
        return jsonify({"status": "error", "message": f"TTS request failed: {exc}"}), 502
    return jsonify({"status": "success", "message": "TTS prompt forwarded"})


@app.route("/auth", methods=["POST"])
def auth():
    data = request.get_json(silent=True) or {}
    provided = str(data.get("password", "")).strip()
    if provided == str(runtime_security["password"]):
        session_key = _create_session()
        return jsonify(
            {
                "status": "success",
                "session_key": session_key,
                "timeout": int(SESSION_TIMEOUT),
                "require_auth": bool(runtime_security["require_auth"]),
            }
        )
    return jsonify({"status": "error", "message": "Invalid password"}), 401


@app.route("/session/rotate", methods=["POST"])
@_auth_required
def rotate_session():
    next_key, invalidated = _rotate_sessions()
    return jsonify(
        {
            "status": "success",
            "message": "Session keys rotated",
            "session_key": next_key,
            "invalidated_sessions": int(invalidated),
            "timeout": int(SESSION_TIMEOUT),
        }
    )


@app.route("/tunnel_info", methods=["GET"])
def tunnel_info():
    tunnel = _tunnel_payload()
    tunnel_base = str(tunnel.get("tunnel_url") or "").strip()
    if tunnel_base:
        status = "success"
        message = "Tunnel URL available"
    elif str(tunnel.get("state") or "") in ("error", "stale"):
        status = "error"
        message = "Tunnel unavailable"
    else:
        status = "pending"
        message = "Tunnel URL not yet available"
    return jsonify(
        {
            "status": status,
            "service": "pipeline_api",
            "message": message,
            "tunnel_url": tunnel_base,
            "auth_url": f"{tunnel_base}/auth" if tunnel_base else "",
            "session_rotate_url": f"{tunnel_base}/session/rotate" if tunnel_base else "",
            "list_url": f"{tunnel_base}/list" if tunnel_base else "",
            "health_url": f"{tunnel_base}/health" if tunnel_base else "",
            "llm_prompt_url": f"{tunnel_base}/llm/prompt" if tunnel_base else "",
            "tts_speak_url": f"{tunnel_base}/tts/speak" if tunnel_base else "",
            "stale_tunnel_url": str(tunnel.get("stale_tunnel_url") or ""),
            "error": str(tunnel.get("error") or ""),
            "running": bool(tunnel.get("running")),
        }
    )


@app.route("/router_info", methods=["GET"])
def router_info():
    local_base, lan_base, publish_base = _endpoint_bases()
    stack = _stack_status()
    tunnel = _tunnel_payload()
    tunnel_base = str(tunnel.get("tunnel_url") or "").strip()
    return jsonify(
        {
            "status": "success",
            "service": "pipeline_api",
            "local": {
                "base_url": publish_base,
                "loopback_base_url": local_base,
                "lan_base_url": lan_base,
                "listen_host": listen_host,
                "listen_port": listen_port,
                "auth_url": f"{publish_base}/auth",
                "session_rotate_url": f"{publish_base}/session/rotate",
                "list_url": f"{publish_base}/list",
                "health_url": f"{publish_base}/health",
                "llm_prompt_url": f"{publish_base}/llm/prompt",
                "tts_speak_url": f"{publish_base}/tts/speak",
                "local_auth_url": f"{local_base}/auth",
                "local_session_rotate_url": f"{local_base}/session/rotate",
                "local_list_url": f"{local_base}/list",
                "local_health_url": f"{local_base}/health",
                "local_llm_prompt_url": f"{local_base}/llm/prompt",
                "local_tts_speak_url": f"{local_base}/tts/speak",
                "lan_auth_url": f"{lan_base}/auth" if lan_base else "",
                "lan_session_rotate_url": f"{lan_base}/session/rotate" if lan_base else "",
                "lan_list_url": f"{lan_base}/list" if lan_base else "",
                "lan_health_url": f"{lan_base}/health" if lan_base else "",
                "lan_llm_prompt_url": f"{lan_base}/llm/prompt" if lan_base else "",
                "lan_tts_speak_url": f"{lan_base}/tts/speak" if lan_base else "",
            },
            "tunnel": {
                **tunnel,
                "auth_url": f"{tunnel_base}/auth" if tunnel_base else "",
                "session_rotate_url": f"{tunnel_base}/session/rotate" if tunnel_base else "",
                "list_url": f"{tunnel_base}/list" if tunnel_base else "",
                "health_url": f"{tunnel_base}/health" if tunnel_base else "",
                "llm_prompt_url": f"{tunnel_base}/llm/prompt" if tunnel_base else "",
                "tts_speak_url": f"{tunnel_base}/tts/speak" if tunnel_base else "",
            },
            "security": _security_payload(publish_base),
            "local_security": _security_payload(local_base),
            "lan_security": _security_payload(lan_base) if lan_base else {},
            "routes": {
                "auth": "/auth",
                "session_rotate": "/session/rotate",
                "llm_prompt": "/llm/prompt",
                "tts_speak": "/tts/speak",
            },
            "stack": stack,
        }
    )


def _apply_runtime_security(saved_config: dict) -> None:
    global SESSION_TIMEOUT, prompt_max_chars, socket_timeout
    runtime_security["password"] = (
        str(_get_nested(saved_config, "pipeline_api.security.password", runtime_security["password"])).strip()
        or DEFAULT_PASSWORD
    )
    runtime_security["require_auth"] = _as_bool(
        _get_nested(saved_config, "pipeline_api.security.require_auth", runtime_security["require_auth"]),
        default=runtime_security["require_auth"],
    )
    SESSION_TIMEOUT = _as_int(
        _get_nested(saved_config, "pipeline_api.security.session_timeout", SESSION_TIMEOUT),
        SESSION_TIMEOUT,
        minimum=30,
        maximum=86400,
    )
    prompt_max_chars = _as_int(
        _get_nested(saved_config, "pipeline_api.limits.prompt_max_chars", prompt_max_chars),
        prompt_max_chars,
        minimum=64,
        maximum=64000,
    )
    socket_timeout = _as_float(
        _get_nested(saved_config, "pipeline_api.limits.socket_timeout_seconds", socket_timeout),
        socket_timeout,
        minimum=0.1,
        maximum=120.0,
    )
    _prune_expired_sessions()
    if ui:
        ui.update_metric("Auth", "Required" if runtime_security["require_auth"] else "Disabled")
        ui.update_metric("Session Timeout", str(SESSION_TIMEOUT))
        ui.update_metric("Prompt Max", str(prompt_max_chars))
        ui.update_metric("Sock Timeout", f"{socket_timeout:.1f}s")
        ui.log("Applied live security updates from config save")


def _build_pipeline_config_spec():
    if not UI_AVAILABLE:
        return None
    return ConfigSpec(
        label="Pipeline API",
        categories=(
            CategorySpec(
                id="network",
                label="Network",
                settings=(
                    SettingSpec(
                        id="listen_host",
                        label="Listen Host",
                        path="pipeline_api.network.listen_host",
                        value_type="str",
                        default="0.0.0.0",
                        description="Bind host for pipeline API.",
                        restart_required=True,
                    ),
                    SettingSpec(
                        id="listen_port",
                        label="Listen Port",
                        path="pipeline_api.network.listen_port",
                        value_type="int",
                        default=6590,
                        min_value=1,
                        max_value=65535,
                        description="Bind port for pipeline API.",
                        restart_required=True,
                    ),
                ),
            ),
            CategorySpec(
                id="limits",
                label="Limits",
                settings=(
                    SettingSpec(
                        id="prompt_max_chars",
                        label="Prompt Max Chars",
                        path="pipeline_api.limits.prompt_max_chars",
                        value_type="int",
                        default=2000,
                        min_value=64,
                        max_value=64000,
                        description="Maximum accepted prompt length.",
                        restart_required=False,
                    ),
                    SettingSpec(
                        id="socket_timeout_seconds",
                        label="Socket Timeout",
                        path="pipeline_api.limits.socket_timeout_seconds",
                        value_type="float",
                        default=8.0,
                        min_value=0.1,
                        max_value=120.0,
                        description="Forwarding timeout for llm/tts sockets.",
                        restart_required=False,
                    ),
                ),
            ),
            CategorySpec(
                id="security",
                label="Security",
                settings=(
                    SettingSpec(
                        id="password",
                        label="Password",
                        path="pipeline_api.security.password",
                        value_type="secret",
                        default=DEFAULT_PASSWORD,
                        description="Password used by /auth.",
                        restart_required=False,
                    ),
                    SettingSpec(
                        id="session_timeout",
                        label="Session Timeout",
                        path="pipeline_api.security.session_timeout",
                        value_type="int",
                        default=DEFAULT_SESSION_TIMEOUT,
                        min_value=30,
                        max_value=86400,
                        description="Seconds before idle session keys expire.",
                        restart_required=False,
                    ),
                    SettingSpec(
                        id="require_auth",
                        label="Require Auth",
                        path="pipeline_api.security.require_auth",
                        value_type="bool",
                        default=DEFAULT_REQUIRE_AUTH,
                        description="Require session_key on protected routes.",
                        restart_required=False,
                    ),
                ),
            ),
            CategorySpec(
                id="tunnel",
                label="Tunnel",
                settings=(
                    SettingSpec(
                        id="enable_tunnel",
                        label="Enable Tunnel",
                        path="pipeline_api.tunnel.enable",
                        value_type="bool",
                        default=DEFAULT_ENABLE_TUNNEL,
                        description="Enable cloudflared trycloudflare tunnel.",
                        restart_required=True,
                    ),
                    SettingSpec(
                        id="auto_install_cloudflared",
                        label="Auto-install Cloudflared",
                        path="pipeline_api.tunnel.auto_install_cloudflared",
                        value_type="bool",
                        default=DEFAULT_AUTO_INSTALL_CLOUDFLARED,
                        description="Install cloudflared binary when missing.",
                        restart_required=True,
                    ),
                ),
            ),
        ),
    )


def _ui_metrics_loop() -> None:
    while ui and ui.running:
        stack = _stack_status()
        ready_count = sum(1 for item in stack.values() if isinstance(item, dict) and item.get("ready"))
        local_base, lan_base, publish_base = _endpoint_bases()
        tunnel = _tunnel_payload()
        ui.update_metric("Service", "pipeline_api")
        ui.update_metric("Bind", f"{listen_host}:{listen_port}")
        ui.update_metric("Local URL", local_base)
        ui.update_metric("LAN URL", lan_base or "N/A")
        ui.update_metric("Public URL", publish_base)
        ui.update_metric("Stack Ready", f"{ready_count}/{len(stack)}")
        ui.update_metric("LLM", "ready" if stack["llm_bridge"]["ready"] else "waiting")
        ui.update_metric("TTS", "ready" if stack["tts_server"]["ready"] else "waiting")
        ui.update_metric("Audio", "ready" if stack["audio_output"]["ready"] else "waiting")
        ui.update_metric("Ollama", "ready" if stack["ollama"]["ready"] else "waiting")
        ui.update_metric("Auth", "Required" if runtime_security["require_auth"] else "Disabled")
        ui.update_metric("Session Timeout", str(SESSION_TIMEOUT))
        ui.update_metric("Prompt Max", str(prompt_max_chars))
        ui.update_metric("Sock Timeout", f"{socket_timeout:.1f}s")
        ui.update_metric("Tunnel", str(tunnel.get("state") or "inactive"))
        ui.update_metric("Tunnel URL", str(tunnel.get("tunnel_url") or str(tunnel.get("stale_tunnel_url") or "N/A")))
        time.sleep(1.0)


def _shutdown_runtime() -> None:
    service_running.clear()
    _stop_cloudflared_tunnel()


def main() -> int:
    global ui
    try:
        service_running.set()
        if tunnel_enabled:
            if not _is_cloudflared_installed():
                if auto_install_cloudflared:
                    if ui:
                        ui.log("Cloudflared not found, attempting install...")
                    if not _install_cloudflared() and ui:
                        ui.log("Cloudflared install failed; tunnel disabled.")
                elif ui:
                    ui.log("Cloudflared missing and auto-install disabled; tunnel disabled.")
            if _is_cloudflared_installed():
                threading.Thread(
                    target=lambda: (time.sleep(2.0), _start_cloudflared_tunnel(listen_port)),
                    daemon=True,
                ).start()
        if UI_AVAILABLE:
            ui = TerminalUI(
                "Pipeline API",
                config_spec=_build_pipeline_config_spec(),
                config_path=CONFIG_PATH,
                refresh_interval_ms=700,
            )
            ui.on_save(_apply_runtime_security)
            ui.log(f"Starting pipeline API on {listen_host}:{listen_port}")
            local_base, lan_base, _ = _endpoint_bases()
            ui.log(f"Local URL: {local_base}")
            if lan_base:
                ui.log(f"LAN URL: {lan_base}")
            flask_thread = threading.Thread(
                target=lambda: app.run(
                    host=listen_host,
                    port=listen_port,
                    debug=False,
                    use_reloader=False,
                    threaded=True,
                ),
                daemon=True,
            )
            flask_thread.start()
            ui.running = True
            threading.Thread(target=_ui_metrics_loop, daemon=True).start()
            ui.start()
        else:
            app.run(host=listen_host, port=listen_port, debug=False, use_reloader=False, threaded=True)
        return 0
    finally:
        _shutdown_runtime()


if __name__ == "__main__":
    raise SystemExit(main())
