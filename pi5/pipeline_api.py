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
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit


PIPELINE_VENV_DIR_NAME = "pipeline_api_venv"
CONFIG_PATH = "pipeline_api_config.json"
LLM_BRIDGE_CONFIG_PATH = "llm_bridge_config.json"
DEFAULT_LLM_BRIDGE_CONFIG = {
    "model": "granite4:350m",
    "stream": True,
    "thinking_enabled": False,
    "max_history_messages": 3,
    "ollama_url": "http://127.0.0.1:11434/api/chat",
}


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
from flask import Flask, Response, jsonify, request
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
llm_pull_lock = threading.Lock()
llm_pull_state = {
    "running": False,
    "status": "idle",
    "model": "",
    "message": "",
    "error": "",
    "ollama_base_url": "",
    "started_at": 0.0,
    "finished_at": 0.0,
    "updated_at": 0.0,
    "events": [],
}
PIPELINE_EVENT_LIMIT = 240
pipeline_state_lock = threading.Lock()
pipeline_state = {
    "seq": 0,
    "updated_at": 0.0,
    "stages": {
        "asr": {"state": "idle", "updated_at": 0.0, "source": "", "detail": "", "last_text": ""},
        "llm": {"state": "idle", "updated_at": 0.0, "source": "", "detail": "", "last_text": ""},
        "tts": {"state": "idle", "updated_at": 0.0, "source": "", "detail": "", "last_text": ""},
    },
    "events": [],
}


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


def _llm_bridge_config_file() -> str:
    return os.path.join(SCRIPT_DIR, LLM_BRIDGE_CONFIG_PATH)


def _load_llm_bridge_config() -> Dict[str, Any]:
    merged = json.loads(json.dumps(DEFAULT_LLM_BRIDGE_CONFIG))
    path = _llm_bridge_config_file()
    if not os.path.exists(path):
        return merged
    try:
        with open(path, "r", encoding="utf-8") as fp:
            loaded = json.load(fp)
        if isinstance(loaded, dict):
            merged.update(loaded)
    except Exception:
        pass
    return merged


def _save_llm_bridge_config(config: Dict[str, Any]) -> None:
    with open(_llm_bridge_config_file(), "w", encoding="utf-8") as fp:
        json.dump(config, fp, indent=2)


def _resolve_ollama_base_url(llm_config: Optional[Dict[str, Any]] = None) -> str:
    candidates: List[str] = []
    if isinstance(llm_config, dict):
        candidates.append(str(llm_config.get("ollama_url", "")).strip())
    candidates.append(str(ollama_health_url).strip())
    candidates.append("http://127.0.0.1:11434/api/tags")

    for raw in candidates:
        if not raw:
            continue
        candidate = raw if "://" in raw else f"http://{raw.lstrip('/')}"
        parsed = urlsplit(candidate)
        if parsed.netloc:
            scheme = parsed.scheme or "http"
            return f"{scheme}://{parsed.netloc}".rstrip("/")
    return "http://127.0.0.1:11434"


def _fetch_ollama_models(llm_config: Optional[Dict[str, Any]] = None) -> Tuple[List[str], str, str]:
    base_url = _resolve_ollama_base_url(llm_config)
    tags_url = f"{base_url}/api/tags"
    try:
        response = requests.get(tags_url, timeout=12)
        response.raise_for_status()
        payload = response.json()
        models_raw = payload.get("models", []) if isinstance(payload, dict) else []
        models: List[str] = []
        seen = set()
        for item in models_raw:
            if not isinstance(item, dict):
                continue
            model_name = str(item.get("name", "")).strip()
            if model_name and model_name not in seen:
                seen.add(model_name)
                models.append(model_name)
        models.sort(key=lambda name: name.lower())
        return models, base_url, ""
    except Exception as exc:
        return [], base_url, str(exc)


def _llm_pull_snapshot() -> Dict[str, Any]:
    with llm_pull_lock:
        snapshot = json.loads(json.dumps(llm_pull_state))
    started_at = float(snapshot.get("started_at") or 0.0)
    finished_at = float(snapshot.get("finished_at") or 0.0)
    if started_at > 0:
        end = finished_at if finished_at > 0 else time.time()
        snapshot["elapsed_seconds"] = round(max(0.0, end - started_at), 2)
    else:
        snapshot["elapsed_seconds"] = 0.0
    return snapshot


def _set_llm_pull_state(**updates: Any) -> None:
    with llm_pull_lock:
        llm_pull_state.update(updates)
        llm_pull_state["updated_at"] = time.time()
        events = llm_pull_state.get("events")
        if isinstance(events, list) and len(events) > 32:
            llm_pull_state["events"] = events[-32:]


def _append_llm_pull_event(message: str) -> None:
    text = str(message or "").strip()
    if not text:
        return
    with llm_pull_lock:
        events = list(llm_pull_state.get("events") or [])
        events.append(
            {
                "timestamp": int(time.time() * 1000),
                "message": text,
            }
        )
        llm_pull_state["events"] = events[-32:]
        llm_pull_state["updated_at"] = time.time()


def _format_pull_progress(payload: Dict[str, Any]) -> str:
    status_text = str(payload.get("status", "")).strip()
    completed = payload.get("completed")
    total = payload.get("total")
    if isinstance(completed, (int, float)) and isinstance(total, (int, float)) and float(total) > 0:
        pct = max(0.0, min(100.0, (float(completed) / float(total)) * 100.0))
        if status_text:
            return f"{status_text} ({pct:.1f}%)"
        return f"{pct:.1f}%"
    if status_text:
        return status_text
    return json.dumps(payload, sort_keys=True)


def _ollama_pull_worker(model_name: str) -> None:
    started_at = time.time()
    llm_config = _load_llm_bridge_config()
    base_url = _resolve_ollama_base_url(llm_config)
    pull_url = f"{base_url}/api/pull"
    _set_llm_pull_state(
        running=True,
        status="running",
        model=model_name,
        message=f"Pulling '{model_name}'",
        error="",
        ollama_base_url=base_url,
        started_at=started_at,
        finished_at=0.0,
        events=[],
    )
    _append_llm_pull_event(f"pull start: {model_name}")
    try:
        with requests.post(
            pull_url,
            json={"model": model_name, "stream": True},
            headers={"Content-Type": "application/json"},
            stream=True,
            timeout=(12, 3600),
        ) as response:
            response.raise_for_status()
            saw_event = False
            for line in response.iter_lines(decode_unicode=True):
                text = str(line or "").strip()
                if not text:
                    continue
                saw_event = True
                progress = text
                try:
                    event = json.loads(text)
                    if isinstance(event, dict):
                        progress = _format_pull_progress(event)
                        event_error = str(event.get("error", "")).strip()
                        if event_error:
                            raise RuntimeError(event_error)
                except json.JSONDecodeError:
                    pass
                _set_llm_pull_state(status="running", message=progress, ollama_base_url=base_url)
                _append_llm_pull_event(progress)
            if not saw_event:
                _append_llm_pull_event("pull completed (no stream output)")
        _set_llm_pull_state(
            running=False,
            status="success",
            message=f"Model '{model_name}' is ready",
            error="",
            finished_at=time.time(),
            ollama_base_url=base_url,
        )
        _append_llm_pull_event(f"pull complete: {model_name}")
    except Exception as exc:
        _set_llm_pull_state(
            running=False,
            status="error",
            message=f"Model pull failed for '{model_name}'",
            error=str(exc),
            finished_at=time.time(),
            ollama_base_url=base_url,
        )
        _append_llm_pull_event(f"pull failed: {exc}")


def _start_ollama_pull(model_name: str) -> Tuple[bool, str, Dict[str, Any]]:
    model = str(model_name or "").strip()
    if not model:
        return False, "Missing model name", _llm_pull_snapshot()
    with llm_pull_lock:
        if llm_pull_state.get("running"):
            snapshot = json.loads(json.dumps(llm_pull_state))
            started_at = float(snapshot.get("started_at") or 0.0)
            finished_at = float(snapshot.get("finished_at") or 0.0)
            if started_at > 0:
                end = finished_at if finished_at > 0 else time.time()
                snapshot["elapsed_seconds"] = round(max(0.0, end - started_at), 2)
            else:
                snapshot["elapsed_seconds"] = 0.0
            return False, "Another pull is already running", snapshot
        llm_pull_state.update(
            {
                "running": True,
                "status": "starting",
                "model": model,
                "message": f"Starting pull for '{model}'",
                "error": "",
                "started_at": time.time(),
                "finished_at": 0.0,
                "updated_at": time.time(),
                "events": [],
            }
        )
    thread = threading.Thread(target=_ollama_pull_worker, args=(model,), daemon=True, name="OllamaPull")
    try:
        thread.start()
    except Exception as exc:
        _set_llm_pull_state(
            running=False,
            status="error",
            message="Failed to start pull worker",
            error=str(exc),
            finished_at=time.time(),
        )
        return False, f"Failed to start pull worker: {exc}", _llm_pull_snapshot()
    return True, f"Started pulling '{model}'", _llm_pull_snapshot()


def _llm_config_payload(llm_config: Dict[str, Any]) -> Dict[str, Any]:
    model = str(llm_config.get("model", DEFAULT_LLM_BRIDGE_CONFIG["model"])).strip() or str(
        DEFAULT_LLM_BRIDGE_CONFIG["model"]
    )
    stream = _as_bool(llm_config.get("stream"), default=True)
    thinking_enabled = _as_bool(
        llm_config.get("thinking_enabled", DEFAULT_LLM_BRIDGE_CONFIG["thinking_enabled"]),
        default=bool(DEFAULT_LLM_BRIDGE_CONFIG["thinking_enabled"]),
    )
    max_history_messages = _as_int(
        llm_config.get("max_history_messages", DEFAULT_LLM_BRIDGE_CONFIG["max_history_messages"]),
        int(DEFAULT_LLM_BRIDGE_CONFIG["max_history_messages"]),
        minimum=0,
        maximum=256,
    )
    ollama_url = str(llm_config.get("ollama_url", DEFAULT_LLM_BRIDGE_CONFIG["ollama_url"])).strip() or str(
        DEFAULT_LLM_BRIDGE_CONFIG["ollama_url"]
    )
    models, ollama_base_url, model_error = _fetch_ollama_models(llm_config)
    return {
        "model": model,
        "stream": stream,
        "thinking_enabled": thinking_enabled,
        "max_history_messages": max_history_messages,
        "ollama_url": ollama_url,
        "ollama_base_url": ollama_base_url,
        "available_models": models,
        "model_installed": model in set(models),
        "model_fetch_error": model_error,
    }


def _short_text(value: str, limit: int = 220) -> str:
    text = str(value or "").replace("\r", " ").replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _record_pipeline_event(
    stage: str,
    state: str,
    *,
    source: str = "pipeline_api",
    text: str = "",
    detail: str = "",
    level: str = "info",
) -> Dict[str, Any]:
    stage_key = str(stage or "").strip().lower()
    if stage_key not in ("asr", "llm", "tts"):
        stage_key = "llm"
    now = time.time()
    event = {}
    with pipeline_state_lock:
        pipeline_state["seq"] = int(pipeline_state.get("seq", 0)) + 1
        seq = int(pipeline_state["seq"])
        event = {
            "seq": seq,
            "timestamp": int(now * 1000),
            "stage": stage_key,
            "state": str(state or "").strip() or "event",
            "source": str(source or "pipeline_api").strip() or "pipeline_api",
            "text": _short_text(text, 320),
            "detail": _short_text(detail, 240),
            "level": str(level or "info").strip().lower() or "info",
        }
        stages = pipeline_state.get("stages", {})
        if isinstance(stages, dict):
            slot = stages.get(stage_key, {})
            if not isinstance(slot, dict):
                slot = {}
            slot.update(
                {
                    "state": event["state"],
                    "updated_at": now,
                    "source": event["source"],
                    "detail": event["detail"],
                    "last_text": event["text"],
                }
            )
            stages[stage_key] = slot
            pipeline_state["stages"] = stages
        events = list(pipeline_state.get("events", []))
        events.append(event)
        if len(events) > PIPELINE_EVENT_LIMIT:
            events = events[-PIPELINE_EVENT_LIMIT:]
        pipeline_state["events"] = events
        pipeline_state["updated_at"] = now

    rendered = f"[PIPELINE] {event['stage'].upper()}:{event['state']} ({event['source']})"
    if event["text"]:
        rendered += f" text=\"{event['text']}\""
    if event["detail"]:
        rendered += f" detail={event['detail']}"
    print(rendered, flush=True)
    if ui:
        try:
            ui.log(rendered)
        except Exception:
            pass
    return event


def _pipeline_snapshot() -> Dict[str, Any]:
    with pipeline_state_lock:
        snap = json.loads(json.dumps(pipeline_state))
    now_ms = int(time.time() * 1000)
    snap["now"] = now_ms
    return snap


def _is_local_request() -> bool:
    remote = str(request.remote_addr or "").strip().lower()
    if not remote:
        return False
    if remote == "::1" or remote.startswith("127.") or remote == "localhost":
        return True
    if remote.startswith("::ffff:127."):
        return True
    return False


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


def _llm_dashboard_html(session_key: str) -> str:
    template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>EGG LLM Dashboard</title>
  <style>
    :root {
      --bg: #111;
      --panel: #1b1b1b;
      --text: #f4f7ff;
      --muted: #9aa9c0;
      --line: #333;
      --accent: #2f7ef5;
      --ok: #00d08a;
      --err: #ff6666;
      --warn: #ffcc66;
      --subpanel: #161616;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: monospace;
      color: var(--text);
      background: var(--bg);
    }
    .wrap {
      max-width: 1080px;
      margin: 0 auto;
      padding: 16px;
      display: grid;
      gap: 12px;
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 14px;
    }
    h1 {
      margin: 0;
      font-size: 1.3rem;
      letter-spacing: 0.02em;
    }
    h2 {
      margin: 0 0 10px 0;
      font-size: 1.02rem;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 12px;
    }
    .row {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }
    label {
      display: block;
      font-size: 0.82rem;
      color: var(--muted);
      margin-bottom: 4px;
      font-weight: 600;
    }
    input, select, button {
      border-radius: 8px;
      border: 1px solid var(--line);
      padding: 9px 10px;
      font-size: 0.93rem;
    }
    input, select {
      width: 100%;
      background: var(--subpanel);
      color: var(--text);
    }
    button {
      background: var(--accent);
      color: #fff;
      border-color: var(--accent);
      font-weight: 600;
      cursor: pointer;
    }
    button.secondary {
      background: #222;
      color: var(--text);
      border-color: var(--line);
    }
    button:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }
    .pill {
      border-radius: 999px;
      padding: 2px 10px;
      border: 1px solid var(--line);
      font-size: 0.78rem;
      color: var(--muted);
      background: #222;
    }
    .ok { color: var(--ok); }
    .err { color: var(--err); }
    .warn { color: var(--warn); }
    .mono { font-family: Consolas, "Courier New", monospace; }
    .kv {
      display: grid;
      grid-template-columns: 160px 1fr;
      gap: 6px 10px;
      align-items: baseline;
      font-size: 0.9rem;
    }
    .log {
      max-height: 220px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--subpanel);
      color: #d6ecff;
      padding: 8px;
      font-family: Consolas, "Courier New", monospace;
      font-size: 0.78rem;
      line-height: 1.3;
      white-space: pre-wrap;
    }
    .stage-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 8px;
      margin-top: 8px;
    }
    .stage-card {
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 8px;
      background: var(--subpanel);
    }
    .stage-title {
      font-size: 0.8rem;
      color: var(--muted);
      text-transform: uppercase;
    }
    .stage-value {
      font-size: 0.96rem;
      margin-top: 4px;
      word-break: break-word;
    }
    .status {
      font-size: 0.88rem;
      color: var(--muted);
      min-height: 1.3em;
    }
    .muted {
      color: var(--muted);
      font-size: 0.84rem;
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <div class="row" style="justify-content: space-between;">
        <h1>LLM Model Control</h1>
        <span id="readyPill" class="pill">loading</span>
      </div>
      <div class="row" style="margin-top: 10px;">
        <div style="flex: 2 1 320px;">
          <label for="sessionKey">Session Key</label>
          <input id="sessionKey" class="mono" placeholder="Paste session_key from /auth" />
        </div>
        <div style="flex: 1 1 180px; min-width: 160px;">
          <label>&nbsp;</label>
          <button id="applySession" class="secondary" style="width: 100%;">Apply Session Key</button>
        </div>
        <div style="flex: 1 1 180px; min-width: 160px;">
          <label>&nbsp;</label>
          <button id="refreshAll" class="secondary" style="width: 100%;">Refresh</button>
        </div>
      </div>
      <div id="topStatus" class="status" style="margin-top: 8px;"></div>
      <div class="muted">Open this URL as <span class="mono">/llm/dashboard?session_key=...</span> when auth is required.</div>
    </div>

    <div class="grid">
      <div class="card">
        <h2>Current Config</h2>
        <div class="kv">
          <div>Model</div><div id="cfgModel" class="mono">-</div>
          <div>Stream</div><div id="cfgStream">-</div>
          <div>Thinking</div><div id="cfgThinking">-</div>
          <div>Max History</div><div id="cfgHistory">-</div>
          <div>Ollama URL</div><div id="cfgOllama" class="mono">-</div>
          <div>Ollama Base</div><div id="cfgOllamaBase" class="mono">-</div>
        </div>
      </div>
      <div class="card">
        <h2>Set Default Model</h2>
        <label for="modelSelect">Available Models</label>
        <select id="modelSelect"></select>
        <div class="row" style="margin-top: 8px;">
          <div style="flex: 1 1 140px;">
            <label for="streamFlag">Stream Responses</label>
            <select id="streamFlag">
              <option value="true">true</option>
              <option value="false">false</option>
            </select>
          </div>
          <div style="flex: 1 1 140px;">
            <label for="thinkingFlag">Thinking</label>
            <select id="thinkingFlag">
              <option value="false">off</option>
              <option value="true">on</option>
            </select>
          </div>
          <div style="flex: 1 1 140px;">
            <label for="maxHistory">Max History</label>
            <input id="maxHistory" type="number" min="0" max="256" step="1" />
          </div>
        </div>
        <div class="row" style="margin-top: 10px;">
          <button id="saveConfig">Save Config</button>
          <button id="reloadModels" class="secondary">Reload Models</button>
        </div>
      </div>
      <div class="card">
        <h2>Pull New Model</h2>
        <label for="pullName">Model Name</label>
        <input id="pullName" class="mono" placeholder="Example: qwen3:0.6b" />
        <div class="row" style="margin-top: 10px;">
          <button id="startPull">Pull Model</button>
          <button id="refreshPull" class="secondary">Refresh Pull Status</button>
        </div>
        <div id="pullSummary" class="status" style="overflow: auto; margin-top: 8px;"></div>
      </div>
    </div>

    <div class="card">
      <h2>Model Pull Log</h2>
      <div id="pullLog" class="log">No pull activity yet.</div>
    </div>

    <div class="card">
      <h2>Pipeline Stages</h2>
      <div class="stage-grid">
        <div class="stage-card">
          <div class="stage-title">ASR</div>
          <div id="asrState" class="stage-value">idle</div>
        </div>
        <div class="stage-card">
          <div class="stage-title">LLM</div>
          <div id="llmState" class="stage-value">idle</div>
        </div>
        <div class="stage-card">
          <div class="stage-title">TTS</div>
          <div id="ttsState" class="stage-value">idle</div>
        </div>
      </div>
      <div class="muted" style="margin-top: 8px;">Live event stream from ASR, LLM bridge, and TTS forwarding.</div>
      <div id="pipelineLog" class="log" style="margin-top: 8px;">No pipeline activity yet.</div>
    </div>
  </div>

  <script>
    const initialSessionKey = __SESSION_KEY_JSON__;
    let sessionKey = initialSessionKey || localStorage.getItem("egg_session_key") || "";
    let refreshTimer = null;

    const els = {
      sessionKey: document.getElementById("sessionKey"),
      applySession: document.getElementById("applySession"),
      refreshAll: document.getElementById("refreshAll"),
      topStatus: document.getElementById("topStatus"),
      readyPill: document.getElementById("readyPill"),
      cfgModel: document.getElementById("cfgModel"),
      cfgStream: document.getElementById("cfgStream"),
      cfgThinking: document.getElementById("cfgThinking"),
      cfgHistory: document.getElementById("cfgHistory"),
      cfgOllama: document.getElementById("cfgOllama"),
      cfgOllamaBase: document.getElementById("cfgOllamaBase"),
      modelSelect: document.getElementById("modelSelect"),
      streamFlag: document.getElementById("streamFlag"),
      thinkingFlag: document.getElementById("thinkingFlag"),
      maxHistory: document.getElementById("maxHistory"),
      saveConfig: document.getElementById("saveConfig"),
      reloadModels: document.getElementById("reloadModels"),
      pullName: document.getElementById("pullName"),
      startPull: document.getElementById("startPull"),
      refreshPull: document.getElementById("refreshPull"),
      pullSummary: document.getElementById("pullSummary"),
      pullLog: document.getElementById("pullLog"),
      asrState: document.getElementById("asrState"),
      llmState: document.getElementById("llmState"),
      ttsState: document.getElementById("ttsState"),
      pipelineLog: document.getElementById("pipelineLog"),
    };

    function setTopStatus(message, level) {
      els.topStatus.textContent = message || "";
      els.topStatus.className = "status";
      if (level === "error") els.topStatus.classList.add("err");
      if (level === "ok") els.topStatus.classList.add("ok");
      if (level === "warn") els.topStatus.classList.add("warn");
    }

    function withSession(url) {
      if (!sessionKey) return url;
      const sep = url.includes("?") ? "&" : "?";
      return `${url}${sep}session_key=${encodeURIComponent(sessionKey)}`;
    }

    async function fetchJson(url, options = {}) {
      const requestOptions = {
        method: options.method || "GET",
        headers: Object.assign({ "Accept": "application/json" }, options.headers || {}),
      };
      if (sessionKey) {
        requestOptions.headers["X-Session-Key"] = sessionKey;
      }
      if (options.body !== undefined) {
        requestOptions.headers["Content-Type"] = "application/json";
        requestOptions.body = JSON.stringify(options.body);
      }
      const response = await fetch(withSession(url), requestOptions);
      const bodyText = await response.text();
      let data = {};
      try {
        data = bodyText ? JSON.parse(bodyText) : {};
      } catch (err) {
        data = { message: bodyText || "Invalid response" };
      }
      if (!response.ok) {
        const msg = (data && (data.message || data.error)) || `${response.status} ${response.statusText}`;
        throw new Error(msg);
      }
      return data;
    }

    function setModels(models, selectedModel) {
      els.modelSelect.innerHTML = "";
      const list = Array.isArray(models) ? models : [];
      if (!list.length) {
        const option = document.createElement("option");
        option.value = "";
        option.textContent = "No models available";
        els.modelSelect.appendChild(option);
        return;
      }
      for (const name of list) {
        const option = document.createElement("option");
        option.value = name;
        option.textContent = name;
        if (name === selectedModel) {
          option.selected = true;
        }
        els.modelSelect.appendChild(option);
      }
    }

    function renderPullStatus(data) {
      const state = data || {};
      const status = String(state.status || "idle");
      const model = String(state.model || "");
      const message = String(state.message || "");
      const error = String(state.error || "");
      const elapsed = Number(state.elapsed_seconds || 0);
      const running = !!state.running;
      const summary = `${status}${model ? ` | ${model}` : ""}${elapsed ? ` | ${elapsed.toFixed(1)}s` : ""}${message ? ` | ${message}` : ""}`;
      els.pullSummary.textContent = summary;
      els.pullSummary.className = "status " + (status === "error" ? "err" : (running ? "warn" : "ok"));
      els.readyPill.textContent = running ? "pulling" : status;
      els.readyPill.className = "pill " + (status === "error" ? "err" : (running ? "warn" : "ok"));
      const events = Array.isArray(state.events) ? state.events : [];
      if (!events.length) {
        els.pullLog.textContent = error || "No pull activity yet.";
      } else {
        const lines = events.map((item) => {
          if (!item || typeof item !== "object") return String(item || "");
          const ts = Number(item.timestamp || 0);
          const date = ts > 0 ? new Date(ts).toLocaleTimeString() : "";
          return `${date}  ${String(item.message || "").trim()}`;
        });
        els.pullLog.textContent = lines.join("\\n");
      }
      els.startPull.disabled = running;
    }

    async function loadConfig() {
      const payload = await fetchJson("/llm/config");
      const cfg = payload.config || {};
      els.cfgModel.textContent = cfg.model || "-";
      els.cfgStream.textContent = String(cfg.stream);
      els.cfgThinking.textContent = cfg.thinking_enabled ? "on" : "off";
      els.cfgHistory.textContent = String(cfg.max_history_messages);
      els.cfgOllama.textContent = cfg.ollama_url || "-";
      els.cfgOllamaBase.textContent = cfg.ollama_base_url || "-";
      els.streamFlag.value = String(!!cfg.stream);
      els.thinkingFlag.value = String(!!cfg.thinking_enabled);
      els.maxHistory.value = String(cfg.max_history_messages ?? 0);
      if (cfg.model && !els.pullName.value) {
        els.pullName.value = cfg.model;
      }
      setModels(cfg.available_models || [], cfg.model || "");
      if (cfg.model && cfg.model_installed === false) {
        setTopStatus(`Selected model '${cfg.model}' is not installed yet. Pull it now.`, "warn");
      }
      if (cfg.model_fetch_error) {
        setTopStatus(`Model list fetch warning: ${cfg.model_fetch_error}`, "warn");
      }
      return payload;
    }

    async function loadModels() {
      const payload = await fetchJson("/llm/models");
      const models = payload.models || [];
      const selected = els.modelSelect.value || "";
      setModels(models, selected);
      return payload;
    }

    async function loadPullStatus() {
      const payload = await fetchJson("/llm/pull/status");
      renderPullStatus(payload.pull || {});
      return payload;
    }

    function formatPipelineStage(slot, fallbackState = "idle") {
      const state = String((slot && slot.state) || fallbackState || "idle");
      const src = String((slot && slot.source) || "").trim();
      const text = String((slot && slot.last_text) || "").trim();
      const detail = String((slot && slot.detail) || "").trim();
      let message = state;
      if (src) message += ` (${src})`;
      if (text) message += ` | ${text}`;
      else if (detail) message += ` | ${detail}`;
      return message;
    }

    async function loadPipelineState() {
      const payload = await fetchJson("/pipeline/state");
      const state = payload.pipeline_state || {};
      const stages = state.stages || {};
      els.asrState.textContent = formatPipelineStage(stages.asr, "idle");
      els.llmState.textContent = formatPipelineStage(stages.llm, "idle");
      els.ttsState.textContent = formatPipelineStage(stages.tts, "idle");
      const events = Array.isArray(state.events) ? state.events : [];
      if (!events.length) {
        els.pipelineLog.textContent = "No pipeline activity yet.";
      } else {
        const lines = events.slice(-120).map((evt) => {
          const ts = Number(evt.timestamp || 0);
          const dt = ts > 0 ? new Date(ts).toLocaleTimeString() : "";
          const stage = String(evt.stage || "pipeline").toUpperCase();
          const status = String(evt.state || "event");
          const src = String(evt.source || "");
          const text = String(evt.text || "").trim();
          const detail = String(evt.detail || "").trim();
          let line = `${dt} [${stage}] ${status}`;
          if (src) line += ` (${src})`;
          if (text) line += ` | ${text}`;
          if (detail) line += ` | ${detail}`;
          return line;
        });
        els.pipelineLog.textContent = lines.join("\\n");
      }
      return payload;
    }

    async function refreshAll() {
      try {
        setTopStatus("Refreshing LLM dashboard...", "warn");
        await loadConfig();
        await loadPullStatus();
        await loadPipelineState();
        setTopStatus("Dashboard updated.", "ok");
      } catch (err) {
        setTopStatus(`Refresh failed: ${err.message}`, "error");
      }
    }

    els.applySession.addEventListener("click", async () => {
      sessionKey = String(els.sessionKey.value || "").trim();
      if (sessionKey) {
        localStorage.setItem("egg_session_key", sessionKey);
        setTopStatus("Session key saved locally.", "ok");
      } else {
        localStorage.removeItem("egg_session_key");
        setTopStatus("Session key cleared.", "warn");
      }
      await refreshAll();
    });

    els.refreshAll.addEventListener("click", async () => {
      await refreshAll();
    });

    els.reloadModels.addEventListener("click", async () => {
      try {
        setTopStatus("Reloading model list...", "warn");
        await loadModels();
        setTopStatus("Model list updated.", "ok");
      } catch (err) {
        setTopStatus(`Model reload failed: ${err.message}`, "error");
      }
    });

    els.saveConfig.addEventListener("click", async () => {
      const model = String(els.modelSelect.value || "").trim();
      const stream = String(els.streamFlag.value || "true").toLowerCase() === "true";
      const thinkingEnabled = String(els.thinkingFlag.value || "false").toLowerCase() === "true";
      const maxHistory = Number(els.maxHistory.value || 0);
      if (!model) {
        setTopStatus("Select a model before saving.", "error");
        return;
      }
      try {
        setTopStatus("Saving LLM config...", "warn");
        await fetchJson("/llm/config", {
          method: "POST",
          body: {
            model,
            stream,
            thinking_enabled: thinkingEnabled,
            max_history_messages: maxHistory,
            auto_pull_missing: true,
          },
        });
        await refreshAll();
        setTopStatus("LLM config saved.", "ok");
      } catch (err) {
        setTopStatus(`Save failed: ${err.message}`, "error");
      }
    });

    els.startPull.addEventListener("click", async () => {
      const model = String(els.pullName.value || "").trim();
      if (!model) {
        setTopStatus("Enter model name to pull.", "error");
        return;
      }
      try {
        setTopStatus(`Starting model pull for ${model}...`, "warn");
        await fetchJson("/llm/pull", { method: "POST", body: { model } });
        await loadPullStatus();
        setTopStatus(`Pull started for ${model}.`, "ok");
      } catch (err) {
        setTopStatus(`Pull start failed: ${err.message}`, "error");
      }
    });

    els.refreshPull.addEventListener("click", async () => {
      try {
        await loadPullStatus();
        setTopStatus("Pull status refreshed.", "ok");
      } catch (err) {
        setTopStatus(`Pull status failed: ${err.message}`, "error");
      }
    });

    function startPolling() {
      if (refreshTimer) clearInterval(refreshTimer);
      refreshTimer = setInterval(async () => {
        try {
          await loadPullStatus();
          await loadPipelineState();
        } catch (err) {
          setTopStatus(`Polling failed: ${err.message}`, "warn");
        }
      }, 3000);
    }

    (async () => {
      els.sessionKey.value = sessionKey;
      await refreshAll();
      startPolling();
    })();
  </script>
</body>
</html>
"""
    return template.replace("__SESSION_KEY_JSON__", json.dumps(str(session_key or "")))


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
                "llm_dashboard": "/llm/dashboard",
                "llm_models": "/llm/models",
                "llm_config": "/llm/config",
                "llm_pull": "/llm/pull",
                "llm_pull_status": "/llm/pull/status",
                "pipeline_state": "/pipeline/state",
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
                "llm_dashboard": "/llm/dashboard",
                "llm_models": "/llm/models",
                "llm_config": "/llm/config",
                "llm_pull": "/llm/pull",
                "llm_pull_status": "/llm/pull/status",
                "pipeline_state": "/pipeline/state",
                "tts_speak": "/tts/speak",
            },
            "endpoints": {
                "health_url": f"{publish_base}/health",
                "auth_url": f"{publish_base}/auth",
                "session_rotate_url": f"{publish_base}/session/rotate",
                "llm_prompt_url": f"{publish_base}/llm/prompt",
                "llm_dashboard_url": f"{publish_base}/llm/dashboard",
                "llm_models_url": f"{publish_base}/llm/models",
                "llm_config_url": f"{publish_base}/llm/config",
                "llm_pull_url": f"{publish_base}/llm/pull",
                "llm_pull_status_url": f"{publish_base}/llm/pull/status",
                "pipeline_state_url": f"{publish_base}/pipeline/state",
                "tts_speak_url": f"{publish_base}/tts/speak",
                "router_info_url": f"{publish_base}/router_info",
                "local_health_url": f"{local_base}/health",
                "local_auth_url": f"{local_base}/auth",
                "local_session_rotate_url": f"{local_base}/session/rotate",
                "local_llm_prompt_url": f"{local_base}/llm/prompt",
                "local_llm_dashboard_url": f"{local_base}/llm/dashboard",
                "local_llm_models_url": f"{local_base}/llm/models",
                "local_llm_config_url": f"{local_base}/llm/config",
                "local_llm_pull_url": f"{local_base}/llm/pull",
                "local_llm_pull_status_url": f"{local_base}/llm/pull/status",
                "local_pipeline_state_url": f"{local_base}/pipeline/state",
                "local_tts_speak_url": f"{local_base}/tts/speak",
                "lan_health_url": f"{lan_base}/health" if lan_base else "",
                "lan_auth_url": f"{lan_base}/auth" if lan_base else "",
                "lan_session_rotate_url": f"{lan_base}/session/rotate" if lan_base else "",
                "lan_llm_prompt_url": f"{lan_base}/llm/prompt" if lan_base else "",
                "lan_llm_dashboard_url": f"{lan_base}/llm/dashboard" if lan_base else "",
                "lan_llm_models_url": f"{lan_base}/llm/models" if lan_base else "",
                "lan_llm_config_url": f"{lan_base}/llm/config" if lan_base else "",
                "lan_llm_pull_url": f"{lan_base}/llm/pull" if lan_base else "",
                "lan_llm_pull_status_url": f"{lan_base}/llm/pull/status" if lan_base else "",
                "lan_pipeline_state_url": f"{lan_base}/pipeline/state" if lan_base else "",
                "lan_tts_speak_url": f"{lan_base}/tts/speak" if lan_base else "",
                "tunnel_health_url": f"{tunnel_base}/health" if tunnel_base else "",
                "tunnel_auth_url": f"{tunnel_base}/auth" if tunnel_base else "",
                "tunnel_session_rotate_url": f"{tunnel_base}/session/rotate" if tunnel_base else "",
                "tunnel_llm_prompt_url": f"{tunnel_base}/llm/prompt" if tunnel_base else "",
                "tunnel_llm_dashboard_url": f"{tunnel_base}/llm/dashboard" if tunnel_base else "",
                "tunnel_llm_models_url": f"{tunnel_base}/llm/models" if tunnel_base else "",
                "tunnel_llm_config_url": f"{tunnel_base}/llm/config" if tunnel_base else "",
                "tunnel_llm_pull_url": f"{tunnel_base}/llm/pull" if tunnel_base else "",
                "tunnel_llm_pull_status_url": f"{tunnel_base}/llm/pull/status" if tunnel_base else "",
                "tunnel_pipeline_state_url": f"{tunnel_base}/pipeline/state" if tunnel_base else "",
                "tunnel_tts_speak_url": f"{tunnel_base}/tts/speak" if tunnel_base else "",
            },
            "tunnel": tunnel,
        }
    )


@app.route("/llm/dashboard", methods=["GET"])
@_auth_required
def llm_dashboard():
    session_key = _get_session_key_from_request()
    return Response(_llm_dashboard_html(session_key), mimetype="text/html")


@app.route("/llm/models", methods=["GET"])
@_auth_required
def llm_models():
    llm_config = _load_llm_bridge_config()
    models, base_url, error = _fetch_ollama_models(llm_config)
    model_selected = str(llm_config.get("model", "")).strip()
    if error:
        return (
            jsonify(
                {
                    "status": "error",
                    "service": "pipeline_api",
                    "message": f"Failed to fetch Ollama models: {error}",
                    "ollama_base_url": base_url,
                    "selected_model": model_selected,
                    "models": [],
                }
            ),
            502,
        )
    return jsonify(
        {
            "status": "success",
            "service": "pipeline_api",
            "ollama_base_url": base_url,
            "selected_model": model_selected,
            "models": models,
        }
    )


@app.route("/llm/config", methods=["GET", "POST"])
@_auth_required
def llm_config():
    if request.method == "GET":
        loaded = _load_llm_bridge_config()
        return jsonify(
            {
                "status": "success",
                "service": "pipeline_api",
                "config_path": _llm_bridge_config_file(),
                "config": _llm_config_payload(loaded),
            }
        )

    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({"status": "error", "message": "Invalid JSON payload"}), 400

    loaded = _load_llm_bridge_config()
    updates: Dict[str, Any] = {}

    if "model" in data:
        model_name = str(data.get("model", "")).strip()
        if not model_name:
            return jsonify({"status": "error", "message": "model cannot be empty"}), 400
        updates["model"] = model_name

    if "stream" in data:
        updates["stream"] = _as_bool(data.get("stream"), default=bool(loaded.get("stream", True)))

    if "thinking_enabled" in data:
        updates["thinking_enabled"] = _as_bool(
            data.get("thinking_enabled"),
            default=bool(loaded.get("thinking_enabled", DEFAULT_LLM_BRIDGE_CONFIG["thinking_enabled"])),
        )

    if "max_history_messages" in data:
        updates["max_history_messages"] = _as_int(
            data.get("max_history_messages"),
            int(loaded.get("max_history_messages", DEFAULT_LLM_BRIDGE_CONFIG["max_history_messages"])),
            minimum=0,
            maximum=256,
        )

    if "ollama_url" in data:
        ollama_url_value = str(data.get("ollama_url", "")).strip()
        if not ollama_url_value:
            return jsonify({"status": "error", "message": "ollama_url cannot be empty"}), 400
        updates["ollama_url"] = ollama_url_value

    if not updates:
        return jsonify({"status": "error", "message": "No supported config fields provided"}), 400

    loaded.update(updates)
    _save_llm_bridge_config(loaded)
    auto_pull_missing = _as_bool(data.get("auto_pull_missing", True), default=True)
    pull_result: Dict[str, Any] = {}
    config_view = _llm_config_payload(loaded)
    selected_model = str(config_view.get("model") or "").strip()
    model_installed = bool(config_view.get("model_installed"))
    if auto_pull_missing and selected_model and not model_installed:
        accepted, pull_message, pull_snapshot = _start_ollama_pull(selected_model)
        pull_result = {
            "status": "success" if accepted else "error",
            "message": pull_message,
            "pull": pull_snapshot,
        }

    return jsonify(
        {
            "status": "success",
            "service": "pipeline_api",
            "message": "LLM bridge configuration saved",
            "config_path": _llm_bridge_config_file(),
            "updated": updates,
            "config": config_view,
            "auto_pull_missing": bool(auto_pull_missing),
            "auto_pull": pull_result,
            "note": "llm_bridge hot-reloads config. Model changes cancel active inference and apply to the next ASR prompt.",
        }
    )


@app.route("/llm/pull", methods=["POST"])
@_auth_required
def llm_pull():
    data = request.get_json(silent=True) or {}
    model_name = ""
    if isinstance(data, dict):
        model_name = str(data.get("model") or data.get("name") or "").strip()
    accepted, message, pull_snapshot = _start_ollama_pull(model_name)
    if not accepted:
        code = 409 if "already" in str(message).lower() else 400
        return (
            jsonify(
                {
                    "status": "error",
                    "service": "pipeline_api",
                    "message": message,
                    "pull": pull_snapshot,
                }
            ),
            code,
        )
    return (
        jsonify(
            {
                "status": "success",
                "service": "pipeline_api",
                "message": message,
                "pull": pull_snapshot,
            }
        ),
        202,
    )


@app.route("/llm/pull/status", methods=["GET"])
@_auth_required
def llm_pull_status():
    return jsonify(
        {
            "status": "success",
            "service": "pipeline_api",
            "pull": _llm_pull_snapshot(),
        }
    )


@app.route("/pipeline/state", methods=["GET"])
@_auth_required
def pipeline_state_view():
    return jsonify(
        {
            "status": "success",
            "service": "pipeline_api",
            "pipeline_state": _pipeline_snapshot(),
        }
    )


@app.route("/pipeline/event", methods=["POST"])
def pipeline_event():
    if not _is_local_request():
        return jsonify({"status": "error", "message": "local requests only"}), 403
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({"status": "error", "message": "Invalid JSON payload"}), 400
    stage = str(data.get("stage", "")).strip().lower() or "llm"
    state = str(data.get("state", "")).strip().lower() or "event"
    source = str(data.get("source", "pipeline_agent")).strip() or "pipeline_agent"
    text = str(data.get("text", "")).strip()
    detail = str(data.get("detail", "")).strip()
    level = str(data.get("level", "info")).strip().lower() or "info"
    event = _record_pipeline_event(
        stage,
        state,
        source=source,
        text=text,
        detail=detail,
        level=level,
    )
    return jsonify({"status": "success", "service": "pipeline_api", "event": event})


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
        _record_pipeline_event(
            "llm",
            "queued",
            source="pipeline_api",
            text=prompt,
            detail="received via /llm/prompt",
        )
        _send_llm_prompt(prompt)
    except Exception as exc:
        _record_pipeline_event(
            "llm",
            "error",
            source="pipeline_api",
            text=prompt,
            detail=f"/llm/prompt forward failed: {exc}",
            level="error",
        )
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
        _record_pipeline_event(
            "tts",
            "requested",
            source="pipeline_api",
            text=prompt,
            detail="received via /tts/speak",
        )
        _send_tts_prompt(prompt)
    except Exception as exc:
        _record_pipeline_event(
            "tts",
            "error",
            source="pipeline_api",
            text=prompt,
            detail=f"/tts/speak forward failed: {exc}",
            level="error",
        )
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
            "llm_dashboard_url": f"{tunnel_base}/llm/dashboard" if tunnel_base else "",
            "llm_models_url": f"{tunnel_base}/llm/models" if tunnel_base else "",
            "llm_config_url": f"{tunnel_base}/llm/config" if tunnel_base else "",
            "llm_pull_url": f"{tunnel_base}/llm/pull" if tunnel_base else "",
            "llm_pull_status_url": f"{tunnel_base}/llm/pull/status" if tunnel_base else "",
            "pipeline_state_url": f"{tunnel_base}/pipeline/state" if tunnel_base else "",
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
                "llm_dashboard_url": f"{publish_base}/llm/dashboard",
                "llm_models_url": f"{publish_base}/llm/models",
                "llm_config_url": f"{publish_base}/llm/config",
                "llm_pull_url": f"{publish_base}/llm/pull",
                "llm_pull_status_url": f"{publish_base}/llm/pull/status",
                "pipeline_state_url": f"{publish_base}/pipeline/state",
                "tts_speak_url": f"{publish_base}/tts/speak",
                "local_auth_url": f"{local_base}/auth",
                "local_session_rotate_url": f"{local_base}/session/rotate",
                "local_list_url": f"{local_base}/list",
                "local_health_url": f"{local_base}/health",
                "local_llm_prompt_url": f"{local_base}/llm/prompt",
                "local_llm_dashboard_url": f"{local_base}/llm/dashboard",
                "local_llm_models_url": f"{local_base}/llm/models",
                "local_llm_config_url": f"{local_base}/llm/config",
                "local_llm_pull_url": f"{local_base}/llm/pull",
                "local_llm_pull_status_url": f"{local_base}/llm/pull/status",
                "local_pipeline_state_url": f"{local_base}/pipeline/state",
                "local_tts_speak_url": f"{local_base}/tts/speak",
                "lan_auth_url": f"{lan_base}/auth" if lan_base else "",
                "lan_session_rotate_url": f"{lan_base}/session/rotate" if lan_base else "",
                "lan_list_url": f"{lan_base}/list" if lan_base else "",
                "lan_health_url": f"{lan_base}/health" if lan_base else "",
                "lan_llm_prompt_url": f"{lan_base}/llm/prompt" if lan_base else "",
                "lan_llm_dashboard_url": f"{lan_base}/llm/dashboard" if lan_base else "",
                "lan_llm_models_url": f"{lan_base}/llm/models" if lan_base else "",
                "lan_llm_config_url": f"{lan_base}/llm/config" if lan_base else "",
                "lan_llm_pull_url": f"{lan_base}/llm/pull" if lan_base else "",
                "lan_llm_pull_status_url": f"{lan_base}/llm/pull/status" if lan_base else "",
                "lan_pipeline_state_url": f"{lan_base}/pipeline/state" if lan_base else "",
                "lan_tts_speak_url": f"{lan_base}/tts/speak" if lan_base else "",
            },
            "tunnel": {
                **tunnel,
                "auth_url": f"{tunnel_base}/auth" if tunnel_base else "",
                "session_rotate_url": f"{tunnel_base}/session/rotate" if tunnel_base else "",
                "list_url": f"{tunnel_base}/list" if tunnel_base else "",
                "health_url": f"{tunnel_base}/health" if tunnel_base else "",
                "llm_prompt_url": f"{tunnel_base}/llm/prompt" if tunnel_base else "",
                "llm_dashboard_url": f"{tunnel_base}/llm/dashboard" if tunnel_base else "",
                "llm_models_url": f"{tunnel_base}/llm/models" if tunnel_base else "",
                "llm_config_url": f"{tunnel_base}/llm/config" if tunnel_base else "",
                "llm_pull_url": f"{tunnel_base}/llm/pull" if tunnel_base else "",
                "llm_pull_status_url": f"{tunnel_base}/llm/pull/status" if tunnel_base else "",
                "pipeline_state_url": f"{tunnel_base}/pipeline/state" if tunnel_base else "",
                "tts_speak_url": f"{tunnel_base}/tts/speak" if tunnel_base else "",
            },
            "security": _security_payload(publish_base),
            "local_security": _security_payload(local_base),
            "lan_security": _security_payload(lan_base) if lan_base else {},
            "routes": {
                "auth": "/auth",
                "session_rotate": "/session/rotate",
                "llm_prompt": "/llm/prompt",
                "llm_dashboard": "/llm/dashboard",
                "llm_models": "/llm/models",
                "llm_config": "/llm/config",
                "llm_pull": "/llm/pull",
                "llm_pull_status": "/llm/pull/status",
                "pipeline_state": "/pipeline/state",
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
        _record_pipeline_event("asr", "idle", source="pipeline_api", detail="pipeline service startup")
        _record_pipeline_event("llm", "idle", source="pipeline_api", detail="pipeline service startup")
        _record_pipeline_event("tts", "idle", source="pipeline_api", detail="pipeline service startup")
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
