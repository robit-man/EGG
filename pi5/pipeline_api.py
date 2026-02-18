#!/usr/bin/env python3
"""
HTTP bridge for Pi5 speech stack:
- Exposes health for ASR/TTS/LLM chain.
- Provides HTTP entrypoints for LLM prompt and direct TTS synthesis.
- Supplies router discovery payloads (/router_info, /tunnel_info).
"""

import json
import os
import socket
import subprocess
import sys
import time


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


@app.route("/", methods=["GET"])
def index():
    return jsonify(
        {
            "status": "ok",
            "service": "pipeline_api",
            "routes": {
                "health": "/health",
                "list": "/list",
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
    return jsonify(
        {
            "status": "ok",
            "service": "pipeline_api",
            "uptime_seconds": round(time.time() - startup_time, 2),
            "ready": bool(stack["llm_bridge"]["ready"] and stack["tts_server"]["ready"] and stack["audio_output"]["ready"]),
            "stack": stack,
        }
    )


@app.route("/list", methods=["GET"])
def list_routes():
    local_base = f"http://127.0.0.1:{listen_port}"
    return jsonify(
        {
            "status": "success",
            "service": "pipeline_api",
            "endpoints": {
                "health_url": f"{local_base}/health",
                "llm_prompt_url": f"{local_base}/llm/prompt",
                "tts_speak_url": f"{local_base}/tts/speak",
                "router_info_url": f"{local_base}/router_info",
            },
        }
    )


@app.route("/llm/prompt", methods=["POST"])
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


@app.route("/tunnel_info", methods=["GET"])
def tunnel_info():
    return jsonify(
        {
            "status": "pending",
            "service": "pipeline_api",
            "message": "Tunnel URL is managed by router service",
            "tunnel_url": "",
        }
    )


@app.route("/router_info", methods=["GET"])
def router_info():
    local_base = f"http://127.0.0.1:{listen_port}"
    stack = _stack_status()
    return jsonify(
        {
            "status": "success",
            "service": "pipeline_api",
            "local": {
                "base_url": local_base,
                "listen_host": listen_host,
                "listen_port": listen_port,
                "list_url": f"{local_base}/list",
                "health_url": f"{local_base}/health",
                "llm_prompt_url": f"{local_base}/llm/prompt",
                "tts_speak_url": f"{local_base}/tts/speak",
            },
            "tunnel": {
                "state": "inactive",
                "tunnel_url": "",
                "list_url": "",
                "health_url": "",
                "llm_prompt_url": "",
                "tts_speak_url": "",
            },
            "stack": stack,
        }
    )


def main() -> int:
    app.run(host=listen_host, port=listen_port, debug=False, use_reloader=False, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
