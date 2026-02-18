#!/usr/bin/env python3
import json
import os
import re
import socket
import subprocess
import threading
import time
from queue import Queue


CONFIG_PATH = "audio_router_config.json"
PIPER_EXECUTABLE = "/opt/piper/build/piper"
PIPER_MODEL_PATH = "/opt/voice/glados_piper_medium.onnx"

DEFAULT_TTS_LISTEN_HOST = "0.0.0.0"
DEFAULT_TTS_LISTEN_PORT = 6434
DEFAULT_RAW_AUDIO_HOST = "127.0.0.1"
DEFAULT_RAW_AUDIO_PORT = 6353
DEFAULT_CONNECT_TIMEOUT_SECONDS = 3.0
DEFAULT_MAX_CONNECT_ATTEMPTS = 4
DEFAULT_RETRY_DELAY_SECONDS = 0.7
VERBOSE_CONNECTION_LOGS = str(os.environ.get("VOICE_SERVER_VERBOSE_CONNECTION_LOGS", "0")).strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)


def _log(message):
    print(str(message), flush=True)


def _debug(message):
    if VERBOSE_CONNECTION_LOGS:
        _log(message)


def _get_nested(data, path, default=None):
    current = data
    for key in str(path or "").split("."):
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


def _as_int(value, default, minimum=None, maximum=None):
    try:
        parsed = int(value)
    except Exception:
        parsed = int(default)
    if minimum is not None and parsed < minimum:
        return int(default)
    if maximum is not None and parsed > maximum:
        return int(default)
    return parsed


def _as_float(value, default, minimum=None, maximum=None):
    try:
        parsed = float(value)
    except Exception:
        parsed = float(default)
    if minimum is not None and parsed < minimum:
        return float(default)
    if maximum is not None and parsed > maximum:
        return float(default)
    return parsed


def _load_runtime_settings():
    cfg_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), CONFIG_PATH)
    payload = {}
    try:
        with open(cfg_path, "r", encoding="utf-8") as fp:
            loaded = json.load(fp)
        if isinstance(loaded, dict):
            payload = loaded
    except Exception:
        payload = {}

    return {
        "tts_listen_host": str(
            _get_nested(payload, "audio_router.audio.tts_listen_host", DEFAULT_TTS_LISTEN_HOST)
        ).strip()
        or DEFAULT_TTS_LISTEN_HOST,
        "tts_listen_port": _as_int(
            _get_nested(payload, "audio_router.integrations.tts_port", DEFAULT_TTS_LISTEN_PORT),
            DEFAULT_TTS_LISTEN_PORT,
            minimum=1,
            maximum=65535,
        ),
        "raw_audio_host": str(
            _get_nested(payload, "audio_router.integrations.audio_out_host", DEFAULT_RAW_AUDIO_HOST)
        ).strip()
        or DEFAULT_RAW_AUDIO_HOST,
        "raw_audio_port": _as_int(
            _get_nested(payload, "audio_router.integrations.audio_out_port", DEFAULT_RAW_AUDIO_PORT),
            DEFAULT_RAW_AUDIO_PORT,
            minimum=1,
            maximum=65535,
        ),
        "connect_timeout_seconds": _as_float(
            _get_nested(payload, "audio_router.audio.output_connect_timeout_seconds", DEFAULT_CONNECT_TIMEOUT_SECONDS),
            DEFAULT_CONNECT_TIMEOUT_SECONDS,
            minimum=0.2,
            maximum=20.0,
        ),
        "connect_attempts": _as_int(
            _get_nested(payload, "audio_router.audio.output_connect_attempts", DEFAULT_MAX_CONNECT_ATTEMPTS),
            DEFAULT_MAX_CONNECT_ATTEMPTS,
            minimum=1,
            maximum=20,
        ),
        "retry_delay_seconds": _as_float(
            _get_nested(payload, "audio_router.audio.output_connect_retry_delay_seconds", DEFAULT_RETRY_DELAY_SECONDS),
            DEFAULT_RETRY_DELAY_SECONDS,
            minimum=0.1,
            maximum=10.0,
        ),
    }


def _extract_prompt(raw_data: str) -> str:
    message = str(raw_data or "").strip()
    if not message:
        return ""
    try:
        parsed = json.loads(message)
        if isinstance(parsed, dict):
            return str(parsed.get("response") or parsed.get("prompt") or parsed.get("text") or "").strip()
    except Exception:
        pass
    json_match = re.search(r"\{.*\}", message, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            if isinstance(parsed, dict):
                return str(parsed.get("response") or parsed.get("prompt") or parsed.get("text") or "").strip()
        except Exception:
            pass
    return message


def _send_audio_to_output(raw_audio: bytes, settings: dict) -> bool:
    host = str(settings.get("raw_audio_host") or DEFAULT_RAW_AUDIO_HOST)
    port = int(settings.get("raw_audio_port") or DEFAULT_RAW_AUDIO_PORT)
    timeout = float(settings.get("connect_timeout_seconds") or DEFAULT_CONNECT_TIMEOUT_SECONDS)
    attempts = int(settings.get("connect_attempts") or DEFAULT_MAX_CONNECT_ATTEMPTS)
    retry_delay = float(settings.get("retry_delay_seconds") or DEFAULT_RETRY_DELAY_SECONDS)

    for attempt in range(1, attempts + 1):
        try:
            with socket.create_connection((host, port), timeout=timeout) as audio_socket:
                audio_socket.sendall(raw_audio)
                return True
        except Exception as exc:
            _log(f"Raw audio send failed attempt={attempt}/{attempts} {host}:{port}: {exc}")
            if attempt < attempts:
                time.sleep(retry_delay)
    return False


def tts_worker(queue: Queue):
    while True:
        text_content = queue.get()
        if text_content is None:
            queue.task_done()
            break
        try:
            _debug(f"Processing text: {text_content}")
            process = subprocess.Popen(
                [
                    PIPER_EXECUTABLE,
                    "--model",
                    PIPER_MODEL_PATH,
                    "--output_raw",
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout, stderr = process.communicate(input=text_content.encode("utf-8"))
            if stderr:
                _log(f"Piper error: {stderr.decode('utf-8', errors='replace')}")

            settings = _load_runtime_settings()
            if _send_audio_to_output(stdout or b"", settings):
                _debug("Raw audio forwarded to output service.")
            else:
                _log("Raw audio forwarding failed after retries.")
        except Exception as exc:
            _log(f"TTS worker error: {exc}")
        finally:
            queue.task_done()


def handle_client_connection(client_socket: socket.socket, queue: Queue):
    try:
        while True:
            data = client_socket.recv(4096)
            if not data:
                break
            prompt = _extract_prompt(data.decode("utf-8", errors="replace"))
            if not prompt:
                continue
            queue.put(prompt)
    except Exception as exc:
        _log(f"Error handling client: {exc}")
    finally:
        try:
            client_socket.close()
        except Exception:
            pass


def main():
    tts_queue: Queue = Queue()
    worker_thread = threading.Thread(target=tts_worker, args=(tts_queue,), daemon=True)
    worker_thread.start()

    text_server = None
    try:
        while True:
            try:
                settings = _load_runtime_settings()
                if text_server is None:
                    text_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    text_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    text_server.bind((settings["tts_listen_host"], int(settings["tts_listen_port"])))
                    text_server.listen(8)
                    _log(
                        f"Listening for text content on "
                        f"{settings['tts_listen_host']}:{settings['tts_listen_port']} "
                        f"(audio->{settings['raw_audio_host']}:{settings['raw_audio_port']})..."
                    )

                client_socket, addr = text_server.accept()
                _debug(f"Connection established with {addr}")
                threading.Thread(
                    target=handle_client_connection,
                    args=(client_socket, tts_queue),
                    daemon=True,
                ).start()
            except Exception as exc:
                _log(f"Server loop error: {exc}")
                if text_server:
                    try:
                        text_server.close()
                    except Exception:
                        pass
                    text_server = None
                time.sleep(2.0)
    except KeyboardInterrupt:
        _log("Shutting down server...")
    finally:
        try:
            tts_queue.put(None)
            tts_queue.join()
        except Exception:
            pass
        try:
            if text_server:
                text_server.close()
        except Exception:
            pass
        _log("Server closed.")


if __name__ == "__main__":
    main()
