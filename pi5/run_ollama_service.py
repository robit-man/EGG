#!/usr/bin/env python3
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request


OLLAMA_HEALTH_URL = "http://127.0.0.1:11434/api/tags"
POLL_SECONDS = 1.5


def _which(binary: str) -> str:
    for part in os.environ.get("PATH", "").split(os.pathsep):
        candidate = os.path.join(part, binary)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return ""


def _ollama_ready(timeout: float = 1.5) -> bool:
    try:
        with urllib.request.urlopen(OLLAMA_HEALTH_URL, timeout=max(0.1, float(timeout))) as resp:
            return 200 <= int(getattr(resp, "status", 0) or 0) < 300
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, ValueError):
        return False


def _hold_existing_service(stop_requested) -> int:
    print("[OLLAMA] existing service detected; entering keepalive wrapper", flush=True)
    while not stop_requested["value"]:
        time.sleep(POLL_SECONDS)
    return 0


def main() -> int:
    ollama = _which("ollama")
    if not ollama:
        print("[OLLAMA] executable not found in PATH", flush=True)
        return 1

    stop_requested = {"value": False}

    if _ollama_ready():
        def _handle_stop_existing(_signum, _frame):
            stop_requested["value"] = True

        signal.signal(signal.SIGINT, _handle_stop_existing)
        signal.signal(signal.SIGTERM, _handle_stop_existing)
        return _hold_existing_service(stop_requested)

    proc = subprocess.Popen([ollama, "serve"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    def _handle_stop(_signum, _frame):
        stop_requested["value"] = True
        try:
            if proc.poll() is None:
                proc.terminate()
        except Exception:
            pass

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    code = proc.wait()
    if code == 0:
        return 0

    output = ""
    try:
        if proc.stdout:
            output = proc.stdout.read() or ""
    except Exception:
        output = ""

    lower = output.lower()
    already_running = ("address already in use" in lower) or ("bind" in lower and "in use" in lower)
    if already_running and _ollama_ready():
        return _hold_existing_service(stop_requested)

    if _ollama_ready():
        return _hold_existing_service(stop_requested)

    if output.strip():
        print(f"[OLLAMA] serve exited with code {code}: {output.strip()}", flush=True)
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
