#!/usr/bin/env python3
import os
import signal
import subprocess
import sys


def _which(binary: str) -> str:
    for part in os.environ.get("PATH", "").split(os.pathsep):
        candidate = os.path.join(part, binary)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return ""


def main() -> int:
    ollama = _which("ollama")
    if not ollama:
        print("[OLLAMA] executable not found in PATH", flush=True)
        return 1

    proc = subprocess.Popen([ollama, "serve"])

    def _handle_stop(_signum, _frame):
        try:
            if proc.poll() is None:
                proc.terminate()
        except Exception:
            pass

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)
    return proc.wait()


if __name__ == "__main__":
    raise SystemExit(main())
