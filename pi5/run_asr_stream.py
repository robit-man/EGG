#!/usr/bin/env python3
import os
import sys


def main() -> int:
    base_dir = os.path.abspath(os.path.dirname(__file__))
    stream_script = os.path.join(base_dir, "whispercpp", "examples", "stream", "stream.py")
    venv_python = os.path.join(base_dir, "whispercpp", "whisper", "bin", "python")

    if not os.path.exists(stream_script):
        print(f"[ASR] Missing stream.py at {stream_script}", flush=True)
        return 1

    python_exe = venv_python if os.path.exists(venv_python) else sys.executable
    target_host = os.environ.get("ASR_TARGET_HOST", "127.0.0.1").strip() or "127.0.0.1"
    target_port = os.environ.get("ASR_TARGET_PORT", "6545").strip() or "6545"
    model_name = os.environ.get("ASR_MODEL_NAME", "tiny.en").strip() or "tiny.en"

    argv = [
        python_exe,
        stream_script,
        "--model_name",
        model_name,
        "--target_host",
        target_host,
        "--target_port",
        target_port,
    ]
    os.execv(python_exe, argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
