#!/usr/bin/env python3
import json
import os
import sys


CONFIG_PATH = "audio_router_config.json"


def _get_nested(data, path, default=None):
    current = data
    for key in str(path or "").split("."):
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


def _load_config(base_dir: str) -> dict:
    path = os.path.join(base_dir, CONFIG_PATH)
    try:
        with open(path, "r", encoding="utf-8") as fp:
            payload = json.load(fp)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _as_int(value, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def main() -> int:
    base_dir = os.path.abspath(os.path.dirname(__file__))
    stream_script = os.path.join(base_dir, "whispercpp", "examples", "stream", "stream.py")
    venv_python = os.path.join(base_dir, "whispercpp", "whisper", "bin", "python")
    cfg = _load_config(base_dir)

    if not os.path.exists(stream_script):
        print(f"[ASR] Missing stream.py at {stream_script}", flush=True)
        return 1

    python_exe = venv_python if os.path.exists(venv_python) else sys.executable
    target_host = (
        os.environ.get("ASR_TARGET_HOST", "").strip()
        or str(_get_nested(cfg, "audio_router.integrations.llm_host", "127.0.0.1")).strip()
        or "127.0.0.1"
    )
    target_port = _as_int(
        os.environ.get("ASR_TARGET_PORT", "").strip() or _get_nested(cfg, "audio_router.integrations.llm_port", 6545),
        6545,
    )
    model_name = os.environ.get("ASR_MODEL_NAME", "tiny.en").strip() or "tiny.en"
    device_id = _as_int(
        os.environ.get("ASR_DEVICE_ID", "").strip() or _get_nested(cfg, "audio_router.audio.asr_device_id", 0),
        0,
    )

    argv = [
        python_exe,
        stream_script,
        "--model_name",
        model_name,
        "--target_host",
        target_host,
        "--target_port",
        str(target_port),
        "--device_id",
        str(device_id),
    ]
    os.execv(python_exe, argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
