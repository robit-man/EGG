#!/usr/bin/env python3
import json
import os
import pathlib
import platform
import subprocess
import sys


CONFIG_PATH = "audio_router_config.json"
PIPELINE_CONFIG_PATH = "pipeline_api_config.json"


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


def _load_pipeline_config(base_dir: str) -> dict:
    path = os.path.join(base_dir, PIPELINE_CONFIG_PATH)
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


def _ensure_whispercpp_available(python_exe: str, base_dir: str) -> bool:
    def can_import() -> bool:
        try:
            check = subprocess.run(
                [python_exe, "-c", "import whispercpp as w; _ = w.api.SAMPLE_RATE"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            return check.returncode == 0
        except Exception:
            return False

    if can_import():
        return True

    whispercpp_dir = os.path.join(base_dir, "whispercpp")
    wheel_dir = os.path.join(whispercpp_dir, "dist")
    wheel_candidates = []
    try:
        machine = platform.machine().lower()
        py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
        preferred_tokens = []
        if "aarch64" in machine or "arm64" in machine:
            preferred_tokens = ["aarch64", "arm64"]
        elif "arm" in machine:
            preferred_tokens = ["armv7", "arm"]
        elif "x86_64" in machine or "amd64" in machine:
            preferred_tokens = ["x86_64", "amd64"]
        for entry in pathlib.Path(wheel_dir).glob("*.whl"):
            name = entry.name.lower()
            score = 0
            if py_tag in name:
                score += 5
            if any(token in name for token in preferred_tokens):
                score += 10
            wheel_candidates.append((score, str(entry)))
        wheel_candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    except Exception:
        wheel_candidates = []

    try:
        subprocess.run([python_exe, "-m", "ensurepip", "--upgrade"], check=False)
    except Exception:
        pass
    try:
        subprocess.run([python_exe, "-m", "pip", "install", "--upgrade", "pip"], check=False)
    except Exception:
        pass

    if wheel_candidates:
        for _, wheel_path in wheel_candidates:
            try:
                print(f"[ASR] Installing whispercpp wheel: {wheel_path}", flush=True)
                install = subprocess.run(
                    [python_exe, "-m", "pip", "install", "--force-reinstall", "--no-deps", wheel_path],
                    check=False,
                )
                if install.returncode == 0 and can_import():
                    return True
            except Exception:
                continue

    try:
        if os.path.isdir(whispercpp_dir):
            print("[ASR] Installing whispercpp from local source...", flush=True)
            install = subprocess.run([python_exe, "-m", "pip", "install", "-e", whispercpp_dir], check=False)
            if install.returncode == 0 and can_import():
                return True
    except Exception:
        pass

    return can_import()


def main() -> int:
    base_dir = os.path.abspath(os.path.dirname(__file__))
    stream_script = os.path.join(base_dir, "whispercpp", "examples", "stream", "stream.py")
    venv_python = os.path.join(base_dir, "whispercpp", "whisper", "bin", "python")
    cfg = _load_config(base_dir)
    pipeline_cfg = _load_pipeline_config(base_dir)

    if not os.path.exists(stream_script):
        print(f"[ASR] Missing stream.py at {stream_script}", flush=True)
        return 1

    python_exe = venv_python if os.path.exists(venv_python) else sys.executable
    if not _ensure_whispercpp_available(python_exe, base_dir):
        print("[ASR] whispercpp import/install failed in runtime python environment", flush=True)
        return 1
    target_host = (
        os.environ.get("ASR_TARGET_HOST", "").strip()
        or str(_get_nested(cfg, "audio_router.integrations.llm_host", "127.0.0.1")).strip()
        or "127.0.0.1"
    )
    target_port = _as_int(
        os.environ.get("ASR_TARGET_PORT", "").strip() or _get_nested(cfg, "audio_router.integrations.llm_port", 6545),
        6545,
    )
    cue_host = (
        os.environ.get("ASR_CUE_HOST", "").strip()
        or str(_get_nested(cfg, "audio_router.integrations.audio_out_host", "127.0.0.1")).strip()
        or "127.0.0.1"
    )
    cue_port = _as_int(
        os.environ.get("ASR_CUE_PORT", "").strip() or _get_nested(cfg, "audio_router.integrations.audio_out_port", 6353),
        6353,
    )
    cue_rate = _as_int(
        os.environ.get("ASR_CUE_RATE", "").strip() or _get_nested(cfg, "audio_router.audio.output_sample_rate", 22050),
        22050,
    )
    cue_channels = _as_int(
        os.environ.get("ASR_CUE_CHANNELS", "").strip() or _get_nested(cfg, "audio_router.audio.output_channels", 1),
        1,
    )
    pipeline_event_host = (
        os.environ.get("ASR_PIPELINE_EVENT_HOST", "").strip()
        or str(_get_nested(pipeline_cfg, "pipeline_api.network.listen_host", "127.0.0.1")).strip()
        or "127.0.0.1"
    )
    if pipeline_event_host in ("0.0.0.0", "::"):
        pipeline_event_host = "127.0.0.1"
    pipeline_event_port = _as_int(
        os.environ.get("ASR_PIPELINE_EVENT_PORT", "").strip()
        or _get_nested(pipeline_cfg, "pipeline_api.network.listen_port", 6590),
        6590,
    )
    model_name = os.environ.get("ASR_MODEL_NAME", "tiny.en").strip() or "tiny.en"
    device_id = _as_int(
        os.environ.get("ASR_DEVICE_ID", "").strip() or _get_nested(cfg, "audio_router.audio.asr_device_id", 0),
        0,
    )

    env = os.environ.copy()
    # Keep runtime imports wheel-first; source path fallback is opt-in only.
    if str(env.get("ASR_USE_SOURCE_PATH", "")).strip().lower() in ("1", "true", "yes", "on"):
        extra_path = os.path.join(base_dir, "whispercpp", "src")
        if os.path.isdir(extra_path):
            existing = str(env.get("PYTHONPATH", "")).strip()
            env["PYTHONPATH"] = f"{extra_path}{os.pathsep}{existing}" if existing else extra_path

    argv = [
        python_exe,
        stream_script,
        "--model_name",
        model_name,
        "--target_host",
        target_host,
        "--target_port",
        str(target_port),
        "--asr_blip_host",
        cue_host,
        "--asr_blip_port",
        str(cue_port),
        "--asr_blip_rate",
        str(cue_rate),
        "--asr_blip_channels",
        str(cue_channels),
        "--pipeline_event_host",
        pipeline_event_host,
        "--pipeline_event_port",
        str(pipeline_event_port),
        "--device_id",
        str(device_id),
    ]
    os.execve(python_exe, argv, env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
