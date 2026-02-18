#!/usr/bin/env python3
import os
import signal
import subprocess
import sys
import time


CONTAINER_NAME = "piper-tts-pi5-service"


def _docker(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )


def _remove_existing_container() -> None:
    _docker("rm", "-f", CONTAINER_NAME)


def main() -> int:
    image = os.environ.get("VOICE_DOCKER_IMAGE", "piper-tts-rpi5").strip() or "piper-tts-rpi5"
    base_dir = os.path.abspath(os.path.dirname(__file__))

    if not shutil_which("docker"):
        print("[TTS] docker is not installed or not in PATH", flush=True)
        return 1

    _remove_existing_container()

    cmd = [
        "docker",
        "run",
        "--name",
        CONTAINER_NAME,
        "--network",
        "host",
        "-v",
        f"{base_dir}:/opt/voice",
        "-w",
        "/opt/voice",
        "--rm",
        image,
        "python3",
        "voice_server.py",
    ]

    proc = subprocess.Popen(cmd)

    def _handle_stop(signum, _frame):
        try:
            _docker("rm", "-f", CONTAINER_NAME)
        except Exception:
            pass
        try:
            if proc.poll() is None:
                proc.terminate()
        except Exception:
            pass

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    try:
        return proc.wait()
    finally:
        time.sleep(0.1)
        _remove_existing_container()


def shutil_which(binary: str) -> str:
    paths = os.environ.get("PATH", "").split(os.pathsep)
    for path in paths:
        candidate = os.path.join(path, binary)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return ""


if __name__ == "__main__":
    raise SystemExit(main())
