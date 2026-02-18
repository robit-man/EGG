#!/usr/bin/env python3
import json
import os
import socket
import subprocess
import sys
import time


VENV_DIR = "output_venv"
CONFIG_PATH = "audio_router_config.json"

DEFAULT_LISTEN_HOST = "0.0.0.0"
DEFAULT_LISTEN_PORT = 6353
DEFAULT_PCM_DEVICE = "default"
DEFAULT_CHUNK = 4096
DEFAULT_RATE = 22050
DEFAULT_CHANNELS = 1
DEFAULT_VOLUME = 0.2
MAX_PIP_ATTEMPTS = 3
VERBOSE_CONNECTION_LOGS = str(os.environ.get("OUTPUT_VERBOSE_CONNECTION_LOGS", "0")).strip().lower() in (
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
    config_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), CONFIG_PATH)
    payload = {}
    try:
        with open(config_path, "r", encoding="utf-8") as fp:
            loaded = json.load(fp)
        if isinstance(loaded, dict):
            payload = loaded
    except Exception:
        payload = {}

    raw_device = _get_nested(payload, "audio_router.audio.output_device", DEFAULT_PCM_DEVICE)
    if isinstance(raw_device, (int, float)):
        pcm_device = DEFAULT_PCM_DEVICE
    else:
        pcm_device = str(raw_device or DEFAULT_PCM_DEVICE).strip() or DEFAULT_PCM_DEVICE

    return {
        "listen_host": str(
            _get_nested(payload, "audio_router.audio.output_listen_host", DEFAULT_LISTEN_HOST)
        ).strip()
        or DEFAULT_LISTEN_HOST,
        "listen_port": _as_int(
            _get_nested(payload, "audio_router.integrations.audio_out_port", DEFAULT_LISTEN_PORT),
            DEFAULT_LISTEN_PORT,
            minimum=1,
            maximum=65535,
        ),
        "pcm_device": pcm_device,
        "chunk": _as_int(
            _get_nested(payload, "audio_router.audio.output_chunk", DEFAULT_CHUNK),
            DEFAULT_CHUNK,
            minimum=256,
            maximum=65536,
        ),
        "rate": _as_int(
            _get_nested(payload, "audio_router.audio.output_sample_rate", DEFAULT_RATE),
            DEFAULT_RATE,
            minimum=8000,
            maximum=192000,
        ),
        "channels": _as_int(
            _get_nested(payload, "audio_router.audio.output_channels", DEFAULT_CHANNELS),
            DEFAULT_CHANNELS,
            minimum=1,
            maximum=2,
        ),
        "volume": _as_float(
            _get_nested(payload, "audio_router.audio.output_volume", DEFAULT_VOLUME),
            DEFAULT_VOLUME,
            minimum=0.0,
            maximum=4.0,
        ),
    }


def is_venv():
    return (
        (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)
        or (hasattr(sys, "real_prefix") and sys.real_prefix != sys.prefix)
    )


def create_venv():
    if not os.path.exists(VENV_DIR):
        _log(f"Creating virtual environment in {VENV_DIR}...")
        subprocess.run([sys.executable, "-m", "venv", VENV_DIR], check=True)


def install_dependencies():
    if sys.platform == "win32":
        pip_executable = os.path.join(VENV_DIR, "Scripts", "pip.exe")
    else:
        pip_executable = os.path.join(VENV_DIR, "bin", "pip")

    if not os.path.exists(pip_executable):
        _log("pip executable not found in the virtual environment.")
        return False

    _log("Installing dependencies in the virtual environment...")
    attempts = 0
    while attempts < MAX_PIP_ATTEMPTS:
        try:
            subprocess.run([pip_executable, "install", "--upgrade", "pip"], check=True)
            subprocess.run([pip_executable, "install", "pyalsaaudio", "numpy"], check=True)
            _log("Dependencies installed successfully.")
            return True
        except subprocess.CalledProcessError as exc:
            attempts += 1
            _log(f"Pip install attempt {attempts} failed: {exc}")
            if attempts < MAX_PIP_ATTEMPTS:
                time.sleep(2)
    _log("Maximum pip install attempts reached. Skipping installation.")
    return False


def activate_venv():
    if sys.platform == "win32":
        venv_site_packages = os.path.join(VENV_DIR, "Lib", "site-packages")
    else:
        venv_site_packages = os.path.join(
            VENV_DIR,
            "lib",
            f"python{sys.version_info.major}.{sys.version_info.minor}",
            "site-packages",
        )
    sys.path.insert(0, venv_site_packages)


def relaunch_in_venv():
    if sys.platform == "win32":
        python_executable = os.path.join(VENV_DIR, "Scripts", "python.exe")
    else:
        python_executable = os.path.join(VENV_DIR, "bin", "python")
    if not os.path.exists(python_executable):
        sys.exit("Error: Python executable not found in the virtual environment.")
    subprocess.check_call([python_executable] + sys.argv)
    sys.exit()


def setup_virtual_environment():
    if not is_venv():
        create_venv()
        install_successful = install_dependencies()
        if not install_successful:
            _log("Proceeding without installing dependencies. Ensure they are already installed.")
        relaunch_in_venv()
    else:
        activate_venv()
        try:
            import alsaaudio  # noqa: F401
            import numpy  # noqa: F401
        except ImportError as exc:
            _log(f"Missing dependencies: {exc}")
            install_successful = install_dependencies()
            if not install_successful:
                _log("Proceeding without installing dependencies. Ensure they are already installed.")


def handle_client_connection(client_socket, settings, addr):
    try:
        import alsaaudio
        import numpy as np

        audio_out = alsaaudio.PCM(alsaaudio.PCM_PLAYBACK, device=settings["pcm_device"])
        audio_out.setchannels(int(settings["channels"]))
        audio_out.setrate(int(settings["rate"]))
        audio_out.setformat(alsaaudio.PCM_FORMAT_S16_LE)
        audio_out.setperiodsize(int(settings["chunk"]))

        chunk = int(settings["chunk"])
        volume = float(settings["volume"])
        received_any = False

        while True:
            data = client_socket.recv(chunk)
            if not data:
                if received_any:
                    _debug(f"Audio client disconnected: {addr}")
                break
            received_any = True

            audio_samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
            audio_samples *= volume
            audio_samples = np.clip(audio_samples, -32768, 32767)
            audio_out.write(audio_samples.astype(np.int16).tobytes())
    except Exception as exc:
        _log(f"Error in client connection: {exc}")
    finally:
        try:
            client_socket.close()
        except Exception:
            pass


def main():
    setup_virtual_environment()
    while True:
        server_socket = None
        try:
            settings = _load_runtime_settings()
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((settings["listen_host"], int(settings["listen_port"])))
            server_socket.listen(1)
            _log(
                f"Listening for raw audio on {settings['listen_host']}:{settings['listen_port']} "
                f"(pcm={settings['pcm_device']} rate={settings['rate']} ch={settings['channels']})..."
            )

            client_socket, addr = server_socket.accept()
            _debug(f"Audio client connected: {addr}")
            handle_client_connection(client_socket, settings, addr)
        except Exception as exc:
            _log(f"An error occurred: {exc}")
            time.sleep(5)
        finally:
            if server_socket:
                try:
                    server_socket.close()
                except Exception:
                    pass


if __name__ == "__main__":
    main()
