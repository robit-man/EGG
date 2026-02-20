#!/usr/bin/env python3
"""
Pi5 camera router:
- Streams one or more camera feeds over MJPEG.
- Provides /health, /list, /router_info, and /tunnel_info endpoints.
- Uses Picamera2 when available, otherwise falls back to OpenCV VideoCapture.
"""

import json
import os
import platform
import random
import re
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time
from functools import wraps
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


CAMERA_VENV_DIR_NAME = "camera_router_venv"
CONFIG_PATH = "camera_router_config.json"


def _venv_includes_system_site_packages(venv_dir: str) -> bool:
    cfg_path = os.path.join(venv_dir, "pyvenv.cfg")
    try:
        with open(cfg_path, "r", encoding="utf-8") as fp:
            for raw_line in fp:
                line = str(raw_line or "").strip().lower()
                if line.startswith("include-system-site-packages"):
                    value = line.split("=", 1)[-1].strip()
                    return value in ("1", "true", "yes", "on")
    except Exception:
        return False
    return False


def ensure_venv() -> None:
    script_dir = os.path.abspath(os.path.dirname(__file__))
    venv_dir = os.path.join(script_dir, CAMERA_VENV_DIR_NAME)

    if os.path.exists(venv_dir) and not _venv_includes_system_site_packages(venv_dir):
        print(
            f"[CAMERA] Rebuilding '{CAMERA_VENV_DIR_NAME}' with system-site-packages for Picamera2 access...",
            flush=True,
        )
        try:
            shutil.rmtree(venv_dir)
        except Exception as exc:
            print(f"[CAMERA] Failed to remove old venv: {exc}", flush=True)

    if os.path.normcase(os.path.abspath(sys.prefix)) == os.path.normcase(os.path.abspath(venv_dir)):
        return

    if os.name == "nt":
        pip_path = os.path.join(venv_dir, "Scripts", "pip.exe")
        python_path = os.path.join(venv_dir, "Scripts", "python.exe")
    else:
        pip_path = os.path.join(venv_dir, "bin", "pip")
        python_path = os.path.join(venv_dir, "bin", "python")

    required = ["Flask", "Flask-CORS", "opencv-python", "numpy"]

    if not os.path.exists(venv_dir):
        print(f"[CAMERA] Creating virtual environment in '{CAMERA_VENV_DIR_NAME}'...", flush=True)
        import venv

        venv.create(venv_dir, with_pip=True, system_site_packages=True)
        subprocess.check_call([pip_path, "install", *required])
    else:
        try:
            check = subprocess.run(
                [python_path, "-c", "import flask, flask_cors, cv2, numpy"],
                capture_output=True,
                timeout=5,
            )
            if check.returncode != 0:
                subprocess.check_call([pip_path, "install", *required])
        except Exception:
            subprocess.check_call([pip_path, "install", *required])

    print("[CAMERA] Re-launching from venv...", flush=True)
    os.execv(python_path, [python_path] + sys.argv)


ensure_venv()

import cv2
import numpy as np
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

PICAMERA2_AVAILABLE = False
Picamera2 = None
try:
    from picamera2 import Picamera2

    PICAMERA2_AVAILABLE = True
except Exception:
    PICAMERA2_AVAILABLE = False

DEFAULT_PASSWORD = "egg"
DEFAULT_SESSION_TIMEOUT = 300
DEFAULT_REQUIRE_AUTH = True
DEFAULT_ENABLE_TUNNEL = True
DEFAULT_AUTO_INSTALL_CLOUDFLARED = True
DEFAULT_TUNNEL_RESTART_DELAY_SECONDS = 3.0
DEFAULT_TUNNEL_RATE_LIMIT_DELAY_SECONDS = 45.0
MAX_TUNNEL_RESTART_DELAY_SECONDS = 300.0
CAMERA_CLOUDFLARED_BASENAME = "camera_router_cloudflared"

DEFAULT_CONFIG = {
    "camera_router": {
        "network": {
            "listen_host": "0.0.0.0",
            "listen_port": 8080,
        },
        "stream": {
            "jpeg_quality": 75,
            "target_fps": 15,
            "use_picamera2": True,
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
        "cameras": [
            {
                "id": "cam0",
                "index": 0,
                "enabled": True,
                "width": 640,
                "height": 480,
                "rotation": 0,
            }
        ],
    }
}


def _get_nested(data: dict, path: str, default=None):
    current = data
    for key in path.split("."):
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


def _set_nested(data: dict, path: str, value) -> None:
    current = data
    keys = path.split(".")
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


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
    host_value = str(_get_nested(merged, "camera_router.network.listen_host", "0.0.0.0")).strip().lower()
    if host_value in ("127.0.0.1", "localhost", "::1", ""):
        _set_nested(merged, "camera_router.network.listen_host", "0.0.0.0")
    save_config(merged)
    return merged


def save_config(cfg: dict) -> None:
    with open(CONFIG_PATH, "w", encoding="utf-8") as fp:
        json.dump(cfg, fp, indent=2)


def _normalize_rotation(value: int) -> int:
    allowed = {0, 90, 180, 270}
    try:
        parsed = int(value)
    except Exception:
        return 0
    return parsed if parsed in allowed else 0


@dataclass
class CameraConfig:
    camera_id: str
    index: int
    enabled: bool
    width: int
    height: int
    rotation: int


class CameraFeed:
    def __init__(self, cfg: CameraConfig, jpeg_quality: int, target_fps: int, use_picamera2: bool):
        self.cfg = cfg
        self.jpeg_quality = max(30, min(95, int(jpeg_quality)))
        self.target_fps = max(1, min(60, int(target_fps)))
        self.use_picamera2 = bool(use_picamera2 and PICAMERA2_AVAILABLE)

        self._lock = threading.Lock()
        self._running = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._frame_jpeg: Optional[bytes] = None
        self._last_frame_at: float = 0.0
        self._frames: int = 0
        self._last_error: str = ""
        self._picam = None
        self._cap = None
        self._backend_name = "unknown"
        self._reopen_attempts = 0
        self._consecutive_failures = 0
        self._avg_jpeg_bytes = 0.0
        self._stream_clients = 0
        self._recover_requested = threading.Event()

    def _open(self) -> None:
        self._close()

        if self.use_picamera2:
            try:
                self._picam = Picamera2(self.cfg.index)
                conf = self._picam.create_video_configuration(
                    main={"size": (self.cfg.width, self.cfg.height)}
                )
                self._picam.configure(conf)
                self._picam.start()
                self._backend_name = "picamera2"
                return
            except Exception as exc:
                self._last_error = f"Picamera2 open failed; falling back to OpenCV: {exc}"
                self._picam = None

        open_errors = []
        attempts = []
        if hasattr(cv2, "CAP_V4L2"):
            attempts.append(("opencv-v4l2", lambda: cv2.VideoCapture(self.cfg.index, cv2.CAP_V4L2)))
        attempts.append(("opencv-default", lambda: cv2.VideoCapture(self.cfg.index)))

        for backend_name, opener in attempts:
            cap = None
            try:
                cap = opener()
                if cap is None or not cap.isOpened():
                    raise RuntimeError("capture not opened")
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
                cap.set(cv2.CAP_PROP_FPS, self.target_fps)

                warm_ok = False
                for _ in range(8):
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        warm_ok = True
                        break
                    time.sleep(0.03)
                if not warm_ok:
                    raise RuntimeError("open succeeded but warmup read failed")

                self._cap = cap
                self._backend_name = backend_name
                return
            except Exception as exc:
                open_errors.append(f"{backend_name}: {exc}")
                try:
                    if cap is not None:
                        cap.release()
                except Exception:
                    pass

        raise RuntimeError(
            f"OpenCV camera open failed (index={self.cfg.index}) [{'; '.join(open_errors)}]"
        )

    def _close(self) -> None:
        try:
            if self._picam is not None:
                self._picam.stop()
                self._picam.close()
        except Exception:
            pass
        self._picam = None
        try:
            if self._cap is not None:
                self._cap.release()
        except Exception:
            pass
        self._cap = None

    def _rotate(self, frame: np.ndarray) -> np.ndarray:
        rotation = self.cfg.rotation
        if rotation == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if rotation == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        if rotation == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    def _read_frame(self) -> np.ndarray:
        if self._picam is not None:
            frame = self._picam.capture_array()
            if frame is None:
                raise RuntimeError("Picamera2 returned no frame")
            return frame

        if self._cap is None:
            raise RuntimeError("OpenCV capture is not initialized")
        for _ in range(3):
            ok, frame = self._cap.read()
            if ok and frame is not None:
                return frame
            time.sleep(0.02)
        raise RuntimeError(f"OpenCV read failed ({self._backend_name})")

    def _reopen_capture(self) -> None:
        self._reopen_attempts += 1
        self._close()
        time.sleep(0.2)
        self._open()

    def _run(self) -> None:
        interval = 1.0 / float(self.target_fps)
        while self._running.is_set():
            start = time.time()
            try:
                frame = self._read_frame()
                frame = self._rotate(frame)
                ok, encoded = cv2.imencode(
                    ".jpg",
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)],
                )
                if not ok:
                    raise RuntimeError("JPEG encode failed")
                jpeg = encoded.tobytes()
                with self._lock:
                    self._frame_jpeg = jpeg
                    self._last_frame_at = time.time()
                    self._frames += 1
                    jpeg_len = float(len(jpeg))
                    if self._avg_jpeg_bytes <= 0:
                        self._avg_jpeg_bytes = jpeg_len
                    else:
                        self._avg_jpeg_bytes = (0.85 * self._avg_jpeg_bytes) + (0.15 * jpeg_len)
                    self._last_error = ""
                    self._consecutive_failures = 0
            except Exception as exc:
                should_reopen = False
                with self._lock:
                    self._last_error = str(exc)
                    self._consecutive_failures += 1
                    if self._consecutive_failures >= 5:
                        should_reopen = True
                if should_reopen:
                    try:
                        self._reopen_capture()
                        with self._lock:
                            self._consecutive_failures = 0
                            self._last_error = "Capture reopened after repeated read failures"
                    except Exception as reopen_exc:
                        with self._lock:
                            self._last_error = f"{exc}; reopen failed: {reopen_exc}"
                time.sleep(0.25)
            if self._recover_requested.is_set():
                self._recover_requested.clear()
                try:
                    self._reopen_capture()
                    with self._lock:
                        self._consecutive_failures = 0
                        self._last_error = "Capture reopened on manual recovery request"
                except Exception as recover_exc:
                    with self._lock:
                        self._last_error = f"Manual recover failed: {recover_exc}"
            elapsed = time.time() - start
            if elapsed < interval:
                time.sleep(interval - elapsed)

    def start(self) -> None:
        if not self.cfg.enabled:
            return
        try:
            self._open()
        except Exception as exc:
            with self._lock:
                self._last_error = str(exc)
            raise
        self._running.set()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        self._close()

    def snapshot(self) -> Optional[bytes]:
        with self._lock:
            return self._frame_jpeg

    def request_recover(self) -> None:
        self._recover_requested.set()

    def is_running(self) -> bool:
        thread_alive = self._thread is not None and self._thread.is_alive()
        return bool(self._running.is_set() and thread_alive)

    def recover(self) -> Tuple[bool, str]:
        if self.is_running():
            self.request_recover()
            return True, "recover requested"
        try:
            self.stop()
        except Exception:
            pass
        try:
            self.start()
            return True, "capture restarted"
        except Exception as exc:
            with self._lock:
                self._last_error = f"Manual recover failed: {exc}"
            return False, str(exc)

    def add_stream_client(self) -> None:
        with self._lock:
            self._stream_clients += 1

    def remove_stream_client(self) -> None:
        with self._lock:
            if self._stream_clients > 0:
                self._stream_clients -= 1

    def status(self) -> dict:
        with self._lock:
            age = time.time() - self._last_frame_at if self._last_frame_at > 0 else -1.0
            fps = float(self.target_fps) if self._frame_jpeg else 0.0
            kbps = float(self._avg_jpeg_bytes * max(1.0, fps) * 8.0 / 1000.0) if self._frame_jpeg else 0.0
            return {
                "id": self.cfg.camera_id,
                "label": self.cfg.camera_id,
                "index": self.cfg.index,
                "enabled": bool(self.cfg.enabled),
                "online": bool(self._frame_jpeg),
                "width": int(self.cfg.width),
                "height": int(self.cfg.height),
                "rotation": int(self.cfg.rotation),
                "frames": int(self._frames),
                "fps": round(fps, 2),
                "kbps": round(kbps, 2),
                "clients": int(self._stream_clients),
                "last_frame_age_seconds": round(age, 3) if age >= 0 else None,
                "last_error": self._last_error,
                "backend": self._backend_name,
                "reopen_attempts": int(self._reopen_attempts),
                "source_type": "default",
                "capture_profile": {
                    "pixel_format": "MJPEG",
                    "width": int(self.cfg.width),
                    "height": int(self.cfg.height),
                    "fps": float(self.target_fps),
                },
                "active_capture": {
                    "backend": self._backend_name,
                    "width": int(self.cfg.width),
                    "height": int(self.cfg.height),
                    "fps": round(fps, 2),
                },
                "available_profiles": [
                    {
                        "pixel_format": "MJPEG",
                        "width": int(self.cfg.width),
                        "height": int(self.cfg.height),
                        "fps": float(self.target_fps),
                    }
                ],
            }


def build_camera_configs(config: dict) -> Dict[str, CameraConfig]:
    rows = _get_nested(config, "camera_router.cameras", []) or []
    out: Dict[str, CameraConfig] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        camera_id = str(row.get("id", "")).strip()
        if not camera_id:
            continue
        try:
            index = int(row.get("index", 0))
        except Exception:
            index = 0
        try:
            width = int(row.get("width", 640))
            height = int(row.get("height", 480))
        except Exception:
            width, height = 640, 480
        out[camera_id] = CameraConfig(
            camera_id=camera_id,
            index=index,
            enabled=bool(row.get("enabled", True)),
            width=max(160, width),
            height=max(120, height),
            rotation=_normalize_rotation(row.get("rotation", 0)),
        )
    return out


config = load_config()
listen_host = str(_get_nested(config, "camera_router.network.listen_host", "0.0.0.0")).strip() or "0.0.0.0"
listen_port = int(_get_nested(config, "camera_router.network.listen_port", 8080) or 8080)
jpeg_quality = int(_get_nested(config, "camera_router.stream.jpeg_quality", 75) or 75)
target_fps = int(_get_nested(config, "camera_router.stream.target_fps", 15) or 15)
use_picamera2 = bool(_get_nested(config, "camera_router.stream.use_picamera2", True))
SESSION_TIMEOUT = _as_int(
    _get_nested(config, "camera_router.security.session_timeout", DEFAULT_SESSION_TIMEOUT),
    DEFAULT_SESSION_TIMEOUT,
    minimum=30,
    maximum=86400,
)
runtime_security = {
    "password": str(_get_nested(config, "camera_router.security.password", DEFAULT_PASSWORD)).strip() or DEFAULT_PASSWORD,
    "require_auth": _as_bool(
        _get_nested(config, "camera_router.security.require_auth", DEFAULT_REQUIRE_AUTH),
        default=DEFAULT_REQUIRE_AUTH,
    ),
}
tunnel_enabled = _as_bool(
    _get_nested(config, "camera_router.tunnel.enable", DEFAULT_ENABLE_TUNNEL),
    default=DEFAULT_ENABLE_TUNNEL,
)
auto_install_cloudflared = _as_bool(
    _get_nested(config, "camera_router.tunnel.auto_install_cloudflared", DEFAULT_AUTO_INSTALL_CLOUDFLARED),
    default=DEFAULT_AUTO_INSTALL_CLOUDFLARED,
)

feeds: Dict[str, CameraFeed] = {}
startup_time = time.time()
service_lock = threading.Lock()
sessions = {}
sessions_lock = threading.Lock()
tunnel_process = None
tunnel_url = None
tunnel_last_error = ""
tunnel_desired = False
tunnel_url_lock = threading.Lock()
tunnel_restart_lock = threading.Lock()
tunnel_restart_failures = 0
service_running = threading.Event()

for camera_id, camera_cfg in build_camera_configs(config).items():
    feed = CameraFeed(camera_cfg, jpeg_quality=jpeg_quality, target_fps=target_fps, use_picamera2=use_picamera2)
    try:
        feed.start()
    except Exception as exc:
        try:
            with feed._lock:
                feed._last_error = str(exc)
        except Exception:
            pass
        print(f"[CAMERA] Failed to start {camera_id}: {exc}", flush=True)
    feeds[camera_id] = feed


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


def _status_rows():
    with service_lock:
        return [feed.status() for feed in feeds.values()]


def _find_feed(camera_id: str) -> Optional[CameraFeed]:
    with service_lock:
        return feeds.get(camera_id)


def _persist_camera_enabled(camera_id: str, enabled: bool) -> bool:
    global config
    camera_key = str(camera_id or "").strip()
    if not camera_key:
        return False
    rows = _get_nested(config, "camera_router.cameras", [])
    if not isinstance(rows, list):
        rows = []
    found = False
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("id", "")).strip() != camera_key:
            continue
        row["enabled"] = bool(enabled)
        found = True
        break
    if not found:
        return False
    _set_nested(config, "camera_router.cameras", rows)
    try:
        save_config(config)
    except Exception:
        return False
    return True


def _set_camera_enabled(camera_id: str, enabled: bool) -> Tuple[bool, str, Optional[dict]]:
    camera_key = str(camera_id or "").strip()
    if not camera_key:
        return False, "camera_id is required", None
    with service_lock:
        feed = feeds.get(camera_key)
        if feed is None:
            return False, f"Camera '{camera_key}' not found", None
        feed.cfg.enabled = bool(enabled)
        if enabled:
            if not feed.is_running():
                try:
                    feed.start()
                except Exception as exc:
                    with feed._lock:
                        feed._last_error = f"Enable failed: {exc}"
                    return False, str(exc), feed.status()
            row = feed.status()
        else:
            try:
                feed.stop()
            except Exception as exc:
                with feed._lock:
                    feed._last_error = f"Disable failed: {exc}"
                return False, str(exc), feed.status()
            with feed._lock:
                feed._frame_jpeg = None
                feed._last_frame_at = 0.0
                feed._last_error = "Disabled by dashboard"
            row = feed.status()
    persisted = _persist_camera_enabled(camera_key, bool(enabled))
    if not persisted:
        return False, "Camera state changed but config save failed", row
    return True, ("camera enabled" if enabled else "camera disabled"), row


def _camera_row_with_urls(row: dict, publish_base: str, local_base: str, lan_base: str, tunnel_base: str) -> dict:
    camera_id = str(row.get("id", "")).strip()
    return {
        **row,
        "snapshot_url": f"{publish_base}/snapshot/{camera_id}",
        "jpeg_url": f"{publish_base}/jpeg/{camera_id}",
        "video_url": f"{publish_base}/video/{camera_id}",
        "mjpeg_url": f"{publish_base}/mjpeg/{camera_id}",
        "local_snapshot_url": f"{local_base}/snapshot/{camera_id}",
        "local_jpeg_url": f"{local_base}/jpeg/{camera_id}",
        "local_video_url": f"{local_base}/video/{camera_id}",
        "local_mjpeg_url": f"{local_base}/mjpeg/{camera_id}",
        "lan_snapshot_url": f"{lan_base}/snapshot/{camera_id}" if lan_base else "",
        "lan_jpeg_url": f"{lan_base}/jpeg/{camera_id}" if lan_base else "",
        "lan_video_url": f"{lan_base}/video/{camera_id}" if lan_base else "",
        "lan_mjpeg_url": f"{lan_base}/mjpeg/{camera_id}" if lan_base else "",
        "tunnel_snapshot_url": f"{tunnel_base}/snapshot/{camera_id}" if tunnel_base else "",
        "tunnel_jpeg_url": f"{tunnel_base}/jpeg/{camera_id}" if tunnel_base else "",
        "tunnel_video_url": f"{tunnel_base}/video/{camera_id}" if tunnel_base else "",
        "tunnel_mjpeg_url": f"{tunnel_base}/mjpeg/{camera_id}" if tunnel_base else "",
    }


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
    next_key = secrets.token_urlsafe(32)
    with sessions_lock:
        invalidated = len(sessions)
        sessions.clear()
        sessions[next_key] = {"created_at": now, "last_used": now}
    return next_key, invalidated


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
            return Response(status=204)
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
        return os.path.join(SCRIPT_DIR, f"{CAMERA_CLOUDFLARED_BASENAME}.exe")
    return os.path.join(SCRIPT_DIR, CAMERA_CLOUDFLARED_BASENAME)


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
        global tunnel_url, tunnel_process, tunnel_last_error, tunnel_restart_failures
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
                                ui.log(f"[TUNNEL] Camera Router URL: {tunnel_url}")

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


def _camera_dashboard_html(session_key: str) -> str:
    template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>EGG Camera Dashboard</title>
  <style>
    :root {
      --bg: #111;
      --panel: #1b1b1b;
      --subpanel: #161616;
      --text: #f4f7ff;
      --muted: #9aa9c0;
      --line: #333;
      --accent: #ffae00;
      --ok: #ffae00;
      --warn: #ffae00;
      --err: #ff6666;
    }
    * {
      box-sizing: border-box;
      font-family: Consolas, "Courier New", monospace;
    }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
    }
    .wrap {
      max-width: 1240px;
      margin: 0 auto;
      padding: 16px;
      display: grid;
      gap: 12px;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }
    .grid > .card:only-child,
    .grid > .card:last-child:nth-child(odd) {
      grid-column: 1 / -1;
    }
    .card {
      border: 1px solid var(--line);
      border-radius: 12px;
      background: var(--panel);
      padding: 12px;
    }
    .row {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }
    .space-between {
      justify-content: space-between;
    }
    .title {
      margin: 0;
      font-size: 1.16rem;
    }
    .muted {
      color: var(--muted);
    }
    .ok {
      color: var(--ok);
    }
    .warn {
      color: var(--warn);
    }
    .err {
      color: var(--err);
    }
    .pill {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 2px 10px;
      background: #222;
      color: var(--muted);
      font-size: 0.8rem;
    }
    .mono {
      font-family: Consolas, "Courier New", monospace;
    }
    input, button, a.btn {
      border-radius: 8px;
      border: 1px solid var(--line);
      padding: 8px 10px;
      font-size: 0.92rem;
    }
    input {
      width: 100%;
      background: var(--subpanel);
      color: var(--text);
    }
    button, a.btn {
      background: var(--accent);
      border-color: var(--accent);
      color: #121212;
      cursor: pointer;
      text-decoration: none;
      display: inline-block;
    }
    button.secondary, a.btn.secondary {
      background: #1f1a0e;
      border-color: rgba(255, 174, 0, 0.45);
      color: var(--accent);
    }
    button:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }
    .status-line {
      min-height: 1.2em;
      font-size: 0.9rem;
      color: var(--muted);
    }
    .camera-title {
      margin: 0;
      font-size: 1.01rem;
    }
    .preview {
      width: 100%;
      aspect-ratio: 4 / 3;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: var(--subpanel);
      overflow: hidden;
      margin-top: 8px;
      margin-bottom: 8px;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .preview img {
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
    }
    .preview .placeholder {
      color: var(--muted);
      font-size: 0.84rem;
      text-align: center;
      padding: 8px;
    }
    .kv {
      display: grid;
      grid-template-columns: 120px 1fr;
      gap: 4px 8px;
      font-size: 0.83rem;
      margin-top: 6px;
    }
    .kv .k {
      color: var(--muted);
    }
    @media (max-width: 960px) {
      .grid {
        grid-template-columns: 1fr;
      }
      .grid > .card:last-child:nth-child(odd) {
        grid-column: auto;
      }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <div class="row space-between">
        <h1 class="title">Camera Router Dashboard</h1>
        <span id="readyPill" class="pill">loading</span>
      </div>
      <div class="row" style="margin-top: 10px;">
        <div style="flex: 2 1 320px;">
          <input id="sessionKey" class="mono" placeholder="Paste session_key from /auth" />
        </div>
        <div style="flex: 1 1 180px; min-width: 160px;">
          <button id="applySession" class="secondary" style="width: 100%;">Apply Session Key</button>
        </div>
        <div style="flex: 1 1 180px; min-width: 160px;">
          <button id="refreshBtn" class="secondary" style="width: 100%;">Refresh</button>
        </div>
      </div>
      <div id="topStatus" class="status-line" style="margin-top: 8px;"></div>
      <div class="muted">Open this URL as <span class="mono">/dashboard?session_key=...</span> when auth is required.</div>
    </div>

    <div class="grid">
      <div class="card">
        <div class="kv">
          <div class="k">Online</div><div id="sumOnline">-</div>
          <div class="k">Total Cameras</div><div id="sumTotal">-</div>
          <div class="k">Tunnel</div><div id="sumTunnel">-</div>
          <div class="k">LAN Base</div><div id="sumLan" class="mono">-</div>
        </div>
      </div>
    </div>

    <div id="cameraGrid" class="grid"></div>
  </div>

  <script>
    const initialSessionKey = __SESSION_KEY_JSON__;
    let sessionKey = initialSessionKey || localStorage.getItem("egg_session_key") || "";
    let pollTimer = null;

    const els = {
      sessionKey: document.getElementById("sessionKey"),
      applySession: document.getElementById("applySession"),
      refreshBtn: document.getElementById("refreshBtn"),
      topStatus: document.getElementById("topStatus"),
      readyPill: document.getElementById("readyPill"),
      sumOnline: document.getElementById("sumOnline"),
      sumTotal: document.getElementById("sumTotal"),
      sumTunnel: document.getElementById("sumTunnel"),
      sumLan: document.getElementById("sumLan"),
      cameraGrid: document.getElementById("cameraGrid"),
    };

    const cameraState = new Map();

    function setTopStatus(msg, level) {
      els.topStatus.textContent = String(msg || "");
      els.topStatus.className = "status-line";
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
      if (sessionKey) requestOptions.headers["X-Session-Key"] = sessionKey;
      if (options.body !== undefined) {
        requestOptions.headers["Content-Type"] = "application/json";
        requestOptions.body = JSON.stringify(options.body);
      }
      const response = await fetch(withSession(url), requestOptions);
      const bodyText = await response.text();
      let data = {};
      try {
        data = bodyText ? JSON.parse(bodyText) : {};
      } catch (_) {
        data = { message: bodyText || "Invalid response" };
      }
      if (!response.ok) {
        const msg = (data && (data.message || data.error)) || `${response.status} ${response.statusText}`;
        throw new Error(msg);
      }
      return data;
    }

    function fmtAge(seconds) {
      const n = Number(seconds);
      if (!Number.isFinite(n) || n < 0) return "-";
      if (n < 1) return `${Math.round(n * 1000)}ms`;
      if (n < 60) return `${n.toFixed(1)}s`;
      return `${(n / 60).toFixed(1)}m`;
    }

    function safeId(value) {
      return String(value || "").replace(/[^a-zA-Z0-9_-]/g, "_");
    }

    function ensureCameraCard(cam) {
      const cameraId = String(cam.id || "");
      const domId = `cam_${safeId(cameraId)}`;
      let card = document.getElementById(domId);
      if (!card) {
        card = document.createElement("div");
        card.id = domId;
        card.className = "card";
        card.innerHTML = `
          <div class="row space-between">
            <h3 class="camera-title mono"></h3>
            <span class="pill cam-pill">-</span>
          </div>
          <div class="preview">
            <img alt="camera stream" />
            <div class="placeholder">Stream unavailable</div>
          </div>
          <div class="row">
            <button class="btn-toggle">Toggle</button>
            <button class="btn-recover secondary">Recover</button>
            <a class="btn secondary btn-open" target="_blank" rel="noopener">Open Stream</a>
          </div>
          <div class="kv">
            <div class="k">Enabled</div><div class="v-enabled">-</div>
            <div class="k">Online</div><div class="v-online">-</div>
            <div class="k">Backend</div><div class="v-backend mono">-</div>
            <div class="k">Resolution</div><div class="v-resolution mono">-</div>
            <div class="k">FPS</div><div class="v-fps">-</div>
            <div class="k">Last Frame</div><div class="v-age">-</div>
            <div class="k">Error</div><div class="v-error mono">-</div>
          </div>
        `;
        els.cameraGrid.appendChild(card);
      }
      return card;
    }

    async function setCameraEnabled(cameraId, enabled) {
      const idText = String(cameraId || "").trim();
      if (!idText) return;
      await fetchJson("/camera/config", {
        method: "POST",
        body: { camera_id: idText, enabled: !!enabled },
      });
    }

    async function recoverCamera(cameraId) {
      const idText = String(cameraId || "").trim();
      if (!idText) return;
      await fetchJson("/camera/recover", {
        method: "POST",
        body: { camera_id: idText, settle_ms: 250, reason: "dashboard_recover" },
      });
    }

    function updateCameraCard(cam) {
      const cameraId = String(cam.id || "");
      const card = ensureCameraCard(cam);
      card.dataset.cameraId = cameraId;
      card.dataset.enabled = String(!!cam.enabled);
      card.querySelector(".camera-title").textContent = `${cameraId} (idx ${cam.index})`;
      const pill = card.querySelector(".cam-pill");
      pill.textContent = cam.online ? "online" : "offline";
      pill.className = `pill cam-pill ${cam.online ? "ok" : "warn"}`;
      card.querySelector(".v-enabled").textContent = cam.enabled ? "yes" : "no";
      card.querySelector(".v-online").textContent = cam.online ? "yes" : "no";
      card.querySelector(".v-backend").textContent = String(cam.backend || "-");
      card.querySelector(".v-resolution").textContent = `${cam.width || "-"}x${cam.height || "-"}`;
      card.querySelector(".v-fps").textContent = String(cam.fps ?? "-");
      card.querySelector(".v-age").textContent = fmtAge(cam.last_frame_age_seconds);
      card.querySelector(".v-error").textContent = String(cam.last_error || "-");

      const img = card.querySelector("img");
      const placeholder = card.querySelector(".placeholder");
      const streamUrl = withSession(`/mjpeg/${encodeURIComponent(cameraId)}`);
      if (cam.enabled) {
        if (!img.dataset.streamUrl || img.dataset.streamUrl !== streamUrl) {
          img.dataset.streamUrl = streamUrl;
          img.src = streamUrl;
        }
        img.style.display = "block";
        placeholder.style.display = "none";
      } else {
        img.removeAttribute("src");
        img.dataset.streamUrl = "";
        img.style.display = "none";
        placeholder.style.display = "block";
        placeholder.textContent = "Camera disabled";
      }

      const openLink = card.querySelector(".btn-open");
      openLink.href = withSession(`/video/${encodeURIComponent(cameraId)}`);

      const toggleBtn = card.querySelector(".btn-toggle");
      toggleBtn.textContent = cam.enabled ? "Disable" : "Enable";
      toggleBtn.className = cam.enabled ? "btn-toggle secondary" : "btn-toggle";
      toggleBtn.onclick = async () => {
        toggleBtn.disabled = true;
        try {
          setTopStatus(`${cam.enabled ? "Disabling" : "Enabling"} ${cameraId}...`, "warn");
          await setCameraEnabled(cameraId, !cam.enabled);
          await loadDashboard();
          setTopStatus(`${cameraId} ${cam.enabled ? "disabled" : "enabled"}.`, "ok");
        } catch (err) {
          setTopStatus(`Camera toggle failed: ${err.message}`, "error");
        } finally {
          toggleBtn.disabled = false;
        }
      };

      const recoverBtn = card.querySelector(".btn-recover");
      recoverBtn.onclick = async () => {
        recoverBtn.disabled = true;
        try {
          setTopStatus(`Recovering ${cameraId}...`, "warn");
          await recoverCamera(cameraId);
          await loadDashboard();
          setTopStatus(`Recovery requested for ${cameraId}.`, "ok");
        } catch (err) {
          setTopStatus(`Recover failed: ${err.message}`, "error");
        } finally {
          recoverBtn.disabled = false;
        }
      };
    }

    function renderCameras(cameras) {
      const list = Array.isArray(cameras) ? cameras : [];
      const seen = new Set();
      list.forEach((cam) => {
        const id = String(cam.id || "");
        if (!id) return;
        seen.add(`cam_${safeId(id)}`);
        cameraState.set(id, cam);
        updateCameraCard(cam);
      });
      const allCards = Array.from(els.cameraGrid.querySelectorAll(".card"));
      allCards.forEach((card) => {
        if (!seen.has(card.id)) {
          card.remove();
        }
      });
      if (!list.length) {
        els.cameraGrid.innerHTML = '<div class="card"><div class="muted">No camera rows returned.</div></div>';
      }
    }

    async function loadDashboard() {
      const payload = await fetchJson("/list");
      const cameras = payload.cameras || [];
      const online = cameras.filter((row) => !!row.online).length;
      els.sumOnline.textContent = `${online}`;
      els.sumTotal.textContent = `${cameras.length}`;
      const tunnel = payload.tunnel || {};
      els.sumTunnel.textContent = String(tunnel.state || "inactive");
      els.sumLan.textContent = String(payload.lan_base_url || "-");
      renderCameras(cameras);
      els.readyPill.textContent = "ready";
      els.readyPill.className = "pill ok";
      return payload;
    }

    async function refreshAll() {
      try {
        setTopStatus("Refreshing camera dashboard...", "warn");
        await loadDashboard();
        setTopStatus("Dashboard updated.", "ok");
      } catch (err) {
        els.readyPill.textContent = "error";
        els.readyPill.className = "pill err";
        setTopStatus(`Refresh failed: ${err.message}`, "error");
      }
    }

    function startPolling() {
      if (pollTimer) clearInterval(pollTimer);
      pollTimer = setInterval(async () => {
        try {
          await loadDashboard();
        } catch (_) {}
      }, 4000);
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

    els.refreshBtn.addEventListener("click", async () => {
      await refreshAll();
    });

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
            "service": "camera_router",
            "routes": {
                "dashboard": "/dashboard",
                "health": "/health",
                "list": "/list",
                "auth": "/auth",
                "session_rotate": "/session/rotate",
                "camera_config": "/camera/config",
                "snapshot": "/snapshot/<camera_id>",
                "jpeg": "/jpeg/<camera_id>",
                "video": "/video/<camera_id>",
                "mjpeg": "/mjpeg/<camera_id>",
                "camera_recover": "/camera/recover",
                "router_info": "/router_info",
                "tunnel_info": "/tunnel_info",
            },
        }
    )


@app.route("/dashboard", methods=["GET"])
@_auth_required
def dashboard():
    session_key = _get_session_key_from_request()
    return Response(_camera_dashboard_html(session_key), mimetype="text/html")


@app.route("/health", methods=["GET"])
def health():
    rows = _status_rows()
    online = sum(1 for row in rows if row.get("online"))
    local_base, lan_base, publish_base = _endpoint_bases()
    tunnel = _tunnel_payload()
    return jsonify(
        {
            "status": "ok",
            "service": "camera_router",
            "uptime_seconds": round(time.time() - startup_time, 2),
            "backend": {
                "picamera2_available": PICAMERA2_AVAILABLE,
                "opencv_available": True,
            },
            "camera_count": len(rows),
            "camera_online_count": online,
            "security": _security_payload(publish_base),
            "local_security": _security_payload(local_base),
            "lan_security": _security_payload(lan_base) if lan_base else {},
            "tunnel": tunnel,
            "cameras": rows,
        }
    )


@app.route("/list", methods=["GET"])
@_auth_required
def list_cameras():
    rows = _status_rows()
    local_base, lan_base, publish_base = _endpoint_bases()
    tunnel = _tunnel_payload()
    tunnel_base = str(tunnel.get("tunnel_url") or "").strip()
    cameras = [_camera_row_with_urls(row, publish_base, local_base, lan_base, tunnel_base) for row in rows]
    return jsonify(
        {
            "status": "success",
            "service": "camera_router",
            "base_url": publish_base,
            "local_base_url": local_base,
            "lan_base_url": lan_base,
            "security": _security_payload(publish_base),
            "local_security": _security_payload(local_base),
            "lan_security": _security_payload(lan_base) if lan_base else {},
            "protocols": {
                "webrtc": False,
                "mjpeg": True,
                "jpeg_snapshot": True,
                "mpegts": False,
            },
            "routes": {
                "dashboard": "/dashboard",
                "auth": "/auth",
                "session_rotate": "/session/rotate",
                "camera_config": "/camera/config",
                "camera_recover": "/camera/recover",
                "snapshot": "/snapshot/<camera_id>",
                "jpeg": "/jpeg/<camera_id>",
                "video": "/video/<camera_id>",
                "mjpeg": "/mjpeg/<camera_id>",
            },
            "tunnel": {
                **tunnel,
                "list_url": f"{tunnel_base}/list" if tunnel_base else "",
                "health_url": f"{tunnel_base}/health" if tunnel_base else "",
                "auth_url": f"{tunnel_base}/auth" if tunnel_base else "",
                "session_rotate_url": f"{tunnel_base}/session/rotate" if tunnel_base else "",
                "dashboard_url": f"{tunnel_base}/dashboard" if tunnel_base else "",
                "camera_config_url": f"{tunnel_base}/camera/config" if tunnel_base else "",
                "camera_recover_url": f"{tunnel_base}/camera/recover" if tunnel_base else "",
            },
            "cameras": cameras,
        }
    )


@app.route("/camera/config", methods=["GET", "POST"])
@_auth_required
def camera_config():
    if request.method == "GET":
        rows = _status_rows()
        local_base, lan_base, publish_base = _endpoint_bases()
        tunnel = _tunnel_payload()
        tunnel_base = str(tunnel.get("tunnel_url") or "").strip()
        cameras = [_camera_row_with_urls(row, publish_base, local_base, lan_base, tunnel_base) for row in rows]
        return jsonify(
            {
                "status": "success",
                "service": "camera_router",
                "message": "Camera runtime config",
                "base_url": publish_base,
                "local_base_url": local_base,
                "lan_base_url": lan_base,
                "tunnel": tunnel,
                "cameras": cameras,
            }
        )

    data = request.get_json(silent=True) or {}
    camera_id = str(data.get("camera_id", "")).strip()
    if not camera_id:
        return jsonify({"status": "error", "message": "camera_id is required"}), 400
    if "enabled" not in data:
        return jsonify({"status": "error", "message": "enabled is required"}), 400
    enabled = _as_bool(data.get("enabled"), default=False)
    ok, detail, row = _set_camera_enabled(camera_id, enabled)

    local_base, lan_base, publish_base = _endpoint_bases()
    tunnel = _tunnel_payload()
    tunnel_base = str(tunnel.get("tunnel_url") or "").strip()
    payload_row = _camera_row_with_urls((row or {"id": camera_id}), publish_base, local_base, lan_base, tunnel_base)

    if ok:
        return jsonify(
            {
                "status": "success",
                "service": "camera_router",
                "message": detail,
                "camera": payload_row,
            }
        )

    status_code = 404 if "not found" in str(detail or "").lower() else 500
    return jsonify(
        {
            "status": "error",
            "service": "camera_router",
            "message": detail,
            "camera": payload_row,
        }
    ), status_code


@app.route("/snapshot/<camera_id>", methods=["GET"])
@_auth_required
def snapshot(camera_id: str):
    feed = _find_feed(camera_id)
    if feed is None:
        return Response("Camera not found", status=404, mimetype="text/plain")
    frame = feed.snapshot()
    if not frame:
        return Response("No frame available yet", status=503, mimetype="text/plain")
    return Response(frame, mimetype="image/jpeg")


def _video_stream_generator(feed: CameraFeed):
    feed.add_stream_client()
    try:
        while True:
            try:
                frame = feed.snapshot()
                if frame:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                    )
                time.sleep(1.0 / float(max(1, target_fps)))
            except GeneratorExit:
                break
            except Exception:
                time.sleep(0.1)
    finally:
        feed.remove_stream_client()


@app.route("/video/<camera_id>", methods=["GET"])
@_auth_required
def video(camera_id: str):
    feed = _find_feed(camera_id)
    if feed is None:
        return Response("Camera not found", status=404, mimetype="text/plain")
    return Response(
        _video_stream_generator(feed),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/jpeg/<camera_id>", methods=["GET"])
@_auth_required
def jpeg(camera_id: str):
    return snapshot(camera_id)


@app.route("/mjpeg/<camera_id>", methods=["GET"])
@_auth_required
def mjpeg(camera_id: str):
    return video(camera_id)


@app.route("/camera/recover", methods=["POST"])
@_auth_required
def camera_recover():
    data = request.get_json(silent=True) or {}
    settle_ms = _as_int(data.get("settle_ms", 350), 350, minimum=0, maximum=5000)
    reason = str(data.get("reason", "")).strip()

    requested_ids = []
    single_id = str(data.get("camera_id", "")).strip()
    if single_id:
        requested_ids.append(single_id)
    id_list = data.get("camera_ids", [])
    if isinstance(id_list, list):
        for value in id_list:
            camera_id = str(value or "").strip()
            if camera_id:
                requested_ids.append(camera_id)
    dedup_ids = []
    seen = set()
    for camera_id in requested_ids:
        if camera_id in seen:
            continue
        seen.add(camera_id)
        dedup_ids.append(camera_id)

    missing_ids = []
    requested = []
    restart_results = []
    with service_lock:
        if dedup_ids:
            values = []
            for camera_id in dedup_ids:
                feed = feeds.get(camera_id)
                if feed is None:
                    missing_ids.append(camera_id)
                else:
                    values.append(feed)
        else:
            values = list(feeds.values())
    if dedup_ids and not values:
        return jsonify(
            {
                "status": "error",
                "service": "camera_router",
                "message": "No requested cameras were found",
                "requested": dedup_ids,
                "missing": missing_ids,
            }
        ), 404
    for feed in values:
        ok, detail = feed.recover()
        requested.append(feed.cfg.camera_id)
        restart_results.append(
            {
                "id": feed.cfg.camera_id,
                "ok": bool(ok),
                "detail": str(detail or "").strip(),
            }
        )
    if settle_ms > 0:
        time.sleep(float(settle_ms) / 1000.0)
    rows = _status_rows()
    online = sum(1 for row in rows if row.get("online"))
    return jsonify(
        {
            "status": "success",
            "service": "camera_router",
            "message": "Recovery requested",
            "reason": reason,
            "requested": requested,
            "missing": missing_ids,
            "results": restart_results,
            "after_online": int(online),
            "after_total": int(len(rows)),
            "elapsed_ms": int(settle_ms),
        }
    )


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
    session_key, invalidated = _rotate_sessions()
    return jsonify(
        {
            "status": "success",
            "message": "Session keys rotated",
            "session_key": session_key,
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
            "service": "camera_router",
            "message": message,
            "tunnel_url": tunnel_base,
            "dashboard_url": f"{tunnel_base}/dashboard" if tunnel_base else "",
            "auth_url": f"{tunnel_base}/auth" if tunnel_base else "",
            "session_rotate_url": f"{tunnel_base}/session/rotate" if tunnel_base else "",
            "list_url": f"{tunnel_base}/list" if tunnel_base else "",
            "health_url": f"{tunnel_base}/health" if tunnel_base else "",
            "camera_config_url": f"{tunnel_base}/camera/config" if tunnel_base else "",
            "camera_recover_url": f"{tunnel_base}/camera/recover" if tunnel_base else "",
            "stale_tunnel_url": str(tunnel.get("stale_tunnel_url") or ""),
            "error": str(tunnel.get("error") or ""),
            "running": bool(tunnel.get("running")),
        }
    )


@app.route("/router_info", methods=["GET"])
def router_info():
    local_base, lan_base, publish_base = _endpoint_bases()
    tunnel = _tunnel_payload()
    tunnel_base = str(tunnel.get("tunnel_url") or "").strip()
    rows = _status_rows()
    camera_routes = [_camera_row_with_urls(row, publish_base, local_base, lan_base, tunnel_base) for row in rows]
    return jsonify(
        {
            "status": "success",
            "service": "camera_router",
            "local": {
                "base_url": publish_base,
                "loopback_base_url": local_base,
                "lan_base_url": lan_base,
                "listen_host": listen_host,
                "listen_port": listen_port,
                "auth_url": f"{publish_base}/auth",
                "session_rotate_url": f"{publish_base}/session/rotate",
                "dashboard_url": f"{publish_base}/dashboard",
                "list_url": f"{publish_base}/list",
                "health_url": f"{publish_base}/health",
                "camera_config_url": f"{publish_base}/camera/config",
                "camera_recover_url": f"{publish_base}/camera/recover",
                "local_dashboard_url": f"{local_base}/dashboard",
                "local_list_url": f"{local_base}/list",
                "local_health_url": f"{local_base}/health",
                "local_auth_url": f"{local_base}/auth",
                "local_session_rotate_url": f"{local_base}/session/rotate",
                "local_camera_config_url": f"{local_base}/camera/config",
                "local_camera_recover_url": f"{local_base}/camera/recover",
                "lan_dashboard_url": f"{lan_base}/dashboard" if lan_base else "",
                "lan_list_url": f"{lan_base}/list" if lan_base else "",
                "lan_health_url": f"{lan_base}/health" if lan_base else "",
                "lan_auth_url": f"{lan_base}/auth" if lan_base else "",
                "lan_session_rotate_url": f"{lan_base}/session/rotate" if lan_base else "",
                "lan_camera_config_url": f"{lan_base}/camera/config" if lan_base else "",
                "lan_camera_recover_url": f"{lan_base}/camera/recover" if lan_base else "",
            },
            "tunnel": {
                **tunnel,
                "dashboard_url": f"{tunnel_base}/dashboard" if tunnel_base else "",
                "list_url": f"{tunnel_base}/list" if tunnel_base else "",
                "health_url": f"{tunnel_base}/health" if tunnel_base else "",
                "auth_url": f"{tunnel_base}/auth" if tunnel_base else "",
                "session_rotate_url": f"{tunnel_base}/session/rotate" if tunnel_base else "",
                "camera_config_url": f"{tunnel_base}/camera/config" if tunnel_base else "",
                "camera_recover_url": f"{tunnel_base}/camera/recover" if tunnel_base else "",
            },
            "security": _security_payload(publish_base),
            "local_security": _security_payload(local_base),
            "lan_security": _security_payload(lan_base) if lan_base else {},
            "protocols": {
                "webrtc": False,
                "mjpeg": True,
                "jpeg_snapshot": True,
                "mpegts": False,
            },
            "routes": {
                "dashboard": "/dashboard",
                "auth": "/auth",
                "session_rotate": "/session/rotate",
                "camera_config": "/camera/config",
                "camera_recover": "/camera/recover",
                "snapshot": "/snapshot/<camera_id>",
                "jpeg": "/jpeg/<camera_id>",
                "video": "/video/<camera_id>",
                "mjpeg": "/mjpeg/<camera_id>",
            },
            "cameras": camera_routes,
        }
    )


def shutdown() -> None:
    service_running.clear()
    _stop_cloudflared_tunnel()
    with service_lock:
        values = list(feeds.values())
    for feed in values:
        try:
            feed.stop()
        except Exception:
            pass


def _apply_runtime_security(saved_config: dict) -> None:
    global SESSION_TIMEOUT
    runtime_security["password"] = (
        str(_get_nested(saved_config, "camera_router.security.password", runtime_security["password"])).strip()
        or DEFAULT_PASSWORD
    )
    runtime_security["require_auth"] = _as_bool(
        _get_nested(saved_config, "camera_router.security.require_auth", runtime_security["require_auth"]),
        default=runtime_security["require_auth"],
    )
    SESSION_TIMEOUT = _as_int(
        _get_nested(saved_config, "camera_router.security.session_timeout", SESSION_TIMEOUT),
        SESSION_TIMEOUT,
        minimum=30,
        maximum=86400,
    )
    _prune_expired_sessions()
    if ui:
        ui.update_metric("Auth", "Required" if runtime_security["require_auth"] else "Disabled")
        ui.update_metric("Session Timeout", str(SESSION_TIMEOUT))
        ui.log("Applied live security updates from config save")


def _build_camera_config_spec():
    if not UI_AVAILABLE:
        return None
    return ConfigSpec(
        label="Camera Router",
        categories=(
            CategorySpec(
                id="network",
                label="Network",
                settings=(
                    SettingSpec(
                        id="listen_host",
                        label="Listen Host",
                        path="camera_router.network.listen_host",
                        value_type="str",
                        default="0.0.0.0",
                        description="Bind host for camera router API.",
                        restart_required=True,
                    ),
                    SettingSpec(
                        id="listen_port",
                        label="Listen Port",
                        path="camera_router.network.listen_port",
                        value_type="int",
                        default=8080,
                        min_value=1,
                        max_value=65535,
                        description="Bind port for camera router API.",
                        restart_required=True,
                    ),
                ),
            ),
            CategorySpec(
                id="stream",
                label="Stream",
                settings=(
                    SettingSpec(
                        id="jpeg_quality",
                        label="JPEG Quality",
                        path="camera_router.stream.jpeg_quality",
                        value_type="int",
                        default=75,
                        min_value=30,
                        max_value=95,
                        description="JPEG quality for MJPEG/jpg responses.",
                        restart_required=True,
                    ),
                    SettingSpec(
                        id="target_fps",
                        label="Target FPS",
                        path="camera_router.stream.target_fps",
                        value_type="int",
                        default=15,
                        min_value=1,
                        max_value=60,
                        description="Target frame rate for capture and stream pacing.",
                        restart_required=True,
                    ),
                    SettingSpec(
                        id="use_picamera2",
                        label="Use Picamera2",
                        path="camera_router.stream.use_picamera2",
                        value_type="bool",
                        default=True,
                        description="Prefer Picamera2 backend before OpenCV fallback.",
                        restart_required=True,
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
                        path="camera_router.security.password",
                        value_type="secret",
                        default=DEFAULT_PASSWORD,
                        description="Password used by /auth.",
                        restart_required=False,
                    ),
                    SettingSpec(
                        id="session_timeout",
                        label="Session Timeout",
                        path="camera_router.security.session_timeout",
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
                        path="camera_router.security.require_auth",
                        value_type="bool",
                        default=DEFAULT_REQUIRE_AUTH,
                        description="Require session_key on camera routes.",
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
                        path="camera_router.tunnel.enable",
                        value_type="bool",
                        default=DEFAULT_ENABLE_TUNNEL,
                        description="Enable cloudflared trycloudflare tunnel.",
                        restart_required=True,
                    ),
                    SettingSpec(
                        id="auto_install_cloudflared",
                        label="Auto-install Cloudflared",
                        path="camera_router.tunnel.auto_install_cloudflared",
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
        rows = _status_rows()
        online = sum(1 for row in rows if row.get("online"))
        local_base, lan_base, publish_base = _endpoint_bases()
        tunnel = _tunnel_payload()
        ui.update_metric("Service", "camera_router")
        ui.update_metric("Bind", f"{listen_host}:{listen_port}")
        ui.update_metric("Local URL", local_base)
        ui.update_metric("LAN URL", lan_base or "N/A")
        ui.update_metric("Public URL", publish_base)
        ui.update_metric("Cameras", f"{online}/{len(rows)}")
        ui.update_metric("Backend", "picamera2" if (PICAMERA2_AVAILABLE and use_picamera2) else "opencv")
        ui.update_metric("Auth", "Required" if runtime_security["require_auth"] else "Disabled")
        ui.update_metric("Session Timeout", str(SESSION_TIMEOUT))
        ui.update_metric("Tunnel", str(tunnel.get("state") or "inactive"))
        ui.update_metric("Tunnel URL", str(tunnel.get("tunnel_url") or str(tunnel.get("stale_tunnel_url") or "N/A")))
        time.sleep(1.0)


def main() -> int:
    global ui
    try:
        service_running.set()
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
                "Camera Router",
                config_spec=_build_camera_config_spec(),
                config_path=CONFIG_PATH,
                refresh_interval_ms=700,
            )
            ui.on_save(_apply_runtime_security)
            ui.log(f"Starting camera router on {listen_host}:{listen_port}")
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
        shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
