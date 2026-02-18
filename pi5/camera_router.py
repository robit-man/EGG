#!/usr/bin/env python3
"""
Pi5 camera router:
- Streams one or more camera feeds over MJPEG.
- Provides /health, /list, /router_info, and /tunnel_info endpoints.
- Uses Picamera2 when available, otherwise falls back to OpenCV VideoCapture.
"""

import json
import os
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional


CAMERA_VENV_DIR_NAME = "camera_router_venv"
CONFIG_PATH = "camera_router_config.json"


def ensure_venv() -> None:
    script_dir = os.path.abspath(os.path.dirname(__file__))
    venv_dir = os.path.join(script_dir, CAMERA_VENV_DIR_NAME)
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

        venv.create(venv_dir, with_pip=True)
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
from flask import Flask, Response, jsonify
from flask_cors import CORS

UI_AVAILABLE = False
TerminalUI = None
ui = None

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
try:
    from terminal_ui import TerminalUI

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

    def _open(self) -> None:
        if self.use_picamera2:
            self._picam = Picamera2(self.cfg.index)
            conf = self._picam.create_video_configuration(
                main={"size": (self.cfg.width, self.cfg.height)}
            )
            self._picam.configure(conf)
            self._picam.start()
            return

        self._cap = cv2.VideoCapture(self.cfg.index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        if not self._cap.isOpened():
            raise RuntimeError(f"OpenCV camera open failed (index={self.cfg.index})")

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
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise RuntimeError("OpenCV read failed")
        return frame

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
                    self._last_error = ""
            except Exception as exc:
                with self._lock:
                    self._last_error = str(exc)
                time.sleep(0.25)
            elapsed = time.time() - start
            if elapsed < interval:
                time.sleep(interval - elapsed)

    def start(self) -> None:
        if not self.cfg.enabled:
            return
        self._open()
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

    def status(self) -> dict:
        with self._lock:
            age = time.time() - self._last_frame_at if self._last_frame_at > 0 else -1.0
            return {
                "id": self.cfg.camera_id,
                "index": self.cfg.index,
                "enabled": bool(self.cfg.enabled),
                "online": bool(self._frame_jpeg),
                "width": int(self.cfg.width),
                "height": int(self.cfg.height),
                "rotation": int(self.cfg.rotation),
                "frames": int(self._frames),
                "last_frame_age_seconds": round(age, 3) if age >= 0 else None,
                "last_error": self._last_error,
                "backend": "picamera2" if self._picam is not None else "opencv",
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

feeds: Dict[str, CameraFeed] = {}
startup_time = time.time()
service_lock = threading.Lock()

for camera_id, camera_cfg in build_camera_configs(config).items():
    feed = CameraFeed(camera_cfg, jpeg_quality=jpeg_quality, target_fps=target_fps, use_picamera2=use_picamera2)
    try:
        feed.start()
    except Exception as exc:
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


@app.route("/", methods=["GET"])
def index():
    return jsonify(
        {
            "status": "ok",
            "service": "camera_router",
            "routes": {
                "health": "/health",
                "list": "/list",
                "snapshot": "/snapshot/<camera_id>",
                "video": "/video/<camera_id>",
                "router_info": "/router_info",
                "tunnel_info": "/tunnel_info",
            },
        }
    )


@app.route("/health", methods=["GET"])
def health():
    rows = _status_rows()
    online = sum(1 for row in rows if row.get("online"))
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
            "cameras": rows,
        }
    )


@app.route("/list", methods=["GET"])
def list_cameras():
    rows = _status_rows()
    local_base, lan_base, publish_base = _endpoint_bases()
    cameras = []
    for row in rows:
        camera_id = row.get("id", "")
        cameras.append(
            {
                **row,
                "snapshot_url": f"{publish_base}/snapshot/{camera_id}",
                "video_url": f"{publish_base}/video/{camera_id}",
                "local_snapshot_url": f"{local_base}/snapshot/{camera_id}",
                "local_video_url": f"{local_base}/video/{camera_id}",
                "lan_snapshot_url": f"{lan_base}/snapshot/{camera_id}" if lan_base else "",
                "lan_video_url": f"{lan_base}/video/{camera_id}" if lan_base else "",
            }
        )
    return jsonify(
        {
            "status": "success",
            "service": "camera_router",
            "base_url": publish_base,
            "local_base_url": local_base,
            "lan_base_url": lan_base,
            "cameras": cameras,
        }
    )


@app.route("/snapshot/<camera_id>", methods=["GET"])
def snapshot(camera_id: str):
    feed = _find_feed(camera_id)
    if feed is None:
        return Response("Camera not found", status=404, mimetype="text/plain")
    frame = feed.snapshot()
    if not frame:
        return Response("No frame available yet", status=503, mimetype="text/plain")
    return Response(frame, mimetype="image/jpeg")


def _video_stream_generator(feed: CameraFeed):
    while True:
        frame = feed.snapshot()
        if frame:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )
        time.sleep(1.0 / float(max(1, target_fps)))


@app.route("/video/<camera_id>", methods=["GET"])
def video(camera_id: str):
    feed = _find_feed(camera_id)
    if feed is None:
        return Response("Camera not found", status=404, mimetype="text/plain")
    return Response(
        _video_stream_generator(feed),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/tunnel_info", methods=["GET"])
def tunnel_info():
    # Tunnel lifecycle is managed by router.py for this stack.
    return jsonify(
        {
            "status": "pending",
            "service": "camera_router",
            "message": "Tunnel URL is managed by router service",
            "tunnel_url": "",
        }
    )


@app.route("/router_info", methods=["GET"])
def router_info():
    local_base, lan_base, publish_base = _endpoint_bases()
    rows = _status_rows()
    camera_routes = []
    for row in rows:
        camera_id = row.get("id", "")
        camera_routes.append(
            {
                "id": camera_id,
                "snapshot_url": f"{publish_base}/snapshot/{camera_id}",
                "video_url": f"{publish_base}/video/{camera_id}",
                "local_snapshot_url": f"{local_base}/snapshot/{camera_id}",
                "local_video_url": f"{local_base}/video/{camera_id}",
                "lan_snapshot_url": f"{lan_base}/snapshot/{camera_id}" if lan_base else "",
                "lan_video_url": f"{lan_base}/video/{camera_id}" if lan_base else "",
                "online": bool(row.get("online")),
            }
        )
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
                "list_url": f"{publish_base}/list",
                "health_url": f"{publish_base}/health",
                "local_list_url": f"{local_base}/list",
                "local_health_url": f"{local_base}/health",
                "lan_list_url": f"{lan_base}/list" if lan_base else "",
                "lan_health_url": f"{lan_base}/health" if lan_base else "",
            },
            "tunnel": {
                "state": "inactive",
                "tunnel_url": "",
                "list_url": "",
                "health_url": "",
            },
            "cameras": camera_routes,
        }
    )


def shutdown() -> None:
    with service_lock:
        values = list(feeds.values())
    for feed in values:
        try:
            feed.stop()
        except Exception:
            pass


def _ui_metrics_loop() -> None:
    while ui and ui.running:
        rows = _status_rows()
        online = sum(1 for row in rows if row.get("online"))
        local_base, lan_base, publish_base = _endpoint_bases()
        ui.update_metric("Service", "camera_router")
        ui.update_metric("Bind", f"{listen_host}:{listen_port}")
        ui.update_metric("Local URL", local_base)
        ui.update_metric("LAN URL", lan_base or "N/A")
        ui.update_metric("Public URL", publish_base)
        ui.update_metric("Cameras", f"{online}/{len(rows)}")
        ui.update_metric("Backend", "picamera2" if (PICAMERA2_AVAILABLE and use_picamera2) else "opencv")
        time.sleep(1.0)


def main() -> int:
    global ui
    try:
        if UI_AVAILABLE:
            ui = TerminalUI("Camera Router", config_path=CONFIG_PATH, refresh_interval_ms=700)
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
