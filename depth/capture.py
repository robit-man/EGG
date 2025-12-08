#!/usr/bin/env python3
import os
import sys
import subprocess

# ----------------- Auto‑venv Bootstrap -----------------
if sys.prefix == sys.base_prefix:
    venv_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv")
    if not os.path.exists(venv_dir):
        print("Creating virtual environment...")
        subprocess.check_call([sys.executable, "-m", "venv", "venv"])
    # Determine pip and python paths inside the venv.
    if os.name == "nt":
        pip_exe = os.path.join(venv_dir, "Scripts", "pip.exe")
        python_exe = os.path.join(venv_dir, "Scripts", "python.exe")
    else:
        pip_exe = os.path.join(venv_dir, "bin", "pip")
        python_exe = os.path.join(venv_dir, "bin", "python")
    print("Installing dependencies...")
    subprocess.check_call([pip_exe, "install", "--upgrade", "pip"])
    subprocess.check_call([pip_exe, "install", "flask"])
    print("Relaunching inside virtual environment...")
    subprocess.check_call([python_exe] + sys.argv)
    sys.exit()

# ----------------- Imports (inside venv) -----------------
import glob
import socket
import json
import atexit
from dataclasses import dataclass, asdict
from flask import Flask, Response, render_template_string, jsonify, request
import subprocess

# ----------------- Configuration -----------------
BASE_VIDEO_PORT = 9000  # Starting TCP port for video streams
AUDIO_PORT = 9100       # TCP port for audio stream
DA3_API_BASE = os.environ.get("DA3_API_BASE", "http://127.0.0.1:5000")
# Depth Anything API base (separate app you uploaded)
DEPTH_API_BASE = os.environ.get("DEPTH_API_BASE", "http://127.0.0.1:5000")

# ----------------- Helper: List Camera Devices -----------------
def get_camera_devices():
    devices = glob.glob("/dev/video*")
    devices.sort()
    return devices

# ----------------- Camera Toolkit -----------------
@dataclass
class CameraSettings:
    """Config for a single physical camera."""
    capture_width: int = 1920
    capture_height: int = 1080
    output_width: int = 1080
    output_height: int = 1920
    framerate: int = 60
    flip_method: int = 3  # Jetson nvvidconv flip-method (3 ≈ rotate 90°)

    @classmethod
    def from_env(cls):
        """Allow overriding defaults via environment variables."""
        return cls(
            capture_width=int(os.getenv("CAM_CAPTURE_WIDTH", "1920")),
            capture_height=int(os.getenv("CAM_CAPTURE_HEIGHT", "1080")),
            output_width=int(os.getenv("CAM_OUTPUT_WIDTH", "1080")),
            output_height=int(os.getenv("CAM_OUTPUT_HEIGHT", "1920")),
            framerate=int(os.getenv("CAM_FRAMERATE", "60")),
            flip_method=int(os.getenv("CAM_FLIP_METHOD", "3")),
        )


class CameraProcess:
    """Wraps a single GStreamer pipeline for one /dev/videoX."""

    def __init__(self, index, device, port, settings: CameraSettings):
        self.index = index
        self.device = device
        self.port = port
        self.settings = settings
        self.proc = None

    def build_pipeline(self) -> str:
        """Build an MJPEG over TCP pipeline using current settings."""
        s = self.settings
        pipeline = (
            f'gst-launch-1.0 nvv4l2camerasrc device={self.device} ! '
            f'"video/x-raw(memory:NVMM), format=(string)UYVY, '
            f'width=(int){s.capture_width}, height=(int){s.capture_height}, '
            f'framerate=(fraction){s.framerate}/1" ! '
            f'nvvidconv flip-method={s.flip_method} ! '
            f'"video/x-raw(memory:NVMM), format=(string)I420, '
            f'width=(int){s.output_width}, height=(int){s.output_height}, '
            f'framerate=(fraction){s.framerate}/1" ! '
            'nvvidconv ! videoconvert ! jpegenc ! multipartmux boundary=frame ! '
            f'tcpserversink host=0.0.0.0 port={self.port}'
        )
        return pipeline

    def start(self):
        """(Re)start the GStreamer process."""
        self.stop()
        command = self.build_pipeline()
        print(f"Starting GStreamer pipeline for {self.device} on port {self.port}:\n{command}\n")
        self.proc = subprocess.Popen(
            command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

    def stop(self):
        """Stop the GStreamer process if still running."""
        if self.proc is not None and self.proc.poll() is None:
            print(f"Stopping GStreamer pipeline for {self.device} on port {self.port}")
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    def to_dict(self):
        return {
            "id": self.index,
            "device": self.device,
            "port": self.port,
            "settings": asdict(self.settings),
        }


class CameraManager:
    """High-level toolkit for handling multiple physical cameras."""

    def __init__(self, base_port: int):
        self.base_port = base_port
        self.cameras = {}
        self.default_settings = CameraSettings.from_env()

    def init_from_devices(self):
        devices = get_camera_devices()
        if not devices:
            print("No camera devices found at /dev/video*")
            sys.exit(1)
        for i, device in enumerate(devices):
            port = self.base_port + i
            # Copy default settings so each camera can diverge later.
            settings = CameraSettings(**asdict(self.default_settings))
            cam = CameraProcess(i, device, port, settings)
            self.cameras[i] = cam
            cam.start()

    @property
    def camera_ports(self):
        return {idx: cam.port for idx, cam in self.cameras.items()}

    def get(self, cam_id: int) -> CameraProcess:
        if cam_id not in self.cameras:
            raise KeyError(f"Unknown camera id {cam_id}")
        return self.cameras[cam_id]

    def update_resolution(self, cam_id: int, width: int, height: int, framerate=None):
        """Update requested camera resolution and restart its pipeline."""
        cam = self.get(cam_id)
        cam.settings.capture_width = width
        cam.settings.capture_height = height
        # Keep 90° rotation assumption: swap for output to keep portrait vs landscape consistent.
        cam.settings.output_width = height
        cam.settings.output_height = width
        if framerate is not None:
            cam.settings.framerate = framerate
        cam.start()

    def to_list(self):
        return [cam.to_dict() for cam in self.cameras.values()]

    def shutdown(self):
        for cam in self.cameras.values():
            cam.stop()


# Instantiate and start camera pipelines.
camera_manager = CameraManager(BASE_VIDEO_PORT)
camera_manager.init_from_devices()
camera_ports = camera_manager.camera_ports

# ----------------- Audio GStreamer Pipeline -----------------
audio_proc = None
audio_command = (
    'gst-launch-1.0 alsasrc device=default ! '
    'audioconvert ! audioresample ! vorbisenc ! oggmux ! '
    f'tcpserversink host=0.0.0.0 port={AUDIO_PORT}'
)
print(f"Starting Audio GStreamer pipeline on port {AUDIO_PORT}:\n{audio_command}\n")
audio_proc = subprocess.Popen(
    audio_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
)


def shutdown_pipelines():
    print("Shutting down pipelines...")
    camera_manager.shutdown()
    global audio_proc
    if audio_proc is not None and audio_proc.poll() is None:
        audio_proc.terminate()
        try:
            audio_proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            audio_proc.kill()


atexit.register(shutdown_pipelines)

# ----------------- Prepare Video Stream URL List -----------------
streams = []
for i in sorted(camera_ports.keys()):
    streams.append(f"/camera/{i}")
#streams.reverse()  # Reverse order (highest index first)

# Fill up to 6 positions with a blank (transparent 1x1 PNG) if needed.
blank_data_url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAE0lEQVR42mP8z/C/HwAE/wH+W+KrAAAAAElFTkSuQmCC"
while len(streams) < 6:
    streams.append(blank_data_url)
streams_json = json.dumps(streams)

# ----------------- Flask Application -----------------
app = Flask(__name__)
app.secret_key = "replace_with_a_random_secret_key"


# ---- REST API for camera control ----
@app.route("/api/cameras")
def api_cameras():
    """Return basic info about all cameras."""
    return jsonify(camera_manager.to_list())


@app.route("/api/cameras/<int:cam_id>/resolution", methods=["POST"])
def api_set_resolution(cam_id: int):
    """Update requested resolution (and optional framerate) for a camera."""
    payload = request.get_json(force=True) or {}
    width = int(payload.get("width", 1920))
    height = int(payload.get("height", 1080))
    framerate = payload.get("framerate")
    if framerate is not None:
        framerate = int(framerate)
    try:
        camera_manager.update_resolution(cam_id, width, height, framerate)
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 404
    return jsonify(camera_manager.get(cam_id).to_dict())


# Index route serves the immersive Three.js room.
@app.route("/")
def index():
    html = """
<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8">
    <title>Multi‑Camera + Depth Anything 3</title>
    <style>
      html, body {
        margin: 0;
        padding: 0;
        width: 100%;
        height: 100%;
        overflow: hidden;
        background: #000;
        color: #fff;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      }

      #overlay-ui {
        position: fixed;
        top: 10px;
        left: 10px;
        padding: 10px 12px;
        background: rgba(0,0,0,0.55);
        border-radius: 8px;
        font-size: 12px;
        z-index: 10;
        backdrop-filter: blur(8px);
      }

      #overlay-ui label {
        display: block;
        margin-bottom: 4px;
      }

      #overlay-ui input[type="number"],
      #overlay-ui input[type="text"] {
        width: 70px;
        font-size: 11px;
        margin-left: 4px;
        border-radius: 4px;
        border: 1px solid rgba(255,255,255,0.1);
        background: rgba(0,0,0,0.4);
        color: #fff;
        padding: 2px 4px;
      }

      #overlay-ui input#api-base {
        width: 180px;
      }

      #overlay-ui input[type="range"] {
        width: 150px;
      }

      #overlay-ui button {
        margin-top: 4px;
        margin-right: 4px;
        padding: 4px 8px;
        font-size: 11px;
        border-radius: 4px;
        border: 1px solid rgba(255,255,255,0.25);
        background: rgba(255,255,255,0.08);
        color: #fff;
        cursor: pointer;
      }

      #overlay-ui button:hover {
        background: rgba(255,255,255,0.18);
      }

      #overlay-ui input[type="checkbox"] {
        vertical-align: middle;
      }

      #status-line {
        margin-top: 6px;
        font-size: 11px;
        opacity: 0.8;
      }
    </style>

    <!-- Import map so the browser knows what "three" means -->
    <script type="importmap">
    {
      "imports": {
        "three": "https://unpkg.com/three@0.161.0/build/three.module.js",
        "three/addons/": "https://unpkg.com/three@0.161.0/examples/jsm/"
      }
    }
    </script>
  </head>
  <body>
    <div id="overlay-ui">
      <div>
        <label>
          Panel FOV
          <input id="panel-fov" type="range" min="20" max="120" step="1" value="60">
          <span id="panel-fov-value">60°</span>
        </label>
      </div>
      <div>
        <label>
          Aspect
          W <input id="aspect-w" type="number" min="1" step="1" value="9">
          H <input id="aspect-h" type="number" min="1" step="1" value="16">
        </label>
      </div>
      <div style="margin-top:4px;">
        <label>
          DA3 API
          <input id="api-base" type="text" value="{{ da3_api_base }}">
        </label>
      </div>
      <div style="margin-top:4px;">
        <button id="depth-once">Depth snapshot (all cams)</button>
        <label style="margin-left:4px;">
          <input id="depth-auto" type="checkbox"> Auto depth
        </label>
      </div>
      <div id="status-line">DA3: idle</div>
    </div>

    <script type="module">
      import * as THREE from 'three';
      import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

      // MJPEG camera stream URLs injected from Flask
      const streams = {{ streams_json|safe }};

      // --- UI elements ---
      const apiBaseInput      = document.getElementById('api-base');
      const fovSlider         = document.getElementById('panel-fov');
      const fovLabel          = document.getElementById('panel-fov-value');
      const aspectWInput      = document.getElementById('aspect-w');
      const aspectHInput      = document.getElementById('aspect-h');
      const depthOnceButton   = document.getElementById('depth-once');
      const depthAutoCheckbox = document.getElementById('depth-auto');
      const statusLine        = document.getElementById('status-line');

      function getApiBase() {
        const v = apiBaseInput.value.trim();
        return v.replace(/\\/+$/, ''); // strip trailing slashes
      }

      // --- Three.js globals ---
      let scene, camera, renderer, controls;
      const cameraPanels = [];
      const depthStates = [];

      // Per-camera depth scheduling
      const depthMinIntervalMs = 1500;
      let da3Ensuring = false;

      init();
      animate();

      function init() {
        // Scene / camera / renderer
        scene = new THREE.Scene();
        scene.background = new THREE.Color(0x000000);

        camera = new THREE.PerspectiveCamera(
          75,
          window.innerWidth / window.innerHeight,
          0.01,
          1000
        );
        // Slight offset so OrbitControls behave nicely
        camera.position.set(0, 0, 0.01);

        renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setPixelRatio(window.devicePixelRatio);
        renderer.setSize(window.innerWidth, window.innerHeight);
        document.body.appendChild(renderer.domElement);

        controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.08;

        // Build camera panels & depth state
        for (let i = 0; i < streams.length; i++) {
          const panel = createCameraPanel(i, streams[i]);
          cameraPanels.push(panel);
          depthStates.push({
            busy: false,
            jobId: null,
            auto: false,
            lastRequestTime: 0,
            group: null
          });
        }

        // UI wiring
        fovSlider.addEventListener('input', () => {
          fovLabel.textContent = `${fovSlider.value}°`;
          rebuildLayout();
        });
        fovLabel.textContent = `${fovSlider.value}°`;

        aspectWInput.addEventListener('change', rebuildLayout);
        aspectHInput.addEventListener('change', rebuildLayout);

        depthOnceButton.addEventListener('click', () => {
          for (let i = 0; i < depthStates.length; i++) {
            requestDepthForCamera(i);
          }
        });

        depthAutoCheckbox.addEventListener('change', () => {
          const enabled = depthAutoCheckbox.checked;
          for (let i = 0; i < depthStates.length; i++) {
            depthStates[i].auto = enabled;
            if (enabled) {
              requestDepthForCamera(i);
            }
          }
        });

        window.addEventListener('resize', onWindowResize);

        // Initial layout: cameras tiled around a circle, default portrait 9:16
        rebuildLayout();
      }

      function onWindowResize() {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
      }

      function getAspect() {
        const w = parseFloat(aspectWInput.value) || 9;
        const h = parseFloat(aspectHInput.value) || 16;
        return w / h;
      }

      // Lay the panels around a circle, sized by FOV & aspect
      function rebuildLayout() {
        const aspect = getAspect();
        const fovDeg = parseFloat(fovSlider.value) || 60;
        const fovRad = THREE.MathUtils.degToRad(fovDeg);
        const radius = 4.0;

        const count = cameraPanels.length;
        const angleStep = (2 * Math.PI) / Math.max(1, count);

        // Panel size so that from the center, they subtend ~fovDeg horizontally
        const panelWidth  = 2 * radius * Math.tan(fovRad / 2);
        const panelHeight = panelWidth / aspect; // portrait-friendly

        for (let i = 0; i < count; i++) {
          const panel = cameraPanels[i];
          const yaw = i * angleStep;

          const x = radius * Math.sin(yaw);
          const z = radius * Math.cos(yaw);

          panel.group.position.set(x, 0, z);
          panel.group.lookAt(0, 0, 0);

          if (panel.mesh.geometry) panel.mesh.geometry.dispose();
          panel.mesh.geometry = new THREE.PlaneGeometry(panelWidth, panelHeight);

          // Forward direction this camera covers in the sphere
          panel.forward.set(
            panel.group.position.x,
            panel.group.position.y,
            panel.group.position.z
          ).normalize();
        }

        // Re-orient any existing point clouds after layout changes
        for (let i = 0; i < depthStates.length; i++) {
          if (depthStates[i].group) {
            orientDepthGroupForCamera(i);
          }
        }
      }

      // Create a streaming panel: MJPEG -> <img> -> <canvas> -> CanvasTexture
      function createCameraPanel(index, url) {
        const group = new THREE.Group();
        scene.add(group);

        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d', { willReadFrequently: true });
        canvas.width  = 0;
        canvas.height = 0;

        const texture = new THREE.CanvasTexture(canvas);
        texture.minFilter = THREE.LinearFilter;
        texture.magFilter = THREE.LinearFilter;
        texture.encoding  = THREE.sRGBEncoding;

        const material = new THREE.MeshBasicMaterial({
          map: texture,
          side: THREE.BackSide, // inside the ring
          transparent: true,
          opacity: 1.0
        });

        const mesh = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), material);
        group.add(mesh);

        // MJPEG image stream
        const img = new Image();
        img.crossOrigin = "anonymous";
        img.decoding = "async";
        img.src = url;

        const forward = new THREE.Vector3(0, 0, 1);

        return { index, group, mesh, img, canvas, ctx, texture, forward };
      }

      // Draw current frame into canvas (center-cropped to target aspect)
      function copyImageToCanvas(img, canvas, ctx) {
        if (!img.naturalWidth || !img.naturalHeight) return;

        if (canvas.width === 0 || canvas.height === 0) {
          const aspect = getAspect();
          const baseWidth  = 640;
          const baseHeight = Math.round(baseWidth / aspect);
          canvas.width  = baseWidth;
          canvas.height = baseHeight;
        }

        const sWidth  = img.naturalWidth;
        const sHeight = img.naturalHeight;
        const destAspect = canvas.width / canvas.height;
        const srcAspect  = sWidth / sHeight;

        let sx, sy, sw, sh;

        if (srcAspect > destAspect) {
          // Source wider than target: crop left/right
          sh = sHeight;
          sw = sh * destAspect;
          sx = (sWidth - sw) * 0.5;
          sy = 0;
        } else {
          // Source taller: crop top/bottom
          sw = sWidth;
          sh = sw / destAspect;
          sx = 0;
          sy = (sHeight - sh) * 0.5;
        }

        ctx.drawImage(img, sx, sy, sw, sh, 0, 0, canvas.width, canvas.height);
      }

      // Main render loop:
      // - just sample whatever frame the browser has *right now* (no manual buffering)
      function animate() {
        requestAnimationFrame(animate);

        for (const panel of cameraPanels) {
          const img = panel.img;
          if (!img.complete || !img.naturalWidth || !img.naturalHeight) continue;
          copyImageToCanvas(img, panel.canvas, panel.ctx);
          panel.texture.needsUpdate = true;
        }

        if (controls) controls.update();
        renderer.render(scene, camera);
      }

      // --- Depth Anything 3 integration (mirrors DA3's own client logic) ---

      function sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
      }

      async function fetchModelStatus() {
        const base = getApiBase();
        if (!base) return null;
        try {
          const res = await fetch(`${base}/api/model_status`);
          if (!res.ok) return null;
          return await res.json();
        } catch (err) {
          console.warn("Model status check failed", err);
          return null;
        }
      }

      async function waitForModelReady(maxAttempts = 60, delayMs = 1500) {
        for (let i = 0; i < maxAttempts; i++) {
          const status = await fetchModelStatus();
          if (status) {
            if (status.status === "ready") {
              statusLine.textContent = "DA3: model ready";
              return true;
            }
            if (status.status === "error") {
              statusLine.textContent = "DA3: model error";
              return false;
            }
            statusLine.textContent = `DA3: loading (${status.progress || 0}%)`;
          }
          await sleep(delayMs);
        }
        statusLine.textContent = "DA3: model load timeout";
        return false;
      }

      // Lightweight variant of DA3's ensureModelReady() that works against the API-only server
      async function ensureDa3ModelReady() {
        if (da3Ensuring) {
          return waitForModelReady();
        }
        da3Ensuring = true;
        try {
          const status = await fetchModelStatus();
          if (status?.status === "ready") {
            statusLine.textContent = "DA3: model ready";
            return true;
          }
          if (status?.status === "loading") {
            return await waitForModelReady();
          }
          const base = getApiBase();
          if (!base) {
            statusLine.textContent = "DA3: API base URL not set";
            return false;
          }
          statusLine.textContent = "DA3: starting model load...";
          const res = await fetch(`${base}/api/load_model`, { method: "POST" });
          if (!res.ok) {
            statusLine.textContent = "DA3: failed to start model load";
            return false;
          }
          return await waitForModelReady();
        } finally {
          da3Ensuring = false;
        }
      }

      // Send the *latest* frame for this camera to /api/v1/infer
      async function requestDepthForCamera(index) {
        const state = depthStates[index];
        const panel = cameraPanels[index];
        const base  = getApiBase();

        if (!panel || !base) {
          statusLine.textContent = "DA3: API base URL not set";
          return;
        }
        if (state.busy) return;

        const now = performance.now();
        if (now - state.lastRequestTime < depthMinIntervalMs) {
          if (state.auto) {
            const remaining = depthMinIntervalMs - (now - state.lastRequestTime);
            setTimeout(() => requestDepthForCamera(index), remaining + 10);
          }
          return;
        }

        if (!panel.img.naturalWidth || !panel.img.naturalHeight) {
          console.warn("Camera", index, "frame not ready yet");
          return;
        }

        const modelOk = await ensureDa3ModelReady();
        if (!modelOk) return;

        state.busy = true;
        state.lastRequestTime = now;
        statusLine.textContent = `DA3: capturing depth for camera ${index}...`;

        // Freeze the current frame into the canvas so the blob is exactly what we show
        copyImageToCanvas(panel.img, panel.canvas, panel.ctx);

        panel.canvas.toBlob(async (blob) => {
          if (!blob) {
            state.busy = false;
            return;
          }

          const file = new File([blob], `camera-${index}-${Date.now()}.jpg`, { type: "image/jpeg" });

          // Match DA3's own form keys (handleFileSelect → /api/process). 
          const formData = new FormData();
          formData.append("file", file);
          formData.append("resolution", "504");
          formData.append("max_points", "250000");
          formData.append("process_res_method", "upper_bound_resize");
          formData.append("align_to_input_ext_scale", "true");
          formData.append("infer_gs", "false");
          formData.append("export_feat_layers", "");
          formData.append("conf_thresh_percentile", "40");
          formData.append("apply_confidence_filter", "false");
          formData.append("include_confidence", "false");
          formData.append("show_cameras", "false");
          formData.append("feat_vis_fps", "15");

          try {
            // /api/v1/infer is an alias for /api/process and accepts multipart form-data. :contentReference[oaicite:4]{index=4}
            const response = await fetch(`${base}/api/v1/infer`, {
              method: "POST",
              body: formData
            });
            const data = await response.json().catch(() => null);

            if (!response.ok) {
              console.error("DA3 infer error", response.status, data);
              statusLine.textContent = `DA3: error ${response.status}`;
              state.busy = false;
              if (state.auto) {
                setTimeout(() => requestDepthForCamera(index), depthMinIntervalMs);
              }
              return;
            }

            if (data.pointcloud) {
              statusLine.textContent = `DA3: depth ready (camera ${index})`;
              handlePointCloudForCamera(index, data.pointcloud);
              state.busy = false;
              if (state.auto) {
                requestDepthForCamera(index);
              }
            } else if (data.job_id) {
              state.jobId = data.job_id;
              statusLine.textContent = `DA3: job ${data.job_id} (camera ${index})`;
              pollDepthJob(index);
            } else {
              console.warn("Unexpected DA3 response", data);
              statusLine.textContent = "DA3: unexpected response";
              state.busy = false;
              if (state.auto) {
                setTimeout(() => requestDepthForCamera(index), depthMinIntervalMs);
              }
            }
          } catch (err) {
            console.error("DA3 infer exception", err);
            statusLine.textContent = "DA3: request failed";
            state.busy = false;
            if (state.auto) {
              setTimeout(() => requestDepthForCamera(index), depthMinIntervalMs);
            }
          }
        }, "image/jpeg", 0.9);
      }

      function pollDepthJob(index) {
        const state = depthStates[index];
        const base  = getApiBase();
        if (!state.jobId || !base) return;

        const jobId = state.jobId;

        const intervalId = setInterval(async () => {
          try {
            const res = await fetch(`${base}/api/v1/jobs/${jobId}`);
            const data = await res.json().catch(() => null);

            if (!res.ok) {
              console.error("DA3 job polling error", res.status, data);
              clearInterval(intervalId);
              state.jobId = null;
              state.busy = false;
              if (state.auto) {
                setTimeout(() => requestDepthForCamera(index), depthMinIntervalMs);
              }
              return;
            }

            if (data.status === "completed") {
              clearInterval(intervalId);
              state.jobId = null;
              statusLine.textContent = `DA3: depth ready (camera ${index})`;
              if (data.pointcloud) {
                handlePointCloudForCamera(index, data.pointcloud);
              }
              state.busy = false;
              if (state.auto) {
                requestDepthForCamera(index);
              }
            } else if (data.status === "error") {
              console.error("DA3 job failed", data.error);
              statusLine.textContent = "DA3: job error";
              clearInterval(intervalId);
              state.jobId = null;
              state.busy = false;
              if (state.auto) {
                setTimeout(() => requestDepthForCamera(index), depthMinIntervalMs);
              }
            }
          } catch (err) {
            console.error("DA3 job poll exception", err);
          }
        }, 1000);
      }

      // Build a THREE.Points from DA3's point cloud JSON and attach it to this camera
      function handlePointCloudForCamera(index, pointcloud) {
        const vertices = pointcloud.vertices || [];
        if (!vertices.length) {
          console.warn("Empty point cloud for camera", index);
          return;
        }

        const colors = pointcloud.colors || [];
        const hasColors = colors.length === vertices.length;

        const positions  = new Float32Array(vertices.length * 3);
        const colorArray = hasColors ? new Float32Array(colors.length * 3) : null;

        for (let i = 0; i < vertices.length; i++) {
          const v = vertices[i];
          positions[3 * i + 0] = v[0];
          positions[3 * i + 1] = v[1];
          positions[3 * i + 2] = v[2];

          if (hasColors) {
            const c = colors[i] || [255, 255, 255];
            colorArray[3 * i + 0] = (c[0] || 0) / 255.0;
            colorArray[3 * i + 1] = (c[1] || 0) / 255.0;
            colorArray[3 * i + 2] = (c[2] || 0) / 255.0;
          }
        }

        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
        if (hasColors) {
          geometry.setAttribute("color", new THREE.BufferAttribute(colorArray, 3));
        }
        geometry.computeBoundingSphere();

        const material = new THREE.PointsMaterial({
          size: 0.005,
          sizeAttenuation: true,
          vertexColors: hasColors,
          depthWrite: false,
          transparent: true
        });

        let group = depthStates[index].group;
        if (!group) {
          group = new THREE.Group();
          depthStates[index].group = group;
          scene.add(group);
        } else {
          while (group.children.length) {
            const child = group.children.pop();
            if (child.geometry) child.geometry.dispose();
            if (child.material) child.material.dispose();
          }
        }

        const points = new THREE.Points(geometry, material);
        group.add(points);

        orientDepthGroupForCamera(index);
      }

      // Align DA3's canonical camera-forward (-Z in OpenGL/Three.js space) to this rig camera's segment
      function orientDepthGroupForCamera(index) {
        const state  = depthStates[index];
        const group  = state.group;
        const panel  = cameraPanels[index];
        if (!group || !panel) return;

        const forwardWorld  = panel.forward.clone().normalize();
        const da3Forward    = new THREE.Vector3(0, 0, -1); // DA3 points are in an OpenGL-like camera space. 
        const quat          = new THREE.Quaternion().setFromUnitVectors(da3Forward, forwardWorld);

        group.quaternion.copy(quat);
        group.position.set(0, 0, 0);
      }
    </script>
  </body>
</html>
    """
    return render_template_string(html, streams_json=streams_json, da3_api_base=DA3_API_BASE)



# The /camera/<id> route proxies the MJPEG video stream.
@app.route("/camera/<int:cam_id>")
def camera_stream(cam_id):
    if cam_id not in camera_ports:
        return "Camera not found", 404
    port = camera_ports[cam_id]
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect(("127.0.0.1", port))
    except Exception as e:
        return f"Error connecting to camera stream on port {port}: {e}", 500

    def generate():
        try:
            while True:
                data = s.recv(1024)
                if not data:
                    break
                yield data
        except Exception as e:
            print(f"Error reading from video socket: {e}")
        finally:
            s.close()
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


# The /audio route proxies the audio stream.
@app.route("/audio")
def audio_stream():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect(("127.0.0.1", AUDIO_PORT))
    except Exception as e:
        return f"Error connecting to audio stream on port {AUDIO_PORT}: {e}", 500

    def generate():
        try:
            while True:
                data = s.recv(1024)
                if not data:
                    break
                yield data
        except Exception as e:
            print(f"Error reading from audio socket: {e}")
        finally:
            s.close()
    return Response(generate(), mimetype="audio/ogg")


# ----------------- Run Flask Server -----------------
if __name__ == "__main__":
    print("Flask server starting on port 8080...")
    app.run(host="0.0.0.0", port=8080)
