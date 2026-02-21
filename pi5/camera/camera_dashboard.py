#!/usr/bin/env python3
"""
Standalone Hailo CSI-camera dashboard launcher.

What this script does:
1. Self-bootstraps a virtual environment (default: ./venv_hailo_apps).
2. Mirrors setup_env.sh behavior (kernel check, PYTHONPATH, /usr/local/hailo/resources/.env).
3. Uses a curses selector for model + camera choice.
4. Runs a selected Hailo pipeline app and serves MJPEG via Flask.
"""

from __future__ import annotations

import argparse
import base64
import curses
import importlib
import json
import os
import platform
import random
import re
import secrets
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
import venv
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Dict, List, Optional


INVALID_KERNELS = {"6.12.21", "6.12.22", "6.12.23", "6.12.24", "6.12.25"}
DEFAULT_ENV_FILE = Path("/usr/local/hailo/resources/.env")
BOOTSTRAP_STAMP_NAME = ".hailo_csi_dashboard_bootstrap"
BOOTSTRAP_STAMP_VALUE = "v1"
VALID_ORIENTATIONS = [0, 90, 180, 270]
DEFAULT_HAILO_REPO_URL = "https://github.com/hailo-ai/hailo-apps.git"
DEFAULT_HAILO_REPO_DIRNAME = "hailo-apps"
DEFAULT_PASSWORD = "egg"
DEFAULT_SESSION_TIMEOUT = 300
DEFAULT_REQUIRE_AUTH = True
DEFAULT_TUNNEL_URL_ENV = "EGG_CAMERA_TUNNEL_URL"
DEFAULT_ENABLE_TUNNEL = True
DEFAULT_AUTO_INSTALL_CLOUDFLARED = True
DEFAULT_TUNNEL_RESTART_DELAY_SECONDS = 3.0
DEFAULT_TUNNEL_RATE_LIMIT_DELAY_SECONDS = 45.0
MAX_TUNNEL_RESTART_DELAY_SECONDS = 300.0
CAMERA_CLOUDFLARED_BASENAME = "camera_router_cloudflared"
SCRIPT_DIR = str(Path(__file__).resolve().parent)
CAMERA_LIFECYCLE_LOCK = threading.Lock()
tunnel_enabled = DEFAULT_ENABLE_TUNNEL
auto_install_cloudflared = DEFAULT_AUTO_INSTALL_CLOUDFLARED
tunnel_process = None
tunnel_url = None
tunnel_last_error = ""
tunnel_desired = False
tunnel_url_lock = threading.Lock()
tunnel_restart_lock = threading.Lock()
tunnel_restart_failures = 0
service_running = threading.Event()
STREAM_PROFILE_OPTIONS = (
    {"pixel_format": "RGB", "width": 640, "height": 480, "fps": 15.0},
    {"pixel_format": "RGB", "width": 1280, "height": 720, "fps": 24.0},
    {"pixel_format": "RGB", "width": 1280, "height": 720, "fps": 30.0},
    {"pixel_format": "RGB", "width": 1920, "height": 1080, "fps": 30.0},
)


@dataclass(frozen=True)
class AppSpec:
    key: str
    display_name: str
    app_name: str
    module_path: str
    class_name: str


@dataclass(frozen=True)
class ModelOption:
    spec: AppSpec
    model_name: str
    label: str


@dataclass(frozen=True)
class CameraOption:
    index: int
    label: str
    details: str = ""


@dataclass(frozen=True)
class LaunchSelection:
    run_mode: str
    model: Optional[ModelOption]
    camera: CameraOption
    orientation: int


APP_SPECS = [
    AppSpec(
        key="detection",
        display_name="Object Detection",
        app_name="detection",
        module_path="hailo_apps.python.pipeline_apps.detection.detection_pipeline",
        class_name="GStreamerDetectionApp",
    ),
    AppSpec(
        key="pose_estimation",
        display_name="Pose Estimation",
        app_name="pose_estimation",
        module_path="hailo_apps.python.pipeline_apps.pose_estimation.pose_estimation_pipeline",
        class_name="GStreamerPoseEstimationApp",
    ),
    AppSpec(
        key="instance_segmentation",
        display_name="Instance Segmentation",
        app_name="instance_segmentation",
        module_path="hailo_apps.python.pipeline_apps.instance_segmentation.instance_segmentation_pipeline",
        class_name="GStreamerInstanceSegmentationApp",
    ),
]


DASHBOARD_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Hailo Dashboard</title>
  <style>
    :root {
      --bg: #1f2328;
      --panel: #2b3138;
      --panel-soft: #343b44;
      --text: #f5f7fa;
      --muted: #9ea7b3;
      --accent: #ffae00;
      --radius: 1rem;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: Consolas, "Courier New", monospace;
      background:
        radial-gradient(circle at top right, #2a3038 0%, #1f2328 45%, #1a1e22 100%);
      color: var(--text);
      padding: 1.5rem;
    }
    .wrap {
      max-width: 1200px;
      margin: 0 auto;
      display: grid;
      gap: 1rem;
    }
    .title {
      display: flex;
      align-items: center;
      justify-content: space-between;
      background: var(--panel);
      border-radius: var(--radius);
      padding: 1rem 1.25rem;
      border: 1px solid #3d4550;
    }
    .title h1 {
      margin: 0;
      font-size: 1.1rem;
      letter-spacing: 0.02em;
      font-weight: 700;
    }
    .btn {
      border: 1px solid rgba(255, 174, 0, 0.55);
      background: rgba(255, 174, 0, 0.12);
      color: var(--accent);
      border-radius: 0.8rem;
      padding: 0.45rem 0.9rem;
      font-weight: 700;
      cursor: pointer;
    }
    .grid { display: grid; grid-template-columns: minmax(320px, 1fr); gap: 1rem; }
    .card {
      background: var(--panel);
      border-radius: var(--radius);
      border: 1px solid #3c444f;
      overflow: hidden;
      display: grid;
      gap: 0.75rem;
      padding: 0.75rem;
    }
    .meta {
      display: grid;
      gap: 0.35rem;
      background: var(--panel-soft);
      border-radius: var(--radius);
      padding: 0.75rem;
    }
    .line { display: flex; justify-content: space-between; gap: 1rem; }
    .k { color: var(--muted); }
    .v { color: var(--text); font-weight: 600; }
    .v.bad { color: #ff6b6b; }
    .stream {
      width: 100%;
      border-radius: var(--radius);
      border: 1px solid #4b5563;
      background: #14181c;
      display: block;
    }
    .drawer {
      position: fixed;
      top: 0;
      right: 0;
      width: min(480px, 100%);
      height: 100vh;
      background: #232a31;
      border-left: 1px solid #424b56;
      transform: translateX(102%);
      transition: transform 0.2s ease;
      padding: 1rem;
      overflow-y: auto;
      z-index: 10;
    }
    .drawer.open { transform: translateX(0); }
    .row {
      display: grid;
      gap: 0.35rem;
      margin-bottom: 0.75rem;
    }
    label {
      font-size: 0.8rem;
      color: var(--muted);
    }
    select, input {
      width: 100%;
      background: #2f363f;
      border: 1px solid #495261;
      color: var(--text);
      border-radius: 0.65rem;
      padding: 0.45rem 0.55rem;
    }
    .actions {
      display: flex;
      gap: 0.5rem;
      flex-wrap: wrap;
      margin-top: 0.7rem;
    }
    .note {
      margin-top: 0.6rem;
      font-size: 0.82rem;
      color: var(--muted);
    }
    @media (max-width: 680px) {
      body { padding: 0.8rem; }
      .title { padding: 0.75rem 1rem; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <section class="title">
      <h1>Hailo CSI Dashboard</h1>
      <button class="btn" id="openConfig">Config</button>
    </section>
    <section class="grid">
      <article class="card">
        <div class="meta">
          <div class="line"><span class="k">Model</span><span class="v" id="modelLabel">{{ initial.session.model_label }}</span></div>
          <div class="line"><span class="k">Camera</span><span class="v" id="cameraLabel">{{ initial.session.camera_label }}</span></div>
          <div class="line"><span class="k">Status</span><span class="v" id="statusText">running</span></div>
          <div class="line"><span class="k">Frames</span><span class="v" id="frames">{{ initial.session.frame_count }}</span></div>
          <div class="line"><span class="k">Avg FPS</span><span class="v" id="avgFps">{{ initial.session.avg_fps }}</span></div>
          <div class="line"><span class="k">Detections</span><span class="v" id="detections">{{ initial.session.detections }}</span></div>
          <div class="line"><span class="k">Config</span><span class="v" id="configPath">{{ config_path }}</span></div>
          <div class="line"><span class="k">Error</span><span class="v bad" id="errorText"></span></div>
          <div class="line"><span class="k">Restart Needed</span><span class="v" id="restartNeeded">no</span></div>
        </div>
        <img class="stream" id="streamImage" alt="Stream" />
      </article>
    </section>
  </div>
  <aside class="drawer" id="configDrawer">
    <section class="title" style="margin-bottom: 0.8rem;">
      <h1>Config</h1>
      <button class="btn" id="closeConfig">Close</button>
    </section>
    <div class="row">
      <label for="runMode">Run Mode</label>
      <select id="runMode">
        <option value="model">Model Pipeline</option>
        <option value="direct_camera">Direct Camera</option>
      </select>
    </div>
    <div class="row">
      <label for="modelFamily">Model Family</label>
      <select id="modelFamily"></select>
    </div>
    <div class="row">
      <label for="modelName">Model Variant</label>
      <select id="modelName"></select>
    </div>
    <div class="row">
      <label for="cameraIndex">Camera</label>
      <select id="cameraIndex"></select>
    </div>
    <div class="row">
      <label for="orientation">Orientation</label>
      <select id="orientation">
        <option value="0">0°</option>
        <option value="90">90°</option>
        <option value="180">180°</option>
        <option value="270">270°</option>
      </select>
    </div>
    <div class="row">
      <label for="cfgWidth">Width</label>
      <input id="cfgWidth" type="number" min="320" />
    </div>
    <div class="row">
      <label for="cfgHeight">Height</label>
      <input id="cfgHeight" type="number" min="240" />
    </div>
    <div class="row">
      <label for="cfgFps">FPS</label>
      <input id="cfgFps" type="number" min="1" />
    </div>
    <div class="row">
      <label for="cfgHost">Host (save only, requires dashboard restart)</label>
      <input id="cfgHost" type="text" />
    </div>
    <div class="row">
      <label for="cfgPort">Port (save only, requires dashboard restart)</label>
      <input id="cfgPort" type="number" min="1" />
    </div>
    <div class="actions">
      <button class="btn" id="saveApply">Save + Apply Live</button>
      <button class="btn" id="saveOnly">Save Only</button>
      <button class="btn" id="reloadDisk">Reload Config</button>
      <button class="btn" id="refreshCameras">Refresh Cameras</button>
    </div>
    <div class="note" id="configNote"></div>
  </aside>
  <script>
    const sessionKey = "{{ session_key }}";
    const requireAuth = {{ require_auth | tojson }};
    let modelCatalog = {};
    let cameras = [];
    let currentConfig = {};
    let serverConfig = {};
    let drawerOpen = false;
    let formDirty = false;

    function setConfigNote(text) {
      document.getElementById('configNote').textContent = text || '';
    }

    function withSession(path) {
      const raw = String(path || '').trim();
      if (!raw) {
        return raw;
      }
      const key = String(sessionKey || '').trim();
      if (!key) {
        return raw;
      }
      const joiner = raw.includes('?') ? '&' : '?';
      return `${raw}${joiner}session_key=${encodeURIComponent(key)}`;
    }

    function markDirty() {
      formDirty = true;
      setConfigNote('Unsaved changes...');
    }

    function updateModelNameOptions() {
      const family = document.getElementById('modelFamily').value;
      const modelSelect = document.getElementById('modelName');
      const models = modelCatalog[family] || [];
      modelSelect.innerHTML = '';
      models.forEach((name) => {
        const option = document.createElement('option');
        option.value = name;
        option.textContent = name;
        modelSelect.appendChild(option);
      });
      if (!models.includes(currentConfig.model_name) && models.length) {
        modelSelect.value = models[0];
      } else {
        modelSelect.value = currentConfig.model_name || '';
      }
    }

    function fillForm() {
      const familySelect = document.getElementById('modelFamily');
      familySelect.innerHTML = '';
      Object.keys(modelCatalog).forEach((family) => {
        const option = document.createElement('option');
        option.value = family;
        option.textContent = family;
        familySelect.appendChild(option);
      });
      familySelect.value = currentConfig.model_family;
      updateModelNameOptions();

      const cameraSelect = document.getElementById('cameraIndex');
      cameraSelect.innerHTML = '';
      cameras.forEach((camera) => {
        const option = document.createElement('option');
        option.value = String(camera.index);
        option.textContent = camera.details ? `${camera.label} (${camera.details})` : camera.label;
        cameraSelect.appendChild(option);
      });
      cameraSelect.value = String(currentConfig.camera_index);

      const orientation = (currentConfig.camera_orientations || {})[String(currentConfig.camera_index)] || 0;
      document.getElementById('orientation').value = String(orientation);
      document.getElementById('runMode').value = currentConfig.run_mode;
      document.getElementById('cfgWidth').value = currentConfig.width;
      document.getElementById('cfgHeight').value = currentConfig.height;
      document.getElementById('cfgFps').value = currentConfig.fps;
      document.getElementById('cfgHost').value = currentConfig.host;
      document.getElementById('cfgPort').value = currentConfig.port;
    }

    function updateStatus(snapshot) {
      const session = snapshot.session;
      document.getElementById('modelLabel').textContent = session.model_label || '';
      document.getElementById('cameraLabel').textContent = session.camera_label || '';
      document.getElementById('statusText').textContent = session.error ? 'error' : 'running';
      document.getElementById('frames').textContent = session.frame_count || 0;
      document.getElementById('avgFps').textContent = (session.avg_fps || 0).toFixed(2);
      document.getElementById('detections').textContent = session.detections || 0;
      document.getElementById('errorText').textContent = session.error || '';
      document.getElementById('restartNeeded').textContent = snapshot.requires_server_restart ? 'yes' : 'no';
    }

    function applySnapshot(snapshot, forceForm = false) {
      modelCatalog = snapshot.model_catalog || {};
      cameras = snapshot.cameras || [];
      serverConfig = snapshot.config || {};
      const shouldFillForm = forceForm || !drawerOpen || !formDirty;
      if (shouldFillForm) {
        currentConfig = serverConfig;
        fillForm();
        formDirty = false;
      }
      updateStatus(snapshot);
    }

    async function fetchSnapshot(forceForm = false) {
      try {
        const response = await fetch(withSession('/api/status'));
        if (response.status === 401) {
          setConfigNote('Unauthorized. Authenticate via /auth and reopen /dashboard?session_key=...');
          return;
        }
        const data = await response.json();
        applySnapshot(data, forceForm);
      } catch (_) {}
    }

    function collectConfigPatch() {
      return {
        run_mode: document.getElementById('runMode').value,
        model_family: document.getElementById('modelFamily').value,
        model_name: document.getElementById('modelName').value,
        camera_index: Number(document.getElementById('cameraIndex').value),
        orientation: Number(document.getElementById('orientation').value),
        width: Number(document.getElementById('cfgWidth').value),
        height: Number(document.getElementById('cfgHeight').value),
        fps: Number(document.getElementById('cfgFps').value),
        host: document.getElementById('cfgHost').value,
        port: Number(document.getElementById('cfgPort').value)
      };
    }

    async function postConfig(path, body) {
      const response = await fetch(withSession(path), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });
      if (response.status === 401) {
        setConfigNote('Unauthorized. Session expired.');
        return;
      }
      const data = await response.json();
      applySnapshot(data, true);
      setConfigNote('Config updated.');
    }

    document.getElementById('openConfig').onclick = async () => {
      document.getElementById('configDrawer').classList.add('open');
      drawerOpen = true;
      formDirty = false;
      await fetchSnapshot(true);
    };
    document.getElementById('closeConfig').onclick = () => {
      document.getElementById('configDrawer').classList.remove('open');
      drawerOpen = false;
      formDirty = false;
      setConfigNote('');
      fetchSnapshot(true);
    };
    document.getElementById('modelFamily').onchange = () => {
      currentConfig.model_family = document.getElementById('modelFamily').value;
      updateModelNameOptions();
      markDirty();
    };
    document.getElementById('cameraIndex').onchange = () => {
      const cameraIndex = document.getElementById('cameraIndex').value;
      const orientation = (currentConfig.camera_orientations || {})[cameraIndex] || 0;
      document.getElementById('orientation').value = String(orientation);
      markDirty();
    };
    document.getElementById('saveApply').onclick = async () => {
      await postConfig('/api/config', { config: collectConfigPatch(), save: true, apply: true });
    };
    document.getElementById('saveOnly').onclick = async () => {
      await postConfig('/api/config', { config: collectConfigPatch(), save: true, apply: false });
    };
    document.getElementById('reloadDisk').onclick = async () => {
      await postConfig('/api/config/reload', { apply: true });
    };
    document.getElementById('refreshCameras').onclick = async () => {
      await postConfig('/api/cameras/refresh', { apply: false });
    };

    ['runMode', 'modelName', 'orientation', 'cfgWidth', 'cfgHeight', 'cfgFps', 'cfgHost', 'cfgPort']
      .forEach((id) => {
        const node = document.getElementById(id);
        if (node) {
          node.addEventListener('input', markDirty);
          node.addEventListener('change', markDirty);
        }
      });

    document.getElementById('streamImage').src = withSession('/stream');
    fetchSnapshot(true);
    setInterval(fetchSnapshot, 1200);
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone Hailo CSI dashboard launcher")
    parser.add_argument("--host", default=None, help="Flask bind host")
    parser.add_argument("--port", type=int, default=None, help="Flask bind port")
    parser.add_argument(
        "--hailo-apps-path",
        default=None,
        help="Path to an existing hailo-apps repo root (auto-detected when omitted).",
    )
    parser.add_argument(
        "--repo-url",
        default=DEFAULT_HAILO_REPO_URL,
        help="Git URL used when auto-cloning hailo-apps.",
    )
    parser.add_argument(
        "--clone-dir-name",
        default=DEFAULT_HAILO_REPO_DIRNAME,
        help="Directory name to use for cloned hailo-apps.",
    )
    parser.add_argument(
        "--no-auto-clone",
        action="store_true",
        help="Disable automatic git clone when hailo-apps repo is not found.",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip running hailo-apps install.sh during auto-setup.",
    )
    parser.add_argument(
        "--force-install",
        action="store_true",
        help="Force running hailo-apps install.sh even if setup markers exist.",
    )
    parser.add_argument(
        "--no-sudo-install",
        action="store_true",
        help="Run install.sh without sudo prefix.",
    )
    parser.add_argument(
        "--venv-dir",
        default="venv_hailo_apps",
        help="Virtual environment path (relative to project root or absolute)",
    )
    parser.add_argument(
        "--config-file",
        default=None,
        help="Path to JSON config file (default: <script_dir>/config.json)",
    )
    parser.add_argument("--skip-bootstrap", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-curses", action="store_true", help="Disable curses selector")
    parser.add_argument(
        "--model",
        choices=[spec.key for spec in APP_SPECS],
        default=None,
        help="Pre-select model family",
    )
    parser.add_argument("--model-name", default=None, help="Pre-select exact model name")
    parser.add_argument(
        "--direct-camera",
        action="store_true",
        help="Run direct camera stream with no Hailo model pipeline",
    )
    parser.add_argument("--camera-index", type=int, default=None, help="Pre-select camera index")
    parser.add_argument(
        "--orientation",
        type=int,
        choices=VALID_ORIENTATIONS,
        default=None,
        help="Camera orientation in degrees (0/90/180/270)",
    )
    parser.add_argument("--width", type=int, default=None, help="Pipeline output width")
    parser.add_argument("--height", type=int, default=None, help="Pipeline output height")
    parser.add_argument("--fps", type=int, default=None, help="Pipeline FPS")
    return parser.parse_args()


def is_hailo_apps_repo(path: Path) -> bool:
    return (
        path.exists()
        and (path / "pyproject.toml").exists()
        and (path / "install.sh").exists()
        and (path / "hailo_apps").exists()
    )


def resolve_project_root(script_dir: Path, explicit_repo: str | None = None) -> Path | None:
    if explicit_repo:
        explicit_path = Path(explicit_repo).expanduser().resolve()
        if is_hailo_apps_repo(explicit_path):
            return explicit_path
        raise RuntimeError(f"--hailo-apps-path is not a valid hailo-apps repo: {explicit_path}")

    candidates: List[Path] = [script_dir, *script_dir.parents]

    # 1) Script inside repo root or subdirectory.
    for candidate in candidates:
        if is_hailo_apps_repo(candidate):
            return candidate

    # 2) Sibling hailo-apps folders at each parent level.
    for parent in [script_dir, *script_dir.parents]:
        sibling = parent / DEFAULT_HAILO_REPO_DIRNAME
        if is_hailo_apps_repo(sibling):
            return sibling

    return None


def clone_hailo_apps_repo(script_dir: Path, args: argparse.Namespace) -> Path:
    if args.no_auto_clone:
        raise RuntimeError(
            "hailo-apps repo was not found and --no-auto-clone is enabled. "
            "Provide --hailo-apps-path or allow auto-clone."
        )

    clone_parent = script_dir.parent if script_dir.name.lower() == "camera" else script_dir
    clone_parent = clone_parent.resolve()
    target_repo = clone_parent / args.clone_dir_name

    if target_repo.exists() and not is_hailo_apps_repo(target_repo):
        raise RuntimeError(
            f"Clone target exists but is not a hailo-apps repo: {target_repo}"
        )
    if is_hailo_apps_repo(target_repo):
        return target_repo

    print(f"[setup] cloning hailo-apps from {args.repo_url} into {target_repo}")
    run_checked(["git", "clone", args.repo_url, str(target_repo)], cwd=clone_parent)
    if not is_hailo_apps_repo(target_repo):
        raise RuntimeError(f"Clone completed but repo markers are missing at: {target_repo}")
    return target_repo


def ensure_hailo_apps_installation(repo_root: Path, args: argparse.Namespace) -> None:
    if args.skip_install:
        return

    install_marker_exists = DEFAULT_ENV_FILE.exists()
    should_install = args.force_install or not install_marker_exists
    if not should_install:
        return

    install_script = repo_root / "install.sh"
    if not install_script.exists():
        raise RuntimeError(f"install.sh missing at expected location: {install_script}")

    install_cmd: List[str] = []
    if not args.no_sudo_install and os.geteuid() != 0:
        install_cmd.append("sudo")
    install_cmd.extend(["bash", str(install_script)])

    print(f"[setup] running install script: {' '.join(shlex.quote(x) for x in install_cmd)}")
    run_checked(install_cmd, cwd=repo_root)


def resolve_config_path(args: argparse.Namespace) -> Path:
    if args.config_file:
        return Path(args.config_file).expanduser().resolve()
    return Path(__file__).resolve().with_name("config.json")


def build_model_catalog(model_options: List[ModelOption]) -> Dict[str, List[ModelOption]]:
    catalog: Dict[str, List[ModelOption]] = {}
    for option in model_options:
        catalog.setdefault(option.spec.key, []).append(option)
    return catalog


def default_config(model_options: List[ModelOption], camera_options: List[CameraOption]) -> dict:
    model_catalog = build_model_catalog(model_options)
    first_family = next(iter(model_catalog))
    first_model = model_catalog[first_family][0].model_name
    first_camera_index = camera_options[0].index
    first_camera_key = str(first_camera_index)
    return {
        "run_mode": "model",
        "model_family": first_family,
        "model_name": first_model,
        "camera_index": first_camera_index,
        "camera_orientations": {first_camera_key: 0},
        "camera_enabled": {first_camera_key: True},
        "camera_profiles": {
            first_camera_key: {
                "pixel_format": "RGB",
                "width": 1280,
                "height": 720,
                "fps": 30.0,
            }
        },
        "host": "0.0.0.0",
        "port": 8080,
        "width": 1280,
        "height": 720,
        "fps": 30,
        "security": {
            "password": DEFAULT_PASSWORD,
            "session_timeout": DEFAULT_SESSION_TIMEOUT,
            "require_auth": DEFAULT_REQUIRE_AUTH,
        },
        "tunnel": {
            "enable": DEFAULT_ENABLE_TUNNEL,
            "auto_install_cloudflared": DEFAULT_AUTO_INSTALL_CLOUDFLARED,
        },
    }


def load_raw_config(config_path: Path) -> dict:
    if not config_path.exists():
        return {}
    try:
        payload = json.loads(config_path.read_text())
        if isinstance(payload, dict):
            legacy = payload.get("camera_router")
            if isinstance(legacy, dict):
                network = legacy.get("network", {}) if isinstance(legacy.get("network"), dict) else {}
                stream = legacy.get("stream", {}) if isinstance(legacy.get("stream"), dict) else {}
                security = legacy.get("security", {}) if isinstance(legacy.get("security"), dict) else {}
                tunnel = legacy.get("tunnel", {}) if isinstance(legacy.get("tunnel"), dict) else {}
                cameras = legacy.get("cameras", []) if isinstance(legacy.get("cameras"), list) else []

                if "host" not in payload and network.get("listen_host") is not None:
                    payload["host"] = network.get("listen_host")
                if "port" not in payload and network.get("listen_port") is not None:
                    payload["port"] = network.get("listen_port")
                if "fps" not in payload and stream.get("target_fps") is not None:
                    payload["fps"] = stream.get("target_fps")
                if "security" not in payload and security:
                    payload["security"] = security
                if "tunnel" not in payload and tunnel:
                    payload["tunnel"] = tunnel

                orientation_map = dict(payload.get("camera_orientations", {}))
                enabled_map = dict(payload.get("camera_enabled", {}))
                profile_map = dict(payload.get("camera_profiles", {}))
                for row in cameras:
                    if not isinstance(row, dict):
                        continue
                    idx = _coerce_int(row.get("index"), -1)
                    if idx < 0:
                        continue
                    key = str(idx)
                    orientation_map.setdefault(key, row.get("rotation", 0))
                    enabled_map.setdefault(key, bool(row.get("enabled", True)))
                    profile_map.setdefault(
                        key,
                        {
                            "pixel_format": "RGB",
                            "width": _coerce_int(row.get("width"), 1280),
                            "height": _coerce_int(row.get("height"), 720),
                            "fps": float(_coerce_int(stream.get("target_fps", row.get("fps", 30)), 30)),
                        },
                    )
                payload["camera_orientations"] = orientation_map
                payload["camera_enabled"] = enabled_map
                payload["camera_profiles"] = profile_map
            return payload
    except Exception:
        pass
    return {}


def save_config(config_path: Path, payload: dict) -> None:
    clean_payload = {key: value for key, value in payload.items() if not str(key).startswith("_")}
    security = clean_payload.get("security", {}) if isinstance(clean_payload.get("security"), dict) else {}
    tunnel = clean_payload.get("tunnel", {}) if isinstance(clean_payload.get("tunnel"), dict) else {}
    camera_orientations = clean_payload.get("camera_orientations", {})
    camera_enabled = clean_payload.get("camera_enabled", {})
    camera_profiles = clean_payload.get("camera_profiles", {})
    camera_rows = []
    for key in sorted(camera_orientations.keys(), key=lambda value: _coerce_int(value, 10_000)):
        idx = _coerce_int(key, -1)
        if idx < 0:
            continue
        profile = camera_profiles.get(key, {}) if isinstance(camera_profiles, dict) else {}
        camera_rows.append(
            {
                "id": _camera_id_for_index(idx),
                "index": idx,
                "enabled": bool(camera_enabled.get(key, True)),
                "width": _coerce_int((profile or {}).get("width"), _coerce_int(clean_payload.get("width"), 1280)),
                "height": _coerce_int((profile or {}).get("height"), _coerce_int(clean_payload.get("height"), 720)),
                "rotation": _coerce_int(camera_orientations.get(key), 0),
            }
        )

    clean_payload["camera_router"] = {
        "network": {
            "listen_host": str(clean_payload.get("host") or "0.0.0.0"),
            "listen_port": _coerce_int(clean_payload.get("port"), 8080),
        },
        "stream": {
            "target_fps": _coerce_int(clean_payload.get("fps"), 30),
            "jpeg_quality": 82,
            "use_picamera2": True,
        },
        "security": {
            "password": str(security.get("password") or DEFAULT_PASSWORD),
            "session_timeout": _coerce_int(security.get("session_timeout"), DEFAULT_SESSION_TIMEOUT),
            "require_auth": _coerce_bool(security.get("require_auth"), DEFAULT_REQUIRE_AUTH),
        },
        "tunnel": {
            "enable": _coerce_bool(tunnel.get("enable"), DEFAULT_ENABLE_TUNNEL),
            "auto_install_cloudflared": _coerce_bool(
                tunnel.get("auto_install_cloudflared"),
                DEFAULT_AUTO_INSTALL_CLOUDFLARED,
            ),
        },
        "cameras": camera_rows,
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(clean_payload, indent=2, sort_keys=True) + "\n")


def _coerce_int(value, fallback: int) -> int:
    try:
        return int(value)
    except Exception:
        return fallback


def _coerce_float(value, fallback: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(fallback)


def _coerce_bool(value, fallback: bool = False) -> bool:
    if isinstance(value, bool):
        return bool(value)
    if value is None:
        return bool(fallback)
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _camera_id_for_index(index: int) -> str:
    return f"cam{int(index)}"


def _camera_index_from_id(camera_id: str) -> int:
    text = str(camera_id or "").strip().lower()
    if text.startswith("cam"):
        text = text[3:]
    if text.startswith("csi"):
        text = text[3:]
    if text.startswith("default_"):
        text = text.split("_", 1)[1]
    if text.startswith("camera"):
        text = text[6:]
    return _coerce_int(text, -1)


def normalize_orientation_map(raw_map, camera_options: List[CameraOption]) -> Dict[str, int]:
    orientation_map: Dict[str, int] = {}
    if isinstance(raw_map, dict):
        for key, value in raw_map.items():
            index = _coerce_int(key, -1)
            angle = _coerce_int(value, 0)
            if index >= 0 and angle in VALID_ORIENTATIONS:
                orientation_map[str(index)] = angle
    for camera in camera_options:
        orientation_map.setdefault(str(camera.index), 0)
    return orientation_map


def normalize_camera_enabled_map(raw_map, camera_options: List[CameraOption]) -> Dict[str, bool]:
    enabled_map: Dict[str, bool] = {}
    if isinstance(raw_map, dict):
        for key, value in raw_map.items():
            index = _coerce_int(key, -1)
            if index >= 0:
                enabled_map[str(index)] = _coerce_bool(value, True)
    for camera in camera_options:
        enabled_map.setdefault(str(camera.index), True)
    return enabled_map


def _normalize_profile(value, default_width: int, default_height: int, default_fps: int) -> dict:
    profile = value if isinstance(value, dict) else {}
    return {
        "pixel_format": str(profile.get("pixel_format", "RGB") or "RGB").strip().upper() or "RGB",
        "width": max(320, _coerce_int(profile.get("width"), int(default_width))),
        "height": max(240, _coerce_int(profile.get("height"), int(default_height))),
        "fps": max(1.0, _coerce_float(profile.get("fps"), float(default_fps))),
    }


def normalize_camera_profiles(
    raw_profiles,
    camera_options: List[CameraOption],
    default_width: int,
    default_height: int,
    default_fps: int,
) -> Dict[str, dict]:
    profiles: Dict[str, dict] = {}
    if isinstance(raw_profiles, dict):
        for key, value in raw_profiles.items():
            index = _coerce_int(key, -1)
            if index >= 0:
                profiles[str(index)] = _normalize_profile(value, default_width, default_height, default_fps)
    for camera in camera_options:
        camera_key = str(camera.index)
        profiles.setdefault(
            camera_key,
            _normalize_profile({}, default_width, default_height, default_fps),
        )
    return profiles


def _normalized_security(raw_security) -> dict:
    security = raw_security if isinstance(raw_security, dict) else {}
    return {
        "password": str(security.get("password") or DEFAULT_PASSWORD).strip() or DEFAULT_PASSWORD,
        "session_timeout": max(30, _coerce_int(security.get("session_timeout"), DEFAULT_SESSION_TIMEOUT)),
        "require_auth": _coerce_bool(security.get("require_auth"), DEFAULT_REQUIRE_AUTH),
    }


def _normalized_tunnel(raw_tunnel) -> dict:
    tunnel = raw_tunnel if isinstance(raw_tunnel, dict) else {}
    return {
        "enable": _coerce_bool(tunnel.get("enable"), DEFAULT_ENABLE_TUNNEL),
        "auto_install_cloudflared": _coerce_bool(
            tunnel.get("auto_install_cloudflared"),
            DEFAULT_AUTO_INSTALL_CLOUDFLARED,
        ),
    }


def _log(message: str) -> None:
    try:
        print(str(message), flush=True)
    except Exception:
        pass


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
        _log(f"[ERROR] Unsupported platform for cloudflared: {system} {machine}")
        return False

    try:
        import urllib.request

        _log(f"[TUNNEL] Downloading cloudflared from {url}")
        urllib.request.urlretrieve(url, cloudflared_path)
        if os.name != "nt":
            os.chmod(cloudflared_path, 0o755)
        _log("[TUNNEL] cloudflared installed successfully")
        return True
    except Exception as exc:
        _log(f"[ERROR] Failed to install cloudflared: {exc}")
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
    _log(f"[START] Launching cloudflared: {' '.join(cmd)}")
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
        _log(f"[ERROR] Failed to start cloudflared tunnel: {exc}")
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
                            captured_url = match.group(0).strip().rstrip("/")
                            tunnel_url = captured_url
                            found_url = True
                            tunnel_last_error = ""
                            tunnel_restart_failures = 0
                            _log(f"[TUNNEL] Camera URL: {tunnel_url}")

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
            _log(f"[WARN] {tunnel_last_error}")
            if tunnel_desired and service_running.is_set():
                delay = _next_tunnel_restart_delay(rate_limited=rate_limited and not found_url)
                _log(f"[WARN] Restarting cloudflared in {delay:.1f}s...")
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

    env_tunnel = str(os.environ.get(DEFAULT_TUNNEL_URL_ENV, "")).strip().rstrip("/")
    source = "cloudflared"
    if env_tunnel and not current_tunnel:
        current_tunnel = env_tunnel
        stale_tunnel = ""
        source = "env"

    state = "active" if current_tunnel else ("starting" if process_running else "inactive")
    if stale_tunnel and not process_running and not current_tunnel:
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
        "source": source if current_tunnel else "",
    }


def _apply_tunnel_runtime_config(config: dict, startup_delay: float = 0.0) -> None:
    global tunnel_enabled, auto_install_cloudflared, tunnel_last_error
    cfg = config if isinstance(config, dict) else {}
    tunnel_cfg = _normalized_tunnel(cfg.get("tunnel", {}))
    tunnel_enabled = bool(tunnel_cfg.get("enable", DEFAULT_ENABLE_TUNNEL))
    auto_install_cloudflared = bool(
        tunnel_cfg.get("auto_install_cloudflared", DEFAULT_AUTO_INSTALL_CLOUDFLARED)
    )
    listen_port = max(1, _coerce_int(cfg.get("port"), 8080))
    env_tunnel = str(os.environ.get(DEFAULT_TUNNEL_URL_ENV, "")).strip().rstrip("/")

    if not tunnel_enabled:
        _stop_cloudflared_tunnel()
        tunnel_last_error = "Tunnel disabled by config"
        return

    if env_tunnel:
        _stop_cloudflared_tunnel()
        tunnel_last_error = ""
        return

    if not _is_cloudflared_installed():
        if auto_install_cloudflared:
            _log("[TUNNEL] cloudflared not found, attempting install...")
            if not _install_cloudflared():
                tunnel_last_error = "cloudflared install failed"
                return
        else:
            tunnel_last_error = "cloudflared missing and auto-install disabled"
            _log("[WARN] cloudflared missing and auto-install disabled")
            return

    running = tunnel_process is not None and tunnel_process.poll() is None
    if running:
        return

    def _launch() -> None:
        if startup_delay > 0:
            time.sleep(max(0.0, float(startup_delay)))
        if not service_running.is_set():
            return
        if not tunnel_enabled:
            return
        _start_cloudflared_tunnel(listen_port)

    threading.Thread(target=_launch, daemon=True).start()


def _camera_row_with_urls(row: dict, publish_base: str, local_base: str, lan_base: str, tunnel_base: str) -> dict:
    camera_id = str(row.get("id", "")).strip()
    return {
        **row,
        "snapshot_url": f"{publish_base}/snapshot/{camera_id}" if publish_base else f"/snapshot/{camera_id}",
        "jpeg_url": f"{publish_base}/jpeg/{camera_id}" if publish_base else f"/jpeg/{camera_id}",
        "video_url": f"{publish_base}/video/{camera_id}" if publish_base else f"/video/{camera_id}",
        "mjpeg_url": f"{publish_base}/mjpeg/{camera_id}" if publish_base else f"/mjpeg/{camera_id}",
        "local_snapshot_url": f"{local_base}/snapshot/{camera_id}" if local_base else "",
        "local_jpeg_url": f"{local_base}/jpeg/{camera_id}" if local_base else "",
        "local_video_url": f"{local_base}/video/{camera_id}" if local_base else "",
        "local_mjpeg_url": f"{local_base}/mjpeg/{camera_id}" if local_base else "",
        "lan_snapshot_url": f"{lan_base}/snapshot/{camera_id}" if lan_base else "",
        "lan_jpeg_url": f"{lan_base}/jpeg/{camera_id}" if lan_base else "",
        "lan_video_url": f"{lan_base}/video/{camera_id}" if lan_base else "",
        "lan_mjpeg_url": f"{lan_base}/mjpeg/{camera_id}" if lan_base else "",
        "tunnel_snapshot_url": f"{tunnel_base}/snapshot/{camera_id}" if tunnel_base else "",
        "tunnel_jpeg_url": f"{tunnel_base}/jpeg/{camera_id}" if tunnel_base else "",
        "tunnel_video_url": f"{tunnel_base}/video/{camera_id}" if tunnel_base else "",
        "tunnel_mjpeg_url": f"{tunnel_base}/mjpeg/{camera_id}" if tunnel_base else "",
    }


def runtime_config_signature(config: dict) -> tuple:
    cfg = config if isinstance(config, dict) else {}
    return (
        str(cfg.get("run_mode") or ""),
        str(cfg.get("model_family") or ""),
        str(cfg.get("model_name") or ""),
        int(_coerce_int(cfg.get("camera_index"), 0)),
        int(_coerce_int(cfg.get("width"), 0)),
        int(_coerce_int(cfg.get("height"), 0)),
        int(_coerce_int(cfg.get("fps"), 0)),
        json.dumps(cfg.get("camera_orientations", {}), sort_keys=True, separators=(",", ":")),
        json.dumps(cfg.get("camera_enabled", {}), sort_keys=True, separators=(",", ":")),
        json.dumps(cfg.get("camera_profiles", {}), sort_keys=True, separators=(",", ":")),
    )


def normalize_config(
    raw_config: dict,
    model_options: List[ModelOption],
    camera_options: List[CameraOption],
) -> dict:
    cfg = default_config(model_options, camera_options)
    catalog = build_model_catalog(model_options)
    allow_custom_model_name = bool(raw_config.get("_allow_custom_model_name", False))

    run_mode = raw_config.get("run_mode", cfg["run_mode"])
    if run_mode not in ("model", "direct_camera"):
        run_mode = cfg["run_mode"]
    cfg["run_mode"] = run_mode

    family = str(raw_config.get("model_family", cfg["model_family"]))
    if family not in catalog:
        family = cfg["model_family"]
    cfg["model_family"] = family

    family_models = [item.model_name for item in catalog[family]]
    model_name_raw = raw_config.get("model_name", cfg["model_name"])
    model_name = str(model_name_raw)
    if model_name not in family_models and allow_custom_model_name:
        model_name = str(model_name_raw)
    elif model_name not in family_models:
        model_name = family_models[0]
    cfg["model_name"] = model_name

    known_camera_indices = {camera.index for camera in camera_options}
    camera_index = _coerce_int(raw_config.get("camera_index", cfg["camera_index"]), cfg["camera_index"])
    if camera_index not in known_camera_indices:
        camera_index = camera_options[0].index
    cfg["camera_index"] = camera_index

    cfg["camera_orientations"] = normalize_orientation_map(
        raw_config.get("camera_orientations", cfg["camera_orientations"]),
        camera_options,
    )
    if str(camera_index) not in cfg["camera_orientations"]:
        cfg["camera_orientations"][str(camera_index)] = 0

    cfg["camera_enabled"] = normalize_camera_enabled_map(
        raw_config.get("camera_enabled", cfg.get("camera_enabled", {})),
        camera_options,
    )

    cfg["host"] = str(raw_config.get("host", cfg["host"]))
    cfg["port"] = _coerce_int(raw_config.get("port", cfg["port"]), cfg["port"])
    cfg["width"] = max(320, _coerce_int(raw_config.get("width", cfg["width"]), cfg["width"]))
    cfg["height"] = max(240, _coerce_int(raw_config.get("height", cfg["height"]), cfg["height"]))
    cfg["fps"] = max(1, _coerce_int(raw_config.get("fps", cfg["fps"]), cfg["fps"]))
    cfg["camera_profiles"] = normalize_camera_profiles(
        raw_config.get("camera_profiles", cfg.get("camera_profiles", {})),
        camera_options,
        cfg["width"],
        cfg["height"],
        cfg["fps"],
    )
    cfg["security"] = _normalized_security(raw_config.get("security", cfg.get("security", {})))
    cfg["tunnel"] = _normalized_tunnel(raw_config.get("tunnel", cfg.get("tunnel", {})))
    return cfg


def apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    merged = dict(config)
    merged["camera_orientations"] = dict(config["camera_orientations"])
    merged["camera_enabled"] = dict(config.get("camera_enabled", {}))
    merged["camera_profiles"] = dict(config.get("camera_profiles", {}))
    merged["security"] = dict(config.get("security", {}))
    merged["tunnel"] = dict(config.get("tunnel", {}))

    if args.direct_camera:
        merged["run_mode"] = "direct_camera"
    if args.model:
        merged["run_mode"] = "model"
        merged["model_family"] = args.model
    if args.model_name:
        merged["model_name"] = args.model_name
        merged["_allow_custom_model_name"] = True
    if args.camera_index is not None:
        merged["camera_index"] = args.camera_index
        merged["camera_orientations"].setdefault(str(args.camera_index), 0)
    if args.orientation is not None:
        merged["camera_orientations"][str(merged["camera_index"])] = args.orientation
    if args.host:
        merged["host"] = args.host
    if args.port is not None:
        merged["port"] = args.port
    if args.width is not None:
        merged["width"] = args.width
    if args.height is not None:
        merged["height"] = args.height
    if args.fps is not None:
        merged["fps"] = args.fps
    selected_key = str(_coerce_int(merged.get("camera_index"), 0))
    selected_profile = dict(merged.get("camera_profiles", {}).get(selected_key, {}))
    selected_profile["width"] = max(320, _coerce_int(merged.get("width"), 1280))
    selected_profile["height"] = max(240, _coerce_int(merged.get("height"), 720))
    selected_profile["fps"] = max(1.0, _coerce_float(merged.get("fps"), 30.0))
    selected_profile.setdefault("pixel_format", "RGB")
    merged.setdefault("camera_profiles", {})
    merged["camera_profiles"][selected_key] = selected_profile
    return merged


def get_orientation_for_camera(config: dict, camera_index: int) -> int:
    value = config.get("camera_orientations", {}).get(str(camera_index), 0)
    return value if value in VALID_ORIENTATIONS else 0


def set_orientation_for_camera(config: dict, camera_index: int, orientation: int) -> None:
    config.setdefault("camera_orientations", {})
    config["camera_orientations"][str(camera_index)] = orientation if orientation in VALID_ORIENTATIONS else 0


def is_camera_enabled(config: dict, camera_index: int) -> bool:
    enabled_map = config.get("camera_enabled", {})
    if not isinstance(enabled_map, dict):
        return True
    return _coerce_bool(enabled_map.get(str(camera_index)), True)


def get_profile_for_camera(config: dict, camera_index: int) -> dict:
    profiles = config.get("camera_profiles", {})
    selected = profiles.get(str(camera_index), {}) if isinstance(profiles, dict) else {}
    return _normalize_profile(
        selected,
        _coerce_int(config.get("width"), 1280),
        _coerce_int(config.get("height"), 720),
        _coerce_int(config.get("fps"), 30),
    )


def select_model_option(
    config: dict,
    model_options: List[ModelOption],
) -> Optional[ModelOption]:
    if config.get("run_mode") == "direct_camera":
        return None
    catalog = build_model_catalog(model_options)
    family = config.get("model_family")
    if family not in catalog:
        return None
    model_name = config.get("model_name")
    for option in catalog[family]:
        if option.model_name == model_name:
            return option
    if model_name:
        base = catalog[family][0]
        return ModelOption(
            spec=base.spec,
            model_name=str(model_name),
            label=f"{base.spec.display_name} | {model_name}",
        )
    return catalog[family][0]


def resolve_camera_option(config: dict, camera_options: List[CameraOption]) -> CameraOption:
    requested_index = config.get("camera_index")
    for camera in camera_options:
        if camera.index == requested_index and is_camera_enabled(config, camera.index):
            return camera
    for camera in camera_options:
        if is_camera_enabled(config, camera.index):
            return camera
    for camera in camera_options:
        if camera.index == requested_index:
            return camera
    return camera_options[0]


def apply_orientation(frame, orientation: int):
    if orientation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if orientation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if orientation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def run_checked(cmd: List[str], cwd: Path | None = None) -> None:
    print(f"[bootstrap] {' '.join(shlex.quote(x) for x in cmd)}")
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def venv_has_runtime(venv_python: Path) -> bool:
    probe = [
        str(venv_python),
        "-c",
        "import flask; import flask_cors; import hailo_apps; import setproctitle; import cv2; import numpy",
    ]
    result = subprocess.run(
        probe, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
    )
    return result.returncode == 0


def ensure_bootstrap_environment(project_root: Path, args: argparse.Namespace) -> None:
    venv_dir = Path(args.venv_dir)
    if not venv_dir.is_absolute():
        venv_dir = (project_root / venv_dir).resolve()
    venv_python = venv_dir / "bin" / "python3"
    stamp_file = venv_dir / BOOTSTRAP_STAMP_NAME
    pyproject_mtime = (project_root / "pyproject.toml").stat().st_mtime_ns
    expected_stamp = f"{BOOTSTRAP_STAMP_VALUE}:{pyproject_mtime}"

    if not venv_dir.exists():
        print(f"[bootstrap] creating venv at {venv_dir}")
        builder = venv.EnvBuilder(with_pip=True, system_site_packages=True)
        builder.create(str(venv_dir))

    if not venv_python.exists():
        raise RuntimeError(f"venv python not found: {venv_python}")

    needs_install = True
    if stamp_file.exists() and stamp_file.read_text().strip() == expected_stamp:
        needs_install = not venv_has_runtime(venv_python)

    if needs_install:
        run_checked([str(venv_python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
        run_checked([str(venv_python), "-m", "pip", "install", "-e", str(project_root)], cwd=project_root)
        run_checked([str(venv_python), "-m", "pip", "install", "flask", "flask-cors"])
        stamp_file.write_text(expected_stamp + "\n")

    current_python = Path(sys.executable).resolve()
    if not args.skip_bootstrap and current_python != venv_python.resolve():
        forwarded = [item for item in sys.argv[1:] if item != "--skip-bootstrap"]
        forwarded.append("--skip-bootstrap")
        os.execv(str(venv_python), [str(venv_python), str(Path(__file__).resolve()), *forwarded])


def apply_setup_env_behavior(project_root: Path, env_file: Path = DEFAULT_ENV_FILE) -> None:
    uname_a = subprocess.run(["uname", "-a"], capture_output=True, text=True, check=False).stdout
    if "Linux raspberrypi" in uname_a:
        current_kernel = (
            subprocess.run(["uname", "-r"], capture_output=True, text=True, check=False)
            .stdout.strip()
            .split("+")[0]
        )
        if current_kernel in INVALID_KERNELS:
            raise RuntimeError(
                "Kernel version "
                f"{current_kernel} is known to be incompatible. "
                "See: https://community.hailo.ai/t/raspberry-pi-kernel-compatibility-issue-temporary-fix/15322"
            )

    project_root_str = str(project_root)
    current_pythonpath = os.environ.get("PYTHONPATH", "")
    pythonpath_entries = [entry for entry in current_pythonpath.split(":") if entry]
    if project_root_str not in pythonpath_entries:
        os.environ["PYTHONPATH"] = (
            f"{project_root_str}:{current_pythonpath}" if current_pythonpath else project_root_str
        )
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

    if env_file.exists():
        for raw_line in env_file.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if not key:
                continue
            os.environ[key] = value
            os.environ[key.upper()] = value
    else:
        print(f"[warn] env file not found: {env_file}", file=sys.stderr)


def build_model_options() -> tuple[str, List[ModelOption]]:
    from hailo_apps.python.core.common.installation_utils import detect_hailo_arch

    arch = os.environ.get("HAILO_ARCH") or detect_hailo_arch() or "hailo8"

    try:
        from hailo_apps.config.config_manager import get_model_names
    except Exception:
        get_model_names = None

    options: List[ModelOption] = []
    for spec in APP_SPECS:
        model_names: List[str] = []
        if get_model_names is not None:
            try:
                model_names = get_model_names(spec.app_name, arch, tier="all") or []
            except Exception:
                model_names = []
        if not model_names:
            model_names = ["default"]
        for model_name in model_names:
            options.append(
                ModelOption(
                    spec=spec,
                    model_name=model_name,
                    label=f"{spec.display_name} | {model_name}",
                )
            )
    return arch, options


def discover_csi_cameras() -> List[CameraOption]:
    cameras: List[CameraOption] = []

    try:
        from picamera2 import Picamera2

        for idx, info in enumerate(Picamera2.global_camera_info()):
            model = str(info.get("Model", "Unknown"))
            location = str(info.get("Location", "")).strip()
            sensor_id = str(info.get("Id", "")).strip()
            details_parts = [part for part in (location, sensor_id) if part]
            details = ", ".join(details_parts)
            cameras.append(CameraOption(index=idx, label=f"CSI {idx}: {model}", details=details))
    except Exception as exc:
        print(f"[warn] Picamera2 camera discovery failed: {exc}", file=sys.stderr)

    if cameras:
        return cameras

    scan = subprocess.run(
        ["rpicam-hello", "--list-cameras"], capture_output=True, text=True, check=False
    )
    if scan.returncode == 0:
        pattern = re.compile(r"^\s*(\d+)\s*:\s*(.+?)\s+\[")
        for line in scan.stdout.splitlines():
            match = pattern.match(line)
            if not match:
                continue
            index = int(match.group(1))
            label = match.group(2).strip()
            cameras.append(CameraOption(index=index, label=f"CSI {index}: {label}"))

    if not cameras:
        cameras.append(
            CameraOption(
                index=0,
                label="CSI 0: default",
                details="Camera metadata unavailable; using default index 0.",
            )
        )
    return cameras


def init_curses_theme() -> dict:
    theme = {
        "title": curses.A_BOLD,
        "hint": curses.A_DIM,
        "selected": curses.A_REVERSE,
        "normal": curses.A_NORMAL,
    }
    if not curses.has_colors():
        return theme
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_YELLOW, -1)
    curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_YELLOW)
    curses.init_pair(3, curses.COLOR_WHITE, -1)
    theme["title"] = curses.color_pair(1) | curses.A_BOLD
    theme["hint"] = curses.color_pair(1)
    theme["selected"] = curses.color_pair(2) | curses.A_BOLD
    theme["normal"] = curses.color_pair(3)
    return theme


def choose_from_menu(
    stdscr: "curses._CursesWindow",
    title: str,
    items: List[str],
    initial_index: int = 0,
) -> int:
    curses.curs_set(0)
    theme = init_curses_theme()
    index = max(0, min(initial_index, len(items) - 1))
    top = 0

    while True:
        stdscr.erase()
        height, width = stdscr.getmaxyx()
        visible_rows = max(1, height - 5)
        if index < top:
            top = index
        elif index >= top + visible_rows:
            top = index - visible_rows + 1

        stdscr.addnstr(0, 2, title, max(1, width - 4), theme["title"])
        stdscr.addnstr(
            1,
            2,
            "Up/Down (j/k), Enter select, q back/quit",
            max(1, width - 4),
            theme["hint"],
        )

        row = 3
        for item_index in range(top, min(len(items), top + visible_rows)):
            is_selected = item_index == index
            marker = "▸ " if is_selected else "  "
            attr = theme["selected"] if is_selected else theme["normal"]
            stdscr.addnstr(row, 2, f"{marker}{items[item_index]}", max(1, width - 4), attr)
            row += 1

        stdscr.refresh()
        key = stdscr.getch()
        if key in (curses.KEY_UP, ord("k")):
            index = (index - 1) % len(items)
        elif key in (curses.KEY_DOWN, ord("j")):
            index = (index + 1) % len(items)
        elif key in (10, 13, curses.KEY_ENTER):
            return index
        elif key in (ord("q"), 27):
            return -1


def open_pipeline_menu(
    stdscr: "curses._CursesWindow",
    config: dict,
    model_options: List[ModelOption],
) -> None:
    catalog = build_model_catalog(model_options)
    families = list(catalog.keys())
    while True:
        mode_label = "Direct Camera (No Model)" if config["run_mode"] == "direct_camera" else "Model Pipeline"
        items = [
            f"Mode: {mode_label}",
            f"Model Family: {config['model_family']}",
            f"Model Variant: {config['model_name']}",
            "Back",
        ]
        choice = choose_from_menu(stdscr, "Pipeline Configuration", items)
        if choice in (-1, 3):
            return
        if choice == 0:
            mode_choice = choose_from_menu(
                stdscr,
                "Run Mode",
                ["Model Pipeline", "Direct Camera (No Model)"],
                0 if config["run_mode"] == "model" else 1,
            )
            if mode_choice == 0:
                config["run_mode"] = "model"
            elif mode_choice == 1:
                config["run_mode"] = "direct_camera"
        elif choice == 1:
            family_idx = choose_from_menu(
                stdscr,
                "Model Family",
                families,
                families.index(config["model_family"]) if config["model_family"] in families else 0,
            )
            if family_idx >= 0:
                config["model_family"] = families[family_idx]
                config["model_name"] = catalog[config["model_family"]][0].model_name
        elif choice == 2:
            family_models = catalog[config["model_family"]]
            labels = [opt.model_name for opt in family_models]
            current_idx = 0
            for idx, option in enumerate(family_models):
                if option.model_name == config["model_name"]:
                    current_idx = idx
                    break
            model_idx = choose_from_menu(stdscr, "Model Variant", labels, current_idx)
            if model_idx >= 0:
                config["model_name"] = family_models[model_idx].model_name


def open_camera_menu(
    stdscr: "curses._CursesWindow",
    config: dict,
    camera_options: List[CameraOption],
) -> None:
    while True:
        camera = resolve_camera_option(config, camera_options)
        orientation = get_orientation_for_camera(config, camera.index)
        items = [
            f"Camera: {camera.label}",
            f"Orientation: {orientation}°",
            "Back",
        ]
        choice = choose_from_menu(stdscr, "Camera Configuration", items)
        if choice in (-1, 2):
            return
        if choice == 0:
            labels = [
                f"{cam.label} ({cam.details})" if cam.details else cam.label for cam in camera_options
            ]
            initial = 0
            for idx, cam in enumerate(camera_options):
                if cam.index == camera.index:
                    initial = idx
                    break
            camera_idx = choose_from_menu(stdscr, "Select Camera", labels, initial)
            if camera_idx >= 0:
                selected_cam = camera_options[camera_idx]
                config["camera_index"] = selected_cam.index
                config["camera_orientations"].setdefault(str(selected_cam.index), 0)
        elif choice == 1:
            current_angle = get_orientation_for_camera(config, config["camera_index"])
            angle_labels = [f"{angle}°" for angle in VALID_ORIENTATIONS]
            start_idx = VALID_ORIENTATIONS.index(current_angle) if current_angle in VALID_ORIENTATIONS else 0
            angle_idx = choose_from_menu(stdscr, "Camera Orientation", angle_labels, start_idx)
            if angle_idx >= 0:
                set_orientation_for_camera(
                    config,
                    config["camera_index"],
                    VALID_ORIENTATIONS[angle_idx],
                )


def open_runtime_menu(stdscr: "curses._CursesWindow", config: dict) -> None:
    host_options = ["0.0.0.0", "127.0.0.1"]
    port_options = [5000, 5050, 8080, 9000]
    resolution_options = [(640, 480), (1280, 720), (1920, 1080)]
    fps_options = [15, 24, 30, 60]

    while True:
        items = [
            f"Host: {config['host']}",
            f"Port: {config['port']}",
            f"Resolution: {config['width']}x{config['height']}",
            f"FPS: {config['fps']}",
            "Back",
        ]
        choice = choose_from_menu(stdscr, "Runtime Configuration", items)
        if choice in (-1, 4):
            return
        if choice == 0:
            current = host_options.index(config["host"]) if config["host"] in host_options else 0
            selected = choose_from_menu(stdscr, "Host", host_options, current)
            if selected >= 0:
                config["host"] = host_options[selected]
        elif choice == 1:
            labels = [str(port) for port in port_options]
            current = port_options.index(config["port"]) if config["port"] in port_options else 0
            selected = choose_from_menu(stdscr, "Port", labels, current)
            if selected >= 0:
                config["port"] = port_options[selected]
        elif choice == 2:
            labels = [f"{width}x{height}" for width, height in resolution_options]
            current = 0
            for idx, (width, height) in enumerate(resolution_options):
                if width == config["width"] and height == config["height"]:
                    current = idx
                    break
            selected = choose_from_menu(stdscr, "Resolution", labels, current)
            if selected >= 0:
                config["width"], config["height"] = resolution_options[selected]
        elif choice == 3:
            labels = [str(fps) for fps in fps_options]
            current = fps_options.index(config["fps"]) if config["fps"] in fps_options else 0
            selected = choose_from_menu(stdscr, "FPS", labels, current)
            if selected >= 0:
                config["fps"] = fps_options[selected]


def choose_selection_with_curses(
    model_options: List[ModelOption],
    camera_options: List[CameraOption],
    initial_config: dict,
    config_path: Path,
) -> tuple[LaunchSelection, dict, List[CameraOption]]:
    result: dict = {}

    def _runner(stdscr: "curses._CursesWindow") -> None:
        config = dict(initial_config)
        config["camera_orientations"] = dict(initial_config["camera_orientations"])
        config["camera_enabled"] = dict(initial_config.get("camera_enabled", {}))
        config["camera_profiles"] = dict(initial_config.get("camera_profiles", {}))
        config["security"] = dict(initial_config.get("security", {}))
        config["tunnel"] = dict(initial_config.get("tunnel", {}))
        dynamic_cameras = list(camera_options)
        menu_index = 0

        while True:
            camera = resolve_camera_option(config, dynamic_cameras)
            orientation = get_orientation_for_camera(config, camera.index)
            if config["run_mode"] == "direct_camera":
                model_label = "Direct Camera (No Model)"
            else:
                model_label = f"{config['model_family']} | {config['model_name']}"

            items = [
                "Start Stream",
                f"Pipeline: {model_label}",
                f"Camera: {camera.label}",
                f"Orientation: {orientation}°",
                f"Runtime: {config['host']}:{config['port']} | {config['width']}x{config['height']}@{config['fps']}",
                "Save Config",
                "Reload Config",
                "Refresh Cameras",
                "Quit",
            ]
            selected = choose_from_menu(stdscr, "Hailo CSI Dashboard (Config)", items, menu_index)
            menu_index = max(0, selected)

            if selected == -1 or selected == 8:
                raise KeyboardInterrupt("Selection cancelled.")
            if selected == 0:
                config["camera_index"] = camera.index
                result["config"] = config
                result["cameras"] = dynamic_cameras
                return
            if selected == 1:
                open_pipeline_menu(stdscr, config, model_options)
            elif selected == 2:
                open_camera_menu(stdscr, config, dynamic_cameras)
            elif selected == 3:
                open_camera_menu(stdscr, config, dynamic_cameras)
            elif selected == 4:
                open_runtime_menu(stdscr, config)
            elif selected == 5:
                save_config(config_path, config)
            elif selected == 6:
                reloaded = normalize_config(load_raw_config(config_path), model_options, dynamic_cameras)
                config.clear()
                config.update(reloaded)
            elif selected == 7:
                refreshed = discover_csi_cameras()
                dynamic_cameras = refreshed
                config["camera_orientations"] = normalize_orientation_map(
                    config.get("camera_orientations", {}),
                    dynamic_cameras,
                )
                config["camera_enabled"] = normalize_camera_enabled_map(
                    config.get("camera_enabled", {}),
                    dynamic_cameras,
                )
                config["camera_profiles"] = normalize_camera_profiles(
                    config.get("camera_profiles", {}),
                    dynamic_cameras,
                    _coerce_int(config.get("width"), 1280),
                    _coerce_int(config.get("height"), 720),
                    _coerce_int(config.get("fps"), 30),
                )
                config["camera_index"] = resolve_camera_option(config, dynamic_cameras).index

    curses.wrapper(_runner)

    resolved_config = normalize_config(result["config"], model_options, result["cameras"])
    selection = resolve_launch_selection(resolved_config, model_options, result["cameras"])
    return selection, resolved_config, result["cameras"]


def resolve_launch_selection(
    config: dict,
    model_options: List[ModelOption],
    camera_options: List[CameraOption],
) -> LaunchSelection:
    camera = resolve_camera_option(config, camera_options)
    model_choice = select_model_option(config, model_options)
    orientation = get_orientation_for_camera(config, camera.index)
    return LaunchSelection(
        run_mode=config["run_mode"],
        model=model_choice,
        camera=camera,
        orientation=orientation,
    )


def merge_config_patch(base_config: dict, patch: dict) -> dict:
    merged = dict(base_config)
    merged["camera_orientations"] = dict(base_config.get("camera_orientations", {}))
    merged["camera_enabled"] = dict(base_config.get("camera_enabled", {}))
    merged["camera_profiles"] = dict(base_config.get("camera_profiles", {}))
    merged["security"] = dict(base_config.get("security", {}))
    merged["tunnel"] = dict(base_config.get("tunnel", {}))

    run_mode = patch.get("run_mode")
    if run_mode in ("model", "direct_camera"):
        merged["run_mode"] = run_mode
    model_family = patch.get("model_family")
    if model_family is not None:
        merged["model_family"] = str(model_family)
    model_name = patch.get("model_name")
    if model_name is not None:
        merged["model_name"] = str(model_name)

    if "camera_index" in patch:
        merged["camera_index"] = _coerce_int(patch.get("camera_index"), merged["camera_index"])
    camera_index = _coerce_int(merged.get("camera_index"), 0)
    camera_key = str(camera_index)
    if "orientation" in patch:
        set_orientation_for_camera(
            merged,
            camera_index,
            _coerce_int(patch.get("orientation"), get_orientation_for_camera(merged, camera_index)),
        )
    if "enabled" in patch:
        merged["camera_enabled"][camera_key] = _coerce_bool(patch.get("enabled"), True)

    incoming_profile = None
    if "profile" in patch and isinstance(patch.get("profile"), dict):
        incoming_profile = patch.get("profile")
    elif any(key in patch for key in ("pixel_format", "width", "height", "fps")):
        incoming_profile = {
            "pixel_format": patch.get("pixel_format", ""),
            "width": patch.get("width"),
            "height": patch.get("height"),
            "fps": patch.get("fps"),
        }
    if incoming_profile is not None:
        merged["camera_profiles"][camera_key] = _normalize_profile(
            incoming_profile,
            _coerce_int(merged.get("width"), 1280),
            _coerce_int(merged.get("height"), 720),
            _coerce_int(merged.get("fps"), 30),
        )

    if "host" in patch:
        merged["host"] = str(patch.get("host"))
    if "port" in patch:
        merged["port"] = _coerce_int(patch.get("port"), merged["port"])
    if "width" in patch:
        merged["width"] = _coerce_int(patch.get("width"), merged["width"])
    if "height" in patch:
        merged["height"] = _coerce_int(patch.get("height"), merged["height"])
    if "fps" in patch:
        merged["fps"] = _coerce_int(patch.get("fps"), merged["fps"])
    if any(key in patch for key in ("width", "height", "fps")) and camera_key:
        existing = dict(merged["camera_profiles"].get(camera_key, {}))
        existing["width"] = max(320, _coerce_int(merged.get("width"), 1280))
        existing["height"] = max(240, _coerce_int(merged.get("height"), 720))
        existing["fps"] = max(1.0, _coerce_float(merged.get("fps"), 30.0))
        existing["pixel_format"] = str(existing.get("pixel_format") or "RGB").strip().upper() or "RGB"
        merged["camera_profiles"][camera_key] = existing

    if "security" in patch and isinstance(patch.get("security"), dict):
        merged["security"].update(dict(patch.get("security", {})))
    if "tunnel" in patch and isinstance(patch.get("tunnel"), dict):
        merged["tunnel"].update(dict(patch.get("tunnel", {})))
    return merged


def build_app_runtime(
    config: dict,
    arch: str,
    model_options: List[ModelOption],
    camera_options: List[CameraOption],
):
    selection = resolve_launch_selection(config, model_options, camera_options)
    if selection.run_mode == "direct_camera":
        return selection, None, []

    if selection.model is None:
        raise RuntimeError("Model mode selected but no model option is available.")

    app_class = load_app_class(selection.model.spec)
    gstreamer_app_module.GST_VIDEO_SINK = "fakesink"
    patch_picamera_thread(selection.camera.index, selection.orientation)
    app_args = [
        "--input",
        "rpi",
        "--disable-sync",
        "--frame-rate",
        str(config["fps"]),
        "--width",
        str(config["width"]),
        "--height",
        str(config["height"]),
        "--arch",
        arch,
    ]
    if selection.model.model_name != "default":
        app_args.extend(["--hef-path", selection.model.model_name])
    return selection, app_class, app_args


class RuntimeController:
    def __init__(
        self,
        arch: str,
        model_options: List[ModelOption],
        camera_options: List[CameraOption],
        config_path: Path,
        initial_config: dict,
    ) -> None:
        self.arch = arch
        self.model_options = model_options
        self.camera_options = list(camera_options)
        self.config_path = config_path
        self.model_catalog = build_model_catalog(model_options)
        self.config = normalize_config(initial_config, model_options, self.camera_options)
        self._lock = threading.RLock()
        self.session: Optional[PipelineSession] = None
        self.requires_server_restart = False

    def _copy_config(self) -> dict:
        config = dict(self.config)
        config["camera_orientations"] = dict(self.config.get("camera_orientations", {}))
        config["camera_enabled"] = dict(self.config.get("camera_enabled", {}))
        config["camera_profiles"] = dict(self.config.get("camera_profiles", {}))
        config["security"] = dict(self.config.get("security", {}))
        config["tunnel"] = dict(self.config.get("tunnel", {}))
        return config

    def _camera_payload(self) -> List[dict]:
        payload: List[dict] = []
        for camera in self.camera_options:
            payload.append(
                {
                    "id": _camera_id_for_index(camera.index),
                    "index": camera.index,
                    "label": camera.label,
                    "details": camera.details,
                    "enabled": is_camera_enabled(self.config, camera.index),
                }
            )
        return payload

    def _catalog_payload(self) -> dict:
        return {
            family: [item.model_name for item in options]
            for family, options in self.model_catalog.items()
        }

    def _camera_by_index(self, camera_index: int) -> Optional[CameraOption]:
        for camera in self.camera_options:
            if camera.index == int(camera_index):
                return camera
        return None

    def camera_option_for_id(self, camera_id: str) -> Optional[CameraOption]:
        idx = _camera_index_from_id(camera_id)
        return self._camera_by_index(idx)

    def _sync_selected_profile_locked(self) -> None:
        camera_index = _coerce_int(self.config.get("camera_index"), 0)
        profile = get_profile_for_camera(self.config, camera_index)
        self.config["width"] = int(profile.get("width", self.config.get("width", 1280)))
        self.config["height"] = int(profile.get("height", self.config.get("height", 720)))
        self.config["fps"] = int(max(1.0, _coerce_float(profile.get("fps"), self.config.get("fps", 30))))

    def security_settings(self) -> dict:
        with self._lock:
            return _normalized_security(self.config.get("security", {}))

    def protocol_capabilities(self) -> dict:
        return {
            "jpeg_snapshot": True,
            "mjpeg": True,
            "webrtc": False,
            "mpegts": False,
            "webrtc_error": "WebRTC not enabled in hailo camera dashboard",
            "mpegts_error": "MPEGTS not enabled in hailo camera dashboard",
        }

    @staticmethod
    def camera_mode_urls(camera_id: str) -> dict:
        camera_id = str(camera_id or "").strip()
        return {
            "jpeg": f"/jpeg/{camera_id}",
            "mjpeg": f"/mjpeg/{camera_id}",
            "stream": f"/video/{camera_id}",
            "snapshot": f"/camera/{camera_id}",
        }

    def camera_statuses(self) -> List[dict]:
        with self._lock:
            active_session = self.session
            config = self._copy_config()
        protocols = self.protocol_capabilities()
        active_index = _coerce_int(config.get("camera_index"), -1)
        active_camera_id = _camera_id_for_index(active_index)
        session_status = active_session.status_payload() if active_session else {}
        selected_online = bool(active_session and not session_status.get("error"))
        selected_clients = int(session_status.get("clients", 0) or 0)
        selected_frames = int(session_status.get("frame_count", 0) or 0)
        selected_fps = float(session_status.get("avg_fps", 0.0) or 0.0)
        selected_error = str(session_status.get("error") or "").strip()

        rows: List[dict] = []
        for camera in self.camera_options:
            camera_id = _camera_id_for_index(camera.index)
            profile = get_profile_for_camera(config, camera.index)
            enabled = is_camera_enabled(config, camera.index)
            is_selected = camera_id == active_camera_id
            online = bool(enabled and is_selected and selected_online)
            last_error = selected_error if is_selected else ("Camera disabled by policy" if not enabled else "")
            rows.append(
                {
                    "id": camera_id,
                    "label": camera.label,
                    "source_type": "default",
                    "device_path": f"/dev/video{camera.index}",
                    "index": int(camera.index),
                    "online": bool(online),
                    "has_frame": bool(online and selected_frames > 0),
                    "frame_size": {
                        "width": int(profile.get("width", 0)),
                        "height": int(profile.get("height", 0)),
                    },
                    "width": int(profile.get("width", 0)),
                    "height": int(profile.get("height", 0)),
                    "fps": round(selected_fps if is_selected else 0.0, 2),
                    "kbps": 0.0,
                    "clients": int(selected_clients if is_selected else 0),
                    "total_frames": int(selected_frames if is_selected else 0),
                    "backend": "hailo" if config.get("run_mode") != "direct_camera" else "direct_camera",
                    "last_frame_age_seconds": 0.0 if online else -1.0,
                    "last_error": last_error,
                    "capture_profile": profile,
                    "active_capture": {
                        "backend": "hailo" if config.get("run_mode") != "direct_camera" else "direct_camera",
                        "pixel_format": str(profile.get("pixel_format") or "RGB"),
                        "width": int(profile.get("width", 0)),
                        "height": int(profile.get("height", 0)),
                        "fps": float(profile.get("fps", 0.0)),
                    },
                    "available_profiles": [dict(item) for item in STREAM_PROFILE_OPTIONS],
                    "profile_query_error": "",
                    "rotation_degrees": int(get_orientation_for_camera(config, camera.index)),
                    "enabled": bool(enabled),
                    "configured_enabled": bool(enabled),
                    "configured_enabled_key": camera_id,
                    "modes": self.camera_mode_urls(camera_id),
                    "protocols": protocols,
                }
            )
        return rows

    def ensure_camera_selected(self, camera_id: str, save: bool = False) -> bool:
        camera = self.camera_option_for_id(camera_id)
        if camera is None:
            return False
        with self._lock:
            current_index = _coerce_int(self.config.get("camera_index"), -1)
            if current_index == camera.index:
                return True
            self.config["camera_index"] = int(camera.index)
            self._sync_selected_profile_locked()
            if save:
                save_config(self.config_path, self.config)
            self._restart_session_locked()
        return True

    def set_camera_enabled(self, camera_id: str, enabled: bool, save: bool = True) -> dict:
        camera = self.camera_option_for_id(camera_id)
        if camera is None:
            raise ValueError("Camera not found")
        with self._lock:
            key = str(camera.index)
            self.config.setdefault("camera_enabled", {})
            self.config["camera_enabled"][key] = bool(enabled)
            if not enabled and _coerce_int(self.config.get("camera_index"), -1) == camera.index:
                fallback = resolve_camera_option(self.config, self.camera_options)
                self.config["camera_index"] = int(fallback.index)
                self._sync_selected_profile_locked()
            if save:
                save_config(self.config_path, self.config)
            self._restart_session_locked()
        return self.snapshot()

    def set_camera_profile(self, camera_id: str, profile: dict, save: bool = True) -> tuple[dict, bool]:
        camera = self.camera_option_for_id(camera_id)
        if camera is None:
            raise ValueError("Camera not found")
        with self._lock:
            key = str(camera.index)
            previous = get_profile_for_camera(self.config, camera.index)
            next_profile = _normalize_profile(
                profile,
                _coerce_int(previous.get("width"), 1280),
                _coerce_int(previous.get("height"), 720),
                int(max(1.0, _coerce_float(previous.get("fps"), 30.0))),
            )
            self.config.setdefault("camera_profiles", {})
            self.config["camera_profiles"][key] = next_profile
            changed = next_profile != previous
            if _coerce_int(self.config.get("camera_index"), -1) == camera.index:
                self._sync_selected_profile_locked()
                if changed:
                    self._restart_session_locked()
            if save:
                save_config(self.config_path, self.config)
        return dict(next_profile), bool(changed)

    def snapshot(self) -> dict:
        with self._lock:
            active_session = self.session
            config = self._copy_config()
            requires_restart = self.requires_server_restart
        status = active_session.status_payload() if active_session else {
            "model_label": "not running",
            "camera_label": "",
            "frame_count": 0,
            "avg_fps": 0.0,
            "detections": 0,
            "error": "No active session",
        }
        return {
            "session": status,
            "config": config,
            "model_catalog": self._catalog_payload(),
            "cameras": self._camera_payload(),
            "requires_server_restart": requires_restart,
        }

    def _restart_session_locked(self) -> None:
        selected_camera = resolve_camera_option(self.config, self.camera_options)
        self.config["camera_index"] = int(selected_camera.index)
        self._sync_selected_profile_locked()
        selection, app_class, app_args = build_app_runtime(
            self.config, self.arch, self.model_options, self.camera_options
        )
        new_session = PipelineSession(
            selection=selection,
            app_class=app_class,
            app_args=app_args,
            width=self.config["width"],
            height=self.config["height"],
            fps=self.config["fps"],
        )
        old_session = self.session
        self.session = new_session
        if old_session is not None:
            old_session.stop()
            time.sleep(0.2)
        new_session.start()

    def start(self) -> None:
        with self._lock:
            self._restart_session_locked()

    def stop(self) -> None:
        with self._lock:
            session = self.session
            self.session = None
        if session is not None:
            session.stop()

    def current_session(self) -> Optional[PipelineSession]:
        with self._lock:
            return self.session

    def update_config(
        self,
        patch: dict,
        save: bool,
        apply_now: bool,
        refresh_cameras: bool = False,
    ) -> dict:
        with self._lock:
            previous_signature = runtime_config_signature(self.config)
            candidate = merge_config_patch(self._copy_config(), patch)
            if refresh_cameras:
                self.camera_options = discover_csi_cameras()
            normalized = normalize_config(candidate, self.model_options, self.camera_options)
            normalized_signature = runtime_config_signature(normalized)

            current_host = self.config.get("host")
            current_port = self.config.get("port")
            host_or_port_changed = (
                normalized.get("host") != current_host or normalized.get("port") != current_port
            )
            self.requires_server_restart = host_or_port_changed

            self.config = normalized
            if save:
                save_config(self.config_path, self.config)
            should_restart = (
                self.session is None
                or refresh_cameras
                or normalized_signature != previous_signature
            )
            if apply_now and should_restart:
                self._restart_session_locked()

        return self.snapshot()

    def reload_from_disk(self, apply_now: bool = True) -> dict:
        with self._lock:
            previous_signature = runtime_config_signature(self.config)
            loaded = normalize_config(
                load_raw_config(self.config_path),
                self.model_options,
                self.camera_options,
            )
            self.config = loaded
            should_restart = self.session is None or runtime_config_signature(loaded) != previous_signature
            if apply_now and should_restart:
                self._restart_session_locked()
        return self.snapshot()


def ensure_port_available(host: str, port: int) -> None:
    bind_host = host if host not in ("0.0.0.0", "::") else ""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((bind_host, port))
        except OSError as exc:
            raise RuntimeError(
                f"Port {port} is already in use. Stop the existing service or choose --port."
            ) from exc


def load_app_class(spec: AppSpec):
    module = importlib.import_module(spec.module_path)
    return getattr(module, spec.class_name)


class PipelineSession:
    def __init__(
        self,
        selection: LaunchSelection,
        app_class,
        app_args: Optional[List[str]],
        width: int,
        height: int,
        fps: int,
    ) -> None:
        self.selection = selection
        self.app_class = app_class
        self.app_args = app_args or []
        self.width = width
        self.height = height
        self.fps = max(1, fps)

        self._lock = threading.Lock()
        self._latest_jpeg: bytes | None = None
        self._last_detection_count = 0
        self._app_instance = None
        self._user_data = None
        self._stop_event = threading.Event()
        self._pipeline_thread: threading.Thread | None = None
        self._frame_thread: threading.Thread | None = None
        self._camera_thread: threading.Thread | None = None
        self._picam2 = None
        self._started_at = time.time()
        self._camera_frame_count = 0
        self._stream_clients = 0
        self.error: str | None = None

    @property
    def model_label(self) -> str:
        if self.selection.model is None:
            return "Direct Camera (No Model)"
        return self.selection.model.label

    @property
    def camera_label(self) -> str:
        return f"{self.selection.camera.label} ({self.selection.orientation}°)"

    def _placeholder_jpeg(self, line: str) -> bytes:
        image = np.full((self.height, self.width, 3), (40, 44, 50), dtype=np.uint8)
        cv2.putText(image, "Hailo Dashboard", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 174, 255), 2)
        cv2.putText(image, line, (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2)
        ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        return encoded.tobytes() if ok else b""

    def latest_jpeg(self) -> bytes:
        with self._lock:
            if self._latest_jpeg is not None:
                return self._latest_jpeg
        return self._placeholder_jpeg("Waiting for frames...")

    def frame_count(self) -> int:
        if self.selection.run_mode == "direct_camera":
            return self._camera_frame_count
        if self._user_data is None:
            return 0
        return self._user_data.get_count()

    def avg_fps(self) -> float:
        elapsed = max(time.time() - self._started_at, 0.001)
        return float(self.frame_count()) / elapsed

    def acquire_client(self) -> None:
        with self._lock:
            self._stream_clients += 1

    def release_client(self) -> None:
        with self._lock:
            self._stream_clients = max(0, self._stream_clients - 1)

    def client_count(self) -> int:
        with self._lock:
            return int(self._stream_clients)

    def _pipeline_callback(self, element, buffer, user_data) -> None:
        if buffer is None:
            return
        pad = element.get_static_pad("src")
        frame_format, width, height = get_caps_from_pad(pad)
        if not user_data.use_frame or frame_format is None or width is None or height is None:
            return

        frame = get_numpy_from_buffer(buffer, frame_format, width, height)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        detections_count = 0
        try:
            roi = hailo.get_roi_from_buffer(buffer)
            detections_count = len(roi.get_objects_typed(hailo.HAILO_DETECTION))
        except Exception:
            detections_count = 0

        model_text = f"{self.selection.model.spec.display_name} [{self.selection.model.model_name}]"
        cv2.putText(frame, model_text, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 174, 255), 2)
        cv2.putText(
            frame,
            f"camera={self.selection.camera.index} orientation={self.selection.orientation} detections={detections_count}",
            (16, 66),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (230, 230, 230),
            2,
        )
        self._last_detection_count = detections_count
        user_data.set_frame(frame)

    def _frame_encoder_loop(self) -> None:
        while not self._stop_event.is_set():
            if self._user_data is None:
                time.sleep(0.02)
                continue
            frame = self._user_data.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if ok:
                with self._lock:
                    self._latest_jpeg = encoded.tobytes()

    def _close_picamera_handle(self, handle=None) -> None:
        picam = handle
        if picam is None:
            with CAMERA_LIFECYCLE_LOCK:
                picam = self._picam2
                self._picam2 = None
        if picam is None:
            return
        try:
            picam.stop()
        except Exception:
            pass
        try:
            picam.close()
        except Exception:
            pass

    def _camera_only_loop(self) -> None:
        from picamera2 import Picamera2
        attempt = 0
        busy_markers = (
            "running state trying acquire",
            "requiring state available",
            "camera __init__ sequence did not complete",
            "device or resource busy",
        )

        while not self._stop_event.is_set():
            attempt += 1
            picam2 = None
            try:
                with CAMERA_LIFECYCLE_LOCK:
                    if self._stop_event.is_set():
                        break
                    picam2 = Picamera2(self.selection.camera.index)
                    self._picam2 = picam2
                    config = picam2.create_preview_configuration(
                        main={"size": (self.width, self.height), "format": "RGB888"},
                        controls={"FrameRate": self.fps},
                    )
                    picam2.configure(config)
                    picam2.start()

                while not self._stop_event.is_set():
                    frame = picam2.capture_array("main")
                    if frame is None:
                        time.sleep(0.01)
                        continue
                    frame = apply_orientation(frame, self.selection.orientation)
                    if frame.shape[1] != self.width or frame.shape[0] != self.height:
                        frame = cv2.resize(frame, (self.width, self.height))

                    cv2.putText(
                        frame,
                        "Direct Camera Mode",
                        (16, 34),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 174, 255),
                        2,
                    )
                    cv2.putText(
                        frame,
                        f"camera={self.selection.camera.index} orientation={self.selection.orientation}",
                        (16, 66),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (230, 230, 230),
                        2,
                    )

                    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
                    if ok:
                        with self._lock:
                            self._latest_jpeg = encoded.tobytes()
                        self._camera_frame_count += 1
                    time.sleep(max(0.0, (1.0 / self.fps) * 0.2))
                break
            except Exception as exc:
                self.error = traceback.format_exc(limit=4)
                text = str(exc or "").strip().lower()
                retriable = any(marker in text for marker in busy_markers)
                if retriable and attempt < 7 and not self._stop_event.is_set():
                    time.sleep(min(1.2, 0.25 * attempt))
                    continue
                break
            finally:
                with CAMERA_LIFECYCLE_LOCK:
                    if self._picam2 is picam2:
                        self._picam2 = None
                self._close_picamera_handle(picam2)

        self._stop_event.set()

    def _pipeline_loop(self) -> None:
        previous_argv = sys.argv[:]
        runtime = self

        class DashboardCallbackData(app_callback_class):
            def __init__(self) -> None:
                super().__init__()
                self.runtime = runtime

        self._user_data = DashboardCallbackData()
        self._user_data.use_frame = True

        try:
            sys.argv = [previous_argv[0], *self.app_args]

            # GStreamerApp registers SIGINT in __init__; avoid that from worker threads.
            if threading.current_thread() is threading.main_thread():
                self._app_instance = self.app_class(self._pipeline_callback, self._user_data)
            else:
                original_signal = signal.signal

                def _thread_safe_signal(sig, handler):
                    if sig == signal.SIGINT:
                        return signal.getsignal(sig)
                    return original_signal(sig, handler)

                signal.signal = _thread_safe_signal
                try:
                    self._app_instance = self.app_class(self._pipeline_callback, self._user_data)
                finally:
                    signal.signal = original_signal

            # Keep callback frame access enabled without triggering gstreamer_app's
            # multiprocessing display process (which can leave stale child processes).
            self._user_data.use_frame = True
            self._app_instance.run()
        except SystemExit:
            pass
        except Exception:
            self.error = traceback.format_exc(limit=4)
        finally:
            sys.argv = previous_argv
            self._stop_event.set()

    def start(self) -> None:
        with self._lock:
            self._latest_jpeg = self._placeholder_jpeg("Starting pipeline...")
        if self.selection.run_mode == "direct_camera":
            self._camera_thread = threading.Thread(target=self._camera_only_loop, daemon=True)
            self._camera_thread.start()
            return
        self._frame_thread = threading.Thread(target=self._frame_encoder_loop, daemon=True)
        self._pipeline_thread = threading.Thread(target=self._pipeline_loop, daemon=True)
        self._frame_thread.start()
        self._pipeline_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._close_picamera_handle()
        if self._app_instance is not None:
            try:
                self._app_instance.shutdown()
            except Exception:
                pass
        if self._pipeline_thread and self._pipeline_thread.is_alive():
            self._pipeline_thread.join(timeout=2.0)
        if self._frame_thread and self._frame_thread.is_alive():
            self._frame_thread.join(timeout=1.0)
        if self._camera_thread and self._camera_thread.is_alive():
            self._camera_thread.join(timeout=5.0)

    def status_payload(self) -> dict:
        return {
            "model_label": self.model_label,
            "camera_label": self.camera_label,
            "frame_count": self.frame_count(),
            "avg_fps": self.avg_fps(),
            "detections": self._last_detection_count,
            "clients": self.client_count(),
            "error": self.error,
        }


def patch_picamera_thread(camera_index: int, orientation: int) -> None:
    import hailo_apps.python.core.gstreamer.gstreamer_app as gstreamer_app_module
    from picamera2 import Picamera2

    def selected_picamera_thread(
        pipeline,
        video_width: int,
        video_height: int,
        video_format: str,
        picamera_config=None,
    ):
        appsrc = pipeline.get_by_name("app_source")
        appsrc.set_property("is-live", True)
        appsrc.set_property("format", Gst.Format.TIME)

        with Picamera2(camera_index) as picam2:
            if picamera_config is None:
                main_stream = {"size": (1280, 720), "format": "RGB888"}
                lores_stream = {"size": (video_width, video_height), "format": "RGB888"}
                controls = {"FrameRate": 30}
                config = picam2.create_preview_configuration(
                    main=main_stream,
                    lores=lores_stream,
                    controls=controls,
                )
            else:
                config = picamera_config

            picam2.configure(config)
            lores = config["lores"]
            format_str = "RGB" if lores["format"] == "RGB888" else video_format
            width, height = lores["size"]
            appsrc.set_property(
                "caps",
                Gst.Caps.from_string(
                    f"video/x-raw, format={format_str}, width={width}, height={height}, "
                    "framerate=30/1, pixel-aspect-ratio=1/1"
                ),
            )
            picam2.start()
            frame_count = 0

            while True:
                frame_data = picam2.capture_array("lores")
                if frame_data is None:
                    break
                frame_data = apply_orientation(frame_data, orientation)
                if frame_data.shape[1] != width or frame_data.shape[0] != height:
                    frame_data = cv2.resize(frame_data, (width, height))
                frame = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
                gst_buffer = Gst.Buffer.new_wrapped(frame.tobytes())
                duration = Gst.util_uint64_scale_int(1, Gst.SECOND, 30)
                gst_buffer.pts = frame_count * duration
                gst_buffer.duration = duration
                ret = appsrc.emit("push-buffer", gst_buffer)
                if ret == Gst.FlowReturn.FLUSHING:
                    break
                if ret != Gst.FlowReturn.OK:
                    break
                frame_count += 1

    gstreamer_app_module.picamera_thread = selected_picamera_thread


def create_flask_app(controller: RuntimeController, config_path: Path):
    app = Flask(__name__)
    if CORS is not None:
        CORS(app, resources={r"/*": {"origins": "*"}})

    startup_time = time.time()
    request_count = {"value": 0}
    sessions: Dict[str, float] = {}
    sessions_lock = threading.Lock()
    profile_revisions: Dict[str, int] = {}

    def _security_settings() -> dict:
        return controller.security_settings()

    def _session_timeout() -> int:
        return max(30, int(_security_settings().get("session_timeout", DEFAULT_SESSION_TIMEOUT)))

    def _security_payload(base_url: str) -> dict:
        return {
            "require_auth": bool(_security_settings().get("require_auth", DEFAULT_REQUIRE_AUTH)),
            "session_timeout": int(_session_timeout()),
            "auth_url": f"{base_url}/auth" if base_url else "/auth",
            "session_rotate_url": f"{base_url}/session/rotate" if base_url else "/session/rotate",
        }

    def _cleanup_expired_sessions() -> None:
        now = time.time()
        with sessions_lock:
            stale = [key for key, expires_at in sessions.items() if now >= float(expires_at)]
            for key in stale:
                sessions.pop(key, None)

    def _create_session() -> str:
        _cleanup_expired_sessions()
        session_key = secrets.token_urlsafe(32)
        with sessions_lock:
            sessions[session_key] = time.time() + float(_session_timeout())
        return session_key

    def _rotate_sessions() -> tuple[str, int]:
        with sessions_lock:
            invalidated = len(sessions)
            sessions.clear()
        session_key = _create_session()
        return session_key, invalidated

    def _get_session_key_from_request() -> str:
        try:
            key = str(request.args.get("session_key", "")).strip()
            if key:
                return key
        except Exception:
            pass
        try:
            key = str(request.headers.get("x-session-key", "")).strip()
            if key:
                return key
        except Exception:
            pass
        try:
            auth_header = str(request.headers.get("Authorization", "")).strip()
            if auth_header.lower().startswith("bearer "):
                token = auth_header[7:].strip()
                if token:
                    return token
        except Exception:
            pass
        try:
            payload = request.get_json(silent=True) or {}
            if isinstance(payload, dict):
                key = str(payload.get("session_key", "")).strip()
                if key:
                    return key
        except Exception:
            pass
        return ""

    def _validate_session(session_key: str) -> bool:
        if not _security_settings().get("require_auth", DEFAULT_REQUIRE_AUTH):
            return True
        key = str(session_key or "").strip()
        if not key:
            return False
        _cleanup_expired_sessions()
        now = time.time()
        with sessions_lock:
            expires_at = sessions.get(key)
            if expires_at is None or now >= float(expires_at):
                sessions.pop(key, None)
                return False
            sessions[key] = now + float(_session_timeout())
        return True

    def require_session(handler):
        @wraps(handler)
        def wrapped(*args, **kwargs):
            if not _security_settings().get("require_auth", DEFAULT_REQUIRE_AUTH):
                return handler(*args, **kwargs)
            if not _validate_session(_get_session_key_from_request()):
                return jsonify({"status": "error", "message": "Invalid or expired session"}), 401
            return handler(*args, **kwargs)

        return wrapped

    def _resolve_lan_host(bind_host: str) -> str:
        host = str(bind_host or "").strip().lower()
        if host in ("0.0.0.0", "::", ""):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
                    probe.settimeout(0.5)
                    probe.connect(("8.8.8.8", 80))
                    ip = str(probe.getsockname()[0] or "").strip()
                    if ip and ip != "127.0.0.1":
                        return ip
            except Exception:
                return ""
            return ""
        if host in ("localhost", "127.0.0.1", "::1"):
            return ""
        return str(bind_host or "").strip()

    def _current_bases() -> tuple[str, str, str]:
        snapshot = controller.snapshot()
        cfg = snapshot.get("config", {}) if isinstance(snapshot, dict) else {}
        host = str(cfg.get("host") or "0.0.0.0").strip() or "0.0.0.0"
        port = _coerce_int(cfg.get("port"), 8080)
        local_base = f"http://127.0.0.1:{port}"
        lan_host = _resolve_lan_host(host)
        lan_base = f"http://{lan_host}:{port}" if lan_host else ""
        tunnel_base = str(_tunnel_payload().get("tunnel_url") or "").strip().rstrip("/")
        return local_base, lan_base, tunnel_base

    def _find_camera_row(camera_id: str) -> Optional[dict]:
        camera_key = str(camera_id or "").strip()
        for row in controller.camera_statuses():
            if str(row.get("id") or "") == camera_key:
                return row
        return None

    def _mjpeg_generator(camera_id: str):
        selected = controller.ensure_camera_selected(camera_id, save=False)
        if not selected:
            return
        target = controller.current_session()
        if target is None:
            return
        target.acquire_client()
        try:
            while True:
                frame = target.latest_jpeg()
                if frame:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        + f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii")
                        + frame
                        + b"\r\n"
                    )
                time.sleep(0.04)
        finally:
            target.release_client()

    @app.before_request
    def _count_requests():
        request_count["value"] += 1

    @app.route("/")
    @app.route("/dashboard", methods=["GET"])
    def index():
        session_key = _get_session_key_from_request()
        if _security_settings().get("require_auth", DEFAULT_REQUIRE_AUTH) and not _validate_session(session_key):
            session_key = ""
        return render_template_string(
            DASHBOARD_TEMPLATE,
            initial=controller.snapshot(),
            config_path=str(config_path),
            session_key=session_key,
            require_auth=bool(_security_settings().get("require_auth", DEFAULT_REQUIRE_AUTH)),
        )

    @app.route("/auth", methods=["POST"])
    def auth():
        data = request.get_json(silent=True) or {}
        provided = str(data.get("password", ""))
        security = _security_settings()
        if provided == str(security.get("password") or DEFAULT_PASSWORD):
            session_key = _create_session()
            return jsonify(
                {
                    "status": "success",
                    "session_key": session_key,
                    "timeout": _session_timeout(),
                    "require_auth": bool(security.get("require_auth", DEFAULT_REQUIRE_AUTH)),
                }
            )
        return jsonify({"status": "error", "message": "Invalid password"}), 401

    @app.route("/session/rotate", methods=["POST"])
    @require_session
    def rotate_session():
        session_key, invalidated = _rotate_sessions()
        return jsonify(
            {
                "status": "success",
                "message": "Session keys rotated",
                "session_key": session_key,
                "timeout": _session_timeout(),
                "invalidated_sessions": int(invalidated),
            }
        )

    @app.route("/api/status")
    @require_session
    def api_status():
        return jsonify(controller.snapshot())

    @app.route("/api/config", methods=["GET", "POST"])
    @require_session
    def api_config():
        if getattr(request, "method", "GET") == "GET":
            return jsonify(controller.snapshot())
        payload = request.get_json(silent=True) or {}
        patch = payload.get("config", {})
        save = bool(payload.get("save", True))
        apply_now = bool(payload.get("apply", True))
        snapshot = controller.update_config(
            patch=patch if isinstance(patch, dict) else {},
            save=save,
            apply_now=apply_now,
            refresh_cameras=False,
        )
        cfg = snapshot.get("config", {}) if isinstance(snapshot, dict) else {}
        _apply_tunnel_runtime_config(cfg, startup_delay=0.0)
        return jsonify(snapshot)

    @app.route("/api/config/reload", methods=["POST"])
    @require_session
    def api_config_reload():
        payload = request.get_json(silent=True) or {}
        apply_now = bool(payload.get("apply", True))
        snapshot = controller.reload_from_disk(apply_now=apply_now)
        cfg = snapshot.get("config", {}) if isinstance(snapshot, dict) else {}
        _apply_tunnel_runtime_config(cfg, startup_delay=0.0)
        return jsonify(snapshot)

    @app.route("/api/cameras/refresh", methods=["POST"])
    @require_session
    def api_refresh_cameras():
        payload = request.get_json(silent=True) or {}
        apply_now = bool(payload.get("apply", False))
        snapshot = controller.update_config(
            patch={},
            save=False,
            apply_now=apply_now,
            refresh_cameras=True,
        )
        cfg = snapshot.get("config", {}) if isinstance(snapshot, dict) else {}
        _apply_tunnel_runtime_config(cfg, startup_delay=0.0)
        return jsonify(snapshot)

    @app.route("/health", methods=["GET"])
    def health():
        cameras = controller.camera_statuses()
        online_count = sum(1 for row in cameras if bool(row.get("online")))
        enabled_count = sum(1 for row in cameras if bool(row.get("enabled", True)))
        total_clients = sum(_coerce_int(row.get("clients"), 0) for row in cameras)
        with sessions_lock:
            sessions_active = len(sessions)
        local_base, lan_base, tunnel_base = _current_bases()
        publish_base = tunnel_base or lan_base or local_base
        tunnel = _tunnel_payload()
        return jsonify(
            {
                "status": "ok",
                "service": "camera_router",
                "uptime_seconds": round(time.time() - startup_time, 2),
                "require_auth": bool(_security_settings().get("require_auth", DEFAULT_REQUIRE_AUTH)),
                "protocols": controller.protocol_capabilities(),
                "base_url": publish_base,
                "local_base_url": local_base,
                "lan_base_url": lan_base,
                "security": _security_payload(publish_base),
                "local_security": _security_payload(local_base),
                "lan_security": _security_payload(lan_base) if lan_base else {},
                "tunnel": tunnel,
                "tunnel_running": bool(tunnel.get("running") or tunnel.get("tunnel_url")),
                "tunnel_error": str(tunnel.get("error") or ""),
                "feeds_total": len(cameras),
                "feeds_online": online_count,
                "feeds_enabled": enabled_count,
                "clients": total_clients,
                "sessions_active": sessions_active,
                "requests_served": int(request_count["value"]),
                "tunnel_url": str(tunnel.get("tunnel_url") or ""),
            }
        )

    @app.route("/list", methods=["GET"])
    @require_session
    def list_cameras():
        rows = controller.camera_statuses()
        local_base, lan_base, tunnel_base = _current_bases()
        publish_base = tunnel_base or lan_base or local_base
        tunnel = _tunnel_payload()
        cameras = [
            _camera_row_with_urls(row, publish_base, local_base, lan_base, tunnel_base)
            for row in rows
        ]
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
                "cameras": cameras,
                "protocols": controller.protocol_capabilities(),
                "stream_format": "multipart/x-mixed-replace; boundary=frame",
                "session_timeout": _session_timeout(),
                "tunnel": {
                    **tunnel,
                    "dashboard_url": f"{tunnel_base}/dashboard" if tunnel_base else "",
                    "auth_url": f"{tunnel_base}/auth" if tunnel_base else "",
                    "session_rotate_url": f"{tunnel_base}/session/rotate" if tunnel_base else "",
                    "list_url": f"{tunnel_base}/list" if tunnel_base else "",
                    "health_url": f"{tunnel_base}/health" if tunnel_base else "",
                    "video_url": f"{tunnel_base}/video/cam0" if tunnel_base else "",
                    "snapshot_url": f"{tunnel_base}/snapshot/cam0" if tunnel_base else "",
                    "jpeg_url": f"{tunnel_base}/jpeg/cam0" if tunnel_base else "",
                    "mjpeg_url": f"{tunnel_base}/mjpeg/cam0" if tunnel_base else "",
                    "frame_packet_template": f"{tunnel_base}/frame_packet/<camera_id>" if tunnel_base else "",
                },
                "routes": {
                    "dashboard": "/dashboard",
                    "auth": "/auth",
                    "session_rotate": "/session/rotate",
                    "health": "/health",
                    "list": "/list",
                    "imu": "/imu",
                    "imu_stream": "/imu/stream",
                    "snapshot": "/camera/<camera_id>",
                    "jpeg": "/jpeg/<camera_id>",
                    "frame_packet": "/frame_packet/<camera_id>",
                    "stream": "/video/<camera_id>",
                    "mjpeg": "/mjpeg/<camera_id>",
                    "stream_options": "/stream_options/<camera_id>",
                    "camera_state": "/camera_state/<camera_id>",
                    "camera_recover": "/camera/recover",
                    "camera_cycle": "/camera/cycle",
                    "router_info": "/router_info",
                    "tunnel_info": "/tunnel_info",
                },
                "local": {
                    "base_url": local_base,
                    "auth_url": f"{local_base}/auth",
                    "session_rotate_url": f"{local_base}/session/rotate",
                    "list_url": f"{local_base}/list",
                    "health_url": f"{local_base}/health",
                    "dashboard_url": f"{local_base}/dashboard",
                    "video_url": f"{local_base}/video/cam0",
                    "snapshot_url": f"{local_base}/snapshot/cam0",
                    "jpeg_url": f"{local_base}/jpeg/cam0",
                    "mjpeg_url": f"{local_base}/mjpeg/cam0",
                    "frame_packet_template": f"{local_base}/frame_packet/<camera_id>",
                },
                "lan": {
                    "base_url": lan_base,
                    "auth_url": f"{lan_base}/auth" if lan_base else "",
                    "session_rotate_url": f"{lan_base}/session/rotate" if lan_base else "",
                    "list_url": f"{lan_base}/list" if lan_base else "",
                    "health_url": f"{lan_base}/health" if lan_base else "",
                    "dashboard_url": f"{lan_base}/dashboard" if lan_base else "",
                    "video_url": f"{lan_base}/video/cam0" if lan_base else "",
                    "snapshot_url": f"{lan_base}/snapshot/cam0" if lan_base else "",
                    "jpeg_url": f"{lan_base}/jpeg/cam0" if lan_base else "",
                    "mjpeg_url": f"{lan_base}/mjpeg/cam0" if lan_base else "",
                    "frame_packet_template": f"{lan_base}/frame_packet/<camera_id>" if lan_base else "",
                },
                "tunnel_url": tunnel_base,
            }
        )

    @app.route("/camera/recover", methods=["POST"])
    @app.route("/camera/cycle", methods=["POST"])
    @require_session
    def camera_recover():
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            payload = {}
        reason = str(payload.get("reason") or payload.get("source") or "api").strip() or "api"
        settle_ms = max(0, _coerce_int(payload.get("settle_ms"), 350))
        camera_id = str(payload.get("camera_id") or payload.get("id") or "").strip()
        requested = []
        before = controller.camera_statuses()
        if camera_id:
            if not controller.ensure_camera_selected(camera_id, save=False):
                return jsonify({"status": "error", "message": "Camera not found"}), 404
            requested.append(camera_id)
        controller.update_config(patch={}, save=False, apply_now=True, refresh_cameras=False)
        if settle_ms > 0:
            time.sleep(float(settle_ms) / 1000.0)
        after = controller.camera_statuses()
        after_online = sum(1 for row in after if bool(row.get("online")))
        after_total = len(after)
        return jsonify(
            {
                "status": "success",
                "service": "camera_router",
                "message": "Recovery requested",
                "reason": reason,
                "requested": requested or [str(row.get("id")) for row in before if row.get("online")],
                "elapsed_ms": int(settle_ms),
                "after_online": int(after_online),
                "after_total": int(after_total),
            }
        )

    @app.route("/camera_state/<camera_id>", methods=["GET", "POST"])
    @require_session
    def camera_state(camera_id: str):
        camera_key = str(camera_id or "").strip()
        row = _find_camera_row(camera_key)
        if row is None:
            return jsonify({"status": "error", "message": "Camera not found", "camera_id": camera_key}), 404
        if request.method == "POST":
            payload = request.get_json(silent=True) or {}
            enabled = payload.get("enabled")
            if enabled is None and isinstance(payload.get("camera"), dict):
                enabled = payload.get("camera", {}).get("enabled")
            if enabled is None:
                return jsonify({"status": "error", "message": "No state change requested", "camera_id": camera_key}), 400
            controller.set_camera_enabled(camera_key, _coerce_bool(enabled, True), save=True)
            row = _find_camera_row(camera_key) or row
            return jsonify(
                {
                    "status": "success",
                    "camera_id": camera_key,
                    "changed": True,
                    "enabled": bool(row.get("enabled", True)),
                    "configured_enabled": bool(row.get("enabled", True)),
                    "configured_enabled_key": camera_key,
                    "message": "Camera state updated",
                }
            )
        return jsonify(
            {
                "status": "success",
                "camera_id": camera_key,
                "enabled": bool(row.get("enabled", True)),
                "configured_enabled": bool(row.get("enabled", True)),
                "configured_enabled_key": camera_key,
            }
        )

    @app.route("/camera/config", methods=["POST"])
    @require_session
    def camera_config():
        payload = request.get_json(silent=True) or {}
        camera_id = str(payload.get("camera_id") or payload.get("id") or "").strip()
        if not camera_id:
            return jsonify({"status": "error", "message": "camera_id is required"}), 400
        controller.set_camera_enabled(camera_id, _coerce_bool(payload.get("enabled"), True), save=True)
        row = _find_camera_row(camera_id)
        return jsonify({"status": "success", "camera": row or {"id": camera_id}})

    @app.route("/stream_options/<camera_id>", methods=["GET", "POST"])
    @require_session
    def stream_options_for_camera(camera_id: str):
        camera_key = str(camera_id or "").strip()
        row = _find_camera_row(camera_key)
        if row is None:
            return jsonify({"status": "error", "message": "Camera not found"}), 404
        profile = dict(row.get("capture_profile") or {})
        profile_revision = int(profile_revisions.get(camera_key, 1))
        rotation = int(row.get("rotation_degrees", 0) or 0)

        if request.method == "POST":
            payload = request.get_json(silent=True) or {}
            if not isinstance(payload, dict):
                return jsonify({"status": "error", "message": "Payload must be an object"}), 400
            changed = False
            profile_changed = False
            rotation_changed = False

            rotation_input = None
            for candidate in ("rotation_degrees", "rotate_degrees", "rotation"):
                if candidate in payload:
                    rotation_input = payload.get(candidate)
                    break
            if rotation_input is None and "rotate_clockwise" in payload:
                rotation_input = 90 if _coerce_bool(payload.get("rotate_clockwise"), False) else 0
            if rotation_input is not None:
                normalized_rotation = _coerce_int(rotation_input, rotation)
                if normalized_rotation not in VALID_ORIENTATIONS:
                    return jsonify(
                        {
                            "status": "error",
                            "message": "rotation_degrees must be one of 0, 90, 180, 270",
                            "camera_id": camera_key,
                        }
                    ), 400
                controller.update_config(
                    patch={
                        "camera_index": _camera_index_from_id(camera_key),
                        "orientation": normalized_rotation,
                    },
                    save=True,
                    apply_now=True,
                    refresh_cameras=False,
                )
                rotation = normalized_rotation
                rotation_changed = True
                changed = True

            profile_payload = None
            if "profile" in payload and isinstance(payload.get("profile"), dict):
                profile_payload = payload.get("profile")
            elif any(key in payload for key in ("pixel_format", "width", "height", "fps")):
                profile_payload = {
                    "pixel_format": payload.get("pixel_format"),
                    "width": payload.get("width"),
                    "height": payload.get("height"),
                    "fps": payload.get("fps"),
                }
            if profile_payload is not None:
                profile, profile_changed = controller.set_camera_profile(camera_key, profile_payload, save=True)
                if profile_changed:
                    profile_revisions[camera_key] = int(profile_revisions.get(camera_key, 1)) + 1
                    profile_revision = int(profile_revisions[camera_key])
                changed = changed or profile_changed

            if not changed:
                return jsonify(
                    {
                        "status": "error",
                        "message": "No changes requested. Provide profile fields and/or rotation_degrees.",
                        "camera_id": camera_key,
                    }
                ), 400

            return jsonify(
                {
                    "status": "success",
                    "camera_id": camera_key,
                    "changed": bool(changed),
                    "profile_changed": bool(profile_changed),
                    "profile_revision": int(profile_revision),
                    "profile": profile,
                    "profile_restart_forced": bool(profile_changed),
                    "rotation_changed": bool(rotation_changed),
                    "configured_rotation_degrees": int(rotation),
                    "configured_rotation_key": camera_key,
                    "effective_rotation_degrees": int(rotation),
                    "default_rotation_degrees": 0,
                    "message": "Camera options updated",
                }
            )

        return jsonify(
            {
                "status": "success",
                "camera_id": camera_key,
                "protocols": controller.protocol_capabilities(),
                "modes": controller.camera_mode_urls(camera_key),
                "profile_mutable": True,
                "current_profile": profile,
                "profile_revision": int(profile_revision),
                "available_profiles": [dict(item) for item in STREAM_PROFILE_OPTIONS],
                "profile_query_error": "",
                "default_rotation_degrees": 0,
                "configured_rotation_degrees": int(rotation),
                "configured_rotation_key": camera_key,
                "effective_rotation_degrees": int(rotation),
            }
        )

    @app.route("/imu", methods=["GET"])
    @require_session
    def imu_endpoint():
        return jsonify({"accel": None, "gyro": None, "server_time_ms": int(time.time() * 1000)})

    @app.route("/imu/stream", methods=["GET"])
    @require_session
    def imu_stream_endpoint():
        hz = max(1, min(120, _coerce_int(request.args.get("hz"), 20)))
        interval_seconds = 1.0 / float(hz)

        def generate():
            while True:
                payload = {
                    "accel": None,
                    "gyro": None,
                    "server_time_ms": int(time.time() * 1000),
                }
                yield f"event: imu\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n"
                time.sleep(interval_seconds)

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.route("/camera/<camera_id>", methods=["GET"])
    @require_session
    def camera_snapshot(camera_id: str):
        camera_key = str(camera_id or "").strip()
        row = _find_camera_row(camera_key)
        if row is None:
            return Response(b"Camera not found", status=404)
        if not bool(row.get("enabled", True)):
            return Response(b"Camera disabled", status=403)
        if not controller.ensure_camera_selected(camera_key, save=False):
            return Response(b"Camera not found", status=404)
        target = controller.current_session()
        frame = target.latest_jpeg() if target else b""
        if not frame:
            return Response(b"No frame", status=503)
        return Response(frame, mimetype="image/jpeg")

    @app.route("/snapshot/<camera_id>", methods=["GET"])
    @require_session
    def snapshot_alias(camera_id: str):
        return camera_snapshot(camera_id)

    @app.route("/jpeg/<camera_id>", methods=["GET"])
    @require_session
    def jpeg_snapshot(camera_id: str):
        return camera_snapshot(camera_id)

    @app.route("/frame_packet/<camera_id>", methods=["GET"])
    @require_session
    def frame_packet(camera_id: str):
        camera_key = str(camera_id or "").strip()
        row = _find_camera_row(camera_key)
        if row is None:
            return jsonify({"status": "error", "message": "Camera not found", "camera_id": camera_key}), 404
        if not bool(row.get("enabled", True)):
            return jsonify({"status": "error", "message": "Camera disabled", "camera_id": camera_key}), 403
        if not controller.ensure_camera_selected(camera_key, save=False):
            return jsonify({"status": "error", "message": "Camera not found", "camera_id": camera_key}), 404
        target = controller.current_session()
        frame = target.latest_jpeg() if target else b""
        if not frame:
            return jsonify({"status": "error", "message": "Frame unavailable", "camera_id": camera_key}), 503
        frame_b64 = base64.b64encode(frame).decode("ascii")
        return jsonify(
            {
                "status": "success",
                "frame_packet": {
                    "camera_id": camera_key,
                    "frame": frame_b64,
                    "encoding": "image/jpeg;base64",
                    "timestamp_ms": int(time.time() * 1000),
                    "width": int(row.get("width") or 0),
                    "height": int(row.get("height") or 0),
                    "quality": 82,
                    "bytes": int(len(frame)),
                    "max_kbps": _coerce_int(request.args.get("max_kbps"), 900),
                    "interval_ms": _coerce_int(request.args.get("interval_ms"), 280),
                },
            }
        )

    @app.route("/video/<camera_id>", methods=["GET"])
    @require_session
    def video(camera_id: str):
        camera_key = str(camera_id or "").strip()
        row = _find_camera_row(camera_key)
        if row is None:
            return Response(b"Camera not found", status=404)
        if not bool(row.get("enabled", True)):
            return Response(b"Camera disabled", status=403)
        return Response(
            _mjpeg_generator(camera_key),
            mimetype="multipart/x-mixed-replace; boundary=frame",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
                "X-Accel-Buffering": "no",
            },
        )

    @app.route("/mjpeg/<camera_id>", methods=["GET"])
    @require_session
    def mjpeg_alias(camera_id: str):
        return video(camera_id)

    @app.route("/stream", methods=["GET"])
    @require_session
    def stream():
        current = controller.snapshot().get("config", {})
        camera_id = _camera_id_for_index(_coerce_int(current.get("camera_index"), 0))
        return video(camera_id)

    @app.route("/tunnel_info", methods=["GET"])
    def tunnel_info():
        tunnel = _tunnel_payload()
        tunnel_base = str(tunnel.get("tunnel_url") or "").strip()
        state = str(tunnel.get("state") or "").strip().lower()
        if tunnel_base:
            status = "success"
            message = "Tunnel URL available"
        elif state in ("error", "stale"):
            status = "unavailable"
            message = "Tunnel unavailable"
        else:
            status = "pending"
            message = "Tunnel URL not yet available"
        return jsonify(
            {
                "status": status,
                "service": "camera_router",
                "message": message,
                "state": str(tunnel.get("state") or ""),
                "tunnel_url": tunnel_base,
                "dashboard_url": f"{tunnel_base}/dashboard" if tunnel_base else "",
                "auth_url": f"{tunnel_base}/auth" if tunnel_base else "",
                "session_rotate_url": f"{tunnel_base}/session/rotate" if tunnel_base else "",
                "list_url": f"{tunnel_base}/list" if tunnel_base else "",
                "health_url": f"{tunnel_base}/health" if tunnel_base else "",
                "video_url": f"{tunnel_base}/video/cam0" if tunnel_base else "",
                "snapshot_url": f"{tunnel_base}/snapshot/cam0" if tunnel_base else "",
                "jpeg_url": f"{tunnel_base}/jpeg/cam0" if tunnel_base else "",
                "mjpeg_url": f"{tunnel_base}/mjpeg/cam0" if tunnel_base else "",
                "frame_packet_template": f"{tunnel_base}/frame_packet/<camera_id>" if tunnel_base else "",
                "stale_tunnel_url": str(tunnel.get("stale_tunnel_url") or ""),
                "error": str(tunnel.get("error") or ""),
                "running": bool(tunnel.get("running")),
                "enabled": bool(tunnel.get("enabled")),
                "source": str(tunnel.get("source") or ""),
            }
        )

    @app.route("/router_info", methods=["GET"])
    def router_info():
        local_base, lan_base, tunnel_base = _current_bases()
        publish_base = tunnel_base or lan_base or local_base
        tunnel = _tunnel_payload()
        rows = controller.camera_statuses()
        camera_routes = [
            _camera_row_with_urls(row, publish_base, local_base, lan_base, tunnel_base)
            for row in rows
        ]
        security = _security_settings()
        selected_transport = "tunnel" if tunnel_base else ("lan" if lan_base else "local")
        selected_base = publish_base
        return jsonify(
            {
                "status": "success",
                "service": "camera_router",
                "transport": selected_transport,
                "base_url": selected_base,
                "list_url": f"{selected_base}/list" if selected_base else "",
                "health_url": f"{selected_base}/health" if selected_base else "",
                "video_url": f"{selected_base}/video/cam0" if selected_base else "",
                "snapshot_url": f"{selected_base}/snapshot/cam0" if selected_base else "",
                "jpeg_url": f"{selected_base}/jpeg/cam0" if selected_base else "",
                "mjpeg_url": f"{selected_base}/mjpeg/cam0" if selected_base else "",
                "local_base_url": local_base,
                "lan_base_url": lan_base,
                "cameras": camera_routes,
                "local": {
                    "base_url": local_base,
                    "dashboard_url": f"{local_base}/dashboard",
                    "auth_url": f"{local_base}/auth",
                    "session_rotate_url": f"{local_base}/session/rotate",
                    "list_url": f"{local_base}/list",
                    "health_url": f"{local_base}/health",
                    "video_url": f"{local_base}/video/cam0",
                    "snapshot_url": f"{local_base}/snapshot/cam0",
                    "jpeg_url": f"{local_base}/jpeg/cam0",
                    "mjpeg_url": f"{local_base}/mjpeg/cam0",
                    "frame_packet_template": f"{local_base}/frame_packet/<camera_id>",
                    "lan_base_url": lan_base,
                    "lan_dashboard_url": f"{lan_base}/dashboard" if lan_base else "",
                    "lan_auth_url": f"{lan_base}/auth" if lan_base else "",
                    "lan_session_rotate_url": f"{lan_base}/session/rotate" if lan_base else "",
                    "lan_list_url": f"{lan_base}/list" if lan_base else "",
                    "lan_health_url": f"{lan_base}/health" if lan_base else "",
                    "lan_video_url": f"{lan_base}/video/cam0" if lan_base else "",
                    "lan_snapshot_url": f"{lan_base}/snapshot/cam0" if lan_base else "",
                    "lan_jpeg_url": f"{lan_base}/jpeg/cam0" if lan_base else "",
                    "lan_mjpeg_url": f"{lan_base}/mjpeg/cam0" if lan_base else "",
                    "lan_frame_packet_template": f"{lan_base}/frame_packet/<camera_id>" if lan_base else "",
                },
                "tunnel": {
                    **tunnel,
                    "dashboard_url": f"{tunnel_base}/dashboard" if tunnel_base else "",
                    "auth_url": f"{tunnel_base}/auth" if tunnel_base else "",
                    "session_rotate_url": f"{tunnel_base}/session/rotate" if tunnel_base else "",
                    "list_url": f"{tunnel_base}/list" if tunnel_base else "",
                    "health_url": f"{tunnel_base}/health" if tunnel_base else "",
                    "video_url": f"{tunnel_base}/video/cam0" if tunnel_base else "",
                    "snapshot_url": f"{tunnel_base}/snapshot/cam0" if tunnel_base else "",
                    "jpeg_url": f"{tunnel_base}/jpeg/cam0" if tunnel_base else "",
                    "mjpeg_url": f"{tunnel_base}/mjpeg/cam0" if tunnel_base else "",
                    "frame_packet_template": f"{tunnel_base}/frame_packet/<camera_id>" if tunnel_base else "",
                },
                "security": {
                    "require_auth": bool(security.get("require_auth", DEFAULT_REQUIRE_AUTH)),
                    "session_timeout": int(security.get("session_timeout", DEFAULT_SESSION_TIMEOUT)),
                },
            }
        )

    return app


def prepare_runtime_imports() -> None:
    global cv2
    global np
    global hailo
    global Gst
    global Flask
    global Response
    global jsonify
    global request
    global render_template_string
    global stream_with_context
    global CORS
    global app_callback_class
    global get_caps_from_pad
    global get_numpy_from_buffer
    global gstreamer_app_module

    import cv2 as _cv2
    import numpy as _np
    import hailo as _hailo
    import gi

    # Some installations may miss setproctitle; fallback to a harmless no-op shim.
    try:
        import setproctitle as _setproctitle  # noqa: F401
    except ModuleNotFoundError:
        class _SetProcTitleShim:
            @staticmethod
            def setproctitle(_title: str) -> None:
                return None

        sys.modules["setproctitle"] = _SetProcTitleShim()

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst as _Gst
    from flask import Flask as _Flask
    from flask import Response as _Response
    from flask import jsonify as _jsonify
    from flask import request as _request
    from flask import render_template_string as _render_template_string
    from flask import stream_with_context as _stream_with_context

    _CORS = None
    try:
        from flask_cors import CORS as _CORS
    except Exception:
        _CORS = None

    from hailo_apps.python.core.common.buffer_utils import (
        get_caps_from_pad as _get_caps_from_pad,
        get_numpy_from_buffer as _get_numpy_from_buffer,
    )
    from hailo_apps.python.core.gstreamer.gstreamer_app import (
        app_callback_class as _app_callback_class,
    )
    import hailo_apps.python.core.gstreamer.gstreamer_app as _gstreamer_app_module

    cv2 = _cv2
    np = _np
    hailo = _hailo
    Gst = _Gst
    Flask = _Flask
    Response = _Response
    jsonify = _jsonify
    request = _request
    render_template_string = _render_template_string
    stream_with_context = _stream_with_context
    CORS = _CORS
    app_callback_class = _app_callback_class
    get_caps_from_pad = _get_caps_from_pad
    get_numpy_from_buffer = _get_numpy_from_buffer
    gstreamer_app_module = _gstreamer_app_module


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    project_root = resolve_project_root(script_dir, args.hailo_apps_path)
    if project_root is None:
        project_root = clone_hailo_apps_repo(script_dir, args)
    ensure_hailo_apps_installation(project_root, args)

    ensure_bootstrap_environment(project_root, args)
    apply_setup_env_behavior(project_root)
    prepare_runtime_imports()

    arch, model_options = build_model_options()
    camera_options = discover_csi_cameras()
    config_path = resolve_config_path(args)

    base_config = normalize_config(load_raw_config(config_path), model_options, camera_options)
    merged_config = normalize_config(
        apply_cli_overrides(base_config, args),
        model_options,
        camera_options,
    )
    if not config_path.exists():
        save_config(config_path, merged_config)

    is_tty = sys.stdin.isatty() and sys.stdout.isatty()
    if args.no_curses or not is_tty:
        active_config = merged_config
    else:
        _, active_config, camera_options = choose_selection_with_curses(
            model_options,
            camera_options,
            merged_config,
            config_path,
        )

    save_config(config_path, active_config)
    ensure_port_available(active_config["host"], active_config["port"])
    controller = RuntimeController(
        arch=arch,
        model_options=model_options,
        camera_options=camera_options,
        config_path=config_path,
        initial_config=active_config,
    )
    controller.start()
    service_running.set()
    _apply_tunnel_runtime_config(active_config, startup_delay=2.0)

    snapshot = controller.snapshot()
    flask_app = create_flask_app(controller, config_path)
    dashboard_url = f"http://{active_config['host']}:{active_config['port']}/"
    print(f"[launch] model={snapshot['session']['model_label']}")
    print(f"[launch] camera={snapshot['session']['camera_label']}")
    print(f"[launch] mode={active_config['run_mode']}")
    print(f"[launch] config={config_path}")
    print(f"[launch] dashboard={dashboard_url}")

    try:
        flask_app.run(
            host=active_config["host"],
            port=active_config["port"],
            debug=False,
            threaded=True,
            use_reloader=False,
        )
    finally:
        service_running.clear()
        _stop_cloudflared_tunnel()
        controller.stop()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
