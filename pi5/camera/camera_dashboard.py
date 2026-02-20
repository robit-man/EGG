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
import curses
import importlib
import json
import os
import re
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
from pathlib import Path
from typing import Dict, List, Optional


INVALID_KERNELS = {"6.12.21", "6.12.22", "6.12.23", "6.12.24", "6.12.25"}
DEFAULT_ENV_FILE = Path("/usr/local/hailo/resources/.env")
BOOTSTRAP_STAMP_NAME = ".hailo_csi_dashboard_bootstrap"
BOOTSTRAP_STAMP_VALUE = "v1"
VALID_ORIENTATIONS = [0, 90, 180, 270]
DEFAULT_HAILO_REPO_URL = "https://github.com/hailo-ai/hailo-apps.git"
DEFAULT_HAILO_REPO_DIRNAME = "hailo-apps"


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
      font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
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
        <img class="stream" src="/stream" alt="Stream" />
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
    let modelCatalog = {};
    let cameras = [];
    let currentConfig = {};
    let serverConfig = {};
    let drawerOpen = false;
    let formDirty = false;

    function setConfigNote(text) {
      document.getElementById('configNote').textContent = text || '';
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
        const response = await fetch('/api/status');
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
      const response = await fetch(path, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });
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
    return {
        "run_mode": "model",
        "model_family": first_family,
        "model_name": first_model,
        "camera_index": first_camera_index,
        "camera_orientations": {str(first_camera_index): 0},
        "host": "0.0.0.0",
        "port": 5000,
        "width": 1280,
        "height": 720,
        "fps": 30,
    }


def load_raw_config(config_path: Path) -> dict:
    if not config_path.exists():
        return {}
    try:
        payload = json.loads(config_path.read_text())
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    return {}


def save_config(config_path: Path, payload: dict) -> None:
    clean_payload = {key: value for key, value in payload.items() if not str(key).startswith("_")}
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(clean_payload, indent=2, sort_keys=True) + "\n")


def _coerce_int(value, fallback: int) -> int:
    try:
        return int(value)
    except Exception:
        return fallback


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

    cfg["host"] = str(raw_config.get("host", cfg["host"]))
    cfg["port"] = _coerce_int(raw_config.get("port", cfg["port"]), cfg["port"])
    cfg["width"] = max(320, _coerce_int(raw_config.get("width", cfg["width"]), cfg["width"]))
    cfg["height"] = max(240, _coerce_int(raw_config.get("height", cfg["height"]), cfg["height"]))
    cfg["fps"] = max(1, _coerce_int(raw_config.get("fps", cfg["fps"]), cfg["fps"]))
    return cfg


def apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    merged = dict(config)
    merged["camera_orientations"] = dict(config["camera_orientations"])

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
    return merged


def get_orientation_for_camera(config: dict, camera_index: int) -> int:
    value = config.get("camera_orientations", {}).get(str(camera_index), 0)
    return value if value in VALID_ORIENTATIONS else 0


def set_orientation_for_camera(config: dict, camera_index: int, orientation: int) -> None:
    config.setdefault("camera_orientations", {})
    config["camera_orientations"][str(camera_index)] = orientation if orientation in VALID_ORIENTATIONS else 0


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
        "import flask; import hailo_apps; import setproctitle; import cv2; import numpy",
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
        run_checked([str(venv_python), "-m", "pip", "install", "flask"])
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
    if "orientation" in patch:
        set_orientation_for_camera(
            merged,
            merged["camera_index"],
            _coerce_int(patch.get("orientation"), get_orientation_for_camera(merged, merged["camera_index"])),
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
        return config

    def _camera_payload(self) -> List[dict]:
        return [
            {"index": camera.index, "label": camera.label, "details": camera.details}
            for camera in self.camera_options
        ]

    def _catalog_payload(self) -> dict:
        return {
            family: [item.model_name for item in options]
            for family, options in self.model_catalog.items()
        }

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
            candidate = merge_config_patch(self._copy_config(), patch)
            if refresh_cameras:
                self.camera_options = discover_csi_cameras()
            normalized = normalize_config(candidate, self.model_options, self.camera_options)

            current_host = self.config.get("host")
            current_port = self.config.get("port")
            host_or_port_changed = (
                normalized.get("host") != current_host or normalized.get("port") != current_port
            )
            self.requires_server_restart = host_or_port_changed

            self.config = normalized
            if save:
                save_config(self.config_path, self.config)
            if apply_now:
                self._restart_session_locked()

        return self.snapshot()

    def reload_from_disk(self, apply_now: bool = True) -> dict:
        with self._lock:
            loaded = normalize_config(
                load_raw_config(self.config_path),
                self.model_options,
                self.camera_options,
            )
            self.config = loaded
            if apply_now:
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
        self._started_at = time.time()
        self._camera_frame_count = 0
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

    def _camera_only_loop(self) -> None:
        from picamera2 import Picamera2

        try:
            with Picamera2(self.selection.camera.index) as picam2:
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
        except Exception:
            self.error = traceback.format_exc(limit=4)
        finally:
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
            self._camera_thread.join(timeout=2.0)

    def status_payload(self) -> dict:
        return {
            "model_label": self.model_label,
            "camera_label": self.camera_label,
            "frame_count": self.frame_count(),
            "avg_fps": self.avg_fps(),
            "detections": self._last_detection_count,
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

    @app.route("/")
    def index():
        return render_template_string(
            DASHBOARD_TEMPLATE,
            initial=controller.snapshot(),
            config_path=str(config_path),
        )

    @app.route("/api/status")
    def api_status():
        return jsonify(controller.snapshot())

    @app.route("/api/config", methods=["GET", "POST"])
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
        return jsonify(snapshot)

    @app.route("/api/config/reload", methods=["POST"])
    def api_config_reload():
        payload = request.get_json(silent=True) or {}
        apply_now = bool(payload.get("apply", True))
        return jsonify(controller.reload_from_disk(apply_now=apply_now))

    @app.route("/api/cameras/refresh", methods=["POST"])
    def api_refresh_cameras():
        payload = request.get_json(silent=True) or {}
        apply_now = bool(payload.get("apply", False))
        return jsonify(
            controller.update_config(
                patch={},
                save=False,
                apply_now=apply_now,
                refresh_cameras=True,
            )
        )

    @app.route("/stream")
    def stream():

        def mjpeg_generator():
            while True:
                target = controller.current_session()
                frame = target.latest_jpeg() if target else b""
                if frame:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        + f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii")
                        + frame
                        + b"\r\n"
                    )
                time.sleep(0.04)

        return Response(mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")

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
        controller.stop()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
