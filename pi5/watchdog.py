#!/usr/bin/env python3
"""
Teleoperation watchdog manager.

Features:
- Creates its own venv and relaunches from it (no external packages required).
- Curses dashboard showing all managed teleoperation services.
- Auto-starts services in their own terminal windows.
- Per-service toggle with arrow keys + space bar.
- Graceful stop (SIGINT -> SIGTERM -> SIGKILL escalation) when disabling.
"""

import os
import sys
import subprocess


# ---------------------------------------------------------------------------
# Virtual environment bootstrap
# ---------------------------------------------------------------------------
WATCHDOG_VENV_DIR_NAME = "watchdog_venv"


def ensure_venv():
    script_dir = os.path.abspath(os.path.dirname(__file__))
    venv_dir = os.path.join(script_dir, WATCHDOG_VENV_DIR_NAME)
    if os.path.normcase(os.path.abspath(sys.prefix)) == os.path.normcase(os.path.abspath(venv_dir)):
        return

    if os.name == "nt":
        python_path = os.path.join(venv_dir, "Scripts", "python.exe")
    else:
        python_path = os.path.join(venv_dir, "bin", "python")

    if not os.path.exists(venv_dir):
        print(f"Creating virtual environment in '{WATCHDOG_VENV_DIR_NAME}'...")
        import venv

        venv.create(venv_dir, with_pip=True)

    print("Re-launching from venv...")
    os.execv(python_path, [python_path] + sys.argv)


ensure_venv()


# ---------------------------------------------------------------------------
# Imports after venv bootstrap
# ---------------------------------------------------------------------------
import curses
import datetime
import json
import pathlib
import platform
import re
import shlex
import shutil
import signal
import socket
import threading
import time
import base64
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Tuple


WATCHDOG_RUNTIME_DIR_NAME = ".watchdog_runtime"
WATCHDOG_STATE_FILE_NAME = "service_state.json"
WATCHDOG_LOCK_FILE_NAME = "watchdog.lock"
WATCHDOG_SERVICE_LOG_DIR_NAME = "service_logs"
CPUFREQ_GLOB = "/sys/devices/system/cpu/cpu*/cpufreq"
DEFAULT_CPU_MIN_KHZ = 1600000
DEFAULT_CPU_MAX_KHZ = 1800000
DEFAULT_OVER_VOLTAGE_DELTA_UV = -25000
MIN_OVER_VOLTAGE_DELTA_UV = -100000
MAX_OVER_VOLTAGE_DELTA_UV = 100000
OVER_VOLTAGE_DELTA_RE = re.compile(r"^\s*over_voltage_delta\s*=\s*([+-]?\d+)\s*(?:#.*)?$")
FIRMWARE_CONFIG_CANDIDATES = (
    "/boot/firmware/config.txt",
    "/boot/config.txt",
)
MONITOR_INTERVAL_SECONDS = 0.25
STOP_SIGINT_GRACE_SECONDS = 5.0
STOP_SIGTERM_GRACE_SECONDS = 10.0
STOP_SIGKILL_GRACE_SECONDS = 14.0
RESTART_BACKOFF_INITIAL_SECONDS = 1.0
RESTART_BACKOFF_MAX_SECONDS = 20.0
LAUNCH_GRACE_SECONDS = 3.0
DEFAULT_ACTIVATION_TIMEOUT_SECONDS = 35.0
DEFAULT_ACTIVATION_STABILITY_SECONDS = 4.0
DEFAULT_ACTIVATION_PROBE_INTERVAL_SECONDS = 1.0
DEFAULT_HEALTH_CHECK_INTERVAL_SECONDS = 2.0
DEFAULT_HEALTH_FAILURE_THRESHOLD = 4
HTTP_PROBE_TIMEOUT_SECONDS = 1.5
RESOURCE_SAMPLE_INTERVAL_SECONDS = 1.0
RESOURCE_GRAPH_HISTORY_POINTS = 240
PORT_DISCOVERY_SCAN_LIMIT = 64
PORT_DISCOVERY_MAX_DELTA = 128
PORT_RECLAIM_TERM_WAIT_SECONDS = 1.4
PORT_RECLAIM_KILL_WAIT_SECONDS = 1.4
AUTO_UPDATE_DEFAULT_REMOTE = "origin"
AUTO_UPDATE_DEFAULT_BRANCH = "main"
AUTO_UPDATE_DEFAULT_REPO_URL = "https://github.com/robit-man/EGG.git"
AUTO_UPDATE_POLL_SECONDS = 20.0
AUTO_UPDATE_FETCH_TIMEOUT_SECONDS = 45.0
AUTO_UPDATE_PULL_TIMEOUT_SECONDS = 120.0
AUTO_UPDATE_RUNTIME_SYNC_TIMEOUT_SECONDS = 180.0
SERVICE_LINK_REFRESH_SECONDS = 15.0
SERVICE_LINK_HTTP_TIMEOUT_SECONDS = 1.2
SERVICE_SESSION_REFRESH_SAFETY_SECONDS = 12.0
PIPELINE_OBS_POLL_SECONDS = 1.2
STATE_EXTERNAL_CPU_RELOAD_INTERVAL_SECONDS = 0.8
DIRECT_LAUNCH_SERVICE_IDS = (
    "tts_output",
    "llm_bridge",
    "tts_voice",
    "asr_stream",
    "ollama",
)
OLLAMA_HARD_STOP_TERM_WAIT_SECONDS = 0.8
OLLAMA_HARD_STOP_KILL_WAIT_SECONDS = 1.2


def _get_nested(data: dict, path: str, default=None):
    current = data
    for key in str(path or "").split("."):
        if not key:
            continue
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


@dataclass(frozen=True)
class ServiceSpec:
    service_id: str
    label: str
    script_relpath: str
    args: Tuple[str, ...] = ()
    auto_start: bool = True
    launch_in_terminal: bool = True
    health_mode: str = "process"  # process | tcp | http
    health_port: int = 0
    health_path: str = ""
    config_relpath: str = ""
    config_port_paths: Tuple[str, ...] = ()
    activation_timeout_seconds: float = DEFAULT_ACTIVATION_TIMEOUT_SECONDS
    activation_stability_seconds: float = DEFAULT_ACTIVATION_STABILITY_SECONDS
    health_check_interval_seconds: float = DEFAULT_HEALTH_CHECK_INTERVAL_SECONDS
    health_failure_threshold: int = DEFAULT_HEALTH_FAILURE_THRESHOLD

    def script_path(self, base_dir: pathlib.Path) -> pathlib.Path:
        return (base_dir / self.script_relpath).resolve()

    def working_dir(self, base_dir: pathlib.Path) -> pathlib.Path:
        return self.script_path(base_dir).parent

    def config_path(self, base_dir: pathlib.Path) -> Optional[pathlib.Path]:
        rel = str(self.config_relpath or "").strip()
        if not rel:
            return None
        return (base_dir / rel).resolve()

    def resolved_health_port(self, base_dir: pathlib.Path) -> int:
        cfg_path = self.config_path(base_dir)
        if cfg_path and cfg_path.exists():
            try:
                payload = json.loads(cfg_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    for path in self.config_port_paths:
                        value = _get_nested(payload, path, None)
                        if isinstance(value, bool):
                            continue
                        try:
                            port = int(value)
                        except Exception:
                            continue
                        if 1 <= port <= 65535:
                            return port
            except Exception:
                pass
        try:
            port = int(self.health_port)
        except Exception:
            port = 0
        return port if 1 <= port <= 65535 else 0

    def resolved_health_target(self, base_dir: pathlib.Path) -> str:
        mode = str(self.health_mode or "process").strip().lower()
        if mode == "process":
            return "process"
        port = self.resolved_health_port(base_dir)
        if port <= 0:
            return ""
        if mode == "tcp":
            return f"127.0.0.1:{port}"
        path = str(self.health_path or "").strip()
        if not path:
            path = "/"
        if not path.startswith("/"):
            path = f"/{path}"
        return f"http://127.0.0.1:{port}{path}"


@dataclass
class ServiceRuntime:
    desired_enabled: bool = False
    restart_after_stop: bool = False
    state: str = "stopped"  # stopped | launching | activating | running | degraded | stopping | error | missing
    pid: Optional[int] = None
    terminal_process: Optional[subprocess.Popen] = None
    stop_stage: int = 0
    stop_requested_at: float = 0.0
    next_restart_at: float = 0.0
    restart_backoff_seconds: float = RESTART_BACKOFF_INITIAL_SECONDS
    launch_grace_until: float = 0.0
    started_at: float = 0.0
    stopped_at: float = 0.0
    last_event: str = ""
    last_error: str = ""
    last_state_change_at: float = 0.0
    start_count: int = 0
    stop_count: int = 0
    crash_count: int = 0
    restart_count: int = 0
    launch_attempts: int = 0
    launch_attempt_started_at: float = 0.0
    activation_deadline: float = 0.0
    activation_checks: int = 0
    activation_failures: int = 0
    activation_method: str = ""
    process_stable_since: float = 0.0
    health_checks: int = 0
    health_failures: int = 0
    consecutive_health_failures: int = 0
    last_health_probe_at: float = 0.0
    last_health_ok_at: float = 0.0
    last_health_error: str = ""
    resolved_health_port: int = 0
    last_port_discovery_at: float = 0.0
    cpu_percent: float = 0.0
    rss_mb: float = 0.0
    peak_cpu_percent: float = 0.0
    peak_cpu_at: float = 0.0
    peak_rss_mb: float = 0.0
    peak_rss_at: float = 0.0


class WatchdogManager:
    def __init__(self, base_dir: pathlib.Path):
        self.base_dir = base_dir
        self.runtime_dir = self.base_dir / WATCHDOG_RUNTIME_DIR_NAME
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.service_log_dir = self.runtime_dir / WATCHDOG_SERVICE_LOG_DIR_NAME
        self.service_log_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.runtime_dir / WATCHDOG_STATE_FILE_NAME
        self.lock_file = self.runtime_dir / WATCHDOG_LOCK_FILE_NAME

        self.services: List[ServiceSpec] = [
            ServiceSpec(
                "camera_router",
                "Camera Router",
                "camera_router.py",
                health_mode="http",
                health_port=8080,
                health_path="/health",
                config_relpath="camera_router_config.json",
                config_port_paths=("camera_router.network.listen_port",),
                activation_timeout_seconds=45.0,
            ),
            ServiceSpec(
                "pipeline_api",
                "Pipeline API",
                "pipeline_api.py",
                health_mode="http",
                health_port=6590,
                health_path="/health",
                config_relpath="pipeline_api_config.json",
                config_port_paths=("pipeline_api.network.listen_port",),
                activation_timeout_seconds=40.0,
            ),
            ServiceSpec(
                "audio_router",
                "Audio Router",
                "audio_router.py",
                health_mode="http",
                health_port=8090,
                health_path="/health",
                config_relpath="audio_router_config.json",
                config_port_paths=("audio_router.network.listen_port",),
                activation_timeout_seconds=50.0,
            ),
            ServiceSpec(
                "tts_output",
                "TTS Output",
                "output.py",
                launch_in_terminal=False,
                health_mode="tcp",
                health_port=6353,
                config_relpath="audio_router_config.json",
                config_port_paths=("audio_router.integrations.audio_out_port",),
                activation_timeout_seconds=300.0,
                health_check_interval_seconds=6.0,
            ),
            ServiceSpec(
                "llm_bridge",
                "LLM Bridge",
                "model_to_tts.py",
                launch_in_terminal=False,
                health_mode="tcp",
                health_port=6545,
                config_relpath="audio_router_config.json",
                config_port_paths=("audio_router.integrations.llm_port",),
                activation_timeout_seconds=300.0,
                health_check_interval_seconds=6.0,
            ),
            ServiceSpec(
                "tts_voice",
                "TTS Voice",
                "run_voice_server.py",
                launch_in_terminal=False,
                health_mode="tcp",
                health_port=6434,
                config_relpath="audio_router_config.json",
                config_port_paths=("audio_router.integrations.tts_port",),
                activation_timeout_seconds=240.0,
                health_check_interval_seconds=6.0,
            ),
            ServiceSpec(
                "asr_stream",
                "ASR Stream",
                "run_asr_stream.py",
                launch_in_terminal=False,
                health_mode="process",
                activation_timeout_seconds=180.0,
            ),
            ServiceSpec(
                "ollama",
                "Ollama",
                "run_ollama_service.py",
                launch_in_terminal=False,
                health_mode="http",
                health_port=11434,
                health_path="/api/tags",
                activation_timeout_seconds=180.0,
            ),
            ServiceSpec(
                "router",
                "NKN Router",
                "router.py",
                health_mode="http",
                health_port=5070,
                health_path="/health",
                config_relpath="router_config.json",
                config_port_paths=("router.network.listen_port",),
                activation_timeout_seconds=35.0,
            ),
        ]
        self.service_by_id: Dict[str, ServiceSpec] = {svc.service_id: svc for svc in self.services}
        self.runtime_by_id: Dict[str, ServiceRuntime] = {}

        self._lock = threading.Lock()
        self._session_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None
        self._service_link_thread: Optional[threading.Thread] = None
        self._logs: Deque[str] = deque(maxlen=300)
        self._click_regions: List[Dict[str, object]] = []
        self._selected_index = 0
        self._bottom_tabs: Tuple[str, str] = ("logs", "resources")
        self._active_bottom_tab = 0
        self._run_started_at = time.time()
        self._instance_lock_acquired = False
        self._instance_lock_handle = None
        self._service_links: Dict[str, Dict[str, str]] = {}
        self._service_link_errors: Dict[str, str] = {}
        self._service_link_last_refresh_at = 0.0
        self._service_sessions: Dict[str, Dict[str, object]] = {}
        self._service_link_refresh_interval = max(
            8.0,
            float(os.environ.get("WATCHDOG_LINK_REFRESH_SECONDS", SERVICE_LINK_REFRESH_SECONDS)),
        )
        self._pipeline_obs_last_seq = 0
        self._pipeline_obs_last_poll_at = 0.0
        self._pipeline_obs_poll_interval = max(
            0.4,
            float(os.environ.get("WATCHDOG_PIPELINE_OBS_POLL_SECONDS", PIPELINE_OBS_POLL_SECONDS)),
        )
        self._pipeline_stage_state: Dict[str, str] = {"asr": "idle", "llm": "idle", "tts": "idle"}
        self._state_file_last_mtime = 0.0
        self._state_file_external_last_poll_at = 0.0
        self._lan_ip = self._detect_lan_ip()
        self.repo_dir = pathlib.Path(
            os.environ.get("WATCHDOG_REPO_DIR", str((self.base_dir / "EGG_repo").resolve()))
        ).resolve()
        self._restart_requested = False
        self._restart_reason = ""
        self._auto_update_remote = AUTO_UPDATE_DEFAULT_REMOTE
        self._auto_update_branch = AUTO_UPDATE_DEFAULT_BRANCH
        self._auto_update_repo_url = AUTO_UPDATE_DEFAULT_REPO_URL
        self._auto_update_poll_seconds = AUTO_UPDATE_POLL_SECONDS
        self._next_auto_update_check_at = 0.0
        self._auto_update_enabled = False
        self._auto_update_stash_dirty = self._env_flag("WATCHDOG_AUTO_UPDATE_STASH_DIRTY", True)
        self._port_reclaim_enabled = self._env_flag("WATCHDOG_RECLAIM_PORTS", True)
        self._port_reclaim_force = self._env_flag("WATCHDOG_RECLAIM_FORCE", False)
        self._tcp_passive_probe = self._env_flag("WATCHDOG_TCP_PASSIVE_PROBE", True)
        self._force_direct = self._env_flag("WATCHDOG_FORCE_DIRECT", False)
        self._last_resource_sample_at = 0.0
        self._resource_sample_interval = max(0.4, float(RESOURCE_SAMPLE_INTERVAL_SECONDS))
        self._resource_cpu_history: Deque[float] = deque(maxlen=RESOURCE_GRAPH_HISTORY_POINTS)
        self._resource_ram_history: Deque[float] = deque(maxlen=RESOURCE_GRAPH_HISTORY_POINTS)
        self._system_cpu_percent = 0.0
        self._system_ram_percent = 0.0
        self._system_mem_total_kb = 0
        self._proc_cpu_prev_total = 0
        self._proc_cpu_prev_idle = 0

        self._os_name = platform.system().lower()
        self._has_passive_tcp_tools = bool(
            self._os_name == "windows" or shutil.which("lsof") or shutil.which("ss")
        )
        self._cpu_cpufreq_paths: List[pathlib.Path] = []
        self._cpu_available_governors: List[str] = []
        self._cpu_hw_min_khz = 0
        self._cpu_hw_max_khz = 0
        self._cpu_current_governor = ""
        self._cpu_current_min_khz = 0
        self._cpu_current_max_khz = 0
        self._cpu_current_over_voltage_delta_uv = 0
        self._firmware_config_path = self._detect_firmware_config_path()
        self._cpu_throttle_supported = False
        self._cpu_throttle_last_apply = ""
        self._cpu_throttle = {
            "governor": "ondemand",
            "min_khz": DEFAULT_CPU_MIN_KHZ,
            "max_khz": DEFAULT_CPU_MAX_KHZ,
            "over_voltage_delta_uv": DEFAULT_OVER_VOLTAGE_DELTA_UV,
            "auto_apply": False,
            "asr_llm_throttle_enable": True,
            "asr_llm_throttle_percent": 65,
            "asr_llm_throttle_cycle_ms": 320,
        }
        self._asr_throttle_paused = False
        self._asr_throttle_last_switch_at = 0.0
        self._asr_throttle_target_pid = 0
        self._refresh_cpu_throttle_support()
        self._refresh_firmware_voltage_setting()
        self._terminal_emulator = self._detect_terminal_emulator()
        if (
            self._os_name not in ("windows", "darwin")
            and not self._terminal_emulator
            and not self._force_direct
        ):
            self._log("[WARN] No terminal emulator detected; Linux service launch will fail")

        self._acquire_instance_lock()
        self._configure_auto_update()

        desired_overrides = self._load_desired_state()
        if self._cpu_throttle.get("auto_apply", False):
            applied, detail = self._apply_cpu_throttle_settings()
            self._cpu_throttle_last_apply = detail
            if applied:
                self._log(f"[CPU] Auto-applied throttle: {detail}")
            else:
                self._log(f"[WARN] CPU throttle auto-apply failed: {detail}")
        now = time.time()
        for svc in self.services:
            desired_enabled = desired_overrides.get(svc.service_id)
            if desired_enabled is None:
                desired_enabled = svc.auto_start
            runtime = ServiceRuntime(desired_enabled=bool(desired_enabled), state="stopped")
            script_path = svc.script_path(self.base_dir)
            if not script_path.exists():
                runtime.desired_enabled = False
                runtime.state = "missing"
                runtime.last_error = f"Missing: {svc.script_relpath}"
            else:
                pid = self._read_pid_file(svc)
                if pid and self._is_pid_running(pid):
                    runtime.pid = pid
                    runtime.state = "activating"
                    runtime.started_at = now
                    runtime.launch_attempt_started_at = now
                    runtime.activation_deadline = now + svc.activation_timeout_seconds
                    runtime.process_stable_since = now
                    runtime.last_event = f"Recovered existing pid={pid}; probing health"
                elif pid:
                    self._remove_pid_file(svc)
            runtime.last_state_change_at = now
            self.runtime_by_id[svc.service_id] = runtime

        self._log("Watchdog initialized")
        self._log(
            "Keys: Up/Down select, Space toggle, R restart, A toggle all, "
            "C copy selected link, L log links, S CPU settings, Q quit"
        )
        if self._port_reclaim_enabled:
            reclaim_mode = "aggressive (includes foreign pids)" if self._port_reclaim_force else "owned-only"
            self._log(f"[PORT] Auto reclaim enabled ({reclaim_mode})")
        else:
            self._log("[PORT] Auto reclaim disabled by WATCHDOG_RECLAIM_PORTS")
        if self._tcp_passive_probe:
            if self._has_passive_tcp_tools:
                self._log("[PORT] Passive TCP health probes enabled")
            else:
                self._log("[WARN] Passive TCP probes requested, but no lsof/ss; falling back to active connects")
        if self._cpu_throttle_supported:
            self._log(
                f"[CPU] cpufreq detected governor={self._cpu_current_governor or 'unknown'} "
                f"min={self._cpu_current_min_khz or '-'} max={self._cpu_current_max_khz or '-'}"
            )
        else:
            self._log("[WARN] CPU throttle control unavailable (cpufreq sysfs not detected)")
        if self._firmware_config_path:
            self._log(
                f"[CPU] firmware voltage config={self._firmware_config_path} "
                f"over_voltage_delta={self._cpu_current_over_voltage_delta_uv}uV"
            )
        else:
            self._log("[WARN] Firmware voltage config unavailable (cannot tune over_voltage_delta)")

    @staticmethod
    def _read_text_file(path: pathlib.Path) -> str:
        try:
            return str(path.read_text(encoding="utf-8")).strip()
        except Exception:
            return ""

    @staticmethod
    def _read_proc_stat_cpu() -> Tuple[int, int]:
        """
        Return (total_jiffies, idle_jiffies) from /proc/stat.
        """
        try:
            with open("/proc/stat", "r", encoding="utf-8") as fp:
                first = str(fp.readline() or "").strip()
            parts = first.split()
            if len(parts) < 5 or parts[0] != "cpu":
                return 0, 0
            values = [int(item) for item in parts[1:]]
            total = int(sum(values))
            idle = int(values[3]) + (int(values[4]) if len(values) > 4 else 0)
            return total, idle
        except Exception:
            return 0, 0

    @staticmethod
    def _read_proc_meminfo() -> Tuple[int, int]:
        """
        Return (mem_total_kb, mem_available_kb) from /proc/meminfo.
        """
        total = 0
        available = 0
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as fp:
                for raw in fp:
                    line = str(raw or "").strip()
                    if line.startswith("MemTotal:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            total = int(parts[1])
                    elif line.startswith("MemAvailable:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            available = int(parts[1])
                    if total > 0 and available > 0:
                        break
        except Exception:
            return 0, 0
        return int(total), int(available)

    @staticmethod
    def _parse_int(value, default: int = 0) -> int:
        try:
            return int(value)
        except Exception:
            return int(default)

    def _detect_firmware_config_path(self) -> Optional[pathlib.Path]:
        if self._os_name in ("windows", "darwin"):
            return None
        env_path = str(os.environ.get("WATCHDOG_FIRMWARE_CONFIG", "")).strip()
        candidates: List[pathlib.Path] = []
        if env_path:
            candidates.append(pathlib.Path(env_path).expanduser())
        for raw in FIRMWARE_CONFIG_CANDIDATES:
            candidates.append(pathlib.Path(str(raw)))

        for path in candidates:
            try:
                expanded = path.expanduser()
            except Exception:
                expanded = path
            if expanded.exists() and expanded.is_file():
                return expanded.resolve()
        for path in candidates:
            try:
                expanded = path.expanduser()
            except Exception:
                expanded = path
            parent = expanded.parent
            if parent.exists() and parent.is_dir():
                try:
                    return expanded.resolve()
                except Exception:
                    return expanded
        return None

    @staticmethod
    def _extract_over_voltage_delta_uv(text: str) -> int:
        value = 0
        for raw in str(text or "").splitlines():
            match = OVER_VOLTAGE_DELTA_RE.match(str(raw or ""))
            if not match:
                continue
            try:
                value = int(match.group(1))
            except Exception:
                continue
        return int(value)

    def _refresh_firmware_voltage_setting(self):
        cfg = self._firmware_config_path
        if cfg is None or not cfg.exists():
            self._cpu_current_over_voltage_delta_uv = 0
            return
        try:
            raw = cfg.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            self._cpu_current_over_voltage_delta_uv = 0
            return
        self._cpu_current_over_voltage_delta_uv = self._extract_over_voltage_delta_uv(raw)

    @staticmethod
    def _render_firmware_config_with_over_voltage_delta(raw_text: str, value_uv: int) -> str:
        target = int(value_uv)
        lines = str(raw_text or "").splitlines()
        output: List[str] = []
        seen = False
        for raw in lines:
            line = str(raw or "")
            if OVER_VOLTAGE_DELTA_RE.match(line):
                if not seen:
                    leading = re.match(r"^\s*", line)
                    prefix = leading.group(0) if leading else ""
                    output.append(f"{prefix}over_voltage_delta={target}")
                    seen = True
                continue
            output.append(line)
        if not seen:
            if output and output[-1].strip():
                output.append("")
            output.append(f"over_voltage_delta={target}")
        return "\n".join(output).rstrip("\n") + "\n"

    def _write_text_with_optional_sudo(self, path: pathlib.Path, content: str) -> Tuple[bool, str]:
        try:
            path.write_text(content, encoding="utf-8")
            return True, ""
        except PermissionError:
            pass
        except Exception as exc:
            return False, str(exc)

        if shutil.which("sudo") and self._os_name != "windows":
            try:
                result = subprocess.run(
                    ["sudo", "-n", "tee", str(path)],
                    input=str(content),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                    timeout=8.0,
                )
                if result.returncode == 0:
                    return True, ""
                detail = (result.stderr or "").strip() or "sudo rejected"
                return False, detail
            except Exception as exc:
                return False, str(exc)
        return False, "permission denied (run watchdog as root or configure passwordless sudo)"

    def _apply_over_voltage_delta_setting(self) -> Tuple[bool, str]:
        cfg = self._firmware_config_path
        if cfg is None:
            return False, "firmware config path unavailable"

        desired = int(self._cpu_throttle.get("over_voltage_delta_uv", DEFAULT_OVER_VOLTAGE_DELTA_UV) or 0)
        desired = max(MIN_OVER_VOLTAGE_DELTA_UV, min(MAX_OVER_VOLTAGE_DELTA_UV, desired))
        self._cpu_throttle["over_voltage_delta_uv"] = int(desired)

        try:
            if cfg.exists():
                raw = cfg.read_text(encoding="utf-8", errors="ignore")
            else:
                raw = ""
        except Exception as exc:
            return False, str(exc)

        current = self._extract_over_voltage_delta_uv(raw)
        if int(current) == int(desired):
            self._cpu_current_over_voltage_delta_uv = int(current)
            return True, f"over_voltage_delta={desired} (already set)"

        rendered = self._render_firmware_config_with_over_voltage_delta(raw, desired)
        ok, detail = self._write_text_with_optional_sudo(cfg, rendered)
        if not ok:
            return False, detail

        self._cpu_current_over_voltage_delta_uv = int(desired)
        return True, f"over_voltage_delta={desired} updated in {cfg} (reboot required)"

    def _refresh_cpu_throttle_support(self):
        cpufreq_paths = []
        for path in sorted(pathlib.Path("/").glob(CPUFREQ_GLOB.lstrip("/"))):
            if not path.exists() or not path.is_dir():
                continue
            if (path / "scaling_governor").exists():
                cpufreq_paths.append(path)
        self._cpu_cpufreq_paths = cpufreq_paths
        self._cpu_throttle_supported = bool(cpufreq_paths)
        if not self._cpu_throttle_supported:
            self._cpu_available_governors = []
            self._cpu_current_governor = ""
            self._cpu_current_min_khz = 0
            self._cpu_current_max_khz = 0
            self._cpu_hw_min_khz = 0
            self._cpu_hw_max_khz = 0
            return

        ref = cpufreq_paths[0]
        available = self._read_text_file(ref / "scaling_available_governors")
        self._cpu_available_governors = [token for token in available.split() if token]
        self._cpu_current_governor = self._read_text_file(ref / "scaling_governor")
        self._cpu_current_min_khz = self._parse_int(self._read_text_file(ref / "scaling_min_freq"), 0)
        self._cpu_current_max_khz = self._parse_int(self._read_text_file(ref / "scaling_max_freq"), 0)
        self._cpu_hw_min_khz = self._parse_int(self._read_text_file(ref / "cpuinfo_min_freq"), 0)
        self._cpu_hw_max_khz = self._parse_int(self._read_text_file(ref / "cpuinfo_max_freq"), 0)

        if not str(self._cpu_throttle.get("governor", "")).strip():
            self._cpu_throttle["governor"] = self._cpu_current_governor or "ondemand"
        if not int(self._cpu_throttle.get("min_khz", 0) or 0):
            self._cpu_throttle["min_khz"] = (
                self._cpu_current_min_khz or self._cpu_hw_min_khz or DEFAULT_CPU_MIN_KHZ
            )
        if not int(self._cpu_throttle.get("max_khz", 0) or 0):
            self._cpu_throttle["max_khz"] = (
                self._cpu_current_max_khz or self._cpu_hw_max_khz or DEFAULT_CPU_MAX_KHZ
            )
        if "over_voltage_delta_uv" not in self._cpu_throttle:
            self._cpu_throttle["over_voltage_delta_uv"] = DEFAULT_OVER_VOLTAGE_DELTA_UV
        self._normalize_cpu_throttle_settings()

    def _normalize_cpu_throttle_settings(self):
        governor = str(self._cpu_throttle.get("governor", "")).strip()
        if self._cpu_available_governors:
            if governor not in self._cpu_available_governors:
                if self._cpu_current_governor in self._cpu_available_governors:
                    governor = self._cpu_current_governor
                else:
                    governor = self._cpu_available_governors[0]
        elif not governor:
            governor = self._cpu_current_governor or "ondemand"
        self._cpu_throttle["governor"] = governor

        min_khz = self._parse_int(
            self._cpu_throttle.get("min_khz"),
            self._cpu_current_min_khz or DEFAULT_CPU_MIN_KHZ,
        )
        max_khz = self._parse_int(
            self._cpu_throttle.get("max_khz"),
            self._cpu_current_max_khz or DEFAULT_CPU_MAX_KHZ,
        )

        if self._cpu_hw_min_khz > 0:
            min_khz = max(self._cpu_hw_min_khz, min_khz)
            max_khz = max(self._cpu_hw_min_khz, max_khz)
        if self._cpu_hw_max_khz > 0:
            min_khz = min(self._cpu_hw_max_khz, min_khz)
            max_khz = min(self._cpu_hw_max_khz, max_khz)
        if min_khz > max_khz:
            min_khz, max_khz = max_khz, min_khz
        self._cpu_throttle["min_khz"] = int(max(100000, min_khz))
        self._cpu_throttle["max_khz"] = int(max(100000, max_khz))
        over_voltage_delta_uv = self._parse_int(
            self._cpu_throttle.get("over_voltage_delta_uv"),
            DEFAULT_OVER_VOLTAGE_DELTA_UV,
        )
        self._cpu_throttle["over_voltage_delta_uv"] = int(
            max(MIN_OVER_VOLTAGE_DELTA_UV, min(MAX_OVER_VOLTAGE_DELTA_UV, over_voltage_delta_uv))
        )
        self._cpu_throttle["auto_apply"] = bool(self._cpu_throttle.get("auto_apply", False))
        self._cpu_throttle["asr_llm_throttle_enable"] = bool(
            self._cpu_throttle.get("asr_llm_throttle_enable", True)
        )
        pct = self._parse_int(self._cpu_throttle.get("asr_llm_throttle_percent"), 65)
        self._cpu_throttle["asr_llm_throttle_percent"] = max(0, min(95, int(pct)))
        cycle_ms = self._parse_int(self._cpu_throttle.get("asr_llm_throttle_cycle_ms"), 320)
        self._cpu_throttle["asr_llm_throttle_cycle_ms"] = max(80, min(2000, int(cycle_ms)))

    def _apply_loaded_cpu_throttle(self, payload: dict):
        if not isinstance(payload, dict):
            return
        for key in (
            "governor",
            "min_khz",
            "max_khz",
            "over_voltage_delta_uv",
            "auto_apply",
            "asr_llm_throttle_enable",
            "asr_llm_throttle_percent",
            "asr_llm_throttle_cycle_ms",
        ):
            if key in payload:
                self._cpu_throttle[key] = payload.get(key)
        self._normalize_cpu_throttle_settings()

    def _apply_cpu_throttle_settings(self) -> Tuple[bool, str]:
        self._normalize_cpu_throttle_settings()
        details: List[str] = []
        failures: List[str] = []
        applicable_count = 0

        governor = str(self._cpu_throttle.get("governor", "")).strip()
        min_khz = int(self._cpu_throttle.get("min_khz", 0) or 0)
        max_khz = int(self._cpu_throttle.get("max_khz", 0) or 0)
        if min_khz > max_khz:
            min_khz, max_khz = max_khz, min_khz
            self._cpu_throttle["min_khz"] = min_khz
            self._cpu_throttle["max_khz"] = max_khz

        cpufreq_applicable = bool(self._cpu_throttle_supported and self._cpu_cpufreq_paths)
        if cpufreq_applicable:
            applicable_count += 1
            if min_khz <= 0 or max_khz <= 0:
                failures.append("cpufreq invalid min/max frequencies")
            else:
                def _write_all():
                    for cpufreq_dir in self._cpu_cpufreq_paths:
                        if governor:
                            (cpufreq_dir / "scaling_governor").write_text(governor, encoding="utf-8")
                        # set max first to avoid kernel rejecting new min > old max
                        (cpufreq_dir / "scaling_max_freq").write_text(str(max_khz), encoding="utf-8")
                        (cpufreq_dir / "scaling_min_freq").write_text(str(min_khz), encoding="utf-8")

                applied = False
                try:
                    _write_all()
                    applied = True
                except PermissionError:
                    applied = False
                except Exception as exc:
                    failures.append(f"cpufreq {exc}")
                    applied = False

                if not applied and not failures:
                    # Fallback through passwordless sudo when available.
                    if shutil.which("sudo") and self._os_name != "windows":
                        script = (
                            "set -euo pipefail; "
                            "for d in /sys/devices/system/cpu/cpu*/cpufreq; do "
                            f"if [ -f \"$d/scaling_governor\" ] && [ -n \"{governor}\" ]; then echo \"{governor}\" > \"$d/scaling_governor\"; fi; "
                            f"if [ -f \"$d/scaling_max_freq\" ]; then echo \"{max_khz}\" > \"$d/scaling_max_freq\"; fi; "
                            f"if [ -f \"$d/scaling_min_freq\" ]; then echo \"{min_khz}\" > \"$d/scaling_min_freq\"; fi; "
                            "done"
                        )
                        try:
                            result = subprocess.run(
                                ["sudo", "-n", "bash", "-lc", script],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True,
                                check=False,
                                timeout=8.0,
                            )
                            if result.returncode == 0:
                                applied = True
                            else:
                                detail = (result.stderr or result.stdout or "").strip() or "sudo rejected"
                                failures.append(f"cpufreq {detail}")
                        except Exception as exc:
                            failures.append(f"cpufreq {exc}")
                    else:
                        failures.append(
                            "cpufreq permission denied (run watchdog as root or configure passwordless sudo)"
                        )

                if applied:
                    self._refresh_cpu_throttle_support()
                    details.append(f"governor={governor} min={min_khz} max={max_khz}")
        else:
            details.append("cpufreq not available")

        undervolt_applicable = bool(self._firmware_config_path)
        if undervolt_applicable:
            applicable_count += 1
            uv_ok, uv_detail = self._apply_over_voltage_delta_setting()
            if uv_ok:
                details.append(uv_detail)
            else:
                failures.append(f"undervolt {uv_detail}")
        else:
            details.append("undervolt config unavailable")

        summary = "; ".join(item for item in (details + failures) if item).strip()
        if not summary:
            summary = "no changes"
        if applicable_count <= 0:
            return False, "cpufreq/undervolt controls unavailable on this host"
        return len(failures) == 0, summary

    def _load_desired_state(self) -> Dict[str, bool]:
        if not self.state_file.exists():
            self._state_file_last_mtime = 0.0
            return {}
        try:
            payload = json.loads(self.state_file.read_text(encoding="utf-8"))
        except Exception as exc:
            self._log(f"[WARN] Failed to read watchdog state: {exc}")
            try:
                self._state_file_last_mtime = float(self.state_file.stat().st_mtime)
            except Exception:
                self._state_file_last_mtime = 0.0
            return {}
        try:
            self._state_file_last_mtime = float(self.state_file.stat().st_mtime)
        except Exception:
            self._state_file_last_mtime = 0.0

        if isinstance(payload, dict):
            raw_services = payload.get("services", payload)
            cpu_payload = payload.get("cpu_throttle", None)
            if isinstance(cpu_payload, dict):
                self._apply_loaded_cpu_throttle(cpu_payload)
        else:
            raw_services = {}
        if not isinstance(raw_services, dict):
            return {}

        service_ids = set(self.service_by_id.keys())
        loaded = {}
        for service_id, value in raw_services.items():
            if service_id in service_ids and isinstance(value, bool):
                loaded[service_id] = value
        return loaded

    def _save_desired_state(self):
        payload = {
            "version": 2,
            "updated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "services": {},
            "cpu_throttle": {
                "governor": str(self._cpu_throttle.get("governor", "")).strip(),
                "min_khz": int(self._cpu_throttle.get("min_khz", 0) or 0),
                "max_khz": int(self._cpu_throttle.get("max_khz", 0) or 0),
                "over_voltage_delta_uv": int(
                    self._cpu_throttle.get("over_voltage_delta_uv", DEFAULT_OVER_VOLTAGE_DELTA_UV)
                    or DEFAULT_OVER_VOLTAGE_DELTA_UV
                ),
                "auto_apply": bool(self._cpu_throttle.get("auto_apply", False)),
                "asr_llm_throttle_enable": bool(self._cpu_throttle.get("asr_llm_throttle_enable", True)),
                "asr_llm_throttle_percent": int(self._cpu_throttle.get("asr_llm_throttle_percent", 65) or 65),
                "asr_llm_throttle_cycle_ms": int(self._cpu_throttle.get("asr_llm_throttle_cycle_ms", 320) or 320),
            },
        }
        for svc in self.services:
            runtime = self.runtime_by_id.get(svc.service_id)
            if runtime is None:
                continue
            payload["services"][svc.service_id] = bool(runtime.desired_enabled)

        tmp_path = self.state_file.with_suffix(".tmp")
        try:
            tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp_path.replace(self.state_file)
            try:
                self._state_file_last_mtime = float(self.state_file.stat().st_mtime)
            except Exception:
                pass
        except Exception as exc:
            self._log(f"[WARN] Failed to persist watchdog state: {exc}")

    def _maybe_reload_external_cpu_throttle(self, now: float):
        if (now - float(self._state_file_external_last_poll_at or 0.0)) < float(
            STATE_EXTERNAL_CPU_RELOAD_INTERVAL_SECONDS
        ):
            return
        self._state_file_external_last_poll_at = now

        if not self.state_file.exists():
            return
        try:
            mtime = float(self.state_file.stat().st_mtime)
        except Exception:
            return
        if mtime <= float(self._state_file_last_mtime or 0.0):
            return
        self._state_file_last_mtime = mtime

        try:
            payload = json.loads(self.state_file.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(payload, dict):
            return
        cpu_payload = payload.get("cpu_throttle", None)
        if not isinstance(cpu_payload, dict):
            return

        tracked_keys = (
            "governor",
            "min_khz",
            "max_khz",
            "over_voltage_delta_uv",
            "auto_apply",
            "asr_llm_throttle_enable",
            "asr_llm_throttle_percent",
            "asr_llm_throttle_cycle_ms",
        )
        previous = {key: self._cpu_throttle.get(key) for key in tracked_keys}
        self._apply_loaded_cpu_throttle(cpu_payload)
        current = {key: self._cpu_throttle.get(key) for key in tracked_keys}
        if previous == current:
            return

        ok, detail = self._apply_cpu_throttle_settings()
        self._cpu_throttle_last_apply = detail
        if ok:
            self._log(f"[CPU] External tuneables applied: {detail}")
        else:
            self._log(f"[WARN] External CPU tuneables apply failed: {detail}")

    # ------------------------------------------------------------------
    # Watchdog self-update helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _env_flag(name: str, default: bool = False) -> bool:
        fallback = "1" if default else "0"
        raw = str(os.environ.get(name, fallback)).strip().lower()
        return raw not in ("0", "false", "no", "off", "")

    @staticmethod
    def _normalize_git_url(url: str) -> str:
        text = str(url or "").strip().lower()
        if text.endswith(".git"):
            text = text[:-4]
        if text.startswith("git@github.com:"):
            text = "https://github.com/" + text.split(":", 1)[1]
        if text.startswith("ssh://git@github.com/"):
            text = "https://github.com/" + text.split("github.com/", 1)[1]
        return text.rstrip("/")

    def _run_git(
        self,
        *args: str,
        timeout_seconds: float = 20.0,
    ) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", *args],
            cwd=str(self.repo_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=max(1.0, float(timeout_seconds)),
        )

    def _git_output(
        self,
        *args: str,
        timeout_seconds: float = 20.0,
    ) -> Tuple[bool, str, str]:
        try:
            result = self._run_git(*args, timeout_seconds=timeout_seconds)
        except Exception as exc:
            return False, "", str(exc)
        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        return result.returncode == 0, stdout, stderr

    def _sync_runtime_from_repo(self) -> Tuple[bool, str]:
        sync_script = (self.repo_dir / "pi5" / "sync_runtime.sh").resolve()
        if not sync_script.exists():
            return False, f"missing sync script: {sync_script}"

        bash_bin = shutil.which("bash")
        if not bash_bin:
            return False, "bash executable not found"

        try:
            mode = sync_script.stat().st_mode
            sync_script.chmod(mode | 0o111)
        except Exception:
            pass

        try:
            result = subprocess.run(
                [bash_bin, str(sync_script), str(self.repo_dir), str(self.base_dir)],
                cwd=str(self.base_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=AUTO_UPDATE_RUNTIME_SYNC_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            return False, str(exc)

        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        if result.returncode != 0:
            detail = stderr or stdout or f"exit code {result.returncode}"
            return False, detail
        return True, stdout or "runtime synchronized"

    def _configure_auto_update(self):
        raw_enabled = str(os.environ.get("WATCHDOG_AUTO_UPDATE", "1")).strip().lower()
        if raw_enabled in ("0", "false", "no", "off"):
            self._log("[UPDATE] Auto-update disabled by WATCHDOG_AUTO_UPDATE")
            self._auto_update_enabled = False
            return

        if not shutil.which("git"):
            self._log("[UPDATE] Auto-update disabled (git executable not found)")
            self._auto_update_enabled = False
            return

        self._auto_update_remote = (
            str(os.environ.get("WATCHDOG_AUTO_UPDATE_REMOTE", AUTO_UPDATE_DEFAULT_REMOTE)).strip()
            or AUTO_UPDATE_DEFAULT_REMOTE
        )
        self._auto_update_branch = (
            str(os.environ.get("WATCHDOG_AUTO_UPDATE_BRANCH", AUTO_UPDATE_DEFAULT_BRANCH)).strip()
            or AUTO_UPDATE_DEFAULT_BRANCH
        )
        self._auto_update_repo_url = (
            str(os.environ.get("WATCHDOG_AUTO_UPDATE_URL", AUTO_UPDATE_DEFAULT_REPO_URL)).strip()
            or AUTO_UPDATE_DEFAULT_REPO_URL
        )

        raw_poll_seconds = str(
            os.environ.get("WATCHDOG_AUTO_UPDATE_POLL_SECONDS", str(AUTO_UPDATE_POLL_SECONDS))
        ).strip()
        try:
            self._auto_update_poll_seconds = max(5.0, float(raw_poll_seconds))
        except Exception:
            self._auto_update_poll_seconds = AUTO_UPDATE_POLL_SECONDS

        git_dir = self.repo_dir / ".git"
        if not git_dir.exists():
            self._log(f"[UPDATE] Auto-update disabled (no git metadata in {self.repo_dir})")
            self._auto_update_enabled = False
            return

        ok, _, err = self._git_output("rev-parse", "--is-inside-work-tree")
        if not ok:
            detail = err or "git checkout validation failed"
            self._log(f"[UPDATE] Auto-update disabled ({detail})")
            self._auto_update_enabled = False
            return

        if self._auto_update_repo_url:
            ok, remote_url, _ = self._git_output("remote", "get-url", self._auto_update_remote)
            if ok:
                expected = self._normalize_git_url(self._auto_update_repo_url)
                actual = self._normalize_git_url(remote_url)
                if expected and actual and expected != actual:
                    self._log(
                        f"[WARN] Auto-update remote mismatch: expected {self._auto_update_repo_url}, got {remote_url}"
                    )
        else:
            ok, remote_url, err = self._git_output("remote", "get-url", self._auto_update_remote)
            if not ok:
                detail = err or f"remote '{self._auto_update_remote}' not found"
                self._log(f"[UPDATE] Auto-update disabled ({detail})")
                self._auto_update_enabled = False
                return
            self._auto_update_repo_url = remote_url

        self._next_auto_update_check_at = time.time() + self._auto_update_poll_seconds
        self._auto_update_enabled = True
        source_label = self._auto_update_repo_url or self._auto_update_remote
        self._log(
            f"[UPDATE] Polling git {source_label} ({self._auto_update_branch}) "
            f"every {self._auto_update_poll_seconds:.0f}s"
        )

    def _request_watchdog_restart(self, reason: str):
        message = str(reason or "").strip() or "Restart requested"
        with self._lock:
            if self._restart_requested:
                return
            self._restart_requested = True
            self._restart_reason = message
            self._log(f"[UPDATE] {message}; restarting watchdog")

    @staticmethod
    def _dirty_paths_from_status(status_text: str, limit: int = 6) -> Tuple[List[str], int]:
        entries = []
        for raw in str(status_text or "").splitlines():
            line = str(raw or "").rstrip()
            if not line:
                continue
            path = line[3:].strip() if len(line) >= 4 else line.strip()
            if "->" in path:
                path = path.split("->", 1)[-1].strip()
            if path:
                entries.append(path)
        total = len(entries)
        if limit > 0 and total > limit:
            return entries[:limit], total
        return entries, total

    def _poll_for_auto_update(self):
        if not self._auto_update_enabled or self._restart_requested:
            return

        now = time.time()
        if now < self._next_auto_update_check_at:
            return
        self._next_auto_update_check_at = now + self._auto_update_poll_seconds

        fetch_source = self._auto_update_repo_url or self._auto_update_remote
        ok, _, err = self._git_output(
            "fetch",
            "--quiet",
            fetch_source,
            self._auto_update_branch,
            timeout_seconds=AUTO_UPDATE_FETCH_TIMEOUT_SECONDS,
        )
        if not ok:
            self._log(f"[WARN] Auto-update fetch failed: {err or 'unknown error'}")
            return

        ok, local_head, err = self._git_output("rev-parse", "HEAD")
        if not ok:
            self._log(f"[WARN] Auto-update skipped (failed to read local head: {err or 'unknown'})")
            return

        remote_ref = f"{self._auto_update_remote}/{self._auto_update_branch}"
        remote_target = "FETCH_HEAD" if self._auto_update_repo_url else remote_ref
        ok, remote_head, err = self._git_output("rev-parse", remote_target)
        if not ok:
            self._log(
                f"[WARN] Auto-update skipped (failed to read {remote_target}: {err or 'unknown'})"
            )
            return

        if local_head == remote_head:
            return

        try:
            ff_probe = self._run_git(
                "merge-base",
                "--is-ancestor",
                "HEAD",
                remote_target,
                timeout_seconds=10.0,
            )
        except Exception as exc:
            self._log(f"[WARN] Auto-update probe failed: {exc}")
            return
        if ff_probe.returncode != 0:
            self._log("[WARN] Auto-update skipped (local branch diverged from remote)")
            return

        ok, status_text, err = self._git_output(
            "status",
            "--porcelain",
            "--untracked-files=no",
        )
        if not ok:
            self._log(f"[WARN] Auto-update skipped (status check failed: {err or 'unknown'})")
            return
        if status_text:
            sample_paths, total_paths = self._dirty_paths_from_status(status_text)
            suffix = ", ".join(sample_paths)
            if total_paths > len(sample_paths):
                suffix = f"{suffix}, +{total_paths - len(sample_paths)} more" if suffix else f"+{total_paths} files"
            if not suffix:
                suffix = f"{total_paths} file(s)"

            if self._auto_update_stash_dirty:
                stash_label = datetime.datetime.now().strftime("watchdog-auto-update-%Y%m%d-%H%M%S")
                ok, stash_out, stash_err = self._git_output("stash", "push", "-m", stash_label)
                if not ok:
                    detail = stash_err or stash_out or "unknown stash error"
                    self._log(f"[WARN] Auto-update skipped (dirty tree and stash failed: {detail})")
                    return
                self._log(f"[UPDATE] Stashed {total_paths} local change(s) before pull ({suffix})")
            else:
                self._log(f"[WARN] Auto-update skipped (local tracked changes present: {suffix})")
                return

        ok, pull_out, pull_err = self._git_output(
            "pull",
            "--ff-only",
            fetch_source,
            self._auto_update_branch,
            timeout_seconds=AUTO_UPDATE_PULL_TIMEOUT_SECONDS,
        )
        if not ok:
            detail = pull_err or pull_out or "unknown error"
            self._log(f"[ERROR] Auto-update pull failed: {detail}")
            return

        lines = [line.strip() for line in pull_out.splitlines() if line.strip()]
        pull_summary = lines[-1] if lines else f"updated to {remote_head[:12]}"
        sync_ok, sync_detail = self._sync_runtime_from_repo()
        if not sync_ok:
            self._log(f"[ERROR] Runtime sync failed after pull: {sync_detail}")
            return
        if sync_detail:
            self._log(f"[UPDATE] Runtime sync completed: {sync_detail}")

        self._request_watchdog_restart(f"Pulled latest code ({pull_summary})")

    def should_restart(self) -> bool:
        with self._lock:
            return bool(self._restart_requested)

    def restart_reason(self) -> str:
        with self._lock:
            return str(self._restart_reason or "")

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _sanitize_log_line(message: str) -> str:
        text = str(message or "")
        text = text.replace("\r", " ").replace("\n", " ")
        text = re.sub(r"\x1B\[[0-?]*[ -/]*[@-~]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _log(self, message: str):
        clean = self._sanitize_log_line(message)
        if not clean:
            return
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self._logs.append(f"[{ts}] {clean}")

    def _state_color(self, state: str) -> int:
        if state == "running":
            return 2
        if state in ("launching", "activating", "degraded"):
            return 3
        if state == "stopping":
            return 4
        if state in ("error", "missing"):
            return 5
        return 1

    def _set_state(
        self,
        svc: ServiceSpec,
        runtime: ServiceRuntime,
        new_state: str,
        now: float,
        event: Optional[str] = None,
        error: Optional[str] = None,
    ):
        previous = runtime.state
        runtime.state = str(new_state or runtime.state)
        runtime.last_state_change_at = now
        if event is not None:
            runtime.last_event = event
        if error is not None:
            runtime.last_error = error
        if previous != runtime.state:
            self._log(f"[STATE] {svc.label}: {previous} -> {runtime.state}")

    def _clear_runtime_timers(self, runtime: ServiceRuntime):
        runtime.launch_grace_until = 0.0
        runtime.launch_attempt_started_at = 0.0
        runtime.activation_deadline = 0.0
        runtime.process_stable_since = 0.0
        runtime.last_health_probe_at = 0.0
        runtime.last_health_ok_at = 0.0
        runtime.last_port_discovery_at = 0.0

    def _reset_runtime_health(self, runtime: ServiceRuntime):
        runtime.consecutive_health_failures = 0
        runtime.last_health_error = ""

    @staticmethod
    def _aggregate_usage_for_pid(
        root_pid: int,
        usage_by_pid: Dict[int, Tuple[float, float]],
        children_by_pid: Dict[int, List[int]],
    ) -> Tuple[float, float]:
        if not root_pid or root_pid <= 0:
            return 0.0, 0.0
        cpu_total = 0.0
        rss_total = 0.0
        stack = [int(root_pid)]
        seen = set()
        while stack:
            pid = int(stack.pop())
            if pid in seen:
                continue
            seen.add(pid)
            sample = usage_by_pid.get(pid)
            if sample:
                cpu_total += float(sample[0])
                rss_total += float(sample[1])
            for child in children_by_pid.get(pid, []):
                if child not in seen:
                    stack.append(int(child))
        return cpu_total, rss_total

    @staticmethod
    def _aggregate_usage_for_pids(
        root_pids: Sequence[int],
        usage_by_pid: Dict[int, Tuple[float, float]],
        children_by_pid: Dict[int, List[int]],
    ) -> Tuple[float, float]:
        cpu_total = 0.0
        rss_total = 0.0
        stack = [int(pid) for pid in root_pids if int(pid or 0) > 0]
        seen = set()
        while stack:
            pid = int(stack.pop())
            if pid in seen:
                continue
            seen.add(pid)
            sample = usage_by_pid.get(pid)
            if sample:
                cpu_total += float(sample[0])
                rss_total += float(sample[1])
            for child in children_by_pid.get(pid, []):
                child_pid = int(child or 0)
                if child_pid > 0 and child_pid not in seen:
                    stack.append(child_pid)
        return cpu_total, rss_total

    def _update_system_resource_snapshot(self):
        total, idle = self._read_proc_stat_cpu()
        if total > 0 and idle >= 0:
            prev_total = int(self._proc_cpu_prev_total or 0)
            prev_idle = int(self._proc_cpu_prev_idle or 0)
            if prev_total > 0 and total > prev_total:
                delta_total = float(total - prev_total)
                delta_idle = float(max(0, idle - prev_idle))
                if delta_total > 0.0:
                    busy_ratio = max(0.0, min(1.0, 1.0 - (delta_idle / delta_total)))
                    self._system_cpu_percent = busy_ratio * 100.0
            self._proc_cpu_prev_total = int(total)
            self._proc_cpu_prev_idle = int(idle)

        mem_total_kb, mem_available_kb = self._read_proc_meminfo()
        if mem_total_kb > 0:
            self._system_mem_total_kb = int(mem_total_kb)
            used_kb = max(0, int(mem_total_kb - max(0, mem_available_kb)))
            self._system_ram_percent = (float(used_kb) / float(mem_total_kb)) * 100.0

        self._resource_cpu_history.append(max(0.0, min(100.0, float(self._system_cpu_percent))))
        self._resource_ram_history.append(max(0.0, min(100.0, float(self._system_ram_percent))))

    def _update_resource_usage(self, now: float):
        if self._os_name == "windows":
            return
        if self._last_resource_sample_at and (
            now - self._last_resource_sample_at
        ) < self._resource_sample_interval:
            return
        self._last_resource_sample_at = now
        if not shutil.which("ps"):
            self._update_system_resource_snapshot()
            return

        try:
            result = subprocess.run(
                ["ps", "-eo", "pid=,ppid=,pcpu=,rss=,args="],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                check=False,
                timeout=2.5,
            )
        except Exception:
            return

        usage_by_pid: Dict[int, Tuple[float, float]] = {}
        children_by_pid: Dict[int, List[int]] = {}
        args_by_pid: Dict[int, str] = {}
        for raw in (result.stdout or "").splitlines():
            line = raw.strip()
            if not line:
                continue
            parts = line.split(None, 4)
            if len(parts) < 4:
                continue
            try:
                pid = int(parts[0])
                ppid = int(parts[1])
                cpu = float(str(parts[2]).replace(",", "."))
                rss_kb = float(parts[3])
            except Exception:
                continue
            cmdline = str(parts[4]).strip() if len(parts) > 4 else ""
            usage_by_pid[pid] = (cpu, rss_kb)
            args_by_pid[pid] = cmdline
            if ppid > 0:
                children_by_pid.setdefault(ppid, []).append(pid)

        for svc in self.services:
            runtime = self.runtime_by_id[svc.service_id]
            root_pids: List[int] = []
            if runtime.pid and runtime.pid > 0:
                root_pids.append(int(runtime.pid))

            # Wrapper services may host the real worker under a different pid.
            candidate_port = int(
                (runtime.resolved_health_port or 0)
                or (svc.resolved_health_port(self.base_dir) or 0)
                or (svc.health_port or 0)
            )
            if candidate_port > 0:
                for pid in self._list_listening_pids_for_port(candidate_port):
                    if int(pid or 0) > 0:
                        root_pids.append(int(pid))

            # LLM Bridge load is mostly on Ollama runners; include those while LLM is busy.
            llm_state = str(self._pipeline_stage_state.get("llm") or "idle").strip().lower()
            llm_busy = llm_state in ("queued", "processing", "streaming", "model_switching")
            if svc.service_id == "llm_bridge" and llm_busy:
                for pid, cmdline in args_by_pid.items():
                    cmd = str(cmdline or "").lower()
                    if "ollama runner" in cmd:
                        root_pids.append(int(pid))
                for pid in self._list_listening_pids_for_port(11434):
                    if int(pid or 0) > 0:
                        root_pids.append(int(pid))

            # Ensure Ollama service row reflects active runner subprocesses too.
            if svc.service_id == "ollama":
                for pid, cmdline in args_by_pid.items():
                    cmd = str(cmdline or "").lower()
                    if "ollama serve" in cmd or "ollama runner" in cmd:
                        root_pids.append(int(pid))
                for pid in self._list_listening_pids_for_port(11434):
                    if int(pid or 0) > 0:
                        root_pids.append(int(pid))

            cpu_total, rss_total = self._aggregate_usage_for_pids(
                root_pids,
                usage_by_pid,
                children_by_pid,
            )

            runtime.cpu_percent = max(0.0, float(cpu_total))
            runtime.rss_mb = max(0.0, float(rss_total) / 1024.0)

            if runtime.cpu_percent > runtime.peak_cpu_percent:
                runtime.peak_cpu_percent = runtime.cpu_percent
                runtime.peak_cpu_at = now
            if runtime.rss_mb > runtime.peak_rss_mb:
                runtime.peak_rss_mb = runtime.rss_mb
                runtime.peak_rss_at = now

        self._update_system_resource_snapshot()

    def _health_probe(self, svc: ServiceSpec) -> Tuple[bool, str]:
        return self._health_probe_with_runtime(svc, None)

    def _build_health_target(self, svc: ServiceSpec, port_override: int = 0) -> str:
        mode = str(svc.health_mode or "process").strip().lower()
        if mode == "process":
            return "process"
        port = int(port_override or 0)
        if port <= 0:
            port = svc.resolved_health_port(self.base_dir)
        if port <= 0:
            return ""
        if mode == "tcp":
            return f"127.0.0.1:{port}"
        path = str(svc.health_path or "").strip()
        if not path:
            path = "/"
        if not path.startswith("/"):
            path = f"/{path}"
        return f"http://127.0.0.1:{port}{path}"

    def _probe_target(self, mode: str, target: str) -> Tuple[bool, str]:
        mode = str(mode or "process").strip().lower()
        if mode == "process":
            return True, "process-only probe"
        if not target:
            return False, "missing probe target"
        if mode == "tcp":
            try:
                host, port_text = target.rsplit(":", 1)
                port = int(port_text)
            except Exception:
                return False, f"invalid tcp target {target!r}"
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(HTTP_PROBE_TIMEOUT_SECONDS)
                try:
                    ok = sock.connect_ex((host, port)) == 0
                except Exception as exc:
                    return False, str(exc)
            return (ok, "tcp open" if ok else "tcp closed")

        # Default HTTP probe.
        request = urllib.request.Request(target, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=HTTP_PROBE_TIMEOUT_SECONDS) as response:
                code = int(getattr(response, "status", 200) or 200)
            if 200 <= code < 400:
                return True, f"http {code}"
            return False, f"http {code}"
        except urllib.error.HTTPError as exc:
            code = int(getattr(exc, "code", 0) or 0)
            # 401/403 still means the service is up; activation should be considered successful.
            if code in (401, 403):
                return True, f"http {code} (auth required)"
            return False, f"http {code}"
        except Exception as exc:
            return False, str(exc)

    def _list_listening_ports_for_pid(self, pid: int) -> List[int]:
        if not pid or pid <= 0:
            return []
        ports = set()
        if self._os_name == "windows":
            try:
                result = subprocess.run(
                    ["netstat", "-ano", "-p", "tcp"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    check=False,
                    timeout=2.5,
                )
                for raw in (result.stdout or "").splitlines():
                    line = raw.strip()
                    if not line or not line.upper().startswith("TCP"):
                        continue
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    state = str(parts[3]).strip().upper()
                    if state != "LISTENING":
                        continue
                    try:
                        line_pid = int(parts[4])
                    except Exception:
                        continue
                    if line_pid != int(pid):
                        continue
                    local = str(parts[1]).strip()
                    if local.startswith("[") and "]:" in local:
                        port_text = local.rsplit("]:", 1)[-1]
                    else:
                        port_text = local.rsplit(":", 1)[-1]
                    try:
                        port = int(port_text)
                    except Exception:
                        continue
                    if 1 <= port <= 65535:
                        ports.add(port)
            except Exception:
                return []
            return sorted(ports)

        # POSIX fallback via lsof when available.
        if shutil.which("lsof"):
            try:
                result = subprocess.run(
                    ["lsof", "-Pan", "-p", str(int(pid)), "-iTCP", "-sTCP:LISTEN"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    check=False,
                    timeout=2.5,
                )
                for raw in (result.stdout or "").splitlines():
                    match = re.search(r":(\d+)\s+\(LISTEN\)", raw)
                    if not match:
                        continue
                    try:
                        port = int(match.group(1))
                    except Exception:
                        continue
                    if 1 <= port <= 65535:
                        ports.add(port)
            except Exception:
                pass
        if not ports and shutil.which("ss"):
            try:
                result = subprocess.run(
                    ["ss", "-ltnp"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    check=False,
                    timeout=2.5,
                )
                pid_pattern = re.compile(r"pid=(\d+)")
                for raw in (result.stdout or "").splitlines():
                    line = raw.strip()
                    if not line or "LISTEN" not in line.upper():
                        continue
                    pids = [int(m.group(1)) for m in pid_pattern.finditer(line) if m.group(1)]
                    if not pids or int(pid) not in pids:
                        continue
                    parts = line.split()
                    if len(parts) < 4:
                        continue
                    local = str(parts[3]).strip()
                    if local.startswith("[") and "]:" in local:
                        port_text = local.rsplit("]:", 1)[-1]
                    else:
                        port_text = local.rsplit(":", 1)[-1]
                    try:
                        port = int(port_text)
                    except Exception:
                        continue
                    if 1 <= port <= 65535:
                        ports.add(port)
            except Exception:
                pass
        return sorted(ports)

    def _list_listening_pids_for_port(self, port: int) -> List[int]:
        try:
            port_value = int(port)
        except Exception:
            return []
        if port_value <= 0 or port_value > 65535:
            return []

        pids = set()
        if self._os_name == "windows":
            try:
                result = subprocess.run(
                    ["netstat", "-ano", "-p", "tcp"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    check=False,
                    timeout=2.5,
                )
                for raw in (result.stdout or "").splitlines():
                    line = raw.strip()
                    if not line or not line.upper().startswith("TCP"):
                        continue
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    state = str(parts[3]).strip().upper()
                    if state != "LISTENING":
                        continue
                    local = str(parts[1]).strip()
                    if local.startswith("[") and "]:" in local:
                        local_port = local.rsplit("]:", 1)[-1]
                    else:
                        local_port = local.rsplit(":", 1)[-1]
                    try:
                        line_port = int(local_port)
                    except Exception:
                        continue
                    if line_port != port_value:
                        continue
                    try:
                        pid = int(parts[4])
                    except Exception:
                        continue
                    if pid > 0:
                        pids.add(pid)
            except Exception:
                pass
            return sorted(pids)

        if shutil.which("lsof"):
            try:
                result = subprocess.run(
                    ["lsof", "-nP", f"-iTCP:{port_value}", "-sTCP:LISTEN", "-t"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    check=False,
                    timeout=2.5,
                )
                for raw in (result.stdout or "").splitlines():
                    try:
                        pid = int(raw.strip())
                    except Exception:
                        continue
                    if pid > 0:
                        pids.add(pid)
            except Exception:
                pass
            if pids:
                return sorted(pids)

        if shutil.which("ss"):
            try:
                result = subprocess.run(
                    ["ss", "-ltnp", "sport", "=", f":{port_value}"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    check=False,
                    timeout=2.5,
                )
                for raw in (result.stdout or "").splitlines():
                    for match in re.findall(r"pid=(\d+)", raw):
                        try:
                            pid = int(match)
                        except Exception:
                            continue
                        if pid > 0:
                            pids.add(pid)
            except Exception:
                pass
            if pids:
                return sorted(pids)

        if shutil.which("netstat"):
            try:
                result = subprocess.run(
                    ["netstat", "-ltnp"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    check=False,
                    timeout=2.5,
                )
                pattern = re.compile(rf":{port_value}\s+.*LISTEN\s+(\d+)/")
                for raw in (result.stdout or "").splitlines():
                    match = pattern.search(raw)
                    if not match:
                        continue
                    try:
                        pid = int(match.group(1))
                    except Exception:
                        continue
                    if pid > 0:
                        pids.add(pid)
            except Exception:
                pass

        return sorted(pids)

    def _read_process_commandline(self, pid: int) -> str:
        if not pid or pid <= 0:
            return ""
        if self._os_name == "windows":
            try:
                ps_script = (
                    f'$p=Get-CimInstance Win32_Process -Filter "ProcessId={int(pid)}"; '
                    "if ($p) { $p.CommandLine }"
                )
                result = subprocess.run(
                    ["powershell", "-NoProfile", "-Command", ps_script],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    check=False,
                    timeout=2.5,
                )
                text = (result.stdout or "").strip()
                if text:
                    return text
            except Exception:
                pass
            try:
                result = subprocess.run(
                    ["wmic", "process", "where", f"processid={int(pid)}", "get", "CommandLine", "/value"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    check=False,
                    timeout=2.5,
                )
                for raw in (result.stdout or "").splitlines():
                    line = raw.strip()
                    if line.lower().startswith("commandline="):
                        return line.split("=", 1)[1].strip()
            except Exception:
                pass
            return ""

        proc_cmdline = pathlib.Path(f"/proc/{int(pid)}/cmdline")
        try:
            if proc_cmdline.exists():
                raw = proc_cmdline.read_bytes()
                text = raw.replace(b"\x00", b" ").decode("utf-8", errors="ignore").strip()
                if text:
                    return text
        except Exception:
            pass
        try:
            result = subprocess.run(
                ["ps", "-p", str(int(pid)), "-o", "command="],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                check=False,
                timeout=2.5,
            )
            return (result.stdout or "").strip()
        except Exception:
            return ""

    def _list_ollama_related_pids(self) -> List[int]:
        pids = set()
        for pid in self._list_listening_pids_for_port(11434):
            try:
                parsed = int(pid)
            except Exception:
                continue
            if parsed > 0 and parsed != os.getpid():
                pids.add(parsed)

        ollama_runtime = self.runtime_by_id.get("ollama")
        if ollama_runtime and ollama_runtime.pid:
            try:
                parsed = int(ollama_runtime.pid)
            except Exception:
                parsed = 0
            if parsed > 0 and parsed != os.getpid():
                pids.add(parsed)

        if self._os_name == "windows":
            return sorted(pids)

        if not shutil.which("ps"):
            return sorted(pids)

        try:
            result = subprocess.run(
                ["ps", "-eo", "pid=,args="],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                check=False,
                timeout=2.5,
            )
            for raw in (result.stdout or "").splitlines():
                line = str(raw or "").strip()
                if not line:
                    continue
                parts = line.split(None, 1)
                if len(parts) < 2:
                    continue
                try:
                    pid = int(parts[0])
                except Exception:
                    continue
                cmd = str(parts[1] or "").strip().lower()
                if not cmd:
                    continue
                if "ollama serve" in cmd or "ollama runner" in cmd:
                    if pid > 0 and pid != os.getpid():
                        pids.add(pid)
        except Exception:
            pass

        return sorted(pids)

    def _force_stop_ollama_processes(self, reason: str) -> int:
        targets = self._list_ollama_related_pids()
        if not targets:
            return 0

        why = str(reason or "").strip() or "requested by user"
        self._log(
            f"[FORCE] Stopping Ollama process(es) ({why}): "
            f"{','.join(str(pid) for pid in targets)}"
        )

        for pid in targets:
            self._send_signal(int(pid), signal.SIGTERM)

        term_deadline = time.time() + max(0.1, float(OLLAMA_HARD_STOP_TERM_WAIT_SECONDS))
        while time.time() < term_deadline:
            remaining = [pid for pid in targets if self._is_pid_running(int(pid))]
            if not remaining:
                break
            time.sleep(0.08)

        remaining = [pid for pid in targets if self._is_pid_running(int(pid))]
        for pid in remaining:
            self._send_signal(int(pid), signal.SIGKILL)

        kill_deadline = time.time() + max(0.1, float(OLLAMA_HARD_STOP_KILL_WAIT_SECONDS))
        while time.time() < kill_deadline:
            survivors = [pid for pid in targets if self._is_pid_running(int(pid))]
            if not survivors:
                break
            time.sleep(0.08)

        survivors = [pid for pid in targets if self._is_pid_running(int(pid))]
        stopped = len(targets) - len(survivors)
        if survivors:
            self._log(
                f"[WARN] Ollama process(es) still alive after forced stop: "
                f"{','.join(str(pid) for pid in survivors)}"
            )
        else:
            self._log("[FORCE] Ollama process stop complete")

        ollama_runtime = self.runtime_by_id.get("ollama")
        if ollama_runtime and ollama_runtime.pid and int(ollama_runtime.pid) in targets:
            if not self._is_pid_running(int(ollama_runtime.pid)):
                ollama_runtime.pid = None
                if ollama_runtime.terminal_process and not self._is_process_handle_running(
                    ollama_runtime.terminal_process
                ):
                    ollama_runtime.terminal_process = None
                ollama_runtime.process_stable_since = 0.0
                ollama_runtime.stop_stage = 0
                ollama_runtime.stop_requested_at = 0.0
                ollama_runtime.resolved_health_port = 0

        # Refresh the resource rows right away so the UI reflects the CPU drop.
        self._last_resource_sample_at = 0.0
        self._update_resource_usage(time.time())

        return int(max(0, stopped))

    def _pid_likely_owned_by_service(self, pid: int, svc: ServiceSpec) -> bool:
        if not pid or pid <= 0:
            return False
        if int(pid) == int(os.getpid()):
            return False
        cmdline = self._read_process_commandline(int(pid))
        if not cmdline:
            return False
        normalized = str(cmdline).replace("\\", "/").lower()
        script_abs = str(svc.script_path(self.base_dir)).replace("\\", "/").lower()
        script_rel = str(svc.script_relpath or "").replace("\\", "/").lower()
        script_name = pathlib.Path(script_abs).name
        if script_abs and script_abs in normalized:
            return True
        if script_rel and script_rel in normalized:
            return True
        if script_name and (
            normalized.endswith(script_name)
            or f"/{script_name}" in normalized
            or f" {script_name} " in normalized
        ):
            return True
        return False

    def _wait_for_pid_exit(self, pid: int, timeout_seconds: float) -> bool:
        deadline = time.time() + max(0.1, float(timeout_seconds))
        while time.time() < deadline:
            if not self._is_pid_running(int(pid)):
                return True
            time.sleep(0.1)
        return not self._is_pid_running(int(pid))

    def _terminate_pid_for_port_reclaim(self, pid: int) -> bool:
        if not pid or pid <= 0:
            return True
        if int(pid) == int(os.getpid()):
            return False
        if not self._is_pid_running(int(pid)):
            return True
        self._send_signal(int(pid), signal.SIGTERM)
        if self._wait_for_pid_exit(int(pid), PORT_RECLAIM_TERM_WAIT_SECONDS):
            return True
        self._send_signal(int(pid), signal.SIGKILL)
        return self._wait_for_pid_exit(int(pid), PORT_RECLAIM_KILL_WAIT_SECONDS)

    def _candidate_service_ports(self, svc: ServiceSpec, runtime: Optional[ServiceRuntime]) -> List[int]:
        mode = str(svc.health_mode or "process").strip().lower()
        if mode == "process":
            return []
        raw_candidates = []
        if runtime is not None:
            raw_candidates.append(getattr(runtime, "resolved_health_port", 0))
        raw_candidates.append(svc.resolved_health_port(self.base_dir))
        raw_candidates.append(getattr(svc, "health_port", 0))

        ports = set()
        for value in raw_candidates:
            try:
                port = int(value)
            except Exception:
                continue
            if 1 <= port <= 65535:
                ports.add(port)
        return sorted(ports)

    def _ensure_service_ports_available_for_launch(
        self,
        svc: ServiceSpec,
        runtime: ServiceRuntime,
    ) -> Tuple[bool, str]:
        ports = self._candidate_service_ports(svc, runtime)
        if not ports:
            return True, ""

        for port in ports:
            observed = [pid for pid in self._list_listening_pids_for_port(port) if pid != os.getpid()]
            if not observed:
                continue

            owned = []
            foreign = []
            for pid in observed:
                if runtime.pid and int(pid) == int(runtime.pid):
                    owned.append(int(pid))
                elif self._pid_likely_owned_by_service(int(pid), svc):
                    owned.append(int(pid))
                else:
                    foreign.append(int(pid))

            reclaim_targets = []
            if self._port_reclaim_enabled:
                reclaim_targets.extend(owned)
                if self._port_reclaim_force:
                    reclaim_targets.extend(foreign)
            reclaim_targets = list(dict.fromkeys(reclaim_targets))

            if reclaim_targets:
                self._log(
                    f"[PORT] {svc.label} reclaiming port {port} from pid(s) "
                    f"{','.join(str(pid) for pid in reclaim_targets)}"
                )
                for pid in reclaim_targets:
                    if self._terminate_pid_for_port_reclaim(pid):
                        self._log(f"[PORT] {svc.label} reclaimed port {port} from pid={pid}")
                    else:
                        self._log(f"[WARN] {svc.label} failed to terminate pid={pid} on port {port}")

            remaining = [pid for pid in self._list_listening_pids_for_port(port) if pid != os.getpid()]
            if not remaining:
                continue

            reason = (
                f"port {port} in use by pid(s) {','.join(str(pid) for pid in remaining)}"
            )
            if foreign and not self._port_reclaim_force:
                reason += " (foreign process; set WATCHDOG_RECLAIM_FORCE=1 to force)"
            return False, reason

        return True, ""

    def _discover_runtime_health_port(
        self,
        svc: ServiceSpec,
        runtime: Optional[ServiceRuntime],
        attempted_port: int,
    ) -> int:
        if runtime is None:
            return 0
        pid = int(runtime.pid or 0)
        if pid <= 0 or not self._is_pid_running(pid):
            return 0

        discovered_ports = self._list_listening_ports_for_pid(pid)
        if not discovered_ports:
            return 0

        preferred = int(svc.resolved_health_port(self.base_dir) or 0)
        current = int(runtime.resolved_health_port or 0)
        attempted = int(attempted_port or 0)

        def _rank(port_value: int):
            delta = abs(port_value - preferred) if preferred else 0
            # Favor previously discovered/current port, then default/config-adjacent ports.
            return (
                0 if current and port_value == current else 1,
                0 if preferred and port_value == preferred else 1,
                0 if preferred and delta <= PORT_DISCOVERY_MAX_DELTA else 1,
                delta,
                port_value,
            )

        mode = str(svc.health_mode or "process").strip().lower()
        for port in sorted(discovered_ports, key=_rank)[:PORT_DISCOVERY_SCAN_LIMIT]:
            if port == attempted:
                continue
            target = self._build_health_target(svc, port_override=port)
            ok, _detail = self._probe_target(mode, target)
            if ok:
                return port
        return 0

    def _health_probe_with_runtime(
        self,
        svc: ServiceSpec,
        runtime: Optional[ServiceRuntime],
    ) -> Tuple[bool, str]:
        mode = str(svc.health_mode or "process").strip().lower()
        if mode == "process":
            return True, "process-only probe"

        preferred_port = int(svc.resolved_health_port(self.base_dir) or 0)
        active_port = int((runtime.resolved_health_port if runtime else 0) or preferred_port or 0)

        # Prefer passive TCP health probing by inspecting PID listen sockets.
        # This avoids creating loopback connections that spam service logs.
        if (
            mode == "tcp"
            and self._tcp_passive_probe
            and runtime is not None
            and runtime.pid
            and runtime.pid > 0
        ):
            listening_ports = self._list_listening_ports_for_pid(int(runtime.pid))
            if listening_ports:
                selected = 0
                for candidate in (
                    active_port,
                    preferred_port,
                    int(svc.health_port or 0),
                ):
                    if candidate > 0 and candidate in listening_ports:
                        selected = candidate
                        break
                if selected <= 0:
                    if preferred_port > 0:
                        selected = min(listening_ports, key=lambda p: (abs(p - preferred_port), p))
                    else:
                        selected = int(listening_ports[0])

                if runtime.resolved_health_port != selected:
                    previous = int(runtime.resolved_health_port or 0)
                    runtime.resolved_health_port = selected
                    runtime.last_port_discovery_at = time.time()
                    if previous > 0 and previous != selected:
                        self._log(f"[DISCOVER] {svc.label} health probe realigned to port {selected}")
                return True, f"tcp listen {selected}"

            # Wrapper-based services (for example docker-run launchers) may keep the
            # control process alive while the listening socket belongs to another pid.
            # If the expected port is listening at all, treat it as healthy.
            for candidate in (
                active_port,
                preferred_port,
                int(svc.health_port or 0),
            ):
                if candidate <= 0:
                    continue
                pids_on_port = self._list_listening_pids_for_port(candidate)
                if not pids_on_port:
                    continue
                if runtime.resolved_health_port != candidate:
                    previous = int(runtime.resolved_health_port or 0)
                    runtime.resolved_health_port = candidate
                    runtime.last_port_discovery_at = time.time()
                    if previous > 0 and previous != candidate:
                        self._log(f"[DISCOVER] {svc.label} health probe realigned to port {candidate}")
                return True, f"tcp listen {candidate}"

            # Passive inspection can miss listeners owned by other users/namespaces
            # (for example docker host-network processes). Fall back to active probe.

        target = self._build_health_target(svc, port_override=active_port)
        ok, detail = self._probe_target(mode, target)
        if ok:
            if runtime is not None and active_port > 0:
                runtime.resolved_health_port = active_port
            return True, detail

        discovered_port = self._discover_runtime_health_port(svc, runtime, active_port)
        if discovered_port > 0 and discovered_port != active_port:
            discovered_target = self._build_health_target(svc, port_override=discovered_port)
            discovered_ok, discovered_detail = self._probe_target(mode, discovered_target)
            if discovered_ok:
                if runtime is not None:
                    runtime.resolved_health_port = discovered_port
                    runtime.last_port_discovery_at = time.time()
                self._log(f"[DISCOVER] {svc.label} health probe realigned to port {discovered_port}")
                return True, f"{discovered_detail} (port {discovered_port})"

        return False, detail

    def _schedule_restart(self, runtime: ServiceRuntime, now: float):
        runtime.next_restart_at = now + runtime.restart_backoff_seconds
        runtime.restart_backoff_seconds = min(
            RESTART_BACKOFF_MAX_SECONDS, runtime.restart_backoff_seconds * 2.0
        )

    def _mark_launch_failure(
        self,
        svc: ServiceSpec,
        runtime: ServiceRuntime,
        now: float,
        reason: str,
        increment_crash: bool = False,
    ):
        runtime.activation_failures += 1
        runtime.health_failures += 1
        runtime.last_health_error = reason
        if increment_crash:
            runtime.crash_count += 1
        runtime.pid = None
        runtime.terminal_process = None
        runtime.resolved_health_port = 0
        runtime.started_at = 0.0
        runtime.stopped_at = now
        self._clear_runtime_timers(runtime)
        self._set_state(svc, runtime, "error", now, event="Waiting to restart", error=reason)
        self._schedule_restart(runtime, now)

    # ------------------------------------------------------------------
    # PID and signaling helpers
    # ------------------------------------------------------------------
    def _service_log_file(self, svc: ServiceSpec) -> pathlib.Path:
        return self.service_log_dir / f"{svc.service_id}.log"

    def _open_service_log_handle(self, svc: ServiceSpec):
        self.service_log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self._service_log_file(svc)
        try:
            handle = open(log_path, "ab", buffering=0)
            stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            header = (
                f"\n[{stamp}] [WATCHDOG] Launching {svc.label} "
                f"(service_id={svc.service_id})\n"
            ).encode("utf-8", errors="replace")
            handle.write(header)
            return handle
        except Exception as exc:
            self._log(f"[WARN] {svc.label} log capture unavailable: {exc}")
            return None

    def _pid_file(self, svc: ServiceSpec) -> pathlib.Path:
        return self.runtime_dir / f"{svc.service_id}.pid"

    def _read_pid_file(self, svc: ServiceSpec) -> Optional[int]:
        path = self._pid_file(svc)
        if not path.exists():
            return None
        try:
            raw = path.read_text(encoding="utf-8").strip()
            pid = int(raw)
            return pid if pid > 0 else None
        except Exception:
            return None

    def _write_pid_file(self, svc: ServiceSpec, pid: int):
        path = self._pid_file(svc)
        try:
            path.write_text(str(int(pid)), encoding="utf-8")
        except Exception:
            pass

    def _remove_pid_file(self, svc: ServiceSpec):
        path = self._pid_file(svc)
        try:
            if path.exists():
                path.unlink()
        except Exception:
            pass

    def _acquire_instance_lock(self):
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        handle = None
        existing_pid = None
        try:
            handle = open(self.lock_file, "a+", encoding="utf-8")
            handle.seek(0)

            if os.name == "nt":
                import msvcrt

                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                except OSError:
                    handle.seek(0)
                    raw = handle.read().strip()
                    try:
                        existing_pid = int(raw) if raw else None
                    except Exception:
                        existing_pid = None
                    suffix = f" (pid {existing_pid})" if existing_pid else ""
                    raise RuntimeError(
                        f"Another watchdog instance is already running{suffix}"
                    )
            else:
                import fcntl

                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    handle.seek(0)
                    raw = handle.read().strip()
                    try:
                        existing_pid = int(raw) if raw else None
                    except Exception:
                        existing_pid = None
                    suffix = f" (pid {existing_pid})" if existing_pid else ""
                    raise RuntimeError(
                        f"Another watchdog instance is already running{suffix}"
                    )

            handle.seek(0)
            handle.truncate()
            handle.write(str(os.getpid()))
            handle.flush()
            self._instance_lock_acquired = True
            self._instance_lock_handle = handle
        except RuntimeError:
            if handle:
                try:
                    handle.close()
                except Exception:
                    pass
            raise
        except Exception as exc:
            if handle:
                try:
                    handle.close()
                except Exception:
                    pass
            raise RuntimeError(f"Failed to acquire watchdog lock: {exc}") from exc

    def _release_instance_lock(self):
        if not self._instance_lock_acquired:
            return
        try:
            handle = self._instance_lock_handle
            if handle:
                try:
                    if os.name == "nt":
                        import msvcrt

                        handle.seek(0)
                        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        import fcntl

                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass
                try:
                    handle.close()
                except Exception:
                    pass
                self._instance_lock_handle = None

            if not self.lock_file.exists():
                return
            raw = self.lock_file.read_text(encoding="utf-8").strip()
            if str(os.getpid()) == raw:
                self.lock_file.unlink()
        except Exception:
            pass
        finally:
            self._instance_lock_acquired = False

    @staticmethod
    def _is_pid_running(pid: int) -> bool:
        if not pid or pid <= 0:
            return False
        if os.name == "nt":
            try:
                import ctypes

                PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
                STILL_ACTIVE = 259
                handle = ctypes.windll.kernel32.OpenProcess(
                    PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid)
                )
                if not handle:
                    return False
                try:
                    exit_code = ctypes.c_ulong()
                    if not ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                        return False
                    return int(exit_code.value) == STILL_ACTIVE
                finally:
                    ctypes.windll.kernel32.CloseHandle(handle)
            except Exception:
                try:
                    result = subprocess.run(
                        ["tasklist", "/FI", f"PID eq {int(pid)}", "/NH"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL,
                        text=True,
                        check=False,
                    )
                    output = (result.stdout or "").strip().lower()
                    return bool(output) and "no tasks are running" not in output
                except Exception:
                    return False
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def _send_signal(self, pid: int, sig: int):
        if not pid:
            return
        if self._os_name == "windows":
            if sig == signal.SIGINT:
                try:
                    os.kill(pid, signal.CTRL_BREAK_EVENT)
                    return
                except Exception:
                    pass
            if sig == signal.SIGKILL:
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
                return
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            return

        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, sig)
            return
        except ProcessLookupError:
            return
        except Exception:
            pass

        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            return
        except Exception:
            return

    # ------------------------------------------------------------------
    # Launch helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _is_process_handle_running(process: Optional[subprocess.Popen]) -> bool:
        if process is None:
            return False
        try:
            return process.poll() is None
        except Exception:
            return False

    def _detect_terminal_emulator(self) -> str:
        if self._os_name == "windows":
            return "windows-console"
        if self._os_name == "darwin":
            return "osascript"

        forced_terminal = str(os.environ.get("WATCHDOG_TERMINAL", "")).strip()
        if forced_terminal:
            if forced_terminal.lower() in ("none", "direct", "headless", "off"):
                return ""
            if shutil.which(forced_terminal):
                return forced_terminal

        env_terminal = os.environ.get("TERMINAL", "").strip()
        if env_terminal and shutil.which(env_terminal):
            return env_terminal

        for candidate in (
            "gnome-terminal",
            "konsole",
            "xfce4-terminal",
            "lxterminal",
            "x-terminal-emulator",
            "xterm",
        ):
            if shutil.which(candidate):
                return candidate
        return ""

    def _build_service_shell_command(self, svc: ServiceSpec) -> str:
        workdir = shlex.quote(str(svc.working_dir(self.base_dir)))
        python_cmd = shlex.quote(sys.executable)
        script_cmd = shlex.quote(str(svc.script_path(self.base_dir)))
        args = " ".join(shlex.quote(arg) for arg in svc.args)
        core = f"cd {workdir} || exit 1; exec {python_cmd} {script_cmd}"
        if args:
            core += f" {args}"
        return core

    def _build_wrapped_shell_command(self, svc: ServiceSpec) -> str:
        pidfile = shlex.quote(str(self._pid_file(svc)))
        core = self._build_service_shell_command(svc)
        # Keep the service in the terminal foreground so curses receives keyboard input.
        return (
            f"echo $$ > {pidfile}; "
            f"trap 'rm -f {pidfile}' EXIT INT TERM; "
            f"{core}"
        )

    def _build_terminal_command(self, title: str, wrapped_cmd: str) -> Optional[List[str]]:
        term = self._terminal_emulator
        if not term:
            return None

        if term == "gnome-terminal":
            return [term, "--title", title, "--", "bash", "-lc", wrapped_cmd]
        if term == "konsole":
            return [term, "--new-tab", "-p", f"tabtitle={title}", "-e", "bash", "-lc", wrapped_cmd]
        if term == "xfce4-terminal":
            return [term, "--title", title, "--command", f"bash -lc {shlex.quote(wrapped_cmd)}"]
        if term == "xterm":
            return [term, "-T", title, "-e", "bash", "-lc", wrapped_cmd]
        if term == "x-terminal-emulator":
            return [term, "-e", "bash", "-lc", wrapped_cmd]
        if term == "lxterminal":
            return [term, "-t", title, "-e", f"bash -lc {shlex.quote(wrapped_cmd)}"]

        return [term, "-e", "bash", "-lc", wrapped_cmd]

    def _launch_service(self, svc: ServiceSpec, runtime: ServiceRuntime, now: float):
        script_path = svc.script_path(self.base_dir)
        if not script_path.exists():
            self._set_state(
                svc,
                runtime,
                "missing",
                now,
                event="Service script missing",
                error=f"Missing: {svc.script_relpath}",
            )
            runtime.desired_enabled = False
            return

        runtime.launch_attempts += 1
        runtime.activation_method = ""
        runtime.activation_checks = 0
        runtime.resolved_health_port = 0
        runtime.launch_attempt_started_at = now
        runtime.activation_deadline = now + max(5.0, float(svc.activation_timeout_seconds))
        runtime.launch_grace_until = now + LAUNCH_GRACE_SECONDS
        runtime.process_stable_since = 0.0
        runtime.last_health_error = ""

        ports_ok, port_reason = self._ensure_service_ports_available_for_launch(svc, runtime)
        if not ports_ok:
            self._mark_launch_failure(
                svc,
                runtime,
                now,
                f"Launch blocked: {port_reason}",
            )
            self._log(f"[ERROR] {svc.label} launch blocked: {port_reason}")
            return

        self._set_state(
            svc,
            runtime,
            "launching",
            now,
            event=f"Launch attempt #{runtime.launch_attempts}",
            error="",
        )

        child_env = os.environ.copy()
        if svc.service_id == "asr_stream":
            child_env.setdefault("ASR_TARGET_HOST", "127.0.0.1")
            child_env.setdefault("ASR_TARGET_PORT", "6545")
        prefer_terminal_launch = bool(
            svc.launch_in_terminal
            and not self._force_direct
            and svc.service_id not in DIRECT_LAUNCH_SERVICE_IDS
        )

        if self._os_name == "windows":
            creationflags = (
                getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
                | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            )
            try:
                proc = subprocess.Popen(
                    [sys.executable, str(script_path)] + list(svc.args),
                    cwd=str(svc.working_dir(self.base_dir)),
                    creationflags=creationflags,
                    env=child_env,
                )
                runtime.terminal_process = proc
                runtime.pid = proc.pid
                runtime.process_stable_since = now
                runtime.stop_stage = 0
                runtime.stop_requested_at = 0.0
                self._set_state(
                    svc,
                    runtime,
                    "activating",
                    now,
                    event=f"Spawned pid={proc.pid}; waiting for activation probe",
                    error="",
                )
                runtime.restart_backoff_seconds = RESTART_BACKOFF_INITIAL_SECONDS
                self._write_pid_file(svc, proc.pid)
                self._log(f"[LAUNCH] {svc.label} attempt={runtime.launch_attempts} pid={proc.pid}")
                return
            except Exception as exc:
                self._mark_launch_failure(svc, runtime, now, f"Launch failed: {exc}")
                self._log(f"[ERROR] {svc.label} launch failed on attempt={runtime.launch_attempts}: {exc}")
                return

        if self._os_name == "darwin" and prefer_terminal_launch:
            wrapped = self._build_wrapped_shell_command(svc)
            escaped = wrapped.replace("\\", "\\\\").replace('"', '\\"')
            apple_script = f'tell application "Terminal" to do script "bash -lc \\"{escaped}\\""'
            try:
                proc = subprocess.Popen(["osascript", "-e", apple_script], env=child_env)
                runtime.terminal_process = proc
                runtime.stop_stage = 0
                runtime.stop_requested_at = 0.0
                self._set_state(
                    svc,
                    runtime,
                    "launching",
                    now,
                    event="Launching terminal host (awaiting child pid)",
                    error="",
                )
                self._log(f"[LAUNCH] {svc.label} attempt={runtime.launch_attempts} via Terminal.app")
                return
            except Exception as exc:
                self._mark_launch_failure(svc, runtime, now, f"Launch failed: {exc}")
                self._log(f"[ERROR] {svc.label} launch failed on attempt={runtime.launch_attempts}: {exc}")
                return

        if not prefer_terminal_launch:
            log_path = self._service_log_file(svc)
            log_handle = self._open_service_log_handle(svc)
            try:
                popen_kwargs = {
                    "cwd": str(svc.working_dir(self.base_dir)),
                    "env": child_env,
                }
                if self._os_name != "windows":
                    popen_kwargs["start_new_session"] = True
                if log_handle is not None:
                    popen_kwargs["stdout"] = log_handle
                    popen_kwargs["stderr"] = subprocess.STDOUT
                else:
                    popen_kwargs["stdout"] = subprocess.DEVNULL
                    popen_kwargs["stderr"] = subprocess.DEVNULL
                proc = subprocess.Popen(
                    [sys.executable, str(script_path)] + list(svc.args),
                    **popen_kwargs,
                )
                runtime.terminal_process = proc
                runtime.pid = proc.pid
                runtime.process_stable_since = now
                runtime.stop_stage = 0
                runtime.stop_requested_at = 0.0
                self._set_state(
                    svc,
                    runtime,
                    "activating",
                    now,
                    event=f"Spawned pid={proc.pid} (direct); waiting for activation probe",
                    error="",
                )
                runtime.restart_backoff_seconds = RESTART_BACKOFF_INITIAL_SECONDS
                self._write_pid_file(svc, proc.pid)
                self._log(
                    f"[LAUNCH] {svc.label} attempt={runtime.launch_attempts} direct "
                    f"pid={proc.pid} log={log_path}"
                )
                return
            except Exception as exc:
                self._mark_launch_failure(svc, runtime, now, f"Launch failed: {exc}")
                self._log(f"[ERROR] {svc.label} direct launch failed on attempt={runtime.launch_attempts}: {exc}")
                return
            finally:
                if log_handle is not None:
                    try:
                        log_handle.close()
                    except Exception:
                        pass

        wrapped = self._build_wrapped_shell_command(svc)
        terminal_cmd = self._build_terminal_command(f"Teleop - {svc.label}", wrapped)
        if not terminal_cmd:
            self._set_state(
                svc,
                runtime,
                "error",
                now,
                event="Launch failed",
                error="No terminal emulator found",
            )
            runtime.desired_enabled = False
            self._clear_runtime_timers(runtime)
            self._log(f"[ERROR] {svc.label}: no terminal emulator found")
            return

        try:
            proc = subprocess.Popen(terminal_cmd, cwd=str(self.base_dir), env=child_env)
            runtime.terminal_process = proc
            runtime.stop_stage = 0
            runtime.stop_requested_at = 0.0
            self._set_state(
                svc,
                runtime,
                "launching",
                now,
                event="Launching terminal host (awaiting child pid)",
                error="",
            )
            self._log(f"[LAUNCH] {svc.label} attempt={runtime.launch_attempts} via {self._terminal_emulator}")
        except Exception as exc:
            self._mark_launch_failure(svc, runtime, now, f"Launch failed: {exc}")
            self._log(f"[ERROR] {svc.label} launch failed on attempt={runtime.launch_attempts}: {exc}")

    # ------------------------------------------------------------------
    # Runtime synchronization
    # ------------------------------------------------------------------
    def _refresh_pid_state(self, svc: ServiceSpec, runtime: ServiceRuntime, now: float):
        if runtime.terminal_process and runtime.terminal_process.poll() is not None:
            runtime.terminal_process = None

        if runtime.pid and not self._is_pid_running(runtime.pid):
            runtime.pid = None

        if (
            runtime.pid is None
            and self._os_name == "windows"
            and self._is_process_handle_running(runtime.terminal_process)
        ):
            runtime.pid = int(runtime.terminal_process.pid)

        if runtime.pid is None:
            pid_from_file = self._read_pid_file(svc)
            if pid_from_file and self._is_pid_running(pid_from_file):
                runtime.pid = pid_from_file
            elif pid_from_file:
                self._remove_pid_file(svc)

        # Some terminal hosts do not expose the wrapped shell child PID reliably.
        # Fall back to tracking the terminal host PID so health probes can proceed.
        if (
            runtime.pid is None
            and self._os_name not in ("windows", "darwin")
            and self._is_process_handle_running(runtime.terminal_process)
            and runtime.terminal_process is not None
        ):
            if runtime.state != "launching" or now >= runtime.launch_grace_until:
                runtime.pid = int(runtime.terminal_process.pid)
                if runtime.state == "launching":
                    self._set_state(
                        svc,
                        runtime,
                        "activating",
                        now,
                        event=f"Using terminal host pid={runtime.pid}; probing readiness",
                        error="",
                    )

        if runtime.pid:
            if runtime.process_stable_since <= 0:
                runtime.process_stable_since = now
            if runtime.state == "launching":
                self._set_state(
                    svc,
                    runtime,
                    "activating",
                    now,
                    event=f"Captured child pid={runtime.pid}; activation probe pending",
                )
        else:
            runtime.process_stable_since = 0.0

    def _begin_stop(self, svc: ServiceSpec, runtime: ServiceRuntime, now: float):
        self._set_state(svc, runtime, "stopping", now, event="Stopping (SIGINT)", error="")
        runtime.stop_count += 1
        runtime.stop_stage = 1
        runtime.stop_requested_at = now
        if runtime.pid:
            self._send_signal(runtime.pid, signal.SIGINT)
            self._log(f"[STOP] {svc.label} SIGINT pid={runtime.pid}")
        elif runtime.terminal_process and runtime.terminal_process.poll() is None:
            try:
                runtime.terminal_process.terminate()
                self._log(f"[STOP] {svc.label} terminal terminate")
            except Exception:
                pass

    def _handle_stop_escalation(self, svc: ServiceSpec, runtime: ServiceRuntime, now: float):
        if runtime.state != "stopping":
            return
        terminal_running = self._is_process_handle_running(runtime.terminal_process)
        pid_running = bool(runtime.pid and self._is_pid_running(runtime.pid))
        if not pid_running and not terminal_running:
            self._set_state(svc, runtime, "stopped", now, event="Stopped", error="")
            runtime.pid = None
            runtime.terminal_process = None
            runtime.resolved_health_port = 0
            runtime.stop_stage = 0
            runtime.stop_requested_at = 0.0
            self._clear_runtime_timers(runtime)
            self._reset_runtime_health(runtime)
            runtime.started_at = 0.0
            runtime.stopped_at = now
            runtime.restart_backoff_seconds = RESTART_BACKOFF_INITIAL_SECONDS
            self._remove_pid_file(svc)
            self._log(f"[STOPPED] {svc.label}")
            return

        if not runtime.pid and terminal_running and runtime.terminal_process:
            runtime.pid = int(runtime.terminal_process.pid)

        elapsed = now - runtime.stop_requested_at
        if runtime.stop_stage == 1 and elapsed >= STOP_SIGINT_GRACE_SECONDS:
            runtime.stop_stage = 2
            runtime.last_event = "Stopping (SIGTERM)"
            if runtime.pid:
                self._send_signal(runtime.pid, signal.SIGTERM)
                self._log(f"[STOP] {svc.label} SIGTERM pid={runtime.pid}")
            return

        if runtime.stop_stage == 2 and elapsed >= STOP_SIGTERM_GRACE_SECONDS:
            runtime.stop_stage = 3
            runtime.last_event = "Stopping (SIGKILL)"
            if runtime.pid:
                self._send_signal(runtime.pid, signal.SIGKILL)
                self._log(f"[STOP] {svc.label} SIGKILL pid={runtime.pid}")
            return

        if runtime.stop_stage == 3 and elapsed >= STOP_SIGKILL_GRACE_SECONDS:
            self._set_state(
                svc,
                runtime,
                "error",
                now,
                event="Stop escalation failed",
                error="Could not stop process",
            )

    def _sync_single_service(self, svc: ServiceSpec, runtime: ServiceRuntime, now: float):
        if runtime.state == "missing":
            return

        self._refresh_pid_state(svc, runtime, now)
        terminal_running = self._is_process_handle_running(runtime.terminal_process)
        if (
            not runtime.pid
            and self._os_name == "windows"
            and terminal_running
            and runtime.terminal_process
        ):
            runtime.pid = int(runtime.terminal_process.pid)
        pid_running = bool(runtime.pid and self._is_pid_running(runtime.pid))
        running = bool(pid_running or (self._os_name == "windows" and terminal_running))

        if not runtime.desired_enabled:
            should_stop = running or (
                runtime.state in ("launching", "activating", "running", "degraded")
                and terminal_running
            )
            if should_stop and runtime.state != "stopping":
                self._begin_stop(svc, runtime, now)
            self._handle_stop_escalation(svc, runtime, now)
            if not should_stop and runtime.restart_after_stop:
                runtime.restart_after_stop = False
                runtime.desired_enabled = True
                runtime.next_restart_at = now
                runtime.last_event = "Restart queued"
                self._save_desired_state()
            elif not should_stop and not runtime.restart_after_stop:
                if runtime.state not in ("stopped", "missing"):
                    self._set_state(svc, runtime, "stopped", now, event="Stopped", error="")
                    runtime.started_at = 0.0
                    runtime.stopped_at = now
                    self._clear_runtime_timers(runtime)
                    self._reset_runtime_health(runtime)
            return

        if not running and runtime.state in ("running", "degraded"):
            self._remove_pid_file(svc)
            self._mark_launch_failure(
                svc,
                runtime,
                now,
                "Exited unexpectedly",
                increment_crash=True,
            )
            self._log(f"[WARN] {svc.label} exited unexpectedly; waiting to restart")
            return

        if runtime.state == "stopping":
            # Desired on while stopping means a restart request.
            runtime.restart_after_stop = True
            self._handle_stop_escalation(svc, runtime, now)
            return

        if runtime.state == "launching":
            if not running:
                if terminal_running and now < runtime.launch_grace_until:
                    return
                if now < runtime.launch_grace_until:
                    return
                if runtime.terminal_process and runtime.terminal_process.poll() is not None:
                    code = runtime.terminal_process.returncode
                    self._mark_launch_failure(
                        svc, runtime, now, f"Terminal exited before child activation ({code})"
                    )
                    self._log(f"[ERROR] {svc.label} terminal host exited ({code})")
                    return
            # Wait for pid capture before activation.
            if not runtime.pid:
                if now > runtime.activation_deadline:
                    self._mark_launch_failure(
                        svc, runtime, now, "Activation timed out waiting for child pid"
                    )
                    self._log(f"[ERROR] {svc.label} activation timeout (no child pid captured)")
                return
            self._set_state(
                svc,
                runtime,
                "activating",
                now,
                event=f"Child pid={runtime.pid} captured; probing readiness",
            )

        if runtime.state == "activating":
            if not running:
                self._mark_launch_failure(
                    svc, runtime, now, "Process exited during activation", increment_crash=True
                )
                self._log(f"[ERROR] {svc.label} exited during activation")
                return

            activation_probe_interval = max(
                0.25,
                min(
                    DEFAULT_ACTIVATION_PROBE_INTERVAL_SECONDS,
                    float(svc.health_check_interval_seconds or DEFAULT_HEALTH_CHECK_INTERVAL_SECONDS),
                ),
            )
            if runtime.last_health_probe_at and (
                now - runtime.last_health_probe_at
            ) < activation_probe_interval:
                return

            runtime.last_health_probe_at = now
            runtime.activation_checks += 1
            probe_ok, probe_detail = self._health_probe_with_runtime(svc, runtime)
            if probe_ok:
                runtime.start_count += 1
                runtime.started_at = now
                runtime.next_restart_at = 0.0
                runtime.restart_backoff_seconds = RESTART_BACKOFF_INITIAL_SECONDS
                runtime.activation_method = f"probe:{probe_detail}"
                runtime.last_health_ok_at = now
                self._reset_runtime_health(runtime)
                self._set_state(
                    svc,
                    runtime,
                    "running",
                    now,
                    event=f"Activated ({probe_detail})",
                    error="",
                )
                self._log(
                    f"[RUNNING] {svc.label} pid={runtime.pid} activated via {probe_detail} "
                    f"attempt={runtime.launch_attempts}"
                )
                return

            runtime.last_health_error = probe_detail
            alive_for = now - (runtime.process_stable_since or now)
            if alive_for >= max(1.5, float(svc.activation_stability_seconds)):
                runtime.start_count += 1
                runtime.started_at = now
                runtime.next_restart_at = 0.0
                runtime.restart_backoff_seconds = RESTART_BACKOFF_INITIAL_SECONDS
                runtime.activation_method = f"stable-process:{alive_for:.1f}s"
                self._set_state(
                    svc,
                    runtime,
                    "degraded",
                    now,
                    event=f"Activated by process stability; health probe failing ({probe_detail})",
                    error="",
                )
                self._log(
                    f"[RUNNING] {svc.label} pid={runtime.pid} treated active after "
                    f"{alive_for:.1f}s stability (probe={probe_detail})"
                )
                return

            if now > runtime.activation_deadline:
                self._mark_launch_failure(
                    svc, runtime, now, f"Activation timeout ({probe_detail})"
                )
                self._log(f"[ERROR] {svc.label} activation timeout: {probe_detail}")
                return

            runtime.last_event = f"Activating... probe={probe_detail}"
            return

        if runtime.state in ("running", "degraded"):
            if not running:
                self._mark_launch_failure(
                    svc, runtime, now, "Exited unexpectedly", increment_crash=True
                )
                self._log(f"[WARN] {svc.label} exited unexpectedly; waiting to restart")
                return

            interval = max(0.6, float(svc.health_check_interval_seconds))
            if runtime.last_health_probe_at and (now - runtime.last_health_probe_at) < interval:
                return

            runtime.health_checks += 1
            runtime.last_health_probe_at = now
            probe_ok, probe_detail = self._health_probe_with_runtime(svc, runtime)
            if probe_ok:
                runtime.last_health_ok_at = now
                runtime.consecutive_health_failures = 0
                runtime.last_health_error = ""
                if runtime.state != "running":
                    self._set_state(
                        svc,
                        runtime,
                        "running",
                        now,
                        event=f"Health restored ({probe_detail})",
                        error="",
                    )
                else:
                    runtime.last_event = f"Healthy ({probe_detail})"
                return

            runtime.health_failures += 1
            runtime.consecutive_health_failures += 1
            runtime.last_health_error = probe_detail
            failure_limit = max(1, int(svc.health_failure_threshold))
            if runtime.consecutive_health_failures >= failure_limit:
                self._set_state(
                    svc,
                    runtime,
                    "degraded",
                    now,
                    event=f"Health degraded ({runtime.consecutive_health_failures} fails)",
                    error="",
                )
            else:
                runtime.last_event = (
                    f"Probe soft-fail {runtime.consecutive_health_failures}/{failure_limit}: {probe_detail}"
                )
            return

        if runtime.state == "error" and runtime.next_restart_at > now:
            return

        if runtime.next_restart_at > now:
            return

        self._launch_service(svc, runtime, now)

    def _resume_asr_if_paused(self, pid: int = 0):
        target = int(pid or self._asr_throttle_target_pid or 0)
        if target <= 0 or not self._asr_throttle_paused:
            return
        if self._os_name == "windows":
            self._asr_throttle_paused = False
            self._asr_throttle_target_pid = 0
            return
        try:
            os.kill(target, signal.SIGCONT)
        except ProcessLookupError:
            pass
        except Exception:
            pass
        self._asr_throttle_paused = False
        self._asr_throttle_target_pid = 0
        self._asr_throttle_last_switch_at = time.time()

    def _apply_asr_llm_throttle(self, now: float):
        if self._os_name == "windows":
            return
        asr_rt = self.runtime_by_id.get("asr_stream")
        if asr_rt is None or not int(asr_rt.pid or 0):
            self._resume_asr_if_paused()
            return
        asr_pid = int(asr_rt.pid or 0)
        if not self._is_pid_running(asr_pid):
            self._resume_asr_if_paused()
            return

        enabled = bool(self._cpu_throttle.get("asr_llm_throttle_enable", True))
        llm_state = str(self._pipeline_stage_state.get("llm") or "idle").strip().lower()
        llm_busy = llm_state in ("queued", "processing", "streaming", "model_switching")
        if not enabled or not llm_busy:
            self._resume_asr_if_paused(asr_pid)
            return

        pct = int(self._cpu_throttle.get("asr_llm_throttle_percent", 65) or 65)
        pct = max(0, min(95, pct))
        if pct <= 0:
            self._resume_asr_if_paused(asr_pid)
            return
        cycle_ms = int(self._cpu_throttle.get("asr_llm_throttle_cycle_ms", 320) or 320)
        cycle_s = max(0.08, min(2.0, float(cycle_ms) / 1000.0))
        pause_s = cycle_s * (float(pct) / 100.0)
        run_s = max(0.02, cycle_s - pause_s)
        elapsed = now - float(self._asr_throttle_last_switch_at or 0.0)

        self._asr_throttle_target_pid = asr_pid
        if self._asr_throttle_paused:
            if elapsed >= pause_s:
                try:
                    os.kill(asr_pid, signal.SIGCONT)
                    self._asr_throttle_paused = False
                    self._asr_throttle_last_switch_at = now
                except ProcessLookupError:
                    self._asr_throttle_paused = False
                except Exception:
                    pass
            return

        if elapsed >= run_s:
            try:
                os.kill(asr_pid, signal.SIGSTOP)
                self._asr_throttle_paused = True
                self._asr_throttle_last_switch_at = now
            except ProcessLookupError:
                self._asr_throttle_paused = False
            except Exception:
                pass

    def _monitor_loop(self):
        while not self._stop_event.is_set():
            now = time.time()
            with self._lock:
                for svc in self.services:
                    runtime = self.runtime_by_id[svc.service_id]
                    self._sync_single_service(svc, runtime, now)
                self._update_resource_usage(now)
                self._maybe_reload_external_cpu_throttle(now)
                self._apply_asr_llm_throttle(now)
            self._poll_pipeline_observability(now)
            self._poll_for_auto_update()
            time.sleep(MONITOR_INTERVAL_SECONDS)

    # ------------------------------------------------------------------
    # Public control API
    # ------------------------------------------------------------------
    def start(self):
        if self._monitor_thread and self._monitor_thread.is_alive():
            return
        self._refresh_service_links()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        self._service_link_thread = threading.Thread(target=self._service_link_loop, daemon=True)
        self._service_link_thread.start()

    def request_shutdown(self):
        self._resume_asr_if_paused()
        with self._lock:
            for svc in self.services:
                runtime = self.runtime_by_id[svc.service_id]
                runtime.desired_enabled = False
                runtime.restart_after_stop = False
                if runtime.state not in ("missing",):
                    runtime.last_event = "Shutdown requested"
        self._stop_event.set()

    def await_shutdown(self, timeout: float = 15.0):
        end = time.time() + timeout
        while time.time() < end:
            all_stopped = True
            with self._lock:
                now = time.time()
                for svc in self.services:
                    runtime = self.runtime_by_id[svc.service_id]
                    self._sync_single_service(svc, runtime, now)
                    if runtime.state not in ("stopped", "missing"):
                        all_stopped = False
            if all_stopped:
                break
            time.sleep(0.25)

        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)
        if self._service_link_thread and self._service_link_thread.is_alive():
            self._service_link_thread.join(timeout=2.0)

    def toggle_selected(self):
        svc = self.services[self._selected_index]
        runtime = self.runtime_by_id[svc.service_id]
        if runtime.state == "missing":
            return
        now = time.time()
        sid = str(svc.service_id or "").strip().lower()
        if runtime.desired_enabled:
            runtime.desired_enabled = False
            runtime.restart_after_stop = False
            runtime.last_event = "Disabled by user"
            self._log(f"[USER] Disabled {svc.label}")
            if sid == "llm_bridge":
                ollama_runtime = self.runtime_by_id.get("ollama")
                if ollama_runtime and ollama_runtime.state != "missing":
                    ollama_runtime.desired_enabled = False
                    ollama_runtime.restart_after_stop = False
                    ollama_runtime.last_event = "Disabled with LLM Bridge"
                    self._log("[USER] Disabled Ollama (linked with LLM Bridge)")
            if sid in ("llm_bridge", "ollama"):
                self._force_stop_ollama_processes(f"{svc.label} disabled")
        else:
            runtime.desired_enabled = True
            runtime.next_restart_at = now
            runtime.last_event = "Enabled by user"
            self._log(f"[USER] Enabled {svc.label}")
            if sid == "llm_bridge":
                ollama_runtime = self.runtime_by_id.get("ollama")
                if ollama_runtime and ollama_runtime.state != "missing":
                    if not ollama_runtime.desired_enabled:
                        ollama_runtime.desired_enabled = True
                        ollama_runtime.next_restart_at = now
                        ollama_runtime.last_event = "Enabled with LLM Bridge"
                        self._log("[USER] Enabled Ollama (required by LLM Bridge)")
        self._save_desired_state()

    def restart_selected(self):
        svc = self.services[self._selected_index]
        runtime = self.runtime_by_id[svc.service_id]
        if runtime.state == "missing":
            return
        runtime.restart_count += 1
        runtime.restart_after_stop = True
        runtime.desired_enabled = False
        runtime.last_event = "Restart requested"
        self._log(f"[USER] Restart {svc.label}")

    def toggle_all(self):
        with self._lock:
            any_disabled = any(
                (rt.state != "missing" and not rt.desired_enabled)
                for rt in self.runtime_by_id.values()
            )
            target_enabled = any_disabled
            for svc in self.services:
                runtime = self.runtime_by_id[svc.service_id]
                if runtime.state == "missing":
                    continue
                runtime.restart_after_stop = False
                runtime.desired_enabled = target_enabled
                runtime.last_event = "Enabled all" if target_enabled else "Disabled all"
            self._log(f"[USER] {'Enabled' if target_enabled else 'Disabled'} all services")
            self._save_desired_state()

    # ------------------------------------------------------------------
    # Service link discovery and clipboard helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _detect_lan_ip() -> str:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
                probe.settimeout(0.5)
                probe.connect(("8.8.8.8", 80))
                ip = str(probe.getsockname()[0] or "").strip()
                if ip and ip != "127.0.0.1":
                    return ip
        except Exception:
            pass
        return ""

    @staticmethod
    def _http_json(method: str, url: str, payload: Optional[Dict[str, object]] = None, timeout: float = 1.2):
        request_data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            try:
                request_data = json.dumps(payload).encode("utf-8")
                headers["Content-Type"] = "application/json"
            except Exception:
                request_data = None
        req = urllib.request.Request(url, data=request_data, method=str(method or "GET").upper(), headers=headers)
        with urllib.request.urlopen(req, timeout=max(0.2, float(timeout))) as resp:
            raw = resp.read()
        text = raw.decode("utf-8", errors="replace").strip() if raw else ""
        return json.loads(text) if text else {}

    @staticmethod
    def _service_default_dashboard_path(service_id: str) -> str:
        sid = str(service_id or "").strip().lower()
        if sid == "router":
            return "/dashboard"
        if sid == "pipeline_api":
            return "/llm/dashboard"
        if sid == "camera_router":
            return "/list"
        if sid == "audio_router":
            return "/"
        if sid == "ollama":
            return "/"
        return ""

    def _collect_url_values(self, payload, prefix: str, out: Dict[str, str], depth: int = 0):
        if depth > 4:
            return
        if isinstance(payload, dict):
            for key, value in payload.items():
                key_text = str(key or "").strip()
                if not key_text:
                    continue
                next_prefix = f"{prefix}_{key_text}" if prefix else key_text
                if isinstance(value, str):
                    url = value.strip()
                    if url.startswith("http://") or url.startswith("https://"):
                        out[next_prefix] = url
                elif isinstance(value, (dict, list, tuple)):
                    self._collect_url_values(value, next_prefix, out, depth + 1)
        elif isinstance(payload, (list, tuple)):
            # Only inspect the first few items to avoid bloating with camera/per-device duplicates.
            for idx, value in enumerate(payload[:3]):
                next_prefix = f"{prefix}_{idx}" if prefix else str(idx)
                self._collect_url_values(value, next_prefix, out, depth + 1)

    @staticmethod
    def _url_with_session_key(url: str, session_key: str) -> str:
        raw = str(url or "").strip()
        key = str(session_key or "").strip()
        if not raw or not key or not (raw.startswith("http://") or raw.startswith("https://")):
            return raw
        try:
            parsed = urllib.parse.urlsplit(raw)
            query_pairs = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
            if not any(str(k) == "session_key" for k, _ in query_pairs):
                query_pairs.append(("session_key", key))
            query = urllib.parse.urlencode(query_pairs)
            return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, query, parsed.fragment))
        except Exception:
            sep = "&" if "?" in raw else "?"
            return f"{raw}{sep}session_key={urllib.parse.quote_plus(key)}"

    def _service_security_settings(self, svc: ServiceSpec) -> Dict[str, object]:
        defaults = {"require_auth": False, "password": "egg", "session_timeout": 300}
        rel = str(svc.config_relpath or "").strip()
        if not rel:
            return defaults
        cfg_path = (self.base_dir / rel).resolve()
        if not cfg_path.exists():
            return defaults
        try:
            payload = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            return defaults
        if not isinstance(payload, dict):
            return defaults

        base_keys = [
            str(svc.service_id or "").strip(),
            "pipeline_api" if svc.service_id == "llm_bridge" else "",
            "security",
        ]
        candidates = []
        for base in base_keys:
            base = str(base or "").strip()
            if not base:
                continue
            if base == "security":
                candidates.append("security")
            else:
                candidates.append(f"{base}.security")

        require_auth = defaults["require_auth"]
        password = defaults["password"]
        session_timeout = defaults["session_timeout"]
        for prefix in candidates:
            req_val = _get_nested(payload, f"{prefix}.require_auth", None)
            if req_val is not None:
                if isinstance(req_val, bool):
                    require_auth = bool(req_val)
                else:
                    require_auth = str(req_val).strip().lower() in ("1", "true", "yes", "on")
            pwd_val = _get_nested(payload, f"{prefix}.password", None)
            if pwd_val is not None and str(pwd_val).strip():
                password = str(pwd_val).strip()
            timeout_val = _get_nested(payload, f"{prefix}.session_timeout", None)
            if timeout_val is not None:
                try:
                    parsed_timeout = int(timeout_val)
                    if parsed_timeout > 0:
                        session_timeout = parsed_timeout
                except Exception:
                    pass

        return {
            "require_auth": bool(require_auth),
            "password": str(password or "egg"),
            "session_timeout": int(session_timeout if int(session_timeout) > 0 else 300),
        }

    def _service_session_key(self, svc: ServiceSpec, local_base_url: str) -> Tuple[str, str]:
        base = str(local_base_url or "").strip().rstrip("/")
        if not base or not base.startswith("http"):
            return "", "invalid local base URL"
        security = self._service_security_settings(svc)
        require_auth = bool(security.get("require_auth", False))
        if not require_auth:
            return "", ""

        now = time.time()
        with self._session_lock:
            cached = dict(self._service_sessions.get(svc.service_id, {}))
        cached_key = str(cached.get("session_key") or "").strip()
        expires_at = float(cached.get("expires_at") or 0.0)
        if cached_key and (expires_at - SERVICE_SESSION_REFRESH_SAFETY_SECONDS) > now:
            return cached_key, ""

        auth_url = f"{base}/auth"
        password = str(security.get("password") or "egg")
        timeout = max(0.3, min(3.0, SERVICE_LINK_HTTP_TIMEOUT_SECONDS))
        try:
            payload = self._http_json(
                "POST",
                auth_url,
                payload={"password": password},
                timeout=timeout,
            )
            if not isinstance(payload, dict):
                return "", "auth payload invalid"
            session_key = str(payload.get("session_key") or "").strip()
            if not session_key:
                message = str(payload.get("message") or "missing session_key").strip()
                return "", message
            session_timeout = payload.get("timeout", security.get("session_timeout", 300))
            try:
                session_timeout = int(session_timeout)
            except Exception:
                session_timeout = int(security.get("session_timeout", 300))
            session_timeout = max(30, session_timeout)
            with self._session_lock:
                self._service_sessions[svc.service_id] = {
                    "session_key": session_key,
                    "expires_at": now + float(session_timeout),
                    "timeout": int(session_timeout),
                }
            return session_key, ""
        except Exception as exc:
            return "", str(exc)

    def _service_port(self, svc: ServiceSpec, runtime: Optional[ServiceRuntime] = None) -> int:
        if runtime is not None:
            try:
                active = int(runtime.resolved_health_port or 0)
            except Exception:
                active = 0
            if 1 <= active <= 65535:
                return active
        try:
            configured = int(svc.resolved_health_port(self.base_dir) or 0)
        except Exception:
            configured = 0
        if 1 <= configured <= 65535:
            return configured
        try:
            fallback = int(svc.health_port or 0)
        except Exception:
            fallback = 0
        return fallback if 1 <= fallback <= 65535 else 0

    def _discover_service_links(self, svc: ServiceSpec, runtime: ServiceRuntime) -> Tuple[Dict[str, str], str]:
        links: Dict[str, str] = {}
        err = ""
        mode = str(svc.health_mode or "").strip().lower()
        port = self._service_port(svc, runtime)
        if mode != "http":
            if port > 0:
                links["local_base_url"] = f"tcp://127.0.0.1:{port}"
                if self._lan_ip:
                    links["lan_base_url"] = f"tcp://{self._lan_ip}:{port}"

            # LLM Bridge is TCP, but model dashboard lives on Pipeline API.
            if svc.service_id == "llm_bridge":
                pipeline_svc = self.service_by_id.get("pipeline_api")
                pipeline_rt = self.runtime_by_id.get("pipeline_api")
                pipeline_port = self._service_port(pipeline_svc, pipeline_rt) if pipeline_svc else 0
                if pipeline_port > 0:
                    pipeline_base = f"http://127.0.0.1:{pipeline_port}"
                    pipeline_key = ""
                    pipeline_err = ""
                    if pipeline_svc:
                        pipeline_key, pipeline_err = self._service_session_key(pipeline_svc, pipeline_base)
                    links["local_llm_dashboard_url"] = self._url_with_session_key(
                        f"{pipeline_base}/llm/dashboard", pipeline_key
                    )
                    links["local_pipeline_state_url"] = self._url_with_session_key(
                        f"{pipeline_base}/pipeline/state", pipeline_key
                    )
                    if self._lan_ip:
                        links["lan_llm_dashboard_url"] = self._url_with_session_key(
                            f"http://{self._lan_ip}:{pipeline_port}/llm/dashboard", pipeline_key
                        )
                        links["lan_pipeline_state_url"] = self._url_with_session_key(
                            f"http://{self._lan_ip}:{pipeline_port}/pipeline/state", pipeline_key
                        )
                    if pipeline_err and not err:
                        err = f"pipeline auth: {pipeline_err}"
            return links, err

        if port <= 0:
            return links, "No HTTP port configured"

        local_base = f"http://127.0.0.1:{port}"
        links["local_base_url"] = local_base
        if self._lan_ip:
            links["lan_base_url"] = f"http://{self._lan_ip}:{port}"

        dashboard_path = self._service_default_dashboard_path(svc.service_id)
        if dashboard_path:
            links["local_dashboard_url"] = f"{local_base}{dashboard_path}"
            if self._lan_ip:
                links["lan_dashboard_url"] = f"http://{self._lan_ip}:{port}{dashboard_path}"

        # Merge URLs exported by service router metadata.
        for route in ("/router_info", "/tunnel_info"):
            url = f"{local_base}{route}"
            try:
                payload = self._http_json("GET", url, timeout=SERVICE_LINK_HTTP_TIMEOUT_SECONDS)
                if isinstance(payload, dict):
                    self._collect_url_values(payload, "", links)
            except Exception as exc:
                err = str(exc)

        session_key, auth_error = self._service_session_key(svc, local_base)
        if session_key:
            for key, value in list(links.items()):
                text = str(value or "").strip()
                if not (text.startswith("http://") or text.startswith("https://")):
                    continue
                if key.endswith("auth_url"):
                    continue
                links[key] = self._url_with_session_key(text, session_key)
        if auth_error and not err:
            err = f"auth: {auth_error}"

        return links, err

    @staticmethod
    def _select_primary_link(links: Dict[str, str]) -> Tuple[str, str]:
        if not links:
            return "", ""

        preferred_exact = (
            "lan_llm_dashboard_url",
            "local_llm_dashboard_url",
            "tunnel_llm_dashboard_url",
            "llm_dashboard_url",
            "lan_dashboard_url",
            "local_dashboard_url",
            "tunnel_dashboard_url",
            "dashboard_url",
            "lan_list_url",
            "local_list_url",
            "tunnel_list_url",
            "list_url",
            "lan_base_url",
            "local_base_url",
            "tunnel_base_url",
        )
        for key in preferred_exact:
            value = str(links.get(key) or "").strip()
            if value:
                return key, value

        best_key = ""
        best_url = ""
        best_rank = (9, 9, 9, 999)
        for key, value in links.items():
            url = str(value or "").strip()
            if not url:
                continue
            key_text = str(key or "").lower()
            if key_text.startswith("lan_"):
                scope_rank = 0
            elif key_text.startswith("tunnel_"):
                scope_rank = 1
            elif key_text.startswith("local_"):
                scope_rank = 2
            else:
                scope_rank = 3
            if "dashboard" in key_text:
                route_rank = 0
            elif key_text.endswith("list_url"):
                route_rank = 1
            elif key_text.endswith("health_url"):
                route_rank = 2
            elif key_text.endswith("_url"):
                route_rank = 3
            else:
                route_rank = 4
            rank = (scope_rank, route_rank, len(key_text), len(url))
            if rank < best_rank:
                best_rank = rank
                best_key = key
                best_url = url
        return best_key, best_url

    @staticmethod
    def _shorten_url(url: str, max_len: int = 44) -> str:
        text = str(url or "").strip()
        if len(text) <= max_len:
            return text
        if max_len <= 7:
            return text[:max_len]
        keep = max_len - 3
        left = max(4, int(keep * 0.63))
        right = max(3, keep - left)
        return f"{text[:left]}...{text[-right:]}"

    def _refresh_service_links(self):
        snapshot = []
        with self._lock:
            for svc in self.services:
                runtime = self.runtime_by_id.get(svc.service_id)
                if runtime is None:
                    continue
                snapshot.append((svc, runtime))

        links_by_id: Dict[str, Dict[str, str]] = {}
        errors_by_id: Dict[str, str] = {}
        for svc, runtime in snapshot:
            links, err = self._discover_service_links(svc, runtime)
            if links:
                key, url = self._select_primary_link(links)
                links["primary_key"] = key
                links["primary_url"] = url
                links_by_id[svc.service_id] = links
            if err:
                errors_by_id[svc.service_id] = err

        with self._lock:
            self._service_links = links_by_id
            self._service_link_errors = errors_by_id
            self._service_link_last_refresh_at = time.time()

    def _poll_pipeline_observability(self, now: float):
        if now - float(self._pipeline_obs_last_poll_at or 0.0) < self._pipeline_obs_poll_interval:
            return
        self._pipeline_obs_last_poll_at = now

        svc = self.service_by_id.get("pipeline_api")
        runtime = self.runtime_by_id.get("pipeline_api")
        if svc is None or runtime is None:
            return
        if runtime.state not in ("running", "degraded", "activating", "launching"):
            return
        port = self._service_port(svc, runtime)
        if port <= 0:
            return
        local_base = f"http://127.0.0.1:{port}"
        session_key, _ = self._service_session_key(svc, local_base)
        state_url = self._url_with_session_key(f"{local_base}/pipeline/state", session_key)
        try:
            payload = self._http_json("GET", state_url, timeout=0.9)
        except Exception:
            return
        if not isinstance(payload, dict):
            return
        state = payload.get("pipeline_state", payload.get("state", payload))
        if not isinstance(state, dict):
            return
        stages = state.get("stages", {})
        if isinstance(stages, dict):
            next_stage_state = {}
            for key in ("asr", "llm", "tts"):
                entry = stages.get(key, {})
                if isinstance(entry, dict):
                    next_stage_state[key] = str(entry.get("state") or "idle").strip().lower() or "idle"
                else:
                    next_stage_state[key] = "idle"
            self._pipeline_stage_state = next_stage_state
        events = state.get("events", [])
        if not isinstance(events, list):
            return

        latest_seq = int(self._pipeline_obs_last_seq or 0)
        try:
            state_seq = int(state.get("seq") or 0)
        except Exception:
            state_seq = 0
        if latest_seq > 0 and state_seq >= 0 and state_seq < latest_seq:
            # pipeline_api likely restarted and reset sequence numbering; resume from fresh stream.
            latest_seq = 0
        new_events = []
        for item in events:
            if not isinstance(item, dict):
                continue
            try:
                seq = int(item.get("seq") or 0)
            except Exception:
                seq = 0
            if seq > latest_seq:
                new_events.append(item)
        if not new_events:
            return

        new_events.sort(key=lambda evt: int(evt.get("seq") or 0))
        for evt in new_events[-12:]:
            stage = str(evt.get("stage") or "").strip().upper() or "PIPELINE"
            step_state = str(evt.get("state") or "").strip() or "event"
            source = str(evt.get("source") or "").strip() or "pipeline"
            text = str(evt.get("text") or "").strip()
            detail = str(evt.get("detail") or "").strip()
            text_short = text if len(text) <= 160 else (text[:157] + "...")
            detail_short = detail if len(detail) <= 120 else (detail[:117] + "...")
            message = f"[PIPELINE] {stage}:{step_state} ({source})"
            if detail_short:
                message += f" | {detail_short}"
            if text_short:
                message += f" | text=\"{text_short}\""
            self._log(message)
            try:
                seq_val = int(evt.get("seq") or 0)
                latest_seq = max(latest_seq, seq_val)
            except Exception:
                pass
        self._pipeline_obs_last_seq = latest_seq

    def _service_link_loop(self):
        while not self._stop_event.is_set():
            try:
                self._refresh_service_links()
            except Exception:
                pass
            if self._stop_event.wait(self._service_link_refresh_interval):
                break

    @staticmethod
    def _register_click_region(regions: List[Dict[str, object]], row: int, col_start: int, col_end: int, payload: str):
        regions.append(
            {
                "row": int(row),
                "col_start": int(min(col_start, col_end)),
                "col_end": int(max(col_start, col_end)),
                "payload": str(payload or ""),
            }
        )

    @staticmethod
    def _copy_to_clipboard(text: str) -> Tuple[bool, str]:
        value = str(text or "").strip()
        if not value:
            return False, "empty link"

        command_sets: List[Tuple[List[str], str]] = []
        if os.name == "nt":
            clip_bin = shutil.which("clip.exe") or shutil.which("clip")
            if clip_bin:
                command_sets.append(([clip_bin], "clip.exe"))
            ps_bin = shutil.which("powershell.exe") or shutil.which("powershell")
            if ps_bin:
                command_sets.append(
                    (
                        [ps_bin, "-NoProfile", "-Command", "Set-Clipboard -Value ([Console]::In.ReadToEnd())"],
                        "powershell",
                    )
                )
        else:
            wl_copy = shutil.which("wl-copy")
            if wl_copy:
                command_sets.append(([wl_copy], "wl-copy"))
            xclip = shutil.which("xclip")
            if xclip:
                command_sets.append(([xclip, "-selection", "clipboard"], "xclip"))
            xsel = shutil.which("xsel")
            if xsel:
                command_sets.append(([xsel, "--clipboard", "--input"], "xsel"))
            pbcopy = shutil.which("pbcopy")
            if pbcopy:
                command_sets.append(([pbcopy], "pbcopy"))

        for command, label in command_sets:
            try:
                subprocess.run(command, input=value, text=True, capture_output=True, check=True)
                return True, label
            except Exception:
                continue

        # Final fallback for terminals supporting OSC52 clipboard control.
        try:
            encoded = base64.b64encode(value.encode("utf-8")).decode("ascii")
            sys.__stdout__.write(f"\033]52;c;{encoded}\a")
            sys.__stdout__.flush()
            return True, "OSC52"
        except Exception as exc:
            return False, str(exc) or "no clipboard backend"

    def _copy_selected_service_link(self):
        with self._lock:
            svc = self.services[self._selected_index]
            link = self._service_links.get(svc.service_id, {})
            primary = str(link.get("primary_url") or "").strip()
        if not primary:
            self._log(f"[LINK] {svc.label}: no link available to copy")
            return
        ok, method = self._copy_to_clipboard(primary)
        if ok:
            self._log(f"[LINK] Copied {svc.label}: {primary} ({method})")
        else:
            self._log(f"[WARN] {svc.label} copy failed: {method}")

    def _log_all_service_links(self):
        with self._lock:
            lines = []
            for svc in self.services:
                data = self._service_links.get(svc.service_id, {})
                url = str(data.get("primary_url") or "").strip()
                if url:
                    lines.append(f"[LINK] {svc.label}: {url}")
                else:
                    lines.append(f"[LINK] {svc.label}: (no HTTP link)")
        self._log("[LINK] Service endpoints:")
        for line in lines:
            self._log(line)

    def _handle_mouse_event(self) -> bool:
        try:
            _, mouse_x, mouse_y, _, button_state = curses.getmouse()
        except Exception:
            return False
        click_mask = (
            getattr(curses, "BUTTON1_CLICKED", 0)
            | getattr(curses, "BUTTON1_RELEASED", 0)
            | getattr(curses, "BUTTON1_PRESSED", 0)
            | getattr(curses, "BUTTON1_DOUBLE_CLICKED", 0)
        )
        if click_mask and not (button_state & click_mask):
            return False

        with self._lock:
            regions = list(self._click_regions)
        for region in reversed(regions):
            if int(region.get("row", -1)) != int(mouse_y):
                continue
            if int(region.get("col_start", 0)) <= int(mouse_x) <= int(region.get("col_end", -1)):
                payload = str(region.get("payload", "")).strip()
                if not payload:
                    return False
                ok, method = self._copy_to_clipboard(payload)
                with self._lock:
                    if ok:
                        self._log(f"[LINK] Copied: {payload} ({method})")
                    else:
                        self._log(f"[WARN] Copy failed: {method}")
                return True
        return False

    # ------------------------------------------------------------------
    # Curses UI
    # ------------------------------------------------------------------
    @staticmethod
    def _format_duration(seconds: float) -> str:
        if seconds <= 0:
            return "-"
        total = int(seconds)
        hrs = total // 3600
        mins = (total % 3600) // 60
        secs = total % 60
        if hrs > 0:
            return f"{hrs:02d}:{mins:02d}:{secs:02d}"
        return f"{mins:02d}:{secs:02d}"

    def _runtime_detail(self, runtime: ServiceRuntime, now: float) -> str:
        base = runtime.last_error if runtime.last_error else runtime.last_event
        tags = []
        if runtime.launch_attempts > 0:
            tags.append(f"a{runtime.launch_attempts}")
        if runtime.state in ("launching", "activating") and runtime.activation_deadline > 0:
            remaining = max(0.0, runtime.activation_deadline - now)
            tags.append(f"t-{remaining:.0f}s")
        if runtime.health_checks > 0:
            tags.append(f"h{runtime.health_checks}/{runtime.health_failures}")
        if runtime.consecutive_health_failures > 0:
            tags.append(f"cf{runtime.consecutive_health_failures}")
        if runtime.resolved_health_port > 0:
            tags.append(f"p{runtime.resolved_health_port}")
        if tags:
            return f"{base} [{' '.join(tags)}]".strip()
        return base

    @staticmethod
    def _safe_addnstr(stdscr, row: int, col: int, text: str, max_width: int, attr: int = 0):
        try:
            height, width = stdscr.getmaxyx()
            if row < 0 or row >= height or col >= width:
                return
            limit = max(0, min(int(max_width), width - col - 1))
            if limit <= 0:
                return
            stdscr.addnstr(row, col, str(text), limit, attr)
        except curses.error:
            return

    @staticmethod
    def _format_khz(freq_khz: int) -> str:
        try:
            value = int(freq_khz)
        except Exception:
            return "-"
        if value <= 0:
            return "-"
        return f"{value/1000.0:.0f}MHz"

    def _resource_summary_line(self) -> str:
        heaviest_cpu_label = "n/a"
        heaviest_cpu_value = 0.0
        heaviest_mem_label = "n/a"
        heaviest_mem_value = 0.0
        peak_cpu_label = "n/a"
        peak_cpu_value = 0.0
        peak_mem_label = "n/a"
        peak_mem_value = 0.0

        for svc in self.services:
            runtime = self.runtime_by_id.get(svc.service_id)
            if not runtime:
                continue
            if runtime.cpu_percent > heaviest_cpu_value:
                heaviest_cpu_value = float(runtime.cpu_percent)
                heaviest_cpu_label = svc.label
            if runtime.rss_mb > heaviest_mem_value:
                heaviest_mem_value = float(runtime.rss_mb)
                heaviest_mem_label = svc.label
            if runtime.peak_cpu_percent > peak_cpu_value:
                peak_cpu_value = float(runtime.peak_cpu_percent)
                peak_cpu_label = svc.label
            if runtime.peak_rss_mb > peak_mem_value:
                peak_mem_value = float(runtime.peak_rss_mb)
                peak_mem_label = svc.label

        return (
            f"Heaviest CPU: {heaviest_cpu_label} {heaviest_cpu_value:.1f}% | "
            f"Heaviest RAM: {heaviest_mem_label} {heaviest_mem_value:.1f}MB | "
            f"Peak CPU: {peak_cpu_label} {peak_cpu_value:.1f}% | "
            f"Peak RAM: {peak_mem_label} {peak_mem_value:.1f}MB"
        )

    def _usage_color_pair(self, value_percent: float, metric: str = "cpu") -> int:
        value = float(value_percent or 0.0)
        if value >= 85.0:
            return 5
        if value >= 60.0:
            return 3
        if str(metric or "").strip().lower() == "ram":
            return 6
        return 2

    def _draw_usage_graph(
        self,
        stdscr,
        top_row: int,
        left_col: int,
        graph_width: int,
        graph_height: int,
        series: Sequence[float],
        title: str,
        metric: str,
    ):
        if graph_width < 8 or graph_height < 4:
            self._safe_addnstr(stdscr, top_row, left_col, f"{title}: terminal too small", graph_width)
            return

        plot_top = top_row + 1
        plot_bottom = top_row + graph_height - 1
        plot_height = max(1, plot_bottom - plot_top + 1)
        plot_width = max(1, graph_width - 2)
        values = list(series)[-plot_width:]
        if len(values) < plot_width:
            values = ([0.0] * (plot_width - len(values))) + values
        latest = float(values[-1] if values else 0.0)

        title_text = f"{title} {latest:5.1f}%"
        self._safe_addnstr(stdscr, top_row, left_col, title_text, graph_width, curses.A_BOLD)
        self._safe_addnstr(stdscr, plot_top - 1, left_col + graph_width - 7, "100%", 6, curses.A_DIM)
        self._safe_addnstr(stdscr, plot_bottom, left_col + graph_width - 5, "0%", 4, curses.A_DIM)

        for col_idx in range(plot_width):
            value = max(0.0, min(100.0, float(values[col_idx])))
            filled = int(round((value / 100.0) * float(plot_height)))
            color = curses.color_pair(self._usage_color_pair(value, metric))
            x = left_col + 1 + col_idx
            for y in range(plot_height):
                row = plot_bottom - y
                if y < filled:
                    self._safe_addnstr(stdscr, row, x, "#", 1, color)
                else:
                    self._safe_addnstr(stdscr, row, x, ".", 1, curses.A_DIM)

    def _draw_resources_tab(self, stdscr, top_row: int, first_row: int, row_count: int, width: int):
        if row_count <= 2:
            return
        cpu_series = list(self._resource_cpu_history)
        ram_series = list(self._resource_ram_history)
        if not cpu_series:
            cpu_series = [0.0]
        if not ram_series:
            ram_series = [0.0]

        graph_height = max(4, min(10, row_count - 5))
        left_width = max(18, (width - 2) // 2)
        right_width = max(18, width - left_width - 1)
        left_col = 0
        right_col = left_width + 1

        self._draw_usage_graph(
            stdscr,
            first_row,
            left_col,
            left_width,
            graph_height,
            cpu_series,
            "CPU Usage",
            "cpu",
        )
        self._draw_usage_graph(
            stdscr,
            first_row,
            right_col,
            right_width,
            graph_height,
            ram_series,
            "RAM Usage",
            "ram",
        )

        sample_count = min(len(cpu_series), len(ram_series))
        cpu_now = float(cpu_series[-1] if cpu_series else 0.0)
        cpu_avg = (sum(cpu_series) / float(len(cpu_series))) if cpu_series else 0.0
        ram_now = float(ram_series[-1] if ram_series else 0.0)
        ram_avg = (sum(ram_series) / float(len(ram_series))) if ram_series else 0.0
        mem_total_mb = float(self._system_mem_total_kb or 0) / 1024.0
        summary_row = first_row + graph_height + 1
        summary = (
            f"Tick {self._resource_sample_interval:.1f}s | Samples {sample_count} | "
            f"CPU now/avg {cpu_now:.1f}/{cpu_avg:.1f}% | "
            f"RAM now/avg {ram_now:.1f}/{ram_avg:.1f}%"
        )
        if mem_total_mb > 0.0:
            summary += f" | Total RAM {mem_total_mb:.0f}MB"
        self._safe_addnstr(stdscr, summary_row, 0, summary, width - 1, curses.A_DIM)

        table_row = summary_row + 1
        remaining_rows = max(0, (first_row + row_count) - table_row)
        if remaining_rows < 2:
            return
        self._safe_addnstr(
            stdscr,
            table_row,
            0,
            "Top Services by CPU (live): Service         CPU%   RAM(MB)   State",
            width - 1,
            curses.A_BOLD | curses.color_pair(1),
        )
        rows_left = remaining_rows - 1
        if rows_left <= 0:
            return
        ranked = []
        for svc in self.services:
            runtime = self.runtime_by_id.get(svc.service_id)
            if not runtime:
                continue
            ranked.append((float(runtime.cpu_percent), float(runtime.rss_mb), svc.label, runtime.state))
        ranked.sort(key=lambda item: (-item[0], -item[1], str(item[2]).lower()))
        for idx, item in enumerate(ranked[:rows_left]):
            cpu, rss, label, state = item
            color = curses.color_pair(self._usage_color_pair(cpu, "cpu"))
            line = f"{label[:14]:14}  {cpu:5.1f}  {rss:8.1f}   {str(state).upper()[:10]}"
            self._safe_addnstr(stdscr, table_row + 1 + idx, 0, line, width - 1, color)

    def _prompt_text_input(self, stdscr, prompt: str, initial: str = "") -> Optional[str]:
        try:
            max_y, max_x = stdscr.getmaxyx()
            row = max(0, max_y - 2)
            col = min(max_x - 2, max(0, len(prompt) + 1))
            self._safe_addnstr(stdscr, row, 0, " " * (max_x - 1), max_x - 1)
            self._safe_addnstr(stdscr, row, 0, prompt, max_x - 1, curses.A_BOLD)
            stdscr.nodelay(False)
            stdscr.timeout(-1)
            curses.echo()
            curses.curs_set(1)
            self._safe_addnstr(stdscr, row, col, initial, max(1, max_x - col - 1))
            stdscr.move(row, col + len(str(initial)))
            stdscr.refresh()
            raw = stdscr.getstr(row, col, max(1, max_x - col - 2))
            if raw is None:
                return None
            return raw.decode("utf-8", errors="ignore").strip()
        except Exception:
            return None
        finally:
            try:
                curses.noecho()
                curses.curs_set(0)
            except Exception:
                pass
            try:
                stdscr.nodelay(True)
                stdscr.timeout(0)
            except Exception:
                pass

    def _open_cpu_settings_modal(self, stdscr):
        with self._lock:
            local_cfg = dict(self._cpu_throttle)
            available_governors = list(self._cpu_available_governors)
            throttle_supported = bool(self._cpu_throttle_supported)
            hw_min = int(self._cpu_hw_min_khz or 0)
            hw_max = int(self._cpu_hw_max_khz or 0)
            current_governor = str(self._cpu_current_governor or "")
            current_min = int(self._cpu_current_min_khz or 0)
            current_max = int(self._cpu_current_max_khz or 0)
            current_undervolt = int(self._cpu_current_over_voltage_delta_uv or 0)
            firmware_cfg_path = str(self._firmware_config_path or "")
            last_apply = str(self._cpu_throttle_last_apply or "")

        if not available_governors:
            available_governors = ["ondemand", "performance", "schedutil", "powersave"]
        selected = 0
        options = [
            "Governor",
            "Min Frequency (kHz)",
            "Max Frequency (kHz)",
            "CPU Undervolt Delta (uV)",
            "Auto Apply On Start",
            "ASR Throttle On LLM",
            "ASR Throttle Percent",
            "ASR Throttle Cycle (ms)",
            "Apply Now",
            "Save",
            "Cancel",
        ]

        while True:
            stdscr.erase()
            max_y, max_x = stdscr.getmaxyx()
            if max_y < 14 or max_x < 70:
                self._safe_addnstr(
                    stdscr,
                    0,
                    0,
                    "Terminal too small for CPU settings modal",
                    max_x - 1,
                    curses.color_pair(5) | curses.A_BOLD,
                )
                stdscr.refresh()
                ch = stdscr.getch()
                if ch in (27, ord("q"), ord("Q"), 10, 13):
                    return
                continue

            title = " CPU Throttle Settings "
            subtitle = "Up/Down select | Enter edit/apply | S save | Esc cancel"
            self._safe_addnstr(stdscr, 0, 0, title, max_x - 1, curses.A_BOLD | curses.color_pair(1))
            self._safe_addnstr(stdscr, 1, 0, subtitle, max_x - 1, curses.A_DIM)

            support_text = "available" if throttle_supported else "unavailable"
            self._safe_addnstr(
                stdscr,
                3,
                0,
                f"cpufreq support: {support_text} | current governor={current_governor or '-'} "
                f"min={current_min or '-'} max={current_max or '-'}",
                max_x - 1,
            )
            self._safe_addnstr(
                stdscr,
                4,
                0,
                f"hardware limits: min={hw_min or '-'} max={hw_max or '-'}",
                max_x - 1,
            )
            self._safe_addnstr(
                stdscr,
                5,
                0,
                f"firmware config: {firmware_cfg_path or '-'} | "
                f"current over_voltage_delta={current_undervolt}uV",
                max_x - 1,
            )

            row = 7
            values = [
                str(local_cfg.get("governor", "")),
                str(local_cfg.get("min_khz", "")),
                str(local_cfg.get("max_khz", "")),
                str(local_cfg.get("over_voltage_delta_uv", DEFAULT_OVER_VOLTAGE_DELTA_UV)),
                "ON" if bool(local_cfg.get("auto_apply", False)) else "OFF",
                "ON" if bool(local_cfg.get("asr_llm_throttle_enable", True)) else "OFF",
                str(local_cfg.get("asr_llm_throttle_percent", 65)),
                str(local_cfg.get("asr_llm_throttle_cycle_ms", 320)),
                "Apply desired values now",
                "Save desired values",
                "Discard changes",
            ]
            for idx, option in enumerate(options):
                prefix = ">" if idx == selected else " "
                attr = curses.A_REVERSE if idx == selected else 0
                line = f"{prefix} {option:23} : {values[idx]}"
                self._safe_addnstr(stdscr, row + idx, 0, line, max_x - 1, attr)

            if last_apply:
                self._safe_addnstr(stdscr, max_y - 3, 0, f"Last apply: {last_apply}", max_x - 1, curses.A_DIM)
            self._safe_addnstr(
                stdscr,
                max_y - 2,
                0,
                f"Desired summary: governor={local_cfg.get('governor')} "
                f"min={local_cfg.get('min_khz')} max={local_cfg.get('max_khz')} "
                f"ov={local_cfg.get('over_voltage_delta_uv')}uV "
                f"auto={'on' if local_cfg.get('auto_apply') else 'off'}",
                max_x - 1,
            )
            stdscr.refresh()

            ch = stdscr.getch()
            if ch in (27, ord("q"), ord("Q")):
                return
            if ch in (curses.KEY_UP, ord("k"), ord("K")):
                selected = max(0, selected - 1)
                continue
            if ch in (curses.KEY_DOWN, ord("j"), ord("J")):
                selected = min(len(options) - 1, selected + 1)
                continue
            if ch in (ord("s"), ord("S")):
                with self._lock:
                    self._cpu_throttle.update(local_cfg)
                    self._normalize_cpu_throttle_settings()
                    self._save_desired_state()
                    self._log(
                        "[CPU] Saved desired throttle "
                        f"gov={self._cpu_throttle['governor']} "
                        f"min={self._cpu_throttle['min_khz']} "
                        f"max={self._cpu_throttle['max_khz']} "
                        f"ov={self._cpu_throttle['over_voltage_delta_uv']}uV "
                        f"auto={'on' if self._cpu_throttle['auto_apply'] else 'off'}"
                    )
                return
            if ch not in (10, 13, curses.KEY_ENTER):
                continue

            if selected == 0:
                try:
                    idx = available_governors.index(str(local_cfg.get("governor", "")))
                except ValueError:
                    idx = -1
                local_cfg["governor"] = available_governors[(idx + 1) % len(available_governors)]
            elif selected == 1:
                typed = self._prompt_text_input(
                    stdscr,
                    "Min kHz: ",
                    str(local_cfg.get("min_khz", "")),
                )
                if typed:
                    parsed = self._parse_int(typed, int(local_cfg.get("min_khz", 0) or 0))
                    local_cfg["min_khz"] = max(100000, parsed)
            elif selected == 2:
                typed = self._prompt_text_input(
                    stdscr,
                    "Max kHz: ",
                    str(local_cfg.get("max_khz", "")),
                )
                if typed:
                    parsed = self._parse_int(typed, int(local_cfg.get("max_khz", 0) or 0))
                    local_cfg["max_khz"] = max(100000, parsed)
            elif selected == 3:
                typed = self._prompt_text_input(
                    stdscr,
                    f"Undervolt delta uV ({MIN_OVER_VOLTAGE_DELTA_UV}..{MAX_OVER_VOLTAGE_DELTA_UV}): ",
                    str(local_cfg.get("over_voltage_delta_uv", DEFAULT_OVER_VOLTAGE_DELTA_UV)),
                )
                if typed:
                    parsed = self._parse_int(
                        typed,
                        int(local_cfg.get("over_voltage_delta_uv", DEFAULT_OVER_VOLTAGE_DELTA_UV)),
                    )
                    local_cfg["over_voltage_delta_uv"] = max(
                        MIN_OVER_VOLTAGE_DELTA_UV,
                        min(MAX_OVER_VOLTAGE_DELTA_UV, parsed),
                    )
            elif selected == 4:
                local_cfg["auto_apply"] = not bool(local_cfg.get("auto_apply", False))
            elif selected == 5:
                local_cfg["asr_llm_throttle_enable"] = not bool(local_cfg.get("asr_llm_throttle_enable", True))
            elif selected == 6:
                typed = self._prompt_text_input(
                    stdscr,
                    "ASR throttle percent (0-95): ",
                    str(local_cfg.get("asr_llm_throttle_percent", "65")),
                )
                if typed:
                    parsed = self._parse_int(typed, int(local_cfg.get("asr_llm_throttle_percent", 65) or 65))
                    local_cfg["asr_llm_throttle_percent"] = max(0, min(95, parsed))
            elif selected == 7:
                typed = self._prompt_text_input(
                    stdscr,
                    "ASR throttle cycle ms (80-2000): ",
                    str(local_cfg.get("asr_llm_throttle_cycle_ms", "320")),
                )
                if typed:
                    parsed = self._parse_int(typed, int(local_cfg.get("asr_llm_throttle_cycle_ms", 320) or 320))
                    local_cfg["asr_llm_throttle_cycle_ms"] = max(80, min(2000, parsed))
            elif selected == 8:
                with self._lock:
                    self._cpu_throttle.update(local_cfg)
                    self._normalize_cpu_throttle_settings()
                    ok, detail = self._apply_cpu_throttle_settings()
                    self._cpu_throttle_last_apply = detail
                    last_apply = detail
                    self._save_desired_state()
                    if ok:
                        self._log(f"[CPU] Applied throttle: {detail}")
                    else:
                        self._log(f"[WARN] CPU throttle apply failed: {detail}")
                    current_governor = str(self._cpu_current_governor or "")
                    current_min = int(self._cpu_current_min_khz or 0)
                    current_max = int(self._cpu_current_max_khz or 0)
                    current_undervolt = int(self._cpu_current_over_voltage_delta_uv or 0)
                    firmware_cfg_path = str(self._firmware_config_path or "")
                    throttle_supported = bool(self._cpu_throttle_supported)
                    hw_min = int(self._cpu_hw_min_khz or 0)
                    hw_max = int(self._cpu_hw_max_khz or 0)
            elif selected == 9:
                with self._lock:
                    self._cpu_throttle.update(local_cfg)
                    self._normalize_cpu_throttle_settings()
                    self._save_desired_state()
                    self._log(
                        "[CPU] Saved desired throttle "
                        f"gov={self._cpu_throttle['governor']} "
                        f"min={self._cpu_throttle['min_khz']} "
                        f"max={self._cpu_throttle['max_khz']} "
                        f"ov={self._cpu_throttle['over_voltage_delta_uv']}uV "
                        f"auto={'on' if self._cpu_throttle['auto_apply'] else 'off'}"
                    )
                return
            elif selected == 10:
                return

            if int(local_cfg.get("min_khz", 0) or 0) > int(local_cfg.get("max_khz", 0) or 0):
                a = int(local_cfg.get("min_khz", 0) or 0)
                b = int(local_cfg.get("max_khz", 0) or 0)
                local_cfg["min_khz"] = min(a, b)
                local_cfg["max_khz"] = max(a, b)
            local_cfg["over_voltage_delta_uv"] = max(
                MIN_OVER_VOLTAGE_DELTA_UV,
                min(
                    MAX_OVER_VOLTAGE_DELTA_UV,
                    int(local_cfg.get("over_voltage_delta_uv", DEFAULT_OVER_VOLTAGE_DELTA_UV) or 0),
                ),
            )

    def _build_exit_report(self, interrupted: bool = False) -> str:
        now = time.time()
        with self._lock:
            rows = []
            for svc in self.services:
                runtime = self.runtime_by_id[svc.service_id]
                if runtime.stopped_at > runtime.started_at > 0 and runtime.state not in (
                    "running",
                    "degraded",
                    "activating",
                    "launching",
                ):
                    active_for = runtime.stopped_at - runtime.started_at
                elif runtime.started_at > 0:
                    active_for = now - runtime.started_at
                else:
                    active_for = 0.0
                rows.append(
                    {
                        "label": svc.label,
                        "desired": "ON" if runtime.desired_enabled else "OFF",
                        "state": runtime.state.upper(),
                        "pid": runtime.pid or "-",
                        "uptime": self._format_duration(active_for),
                        "starts": runtime.start_count,
                        "stops": runtime.stop_count,
                        "restarts": runtime.restart_count,
                        "crashes": runtime.crash_count,
                        "detail": self._runtime_detail(runtime, now),
                    }
                )
            recent_logs = list(self._logs)[-30:]

        run_seconds = max(0.0, now - self._run_started_at)
        header = (
            "Service         Desired  State      PID      Uptime    Starts  Stops  Restarts  Crashes  Last Event / Error"
        )
        lines = []
        lines.append("=" * len(header))
        lines.append("Teleoperation Watchdog Exit Summary")
        lines.append(f"Ended: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Interrupted: {'yes' if interrupted else 'no'}")
        lines.append(f"Session runtime: {self._format_duration(run_seconds)}")
        lines.append("-" * len(header))
        lines.append(header)
        lines.append("-" * len(header))
        for row in rows:
            lines.append(
                f"{row['label'][:14]:14}  "
                f"{row['desired'][:3]:>3}      "
                f"{row['state'][:9]:9}  "
                f"{str(row['pid'])[:8]:8}  "
                f"{row['uptime']:8}  "
                f"{int(row['starts']):6}  "
                f"{int(row['stops']):5}  "
                f"{int(row['restarts']):8}  "
                f"{int(row['crashes']):7}  "
                f"{row['detail']}"
            )
        lines.append("-" * len(header))
        lines.append("Recent Watchdog Logs:")
        if recent_logs:
            lines.extend(recent_logs)
        else:
            lines.append("(no logs)")
        lines.append("=" * len(header))
        return "\n".join(lines)

    def _draw_ui(self, stdscr):
        stdscr.erase()
        height, width = stdscr.getmaxyx()
        now = time.time()
        self._click_regions = []

        if height < 14 or width < 96:
            self._safe_addnstr(
                stdscr,
                0,
                0,
                "Terminal too small for watchdog dashboard (need >= 96x14)",
                width - 1,
                curses.A_BOLD | curses.color_pair(5),
            )
            stdscr.refresh()
            return

        title = " Teleoperation Watchdog "
        info = (
            "Up/Down select | Space toggle | R restart | A toggle all | "
            "C copy link | L log links | S CPU settings | <-/-> tabs | Q quit"
        )
        self._safe_addnstr(stdscr, 0, 0, title, width - 1, curses.A_BOLD | curses.color_pair(1))
        self._safe_addnstr(stdscr, 1, 0, info, width - 1, curses.A_DIM)
        self._safe_addnstr(stdscr, 2, 0, self._resource_summary_line(), width - 1, curses.A_DIM)
        last_links_age = "-"
        if self._service_link_last_refresh_at > 0:
            last_links_age = self._format_duration(now - self._service_link_last_refresh_at)
        self._safe_addnstr(
            stdscr,
            3,
            0,
            f"Service links: click [copy] or link text to copy full URL | Last refresh: {last_links_age}",
            width - 1,
            curses.A_DIM,
        )

        footer_row = height - 1
        table_header_row = 4
        first_service_row = 5
        available_body_rows = max(6, footer_row - first_service_row - 3)
        table_rows = max(4, available_body_rows // 2)
        max_table_rows = max(3, footer_row - first_service_row - 5)
        table_rows = min(table_rows, max_table_rows)
        table_end_row = first_service_row + table_rows
        log_header_row = table_end_row + 1
        log_first_row = log_header_row + 1
        log_rows = max(1, footer_row - log_first_row)

        header = "Sel  Service         Desired  State      PID      Uptime    CPU%   RAM(MB)  Last Event / Error + Link"
        self._safe_addnstr(stdscr, table_header_row, 0, header, width - 1, curses.A_BOLD | curses.color_pair(1))

        visible_services = max(1, table_end_row - first_service_row)
        start_idx = 0
        if self._selected_index >= visible_services:
            start_idx = self._selected_index - visible_services + 1
        end_idx = min(len(self.services), start_idx + visible_services)

        row = first_service_row
        for idx in range(start_idx, end_idx):
            svc = self.services[idx]
            runtime = self.runtime_by_id[svc.service_id]
            selected = ">" if idx == self._selected_index else " "
            desired = "ON " if runtime.desired_enabled else "OFF"
            pid_text = str(runtime.pid) if runtime.pid else "-"
            uptime = self._format_duration(now - runtime.started_at) if runtime.started_at > 0 else "-"
            cpu_text = f"{runtime.cpu_percent:5.1f}" if runtime.cpu_percent > 0 else "  0.0"
            mem_text = f"{runtime.rss_mb:7.1f}" if runtime.rss_mb > 0 else "    0.0"
            state_text = runtime.state.upper()
            message = self._runtime_detail(runtime, now)
            link_data = self._service_links.get(svc.service_id, {})
            link_url = str(link_data.get("primary_url") or "").strip()
            link_display = self._shorten_url(link_url, max_len=max(16, min(46, width // 3)))
            copy_tag = "[copy]" if link_url else "[----]"
            prefix = (
                f"{selected:1}    "
                f"{svc.label[:14]:14}  "
                f"{desired:>3}      "
                f"{state_text[:9]:9}  "
                f"{pid_text[:8]:8}  "
                f"{uptime:8}  "
                f"{cpu_text:>5}  "
                f"{mem_text:>7}  "
            )
            min_message_space = 14
            reserved_for_link = len(copy_tag) + (1 + len(link_display) if link_display else 0) + 3
            message_space = max(
                min_message_space,
                width - len(prefix) - reserved_for_link - 2,
            )
            message_text = str(message or "")
            if len(message_text) > message_space:
                message_text = message_text[: max(0, message_space - 3)] + "..."
            line = f"{prefix}{message_text}  {copy_tag}"
            if link_display:
                line += f" {link_display}"
            attrs = curses.color_pair(self._state_color(runtime.state))
            if idx == self._selected_index:
                attrs |= curses.A_REVERSE
            self._safe_addnstr(stdscr, row, 0, line, width - 1, attrs)
            if link_url:
                copy_col = line.rfind(copy_tag)
                if copy_col >= 0:
                    self._register_click_region(
                        self._click_regions,
                        row,
                        copy_col,
                        copy_col + len(copy_tag) - 1,
                        link_url,
                    )
                if link_display:
                    link_col = line.rfind(link_display)
                    if link_col >= 0:
                        self._register_click_region(
                            self._click_regions,
                            row,
                            link_col,
                            link_col + len(link_display) - 1,
                            link_url,
                        )
            row += 1
            if row >= table_end_row:
                break

        if start_idx > 0:
            self._safe_addnstr(
                stdscr,
                first_service_row,
                max(0, width - 12),
                f"^ +{start_idx}",
                11,
                curses.A_DIM,
            )
        hidden_below = max(0, len(self.services) - end_idx)
        if hidden_below > 0:
            self._safe_addnstr(
                stdscr,
                table_end_row - 1,
                max(0, width - 12),
                f"v +{hidden_below}",
                11,
                curses.A_DIM,
            )

        separator = "-" * max(1, width - 1)
        self._safe_addnstr(stdscr, table_end_row, 0, separator, width - 1, curses.color_pair(1))
        tab_prefix = "[<- / ->] Tabs: "
        self._safe_addnstr(stdscr, log_header_row, 0, tab_prefix, width - 1, curses.A_BOLD | curses.color_pair(1))
        tab_col = len(tab_prefix)
        for idx, tab_name in enumerate(self._bottom_tabs):
            label = str(tab_name or "").strip().capitalize()
            if idx > 0:
                self._safe_addnstr(stdscr, log_header_row, tab_col, "| ", width - tab_col - 1, curses.A_DIM)
                tab_col += 2
            tab_attr = curses.A_BOLD | curses.color_pair(1)
            if idx == self._active_bottom_tab:
                tab_attr |= curses.A_REVERSE
            token = f" {label} "
            self._safe_addnstr(stdscr, log_header_row, tab_col, token, width - tab_col - 1, tab_attr)
            tab_col += len(token)

        active_tab = str(self._bottom_tabs[self._active_bottom_tab]).strip().lower()
        if active_tab == "resources":
            self._draw_resources_tab(stdscr, log_header_row, log_first_row, log_rows, width)
        else:
            logs = list(self._logs)[-log_rows:]
            for i, line in enumerate(logs):
                self._safe_addnstr(stdscr, log_first_row + i, 0, line, width - 1)

        selected_service = self.services[self._selected_index]
        selected_runtime = self.runtime_by_id[selected_service.service_id]
        selected_link_data = self._service_links.get(selected_service.service_id, {})
        selected_link = str(selected_link_data.get("primary_url") or "").strip()
        selected_link_text = self._shorten_url(selected_link, max_len=max(22, width // 3)) if selected_link else "(none)"
        active_tab_label = str(self._bottom_tabs[self._active_bottom_tab]).strip().capitalize()
        status = (
            f"Selected: {selected_service.label} | "
            f"Tab: {active_tab_label} | "
            f"State: {selected_runtime.state.upper()} | "
            f"Desired: {'ON' if selected_runtime.desired_enabled else 'OFF'} | "
            f"CPU {selected_runtime.cpu_percent:.1f}% | RAM {selected_runtime.rss_mb:.1f}MB | "
            f"Link: {selected_link_text}"
        )
        clock = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        footer = f"{status} | {clock}"
        self._safe_addnstr(stdscr, footer_row, 0, " " * (width - 1), width - 1, curses.A_REVERSE)
        self._safe_addnstr(stdscr, footer_row, 0, footer, width - 1, curses.A_REVERSE)
        if selected_link:
            link_token = f"Link: {selected_link_text}"
            token_col = footer.find(link_token)
            if token_col >= 0:
                self._register_click_region(
                    self._click_regions,
                    footer_row,
                    token_col,
                    token_col + len(link_token) - 1,
                    selected_link,
                )
        stdscr.refresh()

    def run_curses(self) -> bool:
        self.start()
        interrupted = False

        def _inner(stdscr):
            curses.curs_set(0)
            stdscr.nodelay(True)
            stdscr.keypad(True)
            try:
                curses.mousemask(curses.ALL_MOUSE_EVENTS | curses.REPORT_MOUSE_POSITION)
            except Exception:
                pass
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_WHITE, -1)
            curses.init_pair(2, curses.COLOR_GREEN, -1)
            curses.init_pair(3, curses.COLOR_YELLOW, -1)
            curses.init_pair(4, curses.COLOR_MAGENTA, -1)
            curses.init_pair(5, curses.COLOR_RED, -1)
            curses.init_pair(6, curses.COLOR_CYAN, -1)

            while True:
                with self._lock:
                    self._draw_ui(stdscr)
                    if self._restart_requested:
                        return

                key = stdscr.getch()
                if key == -1:
                    time.sleep(0.06)
                    continue

                open_cpu_settings = False
                copy_selected_link = False
                log_all_links = False
                with self._lock:
                    if key in (ord("q"), ord("Q")):
                        self._log("[USER] Quit requested")
                        return
                    if key in (curses.KEY_UP, ord("k"), ord("K")):
                        self._selected_index = max(0, self._selected_index - 1)
                    elif key in (curses.KEY_DOWN, ord("j"), ord("J")):
                        self._selected_index = min(len(self.services) - 1, self._selected_index + 1)
                    elif key == curses.KEY_LEFT:
                        self._active_bottom_tab = (self._active_bottom_tab - 1) % len(self._bottom_tabs)
                    elif key == curses.KEY_RIGHT:
                        self._active_bottom_tab = (self._active_bottom_tab + 1) % len(self._bottom_tabs)
                    elif key == ord(" "):
                        self.toggle_selected()
                    elif key in (ord("r"), ord("R")):
                        self.restart_selected()
                    elif key in (ord("a"), ord("A")):
                        self.toggle_all()
                    elif key in (ord("c"), ord("C")):
                        copy_selected_link = True
                    elif key in (ord("l"), ord("L")):
                        log_all_links = True
                    elif key in (ord("s"), ord("S")):
                        open_cpu_settings = True

                if key == curses.KEY_MOUSE:
                    self._handle_mouse_event()
                if copy_selected_link:
                    self._copy_selected_service_link()
                if log_all_links:
                    self._log_all_service_links()

                if open_cpu_settings:
                    self._open_cpu_settings_modal(stdscr)

                time.sleep(0.02)

        try:
            curses.wrapper(_inner)
        except KeyboardInterrupt:
            interrupted = True
            with self._lock:
                self._log("[USER] Ctrl+C interrupt received")
        finally:
            self.request_shutdown()
            self.await_shutdown()
            self._release_instance_lock()
            if self.should_restart():
                reason = self.restart_reason() or "remote update pulled"
                print(f"\n[WATCHDOG] Auto-update applied. Restarting ({reason})", flush=True)
            else:
                report = self._build_exit_report(interrupted=interrupted)
                print("\n" + report, flush=True)

        return self.should_restart()


def main():
    try:
        manager = WatchdogManager(pathlib.Path(__file__).resolve().parent)
    except RuntimeError as exc:
        print(f"[WATCHDOG] {exc}", flush=True)
        return 1
    should_restart = manager.run_curses()
    if should_restart:
        reason = manager.restart_reason() or "remote update pulled"
        python_path = sys.executable
        script_path = str(pathlib.Path(__file__).resolve())
        argv = [python_path, script_path] + sys.argv[1:]
        print(f"[WATCHDOG] Relaunching after update ({reason})", flush=True)
        try:
            os.execv(python_path, argv)
        except Exception as exc:
            print(f"[WATCHDOG] Failed to relaunch watchdog: {exc}", flush=True)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
