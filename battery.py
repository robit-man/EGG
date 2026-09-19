#!/usr/bin/env python3
"""EGG battery sampler and Ubuntu top-bar indicator.

The system daemon owns the serial sensor and atomically publishes a small,
world-readable status document.  The user-session indicator only reads that
document, so other services and applications can consume the same state
without racing for the serial port or scraping a terminal.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import select
import signal
import tempfile
import termios
import time
from pathlib import Path
from typing import Any

DEFAULT_STATE_FILE = Path("/run/egg-battery/status.json")
DEFAULT_BAUD = 9600
FULLY_CHARGED_VOLTAGE = 29.4
EMPTY_VOLTAGE = 22.0
ADC_CALIBRATION_COUNT = 3030.0
ADC_CALIBRATION_VOLTAGE = 29.0

LOGGER = logging.getLogger("egg-battery")
STOP = False


def calculate_voltage(adc_value: int) -> float:
    """Convert the calibrated ESP32 ADC reading to pack voltage."""

    return float(adc_value) / ADC_CALIBRATION_COUNT * ADC_CALIBRATION_VOLTAGE


def calculate_percentage(voltage: float) -> int:
    """Map the configured 7S pack range onto a bounded percentage."""

    span = FULLY_CHARGED_VOLTAGE - EMPTY_VOLTAGE
    return max(0, min(100, round((voltage - EMPTY_VOLTAGE) / span * 100)))


def _stop(_signum: int, _frame: Any) -> None:
    global STOP
    STOP = True


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(value, output, ensure_ascii=False, separators=(",", ":"))
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _device_candidates(explicit: str) -> list[Path]:
    if explicit and explicit != "auto":
        return [Path(explicit)]
    candidates: list[Path] = []
    by_id = Path("/dev/serial/by-id")
    if by_id.is_dir():
        candidates.extend(sorted(by_id.iterdir()))
    candidates.extend(sorted(Path("/dev").glob("ttyACM*")))
    candidates.extend(sorted(Path("/dev").glob("ttyUSB*")))
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = str(candidate.resolve())
        if resolved not in seen:
            unique.append(candidate)
            seen.add(resolved)
    return unique


def _open_serial(path: Path, baud: int) -> int:
    speeds = {
        9600: termios.B9600,
        19200: termios.B19200,
        38400: termios.B38400,
        57600: termios.B57600,
        115200: termios.B115200,
    }
    speed = speeds.get(baud)
    if speed is None:
        raise ValueError(f"unsupported baud rate {baud}")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOCTTY | os.O_NONBLOCK)
    try:
        attributes = termios.tcgetattr(descriptor)
        attributes[0] = termios.IGNPAR
        attributes[1] = 0
        attributes[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
        attributes[3] = 0
        attributes[4] = speed
        attributes[5] = speed
        attributes[6][termios.VMIN] = 0
        attributes[6][termios.VTIME] = 0
        termios.tcflush(descriptor, termios.TCIFLUSH)
        termios.tcsetattr(descriptor, termios.TCSANOW, attributes)
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _status(
    *,
    available: bool,
    device: str = "",
    adc: int | None = None,
    error: str = "",
) -> dict[str, Any]:
    now = time.time()
    value: dict[str, Any] = {
        "schema": "robit.egg.battery.v1",
        "available": available,
        "source": "esp32_serial_adc",
        "device": device,
        "updated_at": now,
        "updated_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(now)),
        "stale_after_seconds": 15,
        "charging": None,
    }
    if adc is not None:
        voltage = calculate_voltage(adc)
        value.update(
            {
                "raw_adc": adc,
                "voltage_v": round(voltage, 2),
                "percentage": calculate_percentage(voltage),
            }
        )
    if error:
        value["error"] = error[:500]
    return value


def run_daemon(device: str, baud: int, state_file: Path) -> int:
    """Reconnect forever, publishing both valid readings and sensor failures."""

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    descriptor: int | None = None
    selected = ""
    buffer = bytearray()
    last_publish = 0.0
    last_adc: int | None = None
    last_error = "battery sensor not connected"
    retry_at = 0.0

    while not STOP:
        now = time.monotonic()
        if descriptor is None and now >= retry_at:
            for candidate in _device_candidates(device):
                try:
                    descriptor = _open_serial(candidate, baud)
                    selected = str(candidate.resolve())
                    buffer.clear()
                    last_error = ""
                    LOGGER.info("reading battery ADC from %s at %d baud", selected, baud)
                    break
                except (OSError, ValueError) as exc:
                    last_error = f"{candidate}: {exc}"
            retry_at = now + 2.0
        if descriptor is not None:
            try:
                readable, _, _ = select.select([descriptor], [], [], 0.5)
                if readable:
                    chunk = os.read(descriptor, 4096)
                    if not chunk:
                        raise OSError("serial device closed")
                    buffer.extend(chunk)
                    while b"\n" in buffer:
                        line, _, remainder = buffer.partition(b"\n")
                        buffer = bytearray(remainder)
                        try:
                            candidate_adc = int(line.decode("ascii", errors="strict").strip())
                        except (UnicodeDecodeError, ValueError):
                            continue
                        if 0 <= candidate_adc <= 4095:
                            last_adc = candidate_adc
                            last_error = ""
            except (OSError, ValueError) as exc:
                last_error = str(exc)
                LOGGER.warning("battery sensor disconnected: %s", exc)
                os.close(descriptor)
                descriptor = None
                selected = ""
                retry_at = time.monotonic() + 1.0
        else:
            time.sleep(0.25)

        now = time.monotonic()
        if now - last_publish >= 1.0:
            available = descriptor is not None and last_adc is not None and not last_error
            _atomic_json(
                state_file,
                _status(
                    available=available,
                    device=selected,
                    adc=last_adc if available else None,
                    error=last_error if not available else "",
                ),
            )
            last_publish = now

    if descriptor is not None:
        os.close(descriptor)
    return 0


def _read_status(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"available": False, "error": "battery service has not published state"}
    if not isinstance(value, dict):
        return {"available": False, "error": "battery state is invalid"}
    try:
        age = time.time() - float(value.get("updated_at") or 0)
        stale_after = float(value.get("stale_after_seconds") or 15)
    except (TypeError, ValueError):
        age, stale_after = 9999.0, 15.0
    if age > stale_after:
        return {**value, "available": False, "error": "battery reading is stale"}
    return value


def run_indicator(state_file: Path) -> int:
    """Render the daemon's state in Ubuntu's AppIndicator-compatible top bar."""

    try:
        import gi

        gi.require_version("Gtk", "3.0")
        try:
            gi.require_version("AyatanaAppIndicator3", "0.1")
            from gi.repository import AyatanaAppIndicator3 as AppIndicator
        except (ImportError, ValueError):
            gi.require_version("AppIndicator3", "0.1")
            from gi.repository import AppIndicator3 as AppIndicator
        from gi.repository import GLib, Gtk
    except (ImportError, ValueError) as exc:
        LOGGER.error("Ubuntu AppIndicator support is unavailable: %s", exc)
        return 1

    indicator = AppIndicator.Indicator.new(
        "egg-battery", "battery-missing-symbolic", AppIndicator.IndicatorCategory.HARDWARE
    )
    indicator.set_status(AppIndicator.IndicatorStatus.ACTIVE)
    menu = Gtk.Menu()
    summary = Gtk.MenuItem(label="Battery service starting…")
    summary.set_sensitive(False)
    menu.append(summary)
    sensor = Gtk.MenuItem(label="Sensor: waiting")
    sensor.set_sensitive(False)
    menu.append(sensor)
    menu.show_all()
    indicator.set_menu(menu)

    def refresh() -> bool:
        state = _read_status(state_file)
        if state.get("available"):
            percentage = max(0, min(100, int(state.get("percentage") or 0)))
            level = min(100, max(0, round(percentage / 10) * 10))
            icon = f"battery-level-{level}-symbolic"
            voltage = float(state.get("voltage_v") or 0)
            label = f"{percentage}%"
            detail = f"{percentage}% — {voltage:.2f} V"
            summary.set_label(detail)
            sensor.set_label(f"Sensor: {state.get('device') or 'serial ADC'}")
        else:
            icon = "battery-missing-symbolic"
            label = "?%"
            detail = str(state.get("error") or "Battery unavailable")
            summary.set_label(detail[:120])
            sensor.set_label("Sensor: unavailable")
        try:
            indicator.set_icon_full(icon, detail)
        except AttributeError:
            indicator.set_icon(icon)
        indicator.set_label(label, "100%")
        return True

    refresh()
    GLib.timeout_add_seconds(2, refresh)
    Gtk.main()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--daemon", action="store_true", help="sample and publish battery state")
    mode.add_argument("--indicator", action="store_true", help="show the desktop indicator")
    mode.add_argument("--status", action="store_true", help="print current JSON state")
    parser.add_argument("--device", default=os.environ.get("EGG_BATTERY_DEVICE", "auto"))
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument(
        "--state-file",
        type=Path,
        default=Path(os.environ.get("EGG_BATTERY_STATE", DEFAULT_STATE_FILE)),
    )
    arguments = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    if arguments.daemon:
        return run_daemon(arguments.device, arguments.baud, arguments.state_file)
    if arguments.indicator:
        return run_indicator(arguments.state_file)
    print(json.dumps(_read_status(arguments.state_file), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
