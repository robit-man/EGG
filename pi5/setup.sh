#!/bin/bash

set -e

# --------------------------- Configuration ---------------------------

REPO_URL="https://github.com/robit-man/EGG.git"
REPO_BRANCH="main"

TARGET_DIR="/home/$(whoami)/voice"
REPO_DIR="$TARGET_DIR/EGG_repo"
SYNC_SCRIPT="$TARGET_DIR/sync_runtime.sh"

WHISPERCPP_DIR="$TARGET_DIR/whispercpp"
VENV_NAME="whisper"
VENV_DIR="$WHISPERCPP_DIR/$VENV_NAME"

MODEL_FILES=(
    "https://raw.githubusercontent.com/robit-man/EGG/main/voice/glados_piper_medium.onnx.json"
    "https://raw.githubusercontent.com/robit-man/EGG/main/voice/glados_piper_medium.onnx"
)

DOCKER_IMAGE_NAME="piper-tts-rpi5"
DOCKERFILE_TARGET="$TARGET_DIR/dockerfile"

AUTOSTART_DIR="$HOME/.config/autostart"
AUTOSTART_FILE="$AUTOSTART_DIR/mini_egg_watchdog.desktop"
LEGACY_AUTOSTART_FILE="$AUTOSTART_DIR/voice_setup.desktop"

TERMINAL_CMD=""
if command -v lxterminal > /dev/null 2>&1; then
    TERMINAL_CMD="lxterminal"
elif command -v gnome-terminal > /dev/null 2>&1; then
    TERMINAL_CMD="gnome-terminal"
elif command -v x-terminal-emulator > /dev/null 2>&1; then
    TERMINAL_CMD="x-terminal-emulator"
fi

# --------------------------- Functions ---------------------------

download_file() {
    local url="$1"
    local dest="$2"
    echo "[SETUP] Downloading $(basename "$dest")..."
    curl -sSL "$url" -o "$dest"
}

ensure_repo() {
    mkdir -p "$TARGET_DIR"
    if [ -d "$REPO_DIR/.git" ]; then
        echo "[SETUP] Updating repository mirror at $REPO_DIR..."
        git -C "$REPO_DIR" fetch --prune origin "$REPO_BRANCH" || true
        git -C "$REPO_DIR" checkout "$REPO_BRANCH" || true
        git -C "$REPO_DIR" pull --ff-only origin "$REPO_BRANCH" || true
    else
        echo "[SETUP] Cloning repository mirror into $REPO_DIR..."
        rm -rf "$REPO_DIR"
        git clone --depth=1 --branch="$REPO_BRANCH" "$REPO_URL" "$REPO_DIR"
    fi
}

sync_runtime_files() {
    if [ ! -f "$REPO_DIR/pi5/sync_runtime.sh" ]; then
        echo "[SETUP] Missing sync script in repo mirror."
        exit 1
    fi
    cp "$REPO_DIR/pi5/sync_runtime.sh" "$SYNC_SCRIPT"
    chmod +x "$SYNC_SCRIPT"
    if [ -f "$REPO_DIR/pi5/setup.sh" ]; then
        cp "$REPO_DIR/pi5/setup.sh" "$TARGET_DIR/setup.sh"
        chmod +x "$TARGET_DIR/setup.sh"
    fi
    if [ -f "$REPO_DIR/pi5/teardown.sh" ]; then
        cp "$REPO_DIR/pi5/teardown.sh" "$TARGET_DIR/teardown.sh"
        chmod +x "$TARGET_DIR/teardown.sh"
    fi
    "$SYNC_SCRIPT" "$REPO_DIR" "$TARGET_DIR"
}

ensure_models() {
    for url in "${MODEL_FILES[@]}"; do
        local filename
        filename="$(basename "$url")"
        local dest="$TARGET_DIR/$filename"
        if [ ! -f "$dest" ]; then
            download_file "$url" "$dest"
        else
            echo "[SETUP] $filename already exists."
        fi
    done
}

ensure_whisper_venv() {
    if [ ! -d "$WHISPERCPP_DIR" ]; then
        echo "[SETUP] whispercpp not found at $WHISPERCPP_DIR."
        exit 1
    fi
    if [ ! -d "$VENV_DIR" ]; then
        echo "[SETUP] Creating whisper virtual environment..."
        python3 -m venv "$VENV_DIR" --system-site-packages
    else
        echo "[SETUP] whisper virtual environment already exists."
    fi

    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip || true
    pip install build || true
    if ls "$WHISPERCPP_DIR"/dist/*.whl > /dev/null 2>&1; then
        local wheel_path=""
        local arch
        arch="$(uname -m 2>/dev/null || true)"
        if [ "$arch" = "aarch64" ] || [ "$arch" = "arm64" ]; then
            wheel_path="$(ls -1 "$WHISPERCPP_DIR"/dist/*aarch64*.whl 2>/dev/null | head -n 1 || true)"
        fi
        if [ -z "$wheel_path" ]; then
            wheel_path="$(ls -1 "$WHISPERCPP_DIR"/dist/*.whl | head -n 1)"
        fi
        echo "[SETUP] Installing whispercpp wheel: $wheel_path"
        pip install --force-reinstall --no-deps "$wheel_path" || true
    else
        echo "[SETUP] Installing whispercpp from local source..."
        pip install -e "$WHISPERCPP_DIR" || true
    fi
    deactivate
}

build_docker_image() {
    if ! command -v docker > /dev/null 2>&1; then
        echo "[SETUP] Docker not found; skipping piper image build."
        return
    fi
    if [ ! -f "$DOCKERFILE_TARGET" ]; then
        echo "[SETUP] Dockerfile not found at $DOCKERFILE_TARGET; skipping build."
        return
    fi
    if ! docker image inspect "$DOCKER_IMAGE_NAME" > /dev/null 2>&1; then
        echo "[SETUP] Building Docker image '$DOCKER_IMAGE_NAME'..."
        docker build -t "$DOCKER_IMAGE_NAME" -f "$DOCKERFILE_TARGET" "$TARGET_DIR"
    else
        echo "[SETUP] Docker image '$DOCKER_IMAGE_NAME' already exists."
    fi
}

upgrade_system_and_install_prereqs() {
    if ! command -v apt-get > /dev/null 2>&1; then
        echo "[SETUP] apt-get not found; skipping system package setup."
        return
    fi

    local apt_prefix=""
    if [ "$(id -u)" -ne 0 ]; then
        if command -v sudo > /dev/null 2>&1; then
            apt_prefix="sudo"
        else
            echo "[SETUP] sudo not found; skipping apt package steps."
            return
        fi
    fi

    echo "[SETUP] Refreshing apt package index..."
    $apt_prefix apt-get update || echo "[WARN] apt-get update failed; continuing."

    if [ "${EGG_SKIP_APT_UPGRADE:-0}" != "1" ]; then
        echo "[SETUP] Upgrading system packages (set EGG_SKIP_APT_UPGRADE=1 to skip)..."
        $apt_prefix apt-get -y dist-upgrade || echo "[WARN] apt dist-upgrade failed; continuing."
    else
        echo "[SETUP] Skipping apt dist-upgrade (EGG_SKIP_APT_UPGRADE=1)."
    fi

    echo "[SETUP] Installing required packages..."
    $apt_prefix apt-get install -y \
        git \
        curl \
        python3 \
        python3-venv \
        python3-pip \
        python3-requests \
        python3-num2words \
        python3-alsaaudio \
        python3-psutil \
        python3-dev \
        nodejs \
        npm \
        libportaudio2 \
        v4l-utils || echo "[WARN] apt package install failed; continuing."

    $apt_prefix apt-get install -y python3-picamera2 || \
        echo "[WARN] python3-picamera2 not available; camera service will use OpenCV fallback."
}

create_watchdog_launcher() {
    cat <<EOF > "$TARGET_DIR/start_watchdog.sh"
#!/bin/bash
set -e
cd "$TARGET_DIR"
if [ -x "$TARGET_DIR/sync_runtime.sh" ] && [ -d "$REPO_DIR" ]; then
  "$TARGET_DIR/sync_runtime.sh" "$REPO_DIR" "$TARGET_DIR" >/dev/null 2>&1 || true
fi
export WATCHDOG_REPO_DIR="$REPO_DIR"
export WATCHDOG_AUTO_UPDATE_URL="$REPO_URL"
export WATCHDOG_AUTO_UPDATE_BRANCH="$REPO_BRANCH"
export WATCHDOG_TERMINAL="$TERMINAL_CMD"
python3 "$TARGET_DIR/watchdog.py"
EOF
    chmod +x "$TARGET_DIR/start_watchdog.sh"
}

ensure_bind_hosts() {
    python3 - "$TARGET_DIR" <<'PY'
import json
import os
import sys

target_dir = sys.argv[1]
configs = [
    ("router_config.json", ("router", "network", "listen_host")),
    ("camera_router_config.json", ("camera_router", "network", "listen_host")),
    ("pipeline_api_config.json", ("pipeline_api", "network", "listen_host")),
    ("audio_router_config.json", ("audio_router", "network", "listen_host")),
]
security_defaults = [
    ("camera_router_config.json", ("camera_router", "security"), {
        "password": "egg",
        "require_auth": True,
        "session_timeout": 300,
    }),
    ("pipeline_api_config.json", ("pipeline_api", "security"), {
        "password": "egg",
        "require_auth": True,
        "session_timeout": 300,
    }),
    ("audio_router_config.json", ("audio_router", "security"), {
        "password": "egg",
        "require_auth": True,
        "session_timeout": 300,
    }),
]
service_url_defaults = [
    (
        "router_config.json",
        ("router", "services", "audio_router_info_url"),
        "http://127.0.0.1:8090/router_info",
        {"http://127.0.0.1:6590/router_info", "http://127.0.0.1:6590/health"},
    ),
]

def update_bind_host(payload, key_path):
    changed = False
    current = payload
    for key in key_path[:-1]:
        if not isinstance(current, dict):
            return changed
        current = current.get(key)
        if current is None:
            return changed
    if not isinstance(current, dict):
        return changed
    leaf = key_path[-1]
    host = str(current.get(leaf, "")).strip().lower()
    if host in ("", "127.0.0.1", "localhost", "::1"):
        current[leaf] = "0.0.0.0"
        changed = True
    return changed

def update_security_defaults(payload, key_path, defaults):
    changed = False
    current = payload
    for key in key_path:
        if not isinstance(current, dict):
            return changed
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
            changed = True
        current = current[key]
    if not isinstance(current, dict):
        return changed
    for key, default_value in defaults.items():
        value = current.get(key)
        if value is None or (isinstance(value, str) and not value.strip()):
            current[key] = default_value
            changed = True
    return changed

def load_payload(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fp:
            payload = json.load(fp)
        if not isinstance(payload, dict):
            return {}
        return payload
    except Exception:
        return {}

def update_service_url(payload, key_path, default_url, legacy_values):
    changed = False
    current = payload
    for key in key_path[:-1]:
        if not isinstance(current, dict):
            return changed
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
            changed = True
        current = current[key]
    leaf = key_path[-1]
    current_value = str(current.get(leaf, "")).strip()
    if not current_value or current_value in legacy_values:
        current[leaf] = default_url
        changed = True
    return changed

for filename, key_path in configs:
    path = os.path.join(target_dir, filename)
    payload = load_payload(path)
    if payload is None:
        continue
    changed = update_bind_host(payload, key_path)
    if changed:
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2)
        print(f"[SETUP] Updated bind host to 0.0.0.0 in {path}")

for filename, key_path, defaults in security_defaults:
    path = os.path.join(target_dir, filename)
    payload = load_payload(path)
    if payload is None:
        continue
    changed = update_security_defaults(payload, key_path, defaults)
    if changed:
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2)
        print(f"[SETUP] Applied security defaults in {path}")

for filename, key_path, default_url, legacy_values in service_url_defaults:
    path = os.path.join(target_dir, filename)
    payload = load_payload(path)
    if payload is None:
        continue
    changed = update_service_url(payload, key_path, default_url, legacy_values)
    if changed:
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2)
        print(f"[SETUP] Updated service URL defaults in {path}")
PY
}

add_to_autostart() {
    mkdir -p "$AUTOSTART_DIR"
    rm -f "$LEGACY_AUTOSTART_FILE"

    local exec_line
    if [ -n "$TERMINAL_CMD" ]; then
        if [ "$TERMINAL_CMD" = "gnome-terminal" ]; then
            exec_line="gnome-terminal -- bash -lc '$TARGET_DIR/start_watchdog.sh; exec bash'"
        elif [ "$TERMINAL_CMD" = "lxterminal" ]; then
            exec_line="lxterminal --working-directory=$TARGET_DIR --command '/bin/bash $TARGET_DIR/start_watchdog.sh'"
        else
            exec_line="$TERMINAL_CMD -e /bin/bash $TARGET_DIR/start_watchdog.sh"
        fi
    else
        exec_line="/bin/bash $TARGET_DIR/start_watchdog.sh"
    fi

    cat <<EOF > "$AUTOSTART_FILE"
[Desktop Entry]
Type=Application
Exec=$exec_line
Hidden=false
NoDisplay=false
X-GNOME-Autostart-enabled=true
Name=Mini EGG Watchdog
Comment=Run Mini EGG watchdog on startup
EOF
    echo "[SETUP] Autostart configured at $AUTOSTART_FILE"
}

launch_watchdog_now() {
    if pgrep -f "$TARGET_DIR/watchdog.py" > /dev/null 2>&1; then
        echo "[SETUP] Watchdog is already running."
        return
    fi

    if [ -n "$TERMINAL_CMD" ]; then
        echo "[SETUP] Launching watchdog in a new terminal..."
        if [ "$TERMINAL_CMD" = "gnome-terminal" ]; then
            gnome-terminal -- bash -lc "$TARGET_DIR/start_watchdog.sh; exec bash" &
        elif [ "$TERMINAL_CMD" = "lxterminal" ]; then
            lxterminal --working-directory="$TARGET_DIR" --command "/bin/bash $TARGET_DIR/start_watchdog.sh" &
        else
            "$TERMINAL_CMD" -e "/bin/bash $TARGET_DIR/start_watchdog.sh" &
        fi
    else
        echo "[SETUP] No terminal emulator detected. Run manually:"
        echo "        /bin/bash $TARGET_DIR/start_watchdog.sh"
    fi
}

# --------------------------- Setup Steps ---------------------------

echo "[SETUP] Starting Mini EGG Pi5 setup..."

upgrade_system_and_install_prereqs
ensure_repo
sync_runtime_files
ensure_bind_hosts
ensure_models
ensure_whisper_venv
build_docker_image
create_watchdog_launcher
add_to_autostart
launch_watchdog_now

echo "[SETUP] Setup completed."
