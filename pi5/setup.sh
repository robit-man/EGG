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
        source "$VENV_DIR/bin/activate"
        pip install --upgrade pip
        pip install build
        deactivate
    else
        echo "[SETUP] whisper virtual environment already exists."
    fi
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
        python3-dev \
        nodejs \
        npm \
        v4l-utils || echo "[WARN] apt package install failed; continuing."

    $apt_prefix apt-get install -y python3-picamera2 || \
        echo "[WARN] python3-picamera2 not available; camera service will use OpenCV fallback."
}

create_watchdog_launcher() {
    cat <<EOF > "$TARGET_DIR/start_watchdog.sh"
#!/bin/bash
set -e
cd "$TARGET_DIR"
export WATCHDOG_REPO_DIR="$REPO_DIR"
export WATCHDOG_AUTO_UPDATE_URL="$REPO_URL"
export WATCHDOG_AUTO_UPDATE_BRANCH="$REPO_BRANCH"
python3 "$TARGET_DIR/watchdog.py"
EOF
    chmod +x "$TARGET_DIR/start_watchdog.sh"
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
ensure_models
ensure_whisper_venv
build_docker_image
create_watchdog_launcher
add_to_autostart
launch_watchdog_now

echo "[SETUP] Setup completed."
