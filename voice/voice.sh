#!/bin/bash

# =====================================================================
# EGG voice auto-installer / launcher
#
# This script is designed to be idempotent: it can be re-run after a
# partial / failed install and will pick up whatever is missing instead
# of skipping all setup once ~/.tempaccess exists.
# =====================================================================

USER_NAME="$(whoami)"
VOICE_DIR="/home/${USER_NAME}/voice"
RAW_BASE="https://raw.githubusercontent.com/robit-man/EGG/main/voice"
JC_DIR="/home/${USER_NAME}/jetson-containers"

# ---------------------------------------------------------------------
# System level dependencies (apt). Always run; apt is idempotent and the
# missing portaudio headers are what break the PyAudio wheel build.
# ---------------------------------------------------------------------
install_system_deps() {
    echo "Installing / verifying system dependencies (apt)..."
    sudo apt-get update
    sudo apt-get install -y \
        git curl wget \
        build-essential \
        python3-dev python3-venv libpython3-dev \
        portaudio19-dev libportaudio2 libportaudiocpp0 \
        libasound2-dev \
        ffmpeg
}

# ---------------------------------------------------------------------
# jetson-containers: clone to a PERMANENT location. The previous version
# cloned to ./jetson-containers, ran install.sh (which symlinks the
# `jetson-containers` and `autotag` commands into /usr/local/bin pointing
# back at the clone) and THEN deleted the clone -- which left dangling
# symlinks and the "command not found" errors. Keep the clone in place.
# ---------------------------------------------------------------------
check_and_install_jetson_containers() {
    if [ ! -d "$JC_DIR" ]; then
        echo "Cloning jetson-containers to $JC_DIR..."
        git clone --depth=1 https://github.com/dusty-nv/jetson-containers "$JC_DIR"
    fi

    if ! command -v jetson-containers &> /dev/null || ! command -v autotag &> /dev/null; then
        echo "Installing jetson-containers..."
        bash "$JC_DIR/install.sh"
    else
        echo "jetson-containers is installed."
    fi

    # Make the commands available in THIS shell as well (covers the case
    # where install.sh updated ~/.bashrc but the current session has not
    # been re-sourced yet).
    export PATH="$JC_DIR:$PATH"
}

# ---------------------------------------------------------------------
# ollama
# ---------------------------------------------------------------------
check_and_install_ollama() {
    if ! command -v ollama &> /dev/null; then
        echo "ollama not found. Installing ollama..."
        curl -fsSL https://ollama.com/install.sh | sh
    else
        echo "ollama is installed."
    fi
}

# ---------------------------------------------------------------------
# jtop / jetson-stats
# ---------------------------------------------------------------------
check_and_install_jtop() {
    if ! command -v jtop &> /dev/null; then
        echo "jtop not found. Installing jetson-stats..."
        sudo pip3 install -U jetson-stats
    else
        echo "jtop is installed."
    fi
}

# ---------------------------------------------------------------------
# Password / sudo caching
# ---------------------------------------------------------------------
if [ ! -f ~/.tempaccess ]; then
    read -sp "Enter your password: " PASSWORD
    echo
    echo "$PASSWORD" > ~/.tempaccess
    chmod 600 ~/.tempaccess
else
    PASSWORD=$(cat ~/.tempaccess)
fi

# Create a script to cache sudo privileges
CACHE_SCRIPT="/tmp/cache_sudo.sh"
echo -e "#!/bin/bash\necho \"$PASSWORD\" | sudo -S true" > $CACHE_SCRIPT
chmod +x $CACHE_SCRIPT

# Run the cache script to ensure sudo privileges are cached
bash $CACHE_SCRIPT

# ---------------------------------------------------------------------
# Run dependency setup every time (idempotent). These guard against a
# previous run that created ~/.tempaccess but failed partway through.
# ---------------------------------------------------------------------
install_system_deps
check_and_install_jetson_containers
check_and_install_ollama
check_and_install_jtop

# ---------------------------------------------------------------------
# Create the voice directory and download required files if missing.
# ---------------------------------------------------------------------
mkdir -p "$VOICE_DIR"

# TTS model + config
[ -f "$VOICE_DIR/glados_piper_medium.onnx" ] || \
    curl -L "$RAW_BASE/glados_piper_medium.onnx" -o "$VOICE_DIR/glados_piper_medium.onnx"
[ -f "$VOICE_DIR/glados_piper_medium.onnx.json" ] || \
    curl -L "$RAW_BASE/glados_piper_medium.onnx.json" -o "$VOICE_DIR/glados_piper_medium.onnx.json"

# Core python scripts
[ -f "$VOICE_DIR/inference.py" ] || \
    curl -L "$RAW_BASE/inference.py" -o "$VOICE_DIR/inference.py"

# model_to_tts.py lives under voice/interaction/ in the repo. It was never
# fetched before, producing: "can't open file '.../voice/model_to_tts.py'".
[ -f "$VOICE_DIR/model_to_tts.py" ] || \
    curl -L "$RAW_BASE/interaction/model_to_tts.py" -o "$VOICE_DIR/model_to_tts.py"

# Whisper capture / server scripts (live under voice/whisper/ in the repo).
[ -f "$VOICE_DIR/audio_stream.py" ] || \
    curl -L "$RAW_BASE/whisper/audio_stream.py" -o "$VOICE_DIR/audio_stream.py"
[ -f "$VOICE_DIR/whisper_server.py" ] || \
    curl -L "$RAW_BASE/whisper/whisper_server.py" -o "$VOICE_DIR/whisper_server.py"

cd "$VOICE_DIR"

# ---------------------------------------------------------------------
# Launch everything in its own GNOME terminal.
# ---------------------------------------------------------------------
gnome-terminal -- bash -c "bash $CACHE_SCRIPT && cd $VOICE_DIR && python3 audio_stream.py; exec bash"
sleep 2
gnome-terminal -- bash -c "bash $CACHE_SCRIPT && cd $VOICE_DIR && python3 model_to_tts.py --stream --history; exec bash"
sleep 2
gnome-terminal -- bash -c "bash $CACHE_SCRIPT && export PATH=$JC_DIR:\$PATH && cd $VOICE_DIR && jetson-containers run -v $VOICE_DIR:/voice \$(autotag piper-tts) bash -c 'cd /voice && python3 inference.py'; exec bash"
sleep 2
gnome-terminal -- bash -c "bash $CACHE_SCRIPT && export PATH=$JC_DIR:\$PATH && cd $VOICE_DIR && jetson-containers run -v $VOICE_DIR:/voice \$(autotag whisper) bash -c 'cd /voice && python3 whisper_server.py'; exec bash"

# ---------------------------------------------------------------------
# Run jtop in the current terminal
# ---------------------------------------------------------------------
echo "Launching jtop in the current terminal..."
check_and_install_jtop
jtop
