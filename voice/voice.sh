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

# Fully non-interactive apt. After the single password entry nothing should
# prompt: -y auto-confirms, DEBIAN_FRONTEND=noninteractive suppresses config
# dialogs, NEEDRESTART_MODE=a auto-restarts services (no "which services to
# restart" menu), and the Dpkg conf options auto-keep existing config files.
# `sudo env ...` is used so the vars survive sudo's environment scrubbing.
APT_GET="sudo env DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a NEEDRESTART_SUSPEND=1 apt-get -y -o Dpkg::Options::=--force-confdef -o Dpkg::Options::=--force-confold"

# ---------------------------------------------------------------------
# System level dependencies (apt). Always run; apt is idempotent and the
# missing portaudio headers are what break the PyAudio wheel build.
# ---------------------------------------------------------------------
install_system_deps() {
    echo "Installing / verifying system dependencies (apt)..."
    $APT_GET update
    $APT_GET install \
        git curl wget \
        build-essential \
        python3-dev python3-venv libpython3-dev \
        portaudio19-dev libportaudio2 libportaudiocpp0 \
        libasound2-dev \
        ffmpeg
}

# ---------------------------------------------------------------------
# Docker + NVIDIA container runtime. jetson-containers (and `autotag`)
# shell out to `sudo docker ...`; without docker installed you get
# "sudo: docker: command not found" and autotag fails on `docker images`.
# On a fresh box JetPack's docker may be missing, so install it and wire
# up the NVIDIA runtime as the default so GPU containers work.
# ---------------------------------------------------------------------
check_and_install_docker() {
    # 1. Docker engine itself. IMPORTANT: install docker.io on its OWN line.
    #    `apt-get install -y docker.io nvidia-container-toolkit ...` is an
    #    all-or-nothing transaction -- if any sibling package name can't be
    #    located (the nvidia packages frequently aren't in the default
    #    sources) apt aborts and docker.io is NOT installed, which is why
    #    "sudo: docker: command not found" kept coming back.
    if ! command -v docker &> /dev/null; then
        echo "docker not found. Installing docker.io..."
        $APT_GET install docker.io
    else
        echo "docker is installed."
    fi

    if ! command -v docker &> /dev/null; then
        echo "ERROR: docker still not installed after apt-get install docker.io." >&2
        echo "       The piper-tts / whisper containers cannot start without it." >&2
    fi

    # 2. NVIDIA container runtime. This is provided by nvidia-container-toolkit
    #    (which also pulls nvidia-container-toolkit-base / the runtime binary).
    #    The standalone nvidia-container-runtime / nvidia-container package names
    #    have no install candidate on JetPack 6 and only produce noise, so we
    #    install just the toolkit and only when it is not already present.
    if ! command -v nvidia-ctk &> /dev/null && ! command -v nvidia-container-runtime &> /dev/null; then
        $APT_GET install nvidia-container-toolkit || true
    fi

    # 3. Register the NVIDIA runtime as the default -- but ONLY if the
    #    runtime binary actually exists. Setting default-runtime=nvidia in
    #    daemon.json without the binary makes dockerd refuse to start.
    if command -v nvidia-container-runtime &> /dev/null; then
        if command -v nvidia-ctk &> /dev/null; then
            sudo nvidia-ctk runtime configure --runtime=docker --set-as-default || true
        else
            sudo mkdir -p /etc/docker
            if [ ! -f /etc/docker/daemon.json ] || ! grep -q '"default-runtime"' /etc/docker/daemon.json 2>/dev/null; then
                echo '{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "default-runtime": "nvidia"
}' | sudo tee /etc/docker/daemon.json > /dev/null
            fi
        fi
    else
        echo "WARNING: nvidia-container-runtime not available; GPU containers" >&2
        echo "         may fall back to CPU. Skipping default-runtime config." >&2
    fi

    # 4. Enable + (re)start the docker daemon so it picks up the runtime
    #    config. IMPORTANT: bound every systemctl call with `timeout` -- on
    #    Jetson `systemctl restart docker` can block indefinitely (containerd
    #    / docker.socket), which wedged the installer with no output right
    #    after nvidia-ctk's "recommended that docker daemon be restarted".
    #    Prefer a config reload (cheap, never blocks); only restart if needed,
    #    and always make sure docker ends up running afterwards.
    sudo timeout 30 systemctl enable docker 2>/dev/null || true
    if ! sudo timeout 30 systemctl reload docker 2>/dev/null; then
        sudo timeout 60 systemctl restart docker 2>/dev/null || true
    fi
    # Guarantee the daemon is up regardless of how the above went.
    if ! sudo docker info >/dev/null 2>&1; then
        sudo timeout 60 systemctl start docker 2>/dev/null || true
    fi

    # 5. Force the reliable `sudo docker` path in jetson-containers.
    #    run.sh decides whether to use sudo with:
    #        id -nG "$USER" | grep -qw docker  ->  SUDO=""  else  SUDO="sudo"
    #    `id -nG` reads the GROUP DATABASE, so adding the user to the docker
    #    group with `usermod -aG` makes run.sh drop sudo IMMEDIATELY -- but the
    #    group is NOT active in the running session until re-login, so plain
    #    `docker` then fails ("command not found" in non-login subshells, or
    #    socket permission denied). `sudo docker` always works (secure_path
    #    finds the binary and root can reach the socket) and sudo is primed by
    #    the cache script before each launch. So we KEEP the user OUT of the
    #    docker group and undo any membership added by earlier script versions.
    if id -nG "$USER_NAME" 2>/dev/null | grep -qw docker; then
        sudo gpasswd -d "$USER_NAME" docker 2>/dev/null || \
        sudo deluser "$USER_NAME" docker 2>/dev/null || true
        echo "Removed $USER_NAME from docker group so jetson-containers uses 'sudo docker'."
    fi
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
        # Full clone (no --depth): jetson-containers refreshes data/containers.json
        # from its `dev` branch (git fetch origin dev && git checkout origin/dev --
        # data/containers.json). A shallow single-branch clone has no origin/dev
        # ref, producing "fatal: invalid reference: origin/dev".
        git clone https://github.com/dusty-nv/jetson-containers "$JC_DIR"
    fi

    # Repair a pre-existing shallow clone (from older versions of this script)
    # so the origin/dev ref exists and the containers.json refresh stops
    # printing "fatal: invalid reference: origin/dev".
    if [ -d "$JC_DIR/.git" ]; then
        if [ -f "$JC_DIR/.git/shallow" ]; then
            git -C "$JC_DIR" fetch --unshallow 2>/dev/null || true
        fi
        git -C "$JC_DIR" remote set-branches origin '*' 2>/dev/null || true
        git -C "$JC_DIR" fetch origin 2>/dev/null || true
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

    # Make sure the ollama server is actually running and reachable on
    # :11434. model_to_tts.py calls wait_for_ollama() (10x2s) and then pulls
    # the model; if nothing is serving it would give up. Prefer the systemd
    # service the installer sets up; fall back to a backgrounded `ollama serve`.
    sudo systemctl enable --now ollama 2>/dev/null || true
    if ! curl -fs http://localhost:11434/api/tags >/dev/null 2>&1; then
        echo "Starting ollama server in the background..."
        nohup ollama serve >/tmp/ollama.log 2>&1 &
        # Give it a moment to bind the port.
        for _ in $(seq 1 10); do
            curl -fs http://localhost:11434/api/tags >/dev/null 2>&1 && break
            sleep 1
        done
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
# Passwordless sudo for this user.
#
# After this single password entry NOTHING should prompt again. The cached
# sudo timestamp (default 15 min, per-tty) is not enough: container builds
# (e.g. whisper pulls in rust + jupyterlab and takes many minutes) outlive
# it and `sudo docker buildx` then re-prompts for a password mid-build with
# no way to answer in an unattended autostart. So install a sudoers drop-in
# granting this user passwordless sudo. Intentional for the appliance/kiosk
# use-case. The first `sudo tee` below uses the timestamp primed above.
# ---------------------------------------------------------------------
SUDOERS_FILE="/etc/sudoers.d/99-egg-voice"
if [ ! -f "$SUDOERS_FILE" ]; then
    echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" | sudo tee "$SUDOERS_FILE" > /dev/null
    sudo chmod 0440 "$SUDOERS_FILE"
    # Validate; if the drop-in is somehow invalid, remove it so sudo keeps working.
    if ! sudo visudo -cf "$SUDOERS_FILE" > /dev/null 2>&1; then
        echo "WARNING: invalid sudoers drop-in, removing it." >&2
        sudo rm -f "$SUDOERS_FILE"
    fi
fi

# ---------------------------------------------------------------------
# Run dependency setup every time (idempotent). These guard against a
# previous run that created ~/.tempaccess but failed partway through.
# ---------------------------------------------------------------------
install_system_deps
check_and_install_docker
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
# `autotag --quiet` auto-confirms the "would you like to pull it? [Y/n]"
# prompt (quiet defaults the answer to yes) while still printing the resolved
# image tag to stdout for command substitution -- so nothing prompts here.
gnome-terminal -- bash -c "bash $CACHE_SCRIPT && export PATH=$JC_DIR:\$PATH && cd $VOICE_DIR && jetson-containers run -v $VOICE_DIR:/voice \$(autotag --quiet piper-tts) bash -c 'cd /voice && python3 inference.py'; exec bash"
sleep 2
gnome-terminal -- bash -c "bash $CACHE_SCRIPT && export PATH=$JC_DIR:\$PATH && cd $VOICE_DIR && jetson-containers run -v $VOICE_DIR:/voice \$(autotag --quiet whisper) bash -c 'cd /voice && python3 whisper_server.py'; exec bash"

# ---------------------------------------------------------------------
# Run jtop in the current terminal
# ---------------------------------------------------------------------
echo "Launching jtop in the current terminal..."
check_and_install_jtop
jtop
