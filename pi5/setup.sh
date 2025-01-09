#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# --------------------------- Configuration ---------------------------

# Repository details
REPO_URL="https://github.com/robit-man/EGG.git"
REPO_BRANCH="main"
CLONE_DIR="/tmp/EGG"

# Target directory where all scripts will reside
TARGET_DIR="/home/$(whoami)/voice"

# Virtual environment name and path
VENV_NAME="whisper"
VENV_DIR="$TARGET_DIR/$VENV_NAME"

# whispercpp directory
WHISPERCPP_DIR="whispercpp"

# Additional scripts and Dockerfile
ADDITIONAL_FILES=(
    "piper/output.py"
    "piper/model_to_tts.py"
    "piper/voice_server.py"
    "piper/dockerfile"
)

# Docker details
DOCKERFILE_TARGET="$TARGET_DIR/dockerfile"
DOCKER_IMAGE_NAME="piper-tts-rpi5"

# Autostart configuration
AUTOSTART_DIR="$HOME/.config/autostart"
AUTOSTART_FILE="$AUTOSTART_DIR/voice_setup.desktop"

# Terminal emulator command (adjust if using a different terminal)
TERMINAL_CMD="lxterminal"

# --------------------------- Functions ---------------------------

# Function to open a new terminal window and run a command
open_new_terminal() {
    local cmd="$1"
    $TERMINAL_CMD --working-directory="$TARGET_DIR" --command "bash -c '$cmd; exec bash'" &
}

# Function to add setup.sh to desktop autostart
add_to_autostart() {
    mkdir -p "$AUTOSTART_DIR"
    if [ ! -f "$AUTOSTART_FILE" ]; then
        cat <<EOF > "$AUTOSTART_FILE"
[Desktop Entry]
Type=Application
Exec=/bin/bash $TARGET_DIR/setup.sh
Hidden=false
NoDisplay=false
X-GNOME-Autostart-enabled=true
Name=Voice Setup
Comment=Run Voice Setup on startup
EOF
        echo "Autostart entry created at $AUTOSTART_FILE."
    else
        echo "Autostart entry already exists. Skipping."
    fi
}

# Function to download individual files using curl
download_file() {
    local url="$1"
    local dest="$2"
    echo "Downloading $(basename "$dest")..."
    curl -sSL "$url" -o "$dest"
    chmod +x "$dest"
}

# Function to build Docker image if not exists
build_docker_image() {
    if ! sudo docker image inspect "$DOCKER_IMAGE_NAME" > /dev/null 2>&1; then
        echo "Building Docker image '$DOCKER_IMAGE_NAME'..."
        sudo docker build -t "$DOCKER_IMAGE_NAME" -f "$DOCKERFILE_TARGET" "$TARGET_DIR"
    else
        echo "Docker image '$DOCKER_IMAGE_NAME' already exists. Skipping build."
    fi
}

# Function to run Docker container
run_docker_container() {
    echo "Running Docker container for voice_server.py..."
    open_new_terminal "sudo docker run --network host -v '$TARGET_DIR':'/opt/voice' -w '/opt/voice' -it '$DOCKER_IMAGE_NAME' python3 voice_server.py"
}

# --------------------------- Setup Steps ---------------------------

echo "Starting setup..."

# Create target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Flag to determine if we need to clone the repository
NEED_CLONE=false

# Check if whispercpp directory exists
if [ ! -d "$TARGET_DIR/$WHISPERCPP_DIR" ]; then
    NEED_CLONE=true
    echo "whispercpp directory does not exist. Will clone repository."
fi

# Check if any additional files are missing
for file in "${ADDITIONAL_FILES[@]}"; do
    BASENAME=$(basename "$file")
    if [ ! -f "$TARGET_DIR/$BASENAME" ]; then
        NEED_CLONE=true
        echo "$BASENAME is missing. Will clone repository."
        break
    fi
done

# Clone the repository if needed
if [ "$NEED_CLONE" = true ]; then
    echo "Cloning repository from $REPO_URL into $CLONE_DIR..."
    git clone --depth=1 --branch="$REPO_BRANCH" "$REPO_URL" "$CLONE_DIR"

    # Copy whispercpp directory
    if [ ! -d "$TARGET_DIR/$WHISPERCPP_DIR" ]; then
        echo "Copying $WHISPERCPP_DIR to $TARGET_DIR..."
        cp -r "$CLONE_DIR/pi5/whisper/$WHISPERCPP_DIR" "$TARGET_DIR/"
    fi

    # Copy additional scripts and Dockerfile
    for file in "${ADDITIONAL_FILES[@]}"; do
        SRC="$CLONE_DIR/pi5/$file"
        DEST="$TARGET_DIR/$(basename "$file")"
        if [ ! -f "$DEST" ]; then
            echo "Copying $(basename "$file") to $TARGET_DIR..."
            cp "$SRC" "$DEST"
            chmod +x "$DEST"
        else
            echo "$(basename "$file") already exists in $TARGET_DIR. Skipping copy."
        fi
    done

    # Cleanup temporary repository
    echo "Cleaning up temporary repository clone..."
    rm -rf "$CLONE_DIR"
else
    echo "All necessary files already exist. Skipping cloning."
fi

# Navigate to the target directory
cd "$TARGET_DIR" || {
    echo "Failed to navigate to $TARGET_DIR. Exiting."
    exit 1
}

# Set up the Python virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating Python virtual environment '$VENV_NAME'..."
    python3 -m venv "$VENV_DIR" --system-site-packages
    echo "Activating virtual environment..."
    source "$VENV_DIR/bin/activate"

    echo "Upgrading pip..."
    pip install --upgrade pip

    echo "Installing required packages..."
    pip install build

    deactivate
else
    echo "Virtual environment '$VENV_NAME' already exists. Skipping creation."
fi

# Function to activate virtual environment and run a Python script
run_stream_py() {
    local script_path="$1"
    echo "Running stream.py in a new terminal with virtual environment..."
    open_new_terminal "source '$VENV_DIR/bin/activate' && python3 '$script_path'"
}

# Run stream.py
STREAM_SCRIPT="$TARGET_DIR/$WHISPERCPP_DIR/examples/stream/stream.py"

if [ -f "$STREAM_SCRIPT" ]; then
    echo "Running stream.py..."
    run_stream_py "$STREAM_SCRIPT"
else
    echo "stream.py not found at $STREAM_SCRIPT. Skipping."
fi

# Run output.py and model_to_tts.py in separate terminals
for script in "output.py" "model_to_tts.py"; do
    SCRIPT_PATH="$TARGET_DIR/$script"
    if [ -f "$SCRIPT_PATH" ]; then
        echo "Running $script in a new terminal..."
        open_new_terminal "python3 '$SCRIPT_PATH'"
    else
        echo "$script not found at $SCRIPT_PATH. Skipping."
    fi
done

# Handle Docker setup for voice_server.py
VOICE_SERVER_SCRIPT="$TARGET_DIR/voice_server.py"

if [ -f "$VOICE_SERVER_SCRIPT" ] && [ -f "$DOCKERFILE_TARGET" ]; then
    # Build Docker image if not exists
    build_docker_image

    # Run Docker container
    run_docker_container
else
    echo "voice_server.py or Dockerfile not found. Skipping Docker setup."
fi

# Add setup.sh to desktop autostart
echo "Configuring autostart..."
add_to_autostart

echo "Setup completed successfully."
