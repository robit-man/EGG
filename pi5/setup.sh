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

# Virtual environment name
VENV_NAME="whisper"

# Directories and scripts
WHISPERCPP_DIR="whispercpp"
PIPER_DIR="piper"

# Scripts to manage
SCRIPTS=(
    "piper/output.py"
    "piper/model_to_tts.py"
    "piper/voice_server.py"
)

# Docker details
DOCKERFILE_SOURCE="piper/dockerfile"
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

# --------------------------- Setup Steps ---------------------------

echo "Starting setup..."

# Create target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Clone the repository if it hasn't been cloned yet
if [ ! -d "$CLONE_DIR/.git" ]; then
    echo "Cloning repository from $REPO_URL into $CLONE_DIR..."
    git clone --depth=1 --branch="$REPO_BRANCH" "$REPO_URL" "$CLONE_DIR"
else
    echo "Repository already cloned. Pulling latest changes..."
    git -C "$CLONE_DIR" pull
fi

# Copy whispercpp directory
if [ ! -d "$TARGET_DIR/$WHISPERCPP_DIR" ]; then
    echo "Copying $WHISPERCPP_DIR to $TARGET_DIR..."
    cp -r "$CLONE_DIR/pi5/whisper/$WHISPERCPP_DIR" "$TARGET_DIR/"
else
    echo "$WHISPERCPP_DIR already exists in $TARGET_DIR. Skipping copy."
fi

# Copy additional scripts
for script in "${SCRIPTS[@]}"; do
    SCRIPT_SOURCE="$CLONE_DIR/pi5/$script"
    SCRIPT_TARGET="$TARGET_DIR/$(basename "$script")"
    
    if [ ! -f "$SCRIPT_TARGET" ]; then
        echo "Copying $(basename "$script") to $TARGET_DIR..."
        cp "$SCRIPT_SOURCE" "$SCRIPT_TARGET"
        chmod +x "$SCRIPT_TARGET"
    else
        echo "$(basename "$script") already exists in $TARGET_DIR. Skipping copy."
    fi
done

# Copy Dockerfile
if [ ! -f "$DOCKERFILE_TARGET" ]; then
    echo "Copying dockerfile to $TARGET_DIR..."
    cp "$CLONE_DIR/pi5/$DOCKERFILE_SOURCE" "$DOCKERFILE_TARGET"
else
    echo "Dockerfile already exists in $TARGET_DIR. Skipping copy."
fi

# Remove the cloned repository from /tmp
echo "Cleaning up temporary repository clone..."
rm -rf "$CLONE_DIR"

# Navigate to the target directory
cd "$TARGET_DIR" || { echo "Failed to navigate to $TARGET_DIR. Exiting."; exit 1; }

# Set up the Python virtual environment if it doesn't exist
if [ ! -d "$TARGET_DIR/$VENV_NAME" ]; then
    echo "Creating Python virtual environment '$VENV_NAME'..."
    python3 -m venv "$VENV_NAME" --system-site-packages
    echo "Activating virtual environment..."
    source "$VENV_NAME/bin/activate"
    
    echo "Upgrading pip..."
    pip install --upgrade pip
    
    echo "Installing required packages..."
    pip install build
    
    deactivate
else
    echo "Virtual environment '$VENV_NAME' already exists. Skipping creation."
fi

# Function to activate virtual environment and run a Python script
run_python_script() {
    local script_path="$1"
    echo "Running $script_path in a new terminal..."
    open_new_terminal "source '$TARGET_DIR/$VENV_NAME/bin/activate' && python3 '$script_path'"
}

# Run stream.py
STREAM_SCRIPT_DIR="$TARGET_DIR/$WHISPERCPP_DIR/examples/stream"
STREAM_SCRIPT="$STREAM_SCRIPT_DIR/stream.py"

if [ -f "$STREAM_SCRIPT" ]; then
    echo "Running stream.py..."
    open_new_terminal "source '$TARGET_DIR/$VENV_NAME/bin/activate' && python3 '$STREAM_SCRIPT'"
else
    echo "stream.py not found at $STREAM_SCRIPT. Skipping."
fi

# Run additional Python scripts
for script in "${SCRIPTS[@]}"; do
    SCRIPT_PATH="$TARGET_DIR/$(basename "$script")"
    
    if [ -f "$SCRIPT_PATH" ]; then
        # Skip voice_server.py as it runs via Docker
        if [ "$(basename "$script")" != "voice_server.py" ]; then
            run_python_script "$SCRIPT_PATH"
        fi
    else
        echo "Script $SCRIPT_PATH does not exist. Skipping."
    fi
done

# Handle Docker setup for voice_server.py
VOICE_SERVER_SCRIPT="$TARGET_DIR/voice_server.py"

if [ -f "$VOICE_SERVER_SCRIPT" ] && [ -f "$DOCKERFILE_TARGET" ]; then
    # Check if Docker image exists
    if ! sudo docker image inspect "$DOCKER_IMAGE_NAME" > /dev/null 2>&1; then
        echo "Building Docker image '$DOCKER_IMAGE_NAME'..."
        sudo docker build -t "$DOCKER_IMAGE_NAME" -f "$DOCKERFILE_TARGET" "$TARGET_DIR"
    else
        echo "Docker image '$DOCKER_IMAGE_NAME' already exists. Skipping build."
    fi
    
    # Run Docker container
    echo "Running Docker container for voice_server.py..."
    open_new_terminal "sudo docker run --network host -v '$TARGET_DIR':'/opt/voice' -w '/opt/voice' -it '$DOCKER_IMAGE_NAME' python3 voice_server.py"
else
    echo "voice_server.py or Dockerfile not found. Skipping Docker setup."
fi

# Add setup.sh to desktop autostart
echo "Configuring autostart..."
add_to_autostart

echo "Setup completed successfully."

