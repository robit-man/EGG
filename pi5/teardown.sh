#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# --------------------------- Configuration ---------------------------

TARGET_DIR="/home/$(whoami)/voice"
WHISPERCPP_DIR="whispercpp"
DOCKER_IMAGE_NAME="piper-tts-rpi5"

AUTOSTART_DIR="$HOME/.config/autostart"
AUTOSTART_FILE="$AUTOSTART_DIR/voice_setup.desktop"

PYTHON_PROCESS_PATTERNS=(
    "$TARGET_DIR/$WHISPERCPP_DIR/examples/stream/stream.py"
    "$TARGET_DIR/output.py"
    "$TARGET_DIR/model_to_tts.py"
    "$TARGET_DIR/voice_server.py"
)

FILES_TO_REMOVE=(
    "$TARGET_DIR/setup.sh"
    "$TARGET_DIR/output.py"
    "$TARGET_DIR/model_to_tts.py"
    "$TARGET_DIR/voice_server.py"
    "$TARGET_DIR/dockerfile"
    "$TARGET_DIR/glados_piper_medium.onnx"
    "$TARGET_DIR/glados_piper_medium.onnx.json"
)

# --------------------------- Functions ---------------------------

stop_python_processes() {
    for pattern in "${PYTHON_PROCESS_PATTERNS[@]}"; do
        if pgrep -f "$pattern" > /dev/null 2>&1; then
            echo "Stopping process pattern: $pattern"
            pkill -f "$pattern" || true
        else
            echo "No running process for: $pattern"
        fi
    done

    sleep 1

    for pattern in "${PYTHON_PROCESS_PATTERNS[@]}"; do
        if pgrep -f "$pattern" > /dev/null 2>&1; then
            echo "Force-killing process pattern: $pattern"
            pkill -9 -f "$pattern" || true
        fi
    done
}

stop_and_remove_docker() {
    if ! command -v docker > /dev/null 2>&1; then
        echo "Docker is not installed. Skipping Docker cleanup."
        return
    fi

    CONTAINERS=$(sudo docker ps -aq --filter "ancestor=$DOCKER_IMAGE_NAME")
    if [ -n "$CONTAINERS" ]; then
        echo "Stopping and removing Docker containers for image '$DOCKER_IMAGE_NAME'..."
        sudo docker rm -f $CONTAINERS > /dev/null
    else
        echo "No Docker containers found for image '$DOCKER_IMAGE_NAME'."
    fi

    if sudo docker image inspect "$DOCKER_IMAGE_NAME" > /dev/null 2>&1; then
        echo "Removing Docker image '$DOCKER_IMAGE_NAME'..."
        sudo docker rmi "$DOCKER_IMAGE_NAME" > /dev/null || true
    else
        echo "Docker image '$DOCKER_IMAGE_NAME' not found."
    fi
}

remove_autostart_entry() {
    if [ -f "$AUTOSTART_FILE" ]; then
        echo "Removing autostart file: $AUTOSTART_FILE"
        rm -f "$AUTOSTART_FILE"
    else
        echo "Autostart file not found: $AUTOSTART_FILE"
    fi
}

remove_setup_files() {
    if [ -d "$TARGET_DIR/$WHISPERCPP_DIR" ]; then
        echo "Removing directory: $TARGET_DIR/$WHISPERCPP_DIR"
        rm -rf "$TARGET_DIR/$WHISPERCPP_DIR"
    else
        echo "Directory not found: $TARGET_DIR/$WHISPERCPP_DIR"
    fi

    for file in "${FILES_TO_REMOVE[@]}"; do
        if [ -e "$file" ]; then
            echo "Removing file: $file"
            rm -f "$file"
        else
            echo "File not found: $file"
        fi
    done

    if [ -d "$TARGET_DIR" ] && [ -z "$(find "$TARGET_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
        echo "Removing empty directory: $TARGET_DIR"
        rmdir "$TARGET_DIR"
    fi
}

# --------------------------- Teardown Steps ---------------------------

echo "Starting teardown..."

stop_python_processes
stop_and_remove_docker
remove_autostart_entry
remove_setup_files

echo "Teardown completed successfully."
