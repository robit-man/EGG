#!/bin/bash

# Define the repository and folder details
REPO_URL="https://github.com/robit-man/EGG.git"
TARGET_DIR="/home/$(whoami)/voice"
WHISPERCPP_DIR="whispercpp"
VENV_NAME="whisper"

# Ensure the target directory exists
mkdir -p "$TARGET_DIR"

# Check if whispercpp already exists
if [ ! -d "$TARGET_DIR/$WHISPERCPP_DIR" ]; then
    echo "whispercpp directory does not exist. Cloning repository into a temporary location..."
    git clone --depth=1 --branch=main $REPO_URL /tmp/EGG

    # Copy the whispercpp folder into the target directory
    echo "Copying whispercpp directory into $TARGET_DIR..."
    cp -r /tmp/EGG/pi5/whisper/$WHISPERCPP_DIR "$TARGET_DIR"

    # Cleanup temporary repository
    rm -rf /tmp/EGG
else
    echo "whispercpp directory already exists. Skipping cloning."
fi

# Navigate to the voice directory
cd "$TARGET_DIR" || {
    echo "Failed to navigate to $TARGET_DIR. Exiting."
    exit 1
}

# Check if whispercpp directory exists
if [ -d "$WHISPERCPP_DIR" ]; then
    cd "$WHISPERCPP_DIR" || {
        echo "Failed to navigate to whispercpp directory. Exiting."
        exit 1
    }
else
    echo "whispercpp directory not found. Exiting."
    exit 1
fi

# Check if the virtual environment exists
if [ ! -d "$VENV_NAME" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_NAME" --system-site-packages
    echo "Activating the virtual environment..."
    source "$VENV_NAME/bin/activate"

    # Install necessary packages
    echo "Installing required packages..."
    pip install --upgrade pip
    pip install build
else
    echo "Virtual environment already exists. Activating it..."
    source "$VENV_NAME/bin/activate"
fi

# Navigate to examples/stream and run stream.py
if [ -d "examples/stream" ]; then
    cd "examples/stream" || {
        echo "Failed to navigate to examples/stream directory. Exiting."
        exit 1
    }
    echo "Running stream.py..."
    python stream.py
else
    echo "examples/stream directory not found. Exiting."
    exit 1
fi

echo "Process completed."
