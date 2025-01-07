#!/bin/bash

# Define the repository and folder details
REPO_URL="https://github.com/robit-man/EGG.git"
TARGET_DIR="/home/$(whoami)/voice"
WHISPERCPP_DIR="whispercpp"
VENV_NAME="whisper"

# Clone the repository if the target directory doesn't exist
if [ ! -d "$TARGET_DIR" ]; then
    echo "Cloning repository into $TARGET_DIR..."
    git clone --depth=1 --branch=main $REPO_URL /tmp/EGG
    mkdir -p "$TARGET_DIR"
    cp -r /tmp/EGG/pi5/whisper/$WHISPERCPP_DIR "$TARGET_DIR"
    rm -rf /tmp/EGG
else
    echo "$TARGET_DIR already exists. Ensuring whispercpp is up to date..."
    git clone --depth=1 --branch=main $REPO_URL /tmp/EGG
    cp -r /tmp/EGG/pi5/whisper/$WHISPERCPP_DIR "$TARGET_DIR"
    rm -rf /tmp/EGG
fi

# Check and navigate to the whispercpp directory
if [ -d "$TARGET_DIR/$WHISPERCPP_DIR" ]; then
    cd "$TARGET_DIR/$WHISPERCPP_DIR" || exit
else
    echo "Failed to find whispercpp directory. Exiting."
    exit 1
fi

# Create a Python virtual environment if it doesn't exist
if [ ! -d "$VENV_NAME" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_NAME" --system-site-packages
else
    echo "Virtual environment already exists. Skipping creation."
fi

# Activate the virtual environment
source "$VENV_NAME/bin/activate"

# Upgrade pip and install the 'build' package
echo "Installing the 'build' package..."
pip install --upgrade pip
pip install build

# Run the build command
echo "Running the build command..."
python3 -m build w

echo "Setup and build process completed."
