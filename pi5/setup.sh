#!/bin/bash

# Define the repository and folder details
REPO_URL="https://github.com/robit-man/EGG.git"
TARGET_DIR="voice"
WHISPERCPP_DIR="whispercpp"
VENV_NAME="whisper"

# Clone the repository if the target directory doesn't exist
if [ ! -d "$TARGET_DIR" ]; then
    echo "Cloning repository into $TARGET_DIR..."
    git clone --depth=1 --branch=main $REPO_URL $TARGET_DIR
else
    echo "$TARGET_DIR already exists. Pulling updates..."
    cd $TARGET_DIR || exit
    git pull
    cd ..
fi

# Navigate to the whispercpp directory
cd "$TARGET_DIR/pi5/whisper/$WHISPERCPP_DIR" || {
    echo "Failed to find whispercpp directory. Exiting."
    exit 1
}

# Create a Python virtual environment if it doesn't exist
if [ ! -d "$VENV_NAME" ]; then
    echo "Creating virtual environment..."
    python3 -m venv $VENV_NAME --system-site-packages
else
    echo "Virtual environment already exists. Skipping creation."
fi

# Activate the virtual environment
source "$VENV_NAME/bin/activate"

# Install the 'build' package
echo "Installing the 'build' package..."
pip install --upgrade pip
pip install build

# Run the build command
echo "Running the build command..."
python3 -m build w

echo "Setup and build process completed."
