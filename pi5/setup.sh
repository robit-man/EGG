#!/bin/bash

# Define the repository and folder details
REPO_URL="https://github.com/robit-man/EGG.git"
TARGET_DIR="/home/$(whoami)/voice"
WHISPERCPP_DIR="whispercpp"
VENV_NAME="whisper"

# Ensure the target directory exists
mkdir -p "$TARGET_DIR"

# Clone the repository to a temporary location
echo "Cloning repository into a temporary location..."
git clone --depth=1 --branch=main $REPO_URL /tmp/EGG

# Copy the whispercpp folder into the target directory
echo "Copying whispercpp directory into $TARGET_DIR..."
cp -r /tmp/EGG/pi5/whisper/$WHISPERCPP_DIR "$TARGET_DIR"

# Cleanup temporary repository
rm -rf /tmp/EGG

# Navigate to the whispercpp directory
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
