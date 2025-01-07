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

# Create a Python virtual environment if it doesn't exist
if [ ! -d "$VENV_NAME" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_NAME" --system-site-packages
else
    echo "Virtual environment already exists. Skipping creation."
fi

# Navigate back to /voice/ and activate the virtual environment explicitly
cd "$TARGET_DIR" || {
    echo "Failed to navigate to $TARGET_DIR. Exiting."
    exit 1
}
echo "Activating the virtual environment..."
source "$WHISPERCPP_DIR/$VENV_NAME/bin/activate"

# Upgrade pip and install the 'build' package
echo "Installing the 'build' package..."
pip install --upgrade pip
pip install build

cd "$WHISPERCPP_DIR" || {
    echo "Failed to navigate to $WHISPERCPP_DIR. Exiting."
    exit 1
}
# Run the build command
echo "Running the build command..."
python3 -m build -w

echo "Setup and build process completed."
