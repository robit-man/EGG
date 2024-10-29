#!/bin/bash
# Create a new directory for the sparse clone
mkdir EGG_Drivers
cd EGG_Drivers

# Initialize a new Git repository
git init

# Add the remote repository
git remote add origin https://github.com/robit-man/EGG.git

# Configure sparse-checkout to include only the drivers directory
git sparse-checkout init --cone
git sparse-checkout set drivers

# Pull the contents of the sparse checkout
git pull origin main

# Change to the drivers directory
cd drivers

# If there's a main script or specific command, you can run it here
# Example: python3 some_driver_script.py
