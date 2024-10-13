#!/bin/bash
# Create a new directory for the sparse clone
mkdir EGG_Orchestrator
cd EGG_Orchestrator

# Initialize a new Git repository
git init

# Add the remote repository
git remote add origin https://github.com/robit-man/EGG.git

# Configure sparse-checkout to include only the Orchestrator directory
git sparse-checkout init --cone
git sparse-checkout set Orchestrator

# Pull the contents of the sparse checkout
git pull origin main

cd Orchestrator
python3 run.py
