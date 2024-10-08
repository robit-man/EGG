#!/bin/bash

# Run dummy_script.sh to cache credentials
bash /home/roko/Startup/dummy_script.sh

# Navigate to the RIVA directory
cd /home/roko/RIVA/

# Start Riva with sudo privileges
echo 'roko' | sudo -S bash ./riva_start.sh

# Add the following to your Startup Applications Preferences Gui
# gnome-terminal -- bash -c "bash /home/roko/Startup/riva_start.sh; exec bash"
