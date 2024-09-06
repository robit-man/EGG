#!/bin/bash

# Function to check if Riva is running
check_riva() {
    RIVA_PORT=$1
    echo "Checking if Riva is up and running on port $RIVA_PORT..."
    until nc -zv localhost $RIVA_PORT; do
        printf '.'
        sleep 5
    done
    echo "Riva is up on port $RIVA_PORT."
}

# Ensure Riva services are running
check_riva 50051  # Change this to the actual port Riva is running on

# Dummy script to cache sudo credentials
./dummy_script.sh

# Function to start the container and handle failures
start_container() {
    while true; do
        jetson-containers run -v /home/roko/container_shared:/container_shared $(autotag nano_llm) /bin/bash -c "
            cd /container_shared && \
            while true; do
                python3 llm_settings_demo.py --input-port=6200 2>&1 | tee python_log.txt | while read -r line; do
                    echo \"\$line\"
                    if echo \"\$line\" | grep -q 'InternalError: Check failed: (offset + needed_size <= this->buffer.size)'; then
                        echo 'Detected memory allocation failure. Restarting Python script...'
                        pkill -f 'python3 llm_settings_demo.py --input-port=6200'
                        sleep 5
                        break
                    elif echo \"\$line\" | grep -q 'Error processing chunk'; then
                        echo 'Detected TTS chunk processing error. Restarting Python script...'
                        pkill -f 'python3 llm_settings_demo.py --input-port=6200'
                        sleep 5
                        break
                    elif echo \"\$line\" | grep -q 'Exception: Watchdog Timer Expired'; then
                        echo 'Detected Watchdog Timer Expired error. Restarting Python script...'
                        pkill -f 'python3 llm_settings_demo.py --input-port=6200'
                        sleep 5
                        break
                    fi
                done
                if [ $? -ne 0 ]; then
                    break
                fi
            done"
        if [ $? -ne 0 ]; then
            echo 'Container failed. Restarting...'
            sleep 5
        else
            break
        fi
    done
}

# Run the container with restart logic
start_container
