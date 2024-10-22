# model.functions.py

import subprocess
import re
from datetime import datetime
import geocoder
import platform
import psutil
import socket
import os
import pygame
from threading import Thread
import time
import shutil
import json

# Initialize pygame for playing sound files
pygame.mixer.init()

# Global variable for controlling sound playback loop
is_playing = False

# Helper Functions


def load_tools_schema():
    """
    Loads the tool schemas from the model.tools.json file.
    Caches the schema after the first load for efficiency.
    """
    if not hasattr(load_tools_schema, "schema_cache"):
        tools_file = "model.tools.json"  # Path to the JSON file
        try:
            with open(tools_file, "r") as f:
                load_tools_schema.schema_cache = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading tools schema: {e}")
            load_tools_schema.schema_cache = {}
    return load_tools_schema.schema_cache

def extract_tool_arguments(arguments, tool_name):
    """
    Extracts and validates arguments for a specific tool based on its schema.
    
    Args:
        arguments (dict): The JSON object containing the arguments.
        tool_name (str): The name of the tool being called.
    
    Returns:
        dict: A cleaned and validated dictionary of arguments ready for use.
    
    Raises:
        ValueError: If a required field is missing or an invalid value is provided.
    """
    # Load the schema for the specified tool
    schema = load_tools_schema().get(tool_name)
    
    if not schema:
        raise ValueError(f"Schema for tool '{tool_name}' not found.")

    # Access the "parameters" section (ignore "description" which is for the LLM)
    parameters = schema.get("parameters", {})
    properties = parameters.get("properties", {})
    required_fields = parameters.get("required", [])

    # Start with an empty dictionary to fill with validated arguments
    cleaned_arguments = {}

    # Flatten nested 'file' argument if exists (flattening logic)
    if 'file' in arguments:
        arguments = arguments['file']

    # Validate each field based on the properties defined in the schema
    for field, field_info in properties.items():
        # Get the value from arguments, or use the default from schema if available
        value = arguments.get(field, field_info.get("default"))

        # If required field is missing, raise an error
        if value is None and field in required_fields:
            raise ValueError(f"Missing required field: {field}")

        cleaned_arguments[field] = value

    return cleaned_arguments


def find_key_in_json(data, parent_key, target_key):
    """
    Recursively search for a key under a specific parent key in a nested JSON and return its value.
    
    Args:
        data (dict): The JSON object or dict to search in.
        parent_key (str): The parent key under which we are searching for the target key.
        target_key (str): The key we are searching for under the parent key.
    
    Returns:
        The value of the target key if found, otherwise None.
    """
    if not isinstance(data, dict):
        return None

    if parent_key in data and isinstance(data[parent_key], dict):
        # If we are inside the parent key, look for the target key
        if target_key in data[parent_key]:
            return data[parent_key][target_key]

    # Recursively check the nested dictionary to find the parent key
    for k, v in data.items():
        if isinstance(v, dict):
            found = find_key_in_json(v, parent_key, target_key)
            if found:
                return found
    return None

def create_file(arguments):
    """
    Create a new file or overwrite an existing file based on the arguments provided.
    """
    try:
        # Debugging: Print the arguments received
        print(f"[Debug] create_file received full arguments: {json.dumps(arguments, indent=2)}")

        # Extract values from the 'file' key in the arguments
        file_name = find_key_in_json(arguments, "file", "name")
        file_path = find_key_in_json(arguments, "file", "path")
        content = find_key_in_json(arguments, "file", "content") or ""  # Default to empty content
        permissions = find_key_in_json(arguments, "file", "permissions") or "w"  # Default to write mode

        # Validate if required fields are present
        if not file_name or not file_path:
            return "Error: 'name' and 'path' are required fields for creating a file."

        # Create the full path
        full_file_path = os.path.join(file_path, file_name)

        # Ensure the directory exists
        os.makedirs(file_path, exist_ok=True)

        # Write to the file
        with open(full_file_path, permissions) as f:
            f.write(content)
        
        return f"File created: {full_file_path}"

    except Exception as e:
        return f"Error creating file: {e}"

def modify_file(arguments):
    """
    Modify an existing file by writing new content or appending to it.
    """
    try:
        # Use the dynamic JSON key extractor to find the required fields
        file_name = find_key_in_json(arguments, "file", "name")
        file_path = find_key_in_json(arguments, "file", "path")
        content = find_key_in_json(arguments, "file", "content")
        permissions = find_key_in_json(arguments, "file", "permissions") or "w"

        # Validate if required fields are present
        if not file_name or not file_path or content is None:
            return "Error: 'name', 'path', and 'content' are required fields for modifying a file."

        # Full file path
        full_file_path = os.path.join(file_path, file_name)

        # Check if file exists
        if not os.path.exists(full_file_path):
            return f"Error: File '{full_file_path}' does not exist."

        # Modify the file (overwrite or append based on permissions)
        with open(full_file_path, permissions) as f:
            f.write(content)

        return f"File modified: {full_file_path}"

    except Exception as e:
        return f"Error modifying file: {e}"


def delete_file(arguments):
    """
    Delete the specified file.
    """
    try:
        # Extract and validate arguments using the schema for 'delete_file'
        file_info = extract_tool_arguments(arguments.get("file", {}), "delete_file")

        file_name = file_info["name"]
        file_path = file_info["path"]

        # Create the full path
        full_file_path = os.path.join(file_path, file_name)

        # Check if the file exists
        if not os.path.exists(full_file_path):
            return f"Error: File '{full_file_path}' does not exist."

        # Delete the file
        os.remove(full_file_path)
        return f"File deleted: {full_file_path}"

    except Exception as e:
        return f"Error deleting file: {e}"


def rename_file(arguments):
    """
    Rename a file based on the arguments provided.
    """
    try:
        # Extract and validate arguments using the schema for 'rename_file'
        file_info = extract_tool_arguments(arguments.get("file", {}), "rename_file")

        old_name = file_info["name"]
        file_path = file_info["path"]
        new_name = file_info["new_name"]

        # Create the full paths
        old_file_path = os.path.join(file_path, old_name)
        new_file_path = os.path.join(file_path, new_name)

        # Check if the old file exists
        if not os.path.exists(old_file_path):
            return f"Error: File '{old_file_path}' does not exist."

        # Rename the file
        os.rename(old_file_path, new_file_path)
        return f"File renamed from '{old_name}' to '{new_name}'."

    except Exception as e:
        return f"Error renaming file: {e}"


def get_cpu_usage(arguments):
    """
    Returns current CPU usage percentage per core and overall.
    """
    cpu_percent = psutil.cpu_percent(percpu=True)
    total_cpu = psutil.cpu_percent()
    return f"CPU Usage: {total_cpu}%\nPer Core: {cpu_percent}"


def get_memory_info(arguments):
    """
    Returns detailed memory usage, available memory, and total memory.
    """
    memory = psutil.virtual_memory()
    return f"Total Memory: {memory.total / (1024 ** 3):.2f} GB, " \
           f"Available: {memory.available / (1024 ** 3):.2f} GB, " \
           f"Used: {memory.used / (1024 ** 3):.2f} GB, " \
           f"Usage: {memory.percent}%"


def get_gpu_usage(arguments):
    """
    Returns GPU usage, memory, temperature, and power (specific to NVIDIA Jetson).
    """
    try:
        gpu_info = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
             '--format=csv,noheader,nounits'], capture_output=True, text=True)
        return gpu_info.stdout.strip()
    except Exception as e:
        return f"Error fetching GPU information: {e}"


def get_disk_usage(arguments):
    """
    Returns information about disk usage, total, used, and free space for all mounted partitions.
    """
    partitions = psutil.disk_partitions()
    disk_info = ""
    for partition in partitions:
        usage = psutil.disk_usage(partition.mountpoint)
        disk_info += f"Partition: {partition.device}, " \
                     f"Total: {usage.total / (1024 ** 3):.2f} GB, " \
                     f"Used: {usage.used / (1024 ** 3):.2f} GB, " \
                     f"Free: {usage.free / (1024 ** 3):.2f} GB, " \
                     f"Usage: {usage.percent}%\n"
    return disk_info


def get_network_info(arguments):
    """
    Returns network-related details like IP address, network interfaces, and data traffic.
    """
    interfaces = psutil.net_if_addrs()
    interface_info = {}
    for interface_name, addresses in interfaces.items():
        for address in addresses:
            if address.family == socket.AF_INET:
                interface_info[interface_name] = {"IP Address": address.address}
    return interface_info


def get_system_uptime(arguments):
    """
    Returns the system uptime since the last boot.
    """
    uptime = time.time() - psutil.boot_time()
    return f"System Uptime: {uptime / 3600:.2f} hours"


def get_temperature_sensors(arguments):
    """
    Returns temperature readings from various sensors on the system (CPU, GPU, etc.).
    """
    try:
        sensors = psutil.sensors_temperatures()
        return {name: [f"{temp.current}°C" for temp in entries] for name, entries in sensors.items()}
    except Exception as e:
        return f"Error fetching temperature sensors: {e}"


def get_usb_devices(arguments):
    """
    Returns a list of connected USB devices (using lsusb).
    """
    try:
        usb_devices = subprocess.run(['lsusb'], capture_output=True, text=True)
        return usb_devices.stdout
    except Exception as e:
        return f"Error fetching USB devices: {e}"


def get_pci_devices(arguments):
    """
    Returns a list of connected PCI devices (using lspci).
    """
    try:
        pci_devices = subprocess.run(['lspci'], capture_output=True, text=True)
        return pci_devices.stdout
    except Exception as e:
        return f"Error fetching PCI devices: {e}"


def get_bluetooth_devices(arguments):
    """
    Returns a list of nearby and paired Bluetooth devices.
    """
    try:
        bluetooth_devices = subprocess.run(['bluetoothctl', 'devices'], capture_output=True, text=True)
        return bluetooth_devices.stdout
    except Exception as e:
        return f"Error fetching Bluetooth devices: {e}"


def get_audio_devices(arguments):
    """
    Returns a list of available audio input/output devices.
    """
    try:
        audio_devices = subprocess.run(['pactl', 'list', 'short', 'sources'], capture_output=True, text=True)
        return audio_devices.stdout
    except Exception as e:
        return f"Error fetching audio devices: {e}"


def get_camera_info(arguments):
    """
    Returns information about connected cameras and their statuses.
    """
    try:
        camera_info = subprocess.run(['v4l2-ctl', '--list-devices'], capture_output=True, text=True)
        return camera_info.stdout
    except Exception as e:
        return f"Error fetching camera information: {e}"


def get_system_info(arguments):
    """
    Get detailed system information tailored for an NVIDIA Jetson AGX Orin.
    """
    try:
        os_info = platform.uname()
        cpu_info = f"Physical cores: {psutil.cpu_count(logical=False)}\nTotal cores: {psutil.cpu_count(logical=True)}\n"
        memory_info = psutil.virtual_memory()
        gpu_info = get_gpu_usage({})
        net_info = get_network_info({})
        return f"System: {os_info.system}\nNode: {os_info.node}\nCPU Info:\n{cpu_info}\nMemory Info:\n{memory_info}\n{gpu_info}\nNetwork Info:\n{net_info}"
    except Exception as e:
        return f"Error fetching system info: {e}"


def get_current_location(arguments):
    """
    Get the current location using geocoder based on IP address.
    """
    try:
        location = geocoder.ip('me')
        if location.ok:
            return f"Current Location: {location.city}, {location.country}"
        return "Error: Unable to determine current location."
    except Exception as e:
        return f"Error fetching location: {e}"


def play_sound(arguments):
    """
    Play a sound file 'still_alive_loop.wav' in a loop.
    """
    global is_playing
    try:
        if not is_playing:
            def loop_sound():
                pygame.mixer.music.load('still_alive_loop.wav')
                pygame.mixer.music.play(-1)

            sound_thread = Thread(target=loop_sound)
            sound_thread.start()
            is_playing = True
            return "Playing sound in a loop."
        else:
            return "Sound is already playing."
    except Exception as e:
        return f"Error playing sound: {e}"


def stop_sound(arguments):
    """
    Stop playing the sound file.
    """
    global is_playing
    try:
        if is_playing:
            pygame.mixer.music.stop()
            is_playing = False
            return "Sound stopped."
        else:
            return "No sound is currently playing."
    except Exception as e:
        return f"Error stopping sound: {e}"


def get_voltage(arguments):
    """
    Run the voltage.py script and return the line containing 'Voltage'.
    """
    try:
        result = subprocess.run(['python3', 'voltage.py'], capture_output=True, text=True, timeout=15)
        if result.returncode == 0:
            voltage_value = re.search(r'Voltage:\s*(\d+\.\d+)\s*V', result.stdout)
            return f"Voltage: {voltage_value.group(1)} V" if voltage_value else "Error: No 'Voltage' found in output."
        return f"Error: voltage.py failed with return code {result.returncode}. Output: {result.stderr.strip()}"
    except Exception as e:
        return f"Error reading voltage: {e}"


def get_current_time(arguments):
    """
    Return the current time.
    """
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"Current Time: {current_time}"


def get_current_weather(arguments):
    """
    Fetch the current weather for a given location.
    """
    location = arguments.get("location")
    format_type = arguments.get("format", "celsius").lower()
    weather_data = {
        "Paris, FR": {"temperature_celsius": 18, "temperature_fahrenheit": 64.4, "condition": "Sunny"},
        "San Francisco, CA": {"temperature_celsius": 16, "temperature_fahrenheit": 60.8, "condition": "Cloudy"}
    }
    try:
        if location not in weather_data:
            return f"Error: Weather data for {location} not found."
        data = weather_data[location]
        temperature = data["temperature_celsius"] if format_type == "celsius" else data["temperature_fahrenheit"]
        condition = data["condition"]
        return f"Current weather in {location}: {temperature}°{'C' if format_type == 'celsius' else 'F'}, {condition}."
    except Exception as e:
        return f"Error fetching weather data: {e}"


def get_battery_status(arguments):
    """
    Check the battery capacity based on the voltage for a 24V nominal lithium-ion pack.
    Returns whether the battery is within a usable range and provides an estimate of charge level.
    """
    # Define voltage ranges for the 24V lithium-ion battery pack
    MAX_VOLTAGE = 28.8  # Fully charged
    MIN_VOLTAGE = 21.0  # Minimum usable voltage (discharged)
    NOMINAL_VOLTAGE = 24.0  # Nominal operating voltage

    try:
        # Replace this with actual battery voltage retrieval logic (e.g., from a sensor or voltage.py)
        result = subprocess.run(['python3', 'voltage.py'], capture_output=True, text=True, timeout=15)
        if result.returncode == 0:
            output = result.stdout.strip()
            voltage_value = re.search(r'Voltage:\s*(\d+\.\d+)\s*V', output)
            if voltage_value:
                voltage = float(voltage_value.group(1))

                # Determine battery state based on voltage
                if voltage >= MAX_VOLTAGE:
                    return f"Battery fully charged: {voltage}V"
                elif voltage > NOMINAL_VOLTAGE:
                    capacity_percentage = ((voltage - NOMINAL_VOLTAGE) / (MAX_VOLTAGE - NOMINAL_VOLTAGE)) * 50 + 50
                    return f"Battery capacity: {capacity_percentage:.2f}% ({voltage}V)"
                elif MIN_VOLTAGE <= voltage <= NOMINAL_VOLTAGE:
                    capacity_percentage = ((voltage - MIN_VOLTAGE) / (NOMINAL_VOLTAGE - MIN_VOLTAGE)) * 50
                    return f"Battery capacity: {capacity_percentage:.2f}% ({voltage}V)"
                else:
                    return f"Warning: Battery voltage critically low at {voltage}V. Recharge immediately."
            else:
                return "Error: No 'Voltage' found in output."
        else:
            return f"Error: voltage.py failed with return code {result.returncode}. Output: {result.stderr.strip()}"
    except subprocess.TimeoutExpired:
        return "Error: voltage.py timed out."
    except Exception as e:
        return f"Error reading battery status: {e}"


def function_list():
    """
    Returns a dictionary of available functions for easy tool calling.
    """
    functions = {
        "create_file": create_file,
        "delete_file": delete_file,
        "modify_file": modify_file,
        "rename_file": rename_file,
        "get_voltage": get_voltage,
        "get_current_time": get_current_time,
        "get_current_weather": get_current_weather,
        "get_current_location": get_current_location,
        "get_system_info": get_system_info,
        "get_battery_status": get_battery_status,
        "play_sound": play_sound,
        "stop_sound": stop_sound,
        "get_cpu_usage": get_cpu_usage,
        "get_memory_info": get_memory_info,
        "get_gpu_usage": get_gpu_usage,
        "get_disk_usage": get_disk_usage,
        "get_network_info": get_network_info,
        "get_system_uptime": get_system_uptime,
        "get_temperature_sensors": get_temperature_sensors,
        "get_usb_devices": get_usb_devices,
        "get_pci_devices": get_pci_devices,
        "get_bluetooth_devices": get_bluetooth_devices,
        "get_audio_devices": get_audio_devices,
        "get_camera_info": get_camera_info
    }
    return functions
