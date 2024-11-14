#!/usr/bin/env python3

import sys
import os
import subprocess
import json
import datetime
import time

# Constants for USB device identification
VENDOR_ID = '1546'
PRODUCT_ID = '01a7'

# Serial communication settings
BAUD_RATE = 57600  # Default baud rate for u-blox GT-U7

# JSON file to store GPS data
JSON_FILE = 'gps_data.json'

# Virtual environment directory
VENV_DIR = 'gps_venv'

# Required Python packages
REQUIRED_PACKAGES = ['pyserial', 'pyudev', 'pynmea2']


def run_command(command, check=True, capture_output=False, text=True):
    """
    Utility function to run shell commands.
    """
    try:
        result = subprocess.run(command, check=check, capture_output=capture_output, text=text)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Command '{' '.join(command)}' failed with error:\n{e.stderr}")
        sys.exit(1)


def setup_virtualenv():
    """
    Sets up the virtual environment and installs required packages.
    """
    import venv

    if not os.path.isdir(VENV_DIR):
        print(f"Creating virtual environment in '{VENV_DIR}'...")
        venv.EnvBuilder(with_pip=True).create(VENV_DIR)
    else:
        print("Virtual environment already exists.")

    # Path to the pip executable within the virtual environment
    pip_executable = os.path.join(VENV_DIR, 'bin', 'pip')

    print("Installing dependencies...")
    run_command([pip_executable, 'install', '--upgrade', 'pip'])
    run_command([pip_executable, 'install'] + REQUIRED_PACKAGES)
    print("Dependencies installed.")


def relaunch_script():
    """
    Relaunches the current script within the virtual environment.
    """
    python_executable = os.path.join(VENV_DIR, 'bin', 'python')
    script_path = os.path.abspath(__file__)

    if not os.path.exists(python_executable):
        print(f"Error: Python executable not found in '{python_executable}'.")
        sys.exit(1)

    print("Relaunching script inside the virtual environment...")
    run_command([python_executable, script_path, '--run'])
    sys.exit()


def find_gps_device(vendor_id, product_id):
    """
    Finds the GPS device node based on USB vendor and product ID.

    Args:
        vendor_id (str): USB vendor ID.
        product_id (str): USB product ID.

    Returns:
        str: Device node (e.g., '/dev/ttyUSB0') if found, else None.
    """
    try:
        import pyudev
    except ImportError:
        print("pyudev is not installed. Please ensure dependencies are installed.")
        sys.exit(1)

    context = pyudev.Context()
    devices = context.list_devices(subsystem='tty')

    for device in devices:
        if device.get('ID_VENDOR_ID') == vendor_id and device.get('ID_MODEL_ID') == product_id:
            return device.device_node

    return None


def parse_nmea_sentence(sentence, latest_date):
    """
    Parses an NMEA sentence and extracts relevant GPS and timing data.

    Args:
        sentence (str): NMEA sentence string.
        latest_date (datetime.date): The latest known date from previous sentences.

    Returns:
        tuple: (dict of extracted data, updated latest_date)
    """
    try:
        import pynmea2
    except ImportError:
        print("pynmea2 is not installed. Please ensure dependencies are installed.")
        sys.exit(1)

    try:
        msg = pynmea2.parse(sentence)
        data = {}
        updated_date = latest_date  # Initialize updated_date with current latest_date

        if isinstance(msg, pynmea2.types.talker.RMC):
            # RMC Sentence: Recommended Minimum Specific GPS/Transit Data
            if msg.datestamp:
                updated_date = msg.datestamp
            timestamp = datetime.datetime.combine(
                updated_date, msg.timestamp
            ).isoformat()
            data['timestamp'] = timestamp
            data['latitude'] = msg.latitude
            data['longitude'] = msg.longitude
            data['speed_over_ground'] = float(msg.spd_over_grnd) if msg.spd_over_grnd else None
            data['course_over_ground'] = float(msg.true_course) if msg.true_course else None

        elif isinstance(msg, pynmea2.types.talker.GGA):
            # GGA Sentence: Global Positioning System Fix Data
            if latest_date:
                timestamp = datetime.datetime.combine(
                    latest_date, msg.timestamp
                ).isoformat()
            else:
                # If no date information is available yet, use UTC date
                timestamp = datetime.datetime.utcnow().isoformat()
            data['timestamp'] = timestamp
            data['latitude'] = msg.latitude
            data['longitude'] = msg.longitude
            data['altitude'] = float(msg.altitude) if msg.altitude else None
            data['num_sats'] = int(msg.num_sats) if msg.num_sats else None
            data['hdop'] = float(msg.horizontal_dil) if msg.horizontal_dil else None

        elif isinstance(msg, pynmea2.types.talker.ZDA):
            # ZDA Sentence: Time & Date
            if msg.datestamp:
                updated_date = msg.datestamp
            timestamp = datetime.datetime(
                updated_date.year,
                updated_date.month,
                updated_date.day,
                msg.timestamp.hour,
                msg.timestamp.minute,
                msg.timestamp.second,
                msg.timestamp.microsecond
            ).isoformat()
            data['timestamp'] = timestamp

        # Add more sentence types as needed

        return data, updated_date if 'updated_date' in locals() else latest_date

    except pynmea2.ParseError as e:
        print(f"Failed to parse sentence: {e}")
        return None, latest_date


def save_to_json(data, json_file):
    """
    Saves the extracted GPS data to a JSON file.

    Args:
        data (dict): GPS data to save.
        json_file (str): Path to the JSON file.
    """
    try:
        with open(json_file, 'a') as f:
            json.dump(data, f)
            f.write('\n')  # Newline for each record
        print(f"Data saved to {json_file}: {data}")
    except IOError as e:
        print(f"Failed to write to {json_file}: {e}")


def main_functionality():
    """
    Main functionality to find the GPS device, read data, parse it, and save to JSON.
    """
    device = find_gps_device(VENDOR_ID, PRODUCT_ID)
    if not device:
        print(f"GPS device with VID:PID {VENDOR_ID}:{PRODUCT_ID} not found.")
        sys.exit(1)

    print(f"GPS device found at {device}")

    try:
        import serial
    except ImportError:
        print("pyserial is not installed. Please ensure dependencies are installed.")
        sys.exit(1)

    try:
        ser = serial.Serial(device, BAUD_RATE, timeout=1)
        print(f"Opened serial connection at {device} with baud rate {BAUD_RATE}.")
    except serial.SerialException as e:
        print(f"Failed to open serial connection: {e}")
        sys.exit(1)

    latest_date = datetime.date.today()  # Initialize to today's date

    try:
        while True:
            try:
                line = ser.readline().decode('ascii', errors='replace').strip()
                if not line:
                    continue  # Skip empty lines

                # Print all NMEA sentences to console
                print(f"NMEA Sentence: {line}")

                # Check if the line is a valid NMEA sentence
                if line.startswith('$'):
                    data, latest_date = parse_nmea_sentence(line, latest_date)
                    if data:
                        try:
                            save_to_json(data, JSON_FILE)
                        except Exception as e:
                            print(f"Error saving data: {e}")

            except Exception as e:
                print(f"An unexpected error occurred while reading data: {e}")
                time.sleep(1)  # Wait before retrying

    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting...")
    finally:
        ser.close()
        print("Serial connection closed.")


def main():
    """
    Main function to orchestrate the setup and run the GPS data extraction.
    """
    if '--run' not in sys.argv:
        # Setup phase: create venv, install dependencies, and relaunch
        setup_virtualenv()
        relaunch_script()
    else:
        # Run phase: execute main functionality within the venv
        main_functionality()


if __name__ == "__main__":
    main()
