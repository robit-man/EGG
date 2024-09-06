from argparse import ArgumentParser
from contextlib import contextmanager
import serial
import os
from datetime import datetime

# Serial
GPS_TTY = '/dev/ttyUSB0'  # Change this to your actual serial port
BAUD_RATE = 4800  # Known baud rate
TIMEOUT = 5

# NMEA
GPS_TALKER_ID = 'GP'
GPS_SENTENCE_IDS = [
    'BOD', 'BWC', 'GGA', 'GLL', 'GSA', 'GSV', 'HDT', 'R00', 'RMA', 'RMB', 'RMC',
    'RTE', 'TRF', 'STN', 'VBW', 'VTG', 'WPL', 'XTE', 'ZDA'
]

# Other
UTF8 = 'utf-8'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOC_FILE = os.path.join(SCRIPT_DIR, 'loc.txt')

def create_loc_file():
    print(f"[INFO] Checking if {LOC_FILE} exists...")
    if not os.path.exists(LOC_FILE):
        with open(LOC_FILE, "w") as file:
            file.write("latitude,longitude,gps_time,local_time,time_delta\n")
        print(f"[INFO] Created {LOC_FILE}")
    else:
        print(f"[INFO] {LOC_FILE} already exists.")

def is_nmea_sentence(decoded_serial_line):
    return len(decoded_serial_line) > 0 and decoded_serial_line[0] == '$'

@contextmanager
def get_serial(port, baud, timeout=TIMEOUT):
    try:
        ser = serial.Serial(port, baud, timeout=timeout)
        print(f"[INFO] Opened serial port {port} at baud rate {baud}")
        yield ser
    except serial.SerialException as e:
        print(f"[ERROR] Could not open serial port: {e}")
        raise
    finally:
        ser.close()
        print(f"[INFO] Closed serial port {port}")

def filter_format_output(sentence, sentence_id_filter):
    if is_nmea_sentence(sentence):
        if not sentence_id_filter:
            print(sentence)
            parse_nmea_sentence(sentence)
        else:
            talker_sentence_id = f'${GPS_TALKER_ID}{sentence_id_filter}'
            if talker_sentence_id == sentence.split(',')[0]:
                print(sentence)
                parse_nmea_sentence(sentence)

def parse_nmea_sentence(sentence):
    try:
        data = sentence.split(',')
        if sentence.startswith('$GPGGA'):
            print(f"Parsing GGA: {sentence}")
            if len(data) >= 6 and data[2] and data[4]:  # Ensure there are enough fields and data is present
                latitude = data[2]
                longitude = data[4]
                gps_time = convert_to_time(data[1])
                if latitude and longitude and gps_time:
                    latitude = f"{latitude} {data[3]}"  # Append N/S direction
                    longitude = f"{longitude} {data[5]}"  # Append E/W direction
                    print(f"Parsed data - Latitude: {latitude}, Longitude: {longitude}, Time: {gps_time}")
                    print(f"Output: LAT: {latitude} LON: {longitude} TIME: {gps_time}")
                    save_location(latitude, longitude, gps_time)
                else:
                    print(f"[ERROR] Invalid data: lat={latitude}, lon={longitude}, gps_time={gps_time}")
            else:
                print(f"[ERROR] Incomplete GGA data: {sentence}")
        elif sentence.startswith('$GPRMC'):
            print(f"Parsing RMC: {sentence}")
            if len(data) >= 6 and data[3] and data[5]:  # Ensure there are enough fields and data is present
                latitude = data[3]
                longitude = data[5]
                gps_time = convert_to_time(data[1])
                if latitude and longitude and gps_time:
                    latitude = f"{latitude} {data[4]}"  # Append N/S direction
                    longitude = f"{longitude} {data[6]}"  # Append E/W direction
                    print(f"Parsed data - Latitude: {latitude}, Longitude: {longitude}, Time: {gps_time}")
                    print(f"Output: LAT: {latitude} LON: {longitude} TIME: {gps_time}")
                    save_location(latitude, longitude, gps_time)
                else:
                    print(f"[ERROR] Invalid data: lat={latitude}, lon={longitude}, gps_time={gps_time}")
            else:
                print(f"[ERROR] Incomplete RMC data: {sentence}")
    except Exception as e:
        print(f"[ERROR] Error parsing NMEA sentence: {e}")

def save_location(lat, lon, gps_time):
    try:
        if not lat or not lon or not gps_time:
            print(f"[ERROR] Invalid data: lat={lat}, lon={lon}, gps_time={gps_time}")
            return
        
        gps_time_dt = datetime.strptime(gps_time, "UTC %H:%M:%S")
        local_time_dt = datetime.now()  # Use local time now
        
        time_delta = local_time_dt - gps_time_dt

        data = f"{lat},{lon},{gps_time},{local_time_dt.isoformat()},{time_delta}"

        print(f"[DEBUG] Attempting to save data: {data}")

        with open(LOC_FILE, "a") as file:
            file.write(data + '\n')

        print(f"[INFO] Location and time successfully saved to {LOC_FILE}")
    except Exception as e:
        print(f"[ERROR] Failed to save location: {e}")

# Ensure LOC_FILE is correctly set up
create_loc_file()  # Ensure the file is created before reading GPS data

def convert_to_time(value):
    if not value:
        return None
    try:
        hours = int(value[:2])
        minutes = int(value[2:4])
        seconds = int(value[4:6])
        return f"UTC {hours:02}:{minutes:02}:{seconds:02}"
    except ValueError as e:
        print(f"[ERROR] ValueError in convert_to_time: {e}")
        return None

def read_output(serial_port, baud_rate, sentence_id_filter):
    with get_serial(serial_port, baud_rate) as ser:
        while True:
            try:
                gps_line = ser.readline().decode(UTF8).strip('\n')
                if gps_line:  # Ensure the line is not empty
                    filter_format_output(gps_line, sentence_id_filter)
            except UnicodeDecodeError as e:
                print('Could not decode serial data. Retrying...')
            except Exception as e:
                print(f'Unexpected error: {e}')
                break

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--sentence_id', dest='sentence_id_filter', choices=GPS_SENTENCE_IDS, help='The NMEA sentence id to parse. If not specified, defaults to all sentences')
    args = parser.parse_args()
    read_output(GPS_TTY, BAUD_RATE, args.sentence_id_filter)
