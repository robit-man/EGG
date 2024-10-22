# voltage.py

import serial
import time
import os
import re

# Replace '/dev/ttyUSB0' with the correct port for your ESP32-C6 on AGX Orin
# You can find the correct port by running ls /dev/ttyUSB* or ls /dev/ttyACM*
port = '/dev/ttyACM0'  # Or /dev/ttyACM0 if that is the correct port
baud_rate = 9600

# Constants for the battery pack and ADC conversion
R1 = 82000.0  # 82 kOhm
R2 = 10000.0  # 10 kOhm
Vref = 3.3  # ADC reference voltage
ADC_RES = 4095  # 12-bit ADC resolution
NOMINAL_VOLTAGE = 24.0  # Nominal voltage for a 7s 6p 18650 battery pack
FULLY_CHARGED_VOLTAGE = 29.4  # Voltage when fully charged
EMPTY_VOLTAGE = 22.0  # Voltage when the battery is considered empty

def calculate_voltage(adc_value):
    """
    Calculate the voltage based on the ADC value.
    
    Args:
        adc_value (int): The ADC reading from the serial port.
    
    Returns:
        float: Calculated voltage.
    """
    # Calculate the voltage based on the observed relationship
    # ADC value ≈ 2400 corresponds to 24.4V
    voltage = (adc_value / 3030) * 29.0
    return voltage

def read_serial_data(port, baud_rate):
    """
    Read a single voltage value from the serial port.
    
    Args:
        port (str): The serial port to read from.
        baud_rate (int): The baud rate for the serial connection.
    
    Returns:
        str: Formatted voltage string or an error message.
    """
    ser = None
    voltage_file = os.path.join(os.path.dirname(__file__), 'voltage.txt')
    
    try:
        ser = serial.Serial(port, baud_rate, timeout=5)
        #print(f"Connected to {port} at {baud_rate} baud")

        # Read a single line from the serial port
        line = ser.readline().decode('utf-8').strip()
        if not line:
            return "Error: No data received from serial port."

        try:
            adc_value = int(line)
            voltage = calculate_voltage(adc_value)
            
            # Print only the voltage in the required format
            voltage_output = f"Voltage: {voltage:.2f} V"
            print(voltage_output)
            
            # Save voltage to file
            with open(voltage_file, 'w') as file:
                file.write(f"{voltage:.2f}\n")
            
            return voltage_output

        except ValueError:
            error_msg = f"Error: Received invalid data: {line}"
            print(error_msg)
            return error_msg

    except serial.SerialException as e:
        error_msg = f"Serial error: {e}"
        print(error_msg)
        return f"Error: Serial exception occurred: {e}"
    except Exception as e:
        error_msg = f"Unexpected error: {e}"
        print(error_msg)
        return f"Error: {e}"
    finally:
        if ser is not None:
            ser.close()
            print("Serial port closed")

if __name__ == "__main__":
    result = read_serial_data(port, baud_rate)
    # Optionally, exit with a non-zero code if there was an error
    if result.startswith("Error"):
        exit(1)
    else:
        exit(0)
