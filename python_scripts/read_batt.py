import serial
import time
import os

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
    # Calculate the voltage based on the observed relationship
    # ADC value ≈ 2400 corresponds to 24.4V
    voltage = (adc_value / 3030) * 29.0
    return voltage

def calculate_percentage(voltage):
    # Calculate battery percentage based on the voltage
    if voltage >= FULLY_CHARGED_VOLTAGE:
        return 100
    elif voltage <= EMPTY_VOLTAGE:
        return 0
    else:
        return int(((voltage - EMPTY_VOLTAGE) / (FULLY_CHARGED_VOLTAGE - EMPTY_VOLTAGE)) * 100)

def generate_battery_meter(percentage):
    # Generate a visual representation of the battery level
    meter_width = 30  # Width of the battery meter
    filled_length = int(meter_width * percentage / 100)
    meter = '█' * filled_length + '-' * (meter_width - filled_length)
    return f'[{meter}] {percentage}%'

def read_serial_data(port, baud_rate):
    ser = None  # Ensure ser is defined before the try block
    voltage_file = os.path.join(os.path.dirname(__file__), 'voltage.txt')
    
    try:
        ser = serial.Serial(port, baud_rate, timeout=2)
        print(f"Connected to {port} at {baud_rate} baud")

        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').strip()
                try:
                    adc_value = int(line)
                    voltage = calculate_voltage(adc_value)
                    percentage = calculate_percentage(voltage)
                    meter = generate_battery_meter(percentage)
                    print(f"Voltage: {voltage:.2f} V  {meter}  RAW ADC: {adc_value}")
                    
                    # Save voltage to file every 10 seconds
                    with open(voltage_file, 'w') as file:
                        file.write(f"{voltage:.2f}\n")
                    
                    time.sleep(10)

                except ValueError:
                    print(f"Received invalid data: {line}")

    except serial.SerialException as e:
        print(f"Serial error: {e}")
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        if ser is not None:  # Ensure ser is closed only if it was opened
            ser.close()
            print("Serial port closed")

if __name__ == "__main__":
    read_serial_data(port, baud_rate)
