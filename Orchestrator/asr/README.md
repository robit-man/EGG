# ASR Engine API Documentation

## Table of Contents

1. [Introduction](#introduction)
2. [Installation and Setup](#installation-and-setup)
3. [Configuration](#configuration)
   - [Configuration File (`asr.cf`)](#configuration-file-asrcf)
   - [Command-Line Arguments](#command-line-arguments)
4. [Running the ASR Engine](#running-the-asr-engine)
5. [Command Interfaces](#command-interfaces)
   - [Port Configuration](#port-configuration)
   - [Orchestrator Connection](#orchestrator-connection)
6. [Supported Commands](#supported-commands)
   - [General Commands](#general-commands)
   - [Special Commands](#special-commands)
7. [Integration with Orchestrator](#integration-with-orchestrator)
   - [Registering with the Orchestrator](#registering-with-the-orchestrator)
   - [Handling Orchestrator Commands](#handling-orchestrator-commands)
8. [Inference Process](#inference-process)
   - [Using the Riva ASR Service](#using-the-riva-asr-service)
9. [Examples](#examples)
   - [Registering the ASR Engine](#registering-the-asr-engine)
   - [Sending Recognized Text to Orchestrator](#sending-recognized-text-to-orchestrator)
10. [Troubleshooting](#troubleshooting)
11. [FAQ](#faq)
12. [Repository Information](#repository-information)

---

## Introduction

The **ASR Engine** is a Python-based server application designed to perform Automatic Speech Recognition (ASR) by interfacing with NVIDIA's Riva ASR service. It functions as a peripheral within a larger orchestrator-managed ecosystem, enabling real-time speech-to-text conversion and seamless communication with other peripherals. The ASR Engine listens for audio input, processes it through the Riva ASR API, and transmits the recognized text back to the orchestrator for further routing or action.

This documentation provides a comprehensive guide on installing, configuring, and effectively using the ASR Engine, including its command interfaces, integration with the orchestrator, and the inference process.

---

## Installation and Setup

### Prerequisites

- **Python 3.6 or Higher**: Ensure that Python is installed on your system. Verify by running:
  ```bash
  python3 --version
  ```
- **Required Python Libraries**:
  - `grpc`: For gRPC communication.
  - `numpy`: For numerical operations.
  - `sounddevice`: For audio input.
  - `riva.client`: NVIDIA Riva client for ASR.
  - `requests`: For HTTP requests.
  - Other standard libraries: `threading`, `time`, `socket`, `json`, `uuid`, `os`, `queue`, `argparse`, `re`, `traceback`.
  
  **Installation**:
  ```bash
  pip install grpcio numpy sounddevice requests
  # riva.client may require specific installation steps as per NVIDIA Riva documentation
  ```

- **NVIDIA Riva ASR Service**: The ASR Engine relies on NVIDIA's Riva ASR service. Ensure that Riva is installed and running. Refer to the [NVIDIA Riva Documentation](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/index.html) for installation and setup instructions.

### Installation Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/robit-man/EGG.git
   cd EGG/asr
   ```

2. **Create a Virtual Environment (Optional but Recommended)**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *(Ensure that `requirements.txt` includes all necessary dependencies.)*

4. **Ensure Riva ASR Service is Running**:
   - Start the Riva ASR server as per the [Riva Documentation](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/index.html).
   - Verify it's running by checking the gRPC endpoint (default: `localhost:50051`).

---

## Configuration

The ASR Engine uses a configuration file (`asr.cf`) to manage its settings. This file is automatically created in the same directory as the script if it does not exist. Additionally, command-line arguments can be used to override configuration parameters.

### Configuration File (`asr.cf`)

#### Default Configuration

```ini
port=6200
port_range=6200-6300
orchestrator_host=localhost
orchestrator_port=6000
language_code=en-US
use_ssl=False
ssl_cert=
server=localhost:50051
input_device=
sample_rate=16000
chunk_size=1600
script_uuid=
```

#### Configuration Parameters

| Parameter           | Description                                                                 | Default Value           |
|---------------------|-----------------------------------------------------------------------------|-------------------------|
| `port`              | Starting port number to listen on for incoming connections.                 | `6200`                  |
| `port_range`        | Range of ports to try if the initial port is unavailable (e.g., `6200-6300`).| `6200-6300`            |
| `orchestrator_host` | Host address of the orchestrator.                                           | `localhost`             |
| `orchestrator_port` | Port number of the orchestrator's command interface.                       | `6000`                  |
| `language_code`     | Language code for ASR recognition (e.g., `en-US`).                          | `en-US`                 |
| `use_ssl`           | Whether to use SSL for gRPC connections (`True` or `False`).                | `False`                 |
| `ssl_cert`          | Path to the SSL certificate file (required if `use_ssl` is `True`).         | *(empty)*               |
| `server`            | Address of the Riva ASR server in the format `host:port`.                    | `localhost:50051`       |
| `input_device`      | Name of the input audio device to use. Leave empty for default.             | *(empty)*               |
| `sample_rate`       | Sample rate for audio input in Hz.                                          | `16000`                 |
| `chunk_size`        | Number of audio samples per chunk (e.g., `1600` for 100ms chunks at 16kHz).  | `1600`                  |
| `script_uuid`       | Unique identifier for the ASR Engine script. Automatically generated if empty. | *(auto-generated)*   |

#### Modifying Configuration

You can manually edit the `asr.cf` file to change configurations. Alternatively, use command-line arguments to override specific parameters during runtime. Configuration changes are saved automatically to the `asr.cf` file.

### Command-Line Arguments

The ASR Engine supports various command-line arguments to override configuration parameters. These arguments take precedence over the configurations specified in the `asr.cf` file.

#### Supported Arguments

| Argument                     | Description                                                                | Example                                        |
|------------------------------|----------------------------------------------------------------------------|------------------------------------------------|
| `--port`                     | Specify the starting port number to listen on.                            | `--port 6200`                                  |
| `--port-range`               | Define the range of ports to try if the initial port is unavailable.      | `--port-range 6200-6300`                       |
| `--orchestrator-host`        | Define the orchestrator's host address.                                   | `--orchestrator-host 192.168.1.10`             |
| `--orchestrator-port`        | Define the orchestrator's command port.                                   | `--orchestrator-port 7000`                     |
| `--language-code`            | Set the language code for ASR recognition.                                | `--language-code en-GB`                         |
| `--use-ssl`                  | Enable SSL for gRPC connections.                                         | `--use-ssl True`                                |
| `--ssl-cert`                 | Path to the SSL certificate file (required if `use_ssl` is `True`).       | `--ssl-cert /path/to/cert.pem`                  |
| `--server`                   | Specify the Riva ASR server address.                                      | `--server localhost:50051`                      |
| `--input-device`             | Define the input audio device to use.                                     | `--input-device "Microphone (Realtek)"`         |
| `--sample-rate`              | Set the sample rate for audio input in Hz.                               | `--sample-rate 44100`                            |
| `--chunk-size`               | Define the number of samples per audio chunk.                            | `--chunk-size 1600`                               |

#### Usage Example

```bash
python3 asr.py --port 6200 --port-range 6200-6300 --orchestrator-host localhost --orchestrator-port 6000 --language-code en-US --use-ssl False --server localhost:50051 --input-device "Microphone (Realtek)" --sample-rate 16000 --chunk-size 1600
```

---

## Running the ASR Engine

To start the ASR Engine, execute the script with optional command-line arguments as needed.

### Basic Execution

```bash
python3 asr.py
```

### With Command-Line Arguments

```bash
python3 asr.py --port 6200 --port-range 6200-6300 --orchestrator-host localhost --orchestrator-port 6000
```

### Execution Steps

1. **Load Configuration**: Reads settings from `asr.cf`. If the file doesn't exist, it generates a new UUID and creates the configuration file with default settings.

2. **Parse Command-Line Arguments**: Overrides configuration parameters based on provided arguments.

3. **Initialize ASR Service**: Connects to the Riva ASR service using the specified server address and SSL settings.

4. **Start Server**: Binds to an available port within the specified range and listens for incoming connections.

5. **Register with Orchestrator**: Sends a `/register` command to the orchestrator to announce its presence and receive an acknowledgment with the data port.

6. **Handle Incoming Connections**: Listens for incoming data or commands from connected clients.

7. **Perform ASR Inference**: Processes received audio data using the Riva ASR service and sends recognized text back to the orchestrator.

**Note**: Ensure that the orchestrator is running and accessible at the specified host and port before starting the ASR Engine.

---

## Command Interfaces

The ASR Engine interacts with the orchestrator and peripherals through specified ports and supports various command interfaces for management and data exchange.

### Port Configuration

- **Purpose**: Defines the range of ports the ASR Engine will attempt to bind to for incoming connections.
- **Configuration**: Specified in the `port` and `port_range` parameters within `asr.cf` or via the `--port` and `--port-range` command-line arguments.
- **Format**: A hyphen-separated range (e.g., `6200-6300`) or a comma-separated list (e.g., `6200,6201,6202`).
- **Example**:
  ```ini
  port_range=6200-6300
  ```

### Orchestrator Connection

- **Orchestrator Host**: Defined by `orchestrator_host` in `asr.cf` or via `--orchestrator-host`.
- **Orchestrator Port**: Defined by `orchestrator_port` in `asr.cf` or via `--orchestrator-port`.
- **Route**: API route used for communication with the orchestrator, defined by `route` in `asr.cf`.

**Example**:
```ini
orchestrator_host=localhost
orchestrator_port=6000
route=/asr
```

**Notes**:
- The ASR Engine must register with the orchestrator upon startup to establish communication channels.
- Ensure that the orchestrator's command port is accessible and not blocked by firewalls.

---

## Supported Commands

The ASR Engine supports a variety of commands for managing its operations and interactions with the orchestrator. These commands can be sent via the orchestrator's command interface or through direct socket connections.

### General Commands

| Command               | Description                                         | Usage Example                 |
|-----------------------|-----------------------------------------------------|-------------------------------|
| `/help`               | Displays help information about available commands. | `/help`                       |
| `/list` or `/available` | Lists all known peripherals connected to the ASR Engine. | `/list` or `/available`   |
| `/exit`               | Exits command mode or gracefully shuts down the engine. | `/exit`                     |

### Special Commands

| Command                                                                                           | Description                                                                         | Usage Example                                                                       |
|---------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| `/register <name> <uuid> <port>`                                                                    | Registers a new peripheral with the orchestrator.                                  | `/register ASR_Engine 123e4567-e89b-12d3-a456-426614174000 6200`                   |
| `/data <uuid> <data>`                                                                               | Sends recognized text from the ASR Engine to be processed by the orchestrator.      | `/data 123e4567-e89b-12d3-a456-426614174000 Temperature is 25°C`                    |
| `/info`                                                                                             | Sends detailed configuration information to the orchestrator.                       | `/info`                                                                             |
| `/exit`                                                                                             | Sends an exit acknowledgment to the orchestrator and shuts down the ASR Engine.      | `/exit`                                                                             |

**Notes**:
- Commands should be sent as plain text terminated by a newline (`\n`).
- The ASR Engine will process commands and respond accordingly.

---

## Integration with Orchestrator

The ASR Engine is designed to work in tandem with the Orchestrator, facilitating seamless data routing and peripheral management within a networked environment.

### Registering with the Orchestrator

Upon startup, the ASR Engine attempts to register itself with the orchestrator to establish communication channels.

#### Registration Process

1. **Generate UUID**: If not already present, the ASR Engine generates a unique UUID to identify itself.

2. **Send Registration Command**: The ASR Engine sends a `/register` command to the orchestrator's command port with its name, UUID, and listening port.
   ```bash
   /register ASR_Engine 123e4567-e89b-12d3-a456-426614174000 6200
   ```

3. **Receive Acknowledgment**: The orchestrator responds with an acknowledgment (`/ack`) containing the data port to be used for sending responses.
   ```
   /ack 6001
   ```

4. **Update Configuration**: The ASR Engine updates its configuration with the received data port for future communications.

**Notes**:
- Registration retries are attempted with exponential backoff in case of failures.
- Successful registration is indicated by receiving a valid `/ack` message from the orchestrator.

### Handling Orchestrator Commands

The ASR Engine listens for incoming connections and commands from the orchestrator on the specified port. It can handle commands such as `/info` to provide configuration details or `/exit` to shut down gracefully.

#### Available Commands

| Command | Description                                          | Response                                     |
|---------|------------------------------------------------------|----------------------------------------------|
| `/info` | Provides detailed configuration information.        | Sends a JSON-formatted response with config. |
| `/exit` | Signals the ASR Engine to shut down gracefully.     | Sends an exit acknowledgment and terminates.  |

**Example**:
```bash
# From orchestrator
/info
```
**Response**:
```json
ASR_Engine
123e4567-e89b-12d3-a456-426614174000
port=6200
port_range=6200-6300
orchestrator_host=localhost
orchestrator_port=6000
language_code=en-US
use_ssl=False
ssl_cert=
server=localhost:50051
input_device=Microphone (Realtek)
sample_rate=16000
chunk_size=1600
script_uuid=123e4567-e89b-12d3-a456-426614174000
```

---

## Inference Process

The core functionality of the ASR Engine revolves around processing incoming audio data using NVIDIA's Riva ASR service to generate recognized text. This section outlines how audio input is handled, processed, and how results are transmitted back to the orchestrator.

### Using the Riva ASR Service

The ASR Engine utilizes the Riva ASR API to perform speech-to-text conversions. Below is the workflow of the inference process:

#### Inference Workflow

1. **Receive Audio Data**: The ASR Engine receives audio data chunks via an input stream (e.g., microphone).

2. **Stream Audio to Riva**: Audio chunks are sent to the Riva ASR service for real-time transcription.

3. **Receive Transcription**: Riva processes the audio and returns recognized text.

4. **Send Recognized Text to Orchestrator**: The ASR Engine formats the recognized text and sends it back to the orchestrator for further routing or action.

#### Detailed Steps

1. **Audio Input Configuration**:
   - **Sample Rate**: Defined by `sample_rate` (default: `16000` Hz).
   - **Chunk Size**: Number of audio samples per chunk (default: `1600` samples, corresponding to 100ms at 16kHz).
   - **Input Device**: Specified by `input_device`. If left empty, the default system input device is used.

2. **Initialize Riva ASR Service**:
   - **Connection**: Uses gRPC to connect to the Riva server (`server` parameter).
   - **SSL Configuration**: If `use_ssl` is `True`, SSL certificates are used for secure communication.

3. **Streaming Recognition**:
   - **Recognition Config**: Sets up recognition parameters such as encoding, sample rate, language code, and model parameters.
   - **Audio Stream**: Captures audio in real-time using the `sounddevice` library and streams it to Riva for processing.

4. **Handling Riva Responses**:
   - **Final Results**: Processes only final recognition results (`is_final` flag).
   - **Transcription**: Extracts the recognized transcript from the Riva response.

5. **Transmitting Results**:
   - **Formatting**: Prepares the recognized text in a format suitable for the orchestrator (e.g., JSON).
   - **Socket Communication**: Connects to the orchestrator's data port and sends the recognized text along with the script UUID.

**Notes**:
- Ensure that the Riva ASR service is running and accessible at the specified `server` address.
- Adjust `sample_rate` and `chunk_size` based on the requirements of your application and the capabilities of your audio hardware.

---

## Examples

### Registering the ASR Engine

**Scenario**: Register the ASR Engine with the orchestrator to enable data routing.

**Command Sent to Orchestrator**:
```bash
/register ASR_Engine 123e4567-e89b-12d3-a456-426614174000 6200
```

**Explanation**:
- **Name**: `ASR_Engine` - Identifies the peripheral.
- **UUID**: `123e4567-e89b-12d3-a456-426614174000` - Unique identifier for the ASR Engine.
- **Port**: `6200` - Port on which the ASR Engine is listening for connections.

**Orchestrator Response**:
```
/ack 6001
```

**Result**:
- The ASR Engine is now registered with the orchestrator.
- The data port for sending responses is set to `6001`.

### Sending Recognized Text to Orchestrator

**Scenario**: The ASR Engine recognizes spoken input and sends the transcribed text to the orchestrator for further processing.

**Process**:

1. **Audio Input**: User speaks "Temperature is 25 degrees Celsius."

2. **ASR Engine Processing**:
   - Captures the audio chunk.
   - Streams it to the Riva ASR service.
   - Receives the recognized text: "Temperature is 25 degrees Celsius."

3. **Sending Data to Orchestrator**:
   - Formats the recognized text.
   - Connects to the orchestrator's data port (`6001`).
   - Sends the following message:
     ```
     /data 123e4567-e89b-12d3-a456-426614174000 Temperature is 25 degrees Celsius.
     ```

**Orchestrator's Handling**:
- Receives the data from the ASR Engine.
- Routes the recognized text to the appropriate peripheral (e.g., a monitoring dashboard or actuator) based on configured routes.

**Result**:
- The orchestrator processes the recognized text, enabling actions such as logging the temperature or triggering alerts.

---

## Troubleshooting

### Common Issues

1. **Unable to Bind to Specified Port Range**
   - **Symptom**: ASR Engine fails to start or displays errors related to port binding.
   - **Solution**:
     - Ensure that the ports in the specified range (`port_range`) are not in use by other applications.
     - Modify the `port_range` in `asr.cf` or use the `--port-range` argument to specify a different range.
     - Example:
       ```bash
       python3 asr.py --port-range 6300-6400
       ```

2. **Orchestrator Registration Failure**
   - **Symptom**: ASR Engine cannot register with the orchestrator and displays retry messages.
   - **Solution**:
     - Verify that the orchestrator is running and accessible at the specified `orchestrator_host` and `orchestrator_port`.
     - Check network connectivity and firewall settings that might block the connection.
     - Ensure that the orchestrator is configured to accept registrations.

3. **Riva ASR Service Unavailable**
   - **Symptom**: ASR Engine cannot perform inferences and logs errors related to the Riva ASR service.
   - **Solution**:
     - Ensure that the Riva ASR service is installed and running.
     - Verify the API endpoint (`localhost:50051` by default) is accessible.
     - Check for any network issues or firewall rules that might block access to the Riva API.

4. **Invalid Configuration Parameters**
   - **Symptom**: ASR Engine fails to start or behaves unexpectedly due to incorrect configurations.
   - **Solution**:
     - Review the `asr.cf` configuration file for syntax errors or invalid parameter values.
     - Use command-line arguments to override and test different configurations.
     - Example:
       ```bash
       python3 asr.py --sample-rate 44100
       ```

5. **Audio Input Issues**
   - **Symptom**: No audio is captured, or audio quality is poor.
   - **Solution**:
     - Verify that the correct `input_device` is specified and is functional.
     - Test the microphone using other applications to ensure it's working.
     - Adjust `sample_rate` and `chunk_size` based on the audio hardware capabilities.

6. **Unexpected Shutdowns or Crashes**
   - **Symptom**: ASR Engine terminates unexpectedly without clear error messages.
   - **Solution**:
     - Review the console logs for any error messages or stack traces.
     - Ensure that all dependencies are correctly installed and up to date.
     - Validate the configuration file (`asr.cf`) for any syntax errors or invalid parameters.
     - Restart the ASR Engine and monitor for recurring issues.

### Viewing Logs

- **Console Output**: All significant events, including registrations, data processing, inferences, and errors, are logged to the console.
- **Redirecting Logs to a File**: For persistent logging, redirect console output to a log file.
  ```bash
  python3 asr.py > asr_engine.log 2>&1
  ```

**Note**: Regularly monitor and manage the log files to prevent excessive storage usage.

---

## FAQ

### 1. **How do I register the ASR Engine with the Orchestrator?**

**Answer**:
- The ASR Engine automatically attempts to register with the orchestrator upon startup.
- Ensure that the orchestrator is running and accessible at the specified `orchestrator_host` and `orchestrator_port`.
- If registration fails, check network connectivity, port configurations, and orchestrator status.

### 2. **Can I change the language code used by the ASR Engine?**

**Answer**:
- Yes. Modify the `language_code` parameter in the `asr.cf` file or use the `--language-code` command-line argument.
  
  **Example**:
  ```bash
  python3 asr.py --language-code en-GB
  ```

### 3. **What happens if the Riva ASR service is not running when I start the ASR Engine?**

**Answer**:
- The ASR Engine will notify you that the Riva ASR service is unavailable.
- It will continue running and periodically attempt to connect to the Riva ASR service.
- Once the Riva ASR service becomes available, the ASR Engine can resume normal operations.

### 4. **How can I view or modify the routes managed by the ASR Engine?**

**Answer**:
- Routes are managed via commands sent to the orchestrator.
- Use the `/routes` commands to add, remove, or list routes.

  **Example**:
  ```bash
  /routes add Route1 ASR_Engine Actuator1
  /routes info
  /routes remove Route1
  ```

### 5. **Is the ASR Engine secure?**

**Answer**:
- **Default Security**: The ASR Engine does not implement authentication or encryption by default.
- **Recommendations**:
  - Run the ASR Engine within a secure, trusted network environment.
  - Implement firewall rules to restrict access to the command and data ports.
  - Enhance the script to include authentication mechanisms if deploying in sensitive environments.

### 6. **Can I run multiple instances of the ASR Engine on the same machine?**

**Answer**:
- Yes, provided each instance is configured to use a unique port within the specified `port_range`.
- Ensure that there are no port conflicts by properly configuring the `port_range` and monitoring active ports.

### 7. **How do I update the configuration after the ASR Engine has started?**

**Answer**:
- Modify the `asr.cf` file with the desired configuration changes.
- Restart the ASR Engine to apply the new configurations.
  
  **Example**:
  ```bash
  python3 asr.py
  ```

### 8. **What is the purpose of the `script_uuid` parameter?**

**Answer**:
- The `script_uuid` uniquely identifies the ASR Engine peripheral within the orchestrator-managed ecosystem.
- It is automatically generated if left empty in the configuration file.
- Ensures that responses sent back to the orchestrator are correctly attributed to the ASR Engine.

### 9. **How does the `chunk_size` parameter affect audio processing?**

**Answer**:
- The `chunk_size` defines the number of audio samples captured per chunk.
- Determines the duration of each audio chunk based on the `sample_rate`.
  
  **Example**:
  - `chunk_size=1600` with `sample_rate=16000` corresponds to 100ms per chunk.

**Notes**:
- Adjust `chunk_size` based on the desired latency and processing capabilities.
- Smaller chunks lead to lower latency but may increase processing overhead.
