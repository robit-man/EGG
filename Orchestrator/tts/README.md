# TTS Engine API Documentation

## Table of Contents

1. [Introduction](#introduction)
2. [Installation and Setup](#installation-and-setup)
   - [Prerequisites](#prerequisites)
   - [Installation Steps](#installation-steps)
3. [Configuration](#configuration)
   - [Configuration File (`tts.cf`)](#configuration-file-ttscf)
   - [Command-Line Arguments](#command-line-arguments)
4. [Running the TTS Engine](#running-the-tts-engine)
   - [Basic Execution](#basic-execution)
   - [With Command-Line Arguments](#with-command-line-arguments)
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
   - [Using the Riva TTS Service](#using-the-riva-tts-service)
9. [Examples](#examples)
   - [Registering the TTS Engine](#registering-the-tts-engine)
   - [Sending Text to the TTS Engine](#sending-text-to-the-tts-engine)
10. [Troubleshooting](#troubleshooting)
11. [FAQ](#faq)
12. [Repository Information](#repository-information)

---

## Introduction

The **TTS Engine** is a Python-based server application designed to perform Text-to-Speech (TTS) conversions using NVIDIA's Riva TTS service. It operates as a peripheral within an orchestrator-managed ecosystem, enabling real-time synthesis of spoken audio from text inputs. The TTS Engine listens for text input via various interfaces, processes it through the Riva TTS API, and outputs the synthesized audio either through speakers or saves it to a file. Additionally, it communicates with an orchestrator for seamless integration and management within a network of peripherals.

This documentation provides a comprehensive guide on installing, configuring, and effectively using the TTS Engine, including its command interfaces, integration with the orchestrator, and the inference process.

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
  - `pyaudio`: For audio playback.
  - `sounddevice`: For audio input (if applicable).
  - `riva.client`: NVIDIA Riva client for TTS.
  - `curses`: For terminal-based UI (typically included on Unix-based systems).
  - Other standard libraries: `threading`, `time`, `socket`, `json`, `uuid`, `os`, `queue`, `argparse`, `re`, `traceback`.

  **Installation**:
  ```bash
  pip install grpcio numpy pyaudio sounddevice
  # riva.client may require specific installation steps as per NVIDIA Riva documentation
  ```

- **NVIDIA Riva TTS Service**: The TTS Engine relies on NVIDIA's Riva TTS service. Ensure that Riva is installed and running. Refer to the [NVIDIA Riva Documentation](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/index.html) for installation and setup instructions.

### Installation Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/robit-man/EGG.git
   cd EGG/tts
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

4. **Ensure Riva TTS Service is Running**:
   - Start the Riva TTS server as per the [Riva Documentation](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/index.html).
   - Verify it's running by checking the gRPC endpoint (default: `localhost:50051`).
   ```bash
   # Example command to check Riva TTS service
   curl http://localhost:50051/api/health
   ```

---

## Configuration

The TTS Engine uses a configuration file (`tts.cf`) to manage its settings. This file is automatically created in the same directory as the script if it does not exist. Additionally, command-line arguments can be used to override configuration parameters.

### Configuration File (`tts.cf`)

#### Default Configuration

```ini
input_mode=terminal
input_format=chunk
output_mode=speaker
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
voice=
route=/tts_route
script_uuid=123e4567-e89b-12d3-a456-426614174000
```

#### Configuration Parameters

| Parameter           | Description                                                                                                                                               | Default Value                      |
|---------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------|
| `input_mode`        | Mode of input reception. Options: `terminal`, `port`, `route`.                                                                                              | `terminal`                         |
| `input_format`      | Format of incoming data. Options: `streaming`, `chunk`. Currently, only `chunk` is implemented.                                                             | `chunk`                            |
| `output_mode`       | Mode of output dispatch. Options: `speaker`, `file`, `stream`.                                                                                             | `speaker`                          |
| `port`              | Starting port number to listen on for incoming connections.                                                                                                | `6200`                             |
| `port_range`        | Range of ports to try if the initial port is unavailable (e.g., `6200-6300`).                                                                                | `6200-6300`                        |
| `orchestrator_host` | Host address of the orchestrator.                                                                                                                           | `localhost`                        |
| `orchestrator_port` | Port number of the orchestrator's command interface.                                                                                                       | `6000`                             |
| `language_code`     | Language code for TTS recognition (e.g., `en-US`).                                                                                                          | `en-US`                            |
| `use_ssl`           | Whether to use SSL/TLS for gRPC connections (`True` or `False`).                                                                                           | `False`                            |
| `ssl_cert`          | Path to the SSL/TLS certificate file (required if `use_ssl` is `True`).                                                                                    | *(empty)*                          |
| `server`            | Address of the Riva TTS server in the format `host:port`.                                                                                                  | `localhost:50051`                  |
| `input_device`      | Name of the input audio device to use. Leave empty for the default device.                                                                                  | *(empty)*                          |
| `sample_rate`       | Sample rate for audio input in Hz.                                                                                                                         | `16000`                            |
| `chunk_size`        | Number of audio samples per chunk (e.g., `1600` for 100ms chunks at 16kHz).                                                                                 | `1600`                             |
| `voice`             | Voice name to use for TTS. An empty string means no voice is set and the user will be prompted to select one.                                               | *(empty)*                          |
| `route`             | API route used for communication with the orchestrator in `route` input mode.                                                                              | `/tts_route`                       |
| `script_uuid`       | Unique identifier for the TTS Engine script. Automatically generated if empty.                                                                               | `123e4567-e89b-12d3-a456-426614174000` |

#### Modifying Configuration

You can manually edit the `tts.cf` file to change configurations. Additionally, command-line arguments can be used to override specific parameters during runtime.

### Command-Line Arguments

The TTS Engine supports various command-line arguments to override configuration parameters. These arguments take precedence over the configurations specified in the `tts.cf` file.

#### Supported Arguments

| Argument          | Description                                                                           | Example                                          |
|-------------------|---------------------------------------------------------------------------------------|--------------------------------------------------|
| `--port`          | Specify the starting port number to listen on.                                       | `--port 6200`                                    |
| `--port-range`    | Define the range of ports to try if the initial port is unavailable (e.g., `6200-6300`). | `--port-range 6200-6300`                        |
| `-o`, `--output`  | Specify the output `.wav` file to write synthesized audio.                           | `--output output.wav`                            |
| `--voice`         | A voice name to use for TTS.                                                         | `--voice "en-US-Wavenet-D"`                      |
| `--language-code` | Set the language code for TTS.                                                       | `--language-code en-GB`                           |
| `--use-ssl`       | Enable SSL/TLS authentication.                                                        | `--use-ssl`                                      |
| `--ssl-cert`      | Path to the SSL/TLS certificate file (required if `--use-ssl` is set).               | `--ssl-cert /path/to/cert.pem`                   |
| `--server`        | Specify the Riva TTS server URI and port.                                            | `--server localhost:50051`                        |
| `--input-device`  | Define the input audio device to use.                                                 | `--input-device "Microphone (Realtek)"`           |
| `--sample-rate`   | Set the sample rate for audio input in Hz.                                          | `--sample-rate 44100`                             |
| `--chunk-size`    | Define the number of samples per audio chunk.                                       | `--chunk-size 1600`                                |

#### Usage Example

```bash
python3 tts.py --port 6200 --port-range 6200-6300 --output output.wav --voice "en-US-Wavenet-D" --language-code en-US --use-ssl --ssl-cert /path/to/cert.pem --server localhost:50051 --input-device "Microphone (Realtek)" --sample-rate 16000 --chunk-size 1600
```

---

## Running the TTS Engine

To start the TTS Engine, execute the script with optional command-line arguments as needed.

### Basic Execution

```bash
python3 tts.py
```

### With Command-Line Arguments

```bash
python3 tts.py --port 6200 --port-range 6200-6300 --orchestrator-host localhost --orchestrator-port 6000 --voice "en-US-Wavenet-D" --language-code en-US --use-ssl --ssl-cert /path/to/cert.pem --server localhost:50051 --input-device "Microphone (Realtek)" --sample-rate 16000 --chunk-size 1600
```

### Execution Steps

1. **Load Configuration**: Reads settings from `tts.cf`. If the file doesn't exist, it generates a new UUID and creates the configuration file with default settings.

2. **Parse Command-Line Arguments**: Overrides configuration parameters based on provided arguments.

3. **Initialize TTS Stub**: Establishes a connection to the Riva TTS service using gRPC, applying SSL/TLS settings if specified.

4. **Retrieve Available Voices**: Queries the Riva TTS service to obtain a list of available voices. If a voice is specified in the configuration, it attempts to use that voice. Otherwise, it prompts the user to select a voice from the available options.

5. **Start TTS Thread**: Launches a background thread to handle TTS generation and audio playback based on the selected voice and output mode.

6. **Handle Input**: Depending on the `input_mode`, the TTS Engine listens for text input via the terminal, a designated port, or an HTTP route.

7. **Process Text and Synthesize Audio**: Received text is processed to replace special characters and numbers, split into manageable chunks, sent to the Riva TTS service for synthesis, and the resulting audio is played through speakers or saved to a file.

8. **Communication with Orchestrator**: Registers with the orchestrator to announce its presence and receives acknowledgments to facilitate data routing.

**Note**: Ensure that the orchestrator and Riva TTS service are running and accessible at the specified hosts and ports before starting the TTS Engine.

---

## Command Interfaces

The TTS Engine interacts with the orchestrator and peripherals through specified ports and supports various command interfaces for management and data exchange.

### Port Configuration

- **Purpose**: Defines the range of ports the TTS Engine will attempt to bind to for incoming connections.
- **Configuration**: Specified in the `port` and `port_range` parameters within `tts.cf` or via the `--port` and `--port-range` command-line arguments.
- **Format**: A hyphen-separated range (e.g., `6200-6300`) or a comma-separated list (e.g., `6200,6201,6202`).
- **Example**:
  ```ini
  port_range=6200-6300
  ```

### Orchestrator Connection

- **Orchestrator Host**: Defined by `orchestrator_host` in `tts.cf` or via `--orchestrator-host`.
- **Orchestrator Port**: Defined by `orchestrator_port` in `tts.cf` or via `--orchestrator-port`.
- **Route**: API route used for communication with the orchestrator, defined by `route` in `tts.cf`.

**Example**:
```ini
orchestrator_host=localhost
orchestrator_port=6000
route=/tts_route
```

**Notes**:
- The TTS Engine must register with the orchestrator upon startup to establish communication channels.
- Ensure that the orchestrator's command port is accessible and not blocked by firewalls.

---

## Supported Commands

The TTS Engine supports a variety of commands for managing its operations and interactions with the orchestrator. These commands can be sent via the orchestrator's command interface or through direct socket connections.

### General Commands

| Command               | Description                                            | Usage Example            |
|-----------------------|--------------------------------------------------------|--------------------------|
| `/help`               | Displays help information about available commands.    | `/help`                  |
| `/list` or `/available` | Lists all known peripherals connected to the TTS Engine. | `/list` or `/available`  |
| `/exit`               | Exits command mode or gracefully shuts down the engine. | `/exit`                  |

### Special Commands

| Command                                                                                           | Description                                                                         | Usage Example                                                                           |
|---------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|
| `/register <name> <uuid> <port>`                                                                    | Registers a new peripheral with the orchestrator.                                  | `/register TTS_Engine 123e4567-e89b-12d3-a456-426614174000 6200`                        |
| `/data <uuid> <data>`                                                                               | Sends recognized text from the TTS Engine to be processed by the orchestrator.      | `/data 123e4567-e89b-12d3-a456-426614174000 Temperature is 25°C`                         |
| `/info`                                                                                            | Sends detailed configuration information to the orchestrator.                       | `/info`                                                                                 |
| `/exit`                                                                                            | Sends an exit acknowledgment to the orchestrator and shuts down the TTS Engine.      | `/exit`                                                                                 |

**Notes**:
- Commands should be sent as plain text terminated by a newline (`\n`).
- The TTS Engine will process commands and respond accordingly.

---

## Integration with Orchestrator

The TTS Engine is designed to work in tandem with the Orchestrator, facilitating seamless data routing and peripheral management within a networked environment.

### Registering with the Orchestrator

Upon startup, the TTS Engine attempts to register itself with the orchestrator to establish communication channels.

#### Registration Process

1. **Generate UUID**: If not already present, the TTS Engine generates a unique UUID to identify itself.

2. **Send Registration Command**: The TTS Engine sends a `/register` command to the orchestrator's command port with its name, UUID, and listening port.
   ```bash
   /register TTS_Engine 123e4567-e89b-12d3-a456-426614174000 6200
   ```

3. **Receive Acknowledgment**: The orchestrator responds with an acknowledgment (`/ack`) containing the data port to be used for sending responses.
   ```
   /ack 6001
   ```

4. **Update Configuration**: The TTS Engine updates its configuration with the received data port for future communications.

**Notes**:
- Registration retries are attempted with exponential backoff in case of failures.
- Successful registration is indicated by receiving a valid `/ack` message from the orchestrator.

### Handling Orchestrator Commands

The TTS Engine listens for incoming connections and commands from the orchestrator on the specified port. It can handle commands such as `/info` to provide configuration details or `/exit` to shut down gracefully.

#### Available Commands

| Command | Description                                          | Response                                     |
|---------|------------------------------------------------------|----------------------------------------------|
| `/info` | Provides detailed configuration information.        | Sends a configuration summary.               |
| `/exit` | Signals the TTS Engine to shut down gracefully.     | Sends an exit acknowledgment and terminates.  |

**Example**:
```bash
# From orchestrator
/info
```
**Response**:
```
TTS_Engine
123e4567-e89b-12d3-a456-426614174000
input_mode=terminal
input_format=chunk
output_mode=speaker
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
voice=en-US-Wavenet-D
route=/tts_route
script_uuid=123e4567-e89b-12d3-a456-426614174000
EOF
```

---

## Inference Process

The core functionality of the TTS Engine revolves around processing incoming text data using NVIDIA's Riva TTS service to generate synthesized audio. This section outlines how text input is handled, processed, and how audio output is generated and transmitted.

### Using the Riva TTS Service

The TTS Engine utilizes the Riva TTS API to perform text-to-speech conversions. Below is the workflow of the inference process:

#### Inference Workflow

1. **Receive Text Input**: The TTS Engine receives text input via the configured input mode (terminal, port, or route).

2. **Process Text**:
   - **Replace Special Characters and Numbers**: Converts special characters and numeric digits in the text to their spelled-out versions to ensure accurate speech synthesis.
     - **Example**: "Temperature is 25°C!" becomes "Temperature is twenty five degrees Celsius exclamation mark."
   
3. **Split Text into Chunks**: Divides long texts into smaller chunks (e.g., 200 characters) to prevent server timeouts and manage processing efficiently.

4. **Send Text to Riva TTS Service**: For each text chunk, the TTS Engine sends a synthesis request to the Riva TTS API with the selected voice and language configurations.

5. **Receive Synthesized Audio**: Receives the synthesized audio data from the Riva TTS service in response to each synthesis request.

6. **Output Audio**:
   - **Speaker Mode**: Plays the audio through the system's speakers in real-time.
   - **File Mode**: Writes the audio data to a specified `.wav` file for later playback.
   - **Stream Mode**: (Future implementation) Streams audio data to another service or peripheral.

7. **Send Recognized Text to Orchestrator**: After synthesis, the TTS Engine sends the recognized text back to the orchestrator for further routing or action.

#### Detailed Steps

1. **Text Input Configuration**:
   - **Input Mode**: Defined by `input_mode` (`terminal`, `port`, `route`).
   - **Input Format**: Defined by `input_format` (`streaming`, `chunk`).
   - **Output Mode**: Defined by `output_mode` (`speaker`, `file`, `stream`).
   - **Voice Selection**: Specified by `voice`. If not set, the user is prompted to select from available voices.

2. **Initialize Riva TTS Service**:
   - **Connection**: Uses gRPC to connect to the Riva TTS server (`server` parameter).
   - **SSL Configuration**: If `use_ssl` is `True`, SSL certificates are used for secure communication.

3. **Text Processing**:
   - **Special Characters**: Replaces special characters with their spelled-out names using a predefined mapping.
   - **Numbers**: Converts numeric digits to their word equivalents using a number-to-words function.

4. **Text Chunking**:
   - Splits processed text into manageable chunks (e.g., 200 characters) to optimize synthesis and prevent overloading the TTS service.

5. **Synthesize Speech**:
   - Sends each text chunk to the Riva TTS API along with the selected voice and language configurations.
   - Receives synthesized audio data in response.

6. **Audio Output**:
   - **Speaker Mode**: Uses `pyaudio` to play the audio through the system's speakers.
   - **File Mode**: Writes the audio data to a `.wav` file specified by the user.
   - **Stream Mode**: (Future implementation) Streams audio data to another service or peripheral.

7. **Communication with Orchestrator**:
   - After successful synthesis, the TTS Engine sends the recognized text back to the orchestrator using socket communication on the configured data port.

**Notes**:
- Ensure that the Riva TTS service is running and accessible at the specified server address.
- Adjust `sample_rate` and `chunk_size` based on the requirements of your application and the capabilities of your audio hardware.

---

## Examples

### Registering the TTS Engine

**Scenario**: Register the TTS Engine with the orchestrator to enable data routing.

**Command Sent to Orchestrator**:
```bash
/register TTS_Engine 123e4567-e89b-12d3-a456-426614174000 6200
```

**Explanation**:
- **Name**: `TTS_Engine` - Identifies the peripheral.
- **UUID**: `123e4567-e89b-12d3-a456-426614174000` - Unique identifier for the TTS Engine.
- **Port**: `6200` - Port on which the TTS Engine is listening for connections.

**Orchestrator Response**:
```
/ack 6001
```

**Result**:
- The TTS Engine is now registered with the orchestrator.
- The data port for sending responses is set to `6001`.

### Sending Text to the TTS Engine

**Scenario**: Send text input to the TTS Engine via terminal input mode for synthesis and playback.

**Process**:

1. **User Input**: The user types the following text into the terminal:
   ```
   Hello, world! The temperature is 25°C.
   ```

2. **TTS Engine Processing**:
   - **Text Replacement**:
     - Special Characters:
       - `,` becomes `comma`
       - `!` becomes `exclamation mark`
     - Numbers:
       - `25` becomes `twenty five`
       - `°C` remains as `degrees Celsius` (assuming it's handled or left as is)
   
   - **Processed Text**:
     ```
     Hello comma world exclamation mark The temperature is twenty five degrees Celsius.
     ```

   - **Chunking**:
     - If the processed text exceeds the `max_chunk_size`, it's split into smaller chunks (e.g., 200 characters).

3. **Synthesis Request**:
   - Sends the processed text chunks to the Riva TTS service for synthesis using the selected voice and language configurations.

4. **Audio Output**:
   - **Speaker Mode**: The synthesized speech is played through the system's speakers in real-time.
   - **File Mode**: The synthesized audio is saved to the specified `.wav` file.

5. **Sending to Orchestrator**:
   - After synthesis, the TTS Engine sends the recognized text back to the orchestrator:
     ```
     /data 123e4567-e89b-12d3-a456-426614174000 Hello, world! The temperature is 25°C.
     ```

**Result**:
- The user hears the synthesized speech corresponding to the input text.
- The orchestrator processes the recognized text for further actions, such as logging or triggering other peripherals.

---

## Troubleshooting

### Common Issues

1. **Unable to Bind to Specified Port Range**
   - **Symptom**: TTS Engine fails to start or displays errors related to port binding.
   - **Solution**:
     - Ensure that the ports in the specified range (`port_range`) are not in use by other applications.
     - Modify the `port_range` in `tts.cf` or use the `--port-range` argument to specify a different range.
     - Example:
       ```bash
       python3 tts.py --port-range 6300-6400
       ```

2. **Orchestrator Registration Failure**
   - **Symptom**: TTS Engine cannot register with the orchestrator and displays retry messages.
   - **Solution**:
     - Verify that the orchestrator is running and accessible at the specified `orchestrator_host` and `orchestrator_port`.
     - Check network connectivity and firewall settings that might block the connection.
     - Ensure that the orchestrator is configured to accept registrations.

3. **Riva TTS Service Unavailable**
   - **Symptom**: TTS Engine cannot perform inferences and logs errors related to the Riva TTS service.
   - **Solution**:
     - Ensure that the Riva TTS service is installed and running.
     - Verify the API endpoint (`localhost:50051` by default) is accessible.
     - Check for any network issues or firewall rules that might block access to the Riva API.

4. **Invalid Configuration Parameters**
   - **Symptom**: TTS Engine fails to start or behaves unexpectedly due to incorrect configurations.
   - **Solution**:
     - Review the `tts.cf` configuration file for syntax errors or invalid parameter values.
     - Use command-line arguments to override and test different configurations.
     - Example:
       ```bash
       python3 tts.py --sample-rate 44100
       ```

5. **Audio Output Issues**
   - **Symptom**: No audio is heard, or audio quality is poor.
   - **Solution**:
     - Verify that the correct `output_mode` is set (`speaker`, `file`, or `stream`).
     - If using `speaker` mode, ensure that speakers are connected and functioning.
     - If using `file` mode, verify that the output file path is correct and writable.
     - Adjust `sample_rate` and `chunk_size` based on audio hardware capabilities.

6. **Unexpected Shutdowns or Crashes**
   - **Symptom**: TTS Engine terminates unexpectedly without clear error messages.
   - **Solution**:
     - Review the console logs for any error messages or stack traces.
     - Ensure that all dependencies are correctly installed and up to date.
     - Validate the configuration file (`tts.cf`) for any syntax errors or invalid parameters.
     - Restart the TTS Engine and monitor for recurring issues.

### Viewing Logs

- **Console Output**: All significant events, including registrations, data processing, inferences, and errors, are logged to the console.
- **Redirecting Logs to a File**: For persistent logging, redirect console output to a log file.
  ```bash
  python3 tts.py > tts_engine.log 2>&1
  ```

**Note**: Regularly monitor and manage the log files to prevent excessive storage usage.

---

## FAQ

### 1. **How do I register the TTS Engine with the Orchestrator?**

**Answer**:
- The TTS Engine automatically attempts to register with the orchestrator upon startup.
- Ensure that the orchestrator is running and accessible at the specified `orchestrator_host` and `orchestrator_port`.
- If registration fails, check network connectivity, port configurations, and orchestrator status.

### 2. **Can I change the language code used by the TTS Engine?**

**Answer**:
- Yes. Modify the `language_code` parameter in the `tts.cf` file or use the `--language-code` command-line argument.
  
  **Example**:
  ```bash
  python3 tts.py --language-code en-GB
  ```

### 3. **What happens if the Riva TTS service is not running when I start the TTS Engine?**

**Answer**:
- The TTS Engine will notify you that the Riva TTS service is unavailable.
- It will continue running and periodically attempt to connect to the Riva TTS service.
- Once the Riva TTS service becomes available, the TTS Engine can resume normal operations.

### 4. **How can I view or modify the routes managed by the TTS Engine?**

**Answer**:
- Routes are managed via commands sent to the orchestrator.
- Use the `/routes` commands to add, remove, or list routes.
  
  **Example**:
  ```bash
  /routes add Route1 TTS_Engine Actuator1
  /routes info
  /routes remove Route1
  ```

### 5. **Is the TTS Engine secure?**

**Answer**:
- **Default Security**: The TTS Engine does not implement authentication or encryption by default.
- **Recommendations**:
  - Run the TTS Engine within a secure, trusted network environment.
  - Implement firewall rules to restrict access to the command and data ports.
  - Enhance the script to include authentication mechanisms if deploying in sensitive environments.

### 6. **Can I run multiple instances of the TTS Engine on the same machine?**

**Answer**:
- Yes, provided each instance is configured to use a unique port within the specified `port_range`.
- Ensure that there are no port conflicts by properly configuring the `port_range` and monitoring active ports.

### 7. **How do I update the configuration after the TTS Engine has started?**

**Answer**:
- Modify the `tts.cf` file with the desired configuration changes.
- Restart the TTS Engine to apply the new configurations.
  
  **Example**:
  ```bash
  python3 tts.py
  ```

### 8. **What is the purpose of the `script_uuid` parameter?**

**Answer**:
- The `script_uuid` uniquely identifies the TTS Engine peripheral within the orchestrator-managed ecosystem.
- It is automatically generated if left empty in the configuration file.
- Ensures that responses sent back to the orchestrator are correctly attributed to the TTS Engine.

### 9. **How does the `chunk_size` parameter affect audio processing?**

**Answer**:
- The `chunk_size` defines the number of audio samples captured per chunk.
- Determines the duration of each audio chunk based on the `sample_rate`.
  
  **Example**:
  - `chunk_size=1600` with `sample_rate=16000` corresponds to 100ms per chunk.

**Notes**:
- Adjust `chunk_size` based on the desired latency and processing capabilities.
- Smaller chunks lead to lower latency but may increase processing overhead.
