# SLM Engine API Documentation

## Table of Contents

1. [Introduction](#introduction)
2. [Installation and Setup](#installation-and-setup)
3. [Configuration](#configuration)
   - [Configuration File (`slm.cf`)](#configuration-file-slmcf)
   - [Command-Line Arguments](#command-line-arguments)
4. [Running the SLM Engine](#running-the-slm-engine)
5. [Command Interfaces](#command-interfaces)
   - [Port Range Configuration](#port-range-configuration)
   - [Orchestrator Connection](#orchestrator-connection)
6. [Supported Commands](#supported-commands)
   - [General Commands](#general-commands)
   - [Special Commands](#special-commands)
7. [Integration with Orchestrator](#integration-with-orchestrator)
   - [Registering with the Orchestrator](#registering-with-the-orchestrator)
   - [Sending Data to the Orchestrator](#sending-data-to-the-orchestrator)
8. [Inference Process](#inference-process)
   - [Using the Ollama API](#using-the-ollama-api)
9. [Examples](#examples)
   - [Registering a Peripheral](#registering-a-peripheral)
   - [Adding a Route](#adding-a-route)
   - [Sending Data Through a Route](#sending-data-through-a-route)
10. [Troubleshooting](#troubleshooting)
11. [FAQ](#faq)
12. [Repository Information](#repository-information)

---

## Introduction

The **SLM Engine** is a Python-based server application designed to interact with the **Orchestrator** for managing and routing data between various peripherals. It leverages socket programming to listen for commands and data, performs AI-driven inferences using the Ollama API, and communicates results back to the orchestrator. The SLM Engine supports dynamic configuration, multiple connection handling, and robust error management to ensure seamless integration within a network of peripherals.

This documentation provides a comprehensive guide on installing, configuring, and effectively using the SLM Engine, including its command interfaces, integration with the orchestrator, and the inference process.

---

## Installation and Setup

### Prerequisites

- **Python 3.6 or Higher**: Ensure that Python is installed on your system. Verify by running:
  ```bash
  python3 --version
  ```
- **Required Python Libraries**:
  - `curses`: Typically included with Python on Unix-based systems.
  - `requests`, `argparse`, `threading`, `socket`, `json`, `uuid`, `os`, `re`, `traceback`: These are standard Python libraries.
  
- **Ollama API**: The SLM Engine utilizes the Ollama API for AI-driven inferences. Ensure that the Ollama API is installed and running on your system. Refer to the [Ollama Documentation](https://ollama.com/docs) for installation instructions.

### Installation Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/robit-man/EGG.git
   cd EGG
   ```
   
2. **Navigate to the SLM Engine Directory**:
   ```bash
   cd slm
   ```
   
3. **Create a Virtual Environment (Optional but Recommended)**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
   
4. **Install Dependencies**:
   - Most required libraries are standard and come pre-installed with Python.
   - If you have additional dependencies, install them using `pip`:
     ```bash
     pip install requests
     ```
   
5. **Ensure Ollama API is Running**:
   - Start the Ollama API service as per the [Ollama Documentation](https://ollama.com/docs).
   - Verify it's running by accessing the API endpoint:
     ```bash
     curl http://localhost:11434/api/tags
     ```

---

## Configuration

The SLM Engine uses a configuration file (`slm.cf`) to manage its settings. This file is automatically created in the same directory as the script if it does not exist. Additionally, command-line arguments can be used to override configuration parameters.

### Configuration File (`slm.cf`)

#### Default Configuration

```ini
model_name=llama3.2:1b
input_mode=port
output_mode=port
input_format=streaming
output_format=streaming
port_range=6200-6300
orchestrator_host=localhost
orchestrator_port=6000
route=/slm
script_uuid=
system_prompt=You Respond Conversationally
temperature=0.7
top_p=0.9
max_tokens=150
repeat_penalty=1.0
inference_timeout=5
```

#### Configuration Parameters

| Parameter           | Description                                                     | Default Value         |
|---------------------|-----------------------------------------------------------------|-----------------------|
| `model_name`        | Name of the language model to use (e.g., `llama3.2:1b`).        | `llama3.2:1b`         |
| `input_mode`        | Mode of input reception (`port`, `terminal`, `route`).          | `port`                |
| `output_mode`       | Mode of output dispatch (`port`, `terminal`, `route`).          | `port`                |
| `input_format`      | Format of incoming data (`streaming`, `chunk`).                 | `streaming`           |
| `output_format`     | Format of outgoing data (`streaming`, `chunk`).                 | `streaming`           |
| `port_range`        | Range of ports to scan for incoming connections (e.g., `6200-6300`). | `6200-6300`         |
| `orchestrator_host` | Host address of the orchestrator.                               | `localhost`           |
| `orchestrator_port` | Port number of the orchestrator's command interface.            | `6000`                |
| `route`             | API route used for communication with the orchestrator.         | `/slm`                |
| `script_uuid`       | Unique identifier for the SLM Engine script. Automatically generated if empty. | *(auto-generated)* |
| `system_prompt`     | System prompt for the language model to guide responses.         | `You Respond Conversationally` |
| `temperature`       | Model parameter influencing randomness of responses.            | `0.7`                 |
| `top_p`             | Model parameter influencing diversity of responses.             | `0.9`                 |
| `max_tokens`        | Maximum number of tokens in model responses.                     | `150`                 |
| `repeat_penalty`    | Model parameter to discourage repetitive outputs.               | `1.0`                 |
| `inference_timeout` | Timeout in seconds for model inference operations.              | `5`                   |

#### Modifying Configuration

You can manually edit the `slm.cf` file to change configurations. Additionally, command-line arguments can be used to override specific parameters during runtime.

### Command-Line Arguments

The SLM Engine supports various command-line arguments to override configuration parameters. These arguments take precedence over the configurations specified in the `slm.cf` file.

#### Supported Arguments

| Argument                    | Description                                                     | Example                                |
|-----------------------------|-----------------------------------------------------------------|----------------------------------------|
| `--port-range`              | Specify the port range for connections.                        | `--port-range 6200-6300`               |
| `--orchestrator-host`       | Define the orchestrator's host address.                        | `--orchestrator-host 192.168.1.10`     |
| `--orchestrator-port`       | Define the orchestrator's command port.                        | `--orchestrator-port 7000`             |
| `--model-name`              | Specify the language model to use.                             | `--model-name llama3.2:1b`              |
| `--system-prompt`           | Set a custom system prompt for the language model.             | `--system-prompt "Assist the user effectively."` |
| `--temperature`             | Set the model's temperature parameter.                         | `--temperature 0.8`                     |
| `--top_p`                   | Set the model's top_p parameter.                               | `--top_p 0.95`                           |
| `--max_tokens`              | Define the maximum tokens in responses.                        | `--max_tokens 200`                       |
| `--repeat_penalty`          | Set the model's repeat penalty parameter.                      | `--repeat_penalty 1.2`                   |
| `--inference_timeout`       | Define the timeout for inference operations in seconds.        | `--inference_timeout 10`                 |

#### Usage Example

```bash
python3 slm.py --port-range 6200-6300 --orchestrator-host localhost --orchestrator-port 6000 --model-name llama3.2:1b --system-prompt "You are a helpful assistant." --temperature 0.7 --top_p 0.9 --max_tokens 150 --repeat_penalty 1.0 --inference_timeout 5
```

---

## Running the SLM Engine

To start the SLM Engine, execute the script with optional command-line arguments as needed.

### Basic Execution

```bash
python3 slm.py
```

### With Command-Line Arguments

```bash
python3 slm.py --port-range 6200-6300 --orchestrator-host localhost --orchestrator-port 6000
```

### Execution Steps

1. **Load Configuration**: Reads settings from `slm.cf`. If the file doesn't exist, it generates a new UUID and creates the configuration file with default settings.
2. **Parse Command-Line Arguments**: Overrides configuration parameters based on provided arguments.
3. **Check Ollama API Status**: Verifies if the Ollama API is running and accessible.
4. **Start Server**: Binds to an available port within the specified range and listens for incoming connections.
5. **Register with Orchestrator**: Sends a `/register` command to the orchestrator to announce its presence and receive an acknowledgment with the data port.
6. **Handle Connections**: Listens for incoming data or commands from connected clients.
7. **Perform Inference**: Processes received data using the Ollama API and sends responses back to the orchestrator.

**Note**: Ensure that the orchestrator is running and accessible at the specified host and port before starting the SLM Engine.

---

## Command Interfaces

The SLM Engine interacts with the orchestrator and peripherals through specified ports and supports various command interfaces for management and data exchange.

### Port Range Configuration

- **Purpose**: Defines the range of ports the SLM Engine will attempt to bind to for incoming connections.
- **Configuration**: Specified in the `port_range` parameter within `slm.cf` or via the `--port-range` command-line argument.
- **Format**: A hyphen-separated range (e.g., `6200-6300`) or a comma-separated list (e.g., `6200,6201,6202`).
- **Example**:
  ```ini
  port_range=6200-6300
  ```

### Orchestrator Connection

- **Orchestrator Host**: Defined by `orchestrator_host` in `slm.cf` or via `--orchestrator-host`.
- **Orchestrator Port**: Defined by `orchestrator_port` in `slm.cf` or via `--orchestrator-port`.
- **Route**: API route used for communication with the orchestrator, defined by `route` in `slm.cf`.
  
**Example**:
```ini
orchestrator_host=localhost
orchestrator_port=6000
route=/slm
```

**Notes**:
- The SLM Engine must register with the orchestrator upon startup to establish communication channels.
- Ensure that the orchestrator's command port is accessible and not blocked by firewalls.

---

## Supported Commands

The SLM Engine supports a variety of commands for managing its operations, peripherals, and routes. These commands can be sent via the Command Port (`6000`) or through the orchestrator's interface.

### General Commands

| Command                   | Description                                              | Usage Example                              |
|---------------------------|----------------------------------------------------------|--------------------------------------------|
| `/help`                   | Displays help information about available commands.      | `/help`                                    |
| `/list` or `/available`    | Lists all known peripherals connected to the SLM Engine. | `/list` or `/available`                    |
| `/exit`                   | Exits command mode or gracefully shuts down the engine. | `/exit`                                    |

### Special Commands

| Command                                       | Description                                                             | Usage Example                                         |
|-----------------------------------------------|-------------------------------------------------------------------------|-------------------------------------------------------|
| `/register <name> <uuid> <port>`              | Registers a new peripheral with the orchestrator.                      | `/register Sensor1 123e4567-e89b-12d3-a456-426614174000 6201` |
| `/data <uuid> <data>`                          | Sends data from a peripheral to be processed by the SLM Engine.         | `/data 123e4567-e89b-12d3-a456-426614174000 Temperature:25°C` |
| `/routes help`                                | Displays help information about route commands.                         | `/routes help`                                       |
| `/routes add <route-name> <incoming-peripheral-name> <outgoing-peripheral-name>` | Adds a new route between two peripherals.                               | `/routes add Route1 Sensor1 Actuator1`                |
| `/routes remove <route-name>`                  | Removes an existing route by name.                                     | `/routes remove Route1`                               |
| `/routes info`                                | Lists all configured routes with details.                              | `/routes info`                                        |

**Notes**:
- Commands should be sent as plain text terminated by a newline (`\n`).
- The SLM Engine will process commands and respond accordingly.

---

## Integration with Orchestrator

The SLM Engine is designed to work in tandem with the Orchestrator, facilitating seamless data routing and peripheral management within a networked environment.

### Registering with the Orchestrator

Upon startup, the SLM Engine attempts to register itself with the orchestrator to establish communication channels.

#### Registration Process

1. **Generate UUID**: If not already present, the SLM Engine generates a unique UUID to identify itself.
2. **Send Registration Command**: The SLM Engine sends a `/register` command to the orchestrator's command port with its name, UUID, and listening port.
   ```bash
   /register SLM_Engine 123e4567-e89b-12d3-a456-426614174000 6201
   ```
3. **Receive Acknowledgment**: The orchestrator responds with an acknowledgment (`/ack`) containing the data port to be used for sending responses.
   ```
   /ack 6001
   ```
4. **Update Configuration**: The SLM Engine updates its configuration with the received data port for future communications.

**Notes**:
- Registration retries are attempted with exponential backoff in case of failures.
- Successful registration is indicated by receiving a valid `/ack` message from the orchestrator.

### Sending Data to the Orchestrator

After successful registration, the SLM Engine can send processed data back to the orchestrator via the specified data port.

#### Data Transmission Process

1. **Establish Connection**: Connect to the orchestrator's data port as specified during registration.
2. **Send Script UUID**: The first line sent is the SLM Engine's UUID to identify the source.
   ```
   123e4567-e89b-12d3-a456-426614174000
   ```
3. **Send Response Data**: Subsequent lines contain the AI model's response data in JSON format.
   ```json
   {
     "response": "The temperature is 25°C."
   }
   ```

**Example Transmission**:
```
123e4567-e89b-12d3-a456-426614174000
{
  "response": "The temperature is 25°C."
}
```

**Notes**:
- Ensure that the orchestrator's data port is correctly configured and accessible.
- Responses are sent as JSON to maintain consistency and ease of parsing.

---

## Inference Process

The core functionality of the SLM Engine revolves around processing incoming data using the Ollama API to generate AI-driven responses. This section outlines how inference is performed and managed.

### Using the Ollama API

The SLM Engine utilizes the Ollama API to perform inferences based on user input or data received from peripherals.

#### Inference Workflow

1. **Receive Data**: The SLM Engine receives data either through socket connections or commands.
2. **Accumulate Input**: Data is accumulated until a timeout is reached, indicating that the user has finished inputting data.
3. **Build Prompt**: Constructs a prompt for the language model by combining the system prompt with the user input.
   ```plaintext
   You Respond Conversationally

   User Input: Temperature:25°C
   Output:
   ```
4. **Send Request to Ollama API**: Sends a POST request to the Ollama API endpoint with the constructed prompt and model parameters.
   ```json
   {
     "model": "llama3.2:1b",
     "prompt": "You Respond Conversationally\n\nUser Input: Temperature:25°C\nOutput:",
     "temperature": 0.7,
     "top_p": 0.9,
     "repeat_penalty": 1.0,
     "max_tokens": 150,
     "format": "json",
     "stream": false
   }
   ```
5. **Handle Response**: Receives the AI model's response, validates it, and formats it as JSON.
6. **Send Response to Orchestrator**: Transmits the formatted response back to the orchestrator via the data port.
7. **Logging**: Logs all activities, including requests sent, responses received, and any errors encountered.

#### Model Parameters

| Parameter         | Description                                               | Default Value |
|-------------------|-----------------------------------------------------------|---------------|
| `temperature`     | Controls randomness of the model's responses. Higher values lead to more random outputs. | `0.7`         |
| `top_p`           | Controls diversity via nucleus sampling.                | `0.9`         |
| `max_tokens`      | Maximum number of tokens in the response.                 | `150`         |
| `repeat_penalty`  | Penalizes repetitive text in the response.               | `1.0`         |
| `inference_timeout` | Timeout in seconds for inference operations.            | `5`           |

**Notes**:
- Adjust model parameters in `slm.cf` or via command-line arguments to fine-tune response behaviors.
- Ensure that the Ollama API is running and accessible to handle inference requests.

---

## Examples

### Registering a Peripheral

**Scenario**: Register the SLM Engine with the orchestrator to enable data routing.

**Command Sent to Orchestrator**:
```bash
/register SLM_Engine 123e4567-e89b-12d3-a456-426614174000 6201
```

**Explanation**:
- **Name**: `SLM_Engine` - Identifies the peripheral.
- **UUID**: `123e4567-e89b-12d3-a456-426614174000` - Unique identifier for the SLM Engine.
- **Port**: `6201` - Port on which the SLM Engine is listening for connections.

**Orchestrator Response**:
```
/ack 6001
```

**Result**:
- The SLM Engine is now registered with the orchestrator.
- The data port for sending responses is set to `6001`.

### Adding a Route

**Scenario**: Define a route that forwards data from the SLM Engine to another peripheral (e.g., an Actuator).

**Command Sent to Orchestrator**:
```bash
/routes add Route1 SLM_Engine Actuator1
```

**Explanation**:
- **Route Name**: `Route1` - Identifier for the route.
- **Incoming Peripheral**: `SLM_Engine` - The source of data.
- **Outgoing Peripheral**: `Actuator1` - The destination for data.

**Result**:
- A new route named `Route1` is created.
- Data received by the SLM Engine is forwarded to `Actuator1` as per the route configuration.

### Sending Data Through a Route

**Scenario**: Send temperature data from the SLM Engine to the orchestrator for routing to `Actuator1`.

**Data Sent to SLM Engine's Listening Port (`6201`)**:
```
Temperature:25°C
```

**SLM Engine's Processing**:
1. **Accumulation**: Receives `Temperature:25°C` and waits for the `inference_timeout` duration.
2. **Inference**: Constructs the prompt and sends it to the Ollama API.
   ```plaintext
   You Respond Conversationally

   User Input: Temperature:25°C
   Output:
   ```
3. **Receive Response**: Receives the AI-generated response, e.g.,
   ```json
   {
     "response": "The temperature is 25°C."
   }
   ```
4. **Send to Orchestrator**:
   ```
   123e4567-e89b-12d3-a456-426614174000
   {
     "response": "The temperature is 25°C."
   }
   ```

**Orchestrator's Handling**:
- Recognizes the UUID and routes the response to `Actuator1` as defined in `Route1`.

**Result**:
- `Actuator1` receives the temperature data for further processing or action.

---

## Troubleshooting

### Common Issues

1. **Unable to Bind to Specified Port Range**
   - **Symptom**: SLM Engine fails to start or displays errors related to port binding.
   - **Solution**:
     - Ensure that the ports in the specified range (`port_range`) are not in use by other applications.
     - Modify the `port_range` in `slm.cf` or use the `--port-range` argument to specify a different range.
     - Example:
       ```bash
       python3 slm.py --port-range 6300-6400
       ```

2. **Orchestrator Registration Failure**
   - **Symptom**: SLM Engine cannot register with the orchestrator and displays retry messages.
   - **Solution**:
     - Verify that the orchestrator is running and accessible at the specified `orchestrator_host` and `orchestrator_port`.
     - Check network connectivity and firewall settings that might block the connection.
     - Ensure that the orchestrator is configured to accept registrations.

3. **Ollama API Unavailable**
   - **Symptom**: SLM Engine cannot perform inferences and logs errors related to the Ollama API.
   - **Solution**:
     - Ensure that the Ollama API service is installed and running.
     - Verify the API endpoint (`http://localhost:11434/api/generate`) is accessible.
     - Check for any network issues or firewall rules that might block access to the API.

4. **Invalid Model Responses**
   - **Symptom**: Received responses from the Ollama API are malformed or not in JSON format.
   - **Solution**:
     - Ensure that the `format` parameter in the inference request is set to `json`.
     - Verify the integrity and configuration of the language model being used.
     - Check for updates or patches for the Ollama API that might affect response formats.

5. **Unexpected Shutdowns or Crashes**
   - **Symptom**: SLM Engine terminates unexpectedly without clear error messages.
   - **Solution**:
     - Review the console logs for any error messages or stack traces.
     - Ensure that all dependencies are correctly installed and up to date.
     - Validate the configuration file (`slm.cf`) for any syntax errors or invalid parameters.
     - Run the script with elevated permissions if port binding issues persist.

### Viewing Logs

- **Activity Log**: Accessible via the console output where the SLM Engine is running. It logs all significant events, including registrations, data processing, inferences, and errors.
- **Error Messages**: Detailed error messages and stack traces are printed to the console to aid in debugging.

**Note**: Redirect console output to a log file for persistent logging.
```bash
python3 slm.py > slm_engine.log 2>&1
```

---

## FAQ

### 1. **How do I register the SLM Engine with the Orchestrator?**

**Answer**:
- The SLM Engine automatically attempts to register with the orchestrator upon startup.
- Ensure that the orchestrator is running and accessible at the specified `orchestrator_host` and `orchestrator_port`.
- If registration fails, check network connectivity, port configurations, and orchestrator status.

### 2. **Can I change the language model used by the SLM Engine?**

**Answer**:
- Yes. Modify the `model_name` parameter in the `slm.cf` file or use the `--model-name` command-line argument.
- Ensure that the specified model is available and compatible with the Ollama API.
  
  **Example**:
  ```bash
  python3 slm.py --model-name llama3.2:2b
  ```

### 3. **What happens if the Ollama API is not running when I start the SLM Engine?**

**Answer**:
- The SLM Engine will notify you that the Ollama API is unavailable.
- It will continue running and periodically check the status of the Ollama API.
- Once the Ollama API becomes available, the SLM Engine can resume normal operations.

### 4. **How can I view or modify the routes managed by the SLM Engine?**

**Answer**:
- Routes are managed via commands sent to the orchestrator.
- Use the `/routes` commands to add, remove, or list routes.
  
  **Example**:
  ```bash
  /routes add Route1 SLM_Engine Actuator1
  /routes info
  /routes remove Route1
  ```

### 5. **Is the SLM Engine secure?**

**Answer**:
- **Default Security**: The SLM Engine does not implement authentication or encryption by default.
- **Recommendations**:
  - Run the SLM Engine within a secure, trusted network environment.
  - Implement firewall rules to restrict access to the command and data ports.
  - Enhance the script to include authentication mechanisms if deploying in sensitive environments.

### 6. **Can I run multiple instances of the SLM Engine on the same machine?**

**Answer**:
- Yes, provided each instance is configured to use a unique port within the specified `port_range`.
- Ensure that there are no port conflicts by properly configuring the `port_range` and monitoring active ports.

### 7. **How do I update the configuration after the SLM Engine has started?**

**Answer**:
- Modify the `slm.cf` file with the desired configuration changes.
- Restart the SLM Engine to apply the new configurations.
  
  **Example**:
  ```bash
  python3 slm.py
  ```

### 8. **What is the purpose of the `system_prompt` parameter?**

**Answer**:
- The `system_prompt` defines the initial instructions or context provided to the language model.
- It guides the model's responses to ensure they align with desired behaviors or formats.
  
  **Example**:
  ```ini
  system_prompt=You are a helpful assistant.
  ```

### 9. **How does the `inference_timeout` parameter affect data processing?**

**Answer**:
- The `inference_timeout` defines the duration (in seconds) the SLM Engine waits for additional data before triggering an inference.
- If no new data is received within this timeout, the accumulated input is processed.
  
  **Example**:
  ```ini
  inference_timeout=5
  ```
