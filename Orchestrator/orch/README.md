# Orchestrator Script API Documentation

## Table of Contents

1. [Introduction](#introduction)
2. [Installation and Setup](#installation-and-setup)
3. [Configuration](#configuration)
   - [Configuration File (`orch.cf`)](#configuration-file-orchcf)
   - [Routes File (`routes.cf`)](#routes-file-routescf)
4. [Running the Orchestrator](#running-the-orchestrator)
5. [Command Interfaces](#command-interfaces)
   - [Command Port (`6000`)](#command-port-6000)
   - [Data Port (`6001`)](#data-port-6001)
6. [Supported Commands](#supported-commands)
   - [General Commands](#general-commands)
   - [Peripheral Commands](#peripheral-commands)
   - [Route Commands](#route-commands)
7. [Curses-Based User Interface](#curses-based-user-interface)
   - [Overview](#overview)
   - [Navigating the Interface](#navigating-the-interface)
   - [Command Mode](#command-mode)
8. [API Endpoints](#api-endpoints)
   - [Registering a Peripheral](#registering-a-peripheral)
   - [Sending Data](#sending-data)
9. [Examples](#examples)
   - [Registering a Peripheral](#registering-a-peripheral-example)
   - [Adding a Route](#adding-a-route-example)
   - [Sending Data Through a Route](#sending-data-through-a-route-example)
10. [Troubleshooting](#troubleshooting)
11. [FAQ](#faq)
12. [Contact and Support](#contact-and-support)

---

## Introduction

The **Orchestrator** is a Python-based server application designed to manage and route data between various peripherals (clients) connected to it. It utilizes socket programming to listen for commands and data, manages configurations and routes, and provides a dynamic, terminal-based user interface using the `curses` library for real-time monitoring and control.

This documentation provides a comprehensive guide on how to install, configure, and effectively use the Orchestrator, including its command interfaces and user interface.

---

## Installation and Setup

### Prerequisites

- **Python 3.6 or higher**: Ensure that Python is installed on your system. You can verify this by running:
  ```bash
  python3 --version
  ```
- **Required Python Libraries**:
  - `curses`: Typically included with Python on Unix-based systems.
  - `asyncio`, `threading`, `socket`, `json`, `uuid`, `os`, `re`, `queue`, etc.: These are standard Python libraries.

### Installation Steps

1. **Clone the Repository** (if applicable):
   ```bash
   git clone https://github.com/your-repo/orchestrator-prototype](https://github.com/robit-man/EGG.git
   cd orchestrator-prototype/orch
   ```

2. **Ensure Dependencies Are Installed**:
   - For most standard Python libraries used in the script, no additional installation is necessary.
   - If using a virtual environment, activate it:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Run the Orchestrator Script**:
   ```bash
   python3 orch.py
   ```

   - On startup, the orchestrator will read the configuration files (`orch.cf` and `routes.cf`). If these files do not exist, they will be created with default settings.

---

## Configuration

### Configuration File (`orch.cf`)

The orchestrator uses a configuration file named `orch.cf` to store its settings. This file is automatically created in the same directory as the script if it does not exist.

#### Default Configuration

```ini
known_ports=2000-8000
scan_interval=5
command_port=6000
data_port=6001
peripherals=[]
```

#### Configuration Parameters

| Parameter       | Description                                                    | Default Value  |
|-----------------|----------------------------------------------------------------|-----------------|
| `known_ports`   | Ports range to scan for peripherals. Format: `start-end`.      | `2000-8000`     |
| `scan_interval` | Time interval (in seconds) between each port scan.             | `5`             |
| `command_port`  | Port to listen for incoming commands and data.                | `6000`          |
| `data_port`     | Port to receive data from peripherals.                        | `6001`          |
| `peripherals`   | JSON-formatted list of known peripherals.                      | `[]`            |

#### Modifying Configuration

You can manually edit the `orch.cf` file to change configurations. Alternatively, use the orchestrator's command interfaces to manage peripherals and routes, which will automatically update this file.

### Routes File (`routes.cf`)

Routes define how data is forwarded from one peripheral to another. The `routes.cf` file stores these configurations in JSON format.

#### Structure

```json
[
    {
        "name": "Route1",
        "incoming": "peripheral_uuid_1",
        "outgoing": "peripheral_uuid_2",
        "incoming_port": 2001,
        "outgoing_port": 2002,
        "last_used": null
    },
    ...
]
```

#### Modifying Routes

Use the `/routes` command via the command interfaces to add, remove, or list routes. Changes are automatically saved to `routes.cf`.

---

## Running the Orchestrator

To start the orchestrator, execute the script:

```bash
python3 orch.py
```

Upon running, the orchestrator will:

1. **Load Configurations**: Read from `orch.cf` and `routes.cf`.
2. **Scan Ports**: Continuously scan the specified range (`known_ports`) at intervals defined by `scan_interval` to discover peripherals.
3. **Listen on Ports**:
   - **Command Port (`6000`)**: For receiving commands and data.
   - **Data Port (`6001`)**: For receiving data from peripherals.
4. **Display UI**: Launch the curses-based user interface for real-time monitoring and management.

**Note**: Ensure that the designated command and data ports (`6000` and `6001` by default) are open and not used by other applications.

---

## Command Interfaces

The orchestrator provides two primary interfaces for interaction:

1. **Command Port (`6000`)**: Receives commands and data from external sources (peripherals).
2. **Data Port (`6001`)**: Receives data specifically from peripherals for routing.

Additionally, the curses-based user interface allows for interactive command input and real-time monitoring.

### Command Port (`6000`)

- **Purpose**: To receive commands from external peripherals or clients.
- **Connection Type**: TCP
- **Usage**:
  - **Clients**: Connect to this port to send commands.
  - **Commands**: Sent as plain text, terminated by a newline (`\n`).

### Data Port (`6001`)

- **Purpose**: To receive raw data from peripherals that need to be routed to other peripherals.
- **Connection Type**: TCP
- **Usage**:
  - **Clients**: Connect to this port to send data.
  - **Data Format**:
    - **First Line**: Peripheral UUID.
    - **Subsequent Lines**: Data payloads, each terminated by a newline (`\n`).

---

## Supported Commands

The orchestrator supports a variety of commands to manage peripherals and routes. Commands can be sent via the Command Port (`6000`) or entered through the curses-based user interface in Command Mode.

### General Commands

| Command | Description                                       | Usage Example           |
|---------|---------------------------------------------------|-------------------------|
| `/help` | Displays help information about available commands. | `/help`                 |
| `/list` or `/available` | Lists all known peripherals.                    | `/list` or `/available` |
| `/exit` | Exits command mode or terminates the orchestrator. | `/exit`                 |

### Peripheral Commands

| Command                | Description                                         | Usage Example                                 |
|------------------------|-----------------------------------------------------|-----------------------------------------------|
| `/register <name> <uuid> <port>` | Registers a new peripheral with the orchestrator. | `/register Sensor1 123e4567-e89b-12d3-a456-426614174000 2001` |
| `/data <uuid> <data>`  | Sends data from a peripheral to be routed.          | `/data 123e4567-e89b-12d3-a456-426614174000 Temperature:25°C` |

### Route Commands

| Command                                                                      | Description                                                       | Usage Example                                               |
|------------------------------------------------------------------------------|-------------------------------------------------------------------|-------------------------------------------------------------|
| `/routes help`                                                               | Displays help information about route commands.                  | `/routes help`                                             |
| `/routes add <route-name> <incoming-peripheral-name> <outgoing-peripheral-name>` | Adds a new route from one peripheral to another.                 | `/routes add Route1 Sensor1 Actuator1`                      |
| `/routes remove <route-name>`                                                | Removes an existing route by name.                                | `/routes remove Route1`                                      |
| `/routes info`                                                               | Lists all configured routes with details.                        | `/routes info`                                              |

---

## Curses-Based User Interface

The orchestrator utilizes the `curses` library to provide a dynamic, terminal-based user interface for real-time monitoring and interaction.

### Overview

Upon running the orchestrator, the curses interface displays:

- **Title and Session UUID**: At the top of the screen.
- **Peripherals**: A list of connected peripherals with details.
- **Routes**: Configured routes between peripherals.
- **Recent Activity Log**: A log of recent actions and events.
- **Commands Received**: A list of recent commands received from external sources.

### Navigating the Interface

- **Real-Time Updates**: The interface refreshes periodically to display the latest information.
- **Entering Command Mode**: Press any key to enter Command Mode.

### Command Mode

- **Activation**: Press any key while in the Overview mode.
- **Purpose**: Allows the user to input commands directly into the orchestrator.
- **Exiting Command Mode**: Type `/exit` to return to the Overview mode.

**Command Mode Layout**:

```
Command Mode (type '/exit' to return to overview):
> 
```

**Example Commands in Command Mode**:

- `/help`
- `/list`
- `/routes add Route1 Sensor1 Actuator1`
- `/routes remove Route1`

**Notes**:

- While in Command Mode, the Overview display is paused.
- Command inputs are processed immediately, and the display updates accordingly.

---

## API Endpoints

While the orchestrator primarily uses socket-based communication for commands and data, understanding the flow and structure of these interactions is crucial for effective use.

### Registering a Peripheral

**Endpoint**: Command Port (`6000`)

**Command**: `/register <name> <uuid> <port>`

**Description**: Registers a new peripheral with the orchestrator. This allows the peripheral to communicate and be part of the routing system.

**Parameters**:

- `<name>`: A unique name for the peripheral.
- `<uuid>`: A universally unique identifier (UUID) for the peripheral.
- `<port>`: The port number on which the peripheral is listening.

**Response**:

- **Success**: Sends an acknowledgment with the data port.
  ```
  /ack 6001
  ```
- **Failure**: Sends an error message detailing the issue.

**Example**:

```bash
/register Sensor1 123e4567-e89b-12d3-a456-426614174000 2001
```

### Sending Data

**Endpoint**: Data Port (`6001`)

**Data Format**:

1. **First Line**: Peripheral UUID.
2. **Subsequent Lines**: Data payloads, each terminated by a newline (`\n`).

**Description**: Sends data from a peripheral to the orchestrator for routing to other peripherals based on configured routes.

**Example**:

```
123e4567-e89b-12d3-a456-426614174000
Temperature:25°C
Humidity:40%
```

**Orchestrator Response**:

- **Success**: Forwards data to the designated outgoing peripheral(s) and logs the activity.
- **Failure**: Logs an error message if routing fails.

---

## Examples

### Registering a Peripheral Example

**Command**:

```bash
/register Sensor1 123e4567-e89b-12d3-a456-426614174000 2001
```

**Explanation**:

- **Name**: `Sensor1`
- **UUID**: `123e4567-e89b-12d3-a456-426614174000`
- **Port**: `2001`

**Response**:

```
/ack 6001
```

**Result**: The peripheral `Sensor1` is now registered and can send data to the orchestrator via port `2001`.

### Adding a Route Example

**Command**:

```bash
/routes add Route1 Sensor1 Actuator1
```

**Explanation**:

- **Route Name**: `Route1`
- **Incoming Peripheral**: `Sensor1`
- **Outgoing Peripheral**: `Actuator1`

**Result**:

- A new route named `Route1` is created, forwarding data from `Sensor1` to `Actuator1`.
- The route details are saved in `routes.cf`.

### Sending Data Through a Route Example

1. **Peripheral `Sensor1` Sends Data**:

   **Data Sent to Data Port (`6001`)**:

   ```
   123e4567-e89b-12d3-a456-426614174000
   Temperature:25°C
   ```

2. **Orchestrator Processes Data**:

   - Identifies that `Sensor1` is connected to `Route1`.
   - Forwards the data `Temperature:25°C` to `Actuator1` on its designated port.

3. **Result**:

   - `Actuator1` receives the temperature data.
   - The activity log is updated accordingly.

---

## Troubleshooting

### Common Issues

1. **Ports Already in Use**:
   - **Issue**: The orchestrator fails to bind to the specified command or data ports.
   - **Solution**:
     - Check if the ports (`6000` or `6001`) are already in use using:
       ```bash
       sudo lsof -iTCP -sTCP:LISTEN -P
       ```
     - Modify the `orch.cf` configuration to use different ports if necessary.

2. **Peripheral Not Receiving Data**:
   - **Issue**: Data sent to a peripheral does not arrive.
   - **Solution**:
     - Ensure that the peripheral is correctly registered and that routes are properly configured.
     - Verify that the peripheral is listening on the designated port.
     - Check firewall settings that might block communication.

3. **Curses Interface Not Displaying Correctly**:
   - **Issue**: Terminal window is too small or display is garbled.
   - **Solution**:
     - Resize the terminal window to a larger size.
     - Ensure that the terminal supports `curses` and has adequate color support.

4. **Unresponsive Orchestrator**:
   - **Issue**: The orchestrator hangs or becomes unresponsive.
   - **Solution**:
     - Check the activity log for any error messages.
     - Ensure that the system has sufficient resources (CPU, memory).
     - Restart the orchestrator and monitor for recurring issues.

### Viewing Logs

- **Activity Log**: Accessible via the curses interface under the "Recent Activity" section.
- **External Commands**: Viewable under the "Commands Received" section in the UI.

**Note**: Ensure that logging does not fill up the terminal buffer by regularly monitoring and managing the activity log.

---

## FAQ

### 1. **How do I add a new peripheral to the orchestrator?**

**Answer**:

- **Step 1**: Register the peripheral using the `/register` command via the Command Port (`6000`) or through the curses interface in Command Mode.
  
  **Example Command**:
  ```bash
  /register Sensor1 123e4567-e89b-12d3-a456-426614174000 2001
  ```
  
- **Step 2**: Verify that the peripheral appears in the peripherals list in the curses UI or by using the `/list` command.

### 2. **How do I create a route between two peripherals?**

**Answer**:

- Use the `/routes add` command specifying the route name, incoming peripheral name, and outgoing peripheral name.

  **Example Command**:
  ```bash
  /routes add Route1 Sensor1 Actuator1
  ```

- Verify the route by listing all routes using the `/routes info` command.

### 3. **Can I change the default command and data ports?**

**Answer**:

- Yes. Modify the `command_port` and `data_port` parameters in the `orch.cf` configuration file.

  **Example**:
  ```ini
  command_port=7000
  data_port=7001
  ```

- After modifying, restart the orchestrator to apply the changes.

### 4. **What happens if a peripheral disconnects?**

**Answer**:

- The orchestrator continuously scans for peripherals based on the `scan_interval`. If a peripheral is not detected during a scan, it updates the `last_seen` timestamp.
- Routes associated with disconnected peripherals will remain but data forwarding will fail until the peripheral reconnects.

### 5. **Is the orchestrator secure?**

**Answer**:

- **Default Security**: The orchestrator does not implement authentication or encryption by default.
- **Recommendations**:
  - Run the orchestrator within a secure, trusted network environment.
  - Implement firewall rules to restrict access to the command and data ports.
  - Enhance the script to include authentication mechanisms if deploying in sensitive environments.
