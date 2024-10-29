# Orchestrator Component Documentation

Welcome to the comprehensive documentation for the **Orchestrator** component of the **Orchestrator and Peripheral System**. This document provides an in-depth overview of the orchestrator script, detailing its functionalities, configurations, components, and usage instructions. Whether you're a developer looking to extend the system or an administrator aiming to manage peripherals effectively, this guide will serve as your primary resource.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [System Architecture](#system-architecture)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Usage](#usage)
7. [Components Overview](#components-overview)
    - [Global Variables](#global-variables)
    - [Configuration Files](#configuration-files)
8. [Core Functionalities](#core-functionalities)
    - [Configuration Management](#configuration-management)
    - [Port Scanning and Peripheral Discovery](#port-scanning-and-peripheral-discovery)
    - [Command Handling](#command-handling)
    - [Data Handling and Routing](#data-handling-and-routing)
    - [Curses-Based User Interface](#curses-based-user-interface)
    - [Logging and Activity Tracking](#logging-and-activity-tracking)
9. [Command Interface](#command-interface)
    - [Available Commands](#available-commands)
    - [Routes Management](#routes-management)
10. [Concurrency and Thread Safety](#concurrency-and-thread-safety)
11. [Extending the Orchestrator](#extending-the-orchestrator)
12. [Error Handling and Troubleshooting](#error-handling-and-troubleshooting)
13. [Frequently Asked Questions (FAQ)](#frequently-asked-questions-faq)
14. [License](#license)
15. [Contributing](#contributing)

---

## Introduction

The **Orchestrator** serves as the central hub in the Orchestrator and Peripheral system, responsible for managing peripheral devices, handling data routing between peripherals, and providing a user interface for monitoring and control. It leverages multi-threading to handle concurrent operations and employs a curses-based interface for real-time system monitoring.

---

## Features

- **Peripheral Discovery:** Automatically scans specified port ranges to discover and register peripherals.
- **Dynamic Configuration:** Reads and writes configurations to JSON files, ensuring persistent and up-to-date settings.
- **Command Interface:** Listens on designated ports for external commands, facilitating remote management.
- **Data Routing:** Manages routes between peripherals, enabling data forwarding based on predefined routes.
- **User Interface:** Provides a real-time, curses-based interface for monitoring system status, peripherals, routes, and recent activities.
- **Thread Safety:** Utilizes threading locks to ensure safe access to shared resources across multiple threads.
- **Activity Logging:** Maintains a log of system activities for auditing and troubleshooting purposes.
- **Extensibility:** Designed to be easily extendable for additional functionalities or integration with other systems.

---

## System Architecture

The Orchestrator operates through multiple threads, each handling distinct responsibilities:

1. **Periodic Port Scanning Thread:** Continuously scans a range of ports to discover peripherals.
2. **Command Listener Threads:** Listens on multiple ports for incoming commands from peripherals or external sources.
3. **Data Listener Threads:** Monitors designated data ports for incoming data from peripherals.
4. **Curses Interface Thread:** Manages the user interface, displaying real-time information and handling user interactions.

Inter-thread communication is facilitated through thread-safe queues and events, ensuring smooth operation without race conditions.

---

## Installation

### Prerequisites

- **Python 3.6+**
- **Unix-like Operating System** (for curses support)
- **Required Python Libraries:**
  - `curses` (usually included in standard Python distributions)
  - `socket`
  - `json`
  - `uuid`
  - `threading`
  - `queue`
  - `os`
  - `sys`
  - `time`
  - `traceback`

### Steps

1. **Run the Orchestrator:**

```
 curl -O https://raw.githubusercontent.com/robit-man/EGG/main/install_and_run_orchestrator.sh && chmod +x install_and_run_orchestrator.sh && ./install_and_run_orchestrator.sh
```

---

## Configuration

The orchestrator relies on two primary configuration files:

1. **`orch.cf`**: Main configuration file containing settings related to port scanning, command ports, peripheral details, and the script UUID.
2. **`routes.cf`**: Configuration file managing the routes between peripherals.

### Default Configuration (`orch.cf`)

If `orch.cf` does not exist, it is automatically created with the following default settings:

```json
{
    "known_ports": "2000-8000",
    "scan_interval": 5,
    "command_port": 6000,
    "data_port_range": "6001-6099",
    "peripherals": [],
    "script_uuid": "generated-uuid"
}
```

### Configuration Parameters

- **`known_ports`**: Defines the range of ports (`start-end`) the orchestrator will scan to discover peripherals. Default is `"2000-8000"`.
  
- **`scan_interval`**: Time interval in seconds between each port scan. Default is `5`.

- **`command_port`**: Primary port to listen for commands and data. Default is `6000`.

- **`data_port_range`**: Range of ports (`start-end`) designated for receiving data from peripherals. Default is `"6001-6099"`.

- **`peripherals`**: List of known peripherals. Initially empty, peripherals are dynamically added upon discovery.

- **`script_uuid`**: Unique UUID string identifying the orchestrator instance. Automatically generated if not present.

### Routes Configuration (`routes.cf`)

Manages the data routing between peripherals. Each route defines:

- **`name`**: Unique identifier for the route.
- **`incoming`**: UUID of the incoming peripheral.
- **`outgoing`**: UUID of the outgoing peripheral.
- **`incoming_port`**: Port number of the incoming peripheral.
- **`outgoing_port`**: Port number of the outgoing peripheral.
- **`last_used`**: Timestamp of the last data transfer through the route.

Example entry:

```json
{
    "name": "route1",
    "incoming": "uuid-incoming",
    "outgoing": "uuid-outgoing",
    "incoming_port": 6001,
    "outgoing_port": 6002,
    "last_used": null
}
```

### Modifying Configuration

To modify configurations, you can directly edit the `orch.cf` and `routes.cf` files or use the command interface provided by the orchestrator.

**Note:** It's recommended to use the orchestrator's command interface for changes to ensure consistency and thread safety.

---

## Usage

### Starting the Orchestrator

Run the orchestrator script using Python:

```bash
python orch.py
```

Upon startup, the orchestrator performs the following actions:

1. **Configuration Loading:** Reads `orch.cf` and initializes settings. If the configuration file is missing or invalid, it resets to default settings.

2. **Routes Loading:** Reads `routes.cf` to initialize existing data routes. If the file is missing or invalid, it creates a new one.

3. **Thread Initialization:** Starts threads for port scanning, command listening, and data listening.

4. **User Interface Launch:** Initializes the curses-based user interface for real-time monitoring and interaction.

### Interacting with the User Interface

- **Overview Screen:**
  - Displays session UUID, list of peripherals, configured routes, and recent activity logs.
  - **Peripherals Section:** Shows each peripheral's name, port, and last seen timestamp.
  - **Routes Section:** Lists each route with source and destination peripherals and last used timestamp.
  - **Recent Activity:** Logs recent system activities and events.

- **Accessing the Menu:**
  - Press `'m'` or `'M'` to open the main menu.

- **Main Menu Options:**
  - **List Peripherals:** Displays all discovered peripherals.
  - **Add Route:** Initiates the process to add a new data route between peripherals.
  - **Edit Route:** Allows modification of existing routes.
  - **Remove Route:** Enables deletion of specified routes.
  - **Remove Peripheral:** Facilitates the removal of a peripheral from the system.
  - **Reset Orchestrator:** Resets the orchestrator by deleting configurations and restarting.
  - **Exit Menu:** Closes the menu and returns to the overview screen.

### Exiting the Orchestrator

- Press `Ctrl+C` in the terminal to terminate the orchestrator gracefully. The curses interface will clean up before exiting.

---

## Components Overview

### Global Variables

The orchestrator script utilizes several global variables to manage state and configurations:

- **Configuration Variables:**
  - `CONFIG_FILE`: Filename for the main configuration (`orch.cf`).
  - `ROUTES_FILE`: Filename for routes configuration (`routes.cf`).
  - `default_config`: Dictionary containing default settings.
  - `config`: Dictionary holding the current configuration.

- **Orchestrator Identification:**
  - `orchestrator_uuid`: Unique UUID string for the orchestrator instance.
  - `orchestrator_name`: Name of the orchestrator, defaulting to `'Orchestrator'`.

- **Routing and Command Handling:**
  - `routes`: List of current data routes.
  - `command_queue`: Queue for processing incoming commands.
  - `external_commands`: List of commands received from external sources.

- **Threading and Synchronization:**
  - `peripherals_lock`: Lock for thread-safe access to peripherals.
  - `routes_lock`: Lock for thread-safe access to routes.
  - `config_lock`: Lock for thread-safe access to configuration.
  - `update_event`: Event to trigger display updates.
  - `in_command_mode`: Event indicating if the system is in command mode.

- **Logging and Interface:**
  - `activity_log`: List maintaining a history of system activities.
  - `peripheral_colors`: Dictionary mapping peripheral names to color pairs for UI.
  - `stdscr`: Curses standard screen object.

### Configuration Files

- **`orch.cf`**: Main configuration file storing system settings and peripheral information. Managed through JSON format.
  
- **`routes.cf`**: Routes configuration file detailing data pathways between peripherals. Also managed via JSON.

---

## Core Functionalities

### Configuration Management

**Functions:**

- `read_config()`: Reads and validates the main configuration from `orch.cf`. If the file is missing or corrupted, it initializes with default settings and writes them to the file.

- `write_config()`: Writes the current configuration state to `orch.cf` in a thread-safe manner.

- `read_routes()`: Reads and validates the routes configuration from `routes.cf`. Initializes an empty list if the file is missing or invalid.

- `write_routes()`: Writes the current routes list to `routes.cf` safely.

- `parse_port_range(port_range_str)`: Parses a string representing port ranges (e.g., `"6001-6099"`) into a list of individual port numbers.

**Behavior:**

- Ensures that configurations are always in a valid state, resetting to defaults if inconsistencies are detected.
  
- Manages unique identification of the orchestrator through UUIDs, persisting them across sessions.

### Port Scanning and Peripheral Discovery

**Functions:**

- `scan_ports()`: Iterates through the specified `known_ports` to identify active peripherals by attempting connections and sending `/info` commands.

- `check_port(port)`: Attempts to connect to a given port, sends an `/info` command, and processes the response to identify peripherals.

- `process_response(response, port)`: Parses the response from a peripheral, validating its structure and UUID, and updates or registers the peripheral accordingly.

- `assign_colors_to_peripherals()`: Assigns unique colors to peripherals based on their names for UI representation.

- `periodic_scan()`: Continuously performs port scanning at intervals defined by `scan_interval`.

**Behavior:**

- Discovers peripherals by connecting to open ports and retrieving their information.
  
- Ensures that each peripheral is uniquely identified and managed within the system.

### Command Handling

**Functions:**

- `command_listener()`: Starts listeners on designated command ports to accept incoming connections for command processing.

- `listen_on_port(port)`: Binds to a specific port and listens for incoming command connections.

- `handle_client_connection(conn, addr)`: Manages individual client connections, parsing and processing incoming commands.

- `process_command(command, source, conn=None)`: Interprets and executes commands received from various sources, such as ports or the console.

- `send_response(message, source, conn)`: Sends responses back to the source of the command, whether it's the console, a port, or the curses interface.

**Behavior:**

- Supports a variety of commands for managing peripherals, routes, and system configurations.
  
- Ensures that commands are processed efficiently and responses are appropriately dispatched.

### Data Handling and Routing

**Functions:**

- `data_listener()`: Initializes listeners on data ports to receive incoming data from peripherals.

- `listen_on_port(port)`: Similar to command listener, but specifically for data connections.

- `handle_data_connection(conn, addr)`: Manages data connections, receiving and processing incoming data.

- `handle_incoming_data(peripheral_uuid, data)`: Forwards data from incoming peripherals to outgoing peripherals based on defined routes.

- `add_route(route_name, incoming_name, outgoing_name)`: Adds a new data route between specified peripherals.

- `remove_route(route_name)`: Removes an existing data route by name.

- `list_routes()`: Lists all configured data routes.

**Behavior:**

- Facilitates seamless data transfer between peripherals based on predefined routing rules.
  
- Maintains the integrity and reliability of data transmission across the system.

### Curses-Based User Interface

**Functions:**

- `run_curses_interface()`: Initializes and manages the curses-based UI.

- `main_overview()`: Renders the main overview screen, displaying peripherals, routes, and activity logs.

- `main_menu()`: Provides a navigable menu for various management tasks.

- `list_peripherals_menu()`: Displays a list of all known peripherals.

- `add_route_menu()`: Guides the user through adding a new data route.

- `edit_route_menu()`: Allows editing existing data routes.

- `remove_route_menu()`: Facilitates the removal of a data route.

- `remove_peripheral_menu()`: Enables the removal of a peripheral from the system.

- `reset_orchestrator_menu()`: Provides options to reset the orchestrator, deleting configurations and restarting.

- `prompt_user(prompt_text)`: Prompts the user for input within the curses interface.

- `select_item(prompt_text, options)`: Allows the user to select an item from a list.

- `select_peripheral(prompt_text, allow_skip=False)`: Facilitates the selection of a peripheral from the list.

- `display_message(title, message)`: Displays informational messages to the user.

**Behavior:**

- Offers an interactive and real-time interface for monitoring system status and performing management tasks.
  
- Enhances user experience by providing intuitive navigation and feedback within the terminal.

### Logging and Activity Tracking

**Functions:**

- `log_message(message)`: Logs messages to the activity log and displays them in the UI or console.

- **Activity Log:** Maintains a history of system events, errors, and informational messages.

**Behavior:**

- Ensures that all significant actions and events are recorded for auditing and troubleshooting.
  
- Provides visibility into the system's operations through the UI and logs.

---

## Command Interface

The orchestrator listens on designated ports for incoming commands, allowing external sources or peripherals to manage system configurations and operations.

### Available Commands

- **`/help`**: Displays a help message listing all available commands.

- **`/list` or `/available`**: Lists all known peripherals.

- **`/routes`**: Manages data routes with subcommands:
  - **`/routes help`**: Displays help for routes commands.
  - **`/routes add <route-name> <incoming-peripheral-name> <outgoing-peripheral-name>`**: Adds a new data route.
  - **`/routes remove <route-name>`**: Removes an existing data route.
  - **`/routes info`**: Lists all configured routes.

- **`/register <name> <peripheral_uuid> <port>`**: Registers a new peripheral with the specified name, UUID, and port.

- **`/data <peripheral_uuid> <data>`**: Sends data from a peripheral identified by its UUID.

- **`/reset`**: Resets the orchestrator by deleting configuration files and restarting.

- **`/exit`**: Exits command mode or terminates the orchestrator.

### Routes Management

**Subcommands:**

- **Add Route:**

  ```bash
  /routes add <route-name> <incoming-peripheral-name> <outgoing-peripheral-name>
  ```

  - **`<route-name>`**: Unique identifier for the route.
  - **`<incoming-peripheral-name>`**: Name of the peripheral sending data.
  - **`<outgoing-peripheral-name>`**: Name of the peripheral receiving data.

- **Remove Route:**

  ```bash
  /routes remove <route-name>
  ```

  - **`<route-name>`**: Name of the route to be removed.

- **List Routes:**

  ```bash
  /routes info
  ```

  - Displays all configured routes with details.

- **Help:**

  ```bash
  /routes help
  ```

  - Shows usage information for routes commands.

### Command Processing

Commands can originate from different sources:

- **Port (`source='port'`)**: External commands received via network ports.
- **Console (`source='console'`)**: Commands entered directly via the console interface.
- **Curses Interface (`source='curses'`)**: Commands entered through the curses-based UI.

**Response Handling:**

- Responses are sent back to the source of the command.
- Informational messages, acknowledgments, or error messages are dispatched appropriately.

---

## Concurrency and Thread Safety

The orchestrator employs multi-threading to handle various tasks concurrently. To ensure thread safety, the following mechanisms are in place:

- **Threading Locks:**
  - **`peripherals_lock`**: Guards access to the peripherals list.
  - **`routes_lock`**: Guards access to the routes list.
  - **`config_lock`**: Guards access to the main configuration.

- **Thread-Safe Queues:**
  - **`command_queue`**: Manages incoming commands from different sources.

- **Events:**
  - **`update_event`**: Signals the UI to refresh when there are updates.
  - **`in_command_mode`**: Indicates whether the system is currently processing commands.

**Best Practices:**

- Always acquire the appropriate lock before accessing or modifying shared resources.
- Use daemon threads to ensure they terminate gracefully when the main program exits.
- Handle exceptions within threads to prevent unexpected terminations.

---

## Extending the Orchestrator

The orchestrator is designed with extensibility in mind. Developers can add new functionalities or integrate with other systems by following these guidelines:

1. **Adding New Commands:**
   - Define the command syntax and processing logic within the `process_command` function.
   - Implement corresponding functions to handle the new commands.
   - Update the help texts to include the new commands.

2. **Integrating Additional Data Sources:**
   - Modify the `scan_ports` and `check_port` functions to accommodate new peripheral types or communication protocols.
   - Ensure that new peripherals are registered and managed using existing locking mechanisms.

3. **Enhancing the User Interface:**
   - Customize the curses-based UI by adding new screens or widgets.
   - Implement additional menu options or real-time data displays as needed.

4. **Improving Data Routing Logic:**
   - Extend the `handle_incoming_data` function to support more complex routing rules.
   - Incorporate filtering, transformation, or conditional routing based on data content.

5. **Implementing Security Measures:**
   - Add authentication mechanisms for incoming command connections.
   - Encrypt data transmissions between the orchestrator and peripherals.

6. **Logging Enhancements:**
   - Integrate with external logging systems or databases for persistent activity tracking.
   - Implement log rotation or archival strategies to manage log sizes.

**Example: Adding a New Command**

To add a `/status` command that returns the current status of the orchestrator:

1. **Update `process_command`:**

   ```python
   elif command == '/status':
       status = get_system_status()
       send_response(status, source, conn)
   ```

2. **Implement `get_system_status`:**

   ```python
   def get_system_status():
       status = {
           "UUID": orchestrator_uuid,
           "Peripherals": len(config['peripherals']),
           "Routes": len(routes),
           "Active Threads": threading.active_count(),
           "Last Scan": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(last_scan_time)),
       }
       return json.dumps(status, indent=4)
   ```

3. **Update Help Text:**

   ```python
   def get_help_text():
       return (
           "Available commands:\n"
           "/help - Show this help message\n"
           "/list or /available - List known peripherals\n"
           "/routes - Manage routes\n"
           "/status - Show system status\n"
           "/reset - Reset the orchestrator by deleting config files and restarting\n"
           "/exit - Exit command mode or exit the orchestrator\n"
       )
   ```

---

## Error Handling and Troubleshooting

The orchestrator includes robust error handling mechanisms to ensure stability and provide informative feedback in case of issues.

### Common Error Scenarios

1. **Invalid Configuration File (`orch.cf`):**
   - **Symptom:** Orchestrator resets to default settings.
   - **Solution:** Verify the JSON structure of `orch.cf`. Ensure all required fields are present and correctly formatted.

2. **Port Binding Failures:**
   - **Symptom:** Error messages indicating inability to bind to specified ports.
   - **Solution:** Check if the ports are already in use by other applications. Modify the `known_ports` or `data_port_range` in `orch.cf` to use available ports.

3. **Peripheral Connection Issues:**
   - **Symptom:** Errors when connecting to peripherals or receiving incomplete responses.
   - **Solution:** Ensure peripherals are online and responsive. Verify network connectivity and port configurations.

4. **Curses Interface Glitches:**
   - **Symptom:** Display anomalies or crashes within the UI.
   - **Solution:** Ensure the terminal supports curses. Resize the terminal window to accommodate the UI layout. Check for unhandled exceptions in the UI code.

5. **Command Processing Errors:**
   - **Symptom:** Unknown commands or failures to execute valid commands.
   - **Solution:** Use `/help` to verify command syntax. Ensure commands are correctly formatted and parameters are valid.

### Logging and Diagnostics

- **Activity Log:** Review the `activity_log` within the UI or check console outputs for real-time logs.
  
- **Exception Tracebacks:** Errors within threads are logged with tracebacks for easier debugging.

### Resetting the Orchestrator

If the orchestrator becomes unresponsive or behaves unexpectedly, you can reset it using the `/reset` command or through the main menu in the UI. This action deletes configuration files and restarts the orchestrator, returning it to a clean state.

**Caution:** Resetting will remove all peripheral registrations and data routes. Ensure you have backups or can reconfigure the system as needed.

---

## Frequently Asked Questions (FAQ)

**Q1: How do I add a new peripheral to the orchestrator?**

**A1:** Peripherals are automatically discovered by scanning the specified port ranges. Ensure your peripheral is configured to listen on a port within the `known_ports` range and responds to `/info` commands. Alternatively, use the `/register` command to manually add a peripheral.

---

**Q2: What happens if two peripherals have the same name?**

**A2:** The orchestrator appends a numeric suffix to the peripheral name to ensure uniqueness (e.g., `Peripheral`, `Peripheral_1`, `Peripheral_2`, etc.).

---

**Q3: Can I change the port ranges after the orchestrator has started?**

**A3:** Yes. Modify the `known_ports` or `data_port_range` in `orch.cf` and restart the orchestrator for changes to take effect.

---

**Q4: How can I view detailed logs outside the UI?**

**A4:** Currently, logs are maintained in the `activity_log` list and displayed in the UI or printed to the console if the UI is not initialized. For persistent logging, consider integrating with external logging frameworks or exporting the logs from the UI.

---

**Q5: Is it possible to run multiple orchestrator instances on the same machine?**

**A5:** Yes, but ensure each instance uses unique `script_uuid` values and does not have overlapping port ranges to prevent conflicts.

---

## License

*Specify the license under which the orchestrator is distributed. For example:*

This project is licensed under the [MIT License](LICENSE).

---

## Contributing

We welcome contributions to enhance the orchestrator's capabilities and improve its functionalities. To contribute:

1. **Fork the Repository:**

   Click the "Fork" button on the repository page to create your own copy.

2. **Create a Feature Branch:**

   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Commit Your Changes:**

   ```bash
   git commit -m "Add feature X"
   ```

4. **Push to Your Fork:**

   ```bash
   git push origin feature/your-feature-name
   ```

5. **Submit a Pull Request:**

   Navigate to the original repository and submit a pull request detailing your changes.

**Guidelines:**

- Ensure your code adheres to the existing coding standards and conventions.
- Provide clear and concise commit messages.
- Include documentation and tests for new features or changes.

---

Thank you for using the Orchestrator component of the Orchestrator and Peripheral System. For further assistance or inquiries, please contact the development team or refer to the project's issue tracker.
