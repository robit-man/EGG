import threading
import time
import socket
import json
import uuid
import os
import re
import queue
import curses
import asyncio

CONFIG_FILE = 'orch.cf'
ROUTES_FILE = 'routes.cf'

# Default configuration
default_config = {
    'known_ports': '2000-8000',  # Scan ports from 2000 to 8000
    'scan_interval': '5',        # Time interval in seconds to scan for peripherals
    'command_port': '6000',      # Port to listen for commands and data
    'data_port': '6001',         # Port to receive data from peripherals
    'peripherals': '[]',         # List of known peripherals (will be stored as JSON)
}

config = {}

# Global variable to store peripherals
peripherals = []

# UUID for the orchestrator
orchestrator_uuid = str(uuid.uuid4())
orchestrator_name = 'Orchestrator'

# Global variable to store routes
routes = []

# Command queue for processing commands from port and console
command_queue = queue.Queue()

# Lock for peripherals list
peripherals_lock = threading.Lock()

# Lock for routes list
routes_lock = threading.Lock()

# Activity logs
activity_log = []

# Event to control the display update
update_event = threading.Event()

# Color mapping for peripherals
peripheral_colors = {}

# Commands received from external sources
external_commands = []

# Flag to indicate if we are in command mode
in_command_mode = threading.Event()

def read_config():
    global config
    config = default_config.copy()
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            for line in f:
                line = line.strip()
                if '=' in line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    if key in config:
                        config[key] = value
    else:
        write_config()
    # Load peripherals from config
    try:
        config['peripherals'] = json.loads(config['peripherals'])
    except json.JSONDecodeError:
        config['peripherals'] = []
    return config

def write_config():
    with peripherals_lock:
        with open(CONFIG_FILE, 'w') as f:
            for key, value in config.items():
                if key == 'peripherals':
                    # Convert peripherals list to JSON string
                    value = json.dumps(value)
                f.write(f"{key}={value}\n")

def read_routes():
    global routes
    if os.path.exists(ROUTES_FILE):
        with open(ROUTES_FILE, 'r') as f:
            try:
                routes = json.load(f)
            except json.JSONDecodeError:
                routes = []
    else:
        routes = []
        write_routes()

def write_routes():
    with routes_lock:
        with open(ROUTES_FILE, 'w') as f:
            json.dump(routes, f, indent=4)

def scan_ports():
    # For testing, we'll use synchronous scanning
    ports = []
    known_ports = config.get('known_ports', '2000-8000')
    for port_entry in known_ports.split(','):
        port_entry = port_entry.strip()
        if '-' in port_entry:
            start_port, end_port = map(int, port_entry.split('-'))
            ports.extend(range(start_port, end_port + 1))
        elif port_entry.isdigit():
            ports.append(int(port_entry))
    for port in ports:
        check_port(port)

def check_port(port):
    host = 'localhost'
    try:
        with socket.create_connection((host, port), timeout=5) as sock:
            sock.sendall(b'/info\n')
            response = ''
            while True:
                data = sock.recv(1024)
                if not data:
                    break
                response += data.decode()
                if 'EOF' in response:
                    break
            if response:
                process_response(response, port)
    except Exception as e:
        pass  # Ignore ports that are not open

def process_response(response, port):
    # Optional logging
    lines = response.strip().split('\n')
    if len(lines) < 3:
        return

    name = lines[0].strip()
    peripheral_uuid = lines[1].strip()
    config_lines = lines[2:]
    peripheral_config = '\n'.join(config_lines)

    # Update peripherals list
    with peripherals_lock:
        peripheral = {
            'name': name,
            'uuid': peripheral_uuid,
            'config': peripheral_config,
            'port': port,
            'last_seen': time.time(),
        }

        # Check for existing peripheral
        existing = next((p for p in config['peripherals'] if p['uuid'] == peripheral_uuid), None)
        if existing:
            existing.update(peripheral)
        else:
            # Handle multiple instances of the same peripheral
            same_name_count = sum(1 for p in config['peripherals'] if p['name'] == name or p['name'].startswith(f"{name}_"))
            if same_name_count > 0:
                peripheral['name'] = f"{name}_{same_name_count + 1}"
            config['peripherals'].append(peripheral)

    write_config()
    assign_colors_to_peripherals()
    if not in_command_mode.is_set():
        update_event.set()  # Signal to update the display

def assign_colors_to_peripherals():
    """Assign colors to peripherals based on their names."""
    with peripherals_lock:
        unique_names = list(set(p['name'].split('_')[0] for p in config['peripherals']))
        for idx, name in enumerate(unique_names):
            color_pair = (idx % 6) + 1  # Use color pairs 1-6
            peripheral_colors[name] = color_pair

def periodic_scan():
    scan_interval = int(config.get('scan_interval', '5'))
    while True:
        scan_ports()
        time.sleep(scan_interval)

def start_orchestrator():
    try:
        read_config()
        read_routes()
        assign_colors_to_peripherals()
        threading.Thread(target=periodic_scan, daemon=True).start()
        threading.Thread(target=command_listener, daemon=True).start()
        threading.Thread(target=data_listener, daemon=True).start()
        run_curses_interface()
    except Exception as e:
        print(f"Error in start_orchestrator: {e}")
        # Ensure curses is cleaned up if an error occurs
        try:
            curses.endwin()
        except Exception:
            pass
        # Restart the orchestrator
        start_orchestrator()

def process_command(command, source, conn=None):
    if source == 'port':
        # Log external command
        with threading.Lock():
            external_commands.append(f"From {conn.getpeername()}: {command}")
            if len(external_commands) > 5:
                external_commands.pop(0)
        if not in_command_mode.is_set():
            update_event.set()
    if command.startswith('/data'):
        # Handle incoming data from peripherals
        tokens = command.strip().split(' ', 2)
        if len(tokens) >= 3:
            peripheral_uuid = tokens[1]
            data = tokens[2]
            handle_incoming_data(peripheral_uuid, data)
        else:
            send_response("Invalid data command format.", source, conn)
    elif command.startswith('/register'):
        # Handle registration of peripherals
        tokens = command.strip().split(' ', 3)
        if len(tokens) == 4:
            name = tokens[1]
            peripheral_uuid = tokens[2]
            port = int(tokens[3])
            register_peripheral(name, peripheral_uuid, port, conn)
        else:
            send_response("Invalid register command format.", source, conn)
    elif command == '/help':
        help_text = get_help_text()
        send_response(help_text, source, conn)
    elif command == '/list' or command == '/available':
        peripherals_list = list_peripherals()
        send_response(peripherals_list, source, conn)
    elif command.startswith('/routes'):
        process_routes_command(command, source, conn)
    elif command == '/exit':
        send_response("Exiting command mode.", source, conn)
        if source == 'curses':
            in_command_mode.clear()  # Exit command mode
            return False  # Signal to exit command mode
        elif source == 'console':
            exit(0)
    else:
        send_response("Unknown command. Type '/help' for available commands.", source, conn)
    return True  # Continue in command mode

def register_peripheral(name, peripheral_uuid, port, conn):
    with peripherals_lock:
        peripheral = {
            'name': name,
            'uuid': peripheral_uuid,
            'config': '',
            'port': port,
            'last_seen': time.time(),
        }
        # Check for existing peripheral
        existing = next((p for p in config['peripherals'] if p['uuid'] == peripheral_uuid), None)
        if existing:
            existing.update(peripheral)
        else:
            # Handle multiple instances of the same peripheral
            same_name_count = sum(1 for p in config['peripherals'] if p['name'] == name or p['name'].startswith(f"{name}_"))
            if same_name_count > 0:
                peripheral['name'] = f"{name}_{same_name_count + 1}"
            config['peripherals'].append(peripheral)
    write_config()
    assign_colors_to_peripherals()
    if not in_command_mode.is_set():
        update_event.set()
    log_message(f"Registered peripheral: {peripheral['name']} on port {port}")

    # Send acknowledgment with data port
    data_port = config.get('data_port', '6001')
    response = f"/ack {data_port}\n"
    if conn:
        try:
            conn.sendall(response.encode())
        except Exception as e:
            log_message(f"Failed to send ack to peripheral: {e}")

def send_response(message, source, conn):
    if source == 'console':
        print(message)
    elif source == 'port' and conn:
        try:
            conn.sendall((message + "\n").encode())
        except Exception:
            pass
    elif source == 'curses':
        global stdscr
        try:
            # Clear the line before printing the message
            stdscr.move(2, 0)
            stdscr.clrtoeol()
            stdscr.addstr(2, 0, message)
            stdscr.refresh()
        except curses.error as e:
            # Handle curses error
            pass

def get_help_text():
    return (
        "Available commands:\n"
        "/help - Show this help message\n"
        "/list or /available - List known peripherals\n"
        "/routes - Manage routes\n"
        "    /routes help - Show routes command help\n"
        "/exit - Exit command mode or exit the orchestrator\n"
    )

def list_peripherals():
    with peripherals_lock:
        if not config['peripherals']:
            return "No peripherals discovered."
        output = "Known peripherals:\n"
        for idx, peripheral in enumerate(config['peripherals']):
            last_seen = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(peripheral['last_seen']))
            output += f"{idx + 1}. {peripheral['name']} (UUID: {peripheral['uuid']}, Port: {peripheral['port']}, Last Seen: {last_seen})\n"
        return output.strip()

def process_routes_command(command, source, conn):
    tokens = command.strip().split()
    if len(tokens) < 2:
        send_response("Invalid routes command. Type '/routes help' for usage.", source, conn)
        return
    action = tokens[1]
    if action == 'help':
        routes_help = get_routes_help_text()
        send_response(routes_help, source, conn)
    elif action == 'add':
        if len(tokens) != 5:
            send_response("Usage: /routes add <route-name> <incoming-peripheral-name> <outgoing-peripheral-name>", source, conn)
            return
        route_name = tokens[2]
        incoming_name = tokens[3]
        outgoing_name = tokens[4]
        result = add_route(route_name, incoming_name, outgoing_name)
        send_response(result, source, conn)
    elif action == 'remove':
        if len(tokens) != 3:
            send_response("Usage: /routes remove <route-name>", source, conn)
            return
        route_name = tokens[2]
        result = remove_route(route_name)
        send_response(result, source, conn)
    elif action == 'info':
        routes_info = list_routes()
        send_response(routes_info, source, conn)
    else:
        send_response("Unknown routes command. Type '/routes help' for usage.", source, conn)

def get_routes_help_text():
    return (
        "Routes command usage:\n"
        "/routes add <route-name> <incoming-peripheral-name> <outgoing-peripheral-name> - Add a new route\n"
        "/routes remove <route-name> - Remove an existing route\n"
        "/routes info - List all routes\n"
        "/routes help - Show this help message\n"
    )

def add_route(route_name, incoming_name, outgoing_name):
    with peripherals_lock:
        incoming = next((p for p in config['peripherals'] if p['name'] == incoming_name), None)
        outgoing = next((p for p in config['peripherals'] if p['name'] == outgoing_name), None)
    if not incoming:
        return f"Incoming peripheral '{incoming_name}' not found."
    if not outgoing:
        return f"Outgoing peripheral '{outgoing_name}' not found."
    with routes_lock:
        # Check if route already exists
        if any(r for r in routes if r['name'] == route_name):
            return f"Route '{route_name}' already exists."
        route = {
            'name': route_name,
            'incoming': incoming['uuid'],
            'outgoing': outgoing['uuid'],
            'incoming_port': incoming['port'],
            'outgoing_port': outgoing['port'],
            'last_used': None,
        }
        routes.append(route)
    write_routes()
    return f"Route '{route_name}' added successfully."

def remove_route(route_name):
    with routes_lock:
        route = next((r for r in routes if r['name'] == route_name), None)
        if not route:
            return f"Route '{route_name}' not found."
        routes.remove(route)
    write_routes()
    return f"Route '{route_name}' removed successfully."

def list_routes():
    with routes_lock:
        if not routes:
            return "No routes configured."
        output = "Configured routes:\n"
        for route in routes:
            incoming_name = get_peripheral_name_by_uuid(route['incoming'])
            outgoing_name = get_peripheral_name_by_uuid(route['outgoing'])
            last_used = route['last_used']
            if last_used:
                last_used_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(last_used))
            else:
                last_used_str = 'Never'
            output += f"Route Name: {route['name']}\n"
            output += f"  From: {incoming_name} (UUID: {route['incoming']}, Port: {route['incoming_port']})\n"
            output += f"  To: {outgoing_name} (UUID: {route['outgoing']}, Port: {route['outgoing_port']})\n"
            output += f"  Last Used: {last_used_str}\n"
        return output.strip()

def get_peripheral_name_by_uuid(uuid_str):
    with peripherals_lock:
        peripheral = next((p for p in config['peripherals'] if p['uuid'] == uuid_str), None)
    if peripheral:
        return peripheral['name']
    else:
        return 'Unknown'

def handle_incoming_data(peripheral_uuid, data):
    # Find routes where this peripheral is the incoming peripheral
    with routes_lock:
        matching_routes = [route for route in routes if route['incoming'] == peripheral_uuid]
    if not matching_routes:
        log_message(f"No routes found for peripheral UUID {peripheral_uuid}")
        return
    for route in matching_routes:
        # Forward data to the outgoing peripheral
        outgoing_port = int(route['outgoing_port'])
        try:
            with socket.create_connection(('localhost', outgoing_port), timeout=5) as s_out:
                s_out.sendall(data.encode())
                # Update last used timestamp
                route['last_used'] = time.time()
                write_routes()
                # Log activity
                incoming_name = get_peripheral_name_by_uuid(route['incoming'])
                outgoing_name = get_peripheral_name_by_uuid(route['outgoing'])
                log_message(f"{incoming_name} sent data to {outgoing_name} via route '{route['name']}'")
        except Exception as e:
            log_message(f"Error forwarding data on route '{route['name']}': {e}")

def command_listener():
    command_port = int(config.get('command_port', '6000'))
    host = '0.0.0.0'
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, command_port))
            s.listen(5)
            log_message(f"Command listener started on {host}:{command_port}")
            while True:
                conn, addr = s.accept()
                log_message(f"Received connection from {addr}")
                threading.Thread(target=handle_client_connection, args=(conn, addr), daemon=True).start()
    except Exception as e:
        log_message(f"Error in command_listener: {e}")

def data_listener():
    data_port = int(config.get('data_port', '6001'))
    host = '0.0.0.0'
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, data_port))
            s.listen(5)
            log_message(f"Data listener started on {host}:{data_port}")
            while True:
                conn, addr = s.accept()
                threading.Thread(target=handle_data_connection, args=(conn, addr), daemon=True).start()
    except Exception as e:
        log_message(f"Error in data_listener: {e}")

def handle_client_connection(conn, addr):
    with conn:
        conn.settimeout(5)  # Increase timeout
        buffer = ''
        while True:
            try:
                data = conn.recv(1024)
                if not data:
                    break
                buffer += data.decode()
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    command = line.strip()
                    if command == '':
                        continue
                    process_command(command, 'port', conn)
            except socket.timeout:
                continue
            except Exception as e:
                break

def handle_data_connection(conn, addr):
    with conn:
        conn.settimeout(5)
        buffer = ''
        peripheral_uuid = None
        while True:
            try:
                data = conn.recv(1024)
                if not data:
                    break
                buffer += data.decode()
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if peripheral_uuid is None:
                        # Expecting peripheral UUID as the first line
                        peripheral_uuid = line.strip()
                    else:
                        handle_incoming_data(peripheral_uuid, line.strip())
            except socket.timeout:
                continue
            except Exception as e:
                break

def run_curses_interface():
    global stdscr
    stdscr = curses.initscr()
    curses.start_color()
    # Initialize color pairs
    curses.init_pair(1, curses.COLOR_RED, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)
    curses.init_pair(4, curses.COLOR_BLUE, curses.COLOR_BLACK)
    curses.init_pair(5, curses.COLOR_MAGENTA, curses.COLOR_BLACK)
    curses.init_pair(6, curses.COLOR_CYAN, curses.COLOR_BLACK)
    curses.noecho()
    curses.cbreak()
    stdscr.nodelay(True)
    stdscr.keypad(True)
    try:
        while True:
            try:
                if not in_command_mode.is_set():
                    display_overview()
                update_event.wait(1)
                if update_event.is_set():
                    update_event.clear()
                c = stdscr.getch()
                if c != -1:
                    # Enter command mode
                    in_command_mode.set()
                    curses.echo()
                    curses.nocbreak()
                    stdscr.nodelay(False)
                    stdscr.clear()
                    stdscr.addstr(0, 0, "Command Mode (type '/exit' to return to overview):")
                    stdscr.addstr(1, 0, "> ")
                    stdscr.refresh()
                    command_input = ""
                    continue_command_mode = True
                    while continue_command_mode:
                        command = stdscr.getstr().decode()
                        # Clear any previous messages
                        stdscr.move(2, 0)
                        stdscr.clrtoeol()
                        continue_command_mode = process_command(command, 'curses')
                        if continue_command_mode:
                            stdscr.addstr(1, 0, "> ")
                            stdscr.clrtoeol()
                            stdscr.refresh()
                    # Return to overview mode
                    curses.noecho()
                    curses.cbreak()
                    stdscr.nodelay(True)
                    in_command_mode.clear()
                    update_event.set()  # Update display after exiting command mode
            except Exception as e:
                # Handle any exception that occurs during the curses interface
                error_message = f"An error occurred: {e}"
                # Log the error
                log_message(error_message)
                # Display the error message in the curses window
                try:
                    stdscr.clear()
                    stdscr.addstr(0, 0, "Error in curses interface:", curses.A_BOLD | curses.A_UNDERLINE)
                    stdscr.addstr(1, 0, error_message)
                    stdscr.refresh()
                    time.sleep(2)
                except curses.error:
                    pass
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up curses
        stdscr.keypad(False)
        curses.echo()
        curses.nocbreak()
        curses.endwin()

def display_overview():
    try:
        stdscr.clear()
        max_y, max_x = stdscr.getmaxyx()
        # Divide the screen into sections
        mid_x = max_x // 2
        # Display title
        stdscr.addstr(0, 0, "Orchestrator Overview", curses.A_BOLD | curses.A_UNDERLINE)
        # Display session UUID in top right
        session_uuid_str = f"Session UUID: {orchestrator_uuid}"
        if len(session_uuid_str) < max_x:
            stdscr.addstr(0, max_x - len(session_uuid_str) - 1, session_uuid_str)
        else:
            truncated_uuid = (session_uuid_str[:max_x - 5] + '...') if len(session_uuid_str) > max_x else session_uuid_str
            stdscr.addstr(0, max_x - len(truncated_uuid) - 1, truncated_uuid)
        # Display peripherals
        stdscr.addstr(2, 0, "Peripherals:", curses.A_BOLD | curses.A_UNDERLINE)
        with peripherals_lock:
            for idx, peripheral in enumerate(config['peripherals']):
                color = peripheral_colors.get(peripheral['name'].split('_')[0], 0)
                color_attr = curses.color_pair(color)
                last_seen = time.strftime('%H:%M:%S', time.localtime(peripheral['last_seen']))
                line = f"{peripheral['name']} (Port: {peripheral['port']}, Last Seen: {last_seen})"
                # Truncate line if it's too long
                if len(line) > max_x - 4:
                    truncated_line = line[:max_x - 7] + '...'
                else:
                    truncated_line = line
                stdscr.addstr(3 + idx, 2, truncated_line, color_attr)
        # Display routes and their relationships
        stdscr.addstr(2, mid_x, "Routes:", curses.A_BOLD | curses.A_UNDERLINE)
        with routes_lock:
            for idx, route in enumerate(routes):
                incoming_name = get_peripheral_name_by_uuid(route['incoming'])
                outgoing_name = get_peripheral_name_by_uuid(route['outgoing'])
                last_used = route['last_used']
                if last_used:
                    last_used_str = time.strftime('%H:%M:%S', time.localtime(last_used))
                else:
                    last_used_str = 'Never'
                # Use colors for peripherals
                in_color = peripheral_colors.get(incoming_name.split('_')[0], 0)
                out_color = peripheral_colors.get(outgoing_name.split('_')[0], 0)
                route_line = f"{route['name']}: "
                stdscr.addstr(3 + idx, mid_x + 2, route_line)
                # Truncate incoming name if necessary
                incoming_display = incoming_name
                if len(incoming_display) > max_x - mid_x - 10:
                    incoming_display = (incoming_display[:max_x - mid_x - 13] + '...') if len(incoming_display) > max_x - mid_x - 10 else incoming_display
                stdscr.addstr(f"{incoming_display}", curses.color_pair(in_color))
                stdscr.addstr(" -> ")
                # Truncate outgoing name if necessary
                outgoing_display = outgoing_name
                if len(outgoing_display) > max_x - mid_x - 10:
                    outgoing_display = (outgoing_display[:max_x - mid_x - 13] + '...') if len(outgoing_display) > max_x - mid_x - 10 else outgoing_display
                stdscr.addstr(f"{outgoing_display}", curses.color_pair(out_color))
                stdscr.addstr(f" | Last Used: {last_used_str}")
        # Calculate starting line for logs
        num_peripherals = len(config['peripherals'])
        num_routes = len(routes)
        content_height = max(num_peripherals, num_routes) + 5
        # Display activity log
        log_start_line = content_height
        stdscr.addstr(log_start_line, 0, "Recent Activity:", curses.A_BOLD | curses.A_UNDERLINE)
        log_start_line += 1
        max_log_lines = max_y - log_start_line - 6  # Reserve lines for external commands and commands received label
        with threading.Lock():
            recent_logs = activity_log[-max_log_lines:]
        for idx, (timestamp, message) in enumerate(recent_logs):
            time_str = time.strftime('%H:%M:%S', time.localtime(timestamp))
            log_line = f"[{time_str}] {message}"
            if len(log_line) > max_x - 4:
                log_line = log_line[:max_x - 7] + '...'
            stdscr.addstr(log_start_line + idx, 2, log_line)
        # Display external commands received
        cmd_recv_line = max_y - 5
        stdscr.addstr(cmd_recv_line, 0, "Commands Received:", curses.A_BOLD | curses.A_UNDERLINE)
        with threading.Lock():
            for idx, cmd in enumerate(external_commands[-4:]):
                # Truncate command to fit the window
                max_cmd_length = max_x - 4  # 2 for indent, 2 for buffer
                if len(cmd) > max_cmd_length:
                    truncated_cmd = cmd[:max_cmd_length - 3] + '...'
                else:
                    truncated_cmd = cmd
                stdscr.addstr(cmd_recv_line + idx + 1, 2, truncated_cmd)
        # Move cursor to the bottom
        stdscr.move(max_y - 1, 0)
        stdscr.refresh()
    except curses.error as e:
        # Handle curses errors (e.g., when window size is too small)
        error_message = f"Curses error: {e}"
        log_message(error_message)
        # Optionally display a message on the screen
        try:
            stdscr.clear()
            stdscr.addstr(0, 0, "Window too small for display.", curses.A_BOLD)
            stdscr.addstr(1, 0, "Please resize the terminal window.")
            stdscr.refresh()
        except curses.error:
            pass

def log_message(message):
    """Logs a message to the activity log and updates the display if not in command mode."""
    timestamp = time.time()
    activity_log.append((timestamp, message))
    if len(activity_log) > 1000:
        activity_log.pop(0)  # Keep activity log from growing indefinitely
    if not in_command_mode.is_set():
        update_event.set()
    # If curses is not initialized, print to console
    if 'stdscr' not in globals():
        print(message)

if __name__ == '__main__':
    try:
        start_orchestrator()
    except KeyboardInterrupt:
        # Clean up curses
        try:
            curses.endwin()
        except Exception:
            pass
        print("Orchestrator terminated by user.")
    except Exception as e:
        # Handle any other exceptions
        print(f"An unexpected error occurred: {e}")
        try:
            curses.endwin()
        except Exception:
            pass
