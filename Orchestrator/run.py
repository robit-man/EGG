#!/usr/bin/env python3

import os
import sys
from pathlib import Path
import yaml
import tempfile
import subprocess
import shutil

# Define the necessary tmux configurations to be added locally
LOCAL_TMUX_CONFIG_LINES = [
    "# Enable mouse support (tmux 2.1 and later)",
    "set -g mouse on",
    "",
    "# Optional: Improve pane border visibility",
    "setw -g pane-border-fg green",
    "setw -g pane-active-border-fg brightgreen",
    "",
    "# Optional: Enable pane resizing with mouse drag",
    "setw -g mouse-resize-pane on",
    "",
    "# Optional: Bind double-click to toggle pane zoom",
    "bind-key -n DoubleClick1Pane resize-pane -Z",
    "",
    "# Optional: Display session name and mode in status-left",
    'set -g status-left "#[fg=green]#{session_name} #[fg=yellow]@mode"',
    "",
    "# Optional: Display date and time in status-right",
    'set -g status-right "#H | %Y-%m-%d %H:%M:%S"',
    "",
    "# Optional: Key bindings for easier navigation (prefix + arrows)",
    "bind-key Left select-pane -L",
    "bind-key Right select-pane -R",
    "bind-key Up select-pane -U",
    "bind-key Down select-pane -D",
    "",
    "# Optional: Bind 'r' to reload configuration",
    'bind r source-file ~/.tmux.conf \\; display-message "Config reloaded."',
    "",
    "# Initialize a variable to track mode",
    'set -g @mode "navigation"',
    "",
    "# Function to toggle mode",
    'bind-key Enter run-shell "tmux set -g @mode \'typing\'"',
    'bind-key Escape run-shell "tmux set -g @mode \'navigation\'"',
    "",
    "# Customize status bar to include mode",
    'set -g status-left "#[fg=green]#{session_name} #[fg=yellow]@mode"',
]

def get_script_folders(parent_dir):
    """
    Returns a list of subdirectories that contain a Python script
    with the same name as the folder.
    """
    folders = []
    for item in parent_dir.iterdir():
        if item.is_dir():
            script = item / f"{item.name}.py"
            if script.is_file():
                folders.append(item)
    return folders

def create_tmuxp_config(scripts, session_name='scripts_session', panes_per_window=4):
    """
    Creates a tmuxp configuration dictionary.
    Organizes scripts into windows with a specified number of panes per window.
    """
    config = {
        "session_name": session_name,
        "windows": []
    }

    window = {
        "window_name": f"window_{len(config['windows']) + 1}",
        "layout": "tiled",  # You can change this to 'even-horizontal', 'even-vertical', etc.
        "panes": []
    }

    for idx, script in enumerate(scripts):
        pane_command = f"cd {script['folder']} && python3 {script['script']}"
        window["panes"].append(pane_command)

        # When panes_per_window is reached, add the window and start a new one
        if (idx + 1) % panes_per_window == 0:
            config["windows"].append(window)
            window = {
                "window_name": f"window_{len(config['windows']) + 1}",
                "layout": "tiled",
                "panes": []
            }

    # Add any remaining panes to the last window
    if window["panes"]:
        config["windows"].append(window)

    return config

def save_config_to_tempfile(config):
    """
    Saves the tmuxp config dictionary to a temporary YAML file.
    Returns the path to the temporary file.
    """
    with tempfile.NamedTemporaryFile('w', delete=False, suffix='.yaml') as tmpfile:
        yaml.dump(config, tmpfile)
        config_path = tmpfile.name
    print(f"tmuxp configuration written to {config_path}")
    return config_path

def is_tmux_installed():
    """
    Checks if tmux is installed and accessible.
    Returns True if installed, False otherwise.
    """
    try:
        subprocess.check_output(["tmux", "-V"], stderr=subprocess.STDOUT)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def session_exists(session_name):
    """
    Checks if a tmux session with the given name exists.
    Returns True if exists, False otherwise.
    """
    try:
        subprocess.check_output(["tmux", "has-session", "-t", session_name], stderr=subprocess.STDOUT)
        return True
    except subprocess.CalledProcessError:
        return False

def kill_existing_session(session_name):
    """
    Kills an existing tmux session with the given name.
    """
    try:
        subprocess.check_call(["tmux", "kill-session", "-t", session_name])
        print(f"Killed existing tmux session '{session_name}'.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to kill existing tmux session '{session_name}': {e}")
        sys.exit(1)

def load_tmuxp_session(config_path, session_name):
    """
    Loads the tmuxp session using the provided configuration file.
    """
    try:
        subprocess.check_call(["tmuxp", "load", config_path])
        print("tmuxp session loaded successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to load tmuxp session: {e}")
        sys.exit(1)

def create_local_tmux_conf(project_dir):
    """
    Creates or updates a local tmux.conf file within the project directory.
    """
    tmux_conf_path = project_dir / "tmux.conf"
    backup_path = project_dir / "tmux.conf.backup"

    # Backup the original local tmux.conf if not already backed up
    if not backup_path.is_file():
        if tmux_conf_path.is_file():
            shutil.copy(tmux_conf_path, backup_path)
            print(f"Backup of original local tmux.conf created at {backup_path}")
        else:
            # Create an empty tmux.conf if it doesn't exist
            tmux_conf_path.touch()
            print(f"Created new tmux.conf at {tmux_conf_path}")

    # Read the existing tmux.conf
    with open(tmux_conf_path, 'r') as f:
        existing_config = f.read()

    # Check if the necessary configurations are already present
    missing_lines = []
    for line in LOCAL_TMUX_CONFIG_LINES:
        stripped_line = line.strip()
        if stripped_line == "" or stripped_line.startswith("#"):
            continue  # Skip empty lines and comments
        if line not in existing_config:
            missing_lines.append(line)

    if missing_lines:
        # Append the missing configurations
        with open(tmux_conf_path, 'a') as f:
            f.write("\n# Added by manage_scripts.py for enhanced navigation and mouse support\n")
            for line in missing_lines:
                f.write(line + "\n")
        print(f"Updated local tmux.conf at {tmux_conf_path} with necessary configurations.")
    else:
        print(f"No updates needed for local tmux.conf at {tmux_conf_path}. All configurations already present.")

def main():
    # Ensure the script is run from its directory
    project_dir = Path.cwd()
    print(f"Scanning directory: {project_dir}")

    # Step 1: Find all script folders
    script_folders = get_script_folders(project_dir)
    if not script_folders:
        print("No script folders found. Ensure each folder contains a Python script with the same name.")
        sys.exit(1)
    print(f"Found script folders: {[f.name for f in script_folders]}")

    # Prepare scripts list for tmuxp config
    scripts = []
    for folder in script_folders:
        script_path = folder / f"{folder.name}.py"
        scripts.append({
            "folder": folder.resolve(),
            "script": f"{folder.name}.py"
        })
        print(f"Prepared script '{script_path}' for tmuxp.")

    # Step 2: Create tmuxp config
    panes_per_window = 4  # You can adjust this number as needed
    session_name = "scripts_session"  # Consistent session name
    tmuxp_config = create_tmuxp_config(scripts, session_name=session_name, panes_per_window=panes_per_window)
    print(f"Generated tmuxp configuration: {tmuxp_config}")

    # Step 3: Save config to temporary file
    config_path = save_config_to_tempfile(tmuxp_config)

    # Step 4: Create or update local tmux.conf
    create_local_tmux_conf(project_dir)

    # Step 5: Check if the session already exists
    if session_exists(session_name):
        print(f"Tmux session '{session_name}' already exists.")
        # Prompt the user for action
        response = input(f"Do you want to kill the existing tmux session '{session_name}' and start a new one? [Y/n] ").strip().lower()
        if response in ['y', 'yes', '']:
            kill_existing_session(session_name)
        elif response in ['n', 'no']:
            print(f"Attaching to existing tmux session '{session_name}'.")
            try:
                subprocess.check_call(["tmux", "attach-session", "-t", session_name])
                sys.exit(0)
            except subprocess.CalledProcessError as e:
                print(f"Failed to attach to tmux session '{session_name}': {e}")
                sys.exit(1)
        else:
            print("Invalid response. Please run the script again and respond with 'y' or 'n'.")
            sys.exit(1)

    # Step 6: Load tmuxp session
    load_tmuxp_session(config_path, session_name)

    # Step 7: Reload tmux configuration within the session
    try:
        subprocess.check_call([
            "tmux",
            "source-file", str(project_dir / "tmux.conf"),
            "-t", session_name
        ])
        print("Reloaded tmux configuration within the session.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to reload tmux configuration within the session: {e}")
        sys.exit(1)

    # Step 8: Cleanup temporary config file
    try:
        os.remove(config_path)
        print(f"Temporary tmuxp configuration file {config_path} removed.")
    except OSError:
        print(f"Could not remove temporary tmuxp configuration file {config_path}.")

    print("All scripts are running in tmux. Use your mouse to select panes or your key bindings to navigate.")
    print(f"To attach to the tmux session, run: tmux attach-session -t {session_name}")

if __name__ == "__main__":
    # Verify tmux installation
    if not is_tmux_installed():
        print("tmux not found. Please install tmux before running this script.")
        sys.exit(1)

    main()
