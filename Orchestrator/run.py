#!/usr/bin/env python3

import os
import sys
from pathlib import Path
import yaml
import tempfile
import subprocess
import shutil
import ast
import importlib.util
import sysconfig
import logging

# Configure logging
logging.basicConfig(
    filename='manage_scripts.log',
    level=logging.DEBUG,  # Changed from INFO to DEBUG for more detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s'
)

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
    # Removed incomplete binding lines to prevent configuration errors
]

# List of local or non-pip-installable modules to exclude
EXCLUDED_MODULES = [
    'riva',     # Custom or non-pip package
    'tuning',   # Local module
    # Add more modules as needed
]

# Convert to a set of lowercase module names for efficient and case-insensitive lookups
EXCLUDED_MODULES = {module.strip().lower() for module in EXCLUDED_MODULES}

# Mapping from import names to PyPI package names (if they differ)
IMPORT_TO_PACKAGE_MAPPING = {
    'grpc': 'grpcio',
    'usb': 'pyusb',
    'pyaudio': 'PyAudio',
    # Add more mappings as needed
}

def is_standard_module(module_name):
    """
    Determines if a module is part of Python's standard library.
    Returns True if it is, False otherwise.
    """
    if module_name in sys.builtin_module_names:
        return True
    try:
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            return False  # Module not found; treat as third-party to handle installation errors
        if spec.origin is None:
            return False
        # Get standard library paths
        stdlib_paths = [Path(p).resolve() for p in sysconfig.get_paths()["stdlib"].split(os.pathsep)]
        module_path = Path(spec.origin).resolve()
        return any(module_path.is_relative_to(stdlib_path) for stdlib_path in stdlib_paths)
    except AttributeError:
        # For Python versions < 3.9 where is_relative_to is not available
        try:
            module_path = Path(spec.origin).resolve()
            for stdlib_path in sysconfig.get_paths()["stdlib"].split(os.pathsep):
                stdlib_path = Path(stdlib_path).resolve()
                if str(module_path).startswith(str(stdlib_path)):
                    return True
        except Exception:
            pass
        return False


def get_script_folders(parent_dir):
    """
    Returns a list of subdirectories that contain a Python script
    with the same name as the folder.
    Excludes the 'venv' directory.
    """
    folders = []
    for item in parent_dir.iterdir():
        if item.is_dir() and item.name != 'venv':
            script = item / f"{item.name}.py"
            if script.is_file():
                folders.append(item)
    logging.info(f"Found {len(folders)} script folders.")
    return folders

def get_all_python_files(script_folders):
    """
    Recursively collects all .py files in the given script folders and their subdirectories.
    Excludes the 'venv' directory.
    """
    python_files = []
    for folder in script_folders:
        files = list(folder.rglob("*.py"))
        python_files.extend(files)
        logging.info(f"Found {len(files)} Python files in folder '{folder}'.")
    return python_files

def get_imports_from_scripts(python_files):
    """
    Extracts all imports from the provided Python files.
    """
    imports = set()
    for script_path in python_files:
        with open(script_path, 'r') as f:
            try:
                tree = ast.parse(f.read(), filename=str(script_path))
            except SyntaxError as e:
                logging.error(f"Syntax error in {script_path}: {e}")
                continue  # Skip scripts with syntax errors
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for n in node.names:
                    imports.add(n.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split('.')[0])
    logging.info(f"Collected imports: {imports}")
    return imports

def install_packages_into_venv(venv_dir, imports):
    """
    Installs the specified packages into the virtual environment.
    Adds new packages to requirements.txt without removing manually added ones.
    Handles installation errors gracefully and logs detailed information.
    Continues installing remaining packages even if some fail.
    """
    venv_pip = venv_dir / 'bin' / 'pip'
    requirements_file = project_dir / 'requirements.txt'

    # Ensure requirements.txt exists
    if not requirements_file.exists():
        logging.info(f"{requirements_file} does not exist. Creating a new one.")
        requirements_file.touch()

    # Determine new packages to add based on imports
    new_packages = set()
    logging.info(f"Excluded modules: {EXCLUDED_MODULES}")
    logging.info(f"Processing imports: {imports}")
    for imp in imports:
        imp_clean = imp.strip().lower()
        if imp_clean in EXCLUDED_MODULES:
            logging.info(f"Excluded module '{imp}' detected; skipping.")
            continue
        if not is_standard_module(imp_clean):
            # Map import name to package name if necessary
            package = IMPORT_TO_PACKAGE_MAPPING.get(imp_clean, imp_clean)
            new_packages.add(package)

    # Read existing requirements from file
    existing_requirements = set()
    with open(requirements_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # Handle cases like 'package==1.2.3' or 'package>=1.0'
                package = line.split('==')[0].split('>=')[0].split('<=')[0].strip().lower()
                existing_requirements.add(package)

    # Identify packages to add (excluding already existing ones)
    packages_to_add = {pkg for pkg in new_packages if pkg.lower() not in existing_requirements}

    if packages_to_add:
        logging.info(f"Adding new packages to requirements.txt: {sorted(packages_to_add)}")
        with open(requirements_file, 'a') as f:
            for package in sorted(packages_to_add):
                f.write(f"{package}\n")
    else:
        logging.info("No new packages to add to requirements.txt.")

    # Log the current contents of requirements.txt
    with open(requirements_file, 'r') as f:
        requirements_content = f.read()
    logging.debug(f"Current requirements.txt content:\n{requirements_content}")

    # Upgrade pip to the latest version to avoid potential issues
    try:
        logging.info("Upgrading pip in the virtual environment...")
        upgrade_pip = subprocess.run(
            [str(venv_pip), 'install', '--upgrade', 'pip'],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        logging.info("pip upgraded successfully.")
        logging.debug(f"pip upgrade output:\n{upgrade_pip.stdout}")
        if upgrade_pip.stderr:
            logging.warning(f"pip upgrade warnings/errors:\n{upgrade_pip.stderr}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to upgrade pip: {e.stderr}")
        print("ERROR: Failed to upgrade pip. Check manage_scripts.log for details.")
        # Proceeding despite pip upgrade failure

    # Read all packages from requirements.txt
    try:
        with open(requirements_file, 'r') as f:
            packages = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except Exception as e:
        logging.error(f"Failed to read requirements.txt: {e}")
        print(f"ERROR: Failed to read requirements.txt: {e}. Check manage_scripts.log for details.")
        packages = []

    # Install each package individually
    for package in packages:
        try:
            logging.info(f"Installing package: {package}")
            install_result = subprocess.run(
                [str(venv_pip), 'install', package],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            logging.info(f"Successfully installed {package}.")
            logging.debug(f"pip install {package} stdout:\n{install_result.stdout}")
            if install_result.stderr:
                logging.warning(f"pip install {package} stderr:\n{install_result.stderr}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to install package '{package}': {e.stderr}")
            print(f"WARNING: Failed to install package '{package}'. Check manage_scripts.log for details.")
            # Continue with the next package

    # Log the installed packages after installation
    try:
        installed_after_install = subprocess.check_output([str(venv_pip), 'freeze'], text=True)
        logging.debug(f"Installed packages after installation:\n{installed_after_install}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to list installed packages after installation: {e}")
        print("ERROR: Failed to list installed packages after installation. Check manage_scripts.log for details.")
        # Proceeding despite failure to list installed packages

def verify_installation(venv_dir):
    """
    Verifies that all packages listed in requirements.txt are installed in the venv.
    Logs any missing packages without exiting the script.
    """
    venv_pip = venv_dir / 'bin' / 'pip'
    requirements_file = project_dir / 'requirements.txt'

    if not requirements_file.exists():
        logging.error(f"{requirements_file} does not exist.")
        print(f"ERROR: {requirements_file} does not exist.")
        return  # Changed from sys.exit(1) to return

    try:
        installed_packages = subprocess.check_output([str(venv_pip), 'freeze'], text=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to list installed packages: {e}")
        print("ERROR: Failed to list installed packages. Check manage_scripts.log for details.")
        return  # Changed from sys.exit(1) to return

    installed_packages_set = set()
    for line in installed_packages.strip().split('\n'):
        if line:
            # Handle cases like 'package==1.2.3' or 'package>=1.0'
            pkg = line.split('==')[0].split('>=')[0].split('<=')[0].strip().lower()
            installed_packages_set.add(pkg)

    # Read requirements.txt
    required_packages = set()
    with open(requirements_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # Handle cases like 'package==1.2.3' or 'package>=1.0'
                pkg = line.split('==')[0].split('>=')[0].split('<=')[0].strip().lower()
                # Handle extras, e.g., 'package[extra]'
                pkg = pkg.split('[')[0]
                required_packages.add(pkg)

    # Identify missing packages
    missing_packages = required_packages - installed_packages_set

    if missing_packages:
        logging.error(f"The following packages are missing in the virtual environment: {missing_packages}")
        print(f"WARNING: The following packages are missing in the virtual environment: {', '.join(missing_packages)}")
    else:
        logging.info("All required packages are installed in the virtual environment.")
        print("All required packages are installed in the virtual environment.")


def list_installed_packages(venv_dir):
    """
    Lists all installed packages in the virtual environment.
    """
    venv_pip = venv_dir / 'bin' / 'pip'
    try:
        installed_packages = subprocess.check_output([str(venv_pip), 'list'], text=True)
        logging.info(f"Installed packages in venv:\n{installed_packages}")
        print(f"Installed packages in venv:\n{installed_packages}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to list installed packages: {e}")
        print("ERROR: Failed to list installed packages. Check manage_scripts.log for details.")

def create_tmuxp_config(script_folders, venv_dir, session_name='scripts_session'):
    """
    Creates a tmuxp configuration dictionary.
    Organizes scripts into panes within a single window.
    The number of panes matches the number of script folders.
    """
    config = {
        "session_name": session_name,
        "windows": []
    }

    window = {
        "window_name": "Scripts Window",
        "layout": "tiled",  # 'tiled' layout arranges panes in a grid
        "panes": []
    }

    for folder in script_folders:
        main_script = folder / f"{folder.name}.py"
        if not main_script.is_file():
            logging.warning(f"No main script found in folder '{folder}'. Expected '{main_script}'. Skipping.")
            continue

        # Use the Python interpreter from the virtual environment
        python_executable = venv_dir / 'bin' / 'python'

        # Construct the pane command to run the main script within the virtual environment
        # Using 'bash -c' to ensure the venv is activated in the pane
        # Added a check to verify the Python executable being used
        pane_command = (
            f"bash -c 'cd \"{folder}\" && "
            f"source \"{venv_dir}/bin/activate\" && "
            f"echo \"Using Python: $(which python)\" && "
            f"python \"{main_script.name}\" && "
            f"echo \"Script {main_script.name} finished execution.\"; "
            f"exec bash'"
        )
        window["panes"].append({
            "shell_command": pane_command
        })
        logging.info(f"Prepared pane command for folder '{folder}': {pane_command}")

    # Add mouse support in the tmuxp configuration
    config["options"] = {
        "mouse": "on"
    }

    # Add the window with all panes
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
    logging.info(f"tmuxp configuration written to {config_path}")
    return config_path

def is_tool_installed(tool):
    """
    Check if a tool is installed and available in PATH.
    """
    return shutil.which(tool) is not None

def is_tmux_installed():
    """
    Checks if tmux is installed and accessible.
    Returns True if installed, False otherwise.
    """
    return is_tool_installed('tmux')

def is_tmuxp_installed():
    """
    Checks if tmuxp is installed and accessible.
    Returns True if installed, False otherwise.
    """
    return is_tool_installed('tmuxp')

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
        logging.info(f"Killed existing tmux session '{session_name}'.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to kill existing tmux session '{session_name}': {e}")
        # Do not exit; proceed

def load_tmuxp_session(config_path, session_name):
    """
    Loads the tmuxp session using the provided configuration file.
    """
    try:
        subprocess.check_call(["tmuxp", "load", config_path])
        logging.info("tmuxp session loaded successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to load tmuxp session: {e}")
        print("ERROR: Failed to load tmuxp session. Check manage_scripts.log for details.")
        # Proceeding despite tmuxp load failure
    
    # Source the local tmux.conf to apply configurations
    try:
        subprocess.check_call(["tmux", "source-file", str(project_dir / "tmux.conf")])
        logging.info("Loaded local tmux.conf into the session.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to source local tmux.conf: {e}")
        print("ERROR: Failed to source local tmux.conf. Check manage_scripts.log for details.")
        # Proceeding despite tmux.conf sourcing failure
    
    # Enable mouse support directly via tmux command
    try:
        subprocess.check_call(["tmux", "set-option", "-g", "mouse", "on", "-t", session_name])
        logging.info("Enabled mouse support in tmux session.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to enable mouse support: {e}")
        print("ERROR: Failed to enable mouse support in tmux session. Check manage_scripts.log for details.")
        # Proceeding despite mouse support failure

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
            logging.info(f"Backup of original local tmux.conf created at {backup_path}")
        else:
            # Create an empty tmux.conf if it doesn't exist
            tmux_conf_path.touch()
            logging.info(f"Created new tmux.conf at {tmux_conf_path}")

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
        logging.info(f"Updated local tmux.conf at {tmux_conf_path} with necessary configurations.")
    else:
        logging.info(f"No updates needed for local tmux.conf at {tmux_conf_path}. All configurations already present.")

def open_gnome_terminal(session_name):
    """
    Open gnome-terminal and attach to the specified tmux session.
    """
    try:
        subprocess.Popen([
            'gnome-terminal',
            '--',
            'bash',
            '-c',
            f'tmux attach-session -t {session_name}; exec bash'
        ])
        logging.info(f"Opened gnome-terminal attached to tmux session '{session_name}'.")
    except Exception as e:
        logging.error(f"Failed to open gnome-terminal: {e}")
        print(f"ERROR: Failed to open gnome-terminal: {e}. Check manage_scripts.log for details.")
        # Do not exit; proceed

def create_venv(project_dir):
    """
    Creates a virtual environment in the project directory.
    """
    venv_dir = project_dir / 'venv'
    if venv_dir.exists():
        logging.info(f"Virtual environment already exists at {venv_dir}")
    else:
        logging.info(f"Creating virtual environment at {venv_dir}")
        import venv
        venv_builder = venv.EnvBuilder(with_pip=True)
        venv_builder.create(str(venv_dir))
    return venv_dir

def main():
    try:
        # Ensure the script is run from its directory
        global project_dir  # To make it accessible in load_tmuxp_session
        project_dir = Path.cwd()
        logging.info(f"Scanning directory: {project_dir}")
        print(f"Scanning directory: {project_dir}")

        # Step 1: Find all script folders
        script_folders = get_script_folders(project_dir)
        if not script_folders:
            logging.error("No script folders found. Ensure each folder contains a Python script with the same name.")
            print("No script folders found. Ensure each folder contains a Python script with the same name.")
            sys.exit(1)
        logging.info(f"Found script folders: {[f.name for f in script_folders]}")
        print(f"Found script folders: {[f.name for f in script_folders]}")

        # Step 2: Collect all Python files across all script folders
        python_files = get_all_python_files(script_folders)
        if not python_files:
            logging.error("No Python files found in the script folders.")
            print("No Python files found in the script folders.")
            sys.exit(1)

        # Step 3: Extract imports from all Python files
        imports = get_imports_from_scripts(python_files)
        if not imports:
            logging.warning("No imports found in the Python scripts.")
            print("No imports found in the Python scripts.")
        else:
            logging.info(f"Total unique imports collected: {len(imports)}")
            print(f"Total unique imports collected: {len(imports)}")

        # Step 4: Create virtual environment
        venv_dir = create_venv(project_dir)

        # Step 5: Install dependencies into the virtual environment
        install_packages_into_venv(venv_dir, imports)

        # Step 6: Verify installation
        verify_installation(venv_dir)

        # Optional Step 7: List installed packages
        list_installed_packages(venv_dir)

        # Step 8: Create tmuxp config
        tmuxp_config = create_tmuxp_config(script_folders, venv_dir, session_name='scripts_session')
        logging.info("Generated tmuxp configuration.")
        print("Generated tmuxp configuration.")

        # Step 9: Save config to temporary file
        config_path = save_config_to_tempfile(tmuxp_config)

        # Step 10: Create or update local tmux.conf
        create_local_tmux_conf(project_dir)

        # Step 11: Check if the session already exists and kill it if so
        session_name = "scripts_session"
        if session_exists(session_name):
            logging.info(f"Tmux session '{session_name}' already exists.")
            print(f"Tmux session '{session_name}' already exists.")
            print(f"Killing the existing tmux session '{session_name}' and starting a new one.")
            kill_existing_session(session_name)

        # Step 12: Load tmuxp session
        load_tmuxp_session(config_path, session_name)

        # Step 13: Cleanup temporary config file
        try:
            os.remove(config_path)
            logging.info(f"Temporary tmuxp configuration file {config_path} removed.")
            print(f"Temporary tmuxp configuration file {config_path} removed.")
        except OSError:
            logging.error(f"Could not remove temporary tmuxp configuration file {config_path}.")
            print(f"Could not remove temporary tmuxp configuration file {config_path}.")

        # Step 14: Open gnome-terminal attached to the tmux session
        open_gnome_terminal(session_name)

        logging.info("All scripts are running in tmux within a single gnome-terminal window.")
        print("All scripts are running in tmux within a single gnome-terminal window.")
        print("Use your mouse to select panes or your key bindings to navigate.")
        print("Check 'manage_scripts.log' for detailed logs.")

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        print(f"ERROR: An unexpected error occurred: {e}. Check manage_scripts.log for details.")
        sys.exit(1)

if __name__ == "__main__":
    # Verify tmux installation
    if not is_tmux_installed():
        print("tmux not found. Please install tmux before running this script.")
        logging.error("tmux not found. Please install tmux before running this script.")
        sys.exit(1)
    
    # Verify tmuxp installation
    if not is_tmuxp_installed():
        print("tmuxp not found. Please install tmuxp using 'pip install tmuxp' before running this script.")
        logging.error("tmuxp not found. Please install tmuxp using 'pip install tmuxp' before running this script.")
        sys.exit(1)
    
    main()
