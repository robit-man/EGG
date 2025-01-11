import os
import subprocess
import sys
import socket
import time
import numpy as np

# Configuration
VENV_DIR = "output_venv"  # Virtual environment directory
LISTEN_HOST = "0.0.0.0"   # Listen on all interfaces
LISTEN_PORT = 6353        # Port to receive raw audio data

# ALSA Playback configuration
PCM_DEVICE = "default"  # ALSA device for playback
CHUNK = 4096            # Buffer size (in bytes). Adjusted for real-time streaming
RATE = 22050            # Ensure this matches the output format of your audio source
CHANNELS = 1

# Audio Processing Configuration
VOLUME = 0.2  # Volume control factor (1.0 = original volume). Adjust as needed

# Maximum number of pip install attempts
MAX_PIP_ATTEMPTS = 3

def is_venv():
    """Check if the script is running inside a virtual environment."""
    return (
        (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) or
        (hasattr(sys, 'real_prefix') and sys.real_prefix != sys.prefix)
    )

def create_venv():
    """Create a virtual environment in VENV_DIR if it doesn't exist."""
    if not os.path.exists(VENV_DIR):
        print(f"Creating virtual environment in {VENV_DIR}...")
        subprocess.run([sys.executable, "-m", "venv", VENV_DIR], check=True)

def install_dependencies():
    """Install required Python packages in the virtual environment with retry logic."""
    if sys.platform == "win32":
        pip_executable = os.path.join(VENV_DIR, "Scripts", "pip.exe")
    else:
        pip_executable = os.path.join(VENV_DIR, "bin", "pip")
    
    # Ensure pip executable exists
    if not os.path.exists(pip_executable):
        print("pip executable not found in the virtual environment.")
        return False

    print("Installing dependencies in the virtual environment...")
    attempts = 0
    while attempts < MAX_PIP_ATTEMPTS:
        try:
            # Upgrade pip first
            subprocess.run([pip_executable, "install", "--upgrade", "pip"], check=True)
            # Install required packages
            subprocess.run([pip_executable, "install", "pyalsaaudio", "numpy"], check=True)
            print("Dependencies installed successfully.")
            return True
        except subprocess.CalledProcessError as e:
            attempts += 1
            print(f"Pip install attempt {attempts} failed: {e}")
            if attempts < MAX_PIP_ATTEMPTS:
                print("Retrying pip install...")
                time.sleep(2)  # Wait before retrying
            else:
                print("Maximum pip install attempts reached. Skipping installation.")
                return False

def activate_venv():
    """Activate the virtual environment by modifying sys.path."""
    if sys.platform == "win32":
        venv_site_packages = os.path.join(
            VENV_DIR, "Lib", "site-packages"
        )
    else:
        venv_site_packages = os.path.join(
            VENV_DIR, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages"
        )
    sys.path.insert(0, venv_site_packages)

def relaunch_in_venv():
    """Relaunch the current script within the virtual environment."""
    if sys.platform == "win32":
        python_executable = os.path.join(VENV_DIR, "Scripts", "python.exe")
    else:
        python_executable = os.path.join(VENV_DIR, "bin", "python")
    if not os.path.exists(python_executable):
        sys.exit("Error: Python executable not found in the virtual environment.")
    subprocess.check_call([python_executable] + sys.argv)
    sys.exit()

def setup_virtual_environment():
    """Ensure that the virtual environment is set up and dependencies are installed."""
    if not is_venv():
        create_venv()
        install_successful = install_dependencies()
        if not install_successful:
            print("Proceeding without installing dependencies. Ensure they are already installed.")
        relaunch_in_venv()
    else:
        activate_venv()
        # Optionally, verify if dependencies are installed
        try:
            import alsaaudio
            import numpy
        except ImportError as e:
            print(f"Missing dependencies: {e}")
            print("Attempting to install dependencies...")
            install_successful = install_dependencies()
            if not install_successful:
                print("Proceeding without installing dependencies. Ensure they are already installed.")
            else:
                print("Dependencies installed successfully.")

def handle_client_connection(client_socket):
    """Handle the incoming client connection and play audio with volume control."""
    try:
        import alsaaudio

        # Setup ALSA for playback
        audio_out = alsaaudio.PCM(alsaaudio.PCM_PLAYBACK, device=PCM_DEVICE)
        audio_out.setchannels(CHANNELS)
        audio_out.setrate(RATE)
        audio_out.setformat(alsaaudio.PCM_FORMAT_S16_LE)
        audio_out.setperiodsize(CHUNK)

        while True:
            # Receive raw audio data
            data = client_socket.recv(CHUNK)
            if not data:
                print("Client disconnected.")
                break

            # Convert byte data to numpy array
            audio_samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)

            # Apply volume control
            audio_samples *= VOLUME

            # Ensure samples are within valid range
            audio_samples = np.clip(audio_samples, -32768, 32767)

            # Convert back to int16
            processed_data = audio_samples.astype(np.int16).tobytes()

            # Play the processed audio data using ALSA
            audio_out.write(processed_data)

    except Exception as e:
        print(f"Error in client connection: {e}")
    finally:
        client_socket.close()

def main():
    while True:
        server_socket = None
        try:
            # Set up the virtual environment
            setup_virtual_environment()

            # Create a socket to listen for raw audio
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((LISTEN_HOST, LISTEN_PORT))
            server_socket.listen(1)
            print(f"Listening for raw audio on port {LISTEN_PORT}...")

            client_socket, addr = server_socket.accept()
            print(f"Connection established with {addr}")

            handle_client_connection(client_socket)

        except Exception as e:
            print(f"An error occurred: {e}")
            time.sleep(5)  # Retry after a short delay

        finally:
            if server_socket:
                server_socket.close()
            print("Server socket closed. Retrying...")

if __name__ == "__main__":
    main()
