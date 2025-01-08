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
CHUNK = 8192            # Buffer size (in bytes). Increased for better processing.
RATE = 22050            # Ensure this matches the output format of your audio source
CHANNELS = 1

# Audio Processing Configuration
VOLUME = 0.8  # Volume control factor (1.0 = original volume). Adjust as needed
PITCH = 1.2   # Pitch control factor (1.0 = original pitch). Adjust as needed

# Pitch Shifting Configuration
MAX_PITCH_SHIFT = 2.0  # Maximum pitch shift factor to prevent excessive artifacts

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
    """Install required Python packages in the virtual environment."""
    pip_executable = os.path.join(VENV_DIR, "bin", "pip")
    if not os.path.exists(pip_executable):
        pip_executable = os.path.join(VENV_DIR, "Scripts", "pip.exe")  # For Windows
    print("Installing dependencies in the virtual environment...")
    subprocess.run([pip_executable, "install", "--upgrade", "pip"], check=True)
    subprocess.run([pip_executable, "install", "pyalsaaudio", "numpy", "librosa"], check=True)

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
        install_dependencies()
        relaunch_in_venv()
    else:
        activate_venv()

def handle_client_connection(client_socket):
    """Handle the incoming client connection and play audio with volume and pitch control."""
    try:
        import alsaaudio
        import librosa

        # Setup ALSA for playback
        audio_out = alsaaudio.PCM(alsaaudio.PCM_PLAYBACK)
        audio_out.setchannels(CHANNELS)
        audio_out.setrate(RATE)
        audio_out.setformat(alsaaudio.PCM_FORMAT_S16_LE)
        audio_out.setperiodsize(CHUNK)

        buffer = b""

        while True:
            # Receive raw audio data
            data = client_socket.recv(CHUNK)
            if not data:
                print("Client disconnected.")
                break

            buffer += data

            # Process in larger blocks (e.g., 32768 bytes)
            PROCESS_BUFFER_SIZE = 32768
            while len(buffer) >= PROCESS_BUFFER_SIZE:
                # Extract a block from the buffer
                block = buffer[:PROCESS_BUFFER_SIZE]
                buffer = buffer[PROCESS_BUFFER_SIZE:]

                # Convert byte data to numpy array
                audio_samples = np.frombuffer(block, dtype=np.int16).astype(np.float32)

                # Apply volume control
                audio_samples *= VOLUME
                audio_samples = np.clip(audio_samples, -32768, 32767)  # Prevent clipping

                # Normalize audio for librosa
                audio_normalized = audio_samples / 32768.0

                # Calculate number of semitones for pitch shift
                if PITCH != 1.0:
                    n_steps = 12 * np.log2(PITCH)
                else:
                    n_steps = 0

                # Apply pitch control if needed
                if n_steps != 0:
                    try:
                        audio_shifted = librosa.effects.pitch_shift(y=audio_normalized, sr=RATE, n_steps=n_steps)
                        # Ensure the audio is still in the correct range
                        audio_shifted = np.clip(audio_shifted, -1.0, 1.0)
                        # Convert back to int16
                        processed_samples = (audio_shifted * 32768).astype(np.int16).tobytes()
                    except TypeError as te:
                        print(f"TypeError during pitch shifting: {te}")
                        print("Check the number and type of arguments passed to pitch_shift.")
                        # Fallback to volume-adjusted samples
                        processed_samples = audio_samples.astype(np.int16).tobytes()
                    except Exception as e:
                        print(f"Error during pitch shifting: {e}")
                        # Fallback to volume-adjusted samples
                        processed_samples = audio_samples.astype(np.int16).tobytes()
                else:
                    # No pitch shifting needed
                    processed_samples = audio_samples.astype(np.int16).tobytes()

                # Play the processed audio data using ALSA
                audio_out.write(processed_samples)

        # Flush remaining samples
        if buffer:
            audio_samples = np.frombuffer(buffer, dtype=np.int16).astype(np.float32)
            audio_samples *= VOLUME
            audio_samples = np.clip(audio_samples, -32768, 32767)

            if PITCH != 1.0:
                n_steps = 12 * np.log2(PITCH)
                try:
                    audio_normalized = audio_samples / 32768.0
                    audio_shifted = librosa.effects.pitch_shift(y=audio_normalized, sr=RATE, n_steps=n_steps)
                    audio_shifted = np.clip(audio_shifted, -1.0, 1.0)
                    processed_samples = (audio_shifted * 32768).astype(np.int16).tobytes()
                except TypeError as te:
                    print(f"TypeError during pitch shifting: {te}")
                    print("Check the number and type of arguments passed to pitch_shift.")
                    processed_samples = audio_samples.astype(np.int16).tobytes()
                except Exception as e:
                    print(f"Error during pitch shifting: {e}")
                    processed_samples = audio_samples.astype(np.int16).tobytes()
            else:
                processed_samples = audio_samples.astype(np.int16).tobytes()

            audio_out.write(processed_samples)

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
