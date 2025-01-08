import os
import subprocess
import sys
import socket
import time
import numpy as np

# Configuration
VENV_DIR = "output_venv"  # Virtual environment directory
LISTEN_HOST = "0.0.0.0"   # Listen on all interfaces
LISTEN_PORT = 6353        # Port to receive raw audio data (matches RAW_AUDIO_PORT in voice_server.py)

# ALSA Playback configuration
PCM_DEVICE = "default"  # ALSA device for playback
CHUNK = 4096            # Buffer size (in bytes). Increased for better processing.
RATE = 22050            # Ensure this matches the output format of piper
CHANNELS = 1

# Audio Processing Configuration
VOLUME = 0.5  # Volume control factor (1.0 = original volume). Adjust as needed
PITCH = 1.0   # Pitch control factor (1.0 = original pitch). Adjust as needed

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
    print("Installing dependencies in the virtual environment...")
    subprocess.run([pip_executable, "install", "--upgrade", "pip"], check=True)
    subprocess.run([pip_executable, "install", "pyalsaaudio", "numpy"], check=True)
    print("Please install pysoundtouch manually by following the provided instructions.")

def activate_venv():
    """Activate the virtual environment by modifying sys.path."""
    venv_site_packages = os.path.join(VENV_DIR, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages")
    sys.path.insert(0, venv_site_packages)

def relaunch_in_venv():
    """Relaunch the current script within the virtual environment."""
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
    """Handle the incoming client connection and play audio with volume and optional pitch control."""
    try:
        import alsaaudio
        # Attempt to import pysoundtouch
        try:
            from pysoundtouch import SoundTouch
            soundtouch_available = True
            print("pysoundtouch successfully imported. Pitch shifting will be applied.")
        except ImportError:
            soundtouch_available = False
            print("pysoundtouch not found. Pitch shifting will be disabled.")

        if soundtouch_available and PITCH != 1.0:
            # Initialize SoundTouch for pitch shifting
            st = SoundTouch(RATE, CHANNELS)
            st.set_pitch(PITCH)
            st.set_speed(1.0)  # Keep speed constant
            st.set_rate(1.0)    # Keep rate constant

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

            # Process in larger blocks (e.g., 16384 bytes)
            if len(buffer) >= 16384:
                # Convert byte data to numpy array
                audio_samples = np.frombuffer(buffer, dtype=np.int16).astype(np.float32)

                # Apply volume control
                audio_samples *= VOLUME
                audio_samples = np.clip(audio_samples, -32768, 32767)  # Prevent clipping

                # Convert back to int16 after volume adjustment
                processed_samples = audio_samples.astype(np.int16).tobytes()

                # Apply pitch shifting if available
                if soundtouch_available and PITCH != 1.0:
                    st.put_samples(processed_samples)
                    shifted_samples = st.receive_samples(len(processed_samples))
                    if shifted_samples:
                        audio_out.write(shifted_samples)
                else:
                    # Play the processed audio data using ALSA
                    audio_out.write(processed_samples)

                # Clear the buffer
                buffer = b""

        # Flush remaining samples
        if buffer:
            audio_samples = np.frombuffer(buffer, dtype=np.int16).astype(np.float32)
            audio_samples *= VOLUME
            audio_samples = np.clip(audio_samples, -32768, 32767)
            processed_samples = audio_samples.astype(np.int16).tobytes()

            if soundtouch_available and PITCH != 1.0:
                st.put_samples(processed_samples)
                while True:
                    shifted_samples = st.receive_samples(len(processed_samples))
                    if not shifted_samples:
                        break
                    audio_out.write(shifted_samples)
            else:
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
