import os
import socket
import subprocess
import time

# Configuration
LISTEN_PORT = 6434  # Port to listen for text content
RAW_AUDIO_PORT = 6353  # Port to send raw audio data
PIPER_EXECUTABLE = "/opt/piper/build/piper"  # Full path to Piper executable
PIPER_MODEL_PATH = "/opt/voice/glados_piper_medium.onnx"

def handle_client_connection(client_socket, audio_socket):
    try:
        while True:
            # Receive text content from the client
            data = client_socket.recv(1024).decode("utf-8").strip()
            if not data:
                break

            print(f"Received text: {data}")

            # Run the Piper command with raw audio output
            process = subprocess.Popen(
                [
                    PIPER_EXECUTABLE,
                    "--model", PIPER_MODEL_PATH,
                    "--output_raw"
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            # Write the received text to Piper's stdin
            stdout, stderr = process.communicate(input=data.encode("utf-8"))

            if stderr:
                print(f"Piper error: {stderr.decode('utf-8')}")

            # Forward raw audio to the parallel port
            audio_socket.sendall(stdout)
            print("Raw audio forwarded to audio socket.")

    except Exception as e:
        print(f"Error handling client: {e}")

def main():
    try:
        # Create a socket to listen for text content
        text_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        text_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        text_server.bind(("0.0.0.0", LISTEN_PORT))
        text_server.listen(1)
        print(f"Listening for text content on port {LISTEN_PORT}...")

        # Create a socket to send raw audio
        audio_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        while True:
            try:
                audio_socket.connect(("127.0.0.1", RAW_AUDIO_PORT))
                print(f"Connected to raw audio port {RAW_AUDIO_PORT}...")
                break
            except ConnectionRefusedError:
                print(f"Raw audio port {RAW_AUDIO_PORT} not available. Retrying in 5 seconds...")
                time.sleep(5)

        while True:
            client_socket, addr = text_server.accept()
            print(f"Connection established with {addr}")
            handle_client_connection(client_socket, audio_socket)
            client_socket.close()

    except KeyboardInterrupt:
        print("Shutting down server...")

    except Exception as e:
        print(f"Server error: {e}")

    finally:
        text_server.close()
        audio_socket.close()
        print("Server closed.")

if __name__ == "__main__":
    main()
