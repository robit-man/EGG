import os
import socket
import subprocess
import time
import json
import re
from queue import Queue
from threading import Thread, Lock

# Configuration
LISTEN_PORT = 6434  # Port to listen for text content
RAW_AUDIO_PORT = 6353  # Port to send raw audio data
PIPER_EXECUTABLE = "/opt/piper/build/piper"  # Full path to Piper executable
PIPER_MODEL_PATH = "/opt/voice/glados_piper_medium.onnx"

def tts_worker(queue, audio_socket, lock):
    """Worker thread for processing TTS queue"""
    while True:
        text_content = queue.get()
        if text_content is None:
            break

        with lock:
            print(f"Processing text: {text_content}")
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

            stdout, stderr = process.communicate(input=text_content.encode("utf-8"))

            if stderr:
                print(f"Piper error: {stderr.decode('utf-8')}")

            audio_socket.sendall(stdout)
            print("Raw audio forwarded to audio socket.")

        queue.task_done()

def handle_client_connection(client_socket, queue):
    try:
        while True:
            # Receive text content from the client
            data = client_socket.recv(4096).decode("utf-8").strip()
            if not data:
                break

            print(f"Received raw data: {data}")

            try:
                json_match = re.search(r'\{.*\}', data, re.DOTALL)
                if json_match:
                    json_data = json.loads(json_match.group())
                    text_content = json_data.get("response") or json_data.get("prompt", "")
                    if not text_content:
                        print("No usable text field ('response' or 'prompt') found in JSON.")
                        continue
                    print(f"Extracted text: {text_content}")
                    queue.put(text_content)  # Add text to the queue
                else:
                    print("No JSON found in the received data.")
                    continue
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Error parsing JSON: {e}")
                continue

    except Exception as e:
        print(f"Error handling client: {e}")

def main():
    try:
        # Create a queue for managing text-to-speech tasks
        tts_queue = Queue()
        lock = Lock()

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

        # Start the TTS worker thread
        worker_thread = Thread(target=tts_worker, args=(tts_queue, audio_socket, lock), daemon=True)
        worker_thread.start()

        while True:
            client_socket, addr = text_server.accept()
            print(f"Connection established with {addr}")
            handle_client_connection(client_socket, tts_queue)
            client_socket.close()

    except KeyboardInterrupt:
        print("Shutting down server...")

    except Exception as e:
        print(f"Server error: {e}")

    finally:
        tts_queue.put(None)  # Signal the worker thread to exit
        tts_queue.join()
        text_server.close()
        audio_socket.close()
        print("Server closed.")

if __name__ == "__main__":
    main()
