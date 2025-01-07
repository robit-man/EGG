import socket
import time

# Configuration
TARGET_HOST = "127.0.0.1"  # Server's IP
TARGET_PORT = 6434         # Port to send text content

def connect_to_server():
    """Attempt to connect to the server."""
    while True:
        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((TARGET_HOST, TARGET_PORT))
            print(f"Connected to server at {TARGET_HOST}:{TARGET_PORT}")
            return client_socket
        except ConnectionRefusedError:
            print(f"Could not connect to server at {TARGET_HOST}:{TARGET_PORT}. Retrying in 5 seconds...")
            time.sleep(5)

def main():
    while True:
        client_socket = connect_to_server()
        try:
            while True:
                # Get text input from the user
                text = input("Enter text to synthesize (or 'exit' to quit): ").strip()

                if text.lower() == 'exit':
                    print("Exiting...")
                    client_socket.close()
                    return

                # Send the text to the server
                client_socket.sendall(text.encode("utf-8"))
                print(f"Sent: {text}")

        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            print("Disconnected from server. Reconnecting...")
            time.sleep(5)

        except Exception as e:
            print(f"An error occurred: {e}. Reconnecting...")
            time.sleep(5)

if __name__ == "__main__":
    main()
