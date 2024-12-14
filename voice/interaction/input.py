#!/usr/bin/env python3
import socket
import sys
import threading

HOST = 'localhost'  # or the hostname/IP where the server runs
PORT = 64162        # must match the port in the provided server script

# Lock to synchronize console output
print_lock = threading.Lock()

def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)

def send_and_receive(prompt):
    """
    Handles sending the prompt to the server and receiving the response.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            s.sendall(prompt.encode('utf-8'))
            # Receive the response from the server
            response = b""
            while True:
                part = s.recv(4096)
                if not part:
                    break
                response += part
            if response:
                safe_print("\nServer Response:")
                safe_print(response.decode('utf-8'))
                safe_print("Enter text to send: ", end='', flush=True)
            else:
                safe_print("\nNo response received from the server.")
                safe_print("Enter text to send: ", end='', flush=True)
    except ConnectionRefusedError:
        safe_print("\nError: Unable to connect to the server. Ensure that the server is running.")
        safe_print("Enter text to send: ", end='', flush=True)
    except Exception as e:
        safe_print(f"\nAn unexpected error occurred: {e}")
        safe_print("Enter text to send: ", end='', flush=True)

def input_thread(stop_event):
    """
    Thread to handle user input.
    """
    while not stop_event.is_set():
        try:
            prompt = input("Enter text to send: ").strip()
        except (EOFError, KeyboardInterrupt):
            safe_print("\nExiting Downstream Client.")
            stop_event.set()
            break

        if prompt.lower() in ['exit', 'quit']:
            safe_print("Exiting Downstream Client.")
            stop_event.set()
            break

        if not prompt:
            safe_print("Error: Empty prompt provided. Please enter some text.")
            continue

        # Start a new thread for each send-receive operation
        threading.Thread(target=send_and_receive, args=(prompt,), daemon=True).start()

def main():
    safe_print("Downstream Client")
    safe_print("Type your text and press Enter to send it to the server.")
    safe_print("Type 'exit' or 'quit' to terminate the client.\n")

    stop_event = threading.Event()
    thread = threading.Thread(target=input_thread, args=(stop_event,))
    thread.start()

    try:
        while thread.is_alive():
            thread.join(timeout=1.0)
    except KeyboardInterrupt:
        safe_print("\nExiting Downstream Client.")
        stop_event.set()
        thread.join()

if __name__ == "__main__":
    main()
