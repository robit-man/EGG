#!/usr/bin/env python3
import socket
import sys

HOST = 'localhost'  # or the hostname/IP where the server runs
PORT = 64162        # must match the port in the provided script

def main():
    print("Downstream Client")
    print("Type your text and press Enter to send it to the server.")
    print("Type 'exit' or 'quit' to terminate the client.\n")

    while True:
        try:
            prompt = input("Enter text to send: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting Downstream Client.")
            break

        if prompt.lower() in ['exit', 'quit']:
            print("Exiting Downstream Client.")
            break

        if not prompt:
            print("Error: Empty prompt provided. Please enter some text.")
            continue

        # Send the prompt to the server
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((HOST, PORT))
                s.sendall(prompt.encode('utf-8'))
                # Receive the response from the server
                response = s.recv(65536)
                if response:
                    print("Server Response:")
                    print(response.decode('utf-8'))
                else:
                    print("No response received from the server.")
        except ConnectionRefusedError:
            print("Error: Unable to connect to the server. Ensure that the server is running.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
