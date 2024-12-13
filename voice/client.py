#!/usr/bin/env python3
import sys
import requests
import subprocess

def synthesize_and_play(prompt, server_url="http://localhost:61637/synthesize"):
    try:
        # Define the JSON payload
        payload = {
            "prompt": prompt
        }

        # Send POST request with streaming response
        with requests.post(server_url, json=payload, stream=True) as response:
            if response.status_code != 200:
                print(f"Error: Received status code {response.status_code}")
                try:
                    error_msg = response.json().get('error', 'No error message provided.')
                    print(f"Error message: {error_msg}")
                except:
                    print("No JSON error message provided.")
                return

            # Initialize subprocess to pipe data to aplay
            # aplay expects raw PCM data with specific parameters
            aplay = subprocess.Popen(
                ['aplay', '-r', '22050', '-f', 'S16_LE', '-t', 'raw'],
                stdin=subprocess.PIPE
            )

            # Stream data to aplay
            try:
                for chunk in response.iter_content(chunk_size=4096):
                    if chunk:
                        aplay.stdin.write(chunk)
            except BrokenPipeError:
                print("Warning: aplay subprocess terminated unexpectedly.")
            finally:
                aplay.stdin.close()
                aplay.wait()

    except requests.exceptions.ConnectionError:
        print("Error: Unable to connect to the TTS server. Ensure that 'server.py' is running.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def main():
    print("Piper TTS Client")
    print("Type your text and press Enter to synthesize and play.")
    print("Type 'exit' or 'quit' to terminate the client.\n")

    while True:
        try:
            prompt = input("Enter text to synthesize: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting Piper TTS Client.")
            break

        if prompt.lower() in ['exit', 'quit']:
            print("Exiting Piper TTS Client.")
            break

        if not prompt:
            print("Error: Empty prompt provided. Please enter some text.")
            continue

        synthesize_and_play(prompt)

if __name__ == "__main__":
    main()
