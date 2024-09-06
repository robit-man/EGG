import argparse
import asyncio
import riva.client
import riva.client.audio_io
from copy import deepcopy
from difflib import SequenceMatcher
import pyaudio
import termcolor

def list_all_devices(selected_input_device_index):
    """List all available audio input and output devices, highlighting the selected input device."""
    p = pyaudio.PyAudio()
    num_devices = p.get_device_count()

    termcolor.cprint("Available Input Devices:", "yellow")
    for i in range(num_devices):
        device_info = p.get_device_info_by_index(i)
        if device_info['maxInputChannels'] > 0:
            if i == selected_input_device_index:
                termcolor.cprint(f"  [Selected Input] Device {i}:", "cyan")
                termcolor.cprint(f"    Name: {device_info['name']}", "cyan")
                termcolor.cprint(f"    Max Input Channels: {device_info['maxInputChannels']}", "cyan")
                termcolor.cprint(f"    Default Sample Rate: {device_info['defaultSampleRate']}", "cyan")
            else:
                termcolor.cprint(f"  Device {i}:", "magenta")
                termcolor.cprint(f"    Name: {device_info['name']}", "red")
                termcolor.cprint(f"    Max Input Channels: {device_info['maxInputChannels']}", "red")
                termcolor.cprint(f"    Default Sample Rate: {device_info['defaultSampleRate']}", "red")

    termcolor.cprint("\nAvailable Output Devices:", "yellow")
    for i in range(num_devices):
        device_info = p.get_device_info_by_index(i)
        if device_info['maxOutputChannels'] > 0:
            termcolor.cprint(f"  Device {i}:", "magenta")
            termcolor.cprint(f"    Name: {device_info['name']}", "red")
            termcolor.cprint(f"    Max Output Channels: {device_info['maxOutputChannels']}", "red")
            termcolor.cprint(f"    Default Sample Rate: {device_info['defaultSampleRate']}", "red")

    p.terminate()

async def send_tokens(writer, message):
    """Send tokens to the connected server."""
    tokens = message.split()
    print("\nStreaming tokens to the server...")

    for token in tokens:
        writer.write(f"{token} ".encode())
        await writer.drain()
        print(f"Sent token: {token}")
        await asyncio.sleep(0.1)  # Brief delay to simulate streaming

    # Send end of token stream marker
    writer.write("<eot>".encode())
    await writer.drain()
    print("Token stream complete. Sent <eot> marker.\n")

async def listen_for_responses(reader):
    """Listen for and print responses from the server."""
    try:
        while True:
            response = await reader.read(4096)
            if not response:
                print("Server closed the connection.")
                break
            print("Received response from server:", response.decode())
    except asyncio.CancelledError:
        pass  # Handle the case where the task is cancelled
    except Exception as e:
        print(f"Error listening for responses: {e}")

def is_similar(text1, text2, threshold=0.8):
    """Check if two strings are similar based on a similarity threshold."""
    return SequenceMatcher(None, text1, text2).ratio() > threshold

async def connect_and_send_message(ip, port, asr_service, streaming_config, input_device):
    """Connect to the server and send messages using ASR input."""
    reader, writer = None, None
    listen_task = None
    last_final_phrase = ""  # Store the last recognized final phrase

    try:
        while True:
            try:
                if writer is None or writer.is_closing():
                    reader, writer = await asyncio.open_connection(ip, port)
                    print(f"Connected to {ip}:{port}\n")

                    if listen_task:
                        listen_task.cancel()  # Cancel the previous listening task if it exists

                    # Start a new task to listen for responses from the server
                    listen_task = asyncio.create_task(listen_for_responses(reader))

                # Start streaming from the microphone
                with riva.client.audio_io.MicrophoneStream(
                    rate=16000,
                    chunk=16000 // 10,  # Chunk size (16000 Hz // 10)
                    device=input_device,
                ) as audio_chunk_iterator:

                    # ASR processing
                    response_generator = asr_service.streaming_response_generator(
                        audio_chunks=audio_chunk_iterator,
                        streaming_config=streaming_config,
                    )

                    for response in response_generator:
                        for result in response.results:
                            # Print each recognized token as it comes in (interim or final)
                            recognized_text = result.alternatives[0].transcript
                            print(f"Recognized (partial/final): {recognized_text}")

                            # Send the recognized text to the server
                            if result.is_final:
                                # Check if the current phrase is similar to the last final phrase
                                if not is_similar(recognized_text, last_final_phrase):
                                    await send_tokens(writer, recognized_text)
                                    last_final_phrase = recognized_text  # Update last final phrase
                                else:
                                    print("Detected similar phrase, ignoring...")

                                # Handle exit condition
                                if recognized_text.strip().lower() == 'exit':
                                    print("Closing connection...")
                                    writer.write("<eot>".encode())
                                    await writer.drain()
                                    return

            except Exception as e:
                print(f"Connection error: {e}. Attempting to reconnect...")
                await asyncio.sleep(5)  # Wait before trying to reconnect

        if listen_task:
            listen_task.cancel()  # Ensure we cancel the listening task when exiting

        if writer:
            writer.close()
            await writer.wait_closed()

    except asyncio.CancelledError:
        if writer:
            writer.close()
            await writer.wait_closed()
        if listen_task:
            listen_task.cancel()

def main():
    parser = argparse.ArgumentParser(description="Connect to a server and send tokens using ASR")
    parser.add_argument('--ip', type=str, default='127.0.0.1', help="IP address of the server (default: 127.0.0.1)")
    parser.add_argument('--port', type=int, default=6200, help="Port to connect to (default: 6200)")
    parser.add_argument("--input-device", type=int, default=0, help="An input audio device to use.")
    parser.add_argument("--server", type=str, default="localhost:50051", help="Riva server address (default: localhost:50051).")
    parser.add_argument("--use-ssl", action="store_true", help="Use SSL to connect to the server.")
    parser.add_argument("--ssl-cert", type=str, help="Path to SSL certificate.")
    args = parser.parse_args()

    # List all devices and highlight the selected input device
    list_all_devices(args.input_device)

    # Authenticate with the Riva server
    auth = riva.client.Auth(uri=args.server, use_ssl=args.use_ssl, ssl_cert=args.ssl_cert)

    # Set up ASR service
    asr_service = riva.client.ASRService(auth)

    # Configure ASR recognition
    offline_config = riva.client.RecognitionConfig(
        encoding=riva.client.AudioEncoding.LINEAR_PCM,
        sample_rate_hertz=16000,  # Set the sample rate to 16000 Hz
        max_alternatives=1,
        enable_automatic_punctuation=True,
        verbatim_transcripts=True,
    )
    streaming_config = riva.client.StreamingRecognitionConfig(
        config=deepcopy(offline_config), interim_results=True
    )

    # Run the connection and message sending process
    asyncio.run(connect_and_send_message(args.ip, args.port, asr_service, streaming_config, args.input_device))

if __name__ == "__main__":
    main()
