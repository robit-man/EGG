import sounddevice as sd
import numpy as np
import riva.client
import asyncio
import argparse
from copy import deepcopy
import grpc
from riva.client.proto import riva_tts_pb2, riva_tts_pb2_grpc
import threading
import queue
import pyaudio
from riva.client.argparse_utils import add_connection_argparse_parameters

# Arguments setup for Riva
def parse_args():
    parser = argparse.ArgumentParser(
        description="Riva ASR and TTS Integration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser = add_connection_argparse_parameters(parser)
    parser.add_argument(
        "--port", type=int, default=6100, help="Port to listen on. Defaults to 6100."
    )
    parser.add_argument(
        "-o", "--output", type=str, help="Output .wav file to write synthesized audio."
    )
    parser.add_argument(
        "--voice",
        type=str,
        default=None,
        help="A voice name to use for TTS."
    )
    parser.add_argument(
        "--language-code",
        type=str,
        default="en-US",
        help="Language code for TTS. Default is 'en-US'."
    )
    parser.add_argument(
        "--interrupt",
        action="store_true",
        default=True,
        help="Whether to interrupt the current TTS when new tokens arrive. Defaults to True."
    )
    parser.add_argument(
        "--input-device",
        type=int,
        default=None,
        help="An input audio device to use."
    )
    return parser.parse_args()

args = parse_args()

# Set the sample rate and channels for ASR and TTS
sample_rate = 16000  # Riva ASR requires 16kHz sample rate
input_channels = 1   # Mono audio for ASR input
output_channels = 1  # Mono audio for TTS output

# Print information about input and output devices
input_device_info = sd.query_devices(args.input_device or sd.default.device[0], kind='input')
output_device_info = sd.query_devices(kind='output')
print("Input Device Info:")
print(input_device_info)
print("\nOutput Device Info:")
print(output_device_info)

# Riva Authentication and Service Setup
auth = riva.client.Auth(uri=args.server, use_ssl=args.use_ssl, ssl_cert=args.ssl_cert)

# Initialize ASR service
try:
    asr_service = riva.client.ASRService(auth)
    print("ASR Service initialized successfully.")
except Exception as e:
    print(f"Failed to initialize ASR Service: {e}")
    exit(1)

# Global Variables
tts_queue = queue.Queue()
cancel_event = threading.Event()
current_tts_thread = None

# Configure ASR Recognition
language_code = args.language_code
offline_config = riva.client.RecognitionConfig(
    encoding=riva.client.AudioEncoding.LINEAR_PCM,
    sample_rate_hertz=sample_rate,
    max_alternatives=1,
    enable_automatic_punctuation=True,
    verbatim_transcripts=True,
    language_code=language_code
)
streaming_config = riva.client.StreamingRecognitionConfig(
    config=deepcopy(offline_config), interim_results=True
)

# Run Riva ASR using the functional component
async def run_riva_asr():
    print("Starting ASR using Riva...")
    try:
        with sd.InputStream(
            samplerate=sample_rate,
            channels=input_channels,
            dtype='int16',
            device=args.input_device
        ) as stream:
            print("Microphone input stream opened successfully.")
            audio_chunk_size = int(sample_rate / 10)  # 100ms chunks
            audio_chunk_iterator = iter(
                lambda: stream.read(audio_chunk_size)[0].tobytes(), b''
            )

            # ASR processing using Riva streaming response generator
            response_generator = asr_service.streaming_response_generator(
                audio_chunks=audio_chunk_iterator,
                streaming_config=streaming_config,
            )

            for response in response_generator:
                for result in response.results:
                    if result.is_final:
                        recognized_text = result.alternatives[0].transcript
                        print(f"Recognized (final): {recognized_text}")
                        tts_queue.put(recognized_text)

    except Exception as e:
        print(f"Error during ASR streaming: {e}")

# Generate TTS and Playback
def tts_generation_and_playback(tts_stub, selected_voice):
    # Set up audio output stream
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=output_channels,
        rate=sample_rate,
        output=True
    )

    while not cancel_event.is_set():
        try:
            text = tts_queue.get(timeout=1)
        except queue.Empty:
            continue

        if text:
            print(f"Generating TTS for: {text}")
            req = riva_tts_pb2.SynthesizeSpeechRequest(
                text=text,
                language_code=selected_voice['language'],
                encoding=riva.client.AudioEncoding.LINEAR_PCM,
                sample_rate_hz=sample_rate,
                voice_name=selected_voice['name']
            )
            try:
                resp = tts_stub.Synthesize(req)
                audio_samples = np.frombuffer(resp.audio, dtype=np.int16)
                stream.write(audio_samples.tobytes())
            except Exception as e:
                print(f"TTS generation failed: {e}")

    stream.stop_stream()
    stream.close()
    p.terminate()

# Start TTS Thread
def start_tts_thread(tts_stub, selected_voice):
    global current_tts_thread
    if current_tts_thread and current_tts_thread.is_alive():
        cancel_event.set()
        current_tts_thread.join()

    cancel_event.clear()
    current_tts_thread = threading.Thread(target=tts_generation_and_playback, args=(tts_stub, selected_voice))
    current_tts_thread.start()

# Get Available TTS Voices
def get_available_voices(tts_stub):
    print("Retrieving available voices from Riva TTS...")

    # Create the request to get synthesis configuration
    request = riva_tts_pb2.RivaSynthesisConfigRequest()

    # Call the GetRivaSynthesisConfig RPC
    try:
        response = tts_stub.GetRivaSynthesisConfig(request)
    except Exception as e:
        print(f"Failed to get TTS synthesis config: {e}")
        return []

    # Parse the response to find available voices
    available_voices = []
    if response.model_config:
        print("Available TTS Models:")
        for model in response.model_config:
            voice_name = model.parameters.get("voice_name", "")
            language_code = model.parameters.get("language_code", "")

            if voice_name:
                available_voices.append({"name": voice_name, "language": language_code})
                print(f"Voice: {voice_name}, Language: {language_code}")

    if not available_voices:
        print("No available voices found in the response.")

    return available_voices

# Main Function to Start ASR and TTS
async def main():
    # Initialize TTS stub
    try:
        if args.use_ssl and args.ssl_cert:
            with open(args.ssl_cert, 'rb') as f:
                creds = grpc.ssl_channel_credentials(f.read())
            channel = grpc.secure_channel(args.server, creds)
        else:
            channel = grpc.insecure_channel(args.server)

        tts_stub = riva_tts_pb2_grpc.RivaSpeechSynthesisStub(channel)
        print("TTS stub initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize TTS stub: {e}")
        return

    # Get available voices
    available_voices = get_available_voices(tts_stub)
    if not available_voices:
        print("No available voices.")
        return

    # Let user select a voice
    print("\nAvailable voices:")
    for idx, voice in enumerate(available_voices):
        print(f"{idx + 1}: {voice['name']} (Language: {voice['language']})")

    selected_voice = None

    # If voice is specified via args.voice, try to use it
    if args.voice:
        # Check if voice exists
        matching_voices = [v for v in available_voices if v['name'] == args.voice]
        if matching_voices:
            selected_voice = matching_voices[0]
        else:
            print(f"Voice '{args.voice}' not found. Please select from the available voices.")

    # If no valid voice selected yet, prompt user to select
    while not selected_voice:
        user_input = input("Select a voice by number or name: ").strip()
        # Try to interpret input as a number
        try:
            idx = int(user_input) - 1
            if 0 <= idx < len(available_voices):
                selected_voice = available_voices[idx]
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            # Not a number, treat as name
            matching_voices = [v for v in available_voices if v['name'] == user_input]
            if matching_voices:
                selected_voice = matching_voices[0]
            else:
                print("Voice not found. Please try again.")

    print(f"Selected Voice: {selected_voice['name']} (Language: {selected_voice['language']})")

    # Start TTS thread with selected voice
    start_tts_thread(tts_stub, selected_voice)

    # Run ASR
    try:
        await run_riva_asr()
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        cancel_event.set()
        if current_tts_thread and current_tts_thread.is_alive():
            current_tts_thread.join()

# Run Main
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Program terminated by user.")
