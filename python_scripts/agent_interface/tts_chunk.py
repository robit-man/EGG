import argparse
import asyncio
import termcolor
import wave
import datetime
import time
import numpy as np
import riva.client
import riva.client.audio_io
import pyaudio
from riva.client.argparse_utils import add_connection_argparse_parameters
import re
import threading
from queue import Queue, Empty

# Global variables for managing tasks
cancel_event = threading.Event()
stop_playback_event = threading.Event()
queue_lock = threading.Lock()
tts_queue = Queue()
chunk_counter = 0  # Counter for chunks, reset with each new generation
current_tts_thread = None  # To track the TTS thread

# Buffer to accumulate incoming tokens
token_buffer = ""
eot_received = threading.Event()
session_active = False
playback_started = threading.Event()

def custom_num2words(number):
    if number == 0:
        return "zero"
    if number == '%':
        return "percent"
    if number == '&':
        return "and"
    
    ones = [
        "", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"
    ]
    teens = [
        "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen"
    ]
    tens = [
        "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"
    ]
    thousands = [
        "", "thousand", "million", "billion", "trillion", "quadrillion", "quintillion"
    ]

    def words_999(n):
        if n < 10:
            return ones[n]
        elif n < 20:
            return teens[n - 10]
        elif n < 100:
            return tens[n // 10] + ("-" + ones[n % 10] if (n % 10 != 0) else "")
        else:
            return ones[n // 100] + " hundred" + (" and " + words_999(n % 100) if (n % 100 != 0) else "")

    def words_large(n):
        if n < 1000:
            return words_999(n)
        else:
            for i, word in enumerate(thousands):
                if n < 1000 ** (i + 1):
                    break
            return words_large(n // (1000 ** i)) + " " + word + (", " + words_large(n % (1000 ** i)) if (n % (1000 ** i)) != 0 else "")

    return words_large(number)

def convert_numbers_to_text(text):
    def replacer(match):
        number = int(match.group(0))
        return custom_num2words(number)

    return re.sub(r'\b\d+\b', replacer, text)

def stretch_audio(audio, factor):
    """Stretch audio by a given factor using linear interpolation."""
    stretched_audio = np.interp(
        np.linspace(0, len(audio), int(len(audio) * factor), endpoint=False),
        np.arange(len(audio)),
        audio
    )
    return stretched_audio.astype(np.int16)

def split_text_into_chunks(text, max_chars=399):
    """Splits text into chunks without exceeding max_chars, ensuring
    chunks end at the last sentence delimiter or word boundary before the max_chars limit."""
    
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())  # Split by sentence boundaries
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_chars:
            # If adding this sentence won't exceed the max length, add it to the current chunk.
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
        else:
            # If adding this sentence would exceed the max length, finalize the current chunk.
            if len(sentence) > max_chars:
                # Split the sentence if it's too long
                while len(sentence) > max_chars:
                    sub_chunk = sentence[:max_chars]
                    cut_index = sub_chunk.rfind(' ')
                    if cut_index == -1:
                        cut_index = max_chars  # Force cut if no space found
                    chunks.append(sub_chunk[:cut_index].strip())
                    sentence = sentence[cut_index:].strip()
                if sentence:
                    current_chunk = sentence
            else:
                # Finalize the current chunk and start a new one with the current sentence
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence

    # Add any remaining text as the last chunk.
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def print_selected_device_info(p, device_index):
    """Print detailed information about the selected output device."""
    if device_index is None:
        device_info = p.get_default_output_device_info()
        termcolor.cprint("Selected Output Device: System Default", "cyan")
    else:
        device_info = p.get_device_info_by_index(device_index)
    
    termcolor.cprint("Selected Output Device Information:", "cyan")
    termcolor.cprint(f"  Name: {device_info['name']}", "cyan")
    termcolor.cprint(f"  Max Output Channels: {device_info['maxOutputChannels']}", "cyan")
    termcolor.cprint(f"  Default Sample Rate: {device_info['defaultSampleRate']}", "cyan")

    return int(device_info['defaultSampleRate'])

async def handle_client(reader, writer, tts_service, tts_voice, language_code, wav_out, output_device_index, resample_rate, interrupt):
    """Handles incoming connections, cancels ongoing TTS, and starts new TTS tasks."""
    global token_buffer, eot_received, session_active, current_tts_thread

    termcolor.cprint("\nConnection established", "green")
    eot_received.clear()  # Reset the end-of-token event
    session_active = False  # Mark the session as inactive

    while True:
        data = await reader.read(100)  # Read incoming data
        if not data:
            break

        # Capture the time right after data is received
        received_time = time.time()

        message = data.decode('utf-8')
        token_buffer += message  # Accumulate the text

        # Calculate the time delta when printing the cyan line
        printed_time = time.time()
        time_delta = printed_time - received_time

        # Print a cyan line with a timestamp and time delta as soon as new tokens are received
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        termcolor.cprint(f"[{timestamp}] New tokens received from client (Time delta: {time_delta:.6f} seconds)", "cyan")

        if "<eot>" in message:  # End of token stream detected
            token_buffer = token_buffer.replace("<eot>", "")
            eot_received.set()  # Signal that end-of-token is received
            session_active = True  # Mark the session as active
            termcolor.cprint("End of token stream detected. Session is now active.", "yellow")
            break  # Stop further reading

    if session_active and interrupt and playback_started.is_set():
        # Interrupt the current session if it's active, playback has started, and new tokens have arrived
        termcolor.cprint("Interrupt: Current TTS session cancelled due to new tokens.", "yellow")
        cancel_event.set()  # Signal cancellation to stop current TTS generation
        stop_playback_event.set()  # Stop current playback

        if current_tts_thread and current_tts_thread.is_alive():
            current_tts_thread.join()  # Wait for the TTS thread to finish

        with queue_lock:
            tts_queue.queue.clear()  # Clear the TTS queue

        # Reset flags to allow the new session to start
        cancel_event.clear()  # Clear the cancel event for the new session
        stop_playback_event.clear()  # Clear the stop playback event
        session_active = False  # Reset the session flag
        playback_started.clear()  # Reset playback started flag
        termcolor.cprint("TTS session has been reset and is ready for new tokens.", "yellow")

        # Process new tokens immediately
        process_tokens_into_chunks()
        start_tts_thread(tts_service, tts_voice, language_code, output_device_index, resample_rate, wav_out)

    writer.close()
    await writer.wait_closed()

def process_tokens_into_chunks():
    """Process the accumulated tokens into chunks and add them to the TTS queue."""
    global token_buffer, tts_queue, session_active

    while not cancel_event.is_set():
        if eot_received.is_set():
            with queue_lock:
                chunks = split_text_into_chunks(token_buffer, max_chars=399)
                for chunk in chunks:
                    if chunk.strip():
                        termcolor.cprint(f"Passing chunk to TTS generation: {chunk}", "green")
                        tts_queue.put(chunk.strip())  # Add the chunk to the queue
            token_buffer = ""  # Clear the token buffer
            eot_received.clear()  # Reset the end-of-token event
            session_active = True  # Ensure session is marked active after processing
            termcolor.cprint("Token buffer processed into chunks and added to TTS queue.", "yellow")
            break

        time.sleep(0.1)  # Small sleep to prevent busy-waiting

def tts_generation_and_playback(tts_service, tts_voice, language_code, output_device_index, resample_rate, wav_out=None):
    """Threaded task for generating and playing TTS sequentially from the queue."""
    global cancel_event, stop_playback_event, chunk_counter, playback_started

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=2,
                    rate=resample_rate,
                    output=True,
                    output_device_index=output_device_index)

    try:
        while not cancel_event.is_set():
            try:
                # Get the next chunk from the queue
                text = tts_queue.get(timeout=1)  # Wait for the next chunk in the queue
            except Empty:
                continue

            if not text.strip():
                continue

            try:
                print(f"Generating TTS for chunk: {text}")  # Print the text being processed
                tts_responses = tts_service.synthesize_online(
                    text, tts_voice, language_code, sample_rate_hz=resample_rate
                )
            except Exception as e:
                termcolor.cprint(f"TTS generation failed: {e}", "red")
                break

            chunk_counter += 1
            termcolor.cprint(f"[chunk {chunk_counter}]", "red")

            playback_started.set()  # Mark playback as started

            for tts_response in tts_responses:
                if cancel_event.is_set() or stop_playback_event.is_set():
                    termcolor.cprint("TTS generation/playback was cancelled.", "red")
                    break

                audio_np = np.frombuffer(tts_response.audio, dtype=np.int16)
                stretched_audio_np = stretch_audio(audio_np, 2)
                
                # Play the audio
                stream.write(stretched_audio_np.tobytes())

                if wav_out is not None:
                    wav_out.writeframesraw(stretched_audio_np.tobytes())

            tts_queue.task_done()

        cancel_event.clear()  # Reset the event after processing
        stop_playback_event.clear()  # Reset the playback stop event
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

def start_tts_thread(tts_service, tts_voice, language_code, output_device_index, resample_rate, wav_out=None):
    """Start a new thread for TTS processing."""
    global current_tts_thread
    if current_tts_thread and current_tts_thread.is_alive():
        # Ensure the previous thread is stopped before starting a new one
        cancel_event.set()
        stop_playback_event.set()
        current_tts_thread.join()

    current_tts_thread = threading.Thread(target=tts_generation_and_playback, args=(tts_service, tts_voice, language_code, output_device_index, resample_rate, wav_out))
    current_tts_thread.daemon = True  # Ensure thread exits when the main program does
    current_tts_thread.start()
    termcolor.cprint("Started a new TTS thread.", "cyan")

def start_processing_thread():
    """Start a thread to process tokens into chunks."""
    processing_thread = threading.Thread(target=process_tokens_into_chunks)
    processing_thread.daemon = True
    processing_thread.start()
    termcolor.cprint("Started a token processing thread.", "cyan")

async def start_server(port, tts_service, tts_voice, language_code, wav_out, output_device_index, resample_rate, interrupt):
    """Starts an asyncio server to listen on the specified port."""
    server = await asyncio.start_server(
        lambda r, w: handle_client(r, w, tts_service, tts_voice, language_code, wav_out, output_device_index, resample_rate, interrupt),
        '127.0.0.1', port
    )
    addr = server.sockets[0].getsockname()
    print(f'Serving on {addr}')

    # Start the TTS processing and token processing threads
    start_tts_thread(tts_service, tts_voice, language_code, output_device_index, resample_rate, wav_out)
    start_processing_thread()

    async with server:
        await server.serve_forever()

def main():
    args = parse_args()
    
    if args.list_devices:
        riva.client.audio_io.list_output_devices()
        return
    
    auth = riva.client.Auth(uri=args.server, use_ssl=args.use_ssl, ssl_cert=args.ssl_cert)
    tts_service = riva.client.SpeechSynthesisService(auth)

    wav_out = None
    if args.output is not None:
        wav_out = wave.open(args.output, 'wb')
        wav_out.setnchannels(2)
        wav_out.setsampwidth(2)

    output_device_index = args.output_device if args.output_device is not None else None

    p = pyaudio.PyAudio()
    resample_rate = print_selected_device_info(p, output_device_index)

    if wav_out is not None:
        wav_out.setframerate(resample_rate)

    asyncio.run(start_server(args.port, tts_service, args.voice, args.language_code, wav_out, output_device_index, resample_rate, args.interrupt))

    if wav_out is not None:
        wav_out.close()

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--list-devices", action="store_true", help="List output audio devices indices.")
    parser.add_argument("--input-device", type=int, default=1, help="An input audio device to use. Default is ReSpeaker 4 Mic Array.")
    parser.add_argument("--output-device", type=int, help="Output device to use. Default is system default.")
    parser.add_argument("--port", type=int, default=6100, help="Port to listen on. Defaults to 6100.")
    parser.add_argument("-o", "--output", type=str, help="Output file .wav file to write synthesized audio.")
    parser.add_argument("--voice", type=str, default="GLaDOS", help="A voice name to use for TTS. Default is 'GLaDOS'.")
    parser.add_argument("--language-code", type=str, default="en-US", help="Language code for TTS. Default is 'en-US'.")
    parser.add_argument("--interrupt", action="store_true", default=True, help="Whether to interrupt the current TTS when new tokens arrive. Defaults to True.")
    parser = add_connection_argparse_parameters(parser)
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
