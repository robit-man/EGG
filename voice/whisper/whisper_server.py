#!/usr/bin/env python3
import os
import threading
import tempfile
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import wave
import time
import socket
import struct

import whisper
import torch

# Configuration
MODEL_NAME = "medium"          # Whisper model size: tiny, base, small, medium, large
MODEL_DIR = "models"           # Directory to store the Whisper model
AUDIO_SAMPLE_RATE = 16000      # Whisper expects 16kHz audio
CHANNELS = 1                   # Mono audio
PORT_AUDIO = 64167             # Updated Port to receive audio streams
PORT_SEND = 64162              # Port to send consolidated transcriptions
TIMEOUT = 1                    # Timeout in seconds to wait before sending transcription

SILENCE_RMS_THRESHOLD = 50    # Threshold for RMS below which audio is considered silent

# File monitoring
SCRIPT_FILE = os.path.realpath(__file__)
CHECK_INTERVAL = 1  # seconds

# Global variables to store the latest transcription and last received time
latest_transcription = ""
transcription_lock = threading.Lock()
last_received_time = time.time()

def compute_rms(audio_bytes):
    """
    Compute the RMS amplitude for 16-bit little-endian PCM audio.
    
    Args:
        audio_bytes (bytes): Raw audio data.
    
    Returns:
        float: The computed RMS value.
    """
    count = len(audio_bytes) // 2
    if count == 0:
        return 0.0
    fmt = "<" + "h" * count
    try:
        samples = struct.unpack(fmt, audio_bytes)
    except Exception:
        return 0.0
    sum_squares = sum(sample * sample for sample in samples)
    return (sum_squares / count) ** 0.5

def load_whisper_model(model_name: str, model_dir: str) -> whisper.Whisper:
    """
    Load the Whisper model, downloading it if necessary.

    Args:
        model_name (str): Name of the Whisper model to load.
        model_dir (str): Directory where models are stored.

    Returns:
        whisper.Whisper: Loaded Whisper model.
    """
    os.makedirs(model_dir, exist_ok=True)
    print(f"Loading Whisper model '{model_name}'...")
    model = whisper.load_model(model_name, download_root=model_dir)
    print(f"Whisper model '{model_name}' loaded successfully.")
    return model

# Load the Whisper model at startup
model = load_whisper_model(MODEL_NAME, MODEL_DIR)

class AudioHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        global latest_transcription, last_received_time

        content_length = int(self.headers.get('Content-Length', 0))
        if content_length == 0:
            self.send_response(400)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            response = {"error": "No audio data received."}
            self.wfile.write(json.dumps(response).encode())
            return

        # Read the audio data from the request
        audio_data = self.rfile.read(content_length)
        
        # Sanity check: Discard transcription if audio is silent
        rms_value = compute_rms(audio_data)
        if rms_value < SILENCE_RMS_THRESHOLD:
            print(f"Audio chunk deemed silent (RMS: {rms_value:.2f}). Discarding transcription.")
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            response = {"status": "No speech detected, transcription discarded due to silence."}
            self.wfile.write(json.dumps(response).encode())
            return

        # Update the last received time
        with transcription_lock:
            last_received_time = time.time()

        # Save the received audio data to a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            try:
                with wave.open(temp_wav.name, "wb") as wf:
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(2)  # 16 bits per sample
                    wf.setframerate(AUDIO_SAMPLE_RATE)
                    wf.writeframes(audio_data)

                # Load and preprocess the audio for Whisper
                audio = whisper.load_audio(temp_wav.name)
                audio = whisper.pad_or_trim(audio)
                
                if MODEL_NAME == "large":
                    mel = whisper.log_mel_spectrogram(audio=audio, n_mels=128).to(model.device)
                    
                if MODEL_NAME in ["medium", "small", "base", "tiny"]:
                    mel = whisper.log_mel_spectrogram(audio).to(model.device)

                # Detect language
                _, probs = model.detect_language(mel)
                detected_lang = max(probs, key=probs.get)
                print(f"Detected language: {detected_lang}")

                # Decode the audio
                options = whisper.DecodingOptions()
                result = whisper.decode(model, mel, options)

                # Update the latest transcription
                with transcription_lock:
                    latest_transcription += result.text + " "

                print(f"Transcription updated: {result.text}")

                # Respond to the client
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                response = {"status": "Transcription successful.", "transcribed_text": result.text}
                self.wfile.write(json.dumps(response).encode())

            except Exception as e:
                print(f"Error processing audio: {e}")
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                response = {"error": str(e)}
                self.wfile.write(json.dumps(response).encode())

            finally:
                # Clean up the temporary file
                os.unlink(temp_wav.name)

    def log_message(self, format, *args):
        # Override to prevent default logging
        return

class TranscriptionSender(threading.Thread):
    """
    Thread to monitor transcription accumulation and send it after a timeout.
    """
    def __init__(self, send_port, timeout):
        super().__init__(daemon=True)
        self.send_port = send_port
        self.timeout = timeout

    def run(self):
        global latest_transcription, last_received_time
        while True:
            time.sleep(1)
            transcription_to_send = ""  # Initialize the variable to ensure it's always defined
            with transcription_lock:
                elapsed = time.time() - last_received_time
                if elapsed >= self.timeout and latest_transcription.strip():
                    # Consolidate the transcription
                    transcription_to_send = latest_transcription.strip()
                    latest_transcription = ""
                    last_received_time = time.time()

            if transcription_to_send:
                self.send_transcription(transcription_to_send)

    def send_transcription(self, transcription):
        """
        Sends the consolidated transcription to the specified port via TCP socket.
        """
        HOST = 'localhost'  # Change if the receiver is on a different host
        PORT = self.send_port

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((HOST, PORT))
                s.sendall(transcription.encode('utf-8'))
                print(f"Sent transcription to {HOST}:{PORT} -> {transcription}")
        except ConnectionRefusedError:
            print(f"Error: Unable to connect to {HOST}:{PORT}. Ensure that the receiver is running.")
        except Exception as e:
            print(f"An unexpected error occurred while sending transcription: {e}")

def run_server(server_class=HTTPServer, handler_class=AudioHandler, port=PORT_AUDIO):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f"Audio server running on port {port}")
    httpd.serve_forever()

class ScriptWatcher(threading.Thread):
    def __init__(self, script_file, interval=1):
        super().__init__(daemon=True)
        self.script_file = script_file
        self.interval = interval
        self.last_mtime = os.path.getmtime(script_file)

    def run(self):
        while True:
            try:
                current_mtime = os.path.getmtime(self.script_file)
                if current_mtime != self.last_mtime:
                    print(f"Script file '{self.script_file}' changed, restarting...")
                    os.execv(sys.executable, [sys.executable, self.script_file])
                time.sleep(self.interval)
            except Exception as e:
                print(f"Error monitoring script file: {e}")
                time.sleep(self.interval)

def main():
    watcher = ScriptWatcher(SCRIPT_FILE, CHECK_INTERVAL)
    watcher.start()

    sender = TranscriptionSender(send_port=PORT_SEND, timeout=TIMEOUT)
    sender.start()

    run_server()

if __name__ == "__main__":
    main()
