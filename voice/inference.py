#!/usr/bin/env python3
import os
import sys
import time
import wave
import io
import traceback
from flask import Flask, request, Response, jsonify
from piper import PiperVoice

# Initialize Flask app
app = Flask(__name__)

# Define the default prompt
DEFAULT_PROMPT = "How we doing bud!"

# Global variables for the PiperVoice instance and model path
voice = None
model_path = None

# Path to store generated audio files (optional, can be omitted if not needed)
script_dir = os.path.dirname(os.path.realpath(__file__))

def load_model():
    global voice, model_path
    # Define the specific model filename
    specific_model = 'glados_piper_medium.onnx'
    model = os.path.join(script_dir, specific_model)

    if not os.path.isfile(model):
        print(f"{specific_model} not found in the script directory. Please add it and restart the server.")
        sys.exit(1)

    model_path = model  # Store the model path globally
    print(f"Detected model in script directory: {specific_model}")

    # Initialize PiperVoice
    try:
        voice = PiperVoice.load(model, config_path=None, use_cuda=True)
        print(f"Model {specific_model} loaded successfully.")
    except Exception as e:
        print(f"Failed to load the model: {e}")
        sys.exit(1)

@app.route('/synthesize', methods=['POST'])
def synthesize():
    global voice, model_path

    if not voice:
        return jsonify({"error": "TTS model is not loaded."}), 500

    data = request.get_json()

    if not data:
        return jsonify({"error": "Invalid JSON payload."}), 400

    prompt = data.get('prompt', DEFAULT_PROMPT)
    speaker = data.get('speaker', 0)
    length_scale = data.get('length_scale', 1.0)
    noise_scale = data.get('noise_scale', 0.667)
    noise_w = data.get('noise_w', 0.8)
    sentence_silence = data.get('sentence_silence', 0.2)

    try:
        # Create an in-memory bytes buffer
        buffer = io.BytesIO()

        # Initialize WAV file parameters
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)       # Mono audio
            wav_file.setsampwidth(2)       # 16-bit audio
            wav_file.setframerate(22050)   # Sample rate

            # Perform synthesis
            start_time = time.perf_counter()
            voice.synthesize(
                prompt,
                wav_file,
                speaker_id=speaker,
                length_scale=length_scale,
                noise_scale=noise_scale,
                noise_w=noise_w,
                sentence_silence=sentence_silence
            )
            end_time = time.perf_counter()

        inference_duration = end_time - start_time

        # Calculate audio duration
        buffer.seek(0)
        with wave.open(buffer, "rb") as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            audio_duration = frames / float(rate)

        # Logging
        print(f"Piper TTS model:    {os.path.basename(model_path)}")
        print(f"Prompt:             {prompt}")
        print(f"Inference duration: {inference_duration:.3f} sec")
        print(f"Audio duration:     {audio_duration:.3f} sec")
        print(f"Realtime factor:    {inference_duration/audio_duration:.3f}")
        print(f"Inverse RTF (RTFX): {audio_duration/inference_duration:.3f}\n")

        # Prepare buffer for sending
        buffer.seek(0)

        # Extract PCM data by reading frames
        with wave.open(buffer, "rb") as wav_file:
            pcm_data = wav_file.readframes(wav_file.getnframes())

        # Define a generator to stream the PCM data in chunks
        def generate():
            chunk_size = 4096  # 4KB chunks
            for i in range(0, len(pcm_data), chunk_size):
                yield pcm_data[i:i+chunk_size]

        # Create a streaming response with appropriate headers
        return Response(generate(),
                        mimetype='audio/raw',
                        headers={
                            'Content-Disposition': f'attachment; filename=piper-{int(time.time())}.raw'
                        })

    except Exception as e:
        # Print full traceback for debugging
        traceback.print_exc()
        print(f"Error during synthesis: {e}")
        return jsonify({"error": "Failed to synthesize audio."}), 500

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"message": "Piper TTS Inference API is running."}), 200

if __name__ == "__main__":
    load_model()
    # Run the Flask app on port 61637 with debug mode enabled
    app.run(host="0.0.0.0", port=61637, debug=True)
