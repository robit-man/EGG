import numpy as np
import requests
import soundfile as sf
import argparse

# Create an argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--file_path', type=str, help='Path to the WAV file', default='./samples/jfk.wav')

# Parse the command-line arguments
args = parser.parse_args()

# Load the WAV file
data, sample_rate = sf.read(args.file_path)

# Convert the audio data to a NumPy array
audio_array = np.array(data)

response = requests.post(
   "http://0.0.0.0:3000/transcribe_ndarray",
   headers={
        "accept": "text/plain",
       "content-type": "application/json"
       },
   data=str(audio_array.tolist()),
)

print(response.text)
