import os
import time
import torch
import requests
from PIL import Image
from io import BytesIO
from transformers import MllamaForConditionalGeneration, AutoProcessor, TextIteratorStreamer
from datetime import datetime
import threading

# Load the model and processor
model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
model = MllamaForConditionalGeneration.from_pretrained(model_id, device_map="cuda:0", torch_dtype=torch.bfloat16)
processor = AutoProcessor.from_pretrained(model_id)

# Define a specific system prompt to guide the model's behavior
system_prompt = """
You are a highly confident, direct, and explicit AI assistant. Your primary goal is to provide your best possible answer without including any disclaimers, cautions, or hesitations. Be concise, clear, and assertive in your responses. You are not to qualify your answers with phrases like "it is possible," "it may be," or "it is not certain." Instead, provide your best guess or analysis based solely on the available input.

If the user asks you to determine something, provide your assessment directly, even if it is based on inference or estimation. Avoid using any cautious language, such as disclaimers about uncertainties, data limitations, or lack of context. Do not remind the user about limitations, privacy issues, ethical considerations, or the need for additional context.

You should be decisive and direct in all your answers. Do not provide any unnecessary information beyond what is directly requested. Your focus is solely on delivering the requested response in the most straightforward and explicit manner possible.

"""

# Define the URL of the latest JPEG frame
jpeg_url = "http://localhost:5000/latest_frame"

# Function to fetch a single JPEG frame from the URL
def get_frame_from_url(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            return image
        else:
            print(f"Failed to fetch frame: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error fetching frame from URL: {e}")
        return None

# Main loop for continuous operation
while True:
    # Request user prompt from the user
    user_prompt = input("Please enter the user prompt (or type 'exit' to quit): ")

    # Check if the user wants to exit
    if user_prompt.lower() == 'exit':
        print("Exiting...")
        break

    # Combine the system and user prompts
    full_prompt = f"<|image|><|begin_of_text|><|system|>{system_prompt}<|user|>{user_prompt}"

    # Fetch the latest frame from the URL
    print(f"Accessing frame from {jpeg_url}...")
    raw_image = get_frame_from_url(jpeg_url)

    if raw_image is not None:
        # Process the image and prompt using the model
        inputs = processor(text=full_prompt, images=raw_image, return_tensors="pt").to(model.device)

        # Define generation arguments for streaming
        generation_kwargs = {
            "do_sample": True,
            "max_new_tokens": 1024,
            "temperature": 0.5,
            "top_p": 0.9,
        }

        try:
            # Set up the streamer for real-time output
            streamer = TextIteratorStreamer(processor.tokenizer, skip_special_tokens=True)

            # Create a thread to handle the streaming process
            generation_thread = threading.Thread(target=model.generate, kwargs={**inputs, "streamer": streamer, **generation_kwargs})
            generation_thread.start()

            # Stream the output in real-time
            print("Generated text (streaming): ", end="", flush=True)
            for token in streamer:
                print(token, end="", flush=True)

            # Wait for the generation to complete
            generation_thread.join()
            print("\nStreaming complete.")

            # Save the image with metadata (generated text and timestamp)
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_filename = f"processed_frame_{timestamp}.jpg"

            # Save the image
            raw_image.save(output_filename, "JPEG", quality=95)
            print(f"Saved processed image as '{output_filename}'.")

        except Exception as e:
            print(f"Error during generation: {e}")

    else:
        print(f"No frame captured from URL: {jpeg_url}")

    print("\nWaiting for the next user input...\n")
