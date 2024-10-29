import usb.core
import usb.util
import time
import requests
import base64
import json
import curses
from tuning import Tuning

# USB device ID for the microphone
MIC_VENDOR_ID = 0x2886
MIC_PRODUCT_ID = 0x0018

# DOA ranges mapped to camera indices
DOA_TO_CAMERA_MAP = {
    0: 5,
    60: 0,
    120: 1,
    180: 2,
    240: 3,
    300: 4
}

DOA_TOLERANCE = 30  # Define a tolerance range to map to the nearest DOA angle

def find_nearest_camera(doa):
    """Map the DOA to the nearest camera index within the tolerance range."""
    for doa_angle, camera_id in DOA_TO_CAMERA_MAP.items():
        if abs(doa - doa_angle) <= DOA_TOLERANCE:
            return camera_id
    return None  # Return None if no valid DOA mapping found

def get_frame_image_url(camera_id):
    """Get the URL for the camera feed based on the camera ID."""
    return f"http://127.0.0.1:{5000 + camera_id}/latest_frame.jpg"

def fetch_image_base64(url):
    """Fetch the image from the camera URL and return it as base64."""
    image_response = requests.get(url)
    if image_response.status_code == 200:
        return base64.b64encode(image_response.content).decode("utf-8")
    return None

def get_next_camera(camera_id):
    """Get the next camera ID, wrapping around if necessary."""
    return (camera_id + 1) % 6

def get_closest_available_camera(camera_id):
    """Check the DOA-based camera and fall back to the closest available one."""
    original_camera_id = camera_id
    for _ in range(6):
        frame_image_url = get_frame_image_url(camera_id)
        image_base64 = fetch_image_base64(frame_image_url)
        if image_base64:
            return camera_id, image_base64
        camera_id = get_next_camera(camera_id)
    return original_camera_id, None  # Return original camera and None if no available cameras

def main(stdscr):
    # Initialize curses settings
    curses.curs_set(1)
    stdscr.clear()
    stdscr.nodelay(True)  # Make getch() non-blocking
    stdscr.timeout(500)   # Refresh rate for DOA display (in milliseconds)

    # Find the USB microphone device for DOA detection
    dev = usb.core.find(idVendor=MIC_VENDOR_ID, idProduct=MIC_PRODUCT_ID)
    if not dev:
        stdscr.addstr(0, 0, "Microphone not found. Please check your USB connection.")
        stdscr.refresh()
        time.sleep(3)
        return

    # Initialize microphone tuning object for DOA reading
    mic_tuning = Tuning(dev)
    current_doa = None
    input_prompt = ""  # User input buffer for the prompt

    # Continuous display loop
    while True:
        # Get and display the current DOA
        try:
            current_doa = mic_tuning.direction
            stdscr.addstr(0, 0, f"Current DOA: {current_doa}     ")

            # Get user input
            key = stdscr.getch()
            if key == curses.KEY_BACKSPACE or key == 127:
                input_prompt = input_prompt[:-1]
            elif key == curses.KEY_ENTER or key == 10:
                if input_prompt:
                    # Process inference with the current DOA and input prompt
                    process_inference(current_doa, input_prompt, stdscr)
                    input_prompt = ""
            elif key != -1:
                input_prompt += chr(key)

            # Display the prompt input line
            stdscr.addstr(2, 0, "Enter your prompt: " + input_prompt + " " * 20)
            stdscr.refresh()

        except KeyboardInterrupt:
            break

def process_inference(current_doa, prompt, stdscr):
    """Handles inference based on the current DOA and user prompt."""
    # Map DOA to camera ID
    camera_id = find_nearest_camera(current_doa)
    if camera_id is None:
        stdscr.addstr(3, 0, "Unrecognized DOA value. Please adjust and try again.")
        stdscr.refresh()
        time.sleep(1)
        return

    # Find the closest available camera
    final_camera_id, image_base64 = get_closest_available_camera(camera_id)
    if image_base64 is None:
        stdscr.addstr(3, 0, "All cameras failed to deliver images. Exiting.")
        stdscr.refresh()
        time.sleep(1)
        return

    stdscr.addstr(3, 0, f"Using camera {final_camera_id} for inference.")
    stdscr.refresh()

    # Prepare JSON payload with prompt and base64 image
    payload = {
        "model": "llava:13b",
        "prompt": prompt,
        "images": [image_base64]
    }

    # Send the request to LLaVA and process response as a stream
    response = requests.post("http://127.0.0.1:11434/api/generate", json=payload, stream=True)
    
    if response.status_code == 200:
        response_text = ""
        for chunk in response.iter_lines():
            if chunk:
                try:
                    # Load JSON for each line and extract 'response' value
                    data = json.loads(chunk.decode("utf-8"))
                    if "response" in data:
                        response_text += data["response"]
                except json.JSONDecodeError:
                    stdscr.addstr(5, 0, "Received malformed JSON chunk.")
        
        stdscr.addstr(6, 0, "Full Response received:\n" + response_text.strip())
        stdscr.refresh()
    else:
        stdscr.addstr(6, 0, f"Failed with status code {response.status_code}: {response.text}")
        stdscr.refresh()

if __name__ == "__main__":
    curses.wrapper(main)
