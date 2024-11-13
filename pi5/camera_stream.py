import os
import sys
import json
import threading
import cv2
import curses
from flask import Flask, Response
from picamera2 import Picamera2, Preview
from time import sleep

# Define the virtual environment name, config file, and required packages
CONFIG_FILE = "cam_config.json"

# Default configuration
DEFAULT_CONFIG = {
    "camera1": {"resolution": [640, 480], "flip": False},
    "camera2": {"resolution": [640, 480], "flip": False}
}

# Initialize Flask applications for camera streaming
app1 = Flask(__name__)
app2 = Flask(__name__)

# Load or create the configuration file
def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    else:
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG

def save_config(config):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=4)

# Start camera streaming using Picamera2 and OpenCV
def start_camera_streams(config):
    # Initialize Picamera2 instances for both cameras
    picam0 = Picamera2(0)
    picam1 = Picamera2(1)

    # Apply configuration settings
    def apply_settings(camera, settings):
        width, height = settings["resolution"]
        camera_config = camera.create_preview_configuration({"size": (width, height)})
        camera.configure(camera_config)
        camera.start_preview(Preview.QTGL if not settings["flip"] else Preview.QT)

    apply_settings(picam0, config["camera1"])
    apply_settings(picam1, config["camera2"])
    
    # Start camera streaming
    picam0.start()
    picam1.start()

    def generate_frames(camera, flip):
        while True:
            frame = camera.capture_array()
            if flip:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    @app1.route('/video_feed')
    def video_feed1():
        return Response(generate_frames(picam0, config["camera1"]["flip"]), mimetype='multipart/x-mixed-replace; boundary=frame')

    @app2.route('/video_feed')
    def video_feed2():
        return Response(generate_frames(picam1, config["camera2"]["flip"]), mimetype='multipart/x-mixed-replace; boundary=frame')

    # Get IP address for the URLs
    ip_address = os.popen("hostname -I").read().strip().split()[0]

    # Start Flask applications in separate threads
    threading.Thread(target=lambda: app1.run(host='0.0.0.0', port=5000)).start()
    threading.Thread(target=lambda: app2.run(host='0.0.0.0', port=5001)).start()

    print(f"\nCamera streams are running:")
    print(f" - Camera 1: http://{ip_address}:5000/video_feed")
    print(f" - Camera 2: http://{ip_address}:5001/video_feed\n")

# Curses-based menu interface for configuration
def curses_menu(stdscr, config):
    def display_menu(stdscr, selected_idx, options):
        stdscr.clear()
        h, w = stdscr.getmaxyx()
        for idx, option in enumerate(options):
            x = w // 2 - len(option["text"]) // 2
            y = h // 2 - len(options) // 2 + idx
            if idx == selected_idx:
                stdscr.attron(curses.color_pair(1))
                stdscr.addstr(y, x, option["text"])
                stdscr.attroff(curses.color_pair(1))
            else:
                stdscr.addstr(y, x, option["text"])
        stdscr.refresh()

    curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_WHITE)
    selected_idx = 0

    while True:
        options = [
            {"text": f"Camera 1 Resolution: {config['camera1']['resolution']}", "action": "set_resolution", "camera": "camera1"},
            {"text": f"Camera 1 Flip: {'On' if config['camera1']['flip'] else 'Off'}", "action": "toggle_flip", "camera": "camera1"},
            {"text": f"Camera 2 Resolution: {config['camera2']['resolution']}", "action": "set_resolution", "camera": "camera2"},
            {"text": f"Camera 2 Flip: {'On' if config['camera2']['flip'] else 'Off'}", "action": "toggle_flip", "camera": "camera2"},
            {"text": "Save and Exit", "action": "save_exit"},
            {"text": "Exit without Saving", "action": "exit"}
        ]

        display_menu(stdscr, selected_idx, options)
        key = stdscr.getch()

        if key == curses.KEY_UP and selected_idx > 0:
            selected_idx -= 1
        elif key == curses.KEY_DOWN and selected_idx < len(options) - 1:
            selected_idx += 1
        elif key in [curses.KEY_ENTER, 10, 13]:  # Enter key
            action = options[selected_idx]["action"]
            if action == "set_resolution":
                config[options[selected_idx]["camera"]]["resolution"] = set_resolution(stdscr)
            elif action == "toggle_flip":
                cam = options[selected_idx]["camera"]
                config[cam]["flip"] = not config[cam]["flip"]
            elif action == "save_exit":
                save_config(config)
                break
            elif action == "exit":
                break

def set_resolution(stdscr):
    resolutions = [(640, 480), (1280, 720), (1920, 1080)]
    idx = 0
    while True:
        stdscr.clear()
        h, w = stdscr.getmaxyx()
        title = "Choose Resolution"
        stdscr.addstr(h // 2 - 2, w // 2 - len(title) // 2, title)
        for i, res in enumerate(resolutions):
            x = w // 2 - len(str(res)) // 2
            y = h // 2 + i
            if i == idx:
                stdscr.attron(curses.color_pair(1))
                stdscr.addstr(y, x, str(res))
                stdscr.attroff(curses.color_pair(1))
            else:
                stdscr.addstr(y, x, str(res))
        stdscr.refresh()
        key = stdscr.getch()
        if key == curses.KEY_UP and idx > 0:
            idx -= 1
        elif key == curses.KEY_DOWN and idx < len(resolutions) - 1:
            idx += 1
        elif key in [curses.KEY_ENTER, 10, 13]:  # Enter key
            return resolutions[idx]

# Main entry point
if __name__ == "__main__":
    config = load_config()
    curses.wrapper(curses_menu, config)
    start_camera_streams(config)
