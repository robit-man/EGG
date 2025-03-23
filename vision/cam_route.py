#!/usr/bin/env python3
import os
import sys
import subprocess

# ----------------- Auto‑venv Bootstrap -----------------
if sys.prefix == sys.base_prefix:
    venv_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv")
    if not os.path.exists(venv_dir):
        print("Creating virtual environment...")
        subprocess.check_call([sys.executable, "-m", "venv", "venv"])
    if os.name == "nt":
        pip_exe = os.path.join(venv_dir, "Scripts", "pip.exe")
        python_exe = os.path.join(venv_dir, "Scripts", "python.exe")
    else:
        pip_exe = os.path.join(venv_dir, "bin", "pip")
        python_exe = os.path.join(venv_dir, "bin", "python")
    print("Installing dependencies...")
    subprocess.check_call([pip_exe, "install", "--upgrade", "pip"])
    subprocess.check_call([pip_exe, "install", "flask"])
    print("Relaunching inside virtual environment...")
    subprocess.check_call([python_exe] + sys.argv)
    sys.exit()

# ----------------- Imports (inside venv) -----------------
import glob
import socket
import time
from flask import Flask, Response, render_template_string

# ----------------- Configuration -----------------
BASE_PORT = 9000  # starting port for GStreamer streams
camera_ports = {}  # mapping: camera index -> TCP port
gst_processes = []  # list to hold gst-launch process objects

# ----------------- Helper: List Camera Devices -----------------
def get_camera_devices():
    devices = glob.glob("/dev/video*")
    devices.sort()
    return devices

# ----------------- Spawn GStreamer Pipelines -----------------
# For each camera device, start a gst-launch process that outputs an MJPEG stream over TCP.
devices = get_camera_devices()
if not devices:
    print("No camera devices found at /dev/video*")
    sys.exit(1)

for i, device in enumerate(devices):
    port = BASE_PORT + i
    camera_ports[i] = port
    # The pipeline is based on your working bash pipeline but modified to encode as JPEG and output via TCP.
    # (We use 'jpegenc' and 'multipartmux' with boundary "frame" to produce a proper MJPEG stream.)
    command = (
        f'gst-launch-1.0 nvv4l2camerasrc device={device} ! '
        f'"video/x-raw(memory:NVMM), format=(string)UYVY, width=(int)1920, height=(int)1080" ! '
        f'nvvidconv flip-method=3 ! '
        f'"video/x-raw(memory:NVMM), format=(string)I420, width=(int)1080, height=(int)1920" ! '
        f'nvvidconv ! videoconvert ! jpegenc ! multipartmux boundary=frame ! '
        f'tcpserversink host=0.0.0.0 port={port}'
    )
    print(f"Starting GStreamer pipeline for {device} on port {port}:\n{command}\n")
    # Start the process (stdout/stderr are captured so you can review logs if needed)
    proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    gst_processes.append(proc)

# ----------------- Flask Application -----------------
app = Flask(__name__)

@app.route("/")
def index():
    html = """
    <!doctype html>
    <html>
      <head>
        <title>Camera Streams</title>
      </head>
      <body>
        <h1>Camera Streams</h1>
        {% for i in camera_ids %}
          <div style="margin-bottom:20px;">
            <h3>Camera {{ i }} (TCP Port {{ camera_ports[i] }})</h3>
            <!-- The img src points to our Flask proxy route -->
            <img src="/camera/{{ i }}" width="1080" height="1920" alt="Camera {{ i }} stream">
          </div>
          <hr>
        {% endfor %}
      </body>
    </html>
    """
    camera_ids = list(camera_ports.keys())
    return render_template_string(html, camera_ids=camera_ids, camera_ports=camera_ports)

@app.route("/camera/<int:cam_id>")
def camera_stream(cam_id):
    if cam_id not in camera_ports:
        return "Camera not found", 404
    port = camera_ports[cam_id]
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect(("127.0.0.1", port))
    except Exception as e:
        return f"Error connecting to camera stream on port {port}: {e}", 500

    def generate():
        try:
            while True:
                data = s.recv(1024)
                if not data:
                    break
                yield data
        except Exception as e:
            print(f"Error reading from socket: {e}")
        finally:
            s.close()

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

# ----------------- Run Flask Server -----------------
if __name__ == "__main__":
    print("Flask server starting on port 8080...")
    app.run(host="0.0.0.0", port=8080)
