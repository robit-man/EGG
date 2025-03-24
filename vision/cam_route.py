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
devices = get_camera_devices()
if not devices:
    print("No camera devices found at /dev/video*")
    sys.exit(1)

for i, device in enumerate(devices):
    port = BASE_PORT + i
    camera_ports[i] = port
    # Pipeline: encode as JPEG, mux into multipart stream over TCP.
    command = (
        f'gst-launch-1.0 nvv4l2camerasrc device={device} ! '
        f'"video/x-raw(memory:NVMM), format=(string)UYVY, width=(int)1920, height=(int)1080" ! '
        f'nvvidconv flip-method=3 ! '
        f'"video/x-raw(memory:NVMM), format=(string)I420, width=(int)1080, height=(int)1920" ! '
        f'nvvidconv ! videoconvert ! jpegenc ! multipartmux boundary=frame ! '
        f'tcpserversink host=0.0.0.0 port={port}'
    )
    print(f"Starting GStreamer pipeline for {device} on port {port}:\n{command}\n")
    proc = subprocess.Popen(command, shell=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    gst_processes.append(proc)

# ----------------- Flask Application -----------------
app = Flask(__name__)

@app.route("/")
def index():
    html = """
    <!doctype html>
    <html>
      <head>
        <title>Camera Snapshots</title>
      </head>
      <body>
        <h1>Camera Snapshots</h1>
        {% for i in camera_ids %}
          <div style="margin-bottom:20px;">
            <h3>Camera {{ i }} (TCP Port {{ camera_ports[i] }})</h3>
            <!-- The img src points to our snapshot route -->
            <img src="/camera/{{ i }}" width="1080" height="1920" alt="Camera {{ i }} snapshot">
          </div>
          <hr>
        {% endfor %}
      </body>
    </html>
    """
    camera_ids = list(camera_ports.keys())
    return render_template_string(html, camera_ids=camera_ids, camera_ports=camera_ports)

@app.route("/camera/<int:cam_id>")
def camera_snapshot(cam_id):
    if cam_id not in camera_ports:
        return Response(b"Camera not found", status=404, mimetype="text/plain")
    port = camera_ports[cam_id]
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect(("127.0.0.1", port))
    except Exception as e:
        error_msg = f"Error connecting to camera stream on port {port}: {e}"
        return Response(error_msg.encode("utf-8"), status=500, mimetype="text/plain")

    # Read from the socket until a complete JPEG is received
    data = b""
    frame = None
    try:
        while True:
            chunk = s.recv(1024)
            if not chunk:
                break
            data += chunk
            start = data.find(b'\xff\xd8')  # JPEG start marker
            end = data.find(b'\xff\xd9')    # JPEG end marker
            if start != -1 and end != -1 and end > start:
                frame = data[start:end+2]
                break
    except Exception as e:
        print(f"Error reading from socket: {e}")
    finally:
        s.close()

    if frame:
        return Response(frame, mimetype="image/jpeg")
    else:
        return Response(b"No frame received", status=500, mimetype="text/plain")

# ----------------- Run Flask Server -----------------
if __name__ == "__main__":
    print("Flask server starting on port 8080...")
    app.run(host="0.0.0.0", port=8080)
