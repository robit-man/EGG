from flask import Flask, send_file
import os
import threading


# Run the following to expost the camera stream as a route
'''
gst-launch-1.0 v4l2src device=/dev/video0 ! \
    videoconvert ! \
    videoflip method=clockwise ! \
    jpegenc ! \
    multifilesink location=/tmp/latest_frame_0.jpg
'''
# Or Not Rotated
'''
gst-launch-1.0 v4l2src device=/dev/video0 ! \
    videoconvert ! \
    jpegenc ! \
    multifilesink location=/tmp/latest_frame_0.jpg
'''
# Or Flipped
'''
gst-launch-1.0 v4l2src device=/dev/video0 ! \
    videoconvert ! \
    videoflip method=rotate-180 ! \
    jpegenc ! \
    multifilesink location=/tmp/latest_frame_0.jpg

'''

# Function to create a Flask app for each stream
def create_app(stream_id):
    app = Flask(__name__)

    @app.route('/latest_frame')
    def latest_frame():
        image_path = f"/tmp/latest_frame_{stream_id}.jpg"
        if os.path.exists(image_path):
            return send_file(image_path, mimetype='image/jpeg')
        else:
            return "No image available", 404

    return app

# Function to run a Flask server for a given stream ID and port
def run_server(stream_id, port):
    app = create_app(stream_id)
    app.run(host="0.0.0.0", port=port)

if __name__ == "__main__":
    threads = []

    # Start a separate Flask server for each stream (ports 5000 to 5005)
    for stream_id in range(6):
        port = 5000 + stream_id
        thread = threading.Thread(target=run_server, args=(stream_id, port))
        thread.start()
        threads.append(thread)

    # Join all threads to keep the servers running
    for thread in threads:
        thread.join()
