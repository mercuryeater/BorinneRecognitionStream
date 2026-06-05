import time
import threading

import cv2
from flask import Flask, Response, render_template_string

app = Flask(__name__)

_frame_lock = threading.Lock()
_current_frame = None

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Borinne Live Stream</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: #0f0f0f;
      color: #eee;
      font-family: sans-serif;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      min-height: 100vh;
      gap: 16px;
    }
    h1 { font-size: 1.4rem; letter-spacing: 0.05em; color: #ccc; }
    .stream-wrapper {
      border: 2px solid #2a2a2a;
      border-radius: 8px;
      overflow: hidden;
      max-width: 960px;
      width: 100%;
    }
    img { display: block; width: 100%; }
    .status { font-size: 0.8rem; color: #555; }
  </style>
</head>
<body>
  <h1>📷 Borinne Live Stream</h1>
  <div class="stream-wrapper">
    <img src="/stream" alt="Live stream" />
  </div>
  <p class="status">Motion detection active &mdash; cat recognition enabled</p>
</body>
</html>
"""


def set_frame(frame):
    """Called by the RTSP reader to push the latest frame."""
    global _current_frame
    with _frame_lock:
        _current_frame = frame.copy()


def _generate_frames():
    while True:
        with _frame_lock:
            if _current_frame is None:
                time.sleep(0.05)
                continue
            frame = _current_frame.copy()

        ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )
        time.sleep(1 / 30)  # cap at ~30 fps


@app.route("/")
def index():
    return render_template_string(HTML_PAGE)


@app.route("/stream")
def stream():
    return Response(
        _generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


def start_server(host: str = "0.0.0.0", port: int = 5000) -> None:
    """Start the Flask MJPEG server. Designed to run in a daemon thread."""
    app.run(host=host, port=port, threaded=True, use_reloader=False)
