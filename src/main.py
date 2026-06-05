import os
import sys
import time
import threading

import cv2
from dotenv import load_dotenv

load_dotenv()

from motion import MotionDetector
from stream_server import set_frame, start_server
from callbacks import test, detectCat


# ---------------------------------------------------------------------------
# RTSP reader
# ---------------------------------------------------------------------------

def read_rtsp_stream(rtsp_url: str, detector: MotionDetector) -> None:
    """
    Read frames from *rtsp_url*, run motion detection, push frames to the web
    stream, and fire callbacks. Raises ConnectionError on any stream failure.
    """
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        raise ConnectionError("Could not open RTSP stream")

    ret, frame1 = cap.read()
    ret2, frame2 = cap.read()
    if not ret or not ret2:
        cap.release()
        raise ConnectionError("Could not read initial frames from stream")

    while True:
        annotated = detector.process(frame1, frame2)
        set_frame(annotated)

        frame1 = frame2
        ok, frame2 = cap.read()
        if not ok:
            cap.release()
            raise ConnectionError("Lost connection to RTSP stream")

        time.sleep(0.01)  # ~100 fps max; avoids pegging one CPU core


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    camera_ip = os.getenv("CAM_IP")
    username = os.getenv("CAM_USERNAME")
    password = os.getenv("CAM_PASSWORD")
    port = int(os.getenv("STREAM_PORT", "5000"))

    if not all([camera_ip, username, password]):
        print("ERROR: CAM_IP, CAM_USERNAME, and CAM_PASSWORD must be set in .env")
        sys.exit(1)

    rtsp_url = f"rtsp://{username}:{password}@{camera_ip}/stream1"

    # --- Motion detector + callbacks ---
    detector = MotionDetector()
    detector.register_callback(test, cooldown=1.0)       # log every 1 s
    detector.register_callback(detectCat, cooldown=5.0)  # YOLO every 5 s

    # --- Web stream server (background thread) ---
    server_thread = threading.Thread(
        target=start_server,
        kwargs={"host": "0.0.0.0", "port": port},
        daemon=True,
    )
    server_thread.start()

    print("=" * 50)
    print("  Borinne Recognition Stream")
    print("=" * 50)
    print(f"  Camera  : {camera_ip}")
    print(f"  Stream  : http://localhost:{port}/")
    print(f"  RTSP    : rtsp://<user>:<pass>@{camera_ip}/stream1")
    print("  Press Ctrl+C to exit")
    print("=" * 50)

    # --- Main RTSP loop with auto-reconnect ---
    retry_delay = 10
    attempt = 0

    while True:
        attempt += 1
        try:
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{ts}] Attempt #{attempt}: connecting ...")
            read_rtsp_stream(rtsp_url, detector)

        except ConnectionError as exc:
            print(f"  Connection error: {exc}")
            print(f"  Retrying in {retry_delay} s ...\n")
            time.sleep(retry_delay)

        except KeyboardInterrupt:
            print("\nExiting ...")
            break

        except Exception as exc:
            print(f"  Unexpected error: {exc}")
            print(f"  Retrying in {retry_delay} s ...\n")
            time.sleep(retry_delay)


if __name__ == "__main__":
    main()
