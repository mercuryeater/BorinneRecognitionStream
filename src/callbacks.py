"""
Motion-detection callbacks.

Register any of these with MotionDetector.register_callback():

    detector.register_callback(test, cooldown=1.0)
    detector.register_callback(detectCat, cooldown=5.0)

To add your own callback just define fn(frame: np.ndarray) -> None and register it.
"""

import os
import threading
import time

import cv2

# ---------------------------------------------------------------------------
# Lazy-loaded YOLO model (downloaded on first use, ~6 MB)
# ---------------------------------------------------------------------------
_yolo_model = None
_model_lock = threading.Lock()

CAT_CLASS_NAME = "cat"   # COCO class label used by YOLOv8
YOLO_MODEL = "yolov8n.pt"  # nano — fast and accurate enough for this task


def _get_model():
    """Return a cached YOLOv8 model, loading it on first call."""
    global _yolo_model
    with _model_lock:
        if _yolo_model is None:
            # Imported here so the rest of the module works even if ultralytics
            # is not yet installed (useful during container build).
            from ultralytics import YOLO  # noqa: PLC0415

            print(f"[isCat] Loading {YOLO_MODEL} … (first run downloads the weights)")
            _yolo_model = YOLO(YOLO_MODEL)
            print("[isCat] Model ready.")
    return _yolo_model


# ---------------------------------------------------------------------------
# Cat-shot output directory (lives next to this file: src/cat_shots/)
# ---------------------------------------------------------------------------
_CAT_SHOTS_DIR = os.path.join(os.path.dirname(__file__), "cat_shots")


def _save_cat_shot(frame) -> str:
    """Persist *frame* to src/cat_shots/ and return the file path."""
    os.makedirs(_CAT_SHOTS_DIR, exist_ok=True)
    filename = os.path.join(
        _CAT_SHOTS_DIR, f"cat_{time.strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
    )
    cv2.imwrite(filename, frame)
    return filename


# ---------------------------------------------------------------------------
# Public callbacks
# ---------------------------------------------------------------------------

def test(frame) -> None:
    """
    Simple diagnostic callback.
    Logs 'Movement detected' to stdout whenever motion is found.
    """
    print(f"[{time.strftime('%H:%M:%S')}] Movement detected")


def isCat(frame) -> bool:
    """
    Run YOLOv8n inference on *frame* and return True if a cat is detected.

    Uses the COCO-trained nano model; no API key or internet access required
    after the one-time weight download (~6 MB).
    """
    model = _get_model()
    results = model(frame, verbose=False)
    for result in results:
        for cls_id in result.boxes.cls:
            if model.names[int(cls_id)] == CAT_CLASS_NAME:
                return True
    return False


def detectCat(frame) -> None:
    """
    Snapshot callback: calls isCat() on the current frame and, if a cat is
    found, saves the image to src/cat_shots/.

    Designed to be registered with a longer cooldown (e.g. 5 s) because
    YOLO inference takes ~50–200 ms depending on hardware.
    """
    ts = time.strftime("%H:%M:%S")
    if isCat(frame):
        path = _save_cat_shot(frame)
        print(f"[{ts}] 🐱 Cat detected! Image saved → {path}")
    else:
        print(f"[{ts}] Motion detected — no cat found.")
