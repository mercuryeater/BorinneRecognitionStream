import threading
import time
from typing import Callable

import cv2


class MotionDetector:
    """
    Detects motion between consecutive frames and fires registered callbacks
    when significant movement is found.

    Usage:
        detector = MotionDetector()
        detector.register_callback(my_fn, cooldown=3.0)
        annotated = detector.process(frame1, frame2)
    """

    MIN_CONTOUR_AREA = 900   # px² — smaller blobs are ignored
    CROP_TOP_PX = 50         # rows to skip (hides camera timestamp overlay)

    def __init__(self) -> None:
        self._callbacks: list[dict] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_callback(self, fn: Callable, cooldown: float = 3.0) -> None:
        """
        Register a function to call when motion is detected.

        Args:
            fn:       Callable with signature fn(frame: np.ndarray) -> None.
            cooldown: Minimum seconds between successive calls of this callback.
        """
        with self._lock:
            self._callbacks.append({"fn": fn, "cooldown": cooldown, "last_called": 0.0})

    def process(self, frame1, frame2):
        """
        Compare two consecutive frames, draw bounding boxes on a copy for the
        stream, and fire registered callbacks with the clean (no boxes) frame.

        Returns the annotated copy for display.
        """
        contours = self._detect(frame1, frame2)
        motion_found = False
        annotated = frame1.copy()

        for contour in contours:
            if cv2.contourArea(contour) < self.MIN_CONTOUR_AREA:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(annotated, (x, y + self.CROP_TOP_PX), (x + w, y + h + self.CROP_TOP_PX), (0, 255, 0), 2)
            motion_found = True

        if motion_found:
            # Pass the clean frame to callbacks so saved images have no boxes
            self._fire_callbacks(frame1)

        return annotated

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect(self, frame1, frame2):
        """Return contours of changed regions (cropped to exclude timestamp)."""
        f1 = frame1[self.CROP_TOP_PX:, :]
        f2 = frame2[self.CROP_TOP_PX:, :]

        diff = cv2.absdiff(f1, f2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def _fire_callbacks(self, frame) -> None:
        """Call each registered callback in its own daemon thread, respecting cooldowns."""
        now = time.time()
        with self._lock:
            for cb in self._callbacks:
                if now - cb["last_called"] >= cb["cooldown"]:
                    cb["last_called"] = now
                    threading.Thread(
                        target=cb["fn"],
                        args=(frame.copy(),),
                        daemon=True,
                    ).start()
