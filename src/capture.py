"""
src/capture.py
Webcam and video file capture wrapper with timestamping and playback control.
"""

import cv2
import time
from typing import Optional, Tuple


class VideoCaptureWrapper:
    """
    Robust video capture supporting live webcams and recorded video playback.
    Provides frame timestamps and frame rate metadata.
    """

    def __init__(
        self,
        source: int | str = 0,
        width: int = 640,
        height: int = 480,
        target_fps: float = 30.0,
    ):
        self.source = source
        self.requested_width = width
        self.requested_height = height
        self.target_fps = target_fps
        self.is_video_file = isinstance(source, str) and not source.isdigit()

        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_index = 0
        self.start_time: float = 0.0
        self.actual_width = width
        self.actual_height = height
        self.fps = target_fps
        self.total_frames = -1

        self._open()

    def _open(self):
        # On Windows, try cv2.CAP_DSHOW for webcam to avoid slow initialization
        if isinstance(self.source, int) or (isinstance(self.source, str) and self.source.isdigit()):
            cam_idx = int(self.source)
            self.cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                # Fallback to default backend
                self.cap = cv2.VideoCapture(cam_idx)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.requested_width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.requested_height)
                self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        else:
            self.cap = cv2.VideoCapture(str(self.source))

        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video capture source: {self.source}")

        self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or self.requested_width
        self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self.requested_height
        detected_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if detected_fps and detected_fps > 0:
            self.fps = detected_fps
        else:
            self.fps = self.target_fps

        if self.is_video_file:
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.start_time = time.time()
        self.frame_index = 0

    def read(self) -> Tuple[bool, Optional[any], float]:
        """
        Read the next frame.
        Returns:
            (ret, frame, timestamp_sec)
        """
        if self.cap is None or not self.cap.isOpened():
            return False, None, 0.0

        ret, frame = self.cap.read()
        if not ret:
            return False, None, 0.0

        # In video replay mode, timestamp is derived deterministically from frame index and fps
        if self.is_video_file:
            timestamp = self.frame_index / max(self.fps, 1.0)
        else:
            timestamp = time.time() - self.start_time

        self.frame_index += 1
        return True, frame, timestamp

    def reset(self):
        """Reset video file back to beginning for deterministic replay."""
        if self.is_video_file and self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.frame_index = 0
            self.start_time = time.time()

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
