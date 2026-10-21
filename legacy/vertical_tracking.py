"""Vertical face-position measurements and the single motor-axis controller."""
from __future__ import annotations

import time
from typing import NamedTuple, Optional

from .face_centering import MQTTMotorClient


class VerticalPosition(NamedTuple):
    nose_offset: float
    face_offset: float
    face_center: float
    direction: str


def measure_vertical(face_box: tuple[int, int, int, int], nose_y: float,
                     frame_height: int, deadband_ratio: float = 0.04) -> VerticalPosition:
    """Measure vertical offsets and direction in image coordinates."""
    if frame_height <= 0:
        raise ValueError("Frame height must be positive")
    _, y1, _, y2 = face_box
    face_center = (y1 + y2) / 2.0
    nose_offset = float(nose_y) - frame_height / 2.0
    face_offset = face_center - frame_height / 2.0
    if abs(nose_offset) < frame_height * deadband_ratio:
        direction = "center"
    else:
        direction = "down" if nose_offset > 0 else "up"
    return VerticalPosition(nose_offset, face_offset, face_center, direction)


class VerticalFaceController:
    """Map vertical movement of the locked face to bounded motor angles."""

    def __init__(self, motor: MQTTMotorClient, center_angle: float = 90.0,
                 min_angle: float = 0.0, max_angle: float = 180.0,
                 direction: int = 1, deadband: float = 0.04,
                 publish_interval: float = 0.12, smoothing: float = 0.35):
        if not 0 <= min_angle < max_angle <= 180:
            raise ValueError("Motor limits must satisfy 0 <= min < max <= 180 degrees")
        if not min_angle <= center_angle <= max_angle:
            raise ValueError("Center angle must be within the configured motor limits")
        if direction not in (-1, 1):
            raise ValueError("Direction must be -1 or 1")
        if not 0 <= deadband < 0.5:
            raise ValueError("Deadband must be between 0 and 0.5 of frame height")
        if not 0 < smoothing <= 1:
            raise ValueError("Smoothing must be greater than 0 and at most 1")
        self.motor = motor
        self.center_angle = center_angle
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.direction = direction
        self.deadband = deadband
        self.publish_interval = publish_interval
        self.smoothing = smoothing
        self._filtered_error: Optional[float] = None
        self._tracking_origin_dy = 0.0
        self._last_publish = 0.0
        self._last_angle: Optional[float] = None
        self._connection_count = motor.connection_count
        self._pending_angle: Optional[float] = None

    def start_tracking(self, face_offset_y_px: float) -> None:
        """Use the locked face's current height as the pointer's center."""
        self._tracking_origin_dy = float(face_offset_y_px)
        self._filtered_error = None
        self._last_angle = None
        self._last_publish = 0.0

    def update(self, face_offset_y_px: float, frame_height: int, track_id: int,
               face_name: str) -> Optional[float]:
        """Publish an angle for the face's vertical position, if changed."""
        if frame_height <= 0:
            return None
        if self._connection_count != self.motor.connection_count:
            self._connection_count = self.motor.connection_count
            self._last_angle = None
            self._last_publish = 0.0
        self._pending_angle = None
        relative_offset = float(face_offset_y_px) - self._tracking_origin_dy
        error = max(-1.0, min(1.0, relative_offset / (frame_height / 2.0)))
        self._filtered_error = (error if self._filtered_error is None else
                                self.smoothing * error + (1.0 - self.smoothing) * self._filtered_error)
        filtered_error = self._filtered_error
        if abs(filtered_error) <= self.deadband:
            target_angle = self.center_angle
        else:
            outside_deadband = (abs(filtered_error) - self.deadband) / (1.0 - self.deadband)
            side = 1 if filtered_error > 0 else -1
            motor_side = side * self.direction
            usable = (self.max_angle - self.center_angle if motor_side > 0 else
                      self.center_angle - self.min_angle)
            target_angle = self.center_angle + motor_side * usable * outside_deadband
        target_angle = max(self.min_angle, min(self.max_angle, target_angle))

        now = time.monotonic()
        if (now - self._last_publish < self.publish_interval or
                self._last_angle is not None and abs(target_angle - self._last_angle) < 1.0):
            return None
        sent = self.motor.publish_angle(
            target_angle, track_id=int(track_id), name=face_name,
            axis="vertical", error_ratio=round(filtered_error, 3),
            error_px=round(relative_offset, 1), frame_height=int(frame_height),
            timestamp=int(time.time() * 1000),
        )
        if sent:
            self._last_publish = now
            self._last_angle = target_angle
            return target_angle
        return None

    def recenter(self) -> Optional[float]:
        """Return the motor to its calibrated center angle after unlocking."""
        self._filtered_error = None
        self._tracking_origin_dy = 0.0
        self._pending_angle = self.center_angle
        return self.flush_pending()

    def flush_pending(self) -> Optional[float]:
        """Publish a queued center command when MQTT is available."""
        if self._pending_angle is None:
            return None
        target_angle = self._pending_angle
        if self.motor.publish_angle(target_angle, mode="recenter"):
            self._last_angle = target_angle
            self._last_publish = time.monotonic()
            self._pending_angle = None
            return target_angle
        return None
