"""Horizontal face-position measurements for the live camera view."""
from __future__ import annotations

from typing import NamedTuple


class HorizontalPosition(NamedTuple):
    nose_offset: float
    face_offset: float
    face_center: float
    direction: str


def measure_horizontal(face_box: tuple[int, int, int, int], nose_x: float,
                       frame_width: int, deadband_ratio: float = 0.04) -> HorizontalPosition:
    """Measure horizontal offsets and direction without controlling the motor."""
    if frame_width <= 0:
        raise ValueError("Frame width must be positive")
    x1, _, x2, _ = face_box
    face_center = (x1 + x2) / 2.0
    nose_offset = float(nose_x) - frame_width / 2.0
    face_offset = face_center - frame_width / 2.0
    if abs(nose_offset) < frame_width * deadband_ratio:
        direction = "center"
    else:
        direction = "right" if nose_offset > 0 else "left"
    return HorizontalPosition(nose_offset, face_offset, face_center, direction)
