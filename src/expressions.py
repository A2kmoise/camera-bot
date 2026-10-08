"""Landmark geometry estimates for expressions and blink counting.

These labels are simple visual cues for a classroom demo, not a trained
emotion-recognition model or a reliable measure of how someone feels.
"""
from __future__ import annotations

import numpy as np


def _distance(points, a, b):
    return float(np.linalg.norm(points[a] - points[b]))


class ExpressionTracker:
    """Estimate face expression and blinks from one face's MediaPipe points."""
    def __init__(self, blink_threshold: float = 0.18, min_closed_frames: int = 1):
        self.blink_threshold = blink_threshold
        self.min_closed_frames = min_closed_frames
        self.closed_frames = 0
        self.blink_count = 0
        self.was_closed = False

    def update(self, points):
        if points is None or len(points) <= 386:
            return None
        p = np.asarray(points, dtype=np.float32)
        # Standard six-point eye aspect ratio for each eye. Averaging two
        # vertical spans is more stable than a single landmark pair when the
        # face turns or individual points jitter slightly.
        def eye_aspect_ratio(outer, upper_a, upper_b, inner, lower_b, lower_a):
            vertical = _distance(p, upper_a, lower_a) + _distance(p, upper_b, lower_b)
            horizontal = max(1e-6, _distance(p, outer, inner))
            return vertical / (2.0 * horizontal)

        left_ear = eye_aspect_ratio(33, 160, 158, 133, 153, 144)
        right_ear = eye_aspect_ratio(362, 385, 387, 263, 373, 380)
        ear = (left_ear + right_ear) / 2.0
        eye_span = max(1e-6, _distance(p, 33, 263))
        # Hysteresis avoids rapid open/closed toggling around the threshold.
        closed = ear < (self.blink_threshold if not self.was_closed else self.blink_threshold * 1.35)
        if closed:
            self.closed_frames += 1
        else:
            if self.was_closed and self.closed_frames >= self.min_closed_frames:
                self.blink_count += 1
            self.closed_frames = 0
        self.was_closed = closed

        mouth_width = _distance(p, 61, 291) / eye_span
        mouth_open = _distance(p, 13, 14) / max(1e-6, _distance(p, 61, 291))
        mid_lip_y = (p[13, 1] + p[14, 1]) / 2
        corner_y = (p[61, 1] + p[291, 1]) / 2
        corner_lift = (mid_lip_y - corner_y) / eye_span
        eye_y = (p[33, 1] + p[263, 1]) / 2
        brow_raise = (eye_y - (p[105, 1] + p[334, 1]) / 2) / eye_span

        # Heuristic ordering gives stretched, tense mouths precedence over smiles.
        if mouth_width >= .82 and mouth_open < .16 and corner_lift < .025:
            expression = "Grimacing"
        elif corner_lift < -.055 and brow_raise >= .23:
            expression = "Sad"
        elif corner_lift < -.035:
            expression = "Frowning"
        elif corner_lift >= .018 or mouth_width >= .78:
            expression = "Smiling"
        else:
            expression = "Neutral"

        return {
            "expression": expression,
            "eyes": "CLOSED" if closed else "OPEN",
            "blink_count": self.blink_count,
            "ear": ear,
        }
