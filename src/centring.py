"""
src/centring.py
Simulated Virtual-Pan Centring Controller.
Calculates horizontal error, proportional command with deadband and smoothing filter,
and integrates command into a virtual camera pan angle.
"""

from typing import Optional, Tuple
import numpy as np
from src.identity import TrackIdentity, State


class VirtualPanController:
    """
    Virtual pan camera controller simulating PTZ tracking without physical hardware.
    """

    def __init__(
        self,
        target_name: str = "",
        kp: float = 1.5,
        deadband: float = 0.03,
        pan_max: float = 30.0,
        max_angle: float = 45.0,
        smoothing_alpha: float = 0.35,
    ):
        self.target_name = target_name
        self.kp = kp
        self.deadband = deadband
        self.pan_max = pan_max
        self.max_angle = max_angle
        self.smoothing_alpha = smoothing_alpha

        # State variables
        self.simulated_pan_angle: float = 0.0  # degrees
        self.smoothed_cmd: float = 0.0  # degrees per second
        self.raw_cmd: float = 0.0
        self.error: float = 0.0  # normalized [-0.5, 0.5]
        self.status: str = "HOLD"  # "TRACKING" or "HOLD"
        self.target_cx: Optional[float] = None
        self.target_cy: Optional[float] = None
        self.locked_track_id: Optional[int] = None
        self.last_time: Optional[float] = None

    def set_target_name(self, name: str):
        """Change target identity to track."""
        self.target_name = name

    def reset_pan(self):
        """Reset simulated pan angle to center (0 degrees)."""
        self.simulated_pan_angle = 0.0
        self.smoothed_cmd = 0.0
        self.raw_cmd = 0.0
        self.error = 0.0
        self.status = "HOLD"

    def update(
        self,
        current_time: float,
        frame_width: int,
        tracks: dict[int, Tuple[TrackIdentity, Tuple[float, float, float, float]]],
    ) -> Tuple[float, float, float, str]:
        """
        Update controller state for current frame.
        Args:
            current_time: current timestamp in seconds
            frame_width: width of camera frame in pixels
            tracks: mapping track_id -> (TrackIdentity, (x, y, w, h))
        Returns:
            (error, smoothed_cmd, simulated_pan_angle, status)
        """
        dt = 0.033  # default ~30fps if first frame
        if self.last_time is not None and current_time > self.last_time:
            dt = min(current_time - self.last_time, 0.5)
        self.last_time = current_time

        # Find target track
        chosen_track = None
        chosen_bbox = None

        # 1. If target_name is specified, look for track holding target_name in Confirmed or Temporary
        if self.target_name:
            for tid, (track, bbox) in tracks.items():
                if track.name == self.target_name and track.state in (State.CONFIRMED, State.TEMPORARY):
                    chosen_track = track
                    chosen_bbox = bbox
                    break
        else:
            # If no target specified, lock onto first Confirmed track, or Temporary track
            confirmed_tracks = [
                (t, b) for t, b in tracks.values() if t.state == State.CONFIRMED and t.name != "Unknown"
            ]
            if confirmed_tracks:
                chosen_track, chosen_bbox = confirmed_tracks[0]
            else:
                temp_tracks = [
                    (t, b) for t, b in tracks.values() if t.state == State.TEMPORARY and t.name != "Unknown"
                ]
                if temp_tracks:
                    chosen_track, chosen_bbox = temp_tracks[0]

        if chosen_track is not None and chosen_bbox is not None:
            self.locked_track_id = chosen_track.track_id
            bx, by, bw, bh = chosen_bbox
            self.target_cx = bx + bw / 2.0
            self.target_cy = by + bh / 2.0

            # Horizontal error: normalized to [-0.5, 0.5]
            raw_err = (self.target_cx - frame_width / 2.0) / float(frame_width)
            self.error = float(np.clip(raw_err, -0.5, 0.5))

            # Apply deadband
            if abs(self.error) < self.deadband:
                effective_err = 0.0
            else:
                # Retain direction outside deadband
                effective_err = self.error

            # Proportional command scaled by pan_max
            # If effective_err is 0.5 (edge of frame), cmd reaches pan_max
            cmd = self.kp * (effective_err / 0.5) * self.pan_max
            self.raw_cmd = float(np.clip(cmd, -self.pan_max, self.pan_max))
            self.status = "TRACKING"
        else:
            # Target is Unknown or absent: Hold position, command 0
            self.locked_track_id = None
            self.target_cx = None
            self.target_cy = None
            self.error = 0.0
            self.raw_cmd = 0.0
            self.status = "HOLD"

        # Smoothing filter: exponential moving average
        self.smoothed_cmd = (
            self.smoothing_alpha * self.raw_cmd
            + (1.0 - self.smoothing_alpha) * self.smoothed_cmd
        )

        # Integrate simulated pan angle: angle += cmd * dt
        # When command > 0 (target is to the right), camera pans right (+angle)
        self.simulated_pan_angle += self.smoothed_cmd * dt
        self.simulated_pan_angle = float(
            np.clip(self.simulated_pan_angle, -self.max_angle, self.max_angle)
        )

        return self.error, self.smoothed_cmd, self.simulated_pan_angle, self.status
