"""
tests/test_centring.py
Unit tests for simulated virtual-pan controller: proportional control,
deadband, command clipping, angle integration limits, and HOLD state.
"""

import pytest
from src.centring import VirtualPanController
from src.identity import TrackIdentity, State


def test_centring_proportional_and_deadband():
    """
    Verify error calculation, deadband behavior, and proportional direction.
    """
    controller = VirtualPanController(
        target_name="Alice",
        kp=1.0,
        deadband=0.04,
        pan_max=30.0,
        max_angle=45.0,
        smoothing_alpha=1.0,  # No smoothing for direct test
    )

    track = TrackIdentity(track_id=1)
    track.state = State.CONFIRMED
    track.name = "Alice"

    # Frame width = 640. Center is 320.
    # 1. Target exactly in center: target_cx = 320 -> err = 0.0 -> pan_cmd = 0.0
    tracks = {1: (track, (270.0, 100.0, 100.0, 200.0))}  # cx = 270 + 50 = 320
    err, cmd, angle, status = controller.update(current_time=1.0, frame_width=640, tracks=tracks)
    assert abs(err) < 1e-4
    assert abs(cmd) < 1e-4
    assert status == "TRACKING"

    # 2. Target within deadband: target_cx = 330 -> err = (330-320)/640 = 0.0156 < 0.04
    tracks = {1: (track, (280.0, 100.0, 100.0, 200.0))}  # cx = 330
    err, cmd, angle, status = controller.update(current_time=1.033, frame_width=640, tracks=tracks)
    assert abs(err) > 0.0
    assert cmd == 0.0  # Zero command due to deadband!

    # 3. Target to the right outside deadband: target_cx = 480 -> err = (480-320)/640 = +0.25
    tracks = {1: (track, (430.0, 100.0, 100.0, 200.0))}  # cx = 480
    err, cmd, angle, status = controller.update(current_time=1.066, frame_width=640, tracks=tracks)
    assert err > 0.20
    assert cmd > 0.0  # Positive command to pan right!

    # 4. Target to the left outside deadband: target_cx = 160 -> err = (160-320)/640 = -0.25
    tracks = {1: (track, (110.0, 100.0, 100.0, 200.0))}  # cx = 160
    err, cmd, angle, status = controller.update(current_time=1.100, frame_width=640, tracks=tracks)
    assert err < -0.20
    assert cmd < 0.0  # Negative command to pan left!


def test_centring_command_and_angle_clipping():
    """
    Verify command is clipped to [-pan_max, pan_max] and angle to [-max_angle, max_angle].
    """
    controller = VirtualPanController(
        target_name="Alice",
        kp=10.0,  # High gain to force clipping
        deadband=0.01,
        pan_max=25.0,
        max_angle=40.0,
        smoothing_alpha=1.0,
    )

    track = TrackIdentity(track_id=1)
    track.state = State.CONFIRMED
    track.name = "Alice"

    # Extreme target at far right edge (cx = 640)
    tracks = {1: (track, (590.0, 100.0, 100.0, 200.0))}
    err, cmd, angle, status = controller.update(current_time=1.0, frame_width=640, tracks=tracks)
    assert cmd == 25.0  # Clipped exactly to +pan_max!

    # Simulate integration over multiple seconds to test angle limit
    for t in range(1, 20):
        _, _, angle, _ = controller.update(current_time=1.0 + t * 0.5, frame_width=640, tracks=tracks)

    assert angle <= 40.0  # Clipped to max_angle!


def test_centring_hold_when_target_unknown_or_absent():
    """
    When the target identity is Unknown, absent, or lost, controller must output
    command 0.0 and report status 'HOLD'.
    """
    controller = VirtualPanController(target_name="Alice")

    # 1. Target track is Unknown state
    track = TrackIdentity(track_id=1)
    track.state = State.UNKNOWN
    track.name = "Unknown"
    tracks = {1: (track, (100.0, 100.0, 100.0, 200.0))}

    err, cmd, angle, status = controller.update(current_time=1.0, frame_width=640, tracks=tracks)
    assert status == "HOLD"
    assert cmd == 0.0

    # 2. No tracks present
    err, cmd, angle, status = controller.update(current_time=1.033, frame_width=640, tracks={})
    assert status == "HOLD"
    assert cmd == 0.0
