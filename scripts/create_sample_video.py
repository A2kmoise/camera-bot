"""
scripts/create_sample_video.py
Generates a realistic synthetic test video (videos/sample_replay.mp4) with moving body
and embedded real face crops to demonstrate and benchmark the 3-state pipeline.
Also generates matching ground-truth labels (videos/sample_replay_labels.csv).
"""

from pathlib import Path
import cv2
import numpy as np

VIDEOS_DIR = Path("videos")
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_PATH = VIDEOS_DIR / "sample_replay.mp4"
GT_PATH = VIDEOS_DIR / "sample_replay_labels.csv"

# Load a real face image from data/enroll/Kaliza
face_sample_path = Path("data/enroll/Kaliza/1791448298557_001.jpg")
if face_sample_path.exists():
    face_img = cv2.imread(str(face_sample_path))
else:
    face_img = np.zeros((100, 100, 3), dtype=np.uint8)


def create_video():
    width = 640
    height = 480
    fps = 30.0
    total_frames = 150

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(VIDEO_PATH), fourcc, fps, (width, height))

    gt_rows = ["frame,timestamp,track_id,gt_name,gt_face_visible,scenario,notes\n"]

    # Background: clean modern office gradient
    bg = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        ratio = y / height
        bg[y, :] = [int(180 - 40 * ratio), int(190 - 30 * ratio), int(200 - 20 * ratio)]

    # Person body parameters
    body_w = 120
    body_h = 320
    body_y = 110

    print(f"Generating synthetic evaluation video: {VIDEO_PATH} ({total_frames} frames)...")

    for f_idx in range(total_frames):
        ts = f_idx / fps
        frame = bg.copy()

        # Motion trajectory: sinusoidal sway left and right
        center_x = int(320 + 80 * np.sin(2 * np.pi * f_idx / 90.0))
        body_x = center_x - body_w // 2

        # 3 Phases:
        # Phase 1: frames 0..44 -> Face visible (Looking at camera)
        # Phase 2: frames 45..104 -> Head turned (Face lost, decay in Temporary)
        # Phase 3: frames 105..149 -> Face visible (Looking back, Confirmed recovery)
        face_visible = not (45 <= f_idx < 105)

        # Draw Person Body (torso + legs)
        # Torso (Navy Blue)
        cv2.rectangle(
            frame,
            (body_x + 10, body_y + 80),
            (body_x + body_w - 10, body_y + 220),
            (80, 50, 40),
            -1,
        )
        # Legs (Dark Denim)
        cv2.rectangle(
            frame,
            (body_x + 15, body_y + 220),
            (body_x + body_w // 2 - 5, body_y + body_h),
            (50, 30, 20),
            -1,
        )
        cv2.rectangle(
            frame,
            (body_x + body_w // 2 + 5, body_y + 220),
            (body_x + body_w - 15, body_y + body_h),
            (50, 30, 20),
            -1,
        )
        # Head boundary
        head_cx = center_x
        head_cy = body_y + 40
        head_r = 35

        if face_visible and face_img is not None and face_img.size > 0:
            # Composite real face crop
            face_resized = cv2.resize(face_img, (60, 70))
            fh, fw = face_resized.shape[:2]
            fx1 = max(0, head_cx - fw // 2)
            fy1 = max(0, head_cy - fh // 2)
            fx2 = min(width, fx1 + fw)
            fy2 = min(height, fy1 + fh)
            frame[fy1:fy2, fx1:fx2] = face_resized[: (fy2 - fy1), : (fx2 - fx1)]
        else:
            # Head turned away: draw back of head (hair color)
            cv2.circle(frame, (head_cx, head_cy), head_r, (30, 30, 35), -1)

        writer.write(frame)

        # Ground truth entry
        gt_vis = 1 if face_visible else 0
        note = "face_seen" if face_visible else "head_turned_away"
        gt_rows.append(f"{f_idx},{ts:.4f},1,kaliza,{gt_vis},head_turn,{note}\n")

    writer.release()
    with open(GT_PATH, "w", encoding="utf-8") as f:
        f.writelines(gt_rows)

    print(f"[SUCCESS] Created {VIDEO_PATH} and {GT_PATH}")


if __name__ == "__main__":
    create_video()
