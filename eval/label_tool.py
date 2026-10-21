"""
eval/label_tool.py
Interactive and batch ground-truth labelling tool for recorded videos and pipeline logs.
Allows annotating true identity (gt_name) and face visibility (gt_face_visible) per track/frame.
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional
import cv2


GROUND_TRUTH_COLUMNS = [
    "frame",
    "timestamp",
    "track_id",
    "gt_name",
    "gt_face_visible",
    "scenario",
    "notes",
]


class LabelTool:
    """
    Step-through video annotation interface for ground-truth identity and visibility labels.
    """

    def __init__(
        self,
        video_path: str,
        output_csv: Optional[str] = None,
        default_identity: str = "Alice",
        scenario: str = "default",
    ):
        self.video_path = Path(video_path)
        if output_csv is None:
            self.output_csv = self.video_path.parent / f"{self.video_path.stem}_labels.csv"
        else:
            self.output_csv = Path(output_csv)

        self.default_identity = default_identity
        self.scenario = scenario
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video file: {self.video_path}")

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

        # Annotations: frame_idx -> dict of track_id -> {gt_name, gt_face_visible, notes}
        self.annotations: Dict[int, Dict[int, dict]] = {}
        self._load_existing_labels()

    def _load_existing_labels(self):
        if self.output_csv.exists():
            with open(self.output_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    f_idx = int(row["frame"])
                    tid = int(row["track_id"]) if row.get("track_id") else 1
                    self.annotations.setdefault(f_idx, {})[tid] = {
                        "gt_name": row["gt_name"],
                        "gt_face_visible": row["gt_face_visible"].lower() in ("1", "true", "yes"),
                        "notes": row.get("notes", ""),
                    }
            print(f"[LabelTool] Loaded existing labels for {len(self.annotations)} frames from {self.output_csv}")

    def save(self):
        """Save annotations to CSV."""
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for f_idx in sorted(self.annotations.keys()):
            ts = f_idx / self.fps
            for tid, data in self.annotations[f_idx].items():
                rows.append([
                    f_idx,
                    f"{ts:.4f}",
                    tid,
                    data.get("gt_name", self.default_identity),
                    1 if data.get("gt_face_visible", True) else 0,
                    self.scenario,
                    data.get("notes", ""),
                ])

        with open(self.output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(GROUND_TRUTH_COLUMNS)
            writer.writerows(rows)
        print(f"[LabelTool] Saved {len(rows)} label entries to {self.output_csv}")

    def run_interactive(self):
        current_frame = 0
        current_tid = 1
        current_face_visible = True
        current_name = self.default_identity

        print("\n" + "=" * 60)
        print(" INTERACTIVE GROUND-TRUTH LABELLING TOOL")
        print(f" Video: {self.video_path.name} ({self.total_frames} frames)")
        print(f" Output: {self.output_csv}")
        print(" Controls:")
        print("   RIGHT / 'd': Next frame")
        print("   LEFT  / 'a': Previous frame")
        print("   UP    / 'w': Skip forward 15 frames")
        print("   DOWN  / 's': Skip backward 15 frames")
        print("   'v': Toggle face visibility (currently Visible/Occluded)")
        print("   '1': Set Name -> Alice")
        print("   '2': Set Name -> Bob")
        print("   '0': Set Name -> Unknown / Impostor")
        print("   'c': Confirm/Store label for current frame")
        print("   's': Save labels to CSV")
        print("   'q': Save and Quit")
        print("=" * 60)

        while True:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = self.cap.read()
            if not ret or frame is None:
                print("[LabelTool] End of video or read error.")
                break

            h, w = frame.shape[:2]
            display = frame.copy()

            # Existing label if present
            existing = self.annotations.get(current_frame, {}).get(current_tid)
            if existing:
                lbl_name = existing["gt_name"]
                lbl_vis = existing["gt_face_visible"]
            else:
                lbl_name = current_name
                lbl_vis = current_face_visible

            # Draw HUD
            cv2.rectangle(display, (0, 0), (w, 55), (20, 20, 20), -1)
            status_text = f"Frame: {current_frame}/{self.total_frames} | Track: {current_tid} | Identity: {lbl_name} | FaceVis: {'YES' if lbl_vis else 'NO'}"
            color = (0, 255, 0) if lbl_vis else (0, 165, 255)
            cv2.putText(display, status_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.imshow("Ground Truth Label Tool", display)
            key = cv2.waitKey(0) & 0xFF

            if key in (ord("q"), 27):  # 'q' or ESC
                break
            elif key in (ord("d"), 83):  # Right arrow or 'd'
                # Auto-record current label and advance
                self.annotations.setdefault(current_frame, {})[current_tid] = {
                    "gt_name": lbl_name,
                    "gt_face_visible": lbl_vis,
                    "notes": "",
                }
                current_frame = min(self.total_frames - 1, current_frame + 1)
            elif key in (ord("a"), 81):  # Left arrow or 'a'
                current_frame = max(0, current_frame - 1)
            elif key in (ord("w"), 82):  # Up arrow
                current_frame = min(self.total_frames - 1, current_frame + 15)
            elif key in (ord("s"), 84):  # Down arrow
                current_frame = max(0, current_frame - 15)
            elif key == ord("v"):
                current_face_visible = not current_face_visible
            elif key == ord("1"):
                current_name = "Alice"
            elif key == ord("2"):
                current_name = "Bob"
            elif key == ord("0"):
                current_name = "Unknown"
            elif key == ord("c"):
                self.annotations.setdefault(current_frame, {})[current_tid] = {
                    "gt_name": current_name,
                    "gt_face_visible": current_face_visible,
                    "notes": "",
                }
                print(f"[LABELLED] Frame {current_frame}: {current_name}, Vis={current_face_visible}")

        self.save()
        self.cap.release()
        cv2.destroyAllWindows()


def generate_synthetic_labels(
    log_csv_path: str,
    output_labels_path: str,
    true_identity: str = "Alice",
    head_turn_ranges: Optional[List[Tuple[int, int]]] = None,
):
    """
    Generate ground-truth labels for a log file by identifying event markers or frame ranges.
    """
    head_turn_ranges = head_turn_ranges or []
    rows = []
    with open(log_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            f_idx = int(row["frame"])
            ts = float(row["timestamp"])
            tid = row.get("track_id", "1")
            if not tid:
                continue

            # Check if within head turn / occlusion ranges
            is_vis = True
            for start_f, end_f in head_turn_ranges:
                if start_f <= f_idx <= end_f:
                    is_vis = False
                    break

            rows.append([
                f_idx,
                f"{ts:.4f}",
                int(tid),
                true_identity,
                1 if is_vis else 0,
                "synthetic_evaluation",
                "",
            ])

    Path(output_labels_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_labels_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(GROUND_TRUTH_COLUMNS)
        writer.writerows(rows)
    print(f"[Synthetic Labels] Generated {len(rows)} ground truth rows in {output_labels_path}")


def main():
    parser = argparse.ArgumentParser(description="Ground Truth Video/Log Labelling Tool")
    parser.add_argument("--video", type=str, default=None, help="Path to video file")
    parser.add_argument("--log", type=str, default=None, help="Path to log CSV (for synthetic generation)")
    parser.add_argument("--output", type=str, default=None, help="Output label CSV path")
    parser.add_argument("--name", type=str, default="Alice", help="Default true identity name")
    parser.add_argument("--scenario", type=str, default="default", help="Scenario name")
    args = parser.parse_args()

    if args.video:
        tool = LabelTool(
            video_path=args.video,
            output_csv=args.output,
            default_identity=args.name,
            scenario=args.scenario,
        )
        tool.run_interactive()
    elif args.log:
        out = args.output if args.output else str(Path(args.log).with_name(f"{Path(args.log).stem}_labels.csv"))
        generate_synthetic_labels(args.log, out, true_identity=args.name)
    else:
        print("Usage: python -m eval.label_tool --video <path_to_video> OR --log <path_to_log.csv>")


if __name__ == "__main__":
    main()
