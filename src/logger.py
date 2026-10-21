"""
src/logger.py
Per-frame CSV logging and run-metadata JSON persistence for experiment reproducibility.
"""

import csv
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


CSV_COLUMNS = [
    "frame",
    "timestamp",
    "track_id",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "state",
    "name",
    "confidence",
    "similarity",
    "face_visible",
    "ambiguous",
    "centring_error",
    "pan_cmd",
    "pan_angle",
    "fps",
    "event_marker",
]


def get_git_hash() -> Optional[str]:
    """Retrieve current git commit hash if repository is present."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        return out.decode("utf-8").strip()
    except Exception:
        return None


class ExperimentLogger:
    """
    Handles synchronized per-frame CSV logging and metadata persistence.
    """

    def __init__(
        self,
        output_dir: str = "logs",
        session_name: Optional[str] = None,
        config_snapshot: Optional[Dict[str, Any]] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if session_name is None:
            session_name = f"run_{time.strftime('%Y%m%d_%H%M%S')}"

        self.session_name = session_name
        self.csv_path = self.output_dir / f"{session_name}.csv"
        self.meta_path = self.output_dir / f"{session_name}_meta.json"

        self.config_snapshot = config_snapshot or {}
        self.current_event_marker: str = ""

        self._init_csv()
        self._save_metadata()

    def _init_csv(self):
        """Write CSV header."""
        with open(self.csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_COLUMNS)

    def _save_metadata(self):
        """Write metadata JSON file."""
        meta = {
            "session_name": self.session_name,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "git_hash": get_git_hash(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "processor": platform.processor(),
            "config": self.config_snapshot,
            "columns": CSV_COLUMNS,
        }
        with open(self.meta_path, mode="w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def set_event_marker(self, marker: str):
        """Set an event marker for subsequent frames or single frame."""
        self.current_event_marker = marker

    def pop_event_marker(self) -> str:
        """Read and clear single-shot event marker."""
        marker = self.current_event_marker
        self.current_event_marker = ""
        return marker

    def log_frame(
        self,
        frame_idx: int,
        timestamp: float,
        track_entries: List[Dict[str, Any]],
        centring_error: float,
        pan_cmd: float,
        pan_angle: float,
        fps: float,
        event_marker: Optional[str] = None,
    ):
        """
        Log all tracks for a single video frame.
        If no tracks are present, logs a single row with empty track fields.
        """
        marker = event_marker if event_marker is not None else self.pop_event_marker()

        rows = []
        if not track_entries:
            # Empty frame log row
            rows.append([
                frame_idx,
                f"{timestamp:.4f}",
                "",
                "",
                "",
                "",
                "",
                "Unknown",
                "None",
                "0.000",
                "0.000",
                0,
                0,
                f"{centring_error:.4f}",
                f"{pan_cmd:.3f}",
                f"{pan_angle:.3f}",
                f"{fps:.2f}",
                marker,
            ])
        else:
            for entry in track_entries:
                bx, by, bw, bh = entry.get("bbox", (0, 0, 0, 0))
                rows.append([
                    frame_idx,
                    f"{timestamp:.4f}",
                    entry.get("track_id", ""),
                    f"{bx:.1f}",
                    f"{by:.1f}",
                    f"{bw:.1f}",
                    f"{bh:.1f}",
                    entry.get("state", "Unknown"),
                    entry.get("name", "Unknown"),
                    f"{entry.get('confidence', 0.0):.4f}",
                    f"{entry.get('similarity', 0.0):.4f}",
                    1 if entry.get("face_visible", False) else 0,
                    1 if entry.get("ambiguous", False) else 0,
                    f"{centring_error:.4f}",
                    f"{pan_cmd:.3f}",
                    f"{pan_angle:.3f}",
                    f"{fps:.2f}",
                    marker,
                ])

        with open(self.csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
