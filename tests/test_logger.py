"""
tests/test_logger.py
Unit tests verifying per-frame CSV logging schema, data integrity, and metadata JSON structure.
"""

import csv
import json
import tempfile
from pathlib import Path
from src.logger import ExperimentLogger, CSV_COLUMNS


def test_logger_csv_schema_and_metadata():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            "seed": 42,
            "identity": {"half_life_s": 2.5, "confidence_floor": 0.3},
        }

        logger = ExperimentLogger(
            output_dir=tmpdir,
            session_name="test_session",
            config_snapshot=config,
        )

        assert logger.csv_path.exists()
        assert logger.meta_path.exists()

        # Check CSV header
        with open(logger.csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            assert header == CSV_COLUMNS

        # Log a sample frame with 2 tracks
        tracks = [
            {
                "track_id": 1,
                "bbox": (10.0, 20.0, 100.0, 200.0),
                "state": "Confirmed",
                "name": "Alice",
                "confidence": 0.88,
                "similarity": 0.88,
                "face_visible": True,
                "ambiguous": False,
            },
            {
                "track_id": 2,
                "bbox": (300.0, 40.0, 80.0, 180.0),
                "state": "Temporary",
                "name": "Bob",
                "confidence": 0.52,
                "similarity": 0.0,
                "face_visible": False,
                "ambiguous": True,
            },
        ]

        logger.set_event_marker("head_turn_start")
        logger.log_frame(
            frame_idx=1,
            timestamp=0.033,
            track_entries=tracks,
            centring_error=0.12,
            pan_cmd=5.4,
            pan_angle=1.2,
            fps=29.8,
        )

        # Check logged rows
        with open(logger.csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
            # Header + 2 track rows = 3 rows
            assert len(rows) == 3

            row1 = rows[1]
            assert row1[0] == "1"  # frame
            assert row1[2] == "1"  # track_id
            assert row1[7] == "Confirmed"
            assert row1[8] == "Alice"
            assert row1[11] == "1"  # face_visible
            assert row1[12] == "0"  # not ambiguous
            assert row1[17] == "head_turn_start"  # event_marker

            row2 = rows[2]
            assert row2[2] == "2"  # track_id
            assert row2[7] == "Temporary"
            assert row2[8] == "Bob"
            assert row2[11] == "0"  # face not visible
            assert row2[12] == "1"  # ambiguous

        # Check Metadata JSON
        with open(logger.meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            assert meta["session_name"] == "test_session"
            assert "created_at" in meta
            assert meta["config"]["seed"] == 42
            assert meta["columns"] == CSV_COLUMNS
