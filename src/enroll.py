"""
src/enroll.py
Enrolment CLI: Interactive face capture tool guiding user through capturing ~20 face samples,
computing the normalized mean SFace embedding, and storing in data/enrolled/<name>.npy.
"""

import argparse
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import yaml

from src.capture import VideoCaptureWrapper
from src.face import FaceEngine


def load_config(config_path: str = "config.yaml") -> dict:
    if Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def enroll_cli():
    parser = argparse.ArgumentParser(description="Enrol a new face identity into data/enrolled/<name>.npy")
    parser.add_argument("--name", type=str, required=True, help="Identity name (e.g. 'alice', 'me')")
    parser.add_argument("--samples", type=int, default=20, help="Number of face samples to capture (default 20)")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default 0)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    face_cfg = cfg.get("face", {})
    cap_cfg = cfg.get("capture", {})

    enrolled_dir = Path(face_cfg.get("enrolled_dir", "data/enrolled"))
    enrolled_dir.mkdir(parents=True, exist_ok=True)
    target_file = enrolled_dir / f"{args.name}.npy"

    print("=" * 60)
    print(f"Enrolment for: {args.name}")
    print(f"Target file: {target_file}")
    print(f"Requested samples: {args.samples}")
    print("=" * 60)

    # Initialize FaceEngine
    engine = FaceEngine(
        yunet_path=face_cfg.get("yunet_model_path", "models/face_detection_yunet_2023mar.onnx"),
        sface_path=face_cfg.get("sface_model_path", "models/face_recognition_sface_2021dec.onnx"),
        score_threshold=face_cfg.get("score_threshold", 0.60),
        min_face_size=face_cfg.get("min_face_size", 20),
    )

    if engine.detector is None or engine.recognizer is None:
        print("[ERROR] YuNet or SFace models could not be loaded!")
        print("Please run: python scripts/download_models.py first.")
        sys.exit(1)

    cam_id = args.camera if args.camera is not None else cap_cfg.get("camera_id", 0)
    cap = VideoCaptureWrapper(
        source=cam_id,
        width=cap_cfg.get("width", 640),
        height=cap_cfg.get("height", 480),
        target_fps=cap_cfg.get("target_fps", 30),
    )

    captured_embeddings = []
    print("\nInstructions:")
    print(" - Look at the camera with varied expressions and slight head angles (left, right, tilt, smile).")
    print(" - Press SPACE to capture a sample.")
    print(" - Press 'q' to abort.")
    print("-" * 60)

    flash_timer = 0
    while len(captured_embeddings) < args.samples:
        ret, frame, _ = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        h, w = frame.shape[:2]
        faces = engine.detect_faces(frame)

        display_frame = frame.copy()
        current_candidate_emb = None

        if faces:
            # Pick largest face
            largest_face = max(faces, key=lambda f: f.bbox_xywh[2] * f.bbox_xywh[3])
            fx, fy, fw, fh = [int(v) for v in largest_face.bbox_xywh]

            _, _, emb = engine.extract_and_match(frame, largest_face)
            if emb is not None:
                current_candidate_emb = emb

            # Draw face box
            color = (0, 255, 0) if current_candidate_emb is not None else (0, 0, 255)
            cv2.rectangle(display_frame, (fx, fy), (fx + fw, fy + fh), color, 2)
            cv2.putText(
                display_frame,
                f"Face ({largest_face.score:.2f})",
                (fx, max(20, fy - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )

        # Flash visual effect on capture
        if flash_timer > 0:
            flash_timer -= 1
            cv2.rectangle(display_frame, (0, 0), (w, h), (255, 255, 255), 8)

        # Status HUD
        progress_text = f"Samples: {len(captured_embeddings)} / {args.samples}"
        cv2.putText(
            display_frame,
            progress_text,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )
        instruction_text = "Press SPACE to capture | 'q' to quit"
        cv2.putText(
            display_frame,
            instruction_text,
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            2,
        )

        cv2.imshow(f"Enrolment - {args.name}", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("[ABORT] Enrolment cancelled by user.")
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)
        elif key == 32:  # SPACE bar
            if current_candidate_emb is not None:
                captured_embeddings.append(current_candidate_emb)
                flash_timer = 5
                print(f"Captured sample {len(captured_embeddings)} / {args.samples}")
            else:
                print("No valid face detected! Adjust position and try again.")

    cap.release()
    cv2.destroyAllWindows()

    # Compute normalized mean embedding
    embeddings_matrix = np.array(captured_embeddings)
    mean_emb = np.mean(embeddings_matrix, axis=0)
    norm = np.linalg.norm(mean_emb)
    if norm > 1e-6:
        mean_emb = mean_emb / norm

    np.save(target_file, mean_emb)
    print("\n" + "=" * 60)
    print(f"[SUCCESS] Enrolment complete for '{args.name}'!")
    print(f"Saved mean embedding ({mean_emb.shape}) to: {target_file}")
    print("=" * 60)


if __name__ == "__main__":
    enroll_cli()
