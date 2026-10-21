#!/usr/bin/env python3
"""
scripts/download_models.py
Download required ONNX models (YuNet and SFace) from the OpenCV Zoo,
and pre-fetch YOLOv8n weights so the pipeline operates completely offline.
"""

import sys
import os
import urllib.request
from pathlib import Path

# Paths
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# OpenCV Zoo Model URLs and fallback mirrors
# SFace full binary is ~38.7 MB (38,696,353 bytes)
# YuNet full binary is ~232 KB (232,589 bytes)
MODELS = {
    "face_detection_yunet_2023mar.onnx": {
        "min_size": 200_000,
        "urls": [
            "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
            "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
            "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        ],
    },
    "face_recognition_sface_2021dec.onnx": {
        "min_size": 35_000_000,
        "urls": [
            "https://huggingface.co/opencv/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx",
            "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        ],
    },
}

YOLO_URLS = [
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
]


def download_file(urls, dest_path: Path, min_size_bytes: int = 100_000) -> bool:
    if dest_path.exists() and dest_path.stat().st_size >= min_size_bytes:
        print(f"[OK] {dest_path.name} already exists ({dest_path.stat().st_size / 1024:.1f} KB).")
        return True

    temp_path = dest_path.with_name(f"{dest_path.name}.tmp")
    for url in urls:
        print(f"Downloading {dest_path.name} from {url}...")
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            )
            with urllib.request.urlopen(req, timeout=60) as response:
                total_size = int(response.info().get("Content-Length", -1))
                downloaded = 0
                chunk_size = 65536
                with open(temp_path, "wb") as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = downloaded * 100 / total_size
                            sys.stdout.write(f"\r  -> {percent:5.1f}% ({downloaded // 1024} KB)")
                            sys.stdout.flush()

            sys.stdout.write("\n")
            if temp_path.stat().st_size >= min_size_bytes:
                if dest_path.exists():
                    dest_path.unlink()
                temp_path.rename(dest_path)
                print(f"[SUCCESS] Saved to {dest_path} ({dest_path.stat().st_size / (1024*1024):.2f} MB)")
                return True
            else:
                print(f"[WARN] File too small ({temp_path.stat().st_size} bytes), might be Git LFS text pointer or incomplete.")
                if temp_path.exists():
                    temp_path.unlink()
        except Exception as e:
            print(f"[ERROR] Failed downloading from {url}: {e}")
            if temp_path.exists():
                temp_path.unlink()

    return False


def setup_yolov8(dest_path: Path):
    if dest_path.exists() and dest_path.stat().st_size > 1_000_000:
        print(f"[OK] {dest_path.name} already exists ({dest_path.stat().st_size / (1024*1024):.1f} MB).")
        return True

    # Try downloading via ultralytics if available
    try:
        from ultralytics import YOLO
        print("Pre-fetching YOLOv8n via ultralytics library...")
        model = YOLO("yolov8n.pt")
        import shutil
        src_pt = Path("yolov8n.pt")
        if src_pt.exists() and src_pt != dest_path:
            shutil.copy(src_pt, dest_path)
            print(f"[SUCCESS] Copied yolov8n.pt to {dest_path}")
            return True
        elif dest_path.exists():
            return True
    except Exception as e:
        print(f"ultralytics import/fetch failed: {e}. Falling back to direct URL download.")

    return download_file(YOLO_URLS, dest_path, min_size_bytes=1_000_000)


def main():
    print("=" * 60)
    print("Downloading models for Person Tracker & Face Association")
    print(f"Target directory: {MODELS_DIR}")
    print("=" * 60)

    success = True
    for filename, info in MODELS.items():
        target = MODELS_DIR / filename
        ok = download_file(info["urls"], target, min_size_bytes=info["min_size"])
        if not ok:
            print(f"[FAIL] Could not download {filename}!")
            success = False

    yolo_target = MODELS_DIR / "yolov8n.pt"
    ok_yolo = setup_yolov8(yolo_target)
    if not ok_yolo:
        print("[FAIL] Could not acquire yolov8n.pt!")
        success = False

    if success:
        print("\nAll models downloaded and verified successfully! Ready for offline execution.")
    else:
        print("\nWarning: One or more models failed to download.")
        sys.exit(1)


if __name__ == "__main__":
    main()
