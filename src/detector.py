"""
src/detector.py
Person detection and tracking using YOLOv8n + ByteTrack with graceful OpenCV HOG + IoU fallback.
Returns Track(id, bbox_xywh, det_conf).
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class Track:
    id: int
    bbox_xywh: Tuple[float, float, float, float]  # (x, y, w, h)
    det_conf: float


class SimpleIoUTracker:
    """
    Lightweight IoU tracker for OpenCV HOG fallback backend.
    """

    def __init__(self, iou_thresh: float = 0.3, max_lost_frames: int = 15):
        self.iou_thresh = iou_thresh
        self.max_lost_frames = max_lost_frames
        self.next_id = 1
        # {track_id: {"bbox": xywh, "lost": int, "conf": float}}
        self.tracks = {}

    @staticmethod
    def _iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
        inter_area = max(0.0, xB - xA) * max(0.0, yB - yA)
        areaA = max(0.0, boxA[2]) * max(0.0, boxA[3])
        areaB = max(0.0, boxB[2]) * max(0.0, boxB[3])
        denom = areaA + areaB - inter_area
        return inter_area / denom if denom > 0 else 0.0

    def update(self, detections: List[Tuple[Tuple[float, float, float, float], float]]) -> List[Track]:
        matched_tracks = set()
        matched_dets = set()
        results = []

        # Greedy IoU match
        for d_idx, (d_box, conf) in enumerate(detections):
            best_iou = self.iou_thresh
            best_tid = None
            for tid, t_data in self.tracks.items():
                if tid in matched_tracks:
                    continue
                iou = self._iou(d_box, t_data["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid

            if best_tid is not None:
                matched_tracks.add(best_tid)
                matched_dets.add(d_idx)
                self.tracks[best_tid]["bbox"] = d_box
                self.tracks[best_tid]["conf"] = conf
                self.tracks[best_tid]["lost"] = 0
                results.append(Track(id=best_tid, bbox_xywh=d_box, det_conf=conf))

        # Unmatched detections -> new tracks
        for d_idx, (d_box, conf) in enumerate(detections):
            if d_idx not in matched_dets:
                new_id = self.next_id
                self.next_id += 1
                self.tracks[new_id] = {"bbox": d_box, "lost": 0, "conf": conf}
                results.append(Track(id=new_id, bbox_xywh=d_box, det_conf=conf))

        # Unmatched tracks -> increment lost
        lost_to_remove = []
        for tid, t_data in self.tracks.items():
            if tid not in matched_tracks:
                t_data["lost"] += 1
                if t_data["lost"] > self.max_lost_frames:
                    lost_to_remove.append(tid)

        for tid in lost_to_remove:
            del self.tracks[tid]

        return results


class PersonDetectorTracker:
    """
    Person detector and tracker with selectable backend (YOLOv8n + ByteTrack vs HOG + IoU).
    Supports running heavy detection every N frames.
    """

    def __init__(
        self,
        backend: str = "yolo",
        model_path: str = "models/yolov8n.pt",
        tracker_config: str = "bytetrack.yaml",
        conf_threshold: float = 0.40,
        iou_threshold: float = 0.50,
        detect_interval: int = 2,
    ):
        self.backend = backend.lower()
        self.model_path = model_path
        self.tracker_config = tracker_config
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.detect_interval = max(1, detect_interval)

        self.yolo_model = None
        self.hog_detector = None
        self.iou_tracker = None
        self.cached_tracks: List[Track] = []

        self._init_backend()

    def _init_backend(self):
        if self.backend == "yolo":
            try:
                from ultralytics import YOLO
                print(f"[PersonDetector] Initializing YOLOv8n from {self.model_path}...")
                self.yolo_model = YOLO(self.model_path)
                print("[PersonDetector] YOLOv8n initialized successfully.")
            except Exception as e:
                print(f"[PersonDetector] Ultralytics YOLO initialization failed: {e}")
                print("[PersonDetector] Falling back to OpenCV HOG person detector.")
                self.backend = "hog"

        if self.backend == "hog":
            print("[PersonDetector] Initializing OpenCV HOG person detector...")
            self.hog_detector = cv2.HOGDescriptor()
            self.hog_detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            self.iou_tracker = SimpleIoUTracker(iou_thresh=0.25, max_lost_frames=20)

    def detect_and_track(self, frame: np.ndarray, frame_idx: int) -> List[Track]:
        """
        Process frame and return persistent tracks.
        Runs heavy detection on interval, yields cached tracks in between to reach target FPS.
        """
        if frame is None:
            return []

        # Run detector on scheduled frames, or if cache is empty
        run_detector = (frame_idx % self.detect_interval == 0) or (len(self.cached_tracks) == 0)

        if not run_detector:
            return list(self.cached_tracks)

        tracks: List[Track] = []

        if self.backend == "yolo" and self.yolo_model is not None:
            try:
                results = self.yolo_model.track(
                    source=frame,
                    persist=True,
                    tracker=self.tracker_config,
                    classes=[0],  # Person class only
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    verbose=False,
                    device="cpu",
                )

                if results and len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes
                    xywhs = boxes.xywh.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()
                    ids = boxes.id.cpu().numpy() if boxes.id is not None else None

                    for i in range(len(xywhs)):
                        cx, cy, w, h = xywhs[i]
                        x = cx - w / 2.0
                        y = cy - h / 2.0
                        conf = float(confs[i])
                        tid = int(ids[i]) if ids is not None else (i + 1)
                        tracks.append(Track(id=tid, bbox_xywh=(float(x), float(y), float(w), float(h)), det_conf=conf))
            except Exception as e:
                # If YOLO run fails, fallback to HOG on next iteration
                print(f"[PersonDetector] YOLO error during track(): {e}")

        elif self.backend == "hog" and self.hog_detector is not None:
            # Resize frame slightly for HOG speedup if large
            h, w = frame.shape[:2]
            scale = 1.0
            work_frame = frame
            if w > 640:
                scale = 640.0 / w
                work_frame = cv2.resize(frame, (640, int(h * scale)))

            rects, weights = self.hog_detector.detectMultiScale(
                work_frame,
                winStride=(8, 8),
                padding=(4, 4),
                scale=1.05,
            )

            detections = []
            for (rx, ry, rw, rh), weight in zip(rects, weights):
                orig_x = rx / scale
                orig_y = ry / scale
                orig_w = rw / scale
                orig_h = rh / scale
                detections.append(((orig_x, orig_y, orig_w, orig_h), float(weight)))

            if self.iou_tracker is not None:
                tracks = self.iou_tracker.update(detections)

        self.cached_tracks = tracks
        return tracks
