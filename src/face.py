"""
src/face.py
Face detection with YuNet, feature extraction with SFace, and cosine similarity matching
against enrolled identities.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


class FaceDetection:
    """Container for a detected face."""

    def __init__(self, bbox_xywh: Tuple[float, float, float, float], score: float, raw_face: np.ndarray):
        self.bbox_xywh = bbox_xywh  # (x, y, w, h)
        self.score = score
        self.raw_face = raw_face
        self.aligned_crop: Optional[np.ndarray] = None
        self.embedding: Optional[np.ndarray] = None
        self.matched_name: Optional[str] = None
        self.similarity: float = 0.0


class FaceEngine:
    """
    Manages YuNet detector and SFace recognizer, face quality filtering,
    and cosine similarity matching against enrolled identities.
    """

    def __init__(
        self,
        yunet_path: str = "models/face_detection_yunet_2023mar.onnx",
        sface_path: str = "models/face_recognition_sface_2021dec.onnx",
        score_threshold: float = 0.60,
        nms_threshold: float = 0.30,
        top_k: int = 5000,
        min_face_size: int = 20,
        match_threshold: float = 0.363,
        enrolled_dir: str = "data/enrolled",
    ):
        self.yunet_path = Path(yunet_path)
        self.sface_path = Path(sface_path)
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self.min_face_size = min_face_size
        self.match_threshold = match_threshold
        self.enrolled_dir = Path(enrolled_dir)
        self.enrolled_dir.mkdir(parents=True, exist_ok=True)

        self.detector: Optional[cv2.FaceDetectorYN] = None
        self.recognizer: Optional[cv2.FaceRecognizerSF] = None
        self.current_input_size: Tuple[int, int] = (640, 480)

        # Enrolled identities: {name: normalized_embedding_128D}
        self.enrolled: Dict[str, np.ndarray] = {}

        self._init_models()
        self.reload_enrolled()

    def _init_models(self):
        """Initialize YuNet and SFace models if present."""
        if self.yunet_path.exists():
            try:
                self.detector = cv2.FaceDetectorYN.create(
                    model=str(self.yunet_path),
                    config="",
                    input_size=self.current_input_size,
                    score_threshold=self.score_threshold,
                    nms_threshold=self.nms_threshold,
                    top_k=self.top_k,
                )
            except Exception as e:
                print(f"[FaceEngine] Error loading YuNet: {e}")
                self.detector = None
        else:
            print(f"[FaceEngine] YuNet model not found at {self.yunet_path}")

        if self.sface_path.exists():
            try:
                self.recognizer = cv2.FaceRecognizerSF.create(
                    model=str(self.sface_path),
                    config="",
                )
            except Exception as e:
                print(f"[FaceEngine] Error loading SFace: {e}")
                self.recognizer = None
        else:
            print(f"[FaceEngine] SFace model not found at {self.sface_path}")

    def reload_enrolled(self):
        """Load or reload all enrolled face embeddings from directory."""
        self.enrolled.clear()
        if not self.enrolled_dir.exists():
            return

        for npy_file in self.enrolled_dir.glob("*.npy"):
            try:
                emb = np.load(npy_file)
                norm = np.linalg.norm(emb)
                if norm > 1e-6:
                    self.enrolled[npy_file.stem] = emb.flatten() / norm
                else:
                    self.enrolled[npy_file.stem] = emb.flatten()
            except Exception as e:
                print(f"[FaceEngine] Error loading {npy_file}: {e}")

    @staticmethod
    def compute_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two 1D vectors."""
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0
        return float(np.dot(emb1, emb2) / (norm1 * norm2))

    def detect_faces(self, frame: np.ndarray) -> List[FaceDetection]:
        """
        Detect faces in frame using YuNet and apply quality filters.
        """
        if self.detector is None or frame is None:
            return []

        h, w = frame.shape[:2]
        if (w, h) != self.current_input_size:
            self.current_input_size = (w, h)
            self.detector.setInputSize(self.current_input_size)

        _, raw_faces = self.detector.detect(frame)
        if raw_faces is None or len(raw_faces) == 0:
            return []

        valid_detections = []
        for face_data in raw_faces:
            fx, fy, fw, fh = float(face_data[0]), float(face_data[1]), float(face_data[2]), float(face_data[3])
            score = float(face_data[-1])

            # Quality filters:
            # 1. Score threshold
            if score < self.score_threshold:
                continue
            # 2. Minimum face size
            if fw < self.min_face_size or fh < self.min_face_size:
                continue

            # Clip bounding box to frame dimensions
            fx_clipped = max(0.0, fx)
            fy_clipped = max(0.0, fy)
            fw_clipped = min(w - fx_clipped, fw)
            fh_clipped = min(h - fy_clipped, fh)

            if fw_clipped <= 0 or fh_clipped <= 0:
                continue

            det = FaceDetection(
                bbox_xywh=(fx_clipped, fy_clipped, fw_clipped, fh_clipped),
                score=score,
                raw_face=face_data,
            )
            valid_detections.append(det)

        return valid_detections

    def extract_and_match(
        self,
        frame: np.ndarray,
        face_det: FaceDetection,
    ) -> Tuple[Optional[str], float, Optional[np.ndarray]]:
        """
        Extract aligned feature embedding using SFace and match against enrolled identities.
        Returns:
            (best_name, max_similarity, normalized_embedding)
        """
        if self.recognizer is None or face_det.raw_face is None:
            return None, 0.0, None

        try:
            # Align face crop
            aligned = self.recognizer.alignCrop(frame, face_det.raw_face)
            face_det.aligned_crop = aligned

            # Extract feature
            feat = self.recognizer.feature(aligned).flatten()
            norm = np.linalg.norm(feat)
            if norm > 1e-6:
                feat = feat / norm
            face_det.embedding = feat

            # Match against enrolled database
            best_name = None
            max_sim = -1.0

            for name, enrolled_emb in self.enrolled.items():
                sim = self.compute_cosine_similarity(feat, enrolled_emb)
                if sim > max_sim:
                    max_sim = sim
                    best_name = name

            if max_sim >= self.match_threshold and best_name is not None:
                face_det.matched_name = best_name
                face_det.similarity = max_sim
                return best_name, max_sim, feat
            else:
                face_det.matched_name = None
                face_det.similarity = max(0.0, max_sim)
                return None, max(0.0, max_sim), feat

        except Exception as e:
            print(f"[FaceEngine] Recognition error: {e}")
            return None, 0.0, None
