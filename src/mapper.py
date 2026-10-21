"""
src/mapper.py
Face-to-Body Association using geometric head constraints, multi-feature scoring,
global bipartite matching (scipy linear_sum_assignment), and body-overlap ambiguity flagging.
"""

from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment


def compute_iou(boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
    """
    Compute Intersection over Union (IoU) of two boxes in xywh format.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = max(0.0, boxA[2]) * max(0.0, boxA[3])
    areaB = max(0.0, boxB[2]) * max(0.0, boxB[3])
    union_area = areaA + areaB - inter_area

    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


class FaceBodyMapper:
    """
    Associates detected face bounding boxes to tracked person bounding boxes.
    """

    def __init__(
        self,
        upper_body_ratio: float = 0.40,
        min_face_width_ratio: float = 0.12,
        max_face_width_ratio: float = 0.65,
        min_assignment_score: float = 0.40,
        overlap_iou_threshold: float = 0.25,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.upper_body_ratio = upper_body_ratio
        self.min_face_width_ratio = min_face_width_ratio
        self.max_face_width_ratio = max_face_width_ratio
        self.min_assignment_score = min_assignment_score
        self.overlap_iou_threshold = overlap_iou_threshold

        default_weights = {
            "containment": 0.35,
            "horizontal": 0.35,
            "size": 0.20,
            "vertical": 0.10,
        }
        self.weights = weights if weights is not None else default_weights

    def find_ambiguous_tracks(
        self,
        tracks: List[Tuple[int, Tuple[float, float, float, float]]],
    ) -> Set[int]:
        """
        Flag tracks whose body bounding boxes overlap significantly (path crossing).
        """
        ambiguous_set: Set[int] = set()
        n = len(tracks)
        for i in range(n):
            tid_i, box_i = tracks[i]
            for j in range(i + 1, n):
                tid_j, box_j = tracks[j]
                iou = compute_iou(box_i, box_j)
                if iou >= self.overlap_iou_threshold:
                    ambiguous_set.add(tid_i)
                    ambiguous_set.add(tid_j)
        return ambiguous_set

    def compute_match_score(
        self,
        face_box: Tuple[float, float, float, float],
        body_box: Tuple[float, float, float, float],
    ) -> float:
        """
        Compute association score in [0.0, 1.0] between a face box and body box.
        Returns 0.0 if hard constraints fail.
        """
        fx, fy, fw, fh = face_box
        bx, by, bw, bh = body_box

        if bw <= 0 or bh <= 0 or fw <= 0 or fh <= 0:
            return 0.0

        fcx = fx + fw / 2.0
        fcy = fy + fh / 2.0

        # Hard Constraint 1: Face center must lie inside body box
        if not (bx <= fcx <= bx + bw and by <= fcy <= by + bh):
            return 0.0

        # Hard Constraint 2: Face center must lie within upper portion (e.g. top 40%)
        head_region_h = bh * self.upper_body_ratio
        if fcy > by + head_region_h:
            return 0.0

        # Size plausibility ratio: face_w / body_w
        size_ratio = fw / float(bw)
        if size_ratio < self.min_face_width_ratio * 0.7 or size_ratio > self.max_face_width_ratio * 1.3:
            return 0.0

        # Feature 1: Containment
        # Fully inside body gives 1.0, center only gives 0.8
        is_fully_inside = (
            fx >= bx and (fx + fw) <= (bx + bw) and
            fy >= by and (fy + fh) <= (by + bh)
        )
        contain_score = 1.0 if is_fully_inside else 0.85

        # Feature 2: Horizontal closeness to head axis (center of body)
        head_axis_x = bx + bw / 2.0
        dx = abs(fcx - head_axis_x)
        horiz_norm = dx / (bw / 2.0)  # 0 at center axis, 1 at body edge
        horiz_score = max(0.0, 1.0 - horiz_norm)

        # Feature 3: Size plausibility
        # Optimal ratio ~ 0.32
        target_ratio = 0.32
        ratio_span = (self.max_face_width_ratio - self.min_face_width_ratio) / 2.0
        size_diff = abs(size_ratio - target_ratio)
        size_score = max(0.0, 1.0 - (size_diff / max(ratio_span, 0.1)))

        # Feature 4: Vertical head closeness
        # Closer to top of head region gives higher score
        vert_norm = (fcy - by) / max(head_region_h, 1e-4)
        vert_score = max(0.0, 1.0 - 0.4 * vert_norm)

        total_score = (
            self.weights["containment"] * contain_score
            + self.weights["horizontal"] * horiz_score
            + self.weights["size"] * size_score
            + self.weights["vertical"] * vert_score
        )

        return float(np.clip(total_score, 0.0, 1.0))

    def match(
        self,
        faces: List[Tuple[float, float, float, float]],
        tracks: List[Tuple[int, Tuple[float, float, float, float]]],
    ) -> Tuple[Dict[int, int], Set[int]]:
        """
        Match detected faces to tracked bodies globally using scipy linear_sum_assignment.
        Args:
            faces: list of face bounding boxes (x, y, w, h)
            tracks: list of (track_id, (x, y, w, h))
        Returns:
            assignments: dict mapping track_id -> face_index
            ambiguous_track_ids: set of track_ids currently flagged as ambiguous
        """
        ambiguous_tracks = self.find_ambiguous_tracks(tracks)

        if not faces or not tracks:
            return {}, ambiguous_tracks

        num_faces = len(faces)
        num_tracks = len(tracks)

        # Cost matrix: minimize cost <=> maximize score
        # For invalid/rejected pairs, set cost to a large penalty
        LARGE_COST = 1000.0
        cost_matrix = np.full((num_faces, num_tracks), fill_value=LARGE_COST, dtype=np.float64)
        score_matrix = np.zeros((num_faces, num_tracks), dtype=np.float64)

        for f_idx, face_box in enumerate(faces):
            for t_idx, (tid, body_box) in enumerate(tracks):
                score = self.compute_match_score(face_box, body_box)
                score_matrix[f_idx, t_idx] = score
                if score >= self.min_assignment_score:
                    cost_matrix[f_idx, t_idx] = 1.0 - score

        # Solve global assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        assignments: Dict[int, int] = {}
        for f_idx, t_idx in zip(row_ind, col_ind):
            score = score_matrix[f_idx, t_idx]
            if score >= self.min_assignment_score and cost_matrix[f_idx, t_idx] < LARGE_COST:
                track_id = tracks[t_idx][0]
                assignments[track_id] = f_idx

        return assignments, ambiguous_tracks
