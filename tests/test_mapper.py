"""
tests/test_mapper.py
Unit tests for face-to-body mapping, geometric head constraints, linear_sum_assignment,
and overlapping body ambiguity detection.
"""

import pytest
from src.mapper import FaceBodyMapper, compute_iou


def test_compute_iou():
    boxA = (100.0, 100.0, 100.0, 100.0)
    boxB = (150.0, 100.0, 100.0, 100.0)
    # Intersection is 50x100 = 5000, union is 10000 + 10000 - 5000 = 15000 -> 1/3
    iou = compute_iou(boxA, boxB)
    assert abs(iou - 0.3333) < 0.01

    # Disjoint boxes
    boxC = (300.0, 300.0, 50.0, 50.0)
    assert compute_iou(boxA, boxC) == 0.0


def test_mapper_upper_body_constraint():
    """
    A face may only be assigned if face centre lies inside the body box AND
    within the upper portion (default top 40%).
    """
    mapper = FaceBodyMapper(upper_body_ratio=0.40)
    body = (100.0, 100.0, 100.0, 300.0)  # y spans 100 to 400. Top 40% spans y=100 to y=220.

    # Face in top 20%: center at (150, 140) -> valid!
    face_top = (135.0, 120.0, 30.0, 40.0)
    score_top = mapper.compute_match_score(face_top, body)
    assert score_top > 0.40

    # Face in middle/lower body: center at (150, 260) -> rejected!
    face_mid = (135.0, 240.0, 30.0, 40.0)
    score_mid = mapper.compute_match_score(face_mid, body)
    assert score_mid == 0.0

    # Face completely outside body horizontally: center at (250, 140) -> rejected!
    face_outside = (240.0, 120.0, 30.0, 40.0)
    score_outside = mapper.compute_match_score(face_outside, body)
    assert score_outside == 0.0


def test_mapper_horizontal_alignment_scoring():
    """
    Face centered on body head axis should score higher than face near lateral edge.
    """
    mapper = FaceBodyMapper(upper_body_ratio=0.40)
    body = (100.0, 100.0, 100.0, 300.0)  # Head axis is x = 150.0

    # Centered face: center x = 150.0
    face_center = (135.0, 120.0, 30.0, 40.0)
    score_center = mapper.compute_match_score(face_center, body)

    # Offset face: center x = 185.0 (near edge x=200)
    face_offset = (170.0, 120.0, 30.0, 40.0)
    score_offset = mapper.compute_match_score(face_offset, body)

    assert score_center > score_offset


def test_mapper_size_plausibility():
    """
    Face width should be roughly 0.15 to 0.6 of body width.
    Extremely large or tiny faces should be rejected.
    """
    mapper = FaceBodyMapper(min_face_width_ratio=0.15, max_face_width_ratio=0.60)
    body = (100.0, 100.0, 100.0, 300.0)  # Body width = 100

    # Plausible face: width = 30 (ratio = 0.30)
    face_good = (135.0, 120.0, 30.0, 40.0)
    assert mapper.compute_match_score(face_good, body) > 0.40

    # Unrealistic face: width = 5 (ratio = 0.05) -> too small
    face_tiny = (147.0, 120.0, 5.0, 6.0)
    assert mapper.compute_match_score(face_tiny, body) == 0.0

    # Unrealistic face: width = 95 (ratio = 0.95) -> too huge
    face_huge = (102.0, 120.0, 95.0, 95.0)
    assert mapper.compute_match_score(face_huge, body) == 0.0


def test_global_bipartite_matching():
    """
    scipy linear_sum_assignment ensures one face maps to at most one body
    and each body gets at most one face.
    """
    mapper = FaceBodyMapper(min_assignment_score=0.40)

    # 2 bodies
    body1 = (50.0, 50.0, 100.0, 300.0)
    body2 = (300.0, 50.0, 100.0, 300.0)
    tracks = [(1, body1), (2, body2)]

    # 2 faces corresponding to bodies
    face1 = (85.0, 70.0, 30.0, 40.0)
    face2 = (335.0, 70.0, 30.0, 40.0)
    faces = [face1, face2]

    assignments, ambig = mapper.match(faces, tracks)
    assert assignments == {1: 0, 2: 1}
    assert len(ambig) == 0

    # Competing faces: 2 faces inside body 1, none inside body 2
    face_better = (85.0, 70.0, 30.0, 40.0)     # centered
    face_worse = (110.0, 70.0, 30.0, 40.0)     # lateral
    competing_faces = [face_worse, face_better]

    assignments, _ = mapper.match(competing_faces, [(1, body1)])
    # Only 1 face gets assigned to body 1, and it must be the higher scoring one (index 1)
    assert assignments == {1: 1}


def test_ambiguous_body_overlap():
    """
    When two body boxes overlap by more than overlap_iou_threshold,
    both must be flagged as ambiguous.
    """
    mapper = FaceBodyMapper(overlap_iou_threshold=0.20)

    # Overlapping bodies (path crossing)
    bodyA = (100.0, 100.0, 100.0, 300.0)
    bodyB = (140.0, 100.0, 100.0, 300.0)  # Significant overlap
    tracks = [(1, bodyA), (2, bodyB)]

    ambig = mapper.find_ambiguous_tracks(tracks)
    assert 1 in ambig
    assert 2 in ambig

    # Non-overlapping bodies
    bodyC = (400.0, 100.0, 100.0, 300.0)
    tracks_clear = [(1, bodyA), (3, bodyC)]
    ambig_clear = mapper.find_ambiguous_tracks(tracks_clear)
    assert len(ambig_clear) == 0
