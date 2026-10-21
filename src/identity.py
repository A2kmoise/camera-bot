"""
src/identity.py
Honest Three-State Identity Machine with Exponential Confidence Decay.
Enforces all 6 Honesty Rules.
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple
import math
import numpy as np


class State(str, Enum):
    CONFIRMED = "Confirmed"
    TEMPORARY = "Temporary"
    UNKNOWN = "Unknown"


class TrackIdentity:
    """
    Maintains the identity state for a single persistent body track ID.
    """

    def __init__(
        self,
        track_id: int,
        half_life_s: float = 2.0,
        confidence_floor: float = 0.30,
        hysteresis_k: int = 2,
        ambiguous_decay_multiplier: float = 1.5,
        ambiguous_penalty: float = 0.05,
    ):
        self.track_id = track_id
        self.half_life_s = max(half_life_s, 0.001)
        self.confidence_floor = confidence_floor
        self.hysteresis_k = hysteresis_k
        self.ambiguous_decay_multiplier = ambiguous_decay_multiplier
        self.ambiguous_penalty = ambiguous_penalty

        # Identity state variables
        self.state: State = State.UNKNOWN
        self.name: str = "Unknown"
        self.confidence: float = 0.0
        self.conf_at_seen: float = 0.0
        self.t_last_seen: float = 0.0
        self.last_update_time: float = 0.0

        # Disappearance counter
        self.absent_frames: int = 0

        # Hysteresis candidate tracking: {candidate_name: consecutive_count}
        self.candidate_name: Optional[str] = None
        self.candidate_matches: int = 0

    @property
    def decay_lambda(self) -> float:
        """Decay constant: lambda = ln(2) / T_half."""
        return math.log(2.0) / self.half_life_s

    def compute_decay(self, current_time: float, is_ambiguous: bool = False) -> float:
        """
        Compute decayed confidence:
        conf(t) = conf_at_seen * exp(-lambda * (t - t_last_seen))
        """
        if self.conf_at_seen <= 0.0:
            return 0.0

        dt = max(0.0, current_time - self.t_last_seen)
        lam = self.decay_lambda
        if is_ambiguous:
            # Rule 5: apply extra decay multiplier while ambiguous overlap without face
            lam *= self.ambiguous_decay_multiplier

        decayed = self.conf_at_seen * math.exp(-lam * dt)

        if is_ambiguous:
            decayed = max(0.0, decayed - self.ambiguous_penalty)

        return float(np.clip(decayed, 0.0, 1.0))

    def reset_to_unknown(self):
        """Immediately reset identity to Unknown."""
        self.state = State.UNKNOWN
        self.name = "Unknown"
        self.confidence = 0.0
        self.conf_at_seen = 0.0
        self.candidate_name = None
        self.candidate_matches = 0

    def update_with_face(
        self,
        current_time: float,
        matched_name: Optional[str],
        similarity: float,
        match_threshold: float,
        is_ambiguous: bool = False,
    ):
        """
        Update state when a face is mapped to this body track.
        """
        self.absent_frames = 0
        self.last_update_time = current_time
        sim_norm = float(np.clip(similarity, 0.0, 1.0))

        # Check if face meets recognition threshold
        face_recognized = (matched_name is not None) and (sim_norm >= match_threshold)

        # Rule 2: If face matches a DIFFERENT enrolled person (or no one) while holding an identity,
        # immediately drop to Unknown.
        if self.state in (State.CONFIRMED, State.TEMPORARY):
            if not face_recognized or matched_name != self.name:
                self.reset_to_unknown()

        if face_recognized and matched_name is not None:
            if self.state == State.CONFIRMED and self.name == matched_name:
                # Refresh confirmed state
                self.conf_at_seen = sim_norm
                self.confidence = sim_norm
                self.t_last_seen = current_time
                self.candidate_name = matched_name
                self.candidate_matches = self.hysteresis_k
            elif self.state == State.TEMPORARY and self.name == matched_name:
                # Identity recovered! Promote back to Confirmed
                self.state = State.CONFIRMED
                self.conf_at_seen = sim_norm
                self.confidence = sim_norm
                self.t_last_seen = current_time
                self.candidate_name = matched_name
                self.candidate_matches = self.hysteresis_k
            else:
                # Currently Unknown (or newly dropped to Unknown): Apply Rule 6 Hysteresis
                if self.candidate_name == matched_name:
                    self.candidate_matches += 1
                else:
                    self.candidate_name = matched_name
                    self.candidate_matches = 1

                if self.candidate_matches >= self.hysteresis_k:
                    # Promoted to Confirmed!
                    self.state = State.CONFIRMED
                    self.name = matched_name
                    self.conf_at_seen = sim_norm
                    self.confidence = sim_norm
                    self.t_last_seen = current_time
                else:
                    # Still awaiting enough consecutive frames
                    self.state = State.UNKNOWN
                    self.name = "Unknown"
                    self.confidence = 0.0
        else:
            # Face mapped, but not recognized above threshold
            # Reset candidate hysteresis counter
            self.candidate_name = None
            self.candidate_matches = 0
            if self.state != State.UNKNOWN:
                self.reset_to_unknown()

    def update_without_face(
        self,
        current_time: float,
        is_ambiguous: bool = False,
    ):
        """
        Update state when no face is mapped to this body track.
        """
        self.absent_frames = 0
        self.last_update_time = current_time
        # Reset any ongoing hysteresis promotion if face lost before confirmation
        if self.state == State.UNKNOWN:
            self.candidate_name = None
            self.candidate_matches = 0
            self.confidence = 0.0
            return

        # If currently Confirmed, transition to Temporary
        if self.state == State.CONFIRMED:
            self.state = State.TEMPORARY

        if self.state == State.TEMPORARY:
            # Exponential decay
            decayed_conf = self.compute_decay(current_time, is_ambiguous=is_ambiguous)
            if decayed_conf < self.confidence_floor:
                # Drop to Unknown
                self.reset_to_unknown()
            else:
                self.confidence = decayed_conf


class IdentityManager:
    """
    Manages identity states for all active tracks and enforces global honesty rules.
    """

    def __init__(
        self,
        half_life_s: float = 2.0,
        confidence_floor: float = 0.30,
        hysteresis_k: int = 2,
        track_lost_frames: int = 30,
        ambiguous_decay_multiplier: float = 1.5,
        ambiguous_penalty: float = 0.05,
    ):
        self.half_life_s = half_life_s
        self.confidence_floor = confidence_floor
        self.hysteresis_k = hysteresis_k
        self.track_lost_frames = track_lost_frames
        self.ambiguous_decay_multiplier = ambiguous_decay_multiplier
        self.ambiguous_penalty = ambiguous_penalty

        self.tracks: Dict[int, TrackIdentity] = {}

    def set_half_life(self, half_life_s: float):
        """Allow dynamic update of T_half live."""
        self.half_life_s = max(0.1, half_life_s)
        for track in self.tracks.values():
            track.half_life_s = self.half_life_s

    def set_confidence_floor(self, floor: float):
        """Allow dynamic update of confidence floor live."""
        self.confidence_floor = float(np.clip(floor, 0.05, 0.95))
        for track in self.tracks.values():
            track.confidence_floor = self.confidence_floor

    def get_or_create_track(self, track_id: int) -> TrackIdentity:
        if track_id not in self.tracks:
            self.tracks[track_id] = TrackIdentity(
                track_id=track_id,
                half_life_s=self.half_life_s,
                confidence_floor=self.confidence_floor,
                hysteresis_k=self.hysteresis_k,
                ambiguous_decay_multiplier=self.ambiguous_decay_multiplier,
                ambiguous_penalty=self.ambiguous_penalty,
            )
        return self.tracks[track_id]

    def step(
        self,
        current_time: float,
        active_track_ids: List[int],
        face_assignments: Dict[int, Tuple[Optional[str], float]],
        match_threshold: float,
        ambiguous_track_ids: Optional[set] = None,
    ) -> Dict[int, TrackIdentity]:
        """
        Advance state machine for one frame.
        Args:
            current_time: current timestamp (seconds)
            active_track_ids: IDs of person tracks detected in this frame
            face_assignments: dict mapping track_id -> (matched_name, similarity)
            match_threshold: minimum cosine similarity threshold
            ambiguous_track_ids: set of track IDs flagged with overlapping body bounding boxes
        """
        if ambiguous_track_ids is None:
            ambiguous_track_ids = set()

        active_set = set(active_track_ids)

        # 1. Update active tracks
        for tid in active_track_ids:
            track = self.get_or_create_track(tid)
            is_ambig = tid in ambiguous_track_ids

            if tid in face_assignments:
                name, sim = face_assignments[tid]
                track.update_with_face(
                    current_time=current_time,
                    matched_name=name,
                    similarity=sim,
                    match_threshold=match_threshold,
                    is_ambiguous=is_ambig,
                )
            else:
                track.update_without_face(
                    current_time=current_time,
                    is_ambiguous=is_ambig,
                )

        # 2. Rule 3: Track disappeared timeout. Delete identity if absent > track_lost_frames
        to_delete = []
        for tid, track in self.tracks.items():
            if tid not in active_set:
                track.absent_frames += 1
                if track.absent_frames > self.track_lost_frames:
                    to_delete.append(tid)

        for tid in to_delete:
            del self.tracks[tid]

        # 3. Rule 4: Identity Uniqueness.
        # If two tracks both claim the same identity, only the one with higher confidence keeps it.
        # Other track is dropped to Unknown.
        identity_claims: Dict[str, List[TrackIdentity]] = {}
        for tid in active_track_ids:
            track = self.tracks.get(tid)
            if track and track.state in (State.CONFIRMED, State.TEMPORARY) and track.name != "Unknown":
                identity_claims.setdefault(track.name, []).append(track)

        for name, claimants in identity_claims.items():
            if len(claimants) > 1:
                # Sort claimants: Confirmed first, then highest confidence, then lowest track_id
                claimants.sort(
                    key=lambda t: (1 if t.state == State.CONFIRMED else 0, t.confidence, -t.track_id),
                    reverse=True,
                )
                winner = claimants[0]
                for loser in claimants[1:]:
                    loser.reset_to_unknown()

        return {tid: self.tracks[tid] for tid in active_track_ids if tid in self.tracks}
