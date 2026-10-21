#!/usr/bin/env python3
"""
scripts/simulate.py
Synthetic unit-level scenario runner for the Three-State Identity Machine.
Validates all state transitions, decay math, and Honesty Rules 1 through 6
completely headless without requiring a camera or physical hardware.
"""

import sys
import math
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, List, Tuple
from src.identity import IdentityManager, State


def run_scenario_1_clean_head_turn() -> bool:
    """
    Scenario 1: Prolonged head turn.
    Confirmed -> Temporary -> (Decays below floor 0.3) -> Unknown -> Reappears -> Confirmed with hysteresis.
    """
    print("\n--- Running Scenario 1: Clean Head Turn (Prolonged Face Loss) ---")
    mgr = IdentityManager(half_life_s=2.0, confidence_floor=0.30, hysteresis_k=2)

    # 1. Establish Confirmed Alice (2 consecutive frames)
    mgr.step(current_time=0.00, active_track_ids=[1], face_assignments={1: ("Alice", 0.90)}, match_threshold=0.363)
    t = mgr.step(current_time=0.033, active_track_ids=[1], face_assignments={1: ("Alice", 0.90)}, match_threshold=0.363)[1]
    if t.state != State.CONFIRMED or t.name != "Alice":
        print(f"  [FAIL] Expected Confirmed Alice, got {t.state} {t.name}")
        return False
    print(f"  [PASS] Frame 2 (t=0.03s): Confirmed Alice (conf={t.confidence:.2f})")

    # 2. Turn head away at t=1.0s (face lost) -> enters Temporary
    t = mgr.step(current_time=1.00, active_track_ids=[1], face_assignments={}, match_threshold=0.363)[1]
    if t.state != State.TEMPORARY:
        print(f"  [FAIL] Expected Temporary Alice, got {t.state}")
        return False
    print(f"  [PASS] Frame (t=1.00s): Entered Temporary state (conf={t.confidence:.2f})")

    # 3. Check exponential decay at dt = T_half = 2.0s (t = 2.033s) -> conf should be exactly 0.45
    t = mgr.step(current_time=2.033, active_track_ids=[1], face_assignments={}, match_threshold=0.363)[1]
    if t.state != State.TEMPORARY or not math.isclose(t.confidence, 0.45, abs_tol=1e-2):
        print(f"  [FAIL] Expected Temporary with conf ~0.45 at T_half, got conf={t.confidence:.3f}")
        return False
    print(f"  [PASS] Frame (t=2.03s, dt=T_half): Decayed to conf={t.confidence:.2f} >= floor (0.30)")

    # 4. Turn head prolonged to t=6.00s (dt = 5.0s > 2*T_half) -> conf < floor -> Unknown
    t = mgr.step(current_time=6.00, active_track_ids=[1], face_assignments={}, match_threshold=0.363)[1]
    if t.state != State.UNKNOWN or t.confidence != 0.0:
        print(f"  [FAIL] Expected Unknown after prolonged loss, got {t.state} (conf={t.confidence:.2f})")
        return False
    print(f"  [PASS] Frame (t=6.00s): Dropped below floor to Unknown (conf=0.00)")

    # 5. Face returns at t=7.00s (Frame 1 of hysteresis)
    t = mgr.step(current_time=7.00, active_track_ids=[1], face_assignments={1: ("Alice", 0.88)}, match_threshold=0.363)[1]
    if t.state != State.UNKNOWN:
        print(f"  [FAIL] Expected Unknown during hysteresis frame 1, got {t.state}")
        return False
    print(f"  [PASS] Frame (t=7.00s): Face re-detected, awaiting hysteresis (State=Unknown)")

    # 6. Face confirmed at t=7.033s (Frame 2 of hysteresis -> Confirmed!)
    t = mgr.step(current_time=7.033, active_track_ids=[1], face_assignments={1: ("Alice", 0.88)}, match_threshold=0.363)[1]
    if t.state != State.CONFIRMED or t.name != "Alice":
        print(f"  [FAIL] Expected Confirmed Alice after hysteresis, got {t.state} {t.name}")
        return False
    print(f"  [PASS] Frame (t=7.03s): Promoted back to Confirmed Alice (conf={t.confidence:.2f})")
    return True


def run_scenario_2_quick_head_turn() -> bool:
    """
    Scenario 2: Quick glance away (dt < time to floor).
    Confirmed -> Temporary (decays but > floor) -> Face reappears -> Immediate Confirmed recovery.
    """
    print("\n--- Running Scenario 2: Quick Head Turn (Loss < Floor) ---")
    mgr = IdentityManager(half_life_s=2.0, confidence_floor=0.30, hysteresis_k=1)
    mgr.step(current_time=0.0, active_track_ids=[1], face_assignments={1: ("Alice", 0.85)}, match_threshold=0.363)

    # 1. Face lost for 1.2 seconds
    t = mgr.step(current_time=1.2, active_track_ids=[1], face_assignments={}, match_threshold=0.363)[1]
    if t.state != State.TEMPORARY or t.confidence <= 0.30:
        print(f"  [FAIL] Expected Temporary above floor, got {t.state} (conf={t.confidence})")
        return False
    print(f"  [PASS] t=1.2s: Decaying in Temporary state (conf={t.confidence:.2f} > floor)")

    # 2. Face reappears before reaching floor -> recovers to Confirmed immediately
    t = mgr.step(current_time=1.3, active_track_ids=[1], face_assignments={1: ("Alice", 0.89)}, match_threshold=0.363)[1]
    if t.state != State.CONFIRMED or not math.isclose(t.confidence, 0.89, abs_tol=1e-3):
        print(f"  [FAIL] Expected immediate recovery to Confirmed, got {t.state} (conf={t.confidence})")
        return False
    print(f"  [PASS] t=1.3s: Fast recovery to Confirmed Alice (conf={t.confidence:.2f})")
    return True


def run_scenario_3_identity_swap_attempt() -> bool:
    """
    Scenario 3: Identity swap attempt (Rule 2).
    Track 1 has Alice. Bob's face detected on Track 1 -> Immediately drops to Unknown.
    """
    print("\n--- Running Scenario 3: Identity Swap Prevention (Rule 2) ---")
    mgr = IdentityManager(hysteresis_k=2)
    # Establish Alice
    mgr.step(current_time=0.00, active_track_ids=[1], face_assignments={1: ("Alice", 0.90)}, match_threshold=0.363)
    mgr.step(current_time=0.03, active_track_ids=[1], face_assignments={1: ("Alice", 0.90)}, match_threshold=0.363)

    # Bob's face detected on Track 1
    t = mgr.step(current_time=0.06, active_track_ids=[1], face_assignments={1: ("Bob", 0.88)}, match_threshold=0.363)[1]
    if t.state != State.UNKNOWN or t.name != "Unknown":
        print(f"  [FAIL] Track did not drop to Unknown on identity conflict! Got {t.state} {t.name}")
        return False
    print(f"  [PASS] Swap attempt caught: Track 1 dropped to Unknown on mismatching face.")
    return True


def run_scenario_4_track_timeout() -> bool:
    """
    Scenario 4: Track lost timeout (Rule 3).
    Track leaves camera view for > track_lost_frames -> Identity deleted.
    """
    print("\n--- Running Scenario 4: Track Lost Timeout (Rule 3) ---")
    mgr = IdentityManager(track_lost_frames=5, hysteresis_k=1)
    mgr.step(current_time=0.0, active_track_ids=[1], face_assignments={1: ("Alice", 0.90)}, match_threshold=0.363)

    # Disappears for 6 frames
    for i in range(1, 7):
        mgr.step(current_time=0.033 * i, active_track_ids=[], face_assignments={}, match_threshold=0.363)

    if 1 in mgr.tracks:
        print("  [FAIL] Track 1 was not deleted after exceeding track_lost_frames!")
        return False
    print("  [PASS] Track 1 successfully purged after disappear timeout.")
    return True


def run_scenario_5_duplicate_claim_resolution() -> bool:
    """
    Scenario 5: Conflict resolution (Rule 4).
    Track 1 (Temporary, conf=0.45) vs Track 2 (Confirmed, conf=0.92) both claim Alice.
    Higher confidence wins; lower confidence drops to Unknown.
    """
    print("\n--- Running Scenario 5: Duplicate Identity Conflict Resolution (Rule 4) ---")
    mgr = IdentityManager(hysteresis_k=1)
    # Track 1 confirmed as Alice, enters Temporary
    mgr.step(current_time=0.0, active_track_ids=[1], face_assignments={1: ("Alice", 0.80)}, match_threshold=0.363)
    mgr.step(current_time=1.5, active_track_ids=[1], face_assignments={}, match_threshold=0.363)

    # Track 2 claims Alice with fresh Confirmed 0.92
    res = mgr.step(
        current_time=1.6,
        active_track_ids=[1, 2],
        face_assignments={2: ("Alice", 0.92)},
        match_threshold=0.363,
    )
    t1 = res[1]
    t2 = res[2]

    if t2.state != State.CONFIRMED or t2.name != "Alice":
        print(f"  [FAIL] Track 2 should be Confirmed Alice, got {t2.state} {t2.name}")
        return False
    if t1.state != State.UNKNOWN or t1.name != "Unknown":
        print(f"  [FAIL] Track 1 should be dropped to Unknown, got {t1.state} {t1.name}")
        return False
    print(f"  [PASS] Conflict resolved: Track 2 kept Alice (conf={t2.confidence:.2f}); Track 1 set to Unknown.")
    return True


def run_scenario_6_ambiguous_crossing() -> bool:
    """
    Scenario 6: Ambiguous body overlap (Rule 5).
    During body overlap, no extension/restoration of Temporary without face evidence,
    and extra confidence penalty applies.
    """
    print("\n--- Running Scenario 6: Ambiguous Body Overlap (Rule 5) ---")
    mgr = IdentityManager(half_life_s=2.0, hysteresis_k=1, ambiguous_decay_multiplier=1.5, ambiguous_penalty=0.05)
    mgr.step(current_time=0.0, active_track_ids=[1], face_assignments={1: ("Alice", 0.90)}, match_threshold=0.363)

    # Overlapping crossing with no face
    res = mgr.step(
        current_time=1.0,
        active_track_ids=[1, 2],
        face_assignments={},
        match_threshold=0.363,
        ambiguous_track_ids={1, 2},
    )
    t1 = res[1]
    normal_decay = 0.90 * math.exp(-(math.log(2) / 2.0) * 1.0)
    if t1.confidence >= normal_decay:
        print(f"  [FAIL] Expected extra decay under ambiguity! conf={t1.confidence:.3f} >= {normal_decay:.3f}")
        return False
    print(f"  [PASS] Extra penalty applied under ambiguity: conf={t1.confidence:.3f} < normal={normal_decay:.3f}")
    return True


def run_scenario_7_hysteresis_flicker_rejection() -> bool:
    """
    Scenario 7: Hysteresis rejection of single-frame flicker (Rule 6).
    A single match frame must not promote Unknown to Confirmed.
    """
    print("\n--- Running Scenario 7: Hysteresis Flicker Rejection (Rule 6) ---")
    mgr = IdentityManager(hysteresis_k=3)
    # Frame 1
    t = mgr.step(current_time=0.00, active_track_ids=[1], face_assignments={1: ("Alice", 0.95)}, match_threshold=0.363)[1]
    if t.state != State.UNKNOWN:
        print(f"  [FAIL] Frame 1 promoted prematurely! State={t.state}")
        return False

    # Interrupted before K=3
    t = mgr.step(current_time=0.03, active_track_ids=[1], face_assignments={}, match_threshold=0.363)[1]
    if t.state != State.UNKNOWN:
        print(f"  [FAIL] State changed unexpectedly on interrupted match: {t.state}")
        return False

    # New match frame -> still Unknown (count reset)
    t = mgr.step(current_time=0.06, active_track_ids=[1], face_assignments={1: ("Alice", 0.95)}, match_threshold=0.363)[1]
    if t.state != State.UNKNOWN:
        print(f"  [FAIL] Premature promotion after reset! State={t.state}")
        return False

    print("  [PASS] Spurious flicker successfully rejected by hysteresis.")
    return True


def main():
    print("=" * 70)
    print(" THREE-STATE IDENTITY MACHINE: SYNTHETIC SCENARIO SIMULATION")
    print(" Validating Honest 3-State Logic & Honesty Rules 1 to 6")
    print("=" * 70)

    scenarios = [
        ("Scenario 1: Clean Head Turn (Prolonged Loss)", run_scenario_1_clean_head_turn),
        ("Scenario 2: Quick Head Turn (Fast Recovery)", run_scenario_2_quick_head_turn),
        ("Scenario 3: Identity Swap Prevention (Rule 2)", run_scenario_3_identity_swap_attempt),
        ("Scenario 4: Track Lost Timeout (Rule 3)", run_scenario_4_track_timeout),
        ("Scenario 5: Conflict Resolution (Rule 4)", run_scenario_5_duplicate_claim_resolution),
        ("Scenario 6: Ambiguous Body Overlap (Rule 5)", run_scenario_6_ambiguous_crossing),
        ("Scenario 7: Hysteresis Flicker Rejection (Rule 6)", run_scenario_7_hysteresis_flicker_rejection),
    ]

    results = []
    for name, func in scenarios:
        try:
            ok = func()
            results.append((name, "PASS" if ok else "FAIL"))
        except Exception as e:
            print(f"  [EXCEPTION] {e}")
            results.append((name, f"ERROR ({e})"))

    print("\n" + "=" * 70)
    print(" SIMULATION SUMMARY TABLE")
    print("=" * 70)
    all_passed = True
    for name, status in results:
        status_str = f"[{status}]"
        print(f" {status_str:<10} | {name}")
        if status != "PASS":
            all_passed = False
    print("-" * 70)

    if all_passed:
        print("[SUCCESS] All 7 synthetic scenarios passed with 100% adherence to honesty rules.")
        sys.exit(0)
    else:
        print("[FAILURE] One or more simulation scenarios failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
