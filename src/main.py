"""
src/main.py
Main entrypoint for live webcam tracking and deterministic video replay.
"""

import argparse
import sys
import time
from pathlib import Path
import cv2
import yaml

from src.capture import VideoCaptureWrapper
from src.pipeline import TrackingPipeline


EVENT_MARKER_KEYS = {
    ord("1"): "head_turn_start",
    ord("2"): "head_turn_end",
    ord("3"): "occlusion_start",
    ord("4"): "occlusion_end",
    ord("5"): "crossing_start",
    ord("6"): "crossing_end",
    ord("7"): "event_marker_7",
    ord("8"): "event_marker_8",
    ord("9"): "event_marker_9",
}


def load_config(config_path: str = "config.yaml") -> dict:
    if Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(description="Person Tracker with 3-State Identity Continuity")
    parser.add_argument("--video", type=str, default=None, help="Path to video file for replay (if omitted, uses webcam)")
    parser.add_argument("--no-display", action="store_true", help="Run in headless mode without GUI window")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    parser.add_argument("--session", type=str, default=None, help="Custom session name for log outputs")
    parser.add_argument("--target", type=str, default=None, help="Target identity name to centre")
    parser.add_argument("--half-life", type=float, default=None, help="Override T_half decay time in seconds")
    parser.add_argument("--floor", type=float, default=None, help="Override confidence floor")
    parser.add_argument("--scenario", type=str, default="live", help="Scenario tag for evaluation logs")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of frames to process")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Apply command line overrides
    if args.target is not None:
        cfg.setdefault("centring", {})["target_name"] = args.target
    if args.half_life is not None:
        cfg.setdefault("identity", {})["half_life_s"] = args.half_life
    if args.floor is not None:
        cfg.setdefault("identity", {})["confidence_floor"] = args.floor

    cap_cfg = cfg.get("capture", {})
    source = args.video if args.video is not None else cap_cfg.get("camera_id", 0)

    try:
        cap = VideoCaptureWrapper(
            source=source,
            width=cap_cfg.get("width", 640),
            height=cap_cfg.get("height", 480),
            target_fps=cap_cfg.get("target_fps", 30),
        )
    except Exception as e:
        print(f"[ERROR] Could not initialize video capture from {source}: {e}")
        sys.exit(1)

    # Update frame width/height in config to match actual capture
    cfg.setdefault("capture", {})["width"] = cap.actual_width
    cfg.setdefault("capture", {})["height"] = cap.actual_height

    pipeline = TrackingPipeline(config=cfg, session_name=args.session)
    pipeline.set_scenario_tag(args.scenario)

    print("\n" + "=" * 65)
    print(" PERSON TRACKER & FACE ASSOCIATION (CPU-ONLY)")
    print(f" Source: {'Video: ' + str(args.video) if args.video else 'Webcam ' + str(source)}")
    print(f" Resolution: {cap.actual_width}x{cap.actual_height} @ {cap.fps:.1f} FPS")
    print(f" T_half: {pipeline.identity_mgr.half_life_s:.1f}s | Floor: {pipeline.identity_mgr.confidence_floor:.2f}")
    print(f" Headless: {args.no_display}")
    print("=" * 65)
    if not args.no_display:
        print(" Controls:")
        print("   'q': Quit")
        print("   'r': Toggle Raw Video & Log Recording")
        print("   '1'..'6': Insert Event Markers (e.g. 1=head_turn_start, 2=head_turn_end)")
        print("   't': Cycle Centring Target Identity")
        print("   '[' / ']': Decrease / Increase T_half Decay Time")
        print("   'd': Toggle Latency Debug Overlay")
        print("-" * 65)

    frame_count = 0
    start_wall_time = time.time()
    current_marker = None

    try:
        while True:
            ret, frame, timestamp = cap.read()
            if not ret or frame is None:
                print("\n[INFO] End of video stream reached.")
                break

            annotated_frame, status = pipeline.step_frame(
                frame=frame,
                timestamp=timestamp,
                frame_idx=frame_count,
                event_marker=current_marker,
            )
            current_marker = None  # Consume single-frame marker
            frame_count += 1

            if not args.no_display:
                cv2.imshow("Person Tracker - CPU Face Lock", annotated_frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    print("[INFO] Quitting on user command.")
                    break
                elif key == ord("r"):
                    pipeline.toggle_recording(fps=cap.fps)
                elif key == ord("d"):
                    pipeline.overlay.toggle_debug()
                elif key == ord("["):
                    new_th = max(0.5, pipeline.identity_mgr.half_life_s - 0.5)
                    pipeline.identity_mgr.set_half_life(new_th)
                    print(f"[CONFIG] Live T_half updated to: {new_th:.1f}s")
                elif key == ord("]"):
                    new_th = min(10.0, pipeline.identity_mgr.half_life_s + 0.5)
                    pipeline.identity_mgr.set_half_life(new_th)
                    print(f"[CONFIG] Live T_half updated to: {new_th:.1f}s")
                elif key == ord("t"):
                    # Cycle through enrolled identities
                    enrolled_names = list(pipeline.face_engine.enrolled.keys())
                    if enrolled_names:
                        curr = pipeline.centring.target_name
                        if curr in enrolled_names:
                            idx = (enrolled_names.index(curr) + 1) % (len(enrolled_names) + 1)
                            new_target = enrolled_names[idx] if idx < len(enrolled_names) else ""
                        else:
                            new_target = enrolled_names[0]
                    else:
                        new_target = ""
                    pipeline.centring.set_target_name(new_target)
                    print(f"[CONFIG] Centring target set to: '{new_target or 'Auto'}'")
                elif key in EVENT_MARKER_KEYS:
                    current_marker = EVENT_MARKER_KEYS[key]
                    print(f"[MARKER] Recorded event: {current_marker} at frame {frame_count}")

            if args.max_frames is not None and frame_count >= args.max_frames:
                print(f"[INFO] Processed max frames ({args.max_frames}). Stopping.")
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        total_time = time.time() - start_wall_time
        avg_fps = frame_count / max(total_time, 1e-4)
        cap.release()
        pipeline.close()
        if not args.no_display:
            cv2.destroyAllWindows()

        print("\n" + "=" * 65)
        print(" RUN SUMMARY")
        print(f" Total Frames Processed: {frame_count}")
        print(f" Elapsed Time: {total_time:.2f}s")
        print(f" Average Processing FPS: {avg_fps:.2f}")
        print(f" Log CSV: {pipeline.logger.csv_path}")
        print(f" Metadata: {pipeline.logger.meta_path}")
        print("=" * 65)


if __name__ == "__main__":
    main()
