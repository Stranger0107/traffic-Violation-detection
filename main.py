import argparse
import cv2
import logging
import os
import time
from dotenv import load_dotenv
load_dotenv()
import sys
import tempfile
import torch
from pathlib import Path

# Fix for PyTorch 2.6+ which defaults weights_only=True and breaks Ultralytics loading
_original_load = torch.load
def _patched_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("tvd.main")

from src.detection import YOLODetector
from src.tracking import build_tracker, TrajectoryStore
from src.violations import ViolationEngine
from src.ocr import PlateRecognizer
from src.model_api import ModelViolationClient
from src.utils import SpeedEstimator

def get_default_device() -> str:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        logger.info(f"GPU detected: {gpu_name} ({vram:.1f} GB VRAM)")
        return "0"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("Apple Silicon GPU (MPS) detected")
        return "mps"
    else:
        logger.info("No GPU found, using CPU")
        return "cpu"

def main():
    parser = argparse.ArgumentParser(description="E-Challan Traffic Violation Detection Pipeline")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--device", type=str, default=None, help="Device: auto-detect if not set. Use 'cpu', '0' (first GPU), 'cuda', or 'mps' (Apple Silicon)")
    parser.add_argument("--weights", type=str, default="weights/best.pt", help="Path to YOLO weights")
    parser.add_argument("--tracker", type=str, default="iou", help="Tracker type (deepsort or iou)")
    parser.add_argument("--show", action="store_true", help="Display the video with bounding boxes for debugging")
    parser.add_argument("--speed-limit", type=float, default=60.0, help="Speed limit in km/h for speeding detection")
    parser.add_argument("--cooldown", type=int, default=900, help="Cooldown in seconds before the same plate can be flagged again (default: 900 = 15 min)")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.is_file():
        logger.error(f"Video file not found: {video_path}")
        sys.exit(1)

    # Auto-detect device if not specified
    if args.device is None:
        args.device = get_default_device()
    
    use_gpu = args.device not in ("cpu", "None")
    logger.info(f"Using device: {args.device} (GPU: {use_gpu})")
    
    logger.info("Initializing components...")
    
    # Initialize components
    detector = YOLODetector(weights=args.weights, device=args.device, conf=0.35)
    tracker = build_tracker(method=args.tracker)
    trajectories = TrajectoryStore(maxlen=60)
    speed_estimator = SpeedEstimator(fps=30.0, pixels_per_meter=8.0)
    
    # Configure Violation Engine
    violation_config = {
        "fps": 30,
        "stop_line": (100, 500, 1000, 500),  # x1, y1, x2, y2
    }
    violation_engine = ViolationEngine(violation_config)
    
    # Use GPU for OCR if available
    ocr = PlateRecognizer(use_gpu=use_gpu)
    api_client = ModelViolationClient()

    # Open Video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_no = 0

    logger.info("Starting inference loop...")

    # Temp directory for evidence images
    temp_dir = Path(tempfile.mkdtemp(prefix="tvd_evidence_"))

    # Plate cooldown tracker: plate_number -> last_submission_timestamp
    # Once a plate is submitted, skip it for --cooldown seconds (default 15 min)
    submitted_plates = {}
    COOLDOWN_SEC = args.cooldown
    logger.info(f"Plate cooldown set to {COOLDOWN_SEC}s ({COOLDOWN_SEC // 60} min)")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_no += 1
            timestamp_sec = frame_no / fps

            # 1. Detect
            detections = detector.detect(frame)

            # 2. Track
            tracked = tracker.update(detections, frame)

            # Update trajectories for motion-based violations
            for det in tracked:
                if det.track_id != -1:
                    trajectories.update(det.track_id, frame_no, det.bottom_center)
            
            # Update speed estimator for all tracked vehicles
            for det in tracked:
                if det.track_id != -1 and det.group == "vehicle":
                    speed = speed_estimator.update(det.track_id, frame_no, det.bottom_center)
                    # Check for speeding violation
                    if speed is not None and speed > args.speed_limit:
                        from src.violations import make_record
                        # Use a cooldown to avoid spamming
                        tid = det.track_id
                        if not hasattr(speed_estimator, '_flagged'):
                            speed_estimator._flagged = {}
                        last_flagged = speed_estimator._flagged.get(tid, -9999)
                        if (frame_no - last_flagged) > 120:  # 4 second cooldown
                            speed_estimator._flagged[tid] = frame_no
                            violation = make_record(
                                frame_no, tid, "SPEEDING",
                                box=det.box.tolist(),
                                extra={"speed_kmh": round(speed, 1), "limit_kmh": args.speed_limit}
                            )
                            # Submit speeding violation immediately
                            vid = tid
                            speed_plate = "UNKNOWN"
                            speed_ocr_result = ocr.process_vehicle(vid, frame, det.box.tolist())
                            if speed_ocr_result:
                                speed_plate, speed_ocr_conf = speed_ocr_result
                            else:
                                speed_ocr_conf = 0.0

                            # Save full frame as evidence
                            evidence_path = temp_dir / f"frame_{frame_no}_veh_{vid}_speeding.jpg"
                            cv2.imwrite(str(evidence_path), frame)

                            logger.info(f"Frame {frame_no}: SPEEDING {speed:.1f} km/h (limit {args.speed_limit}) for Vehicle ID {vid}")
                            res = api_client.submit(
                                frame_number=frame_no,
                                violation_type="SPEEDING",
                                detected_plate=speed_plate,
                                evidence_path=str(evidence_path),
                                video_timestamp_sec=timestamp_sec,
                                ocr_confidence=speed_ocr_conf
                            )
                            if res.get("success"):
                                logger.info(f"Speeding violation submitted to backend.")
                            else:
                                logger.error(f"Failed to submit speeding: {res.get('message')}")
            
            # Clean up old trajectories periodically
            if frame_no % 300 == 0:
                trajectories.cleanup_old_tracks(frame_no)

            # 3. Detect Violations
            violations = violation_engine.process_frame(frame, frame_no, tracked, trajectories)

            # Debug Visualization
            if args.show:
                debug_frame = frame.copy()
                # Draw detections
                for det in tracked:
                    x1, y1, x2, y2 = map(int, det.box)
                    label = f"{det.class_name} {det.conf:.2f} ID:{det.track_id}"
                    cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(debug_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Draw stop line for Red Light reference
                sl = violation_config["stop_line"]
                cv2.line(debug_frame, (sl[0], sl[1]), (sl[2], sl[3]), (0, 0, 255), 3)
                
                # Draw speed info
                for det in tracked:
                    if det.track_id != -1 and det.group == "vehicle":
                        speed = speed_estimator.get_speed(det.track_id)
                        if speed > 0:
                            x1, y1, x2, y2 = map(int, det.box)
                            speed_label = f"{speed:.0f} km/h"
                            color = (0, 0, 255) if speed > args.speed_limit else (0, 255, 0)
                            cv2.putText(debug_frame, speed_label, (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Resize for display if video is too large
                h, w = debug_frame.shape[:2]
                if w > 1280:
                    debug_frame = cv2.resize(debug_frame, (1280, int(h * 1280 / w)))
                    
                cv2.imshow("TVD Detection Debugger", debug_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("User pressed 'q', quitting video.")
                    break

            # 4. Handle Violations
            for v in violations:
                v_type = v.get("violation")
                vid = v.get("vehicle_id")
                box = v.get("box")

                logger.info(f"Frame {frame_no}: Detected {v_type} for Vehicle ID {vid}")

                if box:
                    # Run OCR to get license plate
                    ocr_result = ocr.process_vehicle(vid, frame, box)
                    if ocr_result:
                        plate_text, ocr_conf = ocr_result
                    else:
                        logger.warning(f"Could not read license plate for vehicle {vid}, submitting as UNKNOWN.")
                        plate_text, ocr_conf = "UNKNOWN", 0.0

                    # --- PLATE COOLDOWN CHECK ---
                    # Skip if this plate was already submitted within the cooldown period
                    now = time.time()
                    last_seen = submitted_plates.get(plate_text, 0)
                    remaining = COOLDOWN_SEC - (now - last_seen)

                    if remaining > 0:
                        logger.info(
                            f"SKIPPING: Plate {plate_text} already submitted {int(now - last_seen)}s ago. "
                            f"Cooldown expires in {int(remaining)}s."
                        )
                        continue

                    # Record this plate as submitted
                    submitted_plates[plate_text] = now
                    logger.info(f"Plate {plate_text} recorded. Next submission allowed after {COOLDOWN_SEC}s.")

                    # Save the FULL FRAME as evidence
                    evidence_path = temp_dir / f"frame_{frame_no}_veh_{vid}.jpg"
                    cv2.imwrite(str(evidence_path), frame)

                    # Submit to API
                    logger.info(f"Submitting to API: {v_type}, Plate: {plate_text}")
                    res = api_client.submit(
                        frame_number=frame_no,
                        violation_type=v_type,
                        detected_plate=plate_text,
                        evidence_path=str(evidence_path),
                        video_timestamp_sec=timestamp_sec,
                        ocr_confidence=ocr_conf
                    )

                    if res.get("success"):
                        logger.info(f"Successfully submitted event to backend.")
                    else:
                        logger.error(f"Failed to submit: {res.get('message')}")

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        cap.release()
        logger.info("Inference complete.")

if __name__ == "__main__":
    main()
