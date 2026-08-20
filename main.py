import argparse
import cv2
import logging
import os
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

def main():
    parser = argparse.ArgumentParser(description="E-Challan Traffic Violation Detection Pipeline")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run YOLO on (e.g., cpu, 0, cuda)")
    parser.add_argument("--weights", type=str, default="weights/yolov8n.pt", help="Path to YOLO weights")
    parser.add_argument("--tracker", type=str, default="iou", help="Tracker type (deepsort or iou)")
    parser.add_argument("--show", action="store_true", help="Display the video with bounding boxes for debugging")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.is_file():
        logger.error(f"Video file not found: {video_path}")
        sys.exit(1)

    logger.info("Initializing components...")
    
    # Initialize components
    detector = YOLODetector(weights=args.weights, device=args.device, conf=0.35)
    tracker = build_tracker(method=args.tracker)
    trajectories = TrajectoryStore(maxlen=60)
    
    # Configure Violation Engine
    # Note: stop_line and parking_zones are currently hardcoded for demonstration.
    # In a real setup, these would come from a camera config file.
    violation_config = {
        "fps": 30,
        "stop_line": (100, 500, 1000, 500), # x1, y1, x2, y2
    }
    violation_engine = ViolationEngine(violation_config)
    
    # Use GPU for OCR if device is not strictly 'cpu'
    ocr = PlateRecognizer(use_gpu=(args.device != "cpu"))
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
                        
                    # Save evidence crop
                    evidence_path = temp_dir / f"frame_{frame_no}_veh_{vid}.jpg"
                    x1, y1, x2, y2 = map(int, box)
                    # Expand box slightly for evidence
                    h, w = frame.shape[:2]
                    ex1, ey1 = max(0, x1 - 20), max(0, y1 - 20)
                    ex2, ey2 = min(w, x2 + 20), min(h, y2 + 20)
                    evidence_img = frame[ey1:ey2, ex1:ex2]
                    
                    cv2.imwrite(str(evidence_path), evidence_img)

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
