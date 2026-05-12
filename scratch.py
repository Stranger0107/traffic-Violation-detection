from ultralytics import YOLO

model = YOLO("yolov8m.pt")

model.train(
    data=r"C:\Users\Lenovo\Downloads\4classes.yolov8 (1)\data.yaml",
    epochs=50,        # reduce first
    patience=20,
    imgsz=640,        # safer
    batch=8,          # safer
    mosaic=1.0,
    device=0,        
    project='models/traffic_v2',
    name='train'
)

import os
import cv2
import json
import csv
import datetime
import numpy as np
from pathlib import Path
from collections import Counter

from ultralytics import YOLO

from src.detection import YOLODetector, Detection
from src.tracking import build_tracker, TrajectoryStore
from src.violations import ViolationEngine
from src.ocr import PlateRecognizer
from src.database import TrafficDB

BASE_DIR = Path(".")

OUTPUT_DIR = BASE_DIR / "outputs"
VIDEO_DIR  = OUTPUT_DIR / "videos"
REPORT_DIR = OUTPUT_DIR / "reports"

VIDEO_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Ensure notebook detections are sent to backend ingestion API
os.environ["TVD_BACKEND_URL"] = "http://127.0.0.1:8000"
os.environ["TVD_BACKEND_API_KEY"] = "local-dev-key"

ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

print("✅ Setup ready")

MODEL_PATH = "C:/Users/Lenovo/runs/detect/models/traffic_v2/train7/weights/best.pt"
VIDEO_PATH = "D:/e-challan-system/data/input/video3.mp4"

# Your trained model
detector = YOLODetector(
    weights=MODEL_PATH,
    conf=0.3,
    device="0"
)

# COCO model (for traffic light + vehicles)
coco_model = YOLO("yolov8n.pt")

print("✅ Models loaded")

video_out_path = VIDEO_DIR / f"annotated_{ts}.mp4"
json_out_path  = REPORT_DIR / f"violations_{ts}.json"
csv_out_path   = REPORT_DIR / f"violations_{ts}.csv"

def build_summary(records):
    return dict(Counter([r["violation"] for r in records]))

tracker = build_tracker("deepsort", max_age=50, n_init=1)
traj = TrajectoryStore()

plate_reader = PlateRecognizer(use_gpu=True)

# Initialize Database
db = TrafficDB()

track_memory = {}
IOU_THRESHOLD = 0.5

def iou(boxA, boxB):
    x1 = max(boxA[0], boxB[0])
    y1 = max(boxA[1], boxB[1])
    x2 = min(boxA[2], boxB[2])
    y2 = min(boxA[3], boxB[3])

    inter = max(0, x2-x1) * max(0, y2-y1)
    if inter == 0:
        return 0

    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])

    return inter / (areaA + areaB - inter)

print("✅ Tracker ready")



def auto_find_stop_line(frame):
    h, w = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    mask = np.zeros_like(edges)
    pts = np.array([[(0, int(h*0.4)), (w, int(h*0.4)), (w, h), (0, h)]], dtype=np.int32)
    cv2.fillPoly(mask, pts, 255)
    masked = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(masked, 1, np.pi/180, 100, minLineLength=w//4, maxLineGap=50)

    if lines is None:
        return (0, int(h*0.7), w, int(h*0.7))

    best = max(lines, key=lambda l: np.linalg.norm(l[0][:2]-l[0][2:]))
    x1,y1,x2,y2 = best[0]

    if x1 > x2:
        x1,y1,x2,y2 = x2,y2,x1,y1

    print("🎯 Stop line:", (x1,y1,x2,y2))
    return (x1,y1,x2,y2)

results = {
    "records": [],
    "summary": {},
    "video_out": str(video_out_path)
}

seen = set()

if os.path.exists(VIDEO_PATH):

    cap = cv2.VideoCapture(VIDEO_PATH)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Detect stop line once
    ret, first_frame = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    STOP_LINE = auto_find_stop_line(first_frame)

    # Init engine
    engine = ViolationEngine({
        "fps": fps,
        "stop_line": STOP_LINE
    })

    out = cv2.VideoWriter(
        str(video_out_path),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (w, h)
    )

    frame_no = 0
    print("🚀 Processing video...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # ---------- DETECTION ----------
        custom_dets = detector.detect(frame)
        custom_dets = [d for d in custom_dets if d.conf >= 0.5]

        coco_results = coco_model(frame, conf=0.3)

        coco_dets = []
        for r in coco_results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = coco_model.names[cls]

                if label in ["traffic light", "car", "bus", "truck", "motorcycle", "bicycle"]:
                    x1,y1,x2,y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])

                    group = "traffic_light" if label == "traffic light" else "vehicle"

                    coco_dets.append(
                        Detection(
                            box=[x1,y1,x2,y2],
                            conf=conf,
                            class_id=cls,
                            class_name=label,
                            group=group
                        )
                    )

        detections = custom_dets + coco_dets

        # ---------- TRACKING ----------
        tracked = tracker.update(detections, frame)

        for det in tracked:
            if det.track_id != -1:
                traj.update(det.track_id, frame_no, det.center)

        # ---------- STATIC VIOLATIONS ----------
        for det in tracked:
            tid = det.track_id
            label = det.class_name
            box = det.box
            x1,y1,x2,y2 = map(int, box)

            key = (tid, label)

            if label in ["WITHOUT_HELMET", "USING_MOBILE", "MORE_THAN_TWO_PERSONS"]:
                if key not in seen:
                    seen.add(key)
                    
                    plate_text = plate_reader.process_vehicle(tid, frame, box)
                    
                    if plate_text:
                        owner = db.get_owner_details(plate_text)
                        if owner:
                            print(f"\n🚨 TICKET ISSUED 🚨")
                            print(f"Violation: {label}")
                            print(f"Vehicle: {owner['color']} {owner['model']} (Plate: {plate_text})")
                            print(f"Owner: {owner['name']} | Phone: {owner['phone_number']}")
                        else:
                            print(f"\n⚠️ Violation: {label} (Plate {plate_text} not found in vehicle DB)")

                        ingest_resp = db.log_violation(frame_no, label, plate_text)
                        print(f"📤 Sent to backend queue: {ingest_resp}")
                    else:
                        print(f"\n⚠️ Violation: {label} but plate not readable, not sent")
                    
                    results["records"].append({
                        "vehicle_id": tid,
                        "frame": frame_no,
                        "violation": label,
                        "plate_number": plate_text
                    })

            color = (0,255,0)
            if label == "WITHOUT_HELMET": color = (0,0,255)
            elif label == "USING_MOBILE": color = (255,0,0)
            elif label == "MORE_THAN_TWO_PERSONS": color = (0,255,255)

            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            cv2.putText(frame,f"{tid}-{label}",(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)

        # Cleanup old tracks to prevent memory leaks
        traj.cleanup_old_tracks(frame_no)
        
        # ---------- RED LIGHT ----------
        engine_records = engine.process_frame(frame, frame_no, tracked, traj)

        for r in engine_records:
            vid = r.get("vehicle_id")
            vbox = r.get("box")
            
            if vid is not None and vbox is not None:
                plate_text = plate_reader.process_vehicle(vid, frame, vbox)
                r["plate_number"] = plate_text
                
                if plate_text:
                    owner = db.get_owner_details(plate_text)
                    if owner:
                        print(f"\n🚨 TICKET ISSUED 🚨")
                        print(f"Violation: RED_LIGHT")
                        print(f"Vehicle: {owner['color']} {owner['model']} (Plate: {plate_text})")
                        print(f"Owner: {owner['name']} | Phone: {owner['phone_number']}")
                    else:
                        print(f"\n⚠️ Violation: RED_LIGHT (Plate {plate_text} not found in vehicle DB)")

                    ingest_resp = db.log_violation(frame_no, "RED_LIGHT", plate_text)
                    print(f"📤 Sent to backend queue: {ingest_resp}")
                else:
                    print("\n⚠️ Violation: RED_LIGHT but plate not readable, not sent")
            else:
                r["plate_number"] = None
                
            results["records"].append(r)

            if "box" in r:
                x1,y1,x2,y2 = map(int, r["box"])
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),4)
                cv2.putText(frame,"RED LIGHT!",(x1,y1-30),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),3)

        # Draw stop line
        cv2.line(frame,
                 (STOP_LINE[0],STOP_LINE[1]),
                 (STOP_LINE[2],STOP_LINE[3]),
                 (0,165,255),3)

        out.write(frame)
        cv2.imshow("Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

        frame_no += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

else:
    print("⚠️ Video not found")

results["summary"] = build_summary(results["records"])

with open(json_out_path,"w") as f:
    json.dump(results,f,indent=4)

with open(csv_out_path,"w",newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["vehicle_id","frame","violation"])
    for r in results["records"]:
        writer.writerow([r.get("vehicle_id"),r.get("frame"),r.get("violation")])

print("✅ DONE")
print("Summary:", results["summary"])

from src.database import TrafficDB

# Initialize DB
db = TrafficDB(password="0107@Bbs")

# Caches
owner_cache = {}          # plate → owner details
ticket_logged = set()     # (plate, violation) to avoid duplicates

print("✅ DB + Cache Ready")









