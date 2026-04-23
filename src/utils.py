"""
utils.py — Shared helpers for the Traffic Violation Detection System.

Responsibilities:
  - Geometry utilities (IoU, line crossing, polygon checks)
  - Speed estimation
  - Drawing / annotation helpers
  - Logging setup
  - Config loading
"""

import cv2
import numpy as np
import logging
import json
import csv
import os
import time
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Optional, Any


# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────

def get_logger(name: str = "tvd") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s [%(name)s] %(message)s",
                                datefmt="%H:%M:%S")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


logger = get_logger()


# ─────────────────────────────────────────────
# Class definitions — extend as needed
# ─────────────────────────────────────────────

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

# Custom class names (used when fine-tuning on traffic dataset)
CUSTOM_CLASSES = [
    "car", "motorcycle", "bus", "truck", "person",
    "helmet", "no_helmet", "traffic_light_red", "traffic_light_green",
    "traffic_light_yellow", "mobile_phone", "seatbelt", "no_seatbelt",
]

# COCO class IDs we care about (for pre-trained inference)
VEHICLE_IDS = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
PERSON_ID   = 0
TRAFFIC_LIGHT_ID = 9
CELL_PHONE_ID    = 67
STOP_SIGN_ID     = 11

VIOLATION_COLORS = {
    "Red Light Violation":     (0,   0,   255),
    "No Helmet":               (255, 0,   0),
    "Triple Riding":           (255, 165, 0),
    "No Seatbelt":             (128, 0,   128),
    "Wrong Side Driving":      (0,   128, 255),
    "Over Speeding":           (255, 255, 0),
    "Mobile Phone Usage":      (0,   255, 255),
    "Illegal Parking":         (255, 20,  147),
    "Lane Violation":          (0,   255, 0),
}

DEFAULT_BOX_COLOR  = (0, 255, 0)
TRACK_ID_COLOR     = (255, 255, 255)


# ─────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────

def iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    if inter == 0:
        return 0.0
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter)


def box_center(box: np.ndarray) -> Tuple[float, float]:
    """Return (cx, cy) of a [x1,y1,x2,y2] box."""
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def box_bottom_center(box: np.ndarray) -> Tuple[float, float]:
    """Return bottom-center point — good proxy for vehicle ground position."""
    return ((box[0] + box[2]) / 2, box[3])


def point_side_of_line(px: float, py: float,
                       lx1: float, ly1: float,
                       lx2: float, ly2: float) -> float:
    """
    Sign of the cross product (line_vec × point_vec).
    > 0  → point is LEFT of directed line (lx1,ly1)→(lx2,ly2)
    < 0  → point is RIGHT
    = 0  → on the line
    """
    return (lx2 - lx1) * (py - ly1) - (ly2 - ly1) * (px - lx1)


def segment_crosses_line(p1: Tuple[float, float], p2: Tuple[float, float],
                         l1: Tuple[float, float], l2: Tuple[float, float]) -> bool:
    """
    True if the segment p1→p2 crosses the finite segment l1→l2.
    Uses sign-change of cross-product on both segments.
    """
    s1 = point_side_of_line(p1[0], p1[1], l1[0], l1[1], l2[0], l2[1])
    s2 = point_side_of_line(p2[0], p2[1], l1[0], l1[1], l2[0], l2[1])
    s3 = point_side_of_line(l1[0], l1[1], p1[0], p1[1], p2[0], p2[1])
    s4 = point_side_of_line(l2[0], l2[1], p1[0], p1[1], p2[0], p2[1])
    return (s1 * s2) < 0 and (s3 * s4) < 0


def point_in_polygon(px: float, py: float, polygon: List[Tuple[float, float]]) -> bool:
    """Ray-casting algorithm for point-in-polygon."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-9) + xi):
            inside = not inside
        j = i
    return inside


def pixel_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


# ─────────────────────────────────────────────
# Speed estimation
# ─────────────────────────────────────────────

class SpeedEstimator:
    """
    Estimates vehicle speed using a simple homography-based pixel→meter mapping.

    In production, calibrate `pixels_per_meter` from real road markings.
    Default values are reasonable for a typical surveillance camera at ~6–10 m height.
    """

    def __init__(self, fps: float = 30.0, pixels_per_meter: float = 8.0):
        self.fps = fps
        self.pixels_per_meter = pixels_per_meter
        # track_id → deque of (frame_no, (cx, cy))
        self.history: Dict[int, List[Tuple[int, Tuple[float, float]]]] = {}
        self.speeds: Dict[int, float] = {}   # track_id → smoothed km/h

    def update(self, track_id: int, frame_no: int, center: Tuple[float, float]) -> Optional[float]:
        if track_id not in self.history:
            self.history[track_id] = []
        self.history[track_id].append((frame_no, center))
        # keep last 10 frames
        self.history[track_id] = self.history[track_id][-10:]

        hist = self.history[track_id]
        if len(hist) < 2:
            return None

        f0, c0 = hist[0]
        fn, cn = hist[-1]
        dt = (fn - f0) / self.fps          # seconds
        if dt < 1e-6:
            return None

        dist_px = pixel_distance(c0, cn)
        dist_m  = dist_px / self.pixels_per_meter
        speed_ms = dist_m / dt
        speed_kmh = speed_ms * 3.6

        # exponential smoothing
        prev = self.speeds.get(track_id, speed_kmh)
        smoothed = 0.7 * prev + 0.3 * speed_kmh
        self.speeds[track_id] = smoothed
        return smoothed

    def get_speed(self, track_id: int) -> float:
        return self.speeds.get(track_id, 0.0)


# ─────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────

def draw_box(frame: np.ndarray,
             box: np.ndarray,
             label: str = "",
             color: Tuple[int, int, int] = DEFAULT_BOX_COLOR,
             thickness: int = 2,
             font_scale: float = 0.55) -> np.ndarray:
    x1, y1, x2, y2 = map(int, box[:4])
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def draw_violations_overlay(frame: np.ndarray,
                             violations: List[Dict],
                             frame_no: int) -> np.ndarray:
    """Annotate frame with all active violations."""
    for v in violations:
        box   = v.get("box")
        vtype = v.get("violation", "Unknown")
        tid   = v.get("vehicle_id", -1)
        color = VIOLATION_COLORS.get(vtype, (0, 0, 255))
        label = f"#{tid} {vtype}"
        if box is not None:
            draw_box(frame, np.array(box), label, color, thickness=2)

    # Frame counter
    cv2.putText(frame, f"Frame: {frame_no}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2, cv2.LINE_AA)
    return frame


def draw_roi_lines(frame: np.ndarray,
                   stop_line: Optional[Tuple] = None,
                   lane_lines: Optional[List] = None,
                   roi_zones: Optional[List] = None) -> np.ndarray:
    """Draw configured ROI lines / zones on frame (for debugging)."""
    if stop_line:
        pt1, pt2 = (int(stop_line[0]), int(stop_line[1])), (int(stop_line[2]), int(stop_line[3]))
        cv2.line(frame, pt1, pt2, (0, 0, 255), 2)
        cv2.putText(frame, "STOP LINE", pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    if lane_lines:
        for ll in lane_lines:
            cv2.line(frame, (int(ll[0]), int(ll[1])), (int(ll[2]), int(ll[3])), (0, 255, 255), 1)
    if roi_zones:
        for zone in roi_zones:
            pts = np.array(zone, dtype=np.int32)
            cv2.polylines(frame, [pts], True, (255, 128, 0), 2)
    return frame


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

DEFAULT_CONFIG = {
    "model_weights": "yolov8n.pt",
    "conf_threshold": 0.35,
    "iou_threshold":  0.45,
    "fps": 30,
    "speed_limit_kmh": 60,
    "pixels_per_meter": 8.0,
    # Stop line: (x1, y1, x2, y2) in pixels — override per-video
    "stop_line": None,
    # Lane direction lines list — each: (x1,y1,x2,y2)
    "lane_lines": [],
    # Parking zones — each: [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
    "parking_zones": [],
    # Wrong-side region polygons
    "wrong_side_zones": [],
    # Stationary threshold in frames for illegal parking
    "parking_stationary_frames": 90,
    # Minimum frames a violation must persist before being recorded
    "violation_min_frames": 3,
    "output_dir": "outputs",
}


def load_config(path: Optional[str] = None) -> Dict:
    cfg = DEFAULT_CONFIG.copy()
    if path and os.path.isfile(path):
        import yaml
        with open(path) as f:
            user_cfg = yaml.safe_load(f)
        cfg.update(user_cfg)
        logger.info(f"Config loaded from {path}")
    return cfg


# ─────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────

def save_report_json(records: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(records, f, indent=2, default=str)
    logger.info(f"JSON report saved → {path}")


def save_report_csv(records: List[Dict], path: str) -> None:
    if not records:
        return

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # ✅ Collect ALL keys from ALL records
    all_keys = set()
    for r in records:
        all_keys.update(r.keys())

    keys = sorted(list(all_keys))  # sorted = cleaner CSV

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    logger.info(f"CSV report saved → {path}")


def build_summary(records: List[Dict]) -> Dict[str, int]:
    summary: Dict[str, int] = {}
    for r in records:
        vtype = r.get("violation", "Unknown")
        summary[vtype] = summary.get(vtype, 0) + 1
    return summary


def timestamp_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")
