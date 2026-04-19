"""
detection.py — YOLOv8 wrapper for the Traffic Violation Detection System.

Responsibilities:
  - Load / cache Ultralytics YOLO model
  - Run inference on a single frame or batch
  - Parse raw results into a clean list of Detection dicts
  - Map COCO class IDs → semantic groups (vehicle, person, traffic_light, …)
"""

from __future__ import annotations

import numpy as np
import cv2
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger("tvd.detection")


# ─────────────────────────────────────────────
# Data structure
# ─────────────────────────────────────────────

@dataclass
class Detection:
    """A single detected object in one frame."""
    box:        np.ndarray       # [x1, y1, x2, y2] float32 in pixel coords
    conf:       float
    class_id:   int
    class_name: str
    group:      str              # 'vehicle' | 'person' | 'traffic_light' | 'phone' | 'other'

    # Filled by tracker later
    track_id:   int = -1

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.box[0] + self.box[2]) / 2,
                (self.box[1] + self.box[3]) / 2)

    @property
    def bottom_center(self) -> Tuple[float, float]:
        return ((self.box[0] + self.box[2]) / 2, self.box[3])

    @property
    def area(self) -> float:
        return float((self.box[2] - self.box[0]) * (self.box[3] - self.box[1]))

    def to_dict(self) -> Dict:
        return {
            "box":        self.box.tolist(),
            "conf":       round(float(self.conf), 3),
            "class_id":   self.class_id,
            "class_name": self.class_name,
            "group":      self.group,
            "track_id":   self.track_id,
        }


# ─────────────────────────────────────────────
# Class-ID → group mapping (COCO defaults)
# Override if you fine-tune on a custom dataset.
# ─────────────────────────────────────────────

COCO_GROUP_MAP: Dict[int, str] = {
    0:  "person",
    1:  "vehicle",   # bicycle
    2:  "vehicle",   # car
    3:  "vehicle",   # motorcycle
    5:  "vehicle",   # bus
    7:  "vehicle",   # truck
    9:  "traffic_light",
    11: "stop_sign",
    67: "phone",     # cell phone
}

# Vehicle sub-type map
VEHICLE_TYPE_MAP: Dict[int, str] = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Classes that can carry people on two wheels
TWO_WHEELER_IDS = {1, 3}


# ─────────────────────────────────────────────
# Detector class
# ─────────────────────────────────────────────

class YOLODetector:
    """
    Thin wrapper around Ultralytics YOLO.

    Parameters
    ----------
    weights : str
        Path to .pt file or Ultralytics model name (e.g. 'yolov8n.pt').
    conf    : float
        Minimum confidence threshold.
    iou     : float
        NMS IoU threshold.
    device  : str
        'cpu', '0', 'cuda', etc.
    imgsz   : int
        Inference input size.
    group_map : dict
        Custom class_id → group overrides (for fine-tuned models).
    """

    def __init__(self,
                 weights:   str   = "yolov8n.pt",
                 conf:      float = 0.35,
                 iou:       float = 0.45,
                 device:    str   = "cpu",
                 imgsz:     int   = 640,
                 group_map: Optional[Dict[int, str]] = None):

        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError("Ultralytics not installed. Run: pip install ultralytics") from e

        logger.info(f"Loading YOLO model: {weights} on device={device}")
        self.model     = YOLO(weights)
        self.conf      = conf
        self.iou       = iou
        self.device    = device
        self.imgsz     = imgsz
        self.group_map = {**COCO_GROUP_MAP, **(group_map or {})}

        # Try to extract class names from the model
        if hasattr(self.model, "names") and self.model.names:
            self.class_names: Dict[int, str] = self.model.names
        else:
            self.class_names = {}

        logger.info("Model loaded successfully.")

    # ── core inference ──────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run inference on a single BGR frame (numpy array).

        Returns a list of Detection objects.
        """
        results = self.model.predict(
            source  = frame,
            conf    = self.conf,
            iou     = self.iou,
            device  = self.device,
            imgsz   = self.imgsz,
            verbose = False,
        )
        return self._parse_results(results)

    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """Run inference on a list of BGR frames. Returns one list per frame."""
        if not frames:
            return []
        results = self.model.predict(
            source  = frames,
            conf    = self.conf,
            iou     = self.iou,
            device  = self.device,
            imgsz   = self.imgsz,
            verbose = False,
        )
        return [self._parse_results([r]) for r in results]

    # ── parsing ─────────────────────────────────────────────────────────

    def _parse_results(self, results) -> List[Detection]:
        detections: List[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            xyxy  = boxes.xyxy.cpu().numpy()     # (N, 4)
            confs = boxes.conf.cpu().numpy()     # (N,)
            cls   = boxes.cls.cpu().numpy().astype(int)  # (N,)

            for i in range(len(cls)):
                cid   = int(cls[i])
                cname = self.class_names.get(cid, str(cid))
                group = self.group_map.get(cid, "other")
                det   = Detection(
                    box        = xyxy[i].astype(np.float32),
                    conf       = float(confs[i]),
                    class_id   = cid,
                    class_name = cname,
                    group      = group,
                )
                detections.append(det)
        return detections

    # ── convenience filters ─────────────────────────────────────────────

    @staticmethod
    def filter_by_group(dets: List[Detection], group: str) -> List[Detection]:
        return [d for d in dets if d.group == group]

    @staticmethod
    def filter_vehicles(dets: List[Detection]) -> List[Detection]:
        return [d for d in dets if d.group == "vehicle"]

    @staticmethod
    def filter_persons(dets: List[Detection]) -> List[Detection]:
        return [d for d in dets if d.group == "person"]

    @staticmethod
    def filter_two_wheelers(dets: List[Detection]) -> List[Detection]:
        return [d for d in dets if d.class_id in TWO_WHEELER_IDS]

    @staticmethod
    def filter_traffic_lights(dets: List[Detection]) -> List[Detection]:
        return [d for d in dets if d.group == "traffic_light"]

    @staticmethod
    def filter_phones(dets: List[Detection]) -> List[Detection]:
        return [d for d in dets if d.group == "phone"]


# ─────────────────────────────────────────────
# Traffic-light state inference
# ─────────────────────────────────────────────

def infer_traffic_light_state(frame: np.ndarray,
                               light_box: np.ndarray) -> str:
    """
    Estimate traffic light state (red / green / yellow / unknown)
    by analysing the dominant hue inside the bounding box crop.

    This is a heuristic — a fine-tuned classifier would be more accurate.
    """
    x1, y1, x2, y2 = map(int, light_box[:4])
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return "unknown"

    hsv   = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # Red  — two ranges in HSV hue
    r1 = cv2.inRange(hsv, np.array([0,  120, 120]), np.array([10, 255, 255]))
    r2 = cv2.inRange(hsv, np.array([160, 120, 120]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(r1, r2)

    # Green
    green_mask   = cv2.inRange(hsv, np.array([40, 80, 80]),  np.array([90,  255, 255]))

    # Yellow / amber
    yellow_mask  = cv2.inRange(hsv, np.array([15, 120, 120]), np.array([35, 255, 255]))

    counts = {
        "red":    int(np.sum(red_mask    > 0)),
        "green":  int(np.sum(green_mask  > 0)),
        "yellow": int(np.sum(yellow_mask > 0)),
    }

    best = max(counts, key=counts.get)
    if counts[best] < 30:        # too few pixels → inconclusive
        return "unknown"
    return best
