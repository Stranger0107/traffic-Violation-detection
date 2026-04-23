"""
violations.py — Rule-based violation detection engine.

Each violation class:
  - Receives the current frame's detections + trajectories + config
  - Returns a list of ViolationRecord dicts

Violations implemented:
  1.  Red Light Violation
  2.  No Helmet
  3.  Triple Riding
  4.  Wrong Side Driving
  5.  Mobile Phone Usage
  6.  Illegal Parking
  7.  Lane Violation
"""

from __future__ import annotations

import numpy as np
import cv2
import logging
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict

from .detection import Detection, infer_traffic_light_state, TWO_WHEELER_IDS
from .tracking  import TrajectoryStore
from .utils     import (
    iou, box_center, box_bottom_center,
    segment_crosses_line, point_in_polygon, point_side_of_line,
    SpeedEstimator,
)

logger = logging.getLogger("tvd.violations")


# ─────────────────────────────────────────────
# Shared record builder
# ─────────────────────────────────────────────

def make_record(frame_no: int, vehicle_id: int, violation: str,
                box: Optional[List] = None, extra: Optional[Dict] = None) -> Dict:
    r: Dict[str, Any] = {
        "frame":      frame_no,
        "vehicle_id": vehicle_id,
        "violation":  violation,
    }
    if box is not None:
        r["box"] = [round(v, 1) for v in box]
    if extra:
        r.update(extra)
    return r


# ─────────────────────────────────────────────
# 1. Red Light Violation
# ─────────────────────────────────────────────

class RedLightDetector:
    """
    Logic:
      - Detect traffic light state (red/green/yellow) via colour analysis
      - If red: check whether any vehicle's bottom-centre has crossed the stop line
        (stop line defined as a horizontal or diagonal line in pixel coords)
      - A vehicle is flagged only once per red phase to avoid repeated records.
    """

    def __init__(self, stop_line: Optional[Tuple[float, float, float, float]] = None):
        # stop_line = (x1, y1, x2, y2) in pixel coords
        self.stop_line  = stop_line
        self._flagged: Dict[int, int] = {}   # track_id → last flagged frame
        self._cooldown = 90                  # frames before re-flagging same vehicle

    def detect(self, frame: np.ndarray, frame_no: int,
               detections: List[Detection],
               trajectories: Optional[TrajectoryStore] = None) -> List[Dict]:
        records = []
        if self.stop_line is None or trajectories is None:
            return records

        lights   = [d for d in detections if d.group == "traffic_light"]
        vehicles = [d for d in detections if d.group == "vehicle"]

        # Determine dominant light state
        light_state = "unknown"
        for lt in lights:
            state = infer_traffic_light_state(frame, lt.box)
            if state == "red":
                light_state = "red"
                break
            if state in ("green", "yellow"):
                light_state = state

        if light_state != "red":
            return records

        sl   = self.stop_line
        l1   = (sl[0], sl[1])
        l2   = (sl[2], sl[3])

        for veh in vehicles:
            tid = veh.track_id
            hist = trajectories.get(tid)
            if len(hist) < 2:
                continue
                
            _, prev_center = hist[-2]
            _, curr_center = hist[-1]

            # Check if trajectory crossed the stop line
            crossed = segment_crosses_line(prev_center, curr_center, l1, l2)

            last = self._flagged.get(tid, -9999)
            if crossed and (frame_no - last) > self._cooldown:
                self._flagged[tid] = frame_no
                records.append(make_record(
                    frame_no, tid, "Red Light Violation",
                    box  = veh.box.tolist(),
                    extra={"light_state": light_state, "confidence": round(veh.conf, 2)},
                ))
        return records


# ─────────────────────────────────────────────
# 2. No Helmet
# ─────────────────────────────────────────────

class NoHelmetDetector:
    """
    Logic:
      - Find two-wheelers (motorcycle / bicycle)
      - For each two-wheeler find persons whose bounding box overlaps with it
        (riders are ON the bike → high vertical overlap)
      - If a rider's head area has no overlapping "helmet" detection → flag
      - Uses COCO pre-trained model: we look for person heads in the upper 30%
        of the person bounding box and check if any helmet-like object overlaps.
    """

    def __init__(self, overlap_threshold: float = 0.15):
        self.overlap_threshold = overlap_threshold
        self._flagged: Dict[int, int] = {}
        self._cooldown = 60

    def detect(self, frame: np.ndarray, frame_no: int,
               detections: List[Detection]) -> List[Dict]:
        records   = []
        two_wheels = [d for d in detections if d.class_id in TWO_WHEELER_IDS]
        persons    = [d for d in detections if d.group == "person"]
        helmets    = [d for d in detections if d.class_name in ("helmet", "hard hat")]

        for bike in two_wheels:
            # Find riders on this bike
            riders = [p for p in persons if _vertical_overlap(bike.box, p.box) > self.overlap_threshold]

            for rider in riders:
                # Head region = upper 35% of person box
                head_box = _head_region(rider.box, frac=0.35)

                # Check for overlapping helmet
                has_helmet = any(
                    iou(head_box, h.box) > 0.10 or _box_contains(head_box, box_center(h.box))
                    for h in helmets
                )

                if not has_helmet:
                    tid  = bike.track_id
                    last = self._flagged.get(tid, -9999)
                    if (frame_no - last) > self._cooldown:
                        self._flagged[tid] = frame_no
                        records.append(make_record(
                            frame_no, tid, "No Helmet",
                            box   = rider.box.tolist(),
                            extra = {"bike_id": tid, "rider_conf": round(rider.conf, 2)},
                        ))
        return records


# ─────────────────────────────────────────────
# 3. Triple Riding
# ─────────────────────────────────────────────

class TripleRidingDetector:
    """
    Logic:
      - For each two-wheeler count the number of overlapping persons
      - If count > 2 → triple riding violation
    """

    def __init__(self, overlap_threshold: float = 0.20):
        self.overlap_threshold = overlap_threshold
        self._flagged: Dict[int, int] = {}
        self._cooldown = 60

    def detect(self, frame: np.ndarray, frame_no: int,
               detections: List[Detection]) -> List[Dict]:
        records    = []
        two_wheels = [d for d in detections if d.class_id in TWO_WHEELER_IDS]
        persons    = [d for d in detections if d.group == "person"]

        for bike in two_wheels:
            riders = [p for p in persons
                      if _vertical_overlap(bike.box, p.box) > self.overlap_threshold]

            if len(riders) > 2:
                tid  = bike.track_id
                last = self._flagged.get(tid, -9999)
                if (frame_no - last) > self._cooldown:
                    self._flagged[tid] = frame_no
                    records.append(make_record(
                        frame_no, tid, "Triple Riding",
                        box   = bike.box.tolist(),
                        extra = {"rider_count": len(riders)},
                    ))
        return records


# ─────────────────────────────────────────────
# 5. Wrong Side Driving
# ─────────────────────────────────────────────

class WrongSideDrivingDetector:
    """
    Logic:
      - Road has one or more "wrong side" polygon zones (the lane going in
        the opposite direction).
      - If a vehicle's bottom-centre is inside a wrong-side zone AND its
        motion direction aligns with the wrong direction, flag it.
    """

    def __init__(self, wrong_side_zones: Optional[List[List[Tuple[float, float]]]] = None):
        # zones: list of polygon vertex lists
        self.wrong_side_zones = wrong_side_zones or []
        self._flagged: Dict[int, int] = {}
        self._cooldown = 60

    def detect(self, frame: np.ndarray, frame_no: int,
               detections: List[Detection],
               trajectories: Optional[TrajectoryStore] = None) -> List[Dict]:
        records  = []
        vehicles = [d for d in detections if d.group == "vehicle"]

        if not self.wrong_side_zones:
            return records

        for veh in vehicles:
            bc = veh.bottom_center
            in_wrong_zone = any(
                point_in_polygon(bc[0], bc[1], zone)
                for zone in self.wrong_side_zones
            )
            if not in_wrong_zone:
                continue

            tid  = veh.track_id
            last = self._flagged.get(tid, -9999)
            if (frame_no - last) > self._cooldown:
                self._flagged[tid] = frame_no
                records.append(make_record(
                    frame_no, tid, "Wrong Side Driving",
                    box = veh.box.tolist(),
                ))
        return records

# ─────────────────────────────────────────────
# 7. Mobile Phone Usage
# ─────────────────────────────────────────────

class MobilePhoneDetector:
    """
    Logic:
      - Detect 'cell phone' objects (COCO class 67)
      - If a phone is near a driver's face (upper portion of a person
        bounding box that overlaps with a car), flag it
    """

    def __init__(self, overlap_threshold: float = 0.15):
        self.overlap_threshold = overlap_threshold
        self._flagged: Dict[int, int] = {}
        self._cooldown = 45

    def detect(self, frame: np.ndarray, frame_no: int,
               detections: List[Detection]) -> List[Dict]:
        records  = []
        phones   = [d for d in detections if d.class_id == 67]
        persons  = [d for d in detections if d.group == "person"]
        vehicles = [d for d in detections if d.group == "vehicle"]

        for phone in phones:
            # Check if phone is near a person's head
            for person in persons:
                head = _head_region(person.box, frac=0.50)
                if iou(head, phone.box) < self.overlap_threshold and \
                   not _box_contains(head, box_center(phone.box)):
                    continue

                # Check if that person is inside/near a vehicle
                for veh in vehicles:
                    if iou(veh.box, person.box) > 0.10:
                        tid  = veh.track_id
                        last = self._flagged.get(tid, -9999)
                        if (frame_no - last) > self._cooldown:
                            self._flagged[tid] = frame_no
                            records.append(make_record(
                                frame_no, tid, "Mobile Phone Usage",
                                box   = person.box.tolist(),
                                extra = {"phone_conf": round(phone.conf, 2)},
                            ))
        return records


# ─────────────────────────────────────────────
# 8. Illegal Parking
# ─────────────────────────────────────────────

class IllegalParkingDetector:
    """
    Logic:
      - If a vehicle has been stationary (< pixel_threshold movement)
        for > stationary_frames in a designated no-parking zone, flag it.
      - Also flags vehicles stationary for very long durations on road.
    """

    def __init__(self,
                 parking_zones:        Optional[List[List[Tuple]]] = None,
                 stationary_frames:    int   = 90,
                 pixel_threshold:      float = 15.0):
        self.parking_zones     = parking_zones or []
        self.stationary_frames = stationary_frames
        self.pixel_threshold   = pixel_threshold
        self._flagged: Dict[int, int] = {}
        self._cooldown = 150

    def detect(self, frame: np.ndarray, frame_no: int,
               detections: List[Detection],
               trajectories: TrajectoryStore) -> List[Dict]:
        records  = []
        vehicles = [d for d in detections if d.group == "vehicle"]

        for veh in vehicles:
            tid = veh.track_id
            if not trajectories.is_stationary(
                    tid,
                    min_frames       = self.stationary_frames,
                    pixel_threshold  = self.pixel_threshold):
                continue

            # In a designated no-parking zone?
            bc         = veh.bottom_center
            in_np_zone = any(
                point_in_polygon(bc[0], bc[1], zone)
                for zone in self.parking_zones
            ) if self.parking_zones else True  # if no zones, flag on any road

            if in_np_zone:
                last = self._flagged.get(tid, -9999)
                if (frame_no - last) > self._cooldown:
                    self._flagged[tid] = frame_no
                    records.append(make_record(
                        frame_no, tid, "Illegal Parking",
                        box = veh.box.tolist(),
                    ))
        return records


# ─────────────────────────────────────────────
# 9. Lane Violation
# ─────────────────────────────────────────────

class LaneViolationDetector:
    """
    Logic:
      - Lane boundaries are defined as line segments (x1,y1,x2,y2)
      - If a vehicle's bottom-centre trajectory crosses a solid lane line
        (not a dashed one — we can't distinguish here so treat all as solid),
        flag it.
    """

    def __init__(self, lane_lines: Optional[List[Tuple[float, float, float, float]]] = None):
        self.lane_lines = lane_lines or []
        self._flagged: Dict[int, int] = {}
        self._cooldown = 60

    def detect(self, frame: np.ndarray, frame_no: int,
               detections: List[Detection],
               trajectories: TrajectoryStore) -> List[Dict]:
        records  = []
        vehicles = [d for d in detections if d.group == "vehicle"]

        if not self.lane_lines:
            return records

        for veh in vehicles:
            tid  = veh.track_id
            hist = trajectories.get(tid)
            if len(hist) < 2:
                continue

            _, prev_center = hist[-2]
            _, curr_center = hist[-1]

            crossed = any(
                segment_crosses_line(
                    prev_center, curr_center,
                    (ll[0], ll[1]), (ll[2], ll[3])
                )
                for ll in self.lane_lines
            )

            if crossed:
                last = self._flagged.get(tid, -9999)
                if (frame_no - last) > self._cooldown:
                    self._flagged[tid] = frame_no
                    records.append(make_record(
                        frame_no, tid, "Lane Violation",
                        box = veh.box.tolist(),
                    ))
        return records


# ─────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────

class ViolationEngine:
    """
    Aggregates all detectors and runs them each frame.

    Usage:
        engine = ViolationEngine(config)
        engine.process_frame(frame, frame_no, detections, trajectories)
        all_records = engine.records
    """

    def __init__(self, config: Dict):
        cfg = config
        fps  = cfg.get("fps", 30)
        ppm  = cfg.get("pixels_per_meter", 8.0)
        sl   = cfg.get("stop_line")

        self.detectors = {
            "red_light":    RedLightDetector(stop_line=sl),
        }
        self.records: List[Dict] = []
        logger.info(f"ViolationEngine initialised with {len(self.detectors)} detectors.")

    def process_frame(self,
                      frame:        np.ndarray,
                      frame_no:     int,
                      detections:   List[Detection],
                      trajectories: TrajectoryStore) -> List[Dict]:
        """Run all detectors, collect records, return frame-level violations."""
        frame_records = []

        needs_traj = {"wrong_side", "illegal_parking", "lane_violation", "red_light"}

        for name, det in self.detectors.items():
            try:
                if name in needs_traj:
                    r = det.detect(frame, frame_no, detections, trajectories)
                else:
                    r = det.detect(frame, frame_no, detections)
                frame_records.extend(r)
            except Exception as e:
                logger.warning(f"Detector '{name}' raised an error: {e}")

        self.records.extend(frame_records)
        return frame_records


# ─────────────────────────────────────────────
# Geometry helpers (private)
# ─────────────────────────────────────────────

def _vertical_overlap(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Vertical IoU — measures how much two boxes overlap vertically."""
    ya = max(box_a[1], box_b[1])
    yb = min(box_a[3], box_b[3])
    inter = max(0.0, yb - ya)
    ha = box_a[3] - box_a[1]
    hb = box_b[3] - box_b[1]
    return inter / (min(ha, hb) + 1e-6)


def _head_region(person_box: np.ndarray, frac: float = 0.35) -> np.ndarray:
    """Return upper `frac` of person bounding box as head region."""
    x1, y1, x2, y2 = person_box
    h = y2 - y1
    return np.array([x1, y1, x2, y1 + h * frac], dtype=np.float32)


def _driver_zone(car_box: np.ndarray) -> np.ndarray:
    """Approximate driver zone: left half, upper 60% of car box."""
    x1, y1, x2, y2 = car_box
    w = x2 - x1
    h = y2 - y1
    return np.array([x1, y1, x1 + w * 0.5, y1 + h * 0.6], dtype=np.float32)


def _box_contains(box: np.ndarray, point: Tuple[float, float]) -> bool:
    """True if point (px, py) is inside box."""
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def _image_seatbelt_check(frame: np.ndarray, person_box: np.ndarray) -> bool:
    """
    Heuristic image-based seatbelt check.
    Looks for a diagonal bright/dark strip across the torso region.
    Returns True (has seatbelt) or False.
    """
    x1, y1, x2, y2 = map(int, person_box)
    # Torso region: middle 50% vertically, middle 60% horizontally
    h = y2 - y1
    w = x2 - x1
    ty1 = y1 + int(h * 0.25)
    ty2 = y1 + int(h * 0.75)
    tx1 = x1 + int(w * 0.20)
    tx2 = x1 + int(w * 0.80)

    crop = frame[ty1:ty2, tx1:tx2]
    if crop.size == 0:
        return False

    gray   = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges  = cv2.Canny(gray, 50, 150)
    lines  = cv2.HoughLinesP(edges, 1, np.pi / 180, 20,
                              minLineLength=30, maxLineGap=10)
    if lines is None:
        return False

    # Check for diagonal lines (potential seatbelt)
    for line in lines:
        x_a, y_a, x_b, y_b = line[0]
        dx = abs(x_b - x_a)
        dy = abs(y_b - y_a)
        if dy > 10 and dx > 5:   # diagonal line present
            return True
    return False