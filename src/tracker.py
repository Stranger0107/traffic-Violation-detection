# src/tracking.py

from __future__ import annotations
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

logger = logging.getLogger("tvd.tracking")

# ─────────────────────────────────────────────
# Detection structure (compatible with your code)
# ─────────────────────────────────────────────

class Detection:
    def __init__(self, box, conf, class_name):
        self.box = np.array(box, dtype=float)  # [x1, y1, x2, y2]
        self.conf = float(conf)
        self.class_name = class_name
        self.track_id = -1

    @property
    def center(self):
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2, (y1 + y2) / 2)


# ─────────────────────────────────────────────
# Base tracker
# ─────────────────────────────────────────────

class BaseTracker:
    def update(self, detections: List[Detection], frame):
        raise NotImplementedError


# ─────────────────────────────────────────────
# DeepSORT Tracker
# ─────────────────────────────────────────────

class DeepSORTTracker(BaseTracker):
    def __init__(self, max_age=30, n_init=3):
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
        except ImportError:
            raise ImportError("Install: pip install deep-sort-realtime")

        self.tracker = DeepSort(max_age=max_age, n_init=n_init)
        logger.info("✅ DeepSORT tracker initialised")

    def update(self, detections: List[Detection], frame):

        if not detections:
            return []

        ds_input = []
        for det in detections:
            x1, y1, x2, y2 = det.box
            ds_input.append((
                [x1, y1, x2 - x1, y2 - y1],
                det.conf,
                det.class_name
            ))

        tracks = self.tracker.update_tracks(ds_input, frame=frame)

        results = []
        assigned = set()

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            l, t, r, b = track.to_ltrb()
            t_box = np.array([l, t, r, b])

            # match best detection
            best_iou = 0
            best_idx = -1

            for i, det in enumerate(detections):
                if i in assigned:
                    continue

                iou = _iou(det.box, t_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i

            if best_idx != -1 and best_iou > 0.3:
                detections[best_idx].track_id = track_id
                results.append(detections[best_idx])
                assigned.add(best_idx)

        # unmatched detections
        for i, det in enumerate(detections):
            if i not in assigned:
                results.append(det)

        return results


# ─────────────────────────────────────────────
# IoU Tracker (fallback)
# ─────────────────────────────────────────────

class IoUTracker(BaseTracker):
    def __init__(self, iou_thresh=0.4, max_missing=10):
        self.iou_thresh = iou_thresh
        self.max_missing = max_missing
        self.next_id = 1
        self.tracks: Dict[int, Dict] = {}

    def update(self, detections: List[Detection], frame):

        if not self.tracks:
            for det in detections:
                det.track_id = self._register(det)
            return detections

        used_tracks = set()
        used_dets = set()

        for tid, t in self.tracks.items():
            best_iou = 0
            best_idx = -1

            for i, det in enumerate(detections):
                if i in used_dets:
                    continue

                iou = _iou(t["box"], det.box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i

            if best_iou > self.iou_thresh:
                detections[best_idx].track_id = tid
                self.tracks[tid]["box"] = detections[best_idx].box
                used_tracks.add(tid)
                used_dets.add(best_idx)

        # new tracks
        for i, det in enumerate(detections):
            if i not in used_dets:
                det.track_id = self._register(det)

        return detections

    def _register(self, det):
        tid = self.next_id
        self.next_id += 1
        self.tracks[tid] = {"box": det.box.copy()}
        return tid


# ─────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────

def build_tracker(method="deepsort", **kwargs):

    if method == "deepsort":
        try:
            return DeepSORTTracker(**kwargs)
        except:
            logger.warning("⚠️ Falling back to IoU tracker")
            return IoUTracker(**kwargs)

    elif method == "iou":
        return IoUTracker(**kwargs)

    else:
        raise ValueError("Invalid tracker type")


# ─────────────────────────────────────────────
# Trajectory Store
# ─────────────────────────────────────────────

class TrajectoryStore:
    def __init__(self, maxlen=60):
        self.maxlen = maxlen
        self.store = defaultdict(list)

    def update(self, track_id, frame_no, center):
        self.store[track_id].append((frame_no, center))
        if len(self.store[track_id]) > self.maxlen:
            self.store[track_id].pop(0)

    def get(self, track_id):
        return self.store.get(track_id, [])


# ─────────────────────────────────────────────
# IoU function
# ─────────────────────────────────────────────

def _iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0

    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])

    return inter / (area_a + area_b - inter)