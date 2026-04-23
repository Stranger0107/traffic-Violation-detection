"""
tracking.py — Object tracking wrapper for the Traffic Violation Detection System.

Supports:
  1. DeepSORT (via deep_sort_realtime)
  2. Simple IoU-based fallback tracker (no extra dependencies)

Usage:
    tracker = build_tracker(method="deepsort")   # or "iou"
    tracked = tracker.update(detections, frame)
    # tracked is a list of Detection with .track_id filled in
"""

from __future__ import annotations

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from .detection import Detection

logger = logging.getLogger("tvd.tracking")


# ─────────────────────────────────────────────
# Base interface
# ─────────────────────────────────────────────

class BaseTracker:
    def update(self, detections: List[Detection],
               frame: np.ndarray) -> List[Detection]:
        raise NotImplementedError


# ─────────────────────────────────────────────
# DeepSORT tracker
# ─────────────────────────────────────────────

class DeepSORTTracker(BaseTracker):
    """
    Wraps deep_sort_realtime.DeepSort.

    Install:  pip install deep_sort_realtime
    """

    def __init__(self,
                 max_age:        int   = 30,
                 n_init:         int   = 3,
                 max_cosine_dist: float = 0.4,
                 nn_budget:      int   = 100,
                 embedder:       str   = "mobilenet"):
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
        except ImportError as e:
            raise ImportError(
                "deep_sort_realtime not installed. "
                "Run: pip install deep_sort_realtime"
            ) from e

        self._tracker = DeepSort(
            max_age          = max_age,
            n_init           = n_init,
            max_cosine_distance = max_cosine_dist,
            nn_budget        = nn_budget,
            embedder         = embedder,
            half             = False,
            bgr              = True,
        )
        logger.info("DeepSORT tracker initialised.")

    def update(self, detections: List[Detection],
               frame: np.ndarray) -> List[Detection]:
        """
        Convert Detection list → DeepSort input format,
        run tracker, merge track IDs back.
        """
        if not detections:
            self._tracker.update_tracks([], frame=frame)
            return []

        # DeepSort expects: list of ( [left, top, w, h], confidence, class_name )
        ds_input = []
        for det in detections:
            x1, y1, x2, y2 = det.box
            w = x2 - x1
            h = y2 - y1
            ds_input.append(([x1, y1, w, h], det.conf, det.class_name))

        tracks = self._tracker.update_tracks(ds_input, frame=frame)

        # Build a map from bounding-box IoU match → track_id
        result: List[Detection] = []
        assigned = set()

        for track in tracks:
            if not track.is_confirmed():
                continue
            tlwh     = track.to_tlwh()
            track_id = track.track_id
            tx1 = tlwh[0]
            ty1 = tlwh[1]
            tx2 = tlwh[0] + tlwh[2]
            ty2 = tlwh[1] + tlwh[3]
            t_box = np.array([tx1, ty1, tx2, ty2])

            # Match to detection with highest IoU
            best_iou  = 0.0
            best_idx  = -1
            for idx, det in enumerate(detections):
                if idx in assigned:
                    continue
                overlap = _iou_boxes(det.box, t_box)
                if overlap > best_iou:
                    best_iou = overlap
                    best_idx = idx

            if best_idx >= 0 and best_iou > 0.2:
                det = detections[best_idx]
                det.track_id = track_id
                result.append(det)
                assigned.add(best_idx)

        # Detections not matched still get -1 track_id
        for idx, det in enumerate(detections):
            if idx not in assigned:
                result.append(det)

        return result


# ─────────────────────────────────────────────
# Simple IoU tracker (fallback — no GPU needed)
# ─────────────────────────────────────────────

class IoUTracker(BaseTracker):
    """
    Lightweight IoU-based tracker.
    Assigns consistent integer IDs by matching consecutive-frame bounding boxes
    via maximum IoU.  No appearance embedding — suitable for dense traffic scenes
    where appearance cues matter less.
    """

    def __init__(self,
                 iou_threshold: float = 0.35,
                 max_missing:   int   = 15):
        self.iou_threshold  = iou_threshold
        self.max_missing    = max_missing
        self._next_id       = 1
        # track_id → {box, class_name, missing_frames, last_seen_det}
        self._tracks: Dict[int, Dict] = {}

    # ── public ──────────────────────────────────────────────────────────

    def update(self, detections: List[Detection],
               frame: np.ndarray) -> List[Detection]:
        boxes     = [d.box for d in detections]
        n_det     = len(boxes)
        n_tracks  = len(self._tracks)

        if n_tracks == 0 or n_det == 0:
            # Initialise new tracks for all detections
            for det in detections:
                det.track_id = self._register(det)
            # Age out old tracks
            self._age_tracks(set())
            return detections

        track_ids  = list(self._tracks.keys())
        track_boxes = [self._tracks[t]["box"] for t in track_ids]

        # Build IoU cost matrix (tracks × detections)
        cost = np.zeros((n_tracks, n_det), dtype=np.float32)
        for i, tb in enumerate(track_boxes):
            for j, db in enumerate(boxes):
                cost[i, j] = _iou_boxes(tb, db)

        # Greedy matching (descending IoU)
        matched_tracks  = set()
        matched_dets    = set()
        pairs           = np.dstack(np.unravel_index(
                            np.argsort(-cost, axis=None), cost.shape))[0]

        for ti, di in pairs:
            if cost[ti, di] < self.iou_threshold:
                break
            if ti in matched_tracks or di in matched_dets:
                continue
            tid = track_ids[ti]
            detections[di].track_id = tid
            self._tracks[tid]["box"]           = boxes[di]
            self._tracks[tid]["missing_frames"] = 0
            matched_tracks.add(ti)
            matched_dets.add(di)

        # Register unmatched detections
        for di in range(n_det):
            if di not in matched_dets:
                detections[di].track_id = self._register(detections[di])

        # Age out unmatched tracks
        self._age_tracks({track_ids[ti] for ti in matched_tracks})

        return detections

    # ── internal ────────────────────────────────────────────────────────

    def _register(self, det: Detection) -> int:
        tid = self._next_id
        self._next_id += 1
        self._tracks[tid] = {
            "box":            det.box.copy(),
            "class_name":     det.class_name,
            "missing_frames": 0,
        }
        return tid

    def _age_tracks(self, matched_ids: set) -> None:
        dead = []
        for tid in self._tracks:
            if tid not in matched_ids:
                self._tracks[tid]["missing_frames"] += 1
                if self._tracks[tid]["missing_frames"] > self.max_missing:
                    dead.append(tid)
        for tid in dead:
            del self._tracks[tid]


# ─────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────

def build_tracker(method: str = "deepsort", **kwargs) -> BaseTracker:
    """
    Parameters
    ----------
    method : 'deepsort' | 'iou'
    **kwargs : passed to the tracker constructor
    """
    if method == "deepsort":
        try:
            return DeepSORTTracker(**kwargs)
        except ImportError:
            logger.warning("DeepSORT unavailable — falling back to IoU tracker.")
            return IoUTracker(**kwargs)
    elif method == "iou":
        return IoUTracker(**kwargs)
    else:
        raise ValueError(f"Unknown tracker method: {method!r}. Choose 'deepsort' or 'iou'.")


# ─────────────────────────────────────────────
# Trajectory store
# ─────────────────────────────────────────────

class TrajectoryStore:
    """
    Maintains a rolling history of (frame_no, (cx, cy)) per track_id.
    Used by violation rules that need motion information.
    """

    def __init__(self, maxlen: int = 60):
        self.maxlen  = maxlen
        self._store: Dict[int, List[Tuple[int, Tuple[float, float]]]] = defaultdict(list)

    def update(self, track_id: int, frame_no: int,
               center: Tuple[float, float]) -> None:
        buf = self._store[track_id]
        buf.append((frame_no, center))
        if len(buf) > self.maxlen:
            buf.pop(0)

    def get(self, track_id: int) -> List[Tuple[int, Tuple[float, float]]]:
        return self._store.get(track_id, [])

    def remove_track(self, track_id: int) -> None:
        if track_id in self._store:
            del self._store[track_id]

    def cleanup_old_tracks(self, current_frame: int, max_age: int = 60) -> None:
        dead_ids = []
        for tid, hist in self._store.items():
            if not hist or (current_frame - hist[-1][0]) > max_age:
                dead_ids.append(tid)
        for tid in dead_ids:
            del self._store[tid]

    def direction_vector(self, track_id: int) -> Optional[Tuple[float, float]]:
        """Returns (dx, dy) representing recent motion direction, or None."""
        hist = self.get(track_id)
        if len(hist) < 2:
            return None
        _, p0 = hist[0]
        _, pn = hist[-1]
        dx = pn[0] - p0[0]
        dy = pn[1] - p0[1]
        mag = (dx ** 2 + dy ** 2) ** 0.5
        if mag < 1e-6:
            return (0.0, 0.0)
        return (dx / mag, dy / mag)

    def is_stationary(self, track_id: int,
                      min_frames: int = 30,
                      pixel_threshold: float = 10.0) -> bool:
        """True if the object hasn't moved more than pixel_threshold pixels."""
        hist = self.get(track_id)
        if len(hist) < min_frames:
            return False
        positions = [p for _, p in hist[-min_frames:]]
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        span = max(max(xs) - min(xs), max(ys) - min(ys))
        return span < pixel_threshold


# ─────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────

def _iou_boxes(a: np.ndarray, b: np.ndarray) -> float:
    xa = max(a[0], b[0]);  ya = max(a[1], b[1])
    xb = min(a[2], b[2]);  yb = min(a[3], b[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    if inter == 0:
        return 0.0
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + ab - inter)
