import cv2
import numpy as np
import easyocr
import re
import logging
from typing import Optional, Dict, Tuple, List

logger = logging.getLogger("tvd.ocr")


class PlateRecognizer:
    """
    License plate OCR using EasyOCR.
    
    Strategy:
      1. Extract plate region from vehicle bbox
      2. Find candidate plate areas (bright rectangular regions)
      3. Run OCR on each candidate
      4. Score and pick the best match
    """
    def __init__(self, use_gpu: bool = True):
        logger.info("Initializing EasyOCR for Plate Recognition...")
        self.reader = easyocr.Reader(['en'], gpu=use_gpu)
        self.cache: Dict[int, Optional[Tuple[str, float]]] = {}

    def _clean_text(self, text: str) -> str:
        return re.sub(r'[^A-Z0-9]', '', text.upper())

    def _is_valid_plate(self, text: str) -> bool:
        """Check if text matches an Indian plate pattern."""
        if not text or len(text) < 4:
            return False
        # Full Indian plate: State(2)+District(1-2)+Series(1-2)+Number(1-4)
        if re.match(r'^[A-Z]{2}\d{1,2}[A-Z]{1,2}\d{1,4}$', text):
            return True
        # Newer format
        if re.match(r'^[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}$', text):
            return True
        # Partial: State+District
        if re.match(r'^[A-Z]{2}\d{1,2}$', text) and len(text) >= 4:
            return True
        # Partial: State+District+Series
        if re.match(r'^[A-Z]{2}\d{1,2}[A-Z]{1,2}$', text):
            return True
        return False

    def _score_plate(self, text: str, conf: float) -> float:
        """Score how likely text is a real plate."""
        # Full Indian plate: State+District+Series+Number
        if re.match(r'^[A-Z]{2}\d{1,2}[A-Z]{1,2}\d{1,4}$', text):
            return 100 + conf * 50
        if re.match(r'^[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}$', text):
            return 95 + conf * 50
        # Series+Number (partial plate)
        if re.match(r'^[A-Z]{1,2}\d{3,6}$', text):
            return 80 + conf * 50
        # State+District+Series (partial)
        if re.match(r'^[A-Z]{2}\d{1,2}[A-Z]{1,2}$', text):
            return 75 + conf * 50
        # State+District (partial)
        if re.match(r'^[A-Z]{2}\d{1,2}$', text):
            return 60 + conf * 50
        # Numeric part only
        if re.match(r'^\d{3,6}$', text):
            return 40 + conf * 50
        return 0

    def _merge_plate_fragments(self, texts: List[Tuple[str, float, list]]) -> List[Tuple[str, float]]:
        """
        Merge OCR fragments into plate strings.
        Sort by Y then X (reading order), concatenate, and score.
        """
        if not texts:
            return []
        
        def get_y(pt):
            try:
                return float(pt[1])
            except:
                return 0
        
        def get_x(pt):
            try:
                return float(pt[0])
            except:
                return 0
        
        # Sort by Y (top to bottom), then X (left to right)
        sorted_texts = sorted(texts, key=lambda t: (get_y(t[2][0]) if t[2] else 0, get_x(t[2][0]) if t[2] else 0))
        
        # Filter: keep only alphanumeric fragments (plate-like)
        plate_frags = [(t, c, b) for t, c, b in sorted_texts if re.match(r'^[A-Z0-9]{2,10}$', t)]
        
        # Spatial filter: remove outliers far from the main cluster
        if len(plate_frags) > 2:
            y_positions = [get_y(f[2][0]) for f in plate_frags if f[2]]
            if y_positions:
                y_med = np.median(y_positions)
                y_mad = np.median([abs(y - y_med) for y in y_positions])
                # Keep fragments within 3x MAD of median (robust outlier detection)
                threshold = max(y_mad * 3, 150)  # at least 150px
                plate_frags = [f for f in plate_frags
                               if abs(get_y(f[2][0]) - y_med) < threshold if f[2]]
        
        if not plate_frags:
            return []
        
        results = []
        
        # Strategy 1: merge ALL plate-like fragments
        all_merged = ''.join(t for t, _, _ in plate_frags)
        avg_conf = np.mean([c for _, c, _ in plate_frags])
        score = self._score_plate(all_merged, avg_conf)
        if score > 0:
            results.append((all_merged, avg_conf))
        
        # Strategy 2: merge by Y-lines (within 100px)
        lines = [[plate_frags[0]]]
        for frag in plate_frags[1:]:
            prev_y = get_y(lines[-1][-1][2][0]) if lines[-1][-1][2] else 0
            curr_y = get_y(frag[2][0]) if frag[2] else 0
            if abs(curr_y - prev_y) < 100:
                lines[-1].append(frag)
            else:
                lines.append([frag])
        
        for line in lines:
            line.sort(key=lambda t: get_x(t[2][0]) if t[2] else 0)
            line_text = ''.join(t for t, _, _ in line)
            line_conf = np.mean([c for _, c, _ in line])
            score = self._score_plate(line_text, line_conf)
            if score > 0:
                results.append((line_text, line_conf))
        
        # Strategy 3: individual fragments
        for text, conf, _ in plate_frags:
            score = self._score_plate(text, conf)
            if score > 0:
                results.append((text, conf))
        
        # Deduplicate, keep best
        deduped = {}
        for text, conf in results:
            score = self._score_plate(text, conf)
            if text not in deduped or score > self._score_plate(deduped[text][0], deduped[text][1]):
                deduped[text] = (text, conf)
        
        return sorted(deduped.values(), key=lambda x: -self._score_plate(x[0], x[1]))

    def _find_plate_candidates(self, region: np.ndarray) -> List[np.ndarray]:
        """Find bright rectangular regions that could be plates."""
        if region.size == 0:
            return []
        
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
        
        # Threshold for bright regions (white/yellow plates)
        _, binary = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
        
        # Dilate to connect nearby components
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        dilated = cv2.dilate(binary, kernel, iterations=2)
        
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        rh, rw = region.shape[:2]
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Plate aspect ratio: typically 2:1 to 5:1
            aspect = w / h if h > 0 else 0
            if not (1.5 < aspect < 6.0):
                continue
            
            # Must be reasonable size (at least 2% of region width, not bigger than region)
            if w < rw * 0.05 or w > rw * 0.8:
                continue
            if h < 10 or h > rh * 0.5:
                continue
            
            # Extract with padding
            pad = 10
            y1 = max(0, y - pad)
            y2 = min(rh, y + h + pad)
            x1 = max(0, x - pad)
            x2 = min(rw, x + w + pad)
            
            candidates.append(region[y1:y2, x1:x2])
        
        return candidates

    def _ocr_region(self, region: np.ndarray) -> List[Tuple[str, float]]:
        """Run EasyOCR on a region, return (text, conf) pairs."""
        if region.size == 0:
            return []
        
        # Scale up if too small
        h, w = region.shape[:2]
        if w < 300:
            scale = 300 / w
            region = cv2.resize(region, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # Preprocess: CLAHE + sharpen
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(gray)
        
        # Run OCR with bboxes
        results = self.reader.readtext(enhanced, detail=1, paragraph=False, mag_ratio=2)
        
        texts = []
        for (txt_bbox, text, conf) in results:
            cleaned = self._clean_text(text)
            if cleaned and len(cleaned) >= 2 and conf > 0.15:
                texts.append((cleaned, conf, txt_bbox))
        
        return texts

    def process_vehicle(self, vehicle_id: int, frame: np.ndarray, bbox: list) -> Optional[Tuple[str, float]]:
        """
        Extract license plate from a vehicle.
        
        Returns (plate_text, confidence) or None.
        """
        if vehicle_id in self.cache and self.cache[vehicle_id] is not None:
            return self.cache[vehicle_id]

        x1, y1, x2, y2 = map(int, bbox)
        fh, fw = frame.shape[:2]
        box_h = y2 - y1
        box_w = x2 - x1
        
        # Determine plate location based on box shape
        is_head_only = (box_h > box_w * 1.5) and (box_h < fh * 0.4)
        
        if is_head_only:
            # Plate is below the head
            ny1 = max(0, y1)
            ny2 = min(fh, y2 + int(box_h * 2.5))
            nx1 = max(0, x1 - int(box_w * 0.3))
            nx2 = min(fw, x2 + int(box_w * 0.3))
        else:
            # Standard: take lower portion of box + some below
            ny1 = max(0, y1 + int(box_h * 0.3))
            ny2 = min(fh, y2 + int(box_h * 0.3))
            nx1 = max(0, x1 - int(box_w * 0.2))
            nx2 = min(fw, x2 + int(box_w * 0.2))
        
        if ny2 <= ny1 or nx2 <= nx1:
            return None
        
        region = frame[ny1:ny2, nx1:nx2]
        
        best_plate = None
        best_score = 0.0
        best_conf = 0.0
        
        # Strategy 1: OCR full region + merge fragments
        texts = self._ocr_region(region)
        merged = self._merge_plate_fragments(texts)
        for text, conf in merged:
            score = self._score_plate(text, conf)
            if score > best_score:
                best_plate = text
                best_score = score
                best_conf = conf
        
        # Strategy 2: Find plate contours + merge
        candidates = self._find_plate_candidates(region)
        for candidate in candidates:
            texts = self._ocr_region(candidate)
            merged = self._merge_plate_fragments(texts)
            for text, conf in merged:
                score = self._score_plate(text, conf)
                if score > best_score:
                    best_plate = text
                    best_score = score
                    best_conf = conf
        
        # Strategy 3: Raw image (no preprocessing) + merge
        if region.size > 0:
            h, w = region.shape[:2]
            if w < 300:
                scale = 300 / w
                raw = cv2.resize(region, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            else:
                raw = region
            gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY) if len(raw.shape) == 3 else raw
            results = self.reader.readtext(gray, detail=1, paragraph=False, mag_ratio=2)
            raw_texts = []
            for (txt_bbox, text, conf) in results:
                cleaned = self._clean_text(text)
                if cleaned and len(cleaned) >= 2 and conf > 0.15:
                    raw_texts.append((cleaned, conf, txt_bbox))
            merged = self._merge_plate_fragments(raw_texts)
            for text, conf in merged:
                score = self._score_plate(text, conf)
                if score > best_score:
                    best_plate = text
                    best_score = score
                    best_conf = conf
        
        if best_plate:
            self.cache[vehicle_id] = (best_plate, best_conf)
            logger.info("Vehicle %d -> Plate: %s (conf=%.2f, score=%.1f)", vehicle_id, best_plate, best_conf, best_score)
        
        return (best_plate, best_conf) if best_plate else None
