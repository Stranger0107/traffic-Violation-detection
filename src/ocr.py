import cv2
import numpy as np
import easyocr
import re
import logging
from typing import Optional, Dict

logger = logging.getLogger("tvd.ocr")

class PlateRecognizer:
    """
    Modular component to detect and read license plates from vehicle crops.
    Utilizes EasyOCR for text extraction and basic heuristics for validation.
    """
    def __init__(self, use_gpu: bool = True):
        logger.info("Initializing EasyOCR for Plate Recognition...")
        # Initialize EasyOCR reader for English
        # gpu=True will use CUDA if available, else falls back to CPU
        self.reader = easyocr.Reader(['en'], gpu=use_gpu)
        self.cache: Dict[int, Optional[str]] = {}

    def _clean_text(self, text: str) -> str:
        """Removes noise, spaces, and special characters."""
        return re.sub(r'[^A-Z0-9]', '', text.upper())

    def _is_valid_plate(self, text: str) -> bool:
        """
        Validates if the string roughly matches a license plate pattern.
        - Length between 4 and 10
        - Contains at least one letter and one number
        """
        if not (4 <= len(text) <= 10):
            return False
        if not any(c.isalpha() for c in text):
            return False
        if not any(c.isdigit() for c in text):
            return False
        return True

    def process_vehicle(self, vehicle_id: int, frame: np.ndarray, bbox: list, crop_bottom: bool = True) -> Optional[str]:
        """
        Crops the vehicle from the frame, focuses on the bottom section,
        and runs OCR to extract the license plate.
        
        Args:
            vehicle_id: Tracking ID of the vehicle.
            frame: Full original video frame (BGR).
            bbox: Bounding box of the vehicle [x1, y1, x2, y2].
            crop_bottom: Whether to crop the bottom half of the bounding box.
            
        Returns:
            Extracted plate string, or None if not found.
        """
        # Return cached plate if we have already successfully read it
        if vehicle_id in self.cache and self.cache[vehicle_id] is not None:
            return self.cache[vehicle_id]

        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]
        
        # Clip bounding box to frame dimensions
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
            
        vehicle_crop = frame[y1:y2, x1:x2]
        vh, vw = vehicle_crop.shape[:2]
        
        # Heuristic: License plates are typically in the bottom half of the vehicle
        # Cropping the bottom half speeds up OCR and reduces false positives (e.g. text on shirts/trucks)
        if crop_bottom and vh > 40:
            crop_y_start = int(vh * 0.4)  # Start from 40% down to be safe
            plate_region = vehicle_crop[crop_y_start:vh, 0:vw]
        else:
            plate_region = vehicle_crop

        # Optional: Convert to grayscale or apply contrast to help OCR
        gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
        
        # Read text
        # detail=1 returns [(bbox, text, prob), ...]
        results = self.reader.readtext(gray, detail=1)
        
        best_plate = None
        best_conf = 0.0
        
        for (txt_bbox, text, conf) in results:
            cleaned = self._clean_text(text)
            if self._is_valid_plate(cleaned) and conf > best_conf:
                best_plate = cleaned
                best_conf = conf

        # Only cache if we found a valid plate to avoid missing it if it's clearer in a later frame
        if best_plate:
            self.cache[vehicle_id] = best_plate
            logger.info(f"Vehicle {vehicle_id} -> Plate Detected: {best_plate} (Conf: {best_conf:.2f})")

        return best_plate
