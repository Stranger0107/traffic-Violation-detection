"""Client for submitting detected violations to the e-challan backend."""

import json
import os
import uuid
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger("tvd.model_api")


class ModelViolationClient:
    """Send evidence and model metadata using the backend's ingestion contract."""

    _VIOLATION_TYPES = {
        "No Helmet": "NO_HELMET",
        "Red Light Violation": "RED_LIGHT_JUMP",
        "Wrong Side Driving": "WRONG_SIDE",
        "Illegal Parking": "ILLEGAL_PARKING",
        "Lane Violation": "LANE_VIOLATION",
    }

    def __init__(self):
        self.base_url = os.getenv("TVD_BACKEND_URL", "http://127.0.0.1:5000").rstrip("/")
        self.api_key = os.getenv("MODEL_API_KEY")
        self.camera_id = os.getenv("TVD_CAMERA_ID", "CAM-001")
        self.area_code = os.getenv("TVD_AREA_CODE", "AREA-01")
        self.location_text = os.getenv("TVD_LOCATION_TEXT")
        self.model_version = os.getenv("TVD_MODEL_VERSION", "traffic-v1.0.0")

    def submit(
        self,
        *,
        frame_number: int,
        violation_type: str,
        detected_plate: str,
        evidence_path: str,
        video_timestamp_sec: Optional[float] = None,
        ocr_confidence: float = 0.0,
    ):
        backend_type = self._VIOLATION_TYPES.get(violation_type, violation_type)
        # Check against valid backend enums to ensure safety
        valid_enums = {"NO_HELMET", "RED_LIGHT_JUMP", "WRONG_SIDE", "ILLEGAL_PARKING", "LANE_VIOLATION", "MORE_THAN_2_PEOPLE_ON_BIKE", "SPEEDING", "NO_SEATBELT"}
        if backend_type not in valid_enums:
            logger.warning(f"Skipping unsupported violation type: {violation_type}")
            return {
                "success": False,
                "message": f"Unsupported backend violation type: {violation_type}",
            }
        
        if not self.api_key:
            logger.error("MODEL_API_KEY is not configured")
            return {"success": False, "message": "MODEL_API_KEY is not configured"}

        image_path = Path(evidence_path)
        if not image_path.is_file():
            logger.error(f"Evidence image not found: {image_path}")
            return {"success": False, "message": f"Evidence image not found: {image_path}"}

        # Policy: Auto-verify if OCR confidence > 0.8
        recommendation = "AUTO_VERIFY" if ocr_confidence > 0.8 else "OFFICER_REVIEW"

        metadata = {
            "modelEventId": str(uuid.uuid4()),
            "violationType": backend_type,
            "detectedPlate": detected_plate,
            "ocrConfidence": max(0.0, min(float(ocr_confidence), 1.0)),
            "recommendation": recommendation,
            "frameNumber": frame_number,
            "videoTimestampSec": video_timestamp_sec,
            "detectedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "cameraId": self.camera_id,
            "areaCode": self.area_code,
            "modelVersion": self.model_version,
            "duplicateFlag": False,
            "duplicateConfidence": 0.0,
        }
        if self.location_text:
            metadata["locationText"] = self.location_text

        try:
            with image_path.open("rb") as image:
                logger.info(f"Submitting {backend_type} event to {self.base_url}/api/v1/model/violations")
                response = requests.post(
                    f"{self.base_url}/api/v1/model/violations",
                    headers={"x-model-api-key": self.api_key},
                    files={"image": (image_path.name, image, "image/jpeg")},
                    data={"metadata": json.dumps(metadata)},
                    timeout=15,
                )
            
            # Check for error status
            if not response.ok:
                logger.error(f"Backend API error (HTTP {response.status_code}): {response.text}")
                response.raise_for_status()

            res_json = response.json()
            logger.info(f"Submission successful: {res_json}")
            return res_json
        
        except requests.RequestException as error:
            logger.error(f"Failed to submit violation: {error}")
            return {"success": False, "message": str(error)}
