# src/database.py
import json
import os
import urllib.error
import urllib.request

import mysql.connector
from mysql.connector import pooling

class TrafficDB:
    def __init__(self, host="localhost", user="root", password="0107@Bbs", database="traffic_system_db"):
        self.pool = pooling.MySQLConnectionPool(
            pool_name="tvd_pool",
            pool_size=5,
            host=host,
            user=user,
            password=password,
            database="traffic_system_db"
        )
        self.backend_url = os.getenv("TVD_BACKEND_URL", "http://localhost:8000")
        self.backend_api_key = os.getenv("TVD_BACKEND_API_KEY", "local-dev-key")

    def _get_conn(self):
        return self.pool.get_connection()

    # ─────────────────────────────────────────────
    # OWNER LOOKUP (JOIN)
    # ─────────────────────────────────────────────
    def get_owner_details(self, plate_number: str):
        conn = self._get_conn()
        cur = conn.cursor(dictionary=True)
        try:
            cur.execute("""
                SELECT c.name, c.phone_number, v.model, v.color
                FROM Vehicles v
                JOIN Citizens c ON v.owner_id = c.owner_id
                WHERE v.plate_number = %s
            """, (plate_number,))
            return cur.fetchone()  # dict or None
        except mysql.connector.Error:
            return None
        finally:
            cur.close()
            conn.close()

    # ─────────────────────────────────────────────
    # LOG VIOLATION (INSERT)
    # ─────────────────────────────────────────────
    def log_violation(self, frame_no: int, violation_type: str, plate_number: str):
        payload = {
            "frame_no": frame_no,
            "violation_type": violation_type,
            "plate_number": plate_number,
            "evidence_path": None,
        }

        try:
            request = urllib.request.Request(
                f"{self.backend_url.rstrip('/')}/ml/violations",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "X-API-Key": self.backend_api_key,
                },
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=15) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
            pass

        conn = self._get_conn()
        cur = conn.cursor()
        try:
            cur.execute("""
                INSERT INTO Violations (frame_no, violation_type, plate_number)
                VALUES (%s, %s, %s)
            """, (frame_no, violation_type, plate_number))
            conn.commit()
            return {
                "message": "Violation logged to local MySQL fallback",
                "frame_no": frame_no,
                "violation_type": violation_type,
                "plate_number": plate_number,
            }
        finally:
            cur.close()
            conn.close()

    # ─────────────────────────────────────────────
    # OPTIONAL: CHECK VEHICLE EXISTS
    # ─────────────────────────────────────────────
    def vehicle_exists(self, plate_number: str) -> bool:
        conn = self._get_conn()
        cur = conn.cursor()
        try:
            cur.execute("SELECT 1 FROM Vehicles WHERE plate_number=%s LIMIT 1", (plate_number,))
            return cur.fetchone() is not None
        except mysql.connector.Error:
            return False
        finally:
            cur.close()
            conn.close()