# src/database.py
import mysql.connector
from mysql.connector import pooling

class TrafficDB:
    def __init__(self, host="localhost", user="root", password="YOUR_PASSWORD", database="traffic_system"):
        self.pool = pooling.MySQLConnectionPool(
            pool_name="tvd_pool",
            pool_size=5,
            host=host,
            user=user,
            password=password,
            database=database
        )

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
        finally:
            cur.close()
            conn.close()

    # ─────────────────────────────────────────────
    # LOG VIOLATION (INSERT)
    # ─────────────────────────────────────────────
    def log_violation(self, frame_no: int, violation_type: str, plate_number: str):
        conn = self._get_conn()
        cur = conn.cursor()
        try:
            cur.execute("""
                INSERT INTO Violations (frame_no, violation_type, plate_number)
                VALUES (%s, %s, %s)
            """, (frame_no, violation_type, plate_number))
            conn.commit()
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
        finally:
            cur.close()
            conn.close()