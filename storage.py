import json
import logging
import pickle
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

from config import (
    HISTORY_DB_PATH,
    HISTORY_INTERVAL_SEC,
    LEGACY_POSITIONS_PATH,
    POSITIONS_PATH,
)

logger = logging.getLogger(__name__)

Position = Tuple[int, int]


class PositionStore:
    """Thread-safe parking space coordinate storage (JSON)."""

    def __init__(self, path: Path = POSITIONS_PATH):
        self._path = path
        self._lock = threading.Lock()
        self._positions: List[Position] = []
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            with open(self._path, "r") as f:
                raw = json.load(f)
            self._positions = [(int(p["x"]), int(p["y"])) for p in raw["spaces"]]
            logger.info("Loaded %d spaces from %s", len(self._positions), self._path)
            return

        if self._path == POSITIONS_PATH and LEGACY_POSITIONS_PATH.exists():
            with open(LEGACY_POSITIONS_PATH, "rb") as f:
                legacy = pickle.load(f)
            self._positions = [(int(x), int(y)) for x, y in legacy]
            self._save()
            logger.info(
                "Migrated %d spaces from legacy pickle to %s",
                len(self._positions),
                self._path,
            )
            return

        self._positions = []
        logger.warning("No parking positions found — use /picker to mark spaces")

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"spaces": [{"x": x, "y": y} for x, y in self._positions]}
        with open(self._path, "w") as f:
            json.dump(payload, f, indent=2)

    def list(self) -> List[Position]:
        with self._lock:
            return list(self._positions)

    def add(self, x: int, y: int) -> int:
        with self._lock:
            self._positions.append((x, y))
            self._save()
            return len(self._positions)

    def remove_at(self, cx: int, cy: int, width: int, height: int) -> bool:
        with self._lock:
            for i, (x, y) in enumerate(self._positions):
                if x < cx < x + width and y < cy < y + height:
                    self._positions.pop(i)
                    self._save()
                    return True
            return False

    def clear(self) -> None:
        with self._lock:
            self._positions.clear()
            self._save()

    def count(self) -> int:
        with self._lock:
            return len(self._positions)


class OccupancyHistory:
    """SQLite-backed occupancy snapshots for analytics."""

    def __init__(self, path: Path = HISTORY_DB_PATH):
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._interval = HISTORY_INTERVAL_SEC
        self._last_snapshot: float = 0
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS occupancy_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    free INTEGER NOT NULL,
                    occupied INTEGER NOT NULL,
                    total INTEGER NOT NULL,
                    pct_free REAL NOT NULL
                )
                """
            )
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def maybe_record(
        self,
        free: int,
        occupied: int,
        total: int,
        pct_free: float,
        now: float,
    ) -> None:
        if now - self._last_snapshot < self._interval:
            return
        self._last_snapshot = now
        self.record(free, occupied, total, pct_free)

    def record(
        self,
        free: int,
        occupied: int,
        total: int,
        pct_free: float,
    ) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO occupancy_snapshots
                    (timestamp, free, occupied, total, pct_free)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (ts, free, occupied, total, pct_free),
                )
                conn.commit()

    def recent(self, limit: int = 120) -> list[dict]:
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT timestamp, free, occupied, total, pct_free
                    FROM occupancy_snapshots
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def summary(self) -> dict:
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    """
                    SELECT
                        COUNT(*) AS samples,
                        AVG(pct_free) AS avg_pct_free,
                        MIN(pct_free) AS min_pct_free,
                        MAX(pct_free) AS max_pct_free,
                        AVG(occupied) AS avg_occupied
                    FROM occupancy_snapshots
                    """
                ).fetchone()
        if not row or row["samples"] == 0:
            return {
                "samples": 0,
                "avg_pct_free": 0,
                "min_pct_free": 0,
                "max_pct_free": 0,
                "avg_occupied": 0,
            }
        return {
            "samples": row["samples"],
            "avg_pct_free": round(row["avg_pct_free"], 1),
            "min_pct_free": round(row["min_pct_free"], 1),
            "max_pct_free": round(row["max_pct_free"], 1),
            "avg_occupied": round(row["avg_occupied"], 1),
        }
