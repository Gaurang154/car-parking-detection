import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import cvzone
import numpy as np

from calibrate import load_threshold
from config import (
    DEFAULT_ENGINE,
    PIXEL_THRESHOLD,
    SMOOTHING_FRAMES,
    SPACE_HEIGHT,
    SPACE_WIDTH,
    VIDEO_SOURCE,
    YOLO_CONF_THRESHOLD,
    YOLO_INPUT_SIZE,
    YOLO_IOU_THRESHOLD,
    YOLO_MODEL_PATH,
    YOLO_OCCUPANCY_OVERLAP,
    YOLO_VEHICLE_CLASSES,
)
from engines import ClassicalEngine, EngineOutput, SpaceResult, YoloEngine
from models import ensure_yolo_model
from storage import OccupancyHistory, PositionStore

logger = logging.getLogger(__name__)

Position = Tuple[int, int]


@dataclass
class DetectionStats:
    free: int = 0
    occupied: int = 0
    total: int = 0
    pct_free: int = 0
    pct_occupied: int = 0
    fps: float = 0.0
    frame_count: int = 0
    engine: str = "classical"
    latency_ms: float = 0.0

    def as_dict(self) -> dict:
        return {
            "free": self.free,
            "occupied": self.occupied,
            "total": self.total,
            "pct_free": self.pct_free,
            "pct_occupied": self.pct_occupied,
            "fps": self.fps,
            "frame": self.frame_count,
            "engine": self.engine,
            "latency_ms": round(self.latency_ms, 2),
        }


class ParkingDetector:
    """Dual-engine parking occupancy detector with live engine switching.

    Detection is delegated to interchangeable engines (classical / YOLO).
    Temporal majority-vote smoothing, FPS/latency tracking, SQLite history and
    an SSE publish mechanism live here, independent of which engine runs.
    """

    def __init__(
        self,
        position_store: PositionStore,
        history: OccupancyHistory,
        video_source: str = VIDEO_SOURCE,
        width: int = SPACE_WIDTH,
        height: int = SPACE_HEIGHT,
        threshold: Optional[int] = None,
        smoothing_frames: int = SMOOTHING_FRAMES,
        enable_yolo: bool = True,
    ):
        self.positions = position_store
        self.history = history
        self.video_source = video_source
        self.width = width
        self.height = height
        self.smoothing_frames = max(1, smoothing_frames)

        resolved_threshold = threshold if threshold is not None else load_threshold(PIXEL_THRESHOLD)
        logger.info("Classical threshold = %s", resolved_threshold)

        self.engines: Dict[str, object] = {
            "classical": ClassicalEngine(resolved_threshold, width, height),
        }

        self.yolo = YoloEngine(
            YOLO_MODEL_PATH, width, height,
            input_size=YOLO_INPUT_SIZE,
            conf_threshold=YOLO_CONF_THRESHOLD,
            iou_threshold=YOLO_IOU_THRESHOLD,
            occupancy_overlap=YOLO_OCCUPANCY_OVERLAP,
            vehicle_classes=YOLO_VEHICLE_CLASSES,
        )
        if enable_yolo and ensure_yolo_model() and self.yolo.load():
            self.engines["yolo"] = self.yolo

        self.mode = DEFAULT_ENGINE if DEFAULT_ENGINE in self.engines else "classical"

        self._state_lock = threading.Lock()
        self._current_frame: Optional[bytes] = None
        self._raw_frame: Optional[np.ndarray] = None
        self._stats = DetectionStats(engine=self.mode)
        self._space_history: List[List[bool]] = []
        self._latency: Dict[str, Deque[float]] = {
            "classical": deque(maxlen=60),
            "yolo": deque(maxlen=60),
        }

        # SSE pub/sub
        self._update = threading.Condition()
        self._version = 0

        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ── Engine management ─────────────────────────────────────────────────────
    @property
    def available_engines(self) -> List[str]:
        return list(self.engines.keys())

    @property
    def yolo_available(self) -> bool:
        return "yolo" in self.engines

    def set_mode(self, mode: str) -> bool:
        if mode not in self.engines:
            return False
        with self._state_lock:
            self.mode = mode
            self._space_history.clear()
        logger.info("Engine switched to %s", mode)
        return True

    # ── State access ──────────────────────────────────────────────────────────
    @property
    def stats(self) -> DetectionStats:
        with self._state_lock:
            s = self._stats
            return DetectionStats(**vars(s))

    def get_frame(self) -> Optional[bytes]:
        with self._state_lock:
            return self._current_frame

    def get_raw_frame(self) -> Optional[np.ndarray]:
        with self._state_lock:
            return None if self._raw_frame is None else self._raw_frame.copy()

    def latency_summary(self) -> Dict[str, dict]:
        with self._state_lock:
            out = {}
            for name, samples in self._latency.items():
                if samples:
                    out[name] = {
                        "avg_ms": round(sum(samples) / len(samples), 2),
                        "min_ms": round(min(samples), 2),
                        "max_ms": round(max(samples), 2),
                        "samples": len(samples),
                        "available": name in self.engines,
                    }
                else:
                    out[name] = {"avg_ms": None, "available": name in self.engines}
            return out

    # ── SSE pub/sub ────────────────────────────────────────────────────────────
    def wait_for_update(self, last_version: int, timeout: float = 5.0) -> Tuple[int, dict]:
        with self._update:
            if self._version == last_version:
                self._update.wait(timeout)
            return self._version, self.stats.as_dict()

    def _publish(self) -> None:
        with self._update:
            self._version += 1
            self._update.notify_all()

    # ── Smoothing ──────────────────────────────────────────────────────────────
    def _smooth(self, results: List[SpaceResult]) -> List[SpaceResult]:
        while len(self._space_history) < len(results):
            self._space_history.append([])
        for r in results:
            window = self._space_history[r.index]
            window.append(r.is_free)
            if len(window) > self.smoothing_frames:
                window.pop(0)
            r.is_free = sum(window) >= len(window) / 2
        return results

    def clear_smoothing(self) -> None:
        with self._state_lock:
            self._space_history.clear()

    @staticmethod
    def summarize(results: List[SpaceResult], engine: str, latency_ms: float) -> DetectionStats:
        total = len(results)
        free = sum(1 for r in results if r.is_free)
        occupied = total - free
        pct_free = int((free / total) * 100) if total else 0
        return DetectionStats(
            free=free, occupied=occupied, total=total,
            pct_free=pct_free, pct_occupied=100 - pct_free,
            engine=engine, latency_ms=latency_ms,
        )

    # ── Rendering ──────────────────────────────────────────────────────────────
    def render(
        self,
        frame: np.ndarray,
        positions: List[Position],
        output: EngineOutput,
        stats: DetectionStats,
    ) -> np.ndarray:
        annotated = frame.copy()

        for box in output.car_boxes:
            cv2.rectangle(annotated, (box.x1, box.y1), (box.x2, box.y2), (255, 200, 0), 2)
            cvzone.putTextRect(
                annotated, f"car {box.conf:.2f}", (box.x1, max(0, box.y1 - 4)),
                scale=0.8, thickness=1, offset=3, colorR=(255, 200, 0), colorT=(0, 0, 0),
            )

        for (x, y), r in zip(positions, output.results):
            color, thickness = ((0, 255, 0), 5) if r.is_free else ((0, 0, 255), 2)
            cv2.rectangle(annotated, (x, y), (x + self.width, y + self.height), color, thickness)
            label = str(r.count) if stats.engine == "classical" else f"{r.score:.2f}"
            cvzone.putTextRect(annotated, label, (x, y + self.height - 3),
                               scale=1, thickness=2, offset=0, colorR=color)

        cvzone.putTextRect(annotated, f"Free: {stats.free}/{stats.total}", (100, 50),
                           scale=3, thickness=5, offset=20, colorR=(0, 200, 0))
        cvzone.putTextRect(
            annotated,
            f"{stats.engine.upper()}  |  {stats.latency_ms:.1f} ms  |  Avail: {stats.pct_free}%",
            (100, 110), scale=1.5, thickness=2, offset=10, colorR=(0, 150, 255),
        )
        return annotated

    # ── Inference ──────────────────────────────────────────────────────────────
    def infer(
        self,
        frame: np.ndarray,
        positions: Optional[List[Position]] = None,
        mode: Optional[str] = None,
        smooth: bool = True,
    ) -> Tuple[np.ndarray, DetectionStats, EngineOutput]:
        if positions is None:
            positions = self.positions.list()
        mode = mode or self.mode
        engine = self.engines.get(mode, self.engines["classical"])

        output = engine.detect(frame, positions)
        if smooth:
            output.results = self._smooth(output.results)
        stats = self.summarize(output.results, engine.name, output.latency_ms)
        annotated = self.render(frame, positions, output, stats)
        return annotated, stats, output

    def process_single_frame(self, frame, positions=None, smooth=True):
        annotated, stats, _ = self.infer(frame, positions, smooth=smooth)
        return annotated, stats

    # ── Video loop ───────────────────────────────────────────────────────────────
    def _video_loop(self) -> None:
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            logger.error("Cannot open video source: %s", self.video_source)
            return

        logger.info("Video loop started — source: %s", self.video_source)
        frame_times: Deque[float] = deque(maxlen=30)

        while self._running:
            if cap.get(cv2.CAP_PROP_POS_FRAMES) >= cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.clear_smoothing()

            success, frame = cap.read()
            if not success:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            positions = self.positions.list()
            annotated, stats, _ = self.infer(frame, positions)

            now = time.monotonic()
            frame_times.append(now)
            if len(frame_times) > 1:
                stats.fps = round(len(frame_times) / (frame_times[-1] - frame_times[0]), 1)
            stats.frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            self.history.maybe_record(stats.free, stats.occupied, stats.total, stats.pct_free, now)

            _, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
            with self._state_lock:
                self._current_frame = jpeg.tobytes()
                self._raw_frame = frame
                self._stats = stats
                self._latency[stats.engine].append(stats.latency_ms)
            self._publish()

            time.sleep(0.033)

        cap.release()
        logger.info("Video loop stopped")

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._video_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
