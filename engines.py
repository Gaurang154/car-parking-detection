"""Detection engines.

Two interchangeable engines decide, for each parking space, free vs occupied:

  ClassicalEngine — adaptive-threshold pixel counting (fast, ~ms, light).
  YoloEngine      — YOLOv8n vehicle detection via OpenCV DNN (accurate, heavier).

Both return the same `EngineOutput`, so the detector, renderer, evaluator and
benchmark treat them identically. The non-trivial YOLO post-processing and the
car↔space matching are written as pure functions so they can be unit-tested
without a model or a GPU.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

Box = Tuple[int, int, int, int]  # x1, y1, x2, y2


@dataclass
class SpaceResult:
    index: int
    is_free: bool
    count: int = 0       # classical: non-zero pixel count
    score: float = 0.0   # yolo: best car-overlap ratio for this space


@dataclass
class CarBox:
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float
    class_id: int


@dataclass
class EngineOutput:
    results: List[SpaceResult]
    latency_ms: float
    car_boxes: List[CarBox] = field(default_factory=list)


# ── Pure geometry / parsing helpers (unit-tested) ─────────────────────────────
def overlap_ratio(space: Box, car: Box) -> float:
    """Intersection area / parking-space area. 0 when disjoint."""
    sx1, sy1, sx2, sy2 = space
    cx1, cy1, cx2, cy2 = car
    ix1, iy1 = max(sx1, cx1), max(sy1, cy1)
    ix2, iy2 = min(sx2, cx2), min(sy2, cy2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    space_area = max(1, (sx2 - sx1) * (sy2 - sy1))
    return inter / space_area


def space_occupied_by_cars(
    space: Box, cars: Sequence[Box], threshold: float
) -> Tuple[bool, float]:
    """A space is occupied if any car box covers >= `threshold` of it."""
    best = 0.0
    for car in cars:
        best = max(best, overlap_ratio(space, car))
    return best >= threshold, best


def parse_yolov8_output(
    output: np.ndarray,
    conf_threshold: float,
    vehicle_classes,
    scale_x: float,
    scale_y: float,
) -> Tuple[List[List[int]], List[float], List[int]]:
    """Decode a raw YOLOv8 detection tensor.

    Accepts shape (1, 84, N), (84, N) or (N, 84). Returns boxes as [x, y, w, h]
    in *original-image* pixels (ready for cv2.dnn.NMSBoxes), plus scores and
    class ids — filtered to vehicle classes above the confidence threshold.
    """
    out = np.squeeze(output)
    if out.ndim != 2 or out.size == 0:
        return [], [], []
    # OpenCV emits (features, anchors) e.g. (84, 8400); transpose to rows=anchors.
    if out.shape[0] < out.shape[1]:
        out = out.T

    boxes: List[List[int]] = []
    scores: List[float] = []
    class_ids: List[int] = []

    class_scores = out[:, 4:]
    cids = np.argmax(class_scores, axis=1)
    confs = class_scores[np.arange(len(cids)), cids]

    for i in range(out.shape[0]):
        conf = float(confs[i])
        cid = int(cids[i])
        if conf < conf_threshold or cid not in vehicle_classes:
            continue
        cx, cy, w, h = out[i, 0], out[i, 1], out[i, 2], out[i, 3]
        x = int((cx - w / 2) * scale_x)
        y = int((cy - h / 2) * scale_y)
        boxes.append([x, y, int(w * scale_x), int(h * scale_y)])
        scores.append(conf)
        class_ids.append(cid)

    return boxes, scores, class_ids


# ── Engines ───────────────────────────────────────────────────────────────────
class ClassicalEngine:
    name = "classical"

    def __init__(self, threshold: int, width: int, height: int):
        self.threshold = threshold
        self.width = width
        self.height = height

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 1)
        thresh = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16
        )
        median = cv2.medianBlur(thresh, 5)
        kernel = np.ones((3, 3), np.uint8)
        return cv2.dilate(median, kernel, iterations=1)

    def detect(self, frame: np.ndarray, positions: List[Tuple[int, int]]) -> EngineOutput:
        start = time.perf_counter()
        processed = self.preprocess(frame)
        results: List[SpaceResult] = []
        for i, (x, y) in enumerate(positions):
            crop = processed[y:y + self.height, x:x + self.width]
            count = int(cv2.countNonZero(crop)) if crop.size else 0
            results.append(SpaceResult(index=i, is_free=count < self.threshold, count=count))
        latency = (time.perf_counter() - start) * 1000
        return EngineOutput(results=results, latency_ms=latency)


class YoloEngine:
    name = "yolo"

    def __init__(
        self,
        model_path,
        width: int,
        height: int,
        input_size: int = 640,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        occupancy_overlap: float = 0.15,
        vehicle_classes=frozenset({2, 3, 5, 7}),
    ):
        self.model_path = str(model_path)
        self.width = width
        self.height = height
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.occupancy_overlap = occupancy_overlap
        self.vehicle_classes = vehicle_classes
        self._net: Optional["cv2.dnn_Net"] = None
        self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def load(self) -> bool:
        try:
            self._net = cv2.dnn.readNetFromONNX(self.model_path)
            self._available = True
            logger.info("YOLO engine loaded: %s", self.model_path)
        except Exception as exc:  # noqa: BLE001
            logger.error("YOLO engine load failed: %s", exc)
            self._available = False
        return self._available

    def _infer(self, frame: np.ndarray) -> List[CarBox]:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255.0, (self.input_size, self.input_size),
            swapRB=True, crop=False,
        )
        self._net.setInput(blob)
        output = self._net.forward()

        scale_x = w / self.input_size
        scale_y = h / self.input_size
        boxes, scores, class_ids = parse_yolov8_output(
            output, self.conf_threshold, self.vehicle_classes, scale_x, scale_y
        )
        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold, self.iou_threshold)
        cars: List[CarBox] = []
        for idx in np.array(indices).flatten():
            x, y, bw, bh = boxes[idx]
            cars.append(CarBox(x, y, x + bw, y + bh, float(scores[idx]), int(class_ids[idx])))
        return cars

    def detect(self, frame: np.ndarray, positions: List[Tuple[int, int]]) -> EngineOutput:
        if not self._available:
            raise RuntimeError("YOLO engine not available")
        start = time.perf_counter()
        cars = self._infer(frame)
        car_boxes = [(c.x1, c.y1, c.x2, c.y2) for c in cars]
        results: List[SpaceResult] = []
        for i, (x, y) in enumerate(positions):
            space = (x, y, x + self.width, y + self.height)
            occupied, ratio = space_occupied_by_cars(space, car_boxes, self.occupancy_overlap)
            results.append(SpaceResult(index=i, is_free=not occupied, count=0, score=round(ratio, 3)))
        latency = (time.perf_counter() - start) * 1000
        return EngineOutput(results=results, latency_ms=latency, car_boxes=cars)
