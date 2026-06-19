"""Data-driven threshold calibration.

The occupied/free decision is "non-zero pixels in a space crop vs a threshold".
Instead of hardcoding a magic number, we sample the *distribution* of pixel
counts across many frames and spaces. Empty and occupied spaces form two
clusters; Otsu's method finds the threshold that best separates them.

Run:  python calibrate.py
Output: data/calibration.json  (loaded automatically by the detector)
"""

import json
import logging
from datetime import datetime, timezone
from typing import List, Sequence

import cv2
import numpy as np

from config import CALIBRATION_PATH, SPACE_HEIGHT, SPACE_WIDTH, VIDEO_SOURCE
from storage import PositionStore

logger = logging.getLogger(__name__)


def otsu_threshold(values: Sequence[float]) -> float:
    """Otsu's method on a 1-D distribution of values.

    Returns the threshold that maximizes between-class variance, i.e. the
    split that best separates the two clusters (empty vs occupied counts).
    """
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        raise ValueError("cannot calibrate on an empty sample")
    if np.all(arr == arr[0]):
        return float(arr[0])

    hist, bin_edges = np.histogram(arr, bins=256)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    weights = hist.astype(np.float64)
    total = weights.sum()

    w_cumulative = np.cumsum(weights)
    mean_cumulative = np.cumsum(weights * bin_centers)
    global_mean = mean_cumulative[-1] / total

    best_threshold = bin_centers[0]
    best_variance = -1.0
    for i in range(len(bin_centers) - 1):
        w0 = w_cumulative[i]
        w1 = total - w0
        if w0 == 0 or w1 == 0:
            continue
        mean0 = mean_cumulative[i] / w0
        mean1 = (mean_cumulative[-1] - mean_cumulative[i]) / w1
        between = w0 * w1 * (mean0 - mean1) ** 2
        if between > best_variance:
            best_variance = between
            best_threshold = bin_centers[i]

    return float(best_threshold)


def _preprocess(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 1)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16
    )
    median = cv2.medianBlur(thresh, 5)
    kernel = np.ones((3, 3), np.uint8)
    return cv2.dilate(median, kernel, iterations=1)


def collect_counts(
    video_source: str = VIDEO_SOURCE,
    sample_every: int = 10,
    max_frames: int = 200,
) -> List[int]:
    """Collect per-space non-zero pixel counts across sampled frames."""
    store = PositionStore()
    positions = store.list()
    if not positions:
        raise RuntimeError("no parking spaces configured — mark spaces first")

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video source: {video_source}")

    counts: List[int] = []
    frame_idx = 0
    sampled = 0
    while sampled < max_frames:
        success, frame = cap.read()
        if not success:
            break
        if frame_idx % sample_every == 0:
            processed = _preprocess(frame)
            for x, y in positions:
                crop = processed[y:y + SPACE_HEIGHT, x:x + SPACE_WIDTH]
                if crop.size:
                    counts.append(int(cv2.countNonZero(crop)))
            sampled += 1
        frame_idx += 1

    cap.release()
    logger.info("Collected %d count samples from %d frames", len(counts), sampled)
    return counts


def calibrate(video_source: str = VIDEO_SOURCE) -> dict:
    counts = collect_counts(video_source)
    threshold = otsu_threshold(counts)
    result = {
        "threshold": round(threshold, 1),
        "samples": len(counts),
        "min": int(np.min(counts)),
        "max": int(np.max(counts)),
        "mean": round(float(np.mean(counts)), 1),
        "median": round(float(np.median(counts)), 1),
        "method": "otsu",
        "calibrated_at": datetime.now(timezone.utc).isoformat(),
    }
    CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CALIBRATION_PATH, "w") as f:
        json.dump(result, f, indent=2)
    return result


def load_threshold(default: int) -> int:
    """Return the calibrated threshold if present, else the default."""
    if CALIBRATION_PATH.exists():
        with open(CALIBRATION_PATH) as f:
            data = json.load(f)
        return int(round(data["threshold"]))
    return default


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = calibrate()
    print(json.dumps(result, indent=2))
    print(f"\nCalibrated threshold = {result['threshold']} (was a hardcoded 900)")
    print(f"Saved to {CALIBRATION_PATH}")
