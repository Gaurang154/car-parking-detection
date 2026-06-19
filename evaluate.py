"""Evaluate / benchmark detector engines against hand-labeled ground truth.

Ground truth (data/ground_truth.json):

    {
      "video": "carPark.mp4",
      "frames": {
        "0":   [true, false, true, ...],   # occupied? per space, positions order
        "300": [false, false, true, ...]
      }
    }

Workflow:
    1. python evaluate.py --template            # pre-fill labels from predictions
    2. hand-correct data/ground_truth.json      # flip any wrong booleans
    3. python evaluate.py                        # score the active engine
    4. python evaluate.py --benchmark            # score & time EVERY engine

Pre-labeling then correcting is standard practice — low manual effort, honest
metrics. Accuracy is measured offline against labels; latency is measured here
per engine so the Analytics page can show the accuracy-vs-speed trade-off.
"""

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List

import cv2

from config import BENCHMARK_PATH, GROUND_TRUTH_PATH, VIDEO_SOURCE
from detector import ParkingDetector
from metrics import ConfusionMatrix, confusion_matrix, format_report
from storage import OccupancyHistory, PositionStore

logger = logging.getLogger(__name__)


def _build_detector() -> ParkingDetector:
    return ParkingDetector(PositionStore(), OccupancyHistory())


def _read_frame(cap, frame_index):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    return frame if ok else None


def _predict(engine, frame, positions) -> List[bool]:
    out = engine.detect(frame, positions)
    return [not r.is_free for r in out.results]


def make_template(frame_indices: List[int]) -> dict:
    detector = _build_detector()
    positions = detector.positions.list()
    engine = detector.engines["classical"]

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {VIDEO_SOURCE}")

    frames: Dict[str, List[bool]] = {}
    for idx in frame_indices:
        frame = _read_frame(cap, idx)
        if frame is None:
            logger.warning("frame %d unavailable, skipping", idx)
            continue
        frames[str(idx)] = _predict(engine, frame, positions)
    cap.release()

    template = {"video": VIDEO_SOURCE, "frames": frames}
    GROUND_TRUTH_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GROUND_TRUTH_PATH, "w") as f:
        json.dump(template, f, indent=2)
    logger.info("Wrote label template (%d frames) to %s", len(frames), GROUND_TRUTH_PATH)
    return template


def _load_ground_truth() -> dict:
    if not GROUND_TRUTH_PATH.exists():
        raise FileNotFoundError(
            f"{GROUND_TRUTH_PATH} not found — run `python evaluate.py --template` first"
        )
    with open(GROUND_TRUTH_PATH) as f:
        return json.load(f)


def evaluate_engine(detector: ParkingDetector, engine_name: str, gt: dict) -> dict:
    engine = detector.engines[engine_name]
    positions = detector.positions.list()

    cap = cv2.VideoCapture(gt.get("video", VIDEO_SOURCE))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {gt.get('video', VIDEO_SOURCE)}")

    all_pred: List[bool] = []
    all_true: List[bool] = []
    latencies: List[float] = []

    for frame_key, labels in gt["frames"].items():
        frame = _read_frame(cap, int(frame_key))
        if frame is None:
            continue
        t0 = time.perf_counter()
        out = engine.detect(frame, positions)
        latencies.append((time.perf_counter() - t0) * 1000)
        pred = [not r.is_free for r in out.results]
        n = min(len(pred), len(labels))
        all_pred.extend(pred[:n])
        all_true.extend(bool(v) for v in labels[:n])
    cap.release()

    cm = confusion_matrix(all_pred, all_true)
    metrics = cm.as_dict()
    metrics["avg_latency_ms"] = round(sum(latencies) / len(latencies), 2) if latencies else None
    metrics["available"] = True
    return metrics


def evaluate(engine_name: str = None) -> dict:
    detector = _build_detector()
    gt = _load_ground_truth()
    name = engine_name or detector.mode
    metrics = evaluate_engine(detector, name, gt)
    cm = ConfusionMatrix(metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"])
    print(f"\nEngine: {name}")
    print(format_report(cm))
    if metrics["avg_latency_ms"] is not None:
        print(f"  Avg latency: {metrics['avg_latency_ms']} ms\n")
    return metrics


def benchmark() -> dict:
    detector = _build_detector()
    gt = _load_ground_truth()

    engines = {}
    for name in detector.available_engines:
        logger.info("Benchmarking engine: %s", name)
        engines[name] = evaluate_engine(detector, name, gt)

    result = {"engines": engines, "evaluated_at": datetime.now(timezone.utc).isoformat()}
    BENCHMARK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BENCHMARK_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print("\nBenchmark — accuracy vs. latency")
    print("-" * 64)
    print(f"{'engine':<12}{'accuracy':>10}{'precision':>11}{'recall':>9}{'f1':>8}{'latency':>12}")
    for name, m in engines.items():
        lat = f"{m['avg_latency_ms']} ms" if m["avg_latency_ms"] is not None else "n/a"
        print(f"{name:<12}{m['accuracy']:>10.1%}{m['precision']:>11.1%}"
              f"{m['recall']:>9.1%}{m['f1']:>8.1%}{lat:>12}")
    print(f"\nSaved to {BENCHMARK_PATH}")
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Evaluate / benchmark parking detector")
    parser.add_argument("--template", action="store_true", help="write a label template")
    parser.add_argument("--benchmark", action="store_true", help="evaluate all engines + latency")
    parser.add_argument("--engine", type=str, default=None, help="evaluate a specific engine")
    parser.add_argument("--frames", type=int, nargs="+", default=[0, 150, 300, 450, 600])
    args = parser.parse_args()

    if args.template:
        make_template(args.frames)
        print(f"Template at {GROUND_TRUTH_PATH} — correct the booleans, then re-run.")
    elif args.benchmark:
        benchmark()
    else:
        evaluate(args.engine)
