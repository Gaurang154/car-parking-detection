import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

VIDEO_SOURCE = os.getenv("VIDEO_SOURCE", "assets/carPark.mp4")
IMG_PATH = BASE_DIR / os.getenv("IMG_PATH", "assets/carParkImg.png")
POSITIONS_PATH = BASE_DIR / os.getenv("POSITIONS_PATH", "data/positions.json")
HISTORY_DB_PATH = BASE_DIR / os.getenv("HISTORY_DB_PATH", "data/history.db")
CALIBRATION_PATH = BASE_DIR / os.getenv("CALIBRATION_PATH", "data/calibration.json")
GROUND_TRUTH_PATH = BASE_DIR / os.getenv("GROUND_TRUTH_PATH", "data/ground_truth.json")

SPACE_WIDTH = int(os.getenv("SPACE_WIDTH", "107"))
SPACE_HEIGHT = int(os.getenv("SPACE_HEIGHT", "48"))
PIXEL_THRESHOLD = int(os.getenv("PIXEL_THRESHOLD", "900"))
SMOOTHING_FRAMES = int(os.getenv("SMOOTHING_FRAMES", "3"))
HISTORY_INTERVAL_SEC = float(os.getenv("HISTORY_INTERVAL_SEC", "5"))

# ── Detection engine ──────────────────────────────────────────────────────────
# Modes: "classical" (adaptive-threshold pixel counting) or "yolo" (YOLOv8 ONNX)
DEFAULT_ENGINE = os.getenv("DEFAULT_ENGINE", "classical")

# ── YOLOv8 (OpenCV DNN) ───────────────────────────────────────────────────────
# The bundled footage is a near top-down aerial view. Off-the-shelf COCO YOLO
# detects 0 vehicles from this angle (domain shift), so we use a YOLOv8m model
# fine-tuned on VisDrone (aerial/drone imagery). Build it once with:
#     python scripts/build_model.py
# Runtime stays pure OpenCV DNN — no PyTorch needed to serve.
MODELS_DIR = BASE_DIR / "models"
YOLO_MODEL_PATH = Path(os.getenv("YOLO_MODEL_PATH", MODELS_DIR / "yolov8-visdrone.onnx"))
# Source .pt (exported to ONNX by scripts/build_model.py). No direct ONNX exists.
YOLO_PT_URL = os.getenv(
    "YOLO_PT_URL",
    "https://huggingface.co/mshamrai/yolov8m-visdrone/resolve/main/best.pt?download=true",
)
# Optional fallback: a COCO yolov8n ONNX (works on ground-level footage, not aerial).
YOLO_COCO_URL = os.getenv(
    "YOLO_COCO_URL",
    "https://github.com/CVHub520/X-AnyLabeling/releases/download/v0.1.0/yolov8n.onnx",
)
YOLO_MODEL_SHA1 = os.getenv("YOLO_MODEL_SHA1", "")  # exported models have no fixed hash
YOLO_INPUT_SIZE = int(os.getenv("YOLO_INPUT_SIZE", "640"))
YOLO_CONF_THRESHOLD = float(os.getenv("YOLO_CONF_THRESHOLD", "0.25"))
YOLO_IOU_THRESHOLD = float(os.getenv("YOLO_IOU_THRESHOLD", "0.45"))
# Fraction of a parking space a car box must cover for the space to be "occupied".
YOLO_OCCUPANCY_OVERLAP = float(os.getenv("YOLO_OCCUPANCY_OVERLAP", "0.30"))
# VisDrone vehicle classes: car=3, van=4, truck=5, bus=8
YOLO_VEHICLE_CLASSES = {3, 4, 5, 8}

BENCHMARK_PATH = BASE_DIR / os.getenv("BENCHMARK_PATH", "data/benchmark.json")

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8080"))
DEBUG = os.getenv("DEBUG", "false").lower() in ("1", "true", "yes")
