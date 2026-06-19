"""Build the YOLO ONNX model used by the deep-learning engine.

Downloads the VisDrone-fine-tuned YOLOv8m weights and exports them to ONNX
(opset 12) for OpenCV DNN. Run once after install:

    pip install ultralytics
    python scripts/build_model.py

Runtime serving does NOT need ultralytics/torch — only OpenCV.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging

from models import ensure_yolo_model

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if ensure_yolo_model():
        print("YOLO (VisDrone) model built and ready.")
    else:
        print("Could not build the YOLO model — the app will run classical-only.")
        sys.exit(1)
