"""Model asset management for the YOLO engine.

The bundled footage is a near top-down aerial view, where off-the-shelf COCO
YOLO detects 0 vehicles (domain shift). We therefore serve a YOLOv8m model
fine-tuned on VisDrone (aerial imagery), exported to ONNX.

There is no public direct-download ONNX for this model, so it is built once
from the HuggingFace `.pt` weights via Ultralytics (a build-time dependency).
At runtime only OpenCV DNN is used — no PyTorch required.

YOLO is optional: if the model can't be built/loaded, the app still runs on the
classical engine and the YOLO toggle is disabled (graceful degradation).
"""

import logging
import urllib.request
from pathlib import Path

from config import YOLO_INPUT_SIZE, YOLO_MODEL_PATH, YOLO_PT_URL

logger = logging.getLogger(__name__)


def _build_from_pt(onnx_path: Path) -> bool:
    """Download the VisDrone .pt and export to ONNX. Needs `ultralytics`."""
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.warning(
            "YOLO model missing and `ultralytics` not installed. "
            "Run `pip install ultralytics && python scripts/build_model.py` "
            "to enable the YOLO engine. Falling back to classical CV."
        )
        return False

    try:
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        pt_path = onnx_path.with_suffix(".pt")
        if not pt_path.exists():
            logger.info("Downloading VisDrone weights → %s", pt_path)
            urllib.request.urlretrieve(YOLO_PT_URL, pt_path)

        logger.info("Exporting %s → ONNX (imgsz=%d, opset=12) ...", pt_path.name, YOLO_INPUT_SIZE)
        exported = YOLO(str(pt_path)).export(format="onnx", imgsz=YOLO_INPUT_SIZE, opset=12)
        Path(exported).replace(onnx_path)
        logger.info("YOLO model ready: %s (%.1f MB)", onnx_path, onnx_path.stat().st_size / 1e6)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("YOLO model build failed (%s). Falling back to classical CV.", exc)
        return False


def ensure_yolo_model(path: Path = YOLO_MODEL_PATH) -> bool:
    """Ensure the ONNX model exists locally. Returns True if available."""
    if path.exists():
        logger.info("YOLO model present: %s", path)
        return True
    return _build_from_pt(path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ok = ensure_yolo_model()
    print("YOLO model available" if ok else "YOLO model unavailable — classical only")
