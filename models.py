"""Model asset management — downloads the YOLOv8n ONNX weights on demand.

YOLO is an *optional* engine. If the model can't be downloaded (offline, etc.)
the app still runs on the classical engine; the YOLO toggle is simply disabled.
This graceful degradation is intentional.
"""

import hashlib
import logging
import urllib.request
from pathlib import Path

from config import MODELS_DIR, YOLO_MODEL_PATH, YOLO_MODEL_SHA1, YOLO_MODEL_URL

logger = logging.getLogger(__name__)


def _sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path: Path = YOLO_MODEL_PATH, expected_sha1: str = YOLO_MODEL_SHA1) -> bool:
    if not path.exists():
        return False
    if not expected_sha1:
        return True
    actual = _sha1(path)
    if actual != expected_sha1:
        logger.warning("SHA1 mismatch for %s (got %s)", path.name, actual)
        return False
    return True


def ensure_yolo_model(
    path: Path = YOLO_MODEL_PATH,
    url: str = YOLO_MODEL_URL,
    expected_sha1: str = YOLO_MODEL_SHA1,
) -> bool:
    """Ensure the ONNX model exists locally. Returns True if available."""
    if verify(path, expected_sha1):
        logger.info("YOLO model present: %s", path)
        return True

    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading YOLO model from %s ...", url)
    try:
        tmp = path.with_suffix(".onnx.part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(path)
    except Exception as exc:  # noqa: BLE001 — any failure => fall back to classical
        logger.error("YOLO model download failed (%s). Falling back to classical CV.", exc)
        return False

    if not verify(path, expected_sha1):
        logger.error("Downloaded model failed verification; YOLO disabled.")
        return False

    logger.info("YOLO model ready: %s (%.1f MB)", path, path.stat().st_size / 1e6)
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ok = ensure_yolo_model()
    print("YOLO model available" if ok else "YOLO model unavailable — classical only")
