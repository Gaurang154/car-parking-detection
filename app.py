import json
import logging
import time

from flask import Flask, Response, jsonify, render_template, request, send_file

from config import BENCHMARK_PATH, DEBUG, HOST, IMG_PATH, PORT, SPACE_HEIGHT, SPACE_WIDTH
from detector import ParkingDetector
from storage import OccupancyHistory, PositionStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

position_store = PositionStore()
history = OccupancyHistory()
detector = ParkingDetector(position_store, history)
detector.start()

app = Flask(__name__)


# ── Pages ──────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", total=position_store.count())


@app.route("/analytics")
def analytics():
    return render_template("analytics.html")


@app.route("/picker")
def picker():
    return render_template("picker.html")


# ── Health ───────────────────────────────────────────────────────────────────
@app.route("/health")
def health():
    s = detector.stats
    return jsonify(
        {
            "status": "ok",
            "engine": detector.mode,
            "available_engines": detector.available_engines,
            "yolo_available": detector.yolo_available,
            "spaces_configured": s.total,
            "fps": s.fps,
        }
    )


# ── Live data ──────────────────────────────────────────────────────────────────
@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            frame = detector.get_frame()
            if frame is None:
                time.sleep(0.05)
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            time.sleep(0.033)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/stats")
def get_stats():
    """Polling fallback for clients without SSE."""
    return jsonify(detector.stats.as_dict())


@app.route("/events")
def events():
    """Server-Sent Events: pushes stats the instant a frame is processed."""
    def stream():
        last_version = 0
        while True:
            last_version, payload = detector.wait_for_update(last_version, timeout=5.0)
            yield f"data: {json.dumps(payload)}\n\n"

    return Response(
        stream(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Engine control ─────────────────────────────────────────────────────────────
@app.route("/api/engine", methods=["GET", "POST"])
def engine_control():
    if request.method == "POST":
        data = request.get_json(force=True)
        mode = data.get("mode", "")
        if detector.set_mode(mode):
            return jsonify({"ok": True, "engine": detector.mode})
        return jsonify({"ok": False, "error": f"engine '{mode}' unavailable"}), 400

    return jsonify(
        {
            "engine": detector.mode,
            "available": detector.available_engines,
            "yolo_available": detector.yolo_available,
            "latency": detector.latency_summary(),
        }
    )


@app.route("/api/benchmark")
def api_benchmark():
    """Live side-by-side latency for every engine on the current frame,
    merged with offline accuracy metrics from data/benchmark.json (if present)."""
    frame = detector.get_raw_frame()
    positions = position_store.list()

    engines = {}
    for name, engine in detector.engines.items():
        entry = {"available": True}
        if frame is not None and positions:
            t0 = time.perf_counter()
            engine.detect(frame, positions)
            entry["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        engines[name] = entry

    if BENCHMARK_PATH.exists():
        with open(BENCHMARK_PATH) as f:
            offline = json.load(f)
        for name, metrics in offline.get("engines", {}).items():
            engines.setdefault(name, {"available": name in detector.engines})
            engines[name].update(metrics)
        evaluated_at = offline.get("evaluated_at")
    else:
        evaluated_at = None

    return jsonify({"engines": engines, "evaluated_at": evaluated_at})


@app.route("/api/history")
def api_history():
    limit = request.args.get("limit", 120, type=int)
    return jsonify(
        {"snapshots": history.recent(limit=min(limit, 500)), "summary": history.summary()}
    )


# ── Picker ───────────────────────────────────────────────────────────────────
@app.route("/picker/image")
def picker_image():
    return send_file(IMG_PATH, mimetype="image/png")


@app.route("/picker/spaces")
def picker_spaces():
    return jsonify(
        {"spaces": position_store.list(), "width": SPACE_WIDTH, "height": SPACE_HEIGHT}
    )


@app.route("/picker/add", methods=["POST"])
def picker_add():
    data = request.get_json(force=True)
    count = position_store.add(int(data["x"]), int(data["y"]))
    detector.clear_smoothing()
    return jsonify({"count": count})


@app.route("/picker/remove", methods=["POST"])
def picker_remove():
    data = request.get_json(force=True)
    position_store.remove_at(int(data["x"]), int(data["y"]), SPACE_WIDTH, SPACE_HEIGHT)
    detector.clear_smoothing()
    return jsonify({"count": position_store.count()})


@app.route("/picker/clear", methods=["POST"])
def picker_clear():
    position_store.clear()
    detector.clear_smoothing()
    return jsonify({"count": 0})


if __name__ == "__main__":
    logger.info("PARKX dashboard → http://localhost:%s", PORT)
    logger.info("Analytics       → http://localhost:%s/analytics", PORT)
    logger.info("Space picker    → http://localhost:%s/picker", PORT)
    app.run(host=HOST, port=PORT, debug=DEBUG, threaded=True)
