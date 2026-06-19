import numpy as np
import pytest

from detector import DetectionStats, ParkingDetector
from engines import SpaceResult
from storage import OccupancyHistory, PositionStore


def _detector(tmp_path, **kwargs):
    store = PositionStore(tmp_path / "positions.json")
    history = OccupancyHistory(tmp_path / "history.db")
    kwargs.setdefault("threshold", 900)
    kwargs.setdefault("enable_yolo", False)
    return ParkingDetector(store, history, **kwargs)


class TestSummarize:
  def test_summarize_counts(self, tmp_path):
    detector = _detector(tmp_path)
    results = [
      SpaceResult(0, is_free=True, count=10),
      SpaceResult(1, is_free=False, count=2000),
      SpaceResult(2, is_free=True, count=5),
    ]
    stats = detector.summarize(results, "classical", 1.5)
    assert stats.total == 3
    assert stats.free == 2
    assert stats.occupied == 1
    assert stats.pct_free == 66
    assert stats.engine == "classical"
    assert stats.latency_ms == 1.5


class TestSmoothing:
  def test_majority_vote_reduces_flicker(self, tmp_path):
    detector = _detector(tmp_path, smoothing_frames=3)
    # one space, fed free, then occupied twice
    def step(free):
      out = detector._smooth([SpaceResult(0, is_free=free)])
      return out[0].is_free
    assert step(True) is True
    assert step(False) is True    # 2/3 still free
    assert step(False) is False   # 1/3 now occupied


class TestEngineSelection:
  def test_classical_always_available(self, tmp_path):
    detector = _detector(tmp_path)
    assert "classical" in detector.available_engines
    assert detector.mode == "classical"

  def test_set_unknown_mode_rejected(self, tmp_path):
    detector = _detector(tmp_path)
    assert detector.set_mode("nonexistent") is False
    assert detector.mode == "classical"


class TestStatsSerialization:
  def test_as_dict_has_engine_and_latency(self, tmp_path):
    s = DetectionStats(free=5, occupied=3, total=8, pct_free=62,
                       engine="yolo", latency_ms=42.0)
    d = s.as_dict()
    assert d["engine"] == "yolo"
    assert d["latency_ms"] == 42.0
    assert d["free"] == 5


class TestPositionStore:
  def test_add_and_list(self, tmp_path):
    path = tmp_path / "positions.json"
    store = PositionStore(path)
    store.add(10, 20)
    store.add(30, 40)
    assert store.list() == [(10, 20), (30, 40)]
    assert store.count() == 2

  def test_remove_at(self, tmp_path):
    path = tmp_path / "positions.json"
    store = PositionStore(path)
    store.add(0, 0)
    removed = store.remove_at(50, 24, 107, 48)
    assert removed is True
    assert store.count() == 0


class TestOccupancyHistory:
  def test_record_and_recent(self, tmp_path):
    db = tmp_path / "history.db"
    history = OccupancyHistory(db)
    history.record(10, 5, 15, 66.7)
    history.record(8, 7, 15, 53.3)
    snaps = history.recent()
    assert len(snaps) == 2
    assert snaps[0]["free"] == 10
    assert snaps[1]["free"] == 8

  def test_summary(self, tmp_path):
    db = tmp_path / "history.db"
    history = OccupancyHistory(db)
    history.record(10, 5, 15, 66.7)
    history.record(20, 10, 30, 66.7)
    summary = history.summary()
    assert summary["samples"] == 2
    assert summary["avg_pct_free"] == 66.7
