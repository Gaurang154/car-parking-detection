import numpy as np
import pytest

from engines import (
    ClassicalEngine,
    SpaceResult,
    overlap_ratio,
    parse_yolov8_output,
    space_occupied_by_cars,
)


class TestOverlap:
    def test_full_overlap(self):
        assert overlap_ratio((0, 0, 100, 100), (0, 0, 100, 100)) == 1.0

    def test_no_overlap(self):
        assert overlap_ratio((0, 0, 50, 50), (100, 100, 150, 150)) == 0.0

    def test_half_overlap(self):
        # car covers exactly half of the space horizontally
        assert overlap_ratio((0, 0, 100, 100), (50, 0, 150, 100)) == pytest.approx(0.5)


class TestSpaceOccupancy:
    def test_occupied_when_car_covers_enough(self):
        space = (0, 0, 100, 100)
        cars = [(40, 0, 140, 100)]  # covers 60%
        occupied, ratio = space_occupied_by_cars(space, cars, threshold=0.30)
        assert occupied is True
        assert ratio == pytest.approx(0.6)

    def test_free_when_overlap_below_threshold(self):
        space = (0, 0, 100, 100)
        cars = [(90, 0, 190, 100)]  # covers 10%
        occupied, ratio = space_occupied_by_cars(space, cars, threshold=0.30)
        assert occupied is False

    def test_free_when_no_cars(self):
        occupied, ratio = space_occupied_by_cars((0, 0, 100, 100), [], threshold=0.30)
        assert occupied is False
        assert ratio == 0.0


class TestYoloParsing:
    def _make_output(self, detections, num_classes=80, num_anchors=100):
        """Build a synthetic YOLOv8 tensor (1, 4+num_classes, anchors),
        matching OpenCV's real (features, anchors) layout. Unused anchors stay
        zero (class score 0 -> filtered by the confidence threshold)."""
        out = np.zeros((1, 4 + num_classes, num_anchors), dtype=np.float32)
        for i, (cx, cy, w, h, cid, conf) in enumerate(detections):
            out[0, 0, i] = cx
            out[0, 1, i] = cy
            out[0, 2, i] = w
            out[0, 3, i] = h
            out[0, 4 + cid, i] = conf
        return out

    def test_filters_non_vehicles(self):
        # class 0 (person) should be dropped; class 2 (car) kept
        out = self._make_output([
            (320, 320, 100, 100, 0, 0.9),   # person
            (320, 320, 100, 100, 2, 0.9),   # car
        ])
        boxes, scores, ids = parse_yolov8_output(
            out, conf_threshold=0.25, vehicle_classes={2, 3, 5, 7},
            scale_x=1.0, scale_y=1.0,
        )
        assert len(boxes) == 1
        assert ids == [2]

    def test_filters_low_confidence(self):
        out = self._make_output([(320, 320, 100, 100, 2, 0.10)])
        boxes, _, _ = parse_yolov8_output(
            out, conf_threshold=0.25, vehicle_classes={2}, scale_x=1.0, scale_y=1.0
        )
        assert boxes == []

    def test_box_coordinates_and_scale(self):
        out = self._make_output([(320, 320, 100, 200, 2, 0.9)])
        boxes, scores, ids = parse_yolov8_output(
            out, conf_threshold=0.25, vehicle_classes={2}, scale_x=2.0, scale_y=2.0
        )
        # center (320,320) size (100,200) -> top-left (270,220) *2 -> (540,440)
        assert boxes[0] == [540, 440, 200, 400]
        assert scores[0] == pytest.approx(0.9)

    def test_handles_empty(self):
        boxes, scores, ids = parse_yolov8_output(
            np.zeros((1, 84, 0), dtype=np.float32),
            conf_threshold=0.25, vehicle_classes={2}, scale_x=1.0, scale_y=1.0,
        )
        assert boxes == [] and scores == [] and ids == []


class TestClassicalEngine:
    def test_detect_empty_and_full(self):
        engine = ClassicalEngine(threshold=900, width=107, height=48)
        # plain gray frame -> mostly empty after preprocess
        frame = np.full((200, 400, 3), 127, dtype=np.uint8)
        out = engine.detect(frame, [(0, 0), (150, 100)])
        assert len(out.results) == 2
        assert out.latency_ms >= 0
        assert all(isinstance(r, SpaceResult) for r in out.results)
