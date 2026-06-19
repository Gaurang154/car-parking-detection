import pytest

from metrics import ConfusionMatrix, confusion_matrix


class TestConfusionMatrix:
    def test_perfect_prediction(self):
        pred = [True, False, True, False]
        true = [True, False, True, False]
        cm = confusion_matrix(pred, true)
        assert cm.tp == 2
        assert cm.tn == 2
        assert cm.fp == 0
        assert cm.fn == 0
        assert cm.accuracy == 1.0
        assert cm.precision == 1.0
        assert cm.recall == 1.0
        assert cm.f1 == 1.0

    def test_all_wrong(self):
        pred = [True, True]
        true = [False, False]
        cm = confusion_matrix(pred, true)
        assert cm.fp == 2
        assert cm.accuracy == 0.0
        assert cm.precision == 0.0

    def test_mixed_case(self):
        # 1 TP, 1 FP, 1 FN, 1 TN
        pred = [True, True, False, False]
        true = [True, False, True, False]
        cm = confusion_matrix(pred, true)
        assert (cm.tp, cm.fp, cm.fn, cm.tn) == (1, 1, 1, 1)
        assert cm.accuracy == 0.5
        assert cm.precision == 0.5
        assert cm.recall == 0.5
        assert cm.f1 == pytest.approx(0.5)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            confusion_matrix([True], [True, False])

    def test_empty_is_safe(self):
        cm = ConfusionMatrix()
        assert cm.accuracy == 0.0
        assert cm.precision == 0.0
        assert cm.recall == 0.0
        assert cm.f1 == 0.0

    def test_as_dict_keys(self):
        cm = confusion_matrix([True, False], [True, False])
        d = cm.as_dict()
        for key in ("tp", "fp", "fn", "tn", "accuracy", "precision", "recall", "f1"):
            assert key in d


class TestOtsu:
    def test_separates_bimodal_distribution(self):
        from calibrate import otsu_threshold

        # Two clear clusters: empty spaces (~50) and occupied (~2000)
        empty = [40, 45, 50, 55, 60] * 10
        occupied = [1800, 1900, 2000, 2100, 2200] * 10
        threshold = otsu_threshold(empty + occupied)
        assert 60 < threshold < 1800

    def test_constant_distribution(self):
        from calibrate import otsu_threshold

        assert otsu_threshold([100, 100, 100]) == 100

    def test_empty_raises(self):
        from calibrate import otsu_threshold

        with pytest.raises(ValueError):
            otsu_threshold([])
