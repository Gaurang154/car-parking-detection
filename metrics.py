"""Classification metrics for evaluating parking occupancy detection.

Convention: the positive class is "occupied".
    TP — predicted occupied, actually occupied
    FP — predicted occupied, actually free
    FN — predicted free,     actually occupied
    TN — predicted free,     actually free
"""

from dataclasses import dataclass
from typing import Sequence


@dataclass
class ConfusionMatrix:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.tp + self.tn) / self.total

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def as_dict(self) -> dict:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "total": self.total,
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


def confusion_matrix(
    predicted_occupied: Sequence[bool],
    actual_occupied: Sequence[bool],
) -> ConfusionMatrix:
    """Build a confusion matrix from aligned prediction/label sequences."""
    if len(predicted_occupied) != len(actual_occupied):
        raise ValueError(
            f"length mismatch: {len(predicted_occupied)} predictions "
            f"vs {len(actual_occupied)} labels"
        )

    cm = ConfusionMatrix()
    for pred, actual in zip(predicted_occupied, actual_occupied):
        if pred and actual:
            cm.tp += 1
        elif pred and not actual:
            cm.fp += 1
        elif not pred and actual:
            cm.fn += 1
        else:
            cm.tn += 1
    return cm


def format_report(cm: ConfusionMatrix) -> str:
    d = cm.as_dict()
    return (
        "Evaluation Report (positive class = occupied)\n"
        "---------------------------------------------\n"
        f"  Samples    : {d['total']}\n"
        f"  Accuracy   : {d['accuracy']:.1%}\n"
        f"  Precision  : {d['precision']:.1%}\n"
        f"  Recall     : {d['recall']:.1%}\n"
        f"  F1 score   : {d['f1']:.1%}\n"
        "  Confusion  :\n"
        f"      TP={d['tp']:<5} FP={d['fp']:<5}\n"
        f"      FN={d['fn']:<5} TN={d['tn']:<5}\n"
    )
