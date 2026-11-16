"""Tile-level and farm-level metrics. Thresholds are chosen on validation, reported on test."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score

from exw.constants import CLASSES, EXW, N_CLASSES


def tile_metrics(y_true: np.ndarray, p: np.ndarray) -> dict:
    pred = p.argmax(1)
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, pred, labels=list(range(N_CLASSES)), zero_division=0)
    return dict(accuracy=float((pred == y_true).mean()), macro_f1=float(f1.mean()),
                per_class={c: dict(precision=float(pr[i]), recall=float(rc[i]), f1=float(f1[i]))
                           for i, c in enumerate(CLASSES)},
                confusion=confusion_matrix(y_true, pred, labels=list(range(N_CLASSES))).tolist())


def exw_tile_auc(y_true: np.ndarray, p: np.ndarray) -> float:
    y = (y_true == EXW).astype(int)
    return float(roc_auc_score(y, p[:, EXW])) if 0 < y.sum() < len(y) else float("nan")


def choose_threshold(risk: np.ndarray, y_exw: np.ndarray, min_sens: float = 0.9) -> float:
    """Highest threshold on risk R that keeps sensitivity >= min_sens on VALIDATION; else maximise Youden J."""
    cands = np.unique(np.concatenate([[0.0], risk]))
    best, best_j = None, -1.0
    for t in cands:
        flag = risk >= t if t > 0 else risk > 0
        sens = flag[y_exw == 1].mean() if (y_exw == 1).any() else 0.0
        spec = (~flag)[y_exw == 0].mean() if (y_exw == 0).any() else 0.0
        if sens >= min_sens and (best is None or t > best):
            best = t
        if sens + spec - 1 > best_j and best is None:
            best_j = sens + spec - 1
    if best is None:
        scores = [((risk >= t)[y_exw == 1].mean() + (~(risk >= t))[y_exw == 0].mean() - 1, t) for t in cands]
        best = max(scores)[1]
    return float(best)


def farm_metrics(risk: np.ndarray, y_exw: np.ndarray, thr: float) -> dict:
    flag = risk >= thr if thr > 0 else risk > 0
    tp = int((flag & (y_exw == 1)).sum()); fn = int((~flag & (y_exw == 1)).sum())
    tn = int((~flag & (y_exw == 0)).sum()); fp = int((flag & (y_exw == 0)).sum())
    return dict(threshold=float(thr), sensitivity=tp / max(tp + fn, 1), specificity=tn / max(tn + fp, 1),
                precision=tp / max(tp + fp, 1), tp=tp, fn=fn, tn=tn, fp=fp,
                auc=float(roc_auc_score(y_exw, risk)) if 0 < y_exw.sum() < len(y_exw) else float("nan"))
