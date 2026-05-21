from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "fn_count": int(fn),
        "fp_count": int(fp),
        "tp_count": int(tp),
        "tn_count": int(tn),
    }
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["brier"] = float(brier_score_loss(y_true, y_prob))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["brier"] = float("nan")
    return metrics


def pick_threshold(y_true: np.ndarray, y_prob: np.ndarray, method: str, recall_target: float = 0.85) -> float:
    if method == "fixed":
        return 0.5
    thresholds = np.linspace(0.01, 0.99, 99)
    if method == "recall_target":
        valid = []
        for t in thresholds:
            m = compute_metrics(y_true, y_prob, t)
            if m["recall"] >= recall_target:
                valid.append(t)
        return float(max(valid)) if valid else 0.5
    if method == "youden":
        from sklearn.metrics import roc_curve

        fpr, tpr, thr = roc_curve(y_true, y_prob)
        j = tpr - fpr
        idx = int(np.argmax(j))
        return float(thr[idx])
    return 0.5
