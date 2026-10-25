"""Binary verification metrics and deterministic EER operating point."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def analyze_scores(
    labels: np.ndarray,
    scores: np.ndarray,
    decision_threshold: float | None = None,
) -> dict:
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels.size == 0 or labels.shape != scores.shape:
        raise ValueError("Labels and scores must be non-empty arrays of equal length.")
    if not np.isin(labels, (0, 1)).all() or not np.isfinite(scores).all():
        raise ValueError("Labels must be binary and scores must be finite.")
    if np.unique(labels).size != 2:
        raise ValueError("Both genuine and impostor labels are required.")

    false_accept_rate, true_accept_rate, thresholds = roc_curve(labels, scores)
    false_reject_rate = 1.0 - true_accept_rate
    eer_index = int(np.argmin(np.abs(false_accept_rate - false_reject_rate)))
    eer_threshold = float(thresholds[eer_index])
    threshold = eer_threshold if decision_threshold is None else float(decision_threshold)
    if not np.isfinite(threshold):
        raise ValueError("Decision threshold must be finite.")
    predictions = (scores >= threshold).astype(np.int32)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    true_rejects, false_accepts = (int(value) for value in matrix[0])
    false_rejects, true_accepts = (int(value) for value in matrix[1])
    negatives = false_accepts + true_rejects
    positives = true_accepts + false_rejects

    roc_rows = [
        {
            "false_accept_rate": float(far),
            "true_accept_rate": float(tar),
            "false_reject_rate": float(frr),
            "threshold": float(cutoff),
        }
        for far, tar, frr, cutoff in zip(
            false_accept_rate, true_accept_rate, false_reject_rate, thresholds
        )
    ]
    return {
        "count": int(labels.size),
        "threshold_rule": (
            "ROC point minimizing absolute FAR-FRR difference"
            if decision_threshold is None
            else "frozen threshold selected on validation data"
        ),
        "threshold": threshold,
        "eer_threshold_estimate": eer_threshold,
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "far": float(false_accepts / negatives) if negatives else 0.0,
        "frr": float(false_rejects / positives) if positives else 0.0,
        "eer_estimate": float((false_accept_rate[eer_index] + false_reject_rate[eer_index]) / 2.0),
        "auc": float(roc_auc_score(labels, scores)),
        "confusion_matrix": {
            "labels": ["impostor", "genuine"],
            "matrix": matrix.astype(int).tolist(),
            "true_accepts": true_accepts,
            "false_accepts": false_accepts,
            "true_rejects": true_rejects,
            "false_rejects": false_rejects,
        },
        "predictions": predictions,
        "roc": roc_rows,
    }