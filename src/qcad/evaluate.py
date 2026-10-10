"""Threshold setting, metric computation, and per-attack recall."""

from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

logger = logging.getLogger(__name__)

# Minimum samples in a per-attack bucket before we add a 'low-n' flag.
LOW_N_THRESHOLD = 20


def compute_threshold(benign_val_scores: np.ndarray, target_fpr: float) -> float:
    """Derive a detection threshold from benign validation scores.

    The threshold is the (1 - target_fpr) quantile of the validation scores,
    so that approximately *target_fpr* fraction of benign flows are flagged.

    Parameters
    ----------
    benign_val_scores:
        Anomaly scores of benign-only validation flows (higher = more anomalous).
    target_fpr:
        Desired false-positive rate on benign data (e.g. 0.01 or 0.05).

    Returns
    -------
    float threshold value.
    """
    return float(np.quantile(benign_val_scores, 1 - target_fpr))


def evaluate(
    test_scores: np.ndarray,
    y_true: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict:
    """Compute classification metrics at a fixed threshold.

    Parameters
    ----------
    test_scores:
        Anomaly scores for the test set.
    y_true:
        Binary ground-truth array (1 = attack, 0 = benign).
    labels:
        String label for each test row.
    threshold:
        Detection threshold (scores >= threshold are flagged as anomalous).

    Returns
    -------
    dict with keys: recall, precision, f1, measured_fpr, per_attack.
    """
    predicted = test_scores >= threshold
    attack_mask = y_true == 1
    benign_mask = y_true == 0

    recall = float(predicted[attack_mask].mean()) if attack_mask.any() else 0.0
    precision = (
        float(attack_mask[predicted].sum() / predicted.sum()) if predicted.sum() else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    measured_fpr = float(predicted[benign_mask].mean()) if benign_mask.any() else 0.0

    per_attack: dict[str, dict] = {}
    for attack in np.unique(labels[attack_mask]):
        mask = labels == attack
        count = int(mask.sum())
        per_attack[attack] = {
            "recall": float(predicted[mask].mean()),
            "n": count,
            "low_n": count < LOW_N_THRESHOLD,
        }

    return {
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "measured_fpr": measured_fpr,
        "per_attack": per_attack,
    }


def compute_auc_metrics(test_scores: np.ndarray, y_true: np.ndarray) -> dict:
    """Compute threshold-free PR-AUC and ROC-AUC.

    Parameters
    ----------
    test_scores:
        Anomaly scores for the test set.
    y_true:
        Binary ground-truth array (1 = attack, 0 = benign).

    Returns
    -------
    dict with keys: prauc, rocauc.
    """
    prauc = float(average_precision_score(y_true, test_scores))
    rocauc = float(roc_auc_score(y_true, test_scores))
    return {"prauc": prauc, "rocauc": rocauc}


def full_metrics(
    val_scores: np.ndarray,
    test_scores: np.ndarray,
    y_true: np.ndarray,
    labels: np.ndarray,
    target_fprs: list[float] | None = None,
) -> dict:
    """Run evaluation at every target FPR and compute AUC metrics.

    Parameters
    ----------
    val_scores:
        Benign-only validation scores used to derive thresholds.
    test_scores:
        Anomaly scores for the test set.
    y_true:
        Binary ground-truth (1 = attack, 0 = benign).
    labels:
        String label per test row.
    target_fprs:
        List of target FPR values; defaults to [0.01, 0.05].

    Returns
    -------
    dict with:
      - ``"by_fpr"``: dict keyed by fpr → evaluation result dict
      - ``"prauc"``, ``"rocauc"``: threshold-free metrics
    """
    if target_fprs is None:
        target_fprs = [0.01, 0.05]

    by_fpr: dict[float, dict] = {}
    for fpr in target_fprs:
        thr = compute_threshold(val_scores, fpr)
        result = evaluate(test_scores, y_true, labels, thr)
        result["threshold"] = thr
        result["target_fpr"] = fpr
        by_fpr[fpr] = result

    auc = compute_auc_metrics(test_scores, y_true)

    return {"by_fpr": by_fpr, **auc}
