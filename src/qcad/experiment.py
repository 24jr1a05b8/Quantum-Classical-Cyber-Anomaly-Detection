"""Full experiment runner: grid search, evaluation, result CSVs + charts.

CLI usage
---------
    python -m qcad.experiment --quick   # fast check (~minutes)
    python -m qcad.experiment --full    # full run (seeds 0,1,2; k=4,6,8)
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import average_precision_score

from .config import (
    ENTANGLEMENT,
    FULL_KS,
    FULL_N_TRAIN,
    FULL_N_VAL,
    FULL_PER_TEST,
    FULL_PER_TUNE,
    FULL_SEEDS,
    GAMMA_GRID,
    QUICK_KS,
    QUICK_N_TRAIN,
    QUICK_N_VAL,
    QUICK_PER_TEST,
    QUICK_PER_TUNE,
    QUICK_SEEDS,
    RESULTS_DIR,
    TARGET_FPRS,
)
from .data import build_splits, load_all_data
from .evaluate import compute_auc_metrics, compute_threshold, evaluate
from .models import ClassicalDetector, QuantumDetector
from .preprocess import Preprocessor

logger = logging.getLogger(__name__)


def _warn_edge(name: str, gamma: float, grid: list[float]) -> None:
    if gamma in (grid[0], grid[-1]):
        logger.warning(
            "%s: best gamma=%.4f is at a grid edge %s — consider extending the grid.",
            name, gamma, grid,
        )


def run_experiment(
    *,
    quick: bool = False,
    run_id: str | None = None,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Execute the full grid-search + evaluation experiment.

    Parameters
    ----------
    quick:
        Use a smaller dataset for rapid verification.
    run_id:
        Subdirectory name under ``results/``.  Defaults to ``"v2"`` or ``"v2_quick"``.
    out_dir:
        Override the output directory entirely.

    Returns
    -------
    dict with ``"rows"``, ``"per_rows"``, ``"out_dir"`` keys.
    """
    if run_id is None:
        run_id = "v2_quick" if quick else "v2"

    out = out_dir or (RESULTS_DIR / run_id)
    out.mkdir(parents=True, exist_ok=True)

    seeds     = QUICK_SEEDS    if quick else FULL_SEEDS
    ks        = QUICK_KS       if quick else FULL_KS
    n_train   = QUICK_N_TRAIN  if quick else FULL_N_TRAIN
    n_val     = QUICK_N_VAL    if quick else FULL_N_VAL
    per_test  = QUICK_PER_TEST if quick else FULL_PER_TEST
    per_tune  = QUICK_PER_TUNE if quick else FULL_PER_TUNE

    logger.info("Loading data …")
    mon, pool = load_all_data()

    _, _, test_df, tune_df = build_splits(
        mon, pool,
        n_train=n_train, n_val=n_val,
        per_test=per_test, per_tune=per_tune,
        seed=0,
    )

    y_test  = (test_df["Label"] != "BENIGN").astype(int).to_numpy()
    labels  = test_df["Label"].to_numpy()
    y_tune  = (tune_df["Label"] != "BENIGN").astype(int).to_numpy()

    logger.info(
        "test: %d rows (%d attacks) | tune: %d rows (%d attacks)",
        len(test_df), int(y_test.sum()), len(tune_df), int(y_tune.sum()),
    )
    logger.info("Attack types: %s", dict(pd.Series(labels[y_test == 1]).value_counts()))

    rows: list[dict] = []
    per_rows: list[dict] = []

    for seed in seeds:
        train_df, val_df, _, _ = build_splits(
            mon, pool,
            n_train=n_train, n_val=n_val,
            per_test=per_test, per_tune=per_tune,
            seed=seed,
        )

        for k in ks:
            prep = Preprocessor(k=k).fit(train_df)
            X_tr   = prep.transform(train_df)
            X_val  = prep.transform(val_df)
            X_te   = prep.transform(test_df)
            X_tune = prep.transform(tune_df)

            for model_name in ("classical", "quantum"):
                t0 = time.time()
                grid = GAMMA_GRID[model_name]

                best_ap, best_gamma, _best_scores_tu = -1.0, None, None

                for gamma in grid:
                    if model_name == "classical":
                        det = ClassicalDetector(gamma=gamma)
                    else:
                        det = QuantumDetector(scale=gamma, k=prep.k)

                    det.fit(X_tr)
                    sc_tune = det.score(X_tune)
                    ap = float(average_precision_score(y_tune, sc_tune))

                    if ap > best_ap:
                        best_ap    = ap
                        best_gamma = gamma

                # refit with best gamma
                if model_name == "classical":
                    best_det = ClassicalDetector(gamma=best_gamma)
                else:
                    best_det = QuantumDetector(scale=best_gamma, k=prep.k)
                best_det.fit(X_tr)

                _warn_edge(f"seed={seed} k={k} {model_name}", best_gamma, grid)

                val_sc  = best_det.score(X_val)
                test_sc = best_det.score(X_te)

                auc = compute_auc_metrics(test_sc, y_test)
                r1 = evaluate(test_sc, y_test, labels, compute_threshold(val_sc, 0.01))
                r5 = evaluate(test_sc, y_test, labels, compute_threshold(val_sc, 0.05))

                rows.append({
                    "seed":      seed,
                    "k":         prep.k,
                    "model":     model_name,
                    "gamma":     best_gamma,
                    "Recall_1":  r1["recall"],
                    "Test_FPR_1": r1["measured_fpr"],
                    "Recall_5":  r5["recall"],
                    "Test_FPR_5": r5["measured_fpr"],
                    "F1_5":      r5["f1"],
                    "PRAUC":     auc["prauc"],
                    "ROCAUC":    auc["rocauc"],
                })

                for attack, info in r5["per_attack"].items():
                    per_rows.append({
                        "seed":      seed,
                        "k":         prep.k,
                        "model":     model_name,
                        "attack":    attack,
                        "recall_5":  info["recall"],
                        "n":         info["n"],
                        "low_n":     info["low_n"],
                    })

                logger.info(
                    "seed=%d k=%d %-9s gamma=%.3f  recall@1%%=%.3f fpr@1%%=%.3f  "
                    "recall@5%%=%.3f fpr@5%%=%.3f  PR-AUC=%.3f ROC-AUC=%.3f  (%.0fs)",
                    seed, prep.k, model_name, best_gamma,
                    r1["recall"], r1["measured_fpr"],
                    r5["recall"], r5["measured_fpr"],
                    auc["prauc"], auc["rocauc"],
                    time.time() - t0,
                )

    # ── Write CSVs ───────────────────────────────────────────────────────────
    res_df = pd.DataFrame(rows)
    per_df = pd.DataFrame(per_rows)

    res_df.to_csv(out / "results.csv", index=False)
    per_df.to_csv(out / "per_attack_recall.csv", index=False)

    summary = (
        res_df.groupby(["k", "model"])[
            ["Recall_1", "Recall_5", "F1_5", "PRAUC", "ROCAUC", "Test_FPR_1", "Test_FPR_5"]
        ]
        .agg(["mean", "std"])
        .round(3)
    )
    summary.to_csv(out / "summary.csv")

    logger.info("\n=== SUMMARY (mean ± std over seeds) ===\n%s", summary.to_string())

    # ── Charts ───────────────────────────────────────────────────────────────
    _plot_recall(res_df, out)
    _plot_per_attack(per_df, out)
    _save_circuit(out)

    # ── Config snapshot ───────────────────────────────────────────────────────
    config_snap = {
        "quick":       quick,
        "run_id":      run_id,
        "seeds":       seeds,
        "ks":          list(ks),
        "n_train":     n_train,
        "n_val":       n_val,
        "per_test":    per_test,
        "per_tune":    per_tune,
        "entanglement": ENTANGLEMENT,
        "gamma_grid":  GAMMA_GRID,
        "target_fprs": TARGET_FPRS,
    }
    (out / "config.json").write_text(json.dumps(config_snap, indent=2))

    return {"rows": rows, "per_rows": per_rows, "out_dir": out}


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _plot_recall(res: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, col, title in zip(
        axes,
        ["Recall_1", "Recall_5"],
        ["Recall at 1% target FPR", "Recall at 5% target FPR"],
    ):
        grouped = res.groupby(["k", "model"])[col].agg(["mean", "std"])
        means   = grouped["mean"].unstack()
        errors  = grouped["std"].unstack().fillna(0)
        means.plot.bar(ax=ax, yerr=errors, capsize=3, rot=0)
        ax.set_title(title)
        ax.set_xlabel("PCA components (= qubits)")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Attack recall")
    plt.tight_layout()
    plt.savefig(out / "chart_recall.png", dpi=150)
    plt.close()


def _plot_per_attack(per: pd.DataFrame, out: Path) -> None:
    if per.empty:
        return
    chart_k = int(per["k"].max())
    pa = (
        per[per.k == chart_k]
        .groupby(["attack", "model"])["recall_5"]
        .mean()
        .unstack()
        .round(3)
    )
    pa.plot.bar(figsize=(8, 4), rot=20)
    plt.ylabel("Recall at 5% target FPR")
    plt.title(f"Recall per attack type (k={chart_k})")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out / "chart_per_attack.png", dpi=150)
    plt.close()


def _save_circuit(out: Path) -> None:
    from .kernels import make_fm
    try:
        make_fm(4).decompose().draw("mpl", filename=str(out / "circuit.png"))
    except Exception as exc:
        logger.warning("Circuit image unavailable: %s", exc)
        try:
            (out / "circuit.txt").write_text(str(make_fm(4).draw("text")))
        except Exception as inner:
            logger.warning("Circuit text unavailable: %s", inner)


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Run QCAD experiment")
    group  = parser.add_mutually_exclusive_group()
    group.add_argument("--quick", action="store_true", help="Fast check (~minutes)")
    group.add_argument("--full",  action="store_true", help="Full run (default)")
    parser.add_argument("--run-id", default=None, help="Output subdirectory name")
    args = parser.parse_args()
    run_experiment(quick=args.quick, run_id=args.run_id)


if __name__ == "__main__":
    main()
