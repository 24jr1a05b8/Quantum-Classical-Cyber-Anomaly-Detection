"""Fit final models on full training data and persist artifacts.

CLI usage
---------
    python -m qcad.train --quick    # small training set, fast
    python -m qcad.train --full     # full training set (default)
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import subprocess
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd

from .config import (
    ARTIFACTS_DIR,
    DATA_FILES,
    ENTANGLEMENT,
    FULL_KS,
    FULL_N_TRAIN,
    FULL_N_VAL,
    FULL_PER_TEST,
    FULL_PER_TUNE,
    GAMMA_GRID,
    QUICK_KS,
    QUICK_N_TRAIN,
    QUICK_N_VAL,
    QUICK_PER_TEST,
    QUICK_PER_TUNE,
    REPS,
    TARGET_FPRS,
    VERSION,
)
from .data import build_splits, load_all_data
from .evaluate import compute_auc_metrics, compute_threshold, evaluate
from .models import ClassicalDetector, QuantumDetector
from .preprocess import Preprocessor

logger = logging.getLogger(__name__)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _best_gamma_from_results(results_csv: Path, model: str, k: int) -> float | None:
    """Find the best gamma from experiment results CSV (best mean tuning PR-AUC)."""
    if not results_csv.exists():
        return None
    df = pd.read_csv(results_csv)
    sub = df[(df["model"] == model) & (df["k"] == k)]
    if sub.empty:
        return None
    best = sub.groupby("gamma")["PRAUC"].mean().idxmax()
    return float(best)


def train(
    *,
    quick: bool = False,
    version: str = VERSION,
    results_csv: Path | None = None,
) -> Path:
    """Fit final models and write artifact bundle.

    Parameters
    ----------
    quick:
        Use smaller training sizes (fast path for CI / smoke test).
    version:
        Artifact version string (sub-directory under artifacts/).
    results_csv:
        Optional path to a previously produced results.csv to read best gammas
        from.  If not provided, a fresh grid search is run.

    Returns
    -------
    Path to the artifact directory.
    """
    artifact_dir = ARTIFACTS_DIR / version
    artifact_dir.mkdir(parents=True, exist_ok=True)

    n_train  = QUICK_N_TRAIN  if quick else FULL_N_TRAIN
    n_val    = QUICK_N_VAL    if quick else FULL_N_VAL
    per_test = QUICK_PER_TEST if quick else FULL_PER_TEST
    per_tune = QUICK_PER_TUNE if quick else FULL_PER_TUNE
    ks       = QUICK_KS       if quick else FULL_KS

    logger.info("Loading data …")
    mon, pool = load_all_data()

    train_df, val_df, test_df, tune_df = build_splits(
        mon, pool,
        n_train=n_train, n_val=n_val,
        per_test=per_test, per_tune=per_tune,
        seed=0,
    )
    y_test  = (test_df["Label"] != "BENIGN").astype(int).to_numpy()
    y_tune  = (tune_df["Label"] != "BENIGN").astype(int).to_numpy()
    labels  = test_df["Label"].to_numpy()

    from sklearn.metrics import average_precision_score

    thresholds_all: dict = {}
    metrics_all: dict = {}
    saved_artifacts: list = []

    for k in ks:
        prep = Preprocessor(k=k).fit(train_df)
        X_tr   = prep.transform(train_df)
        X_val  = prep.transform(val_df)
        X_te   = prep.transform(test_df)
        X_tune = prep.transform(tune_df)

        prep_path = artifact_dir / f"preprocessor_k{prep.k}.pkl"
        prep.save(prep_path)
        saved_artifacts.append(str(prep_path))

        for model_name in ("classical", "quantum"):
            # ── find best gamma ───────────────────────────────────────────
            gamma = _best_gamma_from_results(
                results_csv or Path("results/v2/results.csv"),
                model_name, prep.k,
            )

            if gamma is None:
                logger.info(
                    "No results CSV found; running quick gamma grid for %s k=%d",
                    model_name, prep.k,
                )
                grid = GAMMA_GRID[model_name]
                best_ap, gamma = -1.0, grid[0]
                for g in grid:
                    if model_name == "classical":
                        det = ClassicalDetector(gamma=g).fit(X_tr)
                    else:
                        det = QuantumDetector(scale=g, k=prep.k).fit(X_tr)
                    ap = float(average_precision_score(y_tune, det.score(X_tune)))
                    if ap > best_ap:
                        best_ap, gamma = ap, g

            # ── refit ─────────────────────────────────────────────────────
            if model_name == "classical":
                det = ClassicalDetector(gamma=gamma).fit(X_tr)
            else:
                det = QuantumDetector(scale=gamma, k=prep.k).fit(X_tr)

            model_path = artifact_dir / f"{model_name}_k{prep.k}.pkl"
            det.save(model_path)
            saved_artifacts.append(str(model_path))

            # ── thresholds ────────────────────────────────────────────────
            val_sc  = det.score(X_val)
            test_sc = det.score(X_te)
            auc     = compute_auc_metrics(test_sc, y_test)

            key = f"{model_name}_k{prep.k}"
            thresholds_all[key] = {}
            for fpr in TARGET_FPRS:
                thr  = compute_threshold(val_sc, fpr)
                ev   = evaluate(test_sc, y_test, labels, thr)
                thresholds_all[key][str(fpr)] = {
                    "threshold":    thr,
                    "recall":       ev["recall"],
                    "measured_fpr": ev["measured_fpr"],
                    "f1":           ev["f1"],
                }
            metrics_all[key] = {
                "prauc":   auc["prauc"],
                "rocauc":  auc["rocauc"],
                "gamma":   gamma,
                "k":       prep.k,
                "model":   model_name,
                "n_train": len(X_tr),
            }

            logger.info(
                "Saved %s k=%d (gamma=%.3f)  PR-AUC=%.3f  ROC-AUC=%.3f",
                model_name, prep.k, gamma, auc["prauc"], auc["rocauc"],
            )

    # ── thresholds.json ───────────────────────────────────────────────────────
    (artifact_dir / "thresholds.json").write_text(
        json.dumps(thresholds_all, indent=2)
    )

    # ── manifest.json ─────────────────────────────────────────────────────────
    import qiskit
    import sklearn

    manifest = {
        "version":    version,
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "git_commit": _git_commit(),
        "quick":      quick,
        "config": {
            "n_train":     n_train,
            "n_val":       n_val,
            "entanglement": ENTANGLEMENT,
            "reps":        REPS,
            "gamma_grid":  GAMMA_GRID,
            "target_fprs": TARGET_FPRS,
        },
        "library_versions": {
            "qiskit":       qiskit.__version__,
            "sklearn":      sklearn.__version__,
        },
        "data_files": {
            k: {"name": str(p.name), "sha256": _sha256(p) if p.exists() else "missing"}
            for k, p in DATA_FILES.items()
        },
        "metrics":   metrics_all,
        "thresholds": thresholds_all,
        "artifacts": [str(Path(a).name) for a in saved_artifacts],
    }
    (artifact_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    logger.info("Artifacts written to %s", artifact_dir)
    for f in sorted(artifact_dir.iterdir()):
        logger.info("  %s", f.name)

    return artifact_dir


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Train and save QCAD artifacts")
    group  = parser.add_mutually_exclusive_group()
    group.add_argument("--quick", action="store_true")
    group.add_argument("--full",  action="store_true")
    parser.add_argument("--version", default=VERSION)
    parser.add_argument(
        "--results-csv", default=None,
        help="Path to results.csv from a prior experiment run",
    )
    args = parser.parse_args()
    results_csv = Path(args.results_csv) if args.results_csv else None
    train(quick=args.quick, version=args.version, results_csv=results_csv)


if __name__ == "__main__":
    main()
