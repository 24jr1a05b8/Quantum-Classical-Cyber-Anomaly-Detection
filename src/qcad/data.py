"""Data loading, cleaning and split utilities for QCAD."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_FILES, FEATURES

logger = logging.getLogger(__name__)


def load_csv(path: Path | str) -> pd.DataFrame:
    """Load a CICIDS2017 CSV, strip header whitespace, drop Inf/NaN/duplicates.

    Parameters
    ----------
    path:
        Path to a CICIDS2017 flow CSV file.

    Returns
    -------
    pd.DataFrame with exactly the 11 feature columns plus ``Label``.
    """
    wanted = set(FEATURES + ["Label"])
    df = pd.read_csv(
        path,
        usecols=lambda col: col.strip() in wanted,
        skipinitialspace=True,
    )
    df.columns = df.columns.str.strip()

    # Ensure column order is canonical.
    df = df[FEATURES + ["Label"]].copy()

    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan)
    df = df.dropna().drop_duplicates()
    df["Label"] = df["Label"].astype(str).str.strip()

    logger.info("Loaded %s: %d rows after cleaning", Path(path).name, len(df))
    return df


def build_splits(
    mon: pd.DataFrame,
    pool: pd.DataFrame,
    *,
    n_train: int,
    n_val: int,
    per_test: int,
    per_tune: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Derive train / val / test / tune splits.

    Train and val come from Monday (benign-only).
    Test and tune are built from Wednesday + Friday with ~10 % attack prevalence.

    Parameters
    ----------
    mon:
        Monday (benign-only) DataFrame.
    pool:
        Combined Wednesday + Friday DataFrame (benign + attack rows).
    n_train, n_val:
        Sizes of the training and validation benign sets.
    per_test, per_tune:
        Maximum attack rows *per attack class* drawn for test / tune.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    train, val, test, tune DataFrames.
    """
    m = mon.sample(frac=1, random_state=seed).reset_index(drop=True)

    if len(m) < n_train + n_val:
        raise ValueError(
            f"Monday only has {len(m)} rows but need {n_train + n_val} "
            f"(n_train={n_train}, n_val={n_val})."
        )

    train = m.iloc[:n_train]
    val   = m.iloc[n_train : n_train + n_val]

    ben = (
        pool[pool["Label"] == "BENIGN"]
        .sample(frac=1, random_state=0)
        .reset_index(drop=True)
    )
    att = pool[pool["Label"] != "BENIGN"]

    att_te_parts, att_tu_parts = [], []
    for attack_label, group in att.groupby("Label"):
        g = group.sample(frac=1, random_state=0)
        n_test = min(per_test, len(g))
        n_tune = min(per_tune, max(0, len(g) - n_test))
        if n_test:
            att_te_parts.append(g.iloc[:n_test])
        if n_tune:
            att_tu_parts.append(g.iloc[n_test : n_test + n_tune])

    if not att_te_parts or not att_tu_parts:
        raise ValueError(
            "Not enough attack samples for test and tuning. "
            "Reduce per_test or per_tune."
        )

    att_te = pd.concat(att_te_parts, ignore_index=True)
    att_tu = pd.concat(att_tu_parts, ignore_index=True)

    n_te_b = min(9 * len(att_te), len(ben))
    n_tu_b = min(3 * len(att_tu), max(0, len(ben) - n_te_b))

    test = (
        pd.concat([ben.iloc[:n_te_b], att_te], ignore_index=True)
        .sample(frac=1, random_state=10)
        .reset_index(drop=True)
    )
    tune = (
        pd.concat([ben.iloc[n_te_b : n_te_b + n_tu_b], att_tu], ignore_index=True)
        .sample(frac=1, random_state=11)
        .reset_index(drop=True)
    )

    logger.info(
        "Splits — train: %d  val: %d  test: %d (%d attacks)  tune: %d (%d attacks)",
        len(train), len(val), len(test),
        int((test["Label"] != "BENIGN").sum()),
        len(tune),
        int((tune["Label"] != "BENIGN").sum()),
    )

    return train, val, test, tune


def load_all_data(
    files: dict[str, Path] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Monday + pool (Wednesday + Friday).

    Returns
    -------
    mon, pool DataFrames.
    """
    if files is None:
        files = DATA_FILES

    mon  = load_csv(files["mon"])
    pool = pd.concat(
        [load_csv(files["wed"]), load_csv(files["fri"])],
        ignore_index=True,
    )
    return mon, pool
