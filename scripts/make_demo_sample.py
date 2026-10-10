"""Create data/demo_sample.csv from Monday + attack pool (max 300 rows).

Usage
-----
    python scripts/make_demo_sample.py

The script requires the CICIDS2017 CSVs to be present in the project root.
The output file is safe to commit (no PII, standard network-flow statistics).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is on the path when run from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.qcad.config import FEATURES
from src.qcad.data import load_all_data

OUT = Path("data/demo_sample.csv")
N_BENIGN  = 200
N_ATTACK  = 100


def main() -> None:
    print("Loading data …")
    mon, pool = load_all_data()

    benign  = mon.sample(n=min(N_BENIGN, len(mon)), random_state=42)
    attacks = pool[pool["Label"] != "BENIGN"].sample(
        n=min(N_ATTACK, (pool["Label"] != "BENIGN").sum()), random_state=42
    )

    sample = (
        pd.concat([benign, attacks], ignore_index=True)
        .sample(frac=1, random_state=0)
        .reset_index(drop=True)
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    sample[FEATURES + ["Label"]].to_csv(OUT, index=False)
    print(f"Saved {len(sample)} rows to {OUT}")
    print(f"  Benign:  {(sample['Label'] == 'BENIGN').sum()}")
    print(f"  Attacks: {(sample['Label'] != 'BENIGN').sum()}")


if __name__ == "__main__":
    main()
