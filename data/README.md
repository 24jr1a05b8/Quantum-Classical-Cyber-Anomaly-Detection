# CICIDS2017 Dataset

The raw CSV files are **not included** in this repository to respect the
dataset's terms of use and to keep the repo size small.

## Files used

| Key  | Filename                                         | Capture day  | Content                                 |
|------|--------------------------------------------------|--------------|-----------------------------------------|
| mon  | `Monday-WorkingHours.pcap_ISCX.csv`              | Monday       | Benign traffic only (train + val set)   |
| wed  | `Wednesday-workingHours.pcap_ISCX.csv`           | Wednesday    | Benign + DoS attacks (test + tune pool) |
| fri  | `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` | Friday PM | Benign + PortScan (test + tune pool)    |

## How to download

1. Visit the [UNB CICIDS2017 page](https://www.unb.ca/cic/datasets/ids-2017.html).
2. Fill in the access form and download the **MachineLearningCVE** archive.
3. Extract the three files above and place them in the **project root** (same level as `src/`).

## Splits

| Split     | Source                  | Labels     | Size (approximate)     |
|-----------|-------------------------|------------|------------------------|
| Train     | Monday (first N rows)   | Benign     | 300 (quick) / 1 000 (full) |
| Validation| Monday (next N rows)    | Benign     | 1 000                  |
| Test      | Wed + Fri pool          | Benign + attacks | ≈9:1 benign:attack ratio |
| Tune      | Wed + Fri pool          | Benign + attacks | Smaller; used for γ grid search only |

## No data committed

Running `git status` must never show any `.pcap_ISCX.csv` files.
The `.gitignore` excludes all `*.pcap_ISCX.csv` patterns.

## Demo sample

`data/demo_sample.csv` is a small synthetic or real subset (a few hundred rows)
with column names matching the 11 features + `Label`.  It **can** be committed
(check attribution below).  Generate it with:

```bash
python scripts/make_demo_sample.py
```

## Citation

Sharafaldin, I., Lashkari, A.H., Ghorbani, A.A. (2018).
*Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization*.
Proceedings of the 4th International Conference on Information Systems Security and Privacy
(ICISSP 2018), pp. 108–116.
