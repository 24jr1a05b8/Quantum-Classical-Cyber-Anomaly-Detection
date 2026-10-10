# CICIDS2017 Dataset Reference

## Overview

**CICIDS2017** (Canadian Institute for Cybersecurity Intrusion Detection System 2017) is a publicly available benchmark dataset for network intrusion detection research.

- **Source:** University of New Brunswick (UNB), Canadian Institute for Cybersecurity
- **Download:** https://www.unb.ca/cic/datasets/ids-2017.html
- **Citation:** Sharafaldin, I., Lashkari, A.H., Ghorbani, A.A. (2018). *Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization.* ICISSP 2018, pp. 108–116.

---

## Files Used in QCAD

| Filename | Capture Day | Labels | Used for |
|----------|-------------|--------|----------|
| `Monday-WorkingHours.pcap_ISCX.csv` | Monday, 3 July 2017 | BENIGN only | Training + Validation |
| `Wednesday-workingHours.pcap_ISCX.csv` | Wednesday, 5 July 2017 | BENIGN + DoS attacks | Test + Tuning pool |
| `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` | Friday, 7 July 2017 (afternoon) | BENIGN + PortScan | Test + Tuning pool |

---

## Feature Set

QCAD uses exactly 11 flow-level statistical features extracted by CICFlowMeter:

```
Flow Duration
Total Fwd Packets
Total Backward Packets
Total Length of Fwd Packets
Total Length of Bwd Packets
Flow Bytes/s
Flow Packets/s
Packet Length Mean
Packet Length Std
Flow IAT Mean
Flow IAT Std
```

---

## Split Strategy

| Split | Source file(s) | Label filter | Size |
|-------|---------------|-------------|------|
| Train | Monday | BENIGN | 300 (quick) / 1 000 (full) |
| Validation | Monday (after train rows) | BENIGN | 1 000 |
| Test | Wed + Fri pool | BENIGN + attack | ~9:1 ratio |
| Tune | Wed + Fri pool | BENIGN + attack | Smaller subset; per-class capped |

**Key properties:**
- All preprocessing (scaler, PCA, MinMax) is fitted on the **training split only**.
- Thresholds are derived from **validation scores only** (benign, Monday).
- Attack labels are **never used for training** — only for evaluation metrics.
- Hyperparameter tuning uses the **tuning split** (labelled, from same attack families as test but different flows).

---

## Attack Types Present

| Attack | Source file | Approx. test count (per seed) |
|--------|-------------|-------------------------------|
| DoS GoldenEye | Wednesday | 60 |
| DoS Hulk | Wednesday | 60 |
| DoS Slowhttptest | Wednesday | 60 |
| DoS slowloris | Wednesday | 60 |
| Heartbleed | Wednesday | 11 ⚠️ low-n |
| PortScan | Friday | 60 |

> ⚠️ **Heartbleed** has only ~11 flows in the dataset after cleaning. Per-attack recall for Heartbleed should be treated as low-confidence.

---

## Data Availability

The CICIDS2017 files are **not included** in this repository. They must be downloaded separately:

1. Visit https://www.unb.ca/cic/datasets/ids-2017.html
2. Complete the access form
3. Download the **MachineLearningCVE** ZIP archive
4. Extract the three CSV files listed above and place them in the **project root** directory

No data files matching `*.pcap_ISCX.csv` are committed to this repository.

---

## Ethical / Legal Notes

- CICIDS2017 is provided for research and educational use.
- The dataset was captured in a controlled lab environment. Traffic patterns do not represent a specific organisation's production network.
- No personal data or PII is contained in the flow statistics.
