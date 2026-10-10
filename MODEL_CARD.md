# Model Card — Quantum-Classical Cyber Anomaly Detection (QCAD)

## Model Details

| Field | Value |
|-------|-------|
| Model type | One-Class SVM (anomaly detection) |
| Kernels | Classical: RBF · Quantum: ZZFeatureMap fidelity kernel (Qiskit Statevector simulator) |
| Version | v2 |
| Framework | scikit-learn 1.9.1 · Qiskit 2.5.2 |
| Inputs | 11 network-flow statistics (see Features below) |
| Output | Anomaly score (higher = more anomalous) + binary flag at chosen threshold |

> **The quantum model runs on a Qiskit Statevector simulator, not real quantum hardware.
> No quantum advantage or speedup is claimed or demonstrated.**

---

## Intended Use

- **Primary use:** Research / educational demonstration of quantum kernel methods applied to network anomaly detection.
- **Secondary use:** Baseline comparison of classical vs. quantum kernel One-Class SVMs on the CICIDS2017 dataset.
- **Not intended for:** Production intrusion detection, real-time traffic analysis, or any security-critical deployment without further validation.

---

## Training Data

- **Source:** CICIDS2017 (University of New Brunswick)
- **Training split:** Monday benign flows only (300 quick / 1 000 full)
- **Validation split:** Monday benign flows (1 000 rows, used only to set thresholds)
- **No attack labels used during training**

### Features (11 flow statistics)

| Feature | Description |
|---------|-------------|
| Flow Duration | Total duration of the flow (μs) |
| Total Fwd Packets | Packets sent in the forward direction |
| Total Backward Packets | Packets sent in the backward direction |
| Total Length of Fwd Packets | Total bytes in forward packets |
| Total Length of Bwd Packets | Total bytes in backward packets |
| Flow Bytes/s | Bytes per second |
| Flow Packets/s | Packets per second |
| Packet Length Mean | Mean packet size |
| Packet Length Std | Std dev of packet sizes |
| Flow IAT Mean | Mean inter-arrival time |
| Flow IAT Std | Std dev of inter-arrival times |

---

## Metrics (v2 — mean ± std over 3 seeds)

> Measured FPR is always shown next to the target FPR.
> Threshold drift: thresholds derived from Monday benign validation; test benign flows
> come from Wednesday/Friday — measured FPR consistently exceeds target.

| k | Model | Recall @1% target | Measured FPR @1% | Recall @5% target | Measured FPR @5% | PR-AUC | ROC-AUC |
|---|-------|-------------------|------------------|-------------------|------------------|--------|---------|
| 4 | Classical | 0.253 ± 0.086 | 0.024 ± 0.004 | 0.683 ± 0.118 | 0.076 ± 0.012 | 0.462 ± 0.042 | — |
| 4 | Quantum (sim.) | 0.335 ± 0.108 | 0.017 ± 0.002 | 0.617 ± 0.085 | 0.061 ± 0.009 | 0.523 ± 0.062 | — |
| 6 | Classical | 0.289 ± 0.138 | 0.016 ± 0.006 | 0.691 ± 0.091 | 0.064 ± 0.005 | 0.511 ± 0.034 | — |
| 6 | Quantum (sim.) | 0.115 ± 0.074 | 0.013 ± 0.005 | 0.674 ± 0.037 | 0.057 ± 0.002 | 0.499 ± 0.008 | — |
| 8 | Classical | 0.293 ± 0.134 | 0.016 ± 0.006 | 0.763 ± 0.063 | 0.066 ± 0.009 | 0.561 ± 0.060 | — |
| 8 | Quantum (sim.) | 0.149 ± 0.021 | 0.011 ± 0.003 | 0.753 ± 0.030 | 0.060 ± 0.003 | 0.574 ± 0.020 | — |

*ROC-AUC added in v2 package; not available in v1 quick results.*

### Per-Attack Recall @ 5% target FPR (k=8, seed=0)

| Attack | Classical | Quantum (sim.) | Notes |
|--------|-----------|----------------|-------|
| DoS GoldenEye | 0.917 | 0.583 | |
| DoS Hulk | 0.933 | 0.867 | |
| DoS Slowhttptest | 0.933 | 0.517 | |
| DoS slowloris | 0.750 | 0.933 | |
| Heartbleed | 1.000 | 1.000 | ⚠️ low-n (11 test samples) |
| PortScan | 0.367 | 0.700 | |

> **Heartbleed has only 11 test flows — treat results as low-confidence (low-n).**

---

## Known Failure Modes

1. **Threshold drift:** The anomaly threshold is set from Monday benign traffic. Test benign flows from Wednesday and Friday differ in distribution; the measured FPR consistently exceeds the target FPR.
2. **Unseen attack families:** The model has only been evaluated on DoS variants and PortScan. It may miss other attack types (e.g., web attacks, botnet, brute force).
3. **Small training set:** 300–1 000 benign flows is far below what a production system requires. The decision boundary may not generalise to diverse real-world traffic.
4. **Simulator-only quantum kernel:** Real hardware runs would introduce gate noise and decoherence, changing kernel values. Results on a simulator do not predict real hardware performance.
5. **Day-to-day traffic shift:** Network behaviour changes over time. A model trained on Monday 2017 traffic may perform poorly on current traffic.
6. **Hyperparameter leakage note:** γ/scale was tuned on a labelled sample drawn from the same attack families as the test set (different flows). This gives an optimistic estimate of generalisation to new attack families.

---

## Ethical Considerations

- This system flags anomalies, not individuals. Flow-level statistics do not contain personal identifiers.
- False positives in a real deployment would cause alert fatigue; false negatives would miss attacks. Both error types must be understood before operational use.
- The quantum model confers no privacy or security properties beyond what the classical model provides.
- Do not claim this system provides quantum-enhanced security without independent verification on real hardware with current, representative traffic.

---

## Limitations Summary

- Simulated quantum kernel only — not run on real quantum hardware.
- CICIDS2017 is a 2017 dataset captured in a controlled lab environment.
- Only 6 attack types (DoS ×4, Heartbleed, PortScan) evaluated.
- Heartbleed: only 11 test samples (low-n, treat with caution).
- Training size is small by production standards.
- No streaming or real-time inference capability.
- No user authentication or access controls in the demo API.
