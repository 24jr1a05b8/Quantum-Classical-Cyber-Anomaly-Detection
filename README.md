# Quantum-Classical Cyber Anomaly Detection (QCAD)

**Qiskit Fall Fest 2026 · IBM Quantum Hackathon**

> **The quantum model runs on a Qiskit Statevector simulator. No quantum speedup or advantage is claimed or observed. Results are reported exactly as measured.**

---

## Problem Statement

Traditional intrusion detection systems (IDS) match known attack signatures — novel or zero-day threats slip through. This project takes a **behavioural anomaly detection** approach: learn what normal (benign) network traffic looks like using a One-Class SVM, then flag deviations. Two kernels are compared:

- **Classical:** RBF kernel (standard One-Class SVM)
- **Quantum (simulator):** Fidelity kernel built from Qiskit's `ZZFeatureMap` and `Statevector` simulation

## Proposed Solution

Both models share an identical preprocessing pipeline. The only difference is the kernel. No attack labels are used during training; the model learns the boundary of normal behaviour and flags everything outside it.

## Team

| Name | Role |
|------|------|
| *[Your Name]* | ML Engineer / Quantum Developer |
| *[Team Member 2]* | Backend / API |
| *[Team Member 3]* | Frontend / UI |

---

## Qiskit Implementation

- **Feature map:** `qiskit.circuit.library.zz_feature_map` (linear entanglement, reps=2)
- **Simulation:** `qiskit.quantum_info.Statevector` (exact statevector simulation)
- **Kernel:** Fidelity kernel k(x,y) = |⟨φ(x)|φ(y)⟩|²
- **Qubits:** Equal to the number of PCA components (k = 4, 6, or 8)
- **Scale factor:** Multiplicative pre-encoding parameter, tuned by PR-AUC on a small labelled tuning set
- **No real hardware run:** Simulator only

## Dataset

**CICIDS2017** — see [`DATASET.md`](DATASET.md) and [`data/README.md`](data/README.md) for download instructions.

| File | Day | Content |
|------|-----|---------|
| `Monday-WorkingHours.pcap_ISCX.csv` | Monday | Benign only (train + val) |
| `Wednesday-workingHours.pcap_ISCX.csv` | Wednesday | Benign + DoS attacks |
| `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` | Friday PM | Benign + PortScan |

> **Important:** Hyperparameters (γ/scale) were tuned on a small labelled sample drawn from the **same attack families** as the test set (different flows). Models never train on attack data.

## Architecture

```
                  ┌─────────────────────────────────┐
 CICIDS2017 CSVs  │   src/qcad/                     │
  Monday ──train─▶│   data.py  →  preprocess.py     │
  Mon    ──val──▶ │   (log1p, StandardScaler,        │
  Wed+Fri──test──▶│    CorrFilter, PCA, MinMax[0,π]) │
                  │                                  │
                  │   models.py                      │
                  │   ├─ ClassicalDetector (RBF)     │
                  │   └─ QuantumDetector             │
                  │       (ZZFeatureMap + Statevec.) │
                  │                                  │
                  │   evaluate.py (threshold, AUC)   │
                  │   experiment.py (grid search)    │
                  │   train.py (save artifacts)      │
                  └─────────────┬───────────────────┘
                                │ artifacts/v2/
                                ▼
                         src/qcad/api.py  (FastAPI)
                                │
                         frontend/ (React+Vite)
```

## Quick Start (Local)

### 1. Prerequisites

- Python 3.11+
- Node.js 20+ (frontend)
- CICIDS2017 CSV files in the project root (see `data/README.md`)

### 2. Install

```bash
pip install -r requirements.txt -r requirements-dev.txt
```

### 3. Run the experiment (quick mode)

```bash
python -m src.qcad.experiment --quick
```

Output written to `results/v2_quick/`.

### 4. Train and save artifacts

```bash
python -m src.qcad.train --quick --results-csv results/v2_quick/results.csv
```

Artifacts saved to `artifacts/v2/`.

### 5. Start the API

```bash
uvicorn src.qcad.api:app --reload
```

API docs at http://localhost:8000/docs

### 6. Start the frontend (dev)

```bash
cd frontend
npm install
npm run dev
```

App at http://localhost:5173

## Reproduce Results

### v1 Quick (original prototype, preserved untouched)

Results are in `results/v1_quick/`. The original script is at `run_v1_quick/main.py`.

### v2 Full (refactored package)

```bash
python -m src.qcad.experiment --full
```

Output in `results/v2/`. To also save trained artifacts:

```bash
python -m src.qcad.train --full --results-csv results/v2/results.csv
```

## Results (v2 — mean ± std over 3 seeds)

> All numbers from a real run. Measured FPR is shown next to each recall target.
> The quantum model runs on a **simulator**. No quantum advantage is claimed.

| k | Model | Recall @1% FPR | Measured FPR @1% | Recall @5% FPR | Measured FPR @5% | PR-AUC |
|---|-------|---------------|-------------------|----------------|-------------------|--------|
| 4 | Classical | 0.253 ± 0.086 | 0.024 ± 0.004 | 0.683 ± 0.118 | 0.076 ± 0.012 | 0.462 ± 0.042 |
| 4 | Quantum (sim.) | 0.335 ± 0.108 | 0.017 ± 0.002 | 0.617 ± 0.085 | 0.061 ± 0.009 | 0.523 ± 0.062 |
| 6 | Classical | 0.289 ± 0.138 | 0.016 ± 0.006 | 0.691 ± 0.091 | 0.064 ± 0.005 | 0.511 ± 0.034 |
| 6 | Quantum (sim.) | 0.115 ± 0.074 | 0.013 ± 0.005 | 0.674 ± 0.037 | 0.057 ± 0.002 | 0.499 ± 0.008 |
| 8 | Classical | 0.293 ± 0.134 | 0.016 ± 0.006 | 0.763 ± 0.063 | 0.066 ± 0.009 | 0.561 ± 0.060 |
| 8 | Quantum (sim.) | 0.149 ± 0.021 | 0.011 ± 0.003 | 0.753 ± 0.030 | 0.060 ± 0.003 | 0.574 ± 0.020 |

**Observations:**
- At k=1% FPR threshold, the quantum kernel tends to produce lower measured FPR (more conservative).
- Both models achieve >75% recall at 5% target FPR for k=8.
- Classical model wins on recall @1% FPR for k≥6; quantum is competitive at k=4.
- Measured FPR consistently exceeds target (threshold shift from Monday→Wednesday/Friday test set).

## Running Tests

```bash
pytest tests/test_unit.py tests/test_api.py -v
# After building artifacts:
pytest tests/ -v
```

## Docker

```bash
# Build and run
docker compose up --build

# API at http://localhost:8000
# Docs at http://localhost:8000/docs
```

## Deployment Guide

### Render (recommended free tier)

1. Push to GitHub.
2. Create a new **Web Service** → Docker → select your repo.
3. Set environment variables: `CORS_ORIGINS=https://your-frontend.vercel.app`.
4. Artifacts (`artifacts/v2/`) must be baked into the image or mounted as a persistent disk.
5. Note: free tier has cold-start delays of ~30 s. First `/health` call may time out.

### Railway

1. Connect repo → **Deploy from Dockerfile**.
2. Set `PORT=8000` and `CORS_ORIGINS`.
3. Artifacts can be committed or uploaded via Railway volumes.

### Fly.io

```bash
fly launch --dockerfile Dockerfile
fly secrets set CORS_ORIGINS=https://your-frontend.vercel.app
fly deploy
```

Expected image size: ~1.2 GB (Qiskit + NumPy dominant).

### Vercel / Netlify (frontend only)

Set `VITE_API_URL=https://your-api.onrender.com` in the Vercel environment variables, then deploy the `frontend/` directory.

## Limitations

- Quantum model: **simulated**. No real hardware was used. No quantum advantage.
- Small training set (300–1 000 benign flows). Real-world deployments need more data.
- Only DoS and PortScan attack families evaluated. Heartbleed has very few test samples.
- Threshold drift: Monday thresholds applied to Wednesday/Friday test data — measured FPR > target.
- No streaming or real-time inference.

## References

- Sharafaldin, I., Lashkari, A.H., Ghorbani, A.A. (2018). Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization. *ICISSP 2018*, pp. 108–116.
- Qiskit Contributors (2023). Qiskit: An Open-source Framework for Quantum Computing. https://doi.org/10.5281/zenodo.2573505
- Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR* 12, pp. 2825–2830.
- Havlíček et al. (2019). Supervised learning with quantum-enhanced feature spaces. *Nature* 567, pp. 209–212.
