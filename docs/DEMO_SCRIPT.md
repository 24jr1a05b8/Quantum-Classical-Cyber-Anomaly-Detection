# Demo Script — 10-Minute Presentation Guide

This script covers the key talking points and live commands for the final presentation.

---

## Setup (before the demo)

```bash
# 1. Clone / open the repo
cd quantum-cyber-anomaly

# 2. Confirm artifacts exist
ls artifacts/v2/

# 3. Start the API (leave running in background)
uvicorn src.qcad.api:app --reload &

# 4. Start the frontend (separate terminal)
cd frontend && npm run dev &
# → http://localhost:5173
```

---

## Step 1 — Problem statement (1 min)

> "Security tools catch known attack signatures but miss novel threats.
> We train a model on *normal* traffic only, then flag anything that looks unusual."

Point at the About page in the browser.

---

## Step 2 — Architecture overview (1 min)

> "Both models share identical preprocessing — log transform, scaling, PCA to k components.
> The only difference is the kernel:
> classical uses RBF, quantum uses a fidelity kernel built from Qiskit's ZZFeatureMap
> simulated with Statevector."

Show the pipeline diagram on the About page.

---

## Step 3 — Live prediction (3 min)

1. Navigate to **Detect** page.
2. Click **Use Demo Sample** → select **both** models → **5% target FPR** → run.
3. Point at the score histogram: "The red dashed line is the threshold. Flows to the right are flagged."
4. Point at the results table: flagged rows highlighted in red.
5. Show the disclaimer banner: "Quantum model runs on a simulator — no quantum advantage claimed."

Expected output:
- ~5 % of benign flows flagged (measured FPR near target)
- Attack flows scored higher on average

---

## Step 4 — Results comparison (2 min)

1. Navigate to **Compare** page.
2. Toggle between **v1 Quick** and **v2 Full** tabs.
3. Point at recall grouped bar chart: "At k=8, both models achieve ~75 % recall at 5% target FPR."
4. Point at the measured FPR table: "Orange numbers show the *actual* FPR on the test set — always shown next to the target."
5. Point at per-attack recall: "Heartbleed has only 11 test samples — marked with * (low-n)."

---

## Step 5 — API demo (1 min)

```bash
# Health check
curl http://localhost:8000/health

# Expected:
# {"status":"ok","model_version":"v2","artifacts_loaded":true,...}

# Predict via JSON
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"flows":[{"flow_duration":100000,"total_fwd_packets":5,"total_backward_packets":3,
       "total_length_of_fwd_packets":500,"total_length_of_bwd_packets":300,
       "flow_bytes_s":800,"flow_packets_s":8,"packet_length_mean":100,
       "packet_length_std":20,"flow_iat_mean":5000,"flow_iat_std":2000}],
       "model":"both","target_fpr":0.05}' | python -m json.tool
```

Show the OpenAPI docs at http://localhost:8000/docs.

---

## Step 6 — Qiskit circuit (30 sec)

Show `results/v2/circuit.png` or the Compare page circuit image.

> "This is the ZZFeatureMap circuit for k=4 qubits.
> Each feature is encoded as a rotation angle.
> The ZZ entanglement gates create feature interactions.
> The fidelity kernel measures how close two encoded states are in Hilbert space."

---

## Step 7 — Honest assessment (1 min)

> "Classical and quantum achieve similar recall at 5% FPR.
> The quantum kernel shows lower measured FPR in some configurations,
> suggesting it learns a tighter boundary of 'normal' — but this is
> a simulated kernel on a small dataset. We cannot claim quantum advantage.
> What we *can* demonstrate is a complete end-to-end pipeline using Qiskit
> as a drop-in kernel, fully reproducible from a fresh clone."

---

## Expected Commands & Output

```bash
# Run quick experiment (no artifacts needed, ~5 min)
python -m src.qcad.experiment --quick
# → results/v2_quick/results.csv, summary.csv, charts

# Run tests
pytest tests/test_unit.py tests/test_api.py -v
# → All tests pass

# Build artifact (quick)
python -m src.qcad.train --quick
# → artifacts/v2/ created

# Docker
docker compose up --build
curl http://localhost:8000/health
# → {"status":"ok",...}
```

---

## Fallback (if live demo fails)

Show `results/v2/chart_recall.png` and `results/v2/chart_per_attack.png` directly.
Numbers in README.md § Results match those charts exactly.
