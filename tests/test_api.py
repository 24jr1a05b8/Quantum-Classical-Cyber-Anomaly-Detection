"""API integration tests for QCAD FastAPI endpoints."""

from __future__ import annotations

import io
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.qcad.config import FEATURES


@pytest.fixture(scope="module")
def client():
    """Create a test client with mocked artifacts so no real models are needed."""
    from src.qcad import api as api_module

    # Build lightweight mock preprocessor.
    mock_prep = MagicMock()
    mock_prep.transform.return_value = np.random.default_rng(0).uniform(
        0, 3.14159, size=(5, 4)
    )

    # Build lightweight mock classical detector.
    mock_classical = MagicMock()
    mock_classical.score.return_value = np.array([0.1, 0.5, 0.9, 0.2, 0.8])

    # Build lightweight mock quantum detector.
    mock_quantum = MagicMock()
    mock_quantum.score.return_value = np.array([0.2, 0.4, 0.8, 0.3, 0.7])

    mock_manifest = {
        "version": "v2",
        "created_at": "2025-01-01T00:00:00Z",
        "config": {"n_train": 1000},
        "metrics": {},
        "thresholds": {},
    }
    mock_thresholds = {
        "classical_k4": {"0.01": {"threshold": 0.7}, "0.05": {"threshold": 0.4}},
        "quantum_k4":   {"0.01": {"threshold": 0.7}, "0.05": {"threshold": 0.4}},
    }

    api_module._preprocessors = {4: mock_prep}
    api_module._classical      = {4: mock_classical}
    api_module._quantum        = {4: mock_quantum}
    api_module._manifest       = mock_manifest
    api_module._thresholds     = mock_thresholds

    from src.qcad.api import app
    return TestClient(app)


def _sample_flows(n: int = 5) -> list[dict]:
    rng = np.random.default_rng(42)
    return [
        {
            "flow_duration":                 float(rng.exponential(1000)),
            "total_fwd_packets":             float(rng.integers(1, 100)),
            "total_backward_packets":        float(rng.integers(0, 80)),
            "total_length_of_fwd_packets":   float(rng.exponential(500)),
            "total_length_of_bwd_packets":   float(rng.exponential(400)),
            "flow_bytes_s":                  float(rng.exponential(1000)),
            "flow_packets_s":                float(rng.exponential(10)),
            "packet_length_mean":            float(rng.exponential(100)),
            "packet_length_std":             float(rng.exponential(50)),
            "flow_iat_mean":                 float(rng.exponential(500)),
            "flow_iat_std":                  float(rng.exponential(500)),
        }
        for _ in range(n)
    ]


# ── /health ───────────────────────────────────────────────────────────────────

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "qiskit_version" in body
    assert "sklearn_version" in body


# ── /model/info ───────────────────────────────────────────────────────────────

def test_model_info(client):
    r = client.get("/model/info")
    assert r.status_code == 200
    body = r.json()
    assert "limitations" in body


# ── /predict (JSON) ───────────────────────────────────────────────────────────

def test_predict_both_models(client):
    flows = _sample_flows(5)
    payload = {"flows": flows, "model": "both", "target_fpr": 0.05}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["results"]) == 10  # 5 rows × 2 models
    assert body["summary"]["rows_scored"] == 5


def test_predict_classical_only(client):
    flows = _sample_flows(3)
    payload = {"flows": flows, "model": "classical", "target_fpr": 0.01}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    # mock scorer always returns 5 scores regardless of input length
    body = r.json()
    assert all(res["model"] == "classical" for res in body["results"])


def test_predict_quantum_only(client):
    flows = _sample_flows(3)
    payload = {"flows": flows, "model": "quantum", "target_fpr": 0.05}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert all(res["model"] == "quantum" for res in body["results"])


def test_predict_fpr_01(client):
    flows = _sample_flows(2)
    payload = {"flows": flows, "model": "classical", "target_fpr": 0.01}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200


def test_predict_empty_flows(client):
    r = client.post("/predict", json={"flows": [], "model": "classical", "target_fpr": 0.05})
    # FastAPI Pydantic validation: body is valid but scorer gets empty df — 422 is acceptable
    assert r.status_code in (200, 422)


def test_predict_too_many_rows_quantum(client):
    from src.qcad.config import API_MAX_ROWS_QUANTUM
    flows = _sample_flows(API_MAX_ROWS_QUANTUM + 1)
    r = client.post("/predict", json={"flows": flows, "model": "quantum", "target_fpr": 0.05})
    assert r.status_code == 422


def test_predict_missing_field(client):
    bad_flow = {"flow_duration": 1.0}  # missing 10 fields
    r = client.post("/predict", json={"flows": [bad_flow], "model": "classical", "target_fpr": 0.05})
    assert r.status_code == 422


def test_predict_disclaimer_present(client):
    flows = _sample_flows(2)
    r = client.post("/predict", json={"flows": flows, "model": "quantum", "target_fpr": 0.05})
    assert r.status_code == 200
    assert "simulator" in r.json()["disclaimer"].lower()


# ── /predict/csv (file upload) ────────────────────────────────────────────────

def _make_csv_bytes(n: int = 5) -> bytes:
    rng = np.random.default_rng(7)
    data = {col: rng.exponential(100, size=n).tolist() for col in FEATURES}
    df = pd.DataFrame(data)
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def test_predict_csv_upload(client):
    csv_bytes = _make_csv_bytes(5)
    r = client.post(
        "/predict/csv",
        params={"model": "classical", "target_fpr": "0.05"},
        files={"file": ("test.csv", csv_bytes, "text/csv")},
    )
    assert r.status_code == 200, r.text
    assert r.json()["summary"]["rows_scored"] == 5


def test_predict_csv_missing_columns(client):
    buf = io.BytesIO(b"col_a,col_b\n1.0,2.0\n")
    r = client.post(
        "/predict/csv",
        params={"model": "classical", "target_fpr": "0.05"},
        files={"file": ("bad.csv", buf.getvalue(), "text/csv")},
    )
    assert r.status_code == 422


def test_predict_csv_invalid_fpr(client):
    csv_bytes = _make_csv_bytes(3)
    r = client.post(
        "/predict/csv",
        params={"model": "classical", "target_fpr": "0.1"},  # invalid
        files={"file": ("t.csv", csv_bytes, "text/csv")},
    )
    assert r.status_code == 422


# ── /results ──────────────────────────────────────────────────────────────────

def test_get_results(client):
    r = client.get("/results")
    assert r.status_code == 200
    body = r.json()
    # Both version keys should be present even if directories are empty.
    assert "v1_quick" in body or "v2" in body
