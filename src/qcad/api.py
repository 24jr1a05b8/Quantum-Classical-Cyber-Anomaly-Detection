"""FastAPI application for QCAD inference and results serving."""

from __future__ import annotations

import io
import json
import logging
import math
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import (
    API_MAX_ROWS_CLASSICAL,
    API_MAX_ROWS_QUANTUM,
    API_MAX_UPLOAD_BYTES,
    ARTIFACTS_DIR,
    FEATURES,
    RESULTS_DIR,
    VERSION,
)
from .models import ClassicalDetector, QuantumDetector
from .preprocess import Preprocessor

logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────

def _sanitize(obj):
    """Recursively replace NaN/Inf with None so JSON serialization never fails."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    _load_artifacts()
    # Serve pre-built frontend if present.
    frontend_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
    if frontend_dist.exists():
        app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="static")
        logger.info("Serving frontend from %s", frontend_dist)
    yield


app = FastAPI(
    title="Quantum-Classical Cyber Anomaly Detection",
    description=(
        "Anomaly detection API using classical (RBF kernel) and quantum "
        "(ZZFeatureMap fidelity kernel, Qiskit Statevector simulator) "
        "One-Class SVM models trained on CICIDS2017 benign traffic. "
        "**The quantum model runs on a simulator — no quantum advantage is claimed.**"
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────

_cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _cors_origins],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Artifact loading ──────────────────────────────────────────────────────────

_ARTIFACT_DIR: Path = ARTIFACTS_DIR / VERSION
_manifest: dict | None = None
_thresholds: dict | None = None
_preprocessors: dict[int, Preprocessor] = {}
_classical: dict[int, ClassicalDetector] = {}
_quantum: dict[int, QuantumDetector] = {}


def _load_artifacts() -> None:
    global _manifest, _thresholds
    manifest_path = _ARTIFACT_DIR / "manifest.json"
    threshold_path = _ARTIFACT_DIR / "thresholds.json"

    if not manifest_path.exists():
        logger.warning("No manifest found at %s — API running without artifacts", manifest_path)
        return

    _manifest   = json.loads(manifest_path.read_text())
    _thresholds = json.loads(threshold_path.read_text())

    for pkl in sorted(_ARTIFACT_DIR.glob("preprocessor_k*.pkl")):
        k = int(pkl.stem.split("k")[1])
        _preprocessors[k] = Preprocessor.load(pkl)

    for pkl in sorted(_ARTIFACT_DIR.glob("classical_k*.pkl")):
        k = int(pkl.stem.split("k")[1])
        _classical[k] = ClassicalDetector.load(pkl)

    for pkl in sorted(_ARTIFACT_DIR.glob("quantum_k*.pkl")):
        k = int(pkl.stem.split("k")[1])
        _quantum[k] = QuantumDetector.load(pkl)

    logger.info(
        "Artifacts loaded: preprocessors=%s classical=%s quantum=%s",
        list(_preprocessors), list(_classical), list(_quantum),
    )



# ── Schemas ───────────────────────────────────────────────────────────────────

class FlowRecord(BaseModel):
    """A single network-flow feature vector."""
    flow_duration:                  float
    total_fwd_packets:              float
    total_backward_packets:         float
    total_length_of_fwd_packets:    float
    total_length_of_bwd_packets:    float
    flow_bytes_s:                   float
    flow_packets_s:                 float
    packet_length_mean:             float
    packet_length_std:              float
    flow_iat_mean:                  float
    flow_iat_std:                   float


_JSON_FEATURE_MAP = {
    "flow_duration":                "Flow Duration",
    "total_fwd_packets":            "Total Fwd Packets",
    "total_backward_packets":       "Total Backward Packets",
    "total_length_of_fwd_packets":  "Total Length of Fwd Packets",
    "total_length_of_bwd_packets":  "Total Length of Bwd Packets",
    "flow_bytes_s":                 "Flow Bytes/s",
    "flow_packets_s":               "Flow Packets/s",
    "packet_length_mean":           "Packet Length Mean",
    "packet_length_std":            "Packet Length Std",
    "flow_iat_mean":                "Flow IAT Mean",
    "flow_iat_std":                 "Flow IAT Std",
}


class PredictRequest(BaseModel):
    flows:       list[FlowRecord]
    model:       Literal["classical", "quantum", "both"] = "both"
    target_fpr:  Literal[0.01, 0.05] = 0.05


class PredictResult(BaseModel):
    row_index:     int
    score:         float
    flagged:       bool
    threshold:     float
    model:         str
    model_version: str


class PredictResponse(BaseModel):
    results:        list[PredictResult]
    summary:        dict
    rows_received:  int
    rows_dropped:   int
    drop_reasons:   list[str]
    disclaimer:     str


# ── Helpers ───────────────────────────────────────────────────────────────────

def _require_artifacts() -> None:
    if not _preprocessors:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts not loaded. Run `python -m qcad.train` first.",
        )


def _best_k() -> int:
    return max(_preprocessors) if _preprocessors else 8


def _get_threshold(model_name: str, k: int, target_fpr: float) -> float:
    if _thresholds is None:
        raise HTTPException(status_code=503, detail="Thresholds not loaded.")
    key = f"{model_name}_k{k}"
    fpr_key = str(target_fpr)
    return _thresholds.get(key, {}).get(fpr_key, {}).get("threshold", 0.0)


def _df_from_records(flows: list[FlowRecord]) -> pd.DataFrame:
    rows = []
    for f in flows:
        rows.append({FEAT: getattr(f, JSON_KEY) for JSON_KEY, FEAT in _JSON_FEATURE_MAP.items()})
    return pd.DataFrame(rows)


def _clean_df(df: pd.DataFrame) -> tuple[pd.DataFrame, int, list[str]]:
    """Replace Inf/NaN and report how many rows were dropped."""
    n_before = len(df)
    df = df.replace([float("inf"), float("-inf")], float("nan"))
    bad = df.isna().any(axis=1)
    reasons = []
    if bad.sum():
        reasons.append(f"{int(bad.sum())} rows contained Inf or NaN and were dropped")
    df = df[~bad].reset_index(drop=True)
    return df, n_before - len(df), reasons


def _predict_model(
    model_name: str,
    df: pd.DataFrame,
    target_fpr: float,
) -> list[dict]:
    k = _best_k()
    prep = _preprocessors[k]
    X = prep.transform(df)

    if model_name == "classical":
        det = _classical.get(k)
    else:
        det = _quantum.get(k)

    if det is None:
        raise HTTPException(status_code=503, detail=f"Model '{model_name}' k={k} not loaded.")

    scores    = det.score(X)
    threshold = _get_threshold(model_name, k, target_fpr)

    return [
        {
            "score":     float(s),
            "flagged":   bool(s >= threshold),
            "threshold": threshold,
            "model":     model_name,
        }
        for s in scores
    ]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    """Liveness check."""
    import qiskit
    import sklearn
    return {
        "status":        "ok",
        "model_version": VERSION,
        "artifacts_loaded": bool(_preprocessors),
        "qiskit_version":   qiskit.__version__,
        "sklearn_version":  sklearn.__version__,
    }


@app.get("/model/info", tags=["Model"])
def model_info():
    """Return the artifact manifest (training config, metrics, limitations)."""
    _require_artifacts()
    manifest = dict(_manifest)
    manifest["limitations"] = (
        "This model was trained on CICIDS2017 Monday (benign-only). "
        "The quantum model runs on a Qiskit Statevector simulator — "
        "no quantum advantage is claimed or observed. "
        "Hyperparameters were tuned on a small labelled sample of DoS "
        "and PortScan attacks from the same dataset. "
        "Performance may degrade on unseen attack families or real traffic."
    )
    return manifest


@app.get("/results", tags=["Results"])
def get_results():
    """Return v1 and v2 summary tables and per-attack recall."""
    output: dict = {}
    for label, path in [("v1_quick", RESULTS_DIR / "v1_quick"), ("v2", RESULTS_DIR / "v2")]:
        entry: dict = {}
        for fname in ("results.csv", "summary.csv", "per_attack_recall.csv"):
            fp = path / fname
            if fp.exists():
                try:
                    records = pd.read_csv(fp).fillna("").to_dict(orient="records")
                    entry[fname.replace(".csv", "")] = records
                except Exception as e:
                    entry[fname.replace(".csv", "")] = {"error": str(e)}
        output[label] = entry
    return JSONResponse(content=_sanitize(output))


@app.get("/demo-sample", tags=["Data"])
def demo_sample():
    """Return a small sample of flows for testing the predict endpoint."""
    sample_path = Path(__file__).resolve().parents[2] / "data" / "demo_sample.csv"
    if not sample_path.exists():
        raise HTTPException(
            status_code=404,
            detail="demo_sample.csv not found. Run `python scripts/make_demo_sample.py`.",
        )
    df = pd.read_csv(sample_path).head(50)
    return {
        "columns": list(df.columns),
        "rows": df.to_dict(orient="records"),
    }


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict_json(req: PredictRequest):
    """Run inference on a JSON list of flows.

    The quantum model is limited to 200 rows per request.
    """
    _require_artifacts()

    n_flows = len(req.flows)
    max_rows = (
        API_MAX_ROWS_QUANTUM
        if req.model in ("quantum", "both")
        else API_MAX_ROWS_CLASSICAL
    )

    if n_flows > max_rows:
        raise HTTPException(
            status_code=422,
            detail=f"Request has {n_flows} rows; maximum is {max_rows} for model='{req.model}'.",
        )

    df = _df_from_records(req.flows)
    df, n_dropped, reasons = _clean_df(df)

    if df.empty:
        raise HTTPException(status_code=422, detail="All rows were dropped (Inf/NaN).")

    models_to_run = (
        ["classical", "quantum"] if req.model == "both" else [req.model]
    )

    all_results: list[PredictResult] = []
    for model_name in models_to_run:
        preds = _predict_model(model_name, df, req.target_fpr)
        for i, p in enumerate(preds):
            all_results.append(
                PredictResult(
                    row_index=i,
                    score=p["score"],
                    flagged=p["flagged"],
                    threshold=p["threshold"],
                    model=p["model"],
                    model_version=VERSION,
                )
            )

    flagged_count = sum(1 for r in all_results if r.flagged)

    return PredictResponse(
        results=all_results,
        summary={
            "rows_scored":    len(df),
            "flagged_count":  flagged_count,
            "flagged_share":  round(flagged_count / len(all_results), 4) if all_results else 0,
        },
        rows_received=n_flows,
        rows_dropped=n_dropped,
        drop_reasons=reasons,
        disclaimer=(
            "The quantum model runs on a Qiskit Statevector simulator. "
            "No quantum speedup or advantage is claimed."
        ),
    )


@app.post("/predict/csv", response_model=PredictResponse, tags=["Inference"])
async def predict_csv(
    file: UploadFile = File(...),
    model: Literal["classical", "quantum", "both"] = Query("both"),
    target_fpr: float = Query(0.05),
):
    """Run inference on an uploaded CSV file.

    The CSV must contain the 11 feature columns (header names are stripped).
    """
    _require_artifacts()

    if target_fpr not in (0.01, 0.05):
        raise HTTPException(status_code=422, detail="target_fpr must be 0.01 or 0.05.")

    contents = await file.read()
    if len(contents) > API_MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Upload exceeds {API_MAX_UPLOAD_BYTES // 1024 // 1024} MB limit.",
        )

    try:
        df_raw = pd.read_csv(io.BytesIO(contents), skipinitialspace=True)
        df_raw.columns = df_raw.columns.str.strip()
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not parse CSV: {e}")

    missing = [c for c in FEATURES if c not in df_raw.columns]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"CSV is missing required columns: {missing}",
        )

    for col in FEATURES:
        df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")

    df, n_dropped, reasons = _clean_df(df_raw[FEATURES])

    max_rows = (
        API_MAX_ROWS_QUANTUM if model in ("quantum", "both") else API_MAX_ROWS_CLASSICAL
    )
    if len(df) > max_rows:
        df = df.iloc[:max_rows]
        reasons.append(f"Truncated to {max_rows} rows (limit for model='{model}').")

    if df.empty:
        raise HTTPException(status_code=422, detail="All rows were dropped (Inf/NaN).")

    models_to_run = ["classical", "quantum"] if model == "both" else [model]
    all_results: list[PredictResult] = []

    for model_name in models_to_run:
        preds = _predict_model(model_name, df, target_fpr)
        for i, p in enumerate(preds):
            all_results.append(
                PredictResult(
                    row_index=i,
                    score=p["score"],
                    flagged=p["flagged"],
                    threshold=p["threshold"],
                    model=p["model"],
                    model_version=VERSION,
                )
            )

    flagged_count = sum(1 for r in all_results if r.flagged)

    return PredictResponse(
        results=all_results,
        summary={
            "rows_scored":   len(df),
            "flagged_count": flagged_count,
            "flagged_share": round(flagged_count / len(all_results), 4) if all_results else 0,
        },
        rows_received=len(df_raw),
        rows_dropped=n_dropped,
        drop_reasons=reasons,
        disclaimer=(
            "The quantum model runs on a Qiskit Statevector simulator. "
            "No quantum speedup or advantage is claimed."
        ),
    )
