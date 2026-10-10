"""Central configuration for QCAD.  All settings live here."""

from __future__ import annotations

from pathlib import Path

# ── Repository root ───────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]

# ── Data files ────────────────────────────────────────────────────────────────
DATA_FILES = {
    "mon": ROOT / "Monday-WorkingHours.pcap_ISCX.csv",
    "wed": ROOT / "Wednesday-workingHours.pcap_ISCX.csv",
    "fri": ROOT / "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
}

# ── Features (exactly 11 as specified) ───────────────────────────────────────
FEATURES = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Packet Length Mean",
    "Packet Length Std",
    "Flow IAT Mean",
    "Flow IAT Std",
]

# ── Model hyperparameters ─────────────────────────────────────────────────────
NU = 0.05
REPS = 2
CORR_THRESHOLD = 0.95           # drop one feature from a pair above this
ENTANGLEMENT: str = "linear"    # default per spec (was 'full' in v1)

# ── Gamma search grids ────────────────────────────────────────────────────────
GAMMA_GRID = {
    "classical": [0.05, 0.1, 0.5, 1.0, 2.0],
    "quantum":   [0.05, 0.1, 0.25, 0.5],     # narrower range per known issue
}

# ── Target false-positive rates ───────────────────────────────────────────────
TARGET_FPRS = [0.01, 0.05]

# ── Experiment sizes ──────────────────────────────────────────────────────────
QUICK_N_TRAIN  = 300
QUICK_N_VAL    = 1_000
QUICK_KS       = (4, 6)
QUICK_SEEDS    = [0]
QUICK_PER_TEST = 20
QUICK_PER_TUNE = 8

FULL_N_TRAIN  = 1_000
FULL_N_VAL    = 1_000
FULL_KS       = (4, 6, 8)
FULL_SEEDS    = [0, 1, 2]
FULL_PER_TEST = 60
FULL_PER_TUNE = 20

# ── Artifacts ─────────────────────────────────────────────────────────────────
ARTIFACTS_DIR = ROOT / "artifacts"
VERSION       = "v2"

# ── Results ───────────────────────────────────────────────────────────────────
RESULTS_DIR      = ROOT / "results"
V1_RESULTS_DIR   = RESULTS_DIR / "v1_quick"
V2_RESULTS_DIR   = RESULTS_DIR / "v2"

# ── API ───────────────────────────────────────────────────────────────────────
API_MAX_ROWS_QUANTUM   = 200
API_MAX_ROWS_CLASSICAL = 5_000
API_MAX_UPLOAD_BYTES   = 10 * 1024 * 1024   # 10 MB

# ── Random seed for data shuffling ───────────────────────────────────────────
DATA_SEED = 0
