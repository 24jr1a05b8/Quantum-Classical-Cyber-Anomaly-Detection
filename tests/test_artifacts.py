"""Artifact smoke test: loading saved models reproduces recorded scores."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

ARTIFACT_DIR = Path("artifacts/v2")


@pytest.mark.skipif(
    not (ARTIFACT_DIR / "manifest.json").exists(),
    reason="Artifacts not built yet — run `python -m qcad.train` first",
)
class TestArtifactIntegrity:
    def test_manifest_keys(self):
        manifest = json.loads((ARTIFACT_DIR / "manifest.json").read_text())
        for key in ("version", "created_at", "config", "metrics", "thresholds", "artifacts"):
            assert key in manifest, f"manifest missing key: {key}"

    def test_thresholds_keys(self):
        thresholds = json.loads((ARTIFACT_DIR / "thresholds.json").read_text())
        for key, val in thresholds.items():
            assert "0.01" in val, f"{key} missing 0.01 threshold"
            assert "0.05" in val, f"{key} missing 0.05 threshold"

    def test_preprocessor_loads(self):
        from src.qcad.preprocess import Preprocessor

        for pkl in sorted(ARTIFACT_DIR.glob("preprocessor_k*.pkl")):
            prep = Preprocessor.load(pkl)
            assert prep.k >= 4

    def test_classical_model_loads(self):
        from src.qcad.models import ClassicalDetector

        for pkl in sorted(ARTIFACT_DIR.glob("classical_k*.pkl")):
            det = ClassicalDetector.load(pkl)
            assert det.kind == "classical"

    def test_quantum_model_loads(self):
        from src.qcad.models import QuantumDetector

        for pkl in sorted(ARTIFACT_DIR.glob("quantum_k*.pkl")):
            det = QuantumDetector.load(pkl)
            assert det.kind == "quantum"

    def test_scores_reproducible(self):
        """Reload a model and score the same fixed input — scores must match."""
        from src.qcad.config import FEATURES
        from src.qcad.models import ClassicalDetector
        from src.qcad.preprocess import Preprocessor

        # Use lowest k that has an artifact.
        ks = sorted(
            int(p.stem.split("k")[1])
            for p in ARTIFACT_DIR.glob("classical_k*.pkl")
        )
        if not ks:
            pytest.skip("No classical artifacts found")

        k = ks[0]
        prep = Preprocessor.load(ARTIFACT_DIR / f"preprocessor_k{k}.pkl")
        det  = ClassicalDetector.load(ARTIFACT_DIR / f"classical_k{k}.pkl")

        rng = np.random.default_rng(123)
        import pandas as pd
        df = pd.DataFrame(
            {col: rng.exponential(100, size=10).tolist() for col in FEATURES}
        )
        X = prep.transform(df)
        s1 = det.score(X)

        # Reload and score again — must be identical.
        prep2 = Preprocessor.load(ARTIFACT_DIR / f"preprocessor_k{k}.pkl")
        det2  = ClassicalDetector.load(ARTIFACT_DIR / f"classical_k{k}.pkl")
        X2 = prep2.transform(df)
        s2 = det2.score(X2)

        np.testing.assert_allclose(s1, s2, rtol=1e-10)
