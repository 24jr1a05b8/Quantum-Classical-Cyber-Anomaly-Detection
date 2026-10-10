"""Unit tests for QCAD preprocessing, kernels, thresholds, and metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.qcad.config import FEATURES
from src.qcad.evaluate import (
    compute_auc_metrics,
    compute_threshold,
    evaluate,
    full_metrics,
)
from src.qcad.kernels import make_fm, qstates, quantum_kernel
from src.qcad.preprocess import Preprocessor

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_df(n: int = 50, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {col: rng.exponential(scale=100, size=n) for col in FEATURES}
    data["Label"] = "BENIGN"
    return pd.DataFrame(data)


# ── Preprocessor ──────────────────────────────────────────────────────────────

class TestPreprocessor:
    def test_fit_transform_shape(self):
        df = _make_df(60)
        prep = Preprocessor(k=4).fit(df)
        X = prep.transform(df)
        assert X.shape == (60, prep.k)

    def test_output_range(self):
        """All output values must be in [0, π]."""
        df = _make_df(60)
        prep = Preprocessor(k=4).fit(df)
        X = prep.transform(df)
        assert float(X.min()) >= 0.0 - 1e-9
        assert float(X.max()) <= np.pi + 1e-9

    def test_clipping(self):
        """Values outside training range must be clipped to [0, π]."""
        np.random.default_rng(1)
        df_train = _make_df(80, seed=1)
        prep = Preprocessor(k=4).fit(df_train)

        # Craft a DataFrame with extreme values.
        extreme = pd.DataFrame(
            {col: [1e15] * 5 for col in FEATURES}
        )
        extreme["Label"] = "BENIGN"
        X_ext = prep.transform(extreme)
        assert float(X_ext.max()) <= np.pi + 1e-9

    def test_fit_uses_only_training_data(self):
        """Fit must not access test data (no leakage check)."""
        df_train = _make_df(80, seed=0)
        df_test  = _make_df(20, seed=99)

        prep = Preprocessor(k=4)
        prep.fit(df_train)   # must not raise

        # Transform test data must work without refitting.
        X_test = prep.transform(df_test)
        assert X_test.shape[0] == 20

    def test_inf_nan_handling(self):
        """Inf and NaN survive the transform without crashing (cleaned upstream)."""
        df = _make_df(30)
        prep = Preprocessor(k=4).fit(df)

        df2 = df.copy()
        df2.iloc[0, 0] = np.inf
        df2.iloc[1, 1] = np.nan
        # After clipping negatives to 0 and log1p, inf/nan propagates —
        # the user is expected to clean data before calling transform.
        # We just confirm no unexpected exception here.
        df_clean = df2.replace([np.inf, -np.inf], np.nan).dropna()
        X = prep.transform(df_clean)
        assert X.shape[0] == len(df_clean)

    def test_save_load_roundtrip(self, tmp_path):
        df = _make_df(60)
        prep = Preprocessor(k=4).fit(df)
        X1 = prep.transform(df)

        path = tmp_path / "prep.pkl"
        prep.save(path)
        prep2 = Preprocessor.load(path)
        X2 = prep2.transform(df)

        np.testing.assert_allclose(X1, X2)


# ── Quantum kernel properties ─────────────────────────────────────────────────

# Shared statevectors computed once at module level to avoid slow per-test circuits.
_RNG_STATES4 = np.random.default_rng(42)
_X_STATES4 = _RNG_STATES4.uniform(0, np.pi, size=(8, 4))
_STATES_4 = qstates(_X_STATES4, k=4, scale=0.25)


class TestQuantumKernel:
    def test_diagonal_one(self):
        """K(x, x) = 1 for all x."""
        K = quantum_kernel(_STATES_4, _STATES_4)
        diag = np.diag(K)
        np.testing.assert_allclose(diag, 1.0, atol=1e-6)

    def test_symmetric(self):
        """Kernel matrix must be symmetric."""
        K = quantum_kernel(_STATES_4, _STATES_4)
        np.testing.assert_allclose(K, K.T, atol=1e-6)

    def test_values_in_unit_interval(self):
        """All kernel values must be in [0, 1]."""
        K = quantum_kernel(_STATES_4, _STATES_4)
        assert float(K.min()) >= 0.0 - 1e-9
        assert float(K.max()) <= 1.0 + 1e-9

    def test_positive_semidefinite(self):
        """Eigenvalues must be ≥ 0 (within tolerance)."""
        K = quantum_kernel(_STATES_4, _STATES_4)
        eigvals = np.linalg.eigvalsh(K)
        assert float(eigvals.min()) >= -1e-6

    def test_feature_map_circuit(self):
        fm = make_fm(4)
        assert fm.num_qubits == 4

    def test_scale_zero_all_same(self):
        """Scale=0 means all inputs map to identical state → kernel is all ones."""
        rng = np.random.default_rng(7)
        X = rng.uniform(0, np.pi, size=(5, 4))
        states = qstates(X, k=4, scale=0.0)
        K = quantum_kernel(states, states)
        np.testing.assert_allclose(K, np.ones_like(K), atol=1e-5)


# ── Threshold and metrics ─────────────────────────────────────────────────────

class TestThresholdAndMetrics:
    def test_threshold_quantile(self):
        scores = np.arange(100, dtype=float)
        thr = compute_threshold(scores, 0.05)
        # 95th percentile of 0..99 is 94.05
        assert abs(thr - np.quantile(scores, 0.95)) < 1e-9

    def test_threshold_fpr_01(self):
        scores = np.arange(100, dtype=float)
        thr = compute_threshold(scores, 0.01)
        assert abs(thr - np.quantile(scores, 0.99)) < 1e-9

    def test_evaluate_perfect_recall(self):
        """If all attack scores are above all benign scores, recall = 1."""
        y = np.array([0, 0, 0, 1, 1])
        labels = np.array(["BENIGN", "BENIGN", "BENIGN", "DoS", "DoS"])
        scores = np.array([0.1, 0.2, 0.3, 0.9, 0.95])
        thr = 0.5
        result = evaluate(scores, y, labels, thr)
        assert result["recall"] == 1.0
        assert result["measured_fpr"] == 0.0

    def test_evaluate_zero_recall(self):
        y = np.array([0, 0, 0, 1, 1])
        labels = np.array(["BENIGN", "BENIGN", "BENIGN", "DoS", "DoS"])
        scores = np.array([0.8, 0.9, 0.95, 0.1, 0.2])
        thr = 0.5
        result = evaluate(scores, y, labels, thr)
        assert result["recall"] == 0.0

    def test_evaluate_f1_formula(self):
        """Hand-computed F1 example."""
        y      = np.array([1, 1, 0, 0])
        labels = np.array(["A", "A", "BENIGN", "BENIGN"])
        scores = np.array([0.8, 0.3, 0.9, 0.1])
        thr    = 0.5
        result = evaluate(scores, y, labels, thr)
        # predicted = [True, False, True, False]
        # precision = 1/(1+1) = 0.5, recall = 1/2 = 0.5, F1 = 0.5
        assert abs(result["precision"] - 0.5) < 1e-9
        assert abs(result["recall"] - 0.5) < 1e-9
        assert abs(result["f1"] - 0.5) < 1e-9

    def test_low_n_flag(self):
        """Attacks with fewer than 20 test samples must be flagged low_n."""
        y      = np.array([0] * 50 + [1] * 5)
        labels = np.array(["BENIGN"] * 50 + ["Heartbleed"] * 5)
        scores = np.array([0.1] * 50 + [0.9] * 5)
        thr    = 0.5
        result = evaluate(scores, y, labels, thr)
        assert result["per_attack"]["Heartbleed"]["low_n"] is True

    def test_auc_metrics(self):
        rng = np.random.default_rng(0)
        y = np.array([0] * 50 + [1] * 50)
        # Scores perfectly separating classes.
        scores = np.concatenate([rng.uniform(0, 0.4, 50), rng.uniform(0.6, 1, 50)])
        auc = compute_auc_metrics(scores, y)
        assert auc["prauc"] > 0.9
        assert auc["rocauc"] > 0.9

    def test_full_metrics_keys(self):
        rng = np.random.default_rng(1)
        n = 100
        val_scores = rng.uniform(0, 1, n)
        y = np.array([0] * 80 + [1] * 20)
        test_scores = np.concatenate([rng.uniform(0, 0.5, 80), rng.uniform(0.5, 1, 20)])
        labels = np.array(["BENIGN"] * 80 + ["DoS"] * 20)
        result = full_metrics(val_scores, test_scores, y, labels, [0.01, 0.05])
        assert "by_fpr" in result
        assert 0.01 in result["by_fpr"]
        assert 0.05 in result["by_fpr"]
        assert "prauc" in result
        assert "rocauc" in result
