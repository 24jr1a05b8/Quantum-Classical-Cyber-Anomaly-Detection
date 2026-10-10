"""Preprocessing pipeline: log1p → StandardScaler → correlation filter → PCA → MinMaxScaler."""

from __future__ import annotations

import logging
import pickle
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .config import CORR_THRESHOLD, FEATURES

logger = logging.getLogger(__name__)


class Preprocessor:
    """Fit-once, transform-many preprocessing pipeline.

    Steps (all fitted on benign training data only):
    1. log1p with negative clipping to 0.
    2. StandardScaler.
    3. Correlation filter (drop one of a pair with |r| > threshold).
    4. PCA to *k* components.
    5. MinMaxScaler to [0, π] with clipping.

    Parameters
    ----------
    k:
        Number of PCA components (= qubits for the quantum model).
    corr_threshold:
        Absolute Pearson correlation above which one feature is dropped.
    features:
        Ordered list of feature column names (default: ``config.FEATURES``).
    """

    def __init__(
        self,
        k: int,
        corr_threshold: float = CORR_THRESHOLD,
        features: Sequence[str] | None = None,
    ) -> None:
        self.k = k
        self.corr_threshold = corr_threshold
        self.features: list[str] = list(features or FEATURES)

        self._std_scaler: StandardScaler | None = None
        self._kept_indices: list[int] | None = None
        self._pca: PCA | None = None
        self._minmax: MinMaxScaler | None = None

    # ── Fitting ──────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> Preprocessor:
        """Fit all pipeline steps on *benign training data only*.

        Parameters
        ----------
        df:
            DataFrame containing the feature columns.

        Returns
        -------
        self
        """
        X_raw = self._raw(df)

        self._std_scaler = StandardScaler().fit(X_raw)
        Z = self._std_scaler.transform(X_raw)

        # ── correlation filter ────────────────────────────────────────────
        corr = np.abs(np.corrcoef(Z.T))
        corr = np.nan_to_num(corr, nan=0.0)
        keep: list[int] = []
        for i in range(len(self.features)):
            if all(corr[i, j] <= self.corr_threshold for j in keep):
                keep.append(i)
        self._kept_indices = keep
        logger.info(
            "Correlation filter: kept %d of %d features", len(keep), len(self.features)
        )

        k = min(self.k, len(keep))
        if k < self.k:
            logger.warning(
                "Requested k=%d but only %d features passed correlation filter; "
                "using k=%d",
                self.k, len(keep), k,
            )
            self.k = k

        self._pca = PCA(n_components=self.k).fit(Z[:, keep])

        X_pca = self._pca.transform(Z[:, keep])
        self._minmax = MinMaxScaler(feature_range=(0, np.pi)).fit(X_pca)

        return self

    # ── Transform ─────────────────────────────────────────────────────────────

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Apply the fitted pipeline to *new* data.

        Parameters
        ----------
        df:
            DataFrame containing the feature columns.

        Returns
        -------
        np.ndarray of shape (n_samples, k), values in [0, π].
        """
        self._check_fitted()
        X_raw = self._raw(df)
        Z = self._std_scaler.transform(X_raw)
        X_pca = self._pca.transform(Z[:, self._kept_indices])
        return np.clip(self._minmax.transform(X_pca), 0, np.pi)

    # ── Persist ───────────────────────────────────────────────────────────────

    def save(self, path: Path | str) -> None:
        """Pickle the fitted preprocessor to *path*."""
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Preprocessor saved to %s", path)

    @classmethod
    def load(cls, path: Path | str) -> Preprocessor:
        """Load a previously saved preprocessor."""
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected Preprocessor, got {type(obj)}")
        logger.info("Preprocessor loaded from %s", path)
        return obj

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _raw(self, df: pd.DataFrame) -> np.ndarray:
        """log1p with negative clipping."""
        return np.log1p(df[self.features].clip(lower=0).to_numpy())

    def _check_fitted(self) -> None:
        if self._std_scaler is None:
            raise RuntimeError("Preprocessor has not been fitted yet.")

    # ── Info ──────────────────────────────────────────────────────────────────

    def info(self) -> dict:
        """Return a serialisable summary of the fitted pipeline."""
        self._check_fitted()
        return {
            "k": self.k,
            "corr_threshold": self.corr_threshold,
            "kept_features": [self.features[i] for i in self._kept_indices],
            "pca_explained_variance_ratio": self._pca.explained_variance_ratio_.tolist(),
            "minmax_data_min": self._minmax.data_min_.tolist(),
            "minmax_data_max": self._minmax.data_max_.tolist(),
        }
