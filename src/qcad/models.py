"""Classical and Quantum anomaly detection models."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
from sklearn.svm import OneClassSVM

from .config import ENTANGLEMENT, NU, REPS
from .kernels import qstates, quantum_kernel

logger = logging.getLogger(__name__)


class ClassicalDetector:
    """One-Class SVM with an RBF kernel.

    Scores are negated decision function values so that **higher score →
    more anomalous**.

    Parameters
    ----------
    gamma:
        RBF bandwidth parameter.
    nu:
        OneClassSVM nu parameter (upper bound on the training error fraction).
    """

    kind = "classical"

    def __init__(self, gamma: float, nu: float = NU) -> None:
        self.gamma = gamma
        self.nu = nu
        self._svm: OneClassSVM | None = None

    def fit(self, X_train: np.ndarray) -> ClassicalDetector:
        """Fit on *benign* training data.

        Parameters
        ----------
        X_train:
            Array of shape (n_train, k) in [0, π].
        """
        self._svm = OneClassSVM(kernel="rbf", gamma=self.gamma, nu=self.nu)
        self._svm.fit(X_train)
        logger.info("ClassicalDetector fitted (gamma=%.4f, n_train=%d)", self.gamma, len(X_train))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores (higher = more anomalous).

        Parameters
        ----------
        X:
            Array of shape (n_samples, k).
        """
        self._check_fitted()
        return -self._svm.decision_function(X).ravel()

    def save(self, path: Path | str) -> None:
        """Pickle the model."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("ClassicalDetector saved to %s", path)

    @classmethod
    def load(cls, path: Path | str) -> ClassicalDetector:
        """Load a previously saved model."""
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        logger.info("ClassicalDetector loaded from %s", path)
        return obj

    def _check_fitted(self) -> None:
        if self._svm is None:
            raise RuntimeError("ClassicalDetector has not been fitted yet.")

    def info(self) -> dict:
        """Serialisable summary."""
        return {"kind": self.kind, "gamma": self.gamma, "nu": self.nu}


class QuantumDetector:
    """One-Class SVM with a precomputed fidelity kernel.

    Training statevectors are stored at fit-time so inference only needs
    **one circuit evaluation per new sample** (dot-product with stored states).

    Parameters
    ----------
    scale:
        Multiplicative scale factor applied before encoding into the
        ZZFeatureMap.  This is the 'gamma' analogue for the quantum kernel.
    k:
        Number of qubits (= PCA components).
    nu:
        OneClassSVM nu parameter.
    reps:
        Number of ZZFeatureMap repetitions.
    entanglement:
        ZZFeatureMap entanglement pattern.
    """

    kind = "quantum"

    def __init__(
        self,
        scale: float,
        k: int,
        nu: float = NU,
        reps: int = REPS,
        entanglement: str = ENTANGLEMENT,
    ) -> None:
        self.scale = scale
        self.k = k
        self.nu = nu
        self.reps = reps
        self.entanglement = entanglement

        self._train_states: np.ndarray | None = None
        self._svm: OneClassSVM | None = None

    def fit(self, X_train: np.ndarray) -> QuantumDetector:
        """Compute training statevectors and fit precomputed-kernel SVM.

        Parameters
        ----------
        X_train:
            Array of shape (n_train, k), values in [0, π].
        """
        logger.info(
            "QuantumDetector: computing %d training statevectors (k=%d, scale=%.4f) …",
            len(X_train), self.k, self.scale,
        )
        self._train_states = qstates(
            X_train,
            self.k,
            scale=self.scale,
            reps=self.reps,
            entanglement=self.entanglement,
        )
        K_train = quantum_kernel(self._train_states, self._train_states)

        self._svm = OneClassSVM(kernel="precomputed", nu=self.nu)
        self._svm.fit(K_train)
        logger.info("QuantumDetector fitted (scale=%.4f, n_train=%d)", self.scale, len(X_train))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores for new samples (higher = more anomalous).

        Inference cost: one quantum circuit evaluation per row.

        Parameters
        ----------
        X:
            Array of shape (n_samples, k), values in [0, π].
        """
        self._check_fitted()
        test_states = qstates(
            X,
            self.k,
            scale=self.scale,
            reps=self.reps,
            entanglement=self.entanglement,
        )
        K_test = quantum_kernel(test_states, self._train_states)
        return -self._svm.decision_function(K_test).ravel()

    def save(self, path: Path | str) -> None:
        """Pickle the model (includes stored training statevectors)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("QuantumDetector saved to %s", path)

    @classmethod
    def load(cls, path: Path | str) -> QuantumDetector:
        """Load a previously saved model."""
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        logger.info("QuantumDetector loaded from %s", path)
        return obj

    def _check_fitted(self) -> None:
        if self._svm is None:
            raise RuntimeError("QuantumDetector has not been fitted yet.")

    def info(self) -> dict:
        """Serialisable summary."""
        return {
            "kind": self.kind,
            "scale": self.scale,
            "k": self.k,
            "nu": self.nu,
            "reps": self.reps,
            "entanglement": self.entanglement,
        }
