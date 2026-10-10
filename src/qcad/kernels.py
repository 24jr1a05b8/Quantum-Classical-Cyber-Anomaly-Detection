"""Quantum kernel utilities: feature map, statevectors, fidelity kernel."""

from __future__ import annotations

import logging

import numpy as np
from qiskit.circuit.library import zz_feature_map
from qiskit.quantum_info import Statevector

from .config import ENTANGLEMENT, REPS

logger = logging.getLogger(__name__)


def make_fm(k: int, reps: int = REPS, entanglement: str = ENTANGLEMENT):
    """Build a ZZFeatureMap for *k* qubits.

    Parameters
    ----------
    k:
        Number of qubits (= PCA components).
    reps:
        Number of circuit repetitions.
    entanglement:
        Entanglement pattern; defaults to ``config.ENTANGLEMENT`` ('linear').

    Returns
    -------
    qiskit.circuit.QuantumCircuit
    """
    return zz_feature_map(
        feature_dimension=k,
        reps=reps,
        entanglement=entanglement,
    )


def qstates(
    X: np.ndarray,
    k: int,
    scale: float = 1.0,
    reps: int = REPS,
    entanglement: str = ENTANGLEMENT,
) -> np.ndarray:
    """Encode feature rows as quantum statevectors.

    Parameters
    ----------
    X:
        Array of shape (n_samples, k), values already in [0, π].
    k:
        Number of qubits.
    scale:
        Multiplicative scale applied to features before encoding.
    reps, entanglement:
        Passed to :func:`make_fm`.

    Returns
    -------
    np.ndarray of complex128, shape (n_samples, 2**k).
    """
    circuit = make_fm(k, reps=reps, entanglement=entanglement)
    parameters = list(circuit.parameters)
    states = []

    for row in X:
        values = {
            param: float(scale * val)
            for param, val in zip(parameters, row)
        }
        bound = circuit.assign_parameters(values)
        states.append(Statevector(bound).data)

    return np.asarray(states, dtype=np.complex128)


def quantum_kernel(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Fidelity kernel: K[i, j] = |<ψ(A[i])|ψ(B[j])>|².

    Parameters
    ----------
    A:
        Statevector array of shape (m, 2**k).
    B:
        Statevector array of shape (n, 2**k).

    Returns
    -------
    np.ndarray of shape (m, n), values in [0, 1].
    """
    return np.abs(A.conj() @ B.T) ** 2
