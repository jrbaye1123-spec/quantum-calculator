"""Measurement of quantum states (Born rule).

P(outcome i) = |<i|psi>|^2.
"""

import numpy as np


def probabilities(state):
    """Return the Born-rule probabilities P(i) = |amplitude_i|^2.

    Returns a real array of length 2^n that sums to 1 (normalised defensively
    to absorb any numerical drift).
    """
    probs = np.abs(np.asarray(state).ravel()) ** 2
    total = probs.sum()
    if total <= 0:
        raise ValueError("state has zero norm; cannot compute probabilities")
    return (probs / total).real


def measure(state, rng=None):
    """Measure all qubits. Returns (outcome_index, collapsed_state)."""
    if rng is None:
        rng = np.random.default_rng()
    probs = probabilities(state)
    idx = int(rng.choice(len(probs), p=probs))
    collapsed = np.zeros((len(probs), 1), dtype=complex)
    collapsed[idx, 0] = 1.0
    return idx, collapsed


def measure_qubit(state, qubit, n_qubits, rng=None):
    """Measure a single qubit. Returns (outcome_bit, collapsed_state).

    The state is projected onto the subspace consistent with the outcome and
    renormalised, as the Born rule prescribes.
    """
    if rng is None:
        rng = np.random.default_rng()
    probs = probabilities(state)
    p1 = sum(p for i, p in enumerate(probs) if (i >> qubit) & 1)
    outcome = 1 if rng.random() < p1 else 0

    collapsed = np.asarray(state, dtype=complex).copy()
    for i in range(len(probs)):
        if ((i >> qubit) & 1) != outcome:
            collapsed[i, 0] = 0.0
    norm = np.linalg.norm(collapsed)
    if norm > 0:
        collapsed = collapsed / norm
    return outcome, collapsed


def sample(state, shots, rng=None):
    """Measure all qubits `shots` times. Returns {bitstring: count}."""
    if rng is None:
        rng = np.random.default_rng()
    probs = probabilities(state)
    n_qubits = len(probs).bit_length() - 1
    idxs = rng.choice(len(probs), size=shots, p=probs)
    counts = {}
    for i in idxs:
        label = format(i, f"0{n_qubits}b")
        counts[label] = counts.get(label, 0) + 1
    return counts
