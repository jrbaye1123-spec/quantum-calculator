import numpy as np
import pytest

import gates
import measurement
import states


def test_probabilities_plus():
    assert np.allclose(measurement.probabilities(states.plus()), [0.5, 0.5])


def test_probabilities_ket0():
    assert np.allclose(measurement.probabilities(states.ket0()), [1.0, 0.0])


def test_probabilities_sum_to_one():
    assert np.isclose(measurement.probabilities(states.plus()).sum(), 1.0)


def test_probabilities_bell():
    s = gates.apply_cnot(gates.apply(states.zero_state(2), gates.H, 1, 2), 1, 0, 2)
    assert np.allclose(measurement.probabilities(s), [0.5, 0.0, 0.0, 0.5])


def test_probabilities_zero_norm_raises():
    with pytest.raises(ValueError):
        measurement.probabilities(np.zeros((2, 1)))


def test_measure_outcome_in_basis():
    rng = np.random.default_rng(0)
    idx, collapsed = measurement.measure(states.plus(), rng)
    assert idx in (0, 1)
    assert np.isclose(abs(collapsed[idx, 0]), 1.0)
    assert np.count_nonzero(collapsed) == 1


def test_measure_deterministic_ket0():
    rng = np.random.default_rng(0)
    for _ in range(10):
        idx, _ = measurement.measure(states.ket0(), rng)
        assert idx == 0


def test_measure_qubit_zero_state():
    rng = np.random.default_rng(0)
    outcome, collapsed = measurement.measure_qubit(states.zero_state(2), 0, 2, rng)
    assert outcome == 0
    assert np.allclose(collapsed, states.zero_state(2))


def test_measure_qubit_projects_normalized():
    rng = np.random.default_rng(0)
    outcome, collapsed = measurement.measure_qubit(states.plus(), 0, 1, rng)
    assert outcome in (0, 1)
    assert np.isclose(np.linalg.norm(collapsed), 1.0)


def test_sample_deterministic():
    # X|0> = |1>, so every shot must be '1'
    s = gates.X @ states.ket0()
    counts = measurement.sample(s, 100, np.random.default_rng(0))
    assert counts == {"1": 100}


def test_sample_seeded_statistics():
    # |+> should split ~50/50; fixed seed + loose tolerance
    counts = measurement.sample(states.plus(), 20000, np.random.default_rng(1))
    freq0 = counts.get("0", 0) / 20000
    assert 0.45 <= freq0 <= 0.55
