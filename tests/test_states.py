import numpy as np
import pytest

import states


def test_ket0():
    assert np.array_equal(states.ket0(), np.array([[1 + 0j], [0 + 0j]]))


def test_ket1():
    assert np.array_equal(states.ket1(), np.array([[0 + 0j], [1 + 0j]]))


def test_plus():
    assert np.allclose(states.plus(), np.array([[1], [1]]) / np.sqrt(2))


def test_minus():
    assert np.allclose(states.minus(), np.array([[1], [-1]]) / np.sqrt(2))


def test_ket_i():
    assert np.allclose(states.ket_i(), np.array([[1], [1j]]) / np.sqrt(2))


def test_ket_minus_i():
    assert np.allclose(states.ket_minus_i(), np.array([[1], [-1j]]) / np.sqrt(2))


def test_basis_states_normalized():
    for name in ("0", "1", "+", "-", "i", "-i"):
        assert np.isclose(np.linalg.norm(states.basis_state(name)), 1.0), name


def test_basis_state_lookup():
    assert np.allclose(states.basis_state("+"), states.plus())
    assert np.allclose(states.basis_state("-"), states.minus())


def test_basis_state_unknown_raises():
    with pytest.raises(ValueError):
        states.basis_state("?")


def test_zero_state():
    s = states.zero_state(3)
    assert s.shape == (8, 1)
    assert s[0, 0] == 1.0
    assert np.count_nonzero(s) == 1


def test_label():
    assert states.label(1, 2) == "01"
    assert states.label(0, 3) == "000"
    assert states.label(7, 3) == "111"
