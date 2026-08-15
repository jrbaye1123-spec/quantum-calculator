import numpy as np
import pytest

import gates
import states


def test_hadamard_superposition():
    assert np.allclose(gates.H @ states.ket0(), states.plus())


def test_hadamard_self_inverse():
    assert np.allclose(gates.H @ states.plus(), states.ket0())


def test_x_flip():
    assert np.allclose(gates.X @ states.ket0(), states.ket1())
    assert np.allclose(gates.X @ states.ket1(), states.ket0())


def test_y():
    assert np.allclose(gates.Y @ states.ket0(), 1j * states.ket1())


def test_z_phase():
    assert np.allclose(gates.Z @ states.plus(), states.minus())


def test_s_phase():
    assert np.allclose(gates.S @ states.plus(), states.ket_i())


def test_t():
    assert np.allclose(gates.T @ states.ket0(), states.ket0())


def test_phase_zero_is_identity():
    assert np.allclose(gates.phase(0.0), gates.I)


def test_phase_pi_is_z():
    assert np.allclose(gates.phase(np.pi), gates.Z)


def test_rotations_theta_zero_identity():
    for rotation in (gates.rotation_x, gates.rotation_y, gates.rotation_z):
        assert np.allclose(rotation(0.0), gates.I), rotation.__name__


def test_cnot_standard():
    # control=1 (MSB), target=0 (LSB): flips q0 when q1 == 1
    cnot = gates.cnot(1, 0)
    assert np.allclose(cnot @ np.array([[1, 0, 0, 0]]).T, np.array([[1, 0, 0, 0]]).T)
    assert np.allclose(cnot @ np.array([[0, 0, 1, 0]]).T, np.array([[0, 0, 0, 1]]).T)


def test_cnot_reversed():
    # control=0 (LSB), target=1 (MSB): flips q1 when q0 == 1
    cnot = gates.cnot(0, 1)
    assert np.allclose(cnot @ np.array([[0, 1, 0, 0]]).T, np.array([[0, 0, 0, 1]]).T)
    assert np.allclose(cnot @ np.array([[1, 0, 0, 0]]).T, np.array([[1, 0, 0, 0]]).T)


def test_cnot_same_control_target_raises():
    with pytest.raises(ValueError):
        gates.cnot(0, 0)


def test_cnot_out_of_range_raises():
    with pytest.raises(ValueError):
        gates.cnot(0, 2)


def test_bell_state():
    s = gates.apply(states.zero_state(2), gates.H, 1, 2)
    s = gates.apply_cnot(s, 1, 0, 2)
    bell = np.array([[1, 0, 0, 1]], dtype=complex).T / np.sqrt(2)
    assert np.allclose(s, bell)


def test_apply_gate_to_qubit0():
    # X on qubit 0 of |00> -> |01>
    s = gates.apply(states.zero_state(2), gates.X, 0, 2)
    assert np.allclose(s, np.array([[0, 1, 0, 0]], dtype=complex).T)


def test_apply_gate_to_qubit1():
    # X on qubit 1 of |00> -> |10>
    s = gates.apply(states.zero_state(2), gates.X, 1, 2)
    assert np.allclose(s, np.array([[0, 0, 1, 0]], dtype=complex).T)


def test_cnot_unitary():
    assert np.allclose(gates.CNOT @ gates.CNOT.conj().T, np.eye(4))


def test_hadamard_unitary():
    assert np.allclose(gates.H @ gates.H.conj().T, np.eye(2))


def test_norm_preserved():
    assert np.isclose(np.linalg.norm(gates.H @ states.ket0()), 1.0)
    bell = np.array([[1, 0, 0, 1]], dtype=complex).T / np.sqrt(2)
    assert np.isclose(np.linalg.norm(bell), 1.0)


def _ket3(q2, q1, q0):
    s = states.zero_state(3)
    if q2:
        s = gates.apply(s, gates.X, 2, 3)
    if q1:
        s = gates.apply(s, gates.X, 1, 3)
    if q0:
        s = gates.apply(s, gates.X, 0, 3)
    return s


def test_cnot_nonadjacent_control_high():
    # CNOT(2,0): flip q0 when q2 == 1
    assert np.allclose(gates.apply_cnot(_ket3(1, 0, 0), 2, 0, 3), _ket3(1, 0, 1))
    assert np.allclose(gates.apply_cnot(_ket3(0, 1, 0), 2, 0, 3), _ket3(0, 1, 0))


def test_cnot_nonadjacent_control_low():
    # CNOT(0,2): flip q2 when q0 == 1
    assert np.allclose(gates.apply_cnot(_ket3(0, 0, 1), 0, 2, 3), _ket3(1, 0, 1))
    assert np.allclose(gates.apply_cnot(_ket3(0, 1, 0), 0, 2, 3), _ket3(0, 1, 0))


def test_cnot_adjacent_three_qubits():
    # CNOT(2,1): flip q1 when q2 == 1
    assert np.allclose(gates.apply_cnot(_ket3(1, 1, 0), 2, 1, 3), _ket3(1, 0, 0))


def test_ghz_state():
    s = gates.apply(states.zero_state(3), gates.H, 0, 3)
    s = gates.apply_cnot(s, 0, 1, 3)
    s = gates.apply_cnot(s, 0, 2, 3)
    ghz = np.zeros((8, 1), dtype=complex)
    ghz[0, 0] = 1 / np.sqrt(2)
    ghz[7, 0] = 1 / np.sqrt(2)
    assert np.allclose(s, ghz)


def test_apply_swap():
    # SWAP(0,1) on |01> -> |10>
    psi01 = np.array([[0, 1, 0, 0]], dtype=complex).T
    psi10 = np.array([[0, 0, 1, 0]], dtype=complex).T
    assert np.allclose(gates.apply_swap(psi01, 0, 1, 2), psi10)


def test_apply_swap_nonadjacent_raises():
    with pytest.raises(ValueError):
        gates.apply_swap(states.zero_state(3), 0, 2, 3)


def test_cnot_same_qubit_raises_nq():
    with pytest.raises(ValueError):
        gates.apply_cnot(states.zero_state(3), 1, 1, 3)


def test_cnot_out_of_range_raises_nq():
    with pytest.raises(ValueError):
        gates.apply_cnot(states.zero_state(3), 0, 3, 3)
