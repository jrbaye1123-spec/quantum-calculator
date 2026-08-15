"""Quantum basis states as numpy column vectors.

Conventions
-----------
- A single-qubit state is a complex 2x1 column vector [a0, a1]^T.
- An n-qubit state is a complex 2^n x 1 column vector.
- Basis index k = q_{n-1} ... q_1 q_0 in binary, i.e. qubit 0 is the
  least-significant bit (the rightmost qubit in the |q_{n-1}...q_0> label).
"""

import numpy as np


def ket0():
    """|0> = [1, 0]^T"""
    return np.array([[1 + 0j], [0 + 0j]])


def ket1():
    """|1> = [0, 1]^T"""
    return np.array([[0 + 0j], [1 + 0j]])


def plus():
    """|+> = (|0> + |1>)/sqrt(2)  (X eigenstate)"""
    return (ket0() + ket1()) / np.sqrt(2)


def minus():
    """|-> = (|0> - |1>)/sqrt(2)  (X eigenstate)"""
    return (ket0() - ket1()) / np.sqrt(2)


def ket_i():
    """|i> = (|0> + i|1>)/sqrt(2)  (Y eigenstate)"""
    return (ket0() + 1j * ket1()) / np.sqrt(2)


def ket_minus_i():
    """|-i> = (|0> - i|1>)/sqrt(2)  (Y eigenstate)"""
    return (ket0() - 1j * ket1()) / np.sqrt(2)


_BASIS = {
    "0": ket0,
    "1": ket1,
    "+": plus,
    "-": minus,
    "i": ket_i,
    "-i": ket_minus_i,
}


def basis_state(name):
    """Return a named single-qubit basis state: '0', '1', '+', '-', 'i', '-i'."""
    key = name.strip()
    if key not in _BASIS:
        raise ValueError(
            f"unknown basis state {name!r}; choose from {sorted(_BASIS)}"
        )
    return _BASIS[key]()


def zero_state(n_qubits):
    """The n-qubit state |0...0> as a 2^n column vector."""
    if n_qubits < 1:
        raise ValueError("n_qubits must be >= 1")
    state = np.zeros((2 ** n_qubits, 1), dtype=complex)
    state[0, 0] = 1.0
    return state


def label(index, n_qubits):
    """Binary bitstring label for a basis index, e.g. label(1, 2) -> '01'."""
    return format(index, f"0{n_qubits}b")
