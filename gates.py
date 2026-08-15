"""Quantum gates as numpy matrices, plus application helpers.

Conventions
-----------
- Single-qubit gates are 2x2 complex matrices acting as G|psi>.
- Qubit 0 is the least-significant bit (rightmost in |q_{n-1}...q_0>).
- The 2-qubit register is ordered |q1 q0> (index = q1*2 + q0).
- CNOT on arbitrary n-qubit registers is applied via SWAP routing
  (apply_cnot); SWAP and the 4x4 CNOT matrices act on adjacent pairs.
"""

import numpy as np

# --- Pauli / single-qubit Clifford gates ------------------------------------
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)          # Pauli-X (NOT)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)       # Pauli-Y
Z = np.array([[1, 0], [0, -1]], dtype=complex)         # Pauli-Z
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)  # Hadamard

# --- Phase gates -------------------------------------------------------------
S = np.array([[1, 0], [0, 1j]], dtype=complex)         # pi/2 phase
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)  # pi/4 phase


def phase(phi):
    """Arbitrary phase gate P(phi) = diag(1, e^{i phi})."""
    return np.array([[1, 0], [0, np.exp(1j * phi)]], dtype=complex)


# --- Rotation gates ----------------------------------------------------------
def rotation_x(theta):
    """Rx(theta) = exp(-i theta X / 2)."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def rotation_y(theta):
    """Ry(theta) = exp(-i theta Y / 2)."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rotation_z(theta):
    """Rz(theta) = exp(-i theta Z / 2)."""
    return np.array(
        [[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]],
        dtype=complex,
    )


# --- CNOT (two-qubit) --------------------------------------------------------
# Standard CNOT: control = qubit 1 (MSB), target = qubit 0 (LSB).
CNOT = np.array(
    [[1, 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, 0, 1],
     [0, 0, 1, 0]],
    dtype=complex,
)

# CNOT with control = qubit 0 (LSB), target = qubit 1 (MSB).
CNOT_REVERSED = np.array(
    [[1, 0, 0, 0],
     [0, 0, 0, 1],
     [0, 0, 1, 0],
     [0, 1, 0, 0]],
    dtype=complex,
)

# SWAP gate in |q_high q_low> ordering: swaps |01> <-> |10>.
SWAP = np.array(
    [[1, 0, 0, 0],
     [0, 0, 1, 0],
     [0, 1, 0, 0],
     [0, 0, 0, 1]],
    dtype=complex,
)


def cnot(control, target):
    """Return the 4x4 CNOT matrix for a 2-qubit register ordered |q1 q0>.

    control, target in {0, 1} and must differ.
    """
    if control not in (0, 1) or target not in (0, 1):
        raise ValueError("control and target must be 0 or 1")
    if control == target:
        raise ValueError("control and target must differ")
    if (control, target) == (1, 0):
        return CNOT
    return CNOT_REVERSED


# --- Application helpers -----------------------------------------------------
def _apply_adjacent_two_qubit(state, gate, low, n_qubits):
    """Apply a 4x4 gate to adjacent qubits (low, low + 1).

    `gate` is in |q_{low+1} q_{low}> basis ordering (index = high*2 + low).
    """
    if low < 0 or low + 1 >= n_qubits:
        raise ValueError("qubits must be adjacent within the register")
    ops = [I] * n_qubits
    ops[n_qubits - 2 - low:n_qubits - low] = [gate]
    full = ops[0]
    for op in ops[1:]:
        full = np.kron(full, op)
    return full @ state


def apply(state, gate, target, n_qubits):
    """Apply a single-qubit gate to `target` of an n-qubit state.

    Builds the full 2^n operator (identity on all other qubits) and returns
    the new state vector.
    """
    ops = [I] * n_qubits
    ops[n_qubits - 1 - target] = gate
    full = ops[0]
    for op in ops[1:]:
        full = np.kron(full, op)
    return full @ state


def apply_swap(state, q0, q1, n_qubits):
    """Swap adjacent qubits q0, q1 via the SWAP gate."""
    if abs(q0 - q1) != 1:
        raise ValueError("SWAP is applied to adjacent qubits only")
    return _apply_adjacent_two_qubit(state, SWAP, min(q0, q1), n_qubits)


def apply_cnot(state, control, target, n_qubits):
    """Apply CNOT with arbitrary control/target qubits via SWAP routing.

    Moves the control adjacent to the target with SWAPs, applies the local
    2-qubit CNOT, then undoes the SWAPs.
    """
    if n_qubits < 2:
        raise ValueError("CNOT requires at least 2 qubits")
    if not (0 <= control < n_qubits and 0 <= target < n_qubits):
        raise ValueError("control/target out of range")
    if control == target:
        raise ValueError("control and target must differ")

    s = np.asarray(state, dtype=complex)
    swaps = []
    c = control

    # Route the control qubit adjacent to the target.
    if c < target:
        while c < target - 1:
            swaps.append((c, c + 1))
            c += 1
    else:
        while c > target + 1:
            swaps.append((c - 1, c))
            c -= 1

    for a, b in swaps:
        s = _apply_adjacent_two_qubit(s, SWAP, min(a, b), n_qubits)

    if c == target - 1:          # control is now the lower of the pair
        gate, low = CNOT_REVERSED, c
    else:                        # control is now the higher of the pair
        gate, low = CNOT, target

    s = _apply_adjacent_two_qubit(s, gate, low, n_qubits)

    for a, b in reversed(swaps):
        s = _apply_adjacent_two_qubit(s, SWAP, min(a, b), n_qubits)

    return s


# Named single-qubit gates available to the CLI / circuits.
GATE_TABLE = {
    "I": I,
    "X": X,
    "Y": Y,
    "Z": Z,
    "H": H,
    "S": S,
    "T": T,
}
