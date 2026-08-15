# Quantum Physics Calculator

A small quantum physics project built in tiers. Each tier is a working,
verified module that the next tier builds on.

## Tier 1 — Formula calculator (quantum_calc.py)

Numbers in, known physics equations applied, numbers out. No dependencies.

| Command               | Equation                          | Inputs                       |
|-----------------------|-----------------------------------|------------------------------|
| photon-wavelength     | E = h·c / λ                       | λ (nm)                       |
| photon-frequency      | E = h·f                           | f (Hz)                       |
| de-broglie            | λ = h / (m·v)                     | m (kg), v (m/s)              |
| hydrogen-level        | Eₙ = −13.606 eV / n²              | n (integer ≥ 1)              |
| hydrogen-transition   | E = E(nᵢ) − E(n_f)                | nᵢ, n_f                      |
| uncertainty           | Δp_min = ħ / (2·Δx)               | Δx (m)                       |
| photoelectric         | K_max = h·f − φ                   | f (Hz), φ (eV)               |
| harmonic              | Eₙ = (n + ½)·ħ·ω                  | n (≥ 0), ω (rad/s)           |

Usage:

    python3 quantum_calc.py                  # interactive menu
    python3 quantum_calc.py demo             # worked examples
    python3 quantum_calc.py photon-wavelength 500

## Tier 2 — Qubit state-vector simulator (qubit.py)

State vector → gates → new state vector → measurement probabilities.
Requires numpy (install into the project venv).

Modules:

    states.py        basis states |0> |1> |+> |-> |i> |-i>, zero_state
    gates.py         I, X, Y, Z, H, S, T, phase(φ), rotations, CNOT (n-qubit,
                     via SWAP routing), SWAP, apply
    measurement.py   probabilities, measure, measure_qubit, sample
    qubit.py         CLI (demo / state / run / gates)

Convention: qubit 0 is the least-significant bit (rightmost in |q1 q0>).

Usage:

    .venv/bin/python qubit.py demo                 # worked examples
    .venv/bin/python qubit.py state +              # print a basis state
    .venv/bin/python qubit.py gates                # list available gates
    .venv/bin/python qubit.py run 2 "H:1,CNOT:1:0" # Bell state, sample it

Circuit DSL (comma-separated, applied left to right):

    H            Hadamard on qubit 0
    X:1          Pauli-X on qubit 1
    CNOT:1:0     CNOT, control=1, target=0 (any qubits, via SWAP routing)
    SWAP:0:1     SWAP adjacent qubits

Example (Bell state):

    .venv/bin/python qubit.py run 2 "H:1,CNOT:1:0" --shots 4000
    # |00> ~50%, |11> ~50%

## Tier 3 — 1D Schrödinger solver (schrodinger.py)

Solves the time-independent Schrödinger equation for bound states by
discretising position and diagonalising the Hamiltonian (finite differences +
scipy.sparse.linalg.eigsh). Requires scipy.

Potentials: infinite_well, harmonic_oscillator(omega), finite_well(depth, width)

Usage:

    .venv/bin/python schrodinger.py demo
    .venv/bin/python schrodinger.py well 1.0 --states 5
    .venv/bin/python schrodinger.py harmonic 1.0 --states 5
    .venv/bin/python schrodinger.py finite-well 10.0 2.0

Plot wavefunctions (optional, requires matplotlib):

    .venv/bin/python schrodinger.py harmonic 1.0 --states 4 --plot --output harmonic.png

The CLI uses natural units (hbar = m = 1). Analytical references for the
infinite well (E_n = n^2 pi^2 / 2L^2) and harmonic oscillator (E_n = n + 1/2)
are provided and checked in tests/test_schrodinger.py.

## Setup

    cd /home/nakamichi/quantum-calc
    python3 -m venv .venv
    .venv/bin/pip install numpy
    .venv/bin/pip install scipy
    .venv/bin/pip install pytest
    .venv/bin/pip install matplotlib   # optional, for wavefunction plots

## Testing

    .venv/bin/python -m pytest -q

This runs the permanent regression suite under tests/:

    tests/test_states.py        basis states and normalisation
    tests/test_gates.py         gate identities, Bell state, CNOT, unitarity
    tests/test_measurement.py   Born-rule probabilities, collapse, sampling
    tests/test_tier1.py         Tier 1 formula regressions
    tests/test_schrodinger.py   Schrödinger solver vs analytical references

Probability checks are exact (np.allclose). Sampling tests use a fixed seed
and a loose tolerance so they are stable across runs.

## Roadmap

- 2D Schrödinger solver (extend the finite-difference Hamiltonian to 2D).
- Time-dependent Schrödinger propagation (Crank-Nicolson or split-operator).
- Qubit: controlled-Z (CZ) and controlled-phase gates.
