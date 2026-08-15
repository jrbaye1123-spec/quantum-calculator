#!/usr/bin/env python3
"""Tier 2: minimal qubit state-vector simulator (CLI).

Circuit DSL: a comma-separated list of gate specs, applied left to right.
  H            Hadamard on qubit 0
  X:1          Pauli-X on qubit 1
  CNOT:1:0     CNOT, control=1, target=0 (any qubits, via SWAP routing)
  SWAP:0:1     SWAP adjacent qubits
  S, T, Y, Z, I likewise; multi-qubit registers supported via <gate>:<target>.

Examples:
  python3 qubit.py demo
  python3 qubit.py state +
  python3 qubit.py run 2 "H:1,CNOT:1:0" --shots 1000
  python3 qubit.py gates
"""

import argparse
import sys

import numpy as np

import gates
import measurement
import states


def run_circuit(n_qubits, gate_specs):
    """Build |0...0> and apply gate specs left to right. Returns the state."""
    state = states.zero_state(n_qubits)
    for spec in gate_specs:
        parts = [p.strip() for p in spec.split(":")]
        name = parts[0].upper()

        if name == "CNOT":
            if len(parts) != 3:
                raise ValueError("CNOT requires CNOT:control:target")
            control, target = int(parts[1]), int(parts[2])
            state = gates.apply_cnot(state, control, target, n_qubits)
            continue

        if name == "SWAP":
            if len(parts) != 3:
                raise ValueError("SWAP requires SWAP:q0:q1")
            q0, q1 = int(parts[1]), int(parts[2])
            state = gates.apply_swap(state, q0, q1, n_qubits)
            continue

        gate = gates.GATE_TABLE.get(name)
        if gate is None:
            raise ValueError(f"unknown gate {name!r}; see `qubit.py gates`")
        target = int(parts[1]) if len(parts) > 1 else 0
        state = gates.apply(state, gate, target, n_qubits)

    return state


def format_state(state):
    """Render the state vector with amplitudes and probabilities per basis."""
    n_qubits = len(state).bit_length() - 1
    lines = []
    for i, amp in enumerate(np.asarray(state).ravel()):
        label = states.label(i, n_qubits)
        prob = abs(amp) ** 2
        lines.append(f"  |{label}>  {amp.real:+.3f}{amp.imag:+.3f}i   P={prob:.4f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------
DEMOS = [
    ("H on |0> -> |+> (equal superposition)", 1, ["H"]),
    ("H then H on |0> -> |0> (H is self-inverse)", 1, ["H", "H"]),
    ("X on |0> -> |1> (bit flip)", 1, ["X"]),
    ("Z on |+> -> |-> (phase flip)", 1, ["H", "Z"]),
    ("S on |+> -> |i> (Y eigenstate)", 1, ["H", "S"]),
    ("Bell state |00>+|11>: H on q1, CNOT(1,0)", 2, ["H:1", "CNOT:1:0"]),
    ("GHZ state |000>+|111>: H, CNOT(0,1), CNOT(0,2)",
     3, ["H", "CNOT:0:1", "CNOT:0:2"]),
]


def run_demo(shots=2000):
    for title, n_qubits, specs in DEMOS:
        state = run_circuit(n_qubits, specs)
        print(f"\n== {title} ==")
        print(f"   circuit: {' -> '.join(specs)}  (on |0...0>)")
        print(format_state(state))
        if n_qubits == 1:
            counts = measurement.sample(state, shots)
            print(f"   sampled {shots} shots: {dict(sorted(counts.items()))}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        prog="qubit",
        description="Tier 2 qubit state-vector simulator.",
    )
    sub = p.add_subparsers(dest="command")

    sub.add_parser("demo", help="run worked examples")

    sp = sub.add_parser("state", help="print a named basis state")
    sp.add_argument("name", help="0, 1, +, -, i, -i")

    sp = sub.add_parser("run", help="apply a circuit and print the result")
    sp.add_argument("n_qubits", type=int, help="number of qubits")
    sp.add_argument("gates", help="comma-separated gate specs, e.g. H:1,CNOT:1:0")
    sp.add_argument("--shots", type=int, default=1000,
                    help="number of measurement shots (default 1000)")

    sub.add_parser("gates", help="list available gates")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    try:
        if args.command == "demo":
            run_demo()
        elif args.command == "state":
            print(format_state(states.basis_state(args.name)))
        elif args.command == "run":
            specs = [s for s in args.gates.split(",") if s.strip()]
            state = run_circuit(args.n_qubits, specs)
            print("State after circuit:")
            print(format_state(state))
            print(f"\nMeasured probabilities: {measurement.probabilities(state)}")
            counts = measurement.sample(state, args.shots)
            print(f"Sampled {args.shots} shots:")
            for label, count in sorted(counts.items()):
                print(f"  |{label}>  {count}  ({100 * count / args.shots:.1f}%)")
        elif args.command == "gates":
            print("Single-qubit gates:")
            for name in sorted(gates.GATE_TABLE):
                print(f"  {name}")
            print("Two-qubit gates:")
            print("  CNOT (usage: CNOT:control:target, any qubits)")
            print("  SWAP (usage: SWAP:q0:q1, adjacent qubits)")
        else:
            build_parser().print_help()
    except (ValueError, NotImplementedError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
