#!/usr/bin/env python3
"""Tier 3: 1D time-independent Schrodinger equation solver.

Solves
    [ -hbar^2/(2m) d^2/dx^2 + V(x) ] psi(x) = E psi(x)
for bound states with Dirichlet boundaries psi(x_min) = psi(x_max) = 0.

Method: finite-difference discretisation of position + diagonalisation of the
resulting sparse Hamiltonian (scipy.sparse.linalg.eigsh, lowest eigenvalues).

The CLI uses natural units (hbar = m = 1); the solver itself accepts hbar and
mass as arguments.
"""

import argparse
import sys

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla


# --- Discretisation ---------------------------------------------------------
def make_grid(x_min, x_max, n_points):
    """Interior grid points (the two Dirichlet boundaries are excluded)."""
    return np.linspace(x_min, x_max, n_points + 2)[1:-1]


def kinetic_operator(n_points, dx, hbar=1.0, mass=1.0):
    """Sparse matrix for -hbar^2/(2m) d^2/dx^2 (2nd-order central difference)."""
    coeff = hbar ** 2 / (2.0 * mass * dx ** 2)
    diag = 2.0 * coeff
    off = -coeff
    return sparse.diags([off, diag, off], [-1, 0, 1],
                        shape=(n_points, n_points), format="csr")


# --- Solver -----------------------------------------------------------------
def solve_tise(potential, x_min, x_max, n_points, n_states=5,
               hbar=1.0, mass=1.0):
    """Solve the 1D TISE; return (x, energies, wavefunctions).

    `potential` is a callable V(x) -> array. Wavefunctions are normalised so
    that int |psi|^2 dx = 1.
    """
    if n_points < 2:
        raise ValueError("n_points must be >= 2")
    x = make_grid(x_min, x_max, n_points)
    dx = (x_max - x_min) / (n_points + 1)

    H = kinetic_operator(n_points, dx, hbar, mass)
    H = H + sparse.diags(potential(x), 0, format="csr")

    k = min(n_states, n_points - 1)
    # 'SA' = smallest algebraic eigenvalue (most negative). This correctly
    # returns bound states for potentials like the finite well, which have
    # negative energy; 'SM' (smallest magnitude) would miss them.
    energies, vectors = spla.eigsh(H, k=k, which="SA")

    order = np.argsort(energies)
    energies = energies[order]
    vectors = vectors[:, order]

    vectors = vectors / np.sqrt(dx)  # int |psi|^2 dx = 1

    return x, energies, vectors


# --- Potentials -------------------------------------------------------------
def infinite_well(x):
    """V = 0 inside; the walls are enforced by the Dirichlet boundaries."""
    return np.zeros_like(np.asarray(x))


def harmonic_oscillator(omega, mass=1.0):
    """V(x) = (1/2) m omega^2 x^2."""
    def v(x):
        x = np.asarray(x)
        return 0.5 * mass * omega ** 2 * x ** 2
    return v


def finite_well(depth, width):
    """V(x) = -depth for |x| < width/2, else 0."""
    def v(x):
        x = np.asarray(x)
        return np.where(np.abs(x) < width / 2.0, -depth, 0.0)
    return v


# --- Analytical references --------------------------------------------------
def well_energies_analytical(n, length, hbar=1.0, mass=1.0):
    """E_n = n^2 pi^2 hbar^2 / (2 m L^2), n = 1, 2, ..."""
    ns = np.arange(1, n + 1)
    return hbar ** 2 * (ns * np.pi) ** 2 / (2.0 * mass * length ** 2)


def harmonic_energies_analytical(n, omega, hbar=1.0):
    """E_n = (n + 1/2) hbar omega, n = 0, 1, ..."""
    return (np.arange(n) + 0.5) * hbar * omega


def well_wavefunction_analytical(x, n, length):
    """psi_n(x) = sqrt(2/L) sin(n pi x / L) on [0, L]."""
    x = np.asarray(x)
    return np.sqrt(2.0 / length) * np.sin(n * np.pi * x / length)


def harmonic_wavefunction_analytical(x, omega, hbar=1.0, mass=1.0):
    """Ground state psi_0(x) = (m omega / pi hbar)^(1/4) exp(-m omega x^2/2hbar)."""
    x = np.asarray(x)
    a = (mass * omega / (np.pi * hbar)) ** 0.25
    return a * np.exp(-mass * omega * x ** 2 / (2.0 * hbar))


# --- Plotting ---------------------------------------------------------------
def plot_states(x, potential, energies, wavefunctions, n_states=3,
                filename="wavefunctions.png"):
    """Plot V(x) and the first `n_states` wavefunctions (energy-level style).

    matplotlib is imported lazily; a non-interactive Agg backend is used so
    this works headless. Returns the filename written.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plotting; install it with "
            "`.venv/bin/pip install matplotlib`"
        ) from exc

    V = np.asarray(potential(x))
    n = min(n_states, len(energies))
    gaps = np.diff(energies)
    amp = 0.5 * np.min(gaps) if gaps.size and np.min(gaps) > 0 else 0.5

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, V, color="0.55", lw=1.5, label="V(x)")
    for k in range(n):
        psi = wavefunctions[:, k]
        peak = np.max(np.abs(psi))
        if peak > 0:
            psi = psi / peak * amp
        ax.plot(x, psi + energies[k], lw=1.5,
                label=f"n={k}, E={energies[k]:.3f}")
        ax.axhline(energies[k], color="0.8", lw=0.6)
    ax.set_xlabel("x")
    ax.set_ylabel("energy")
    ax.set_title("Wavefunctions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename, dpi=100)
    plt.close(fig)
    return filename


# --- Reporting --------------------------------------------------------------
def _print_compare(title, ns, numerical, analytical):
    print(title)
    print(f"  {'state':>5} {'numerical':>14} {'analytical':>14} {'rel.err':>10}")
    for n, num, ana in zip(ns, numerical, analytical):
        rel = abs(num - ana) / abs(ana)
        print(f"  {n:>5} {num:>14.6f} {ana:>14.6f} {rel:>10.2e}")
    print()


def run_demo():
    # Infinite square well, L = 1, natural units
    length = 1.0
    x, en, wf = solve_tise(infinite_well, 0.0, length, 200, n_states=5)
    _print_compare("Infinite square well, L=1 (hbar=m=1), N=200 interior points",
                   range(1, 6), en, well_energies_analytical(5, length))

    ana = well_wavefunction_analytical(x, 1, length)
    err = np.max(np.abs(np.abs(wf[:, 0]) - ana))
    print(f"  ground-state |psi| vs sqrt(2/L) sin(pi x/L): max err = {err:.2e}\n")

    # Harmonic oscillator, omega = 1, box [-8, 8]
    xh, eh, wfh = solve_tise(harmonic_oscillator(1.0), -8.0, 8.0, 400, n_states=5)
    _print_compare("Harmonic oscillator, omega=1, box [-8,8] (hbar=m=1), N=400",
                   range(5), eh, harmonic_energies_analytical(5, 1.0))


# --- CLI --------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        prog="schrodinger",
        description="1D time-independent Schrodinger equation solver.",
    )
    sub = p.add_subparsers(dest="command")

    sub.add_parser("demo", help="well + harmonic oscillator vs analytic")

    sp = sub.add_parser("well", help="particle in an infinite square well")
    sp.add_argument("length", type=float, help="well width L")
    sp.add_argument("--states", type=int, default=5)
    sp.add_argument("--points", type=int, default=200)
    sp.add_argument("--plot", action="store_true", help="save a wavefunction plot")
    sp.add_argument("--output", default="well.png", help="plot filename (with --plot)")

    sp = sub.add_parser("harmonic", help="quantum harmonic oscillator")
    sp.add_argument("omega", type=float, help="angular frequency")
    sp.add_argument("--states", type=int, default=5)
    sp.add_argument("--points", type=int, default=400)
    sp.add_argument("--xmax", type=float, default=8.0)
    sp.add_argument("--plot", action="store_true", help="save a wavefunction plot")
    sp.add_argument("--output", default="harmonic.png", help="plot filename (with --plot)")

    sp = sub.add_parser("finite-well", help="finite square well")
    sp.add_argument("depth", type=float, help="well depth (positive)")
    sp.add_argument("width", type=float, help="well width")
    sp.add_argument("--states", type=int, default=5)
    sp.add_argument("--points", type=int, default=400)
    sp.add_argument("--xmax", type=float, default=8.0)
    sp.add_argument("--plot", action="store_true", help="save a wavefunction plot")
    sp.add_argument("--output", default="finite-well.png", help="plot filename (with --plot)")

    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    try:
        if args.command == "demo":
            run_demo()
            return 0

        if args.command == "well":
            potential = infinite_well
            x, en, wf = solve_tise(potential, 0.0, args.length,
                                   args.points, args.states)
            _print_compare("Infinite square well (natural units)",
                           range(1, args.states + 1), en,
                           well_energies_analytical(args.states, args.length))
        elif args.command == "harmonic":
            potential = harmonic_oscillator(args.omega)
            x, en, wf = solve_tise(potential, -args.xmax, args.xmax,
                                   args.points, args.states)
            _print_compare("Harmonic oscillator (natural units)",
                           range(args.states), en,
                           harmonic_energies_analytical(args.states, args.omega))
        elif args.command == "finite-well":
            potential = finite_well(args.depth, args.width)
            x, en, wf = solve_tise(potential, -args.xmax, args.xmax,
                                   args.points, args.states)
            print("Finite square well bound states (numerical):")
            for i, e in enumerate(en):
                print(f"  state {i}: {e:>12.6f}")
        else:
            build_parser().print_help()
            return 0

        if getattr(args, "plot", False):
            out = plot_states(x, potential, en, wf,
                              n_states=args.states, filename=args.output)
            print(f"plot saved: {out}")
    except (ValueError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
