#!/usr/bin/env python3
"""Tier 1 quantum physics calculator.

Formula calculator: numbers in, known physics equations applied, numbers out.
No third-party dependencies -- runs on the Python standard library.

Usage:
  python3 quantum_calc.py                      # interactive menu
  python3 quantum_calc.py demo                 # print worked examples
  python3 quantum_calc.py <formula> <args>     # single calculation (see --help)
"""

import argparse
import math
import sys

# ---------------------------------------------------------------------------
# Physical constants (CODATA 2018, SI units)
# ---------------------------------------------------------------------------
H_PLANCK = 6.62607015e-34        # Planck constant, J.s
HBAR = H_PLANCK / (2 * math.pi)  # reduced Planck constant, J.s
C_LIGHT = 299_792_458.0          # speed of light, m/s
EV_JOULE = 1.602176634e-19       # one electron-volt, J
M_ELECTRON = 9.1093837015e-31    # electron rest mass, kg
RYDBERG_EV = 13.605693122994     # Rydberg energy for hydrogen, eV


# ---------------------------------------------------------------------------
# Formulas (each returns a float in its natural display unit)
# ---------------------------------------------------------------------------
def photon_energy_from_wavelength(wavelength_nm: float) -> float:
    """E = h c / lambda. Wavelength in nm -> photon energy in eV."""
    if wavelength_nm <= 0:
        raise ValueError("wavelength must be positive")
    lam = wavelength_nm * 1e-9          # nm -> m
    energy_j = H_PLANCK * C_LIGHT / lam
    return energy_j / EV_JOULE          # J -> eV


def photon_energy_from_frequency(frequency_hz: float) -> float:
    """E = h f. Frequency in Hz -> photon energy in eV."""
    if frequency_hz <= 0:
        raise ValueError("frequency must be positive")
    energy_j = H_PLANCK * frequency_hz
    return energy_j / EV_JOULE


def de_broglie_wavelength(mass_kg: float, velocity_mps: float) -> float:
    """lambda = h / (m v). Returns wavelength in metres."""
    if mass_kg <= 0:
        raise ValueError("mass must be positive")
    return H_PLANCK / (mass_kg * velocity_mps)


def hydrogen_level(n: int) -> float:
    """E_n = -13.606 eV / n^2 for a hydrogen atom."""
    if n < 1:
        raise ValueError("principal quantum number n must be >= 1")
    return -RYDBERG_EV / (n * n)


def hydrogen_transition(n_initial: int, n_final: int) -> float:
    """Photon energy emitted when an electron drops from n_initial to n_final."""
    return hydrogen_level(n_initial) - hydrogen_level(n_final)


def min_momentum_uncertainty(dx_m: float) -> float:
    """Heisenberg: dx * dp >= hbar/2. Given dx, returns minimum dp (kg.m/s)."""
    if dx_m <= 0:
        raise ValueError("position uncertainty must be positive")
    return HBAR / (2 * dx_m)


def photoelectric_ke(frequency_hz: float, work_function_ev: float) -> float:
    """K_max = h f - phi. Frequency in Hz, work function in eV -> K_max in eV."""
    photon_ev = photon_energy_from_frequency(frequency_hz)
    if photon_ev < work_function_ev:
        raise ValueError(
            "photon energy (%.3f eV) is below the work function (%.3f eV); "
            "no electrons are ejected" % (photon_ev, work_function_ev)
        )
    return photon_ev - work_function_ev


def harmonic_oscillator_energy(n: int, omega_rad_s: float) -> tuple:
    """E_n = (n + 1/2) hbar omega. Returns (joules, electron-volts)."""
    if n < 0:
        raise ValueError("quantum number n must be >= 0")
    energy_j = (n + 0.5) * HBAR * omega_rad_s
    return energy_j, energy_j / EV_JOULE


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def fmt(value: float, digits: int = 5) -> str:
    return f"{value:.{digits}g}"


# ---------------------------------------------------------------------------
# Formula registry for the menu and the demo
# ---------------------------------------------------------------------------
def _run_all_demos() -> list:
    """Return a list of (label, result_string) for each formula."""
    out = []

    ev = photon_energy_from_wavelength(500.0)
    out.append(("Photon energy, 500 nm (green light)", f"{fmt(ev)} eV"))

    ev = photon_energy_from_frequency(6.0e14)
    out.append(("Photon energy, 6e14 Hz", f"{fmt(ev)} eV"))

    lam = de_broglie_wavelength(M_ELECTRON, 1.0e6)
    out.append(("de Broglie wavelength, electron at 1e6 m/s",
                f"{fmt(lam)} m  =  {fmt(lam * 1e9)} nm"))

    out.append(("Hydrogen level n=2", f"{fmt(hydrogen_level(2))} eV"))
    out.append(("Hydrogen transition n=3 -> n=2 (H-alpha)",
                f"{fmt(hydrogen_transition(3, 2))} eV"))

    dp = min_momentum_uncertainty(1.0e-9)
    out.append(("Minimum momentum uncertainty for dx = 1 nm",
                f"{fmt(dp)} kg.m/s"))

    ke = photoelectric_ke(1.2e15, 4.5)
    out.append(("Photoelectric K_max, f=1.2e15 Hz, phi=4.5 eV (sodium-like)",
                f"{fmt(ke)} eV"))

    e_j, e_ev = harmonic_oscillator_energy(0, 1.0e15)
    out.append(("Harmonic oscillator ground state, omega=1e15 rad/s",
                f"{fmt(e_j)} J  =  {fmt(e_ev)} eV"))

    return out


# ---------------------------------------------------------------------------
# Interactive menu
# ---------------------------------------------------------------------------
MENU = [
    ("1", "Photon energy from wavelength", "photon-wavelength"),
    ("2", "Photon energy from frequency", "photon-frequency"),
    ("3", "de Broglie wavelength", "de-broglie"),
    ("4", "Hydrogen energy level", "hydrogen-level"),
    ("5", "Hydrogen transition energy", "hydrogen-transition"),
    ("6", "Uncertainty principle (min momentum)", "uncertainty"),
    ("7", "Photoelectric effect (K_max)", "photoelectric"),
    ("8", "Quantum harmonic oscillator energy", "harmonic"),
]


def _prompt_float(label: str) -> float:
    while True:
        raw = input(f"  {label}: ").strip()
        try:
            return float(raw)
        except ValueError:
            print("  Please enter a number.")


def _prompt_int(label: str, minimum: int) -> int:
    while True:
        raw = input(f"  {label}: ").strip()
        try:
            value = int(raw)
        except ValueError:
            print("  Please enter an integer.")
            continue
        if value < minimum:
            print(f"  Must be >= {minimum}.")
            continue
        return value


def _interactive_menu() -> None:
    print("Tier 1 quantum physics calculator")
    print("--------------------------------")
    while True:
        print()
        for key, label, _ in MENU:
            print(f"  {key}. {label}")
        print("  d. Run all demo calculations")
        print("  q. Quit")
        choice = input("\nChoose: ").strip().lower()

        if choice == "q":
            print("Bye.")
            return
        if choice == "d":
            print()
            for label, result in _run_all_demos():
                print(f"  {label:<52} {result}")
            continue

        formula = next((f for k, _, f in MENU if k == choice), None)
        if formula is None:
            print("  Unknown choice.")
            continue

        print()
        try:
            if formula == "photon-wavelength":
                wl = _prompt_float("Wavelength (nm): ")
                print(f"  E = {fmt(photon_energy_from_wavelength(wl))} eV")
            elif formula == "photon-frequency":
                f = _prompt_float("Frequency (Hz): ")
                print(f"  E = {fmt(photon_energy_from_frequency(f))} eV")
            elif formula == "de-broglie":
                m = _prompt_float("Mass (kg): ")
                v = _prompt_float("Velocity (m/s): ")
                lam = de_broglie_wavelength(m, v)
                print(f"  lambda = {fmt(lam)} m  =  {fmt(lam * 1e9)} nm")
            elif formula == "hydrogen-level":
                n = _prompt_int("Principal quantum number n: ", 1)
                print(f"  E_{n} = {fmt(hydrogen_level(n))} eV")
            elif formula == "hydrogen-transition":
                n1 = _prompt_int("Initial level n_i: ", 1)
                n2 = _prompt_int("Final level n_f: ", 1)
                print(f"  photon energy = {fmt(hydrogen_transition(n1, n2))} eV")
            elif formula == "uncertainty":
                dx = _prompt_float("Position uncertainty dx (m): ")
                print(f"  dp_min = {fmt(min_momentum_uncertainty(dx))} kg.m/s")
            elif formula == "photoelectric":
                f = _prompt_float("Frequency (Hz): ")
                phi = _prompt_float("Work function (eV): ")
                print(f"  K_max = {fmt(photoelectric_ke(f, phi))} eV")
            elif formula == "harmonic":
                n = _prompt_int("Quantum number n: ", 0)
                w = _prompt_float("Angular frequency omega (rad/s): ")
                e_j, e_ev = harmonic_oscillator_energy(n, w)
                print(f"  E_{n} = {fmt(e_j)} J  =  {fmt(e_ev)} eV")
        except ValueError as exc:
            print(f"  Error: {exc}")


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="quantum_calc",
        description="Tier 1 quantum physics calculator.",
    )
    sub = p.add_subparsers(dest="command")

    sp = sub.add_parser("photon-wavelength", help="E = hc/lambda (wavelength in nm)")
    sp.add_argument("wavelength", type=float, help="wavelength in nm")

    sp = sub.add_parser("photon-frequency", help="E = hf (frequency in Hz)")
    sp.add_argument("frequency", type=float, help="frequency in Hz")

    sp = sub.add_parser("de-broglie", help="lambda = h/(mv)")
    sp.add_argument("mass", type=float, help="mass in kg")
    sp.add_argument("velocity", type=float, help="velocity in m/s")

    sp = sub.add_parser("hydrogen-level", help="E_n = -13.606 eV / n^2")
    sp.add_argument("n", type=int, help="principal quantum number")

    sp = sub.add_parser("hydrogen-transition",
                        help="photon energy for n_i -> n_f transition")
    sp.add_argument("n_initial", type=int)
    sp.add_argument("n_final", type=int)

    sp = sub.add_parser("uncertainty", help="minimum momentum uncertainty given dx")
    sp.add_argument("dx", type=float, help="position uncertainty in m")

    sp = sub.add_parser("photoelectric", help="K_max = hf - phi")
    sp.add_argument("frequency", type=float, help="frequency in Hz")
    sp.add_argument("work_function", type=float, help="work function in eV")

    sp = sub.add_parser("harmonic", help="E_n = (n + 1/2) hbar omega")
    sp.add_argument("n", type=int, help="quantum number (>= 0)")
    sp.add_argument("omega", type=float, help="angular frequency in rad/s")

    sub.add_parser("demo", help="print worked examples for every formula")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.command is None:
        _interactive_menu()
        return 0

    try:
        if args.command == "demo":
            for label, result in _run_all_demos():
                print(f"{label:<52} {result}")
        elif args.command == "photon-wavelength":
            print(f"{fmt(photon_energy_from_wavelength(args.wavelength))} eV")
        elif args.command == "photon-frequency":
            print(f"{fmt(photon_energy_from_frequency(args.frequency))} eV")
        elif args.command == "de-broglie":
            lam = de_broglie_wavelength(args.mass, args.velocity)
            print(f"{fmt(lam)} m  =  {fmt(lam * 1e9)} nm")
        elif args.command == "hydrogen-level":
            print(f"{fmt(hydrogen_level(args.n))} eV")
        elif args.command == "hydrogen-transition":
            print(f"{fmt(hydrogen_transition(args.n_initial, args.n_final))} eV")
        elif args.command == "uncertainty":
            print(f"{fmt(min_momentum_uncertainty(args.dx))} kg.m/s")
        elif args.command == "photoelectric":
            print(f"{fmt(photoelectric_ke(args.frequency, args.work_function))} eV")
        elif args.command == "harmonic":
            e_j, e_ev = harmonic_oscillator_energy(args.n, args.omega)
            print(f"{fmt(e_j)} J  =  {fmt(e_ev)} eV")
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
