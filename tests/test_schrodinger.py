import numpy as np
import pytest

import schrodinger as s


def test_make_grid_interior_points():
    x = s.make_grid(0.0, 1.0, 3)
    assert np.allclose(x, [0.25, 0.5, 0.75])


def test_well_energies_match_analytical():
    x, en, wf = s.solve_tise(s.infinite_well, 0.0, 1.0, 200, n_states=5)
    ana = s.well_energies_analytical(5, 1.0)
    assert np.allclose(en, ana, rtol=1e-3)


def test_well_ground_state_wavefunction():
    x, en, wf = s.solve_tise(s.infinite_well, 0.0, 1.0, 200, n_states=1)
    ana = s.well_wavefunction_analytical(x, 1, 1.0)
    assert np.allclose(np.abs(wf[:, 0]), ana, atol=1e-6)


def test_well_wavefunctions_normalized():
    x, en, wf = s.solve_tise(s.infinite_well, 0.0, 1.0, 200, n_states=3)
    for k in range(3):
        norm = np.trapezoid(np.abs(wf[:, k]) ** 2, x)
        assert np.isclose(norm, 1.0, atol=1e-4), k


def test_well_eigenstates_orthogonal():
    x, en, wf = s.solve_tise(s.infinite_well, 0.0, 1.0, 200, n_states=3)
    overlap = np.trapezoid(np.conj(wf[:, 0]) * wf[:, 1], x)
    assert abs(overlap) < 1e-6


def test_harmonic_energies_match_analytical():
    x, en, wf = s.solve_tise(s.harmonic_oscillator(1.0), -8.0, 8.0, 400, n_states=5)
    ana = s.harmonic_energies_analytical(5, 1.0)
    assert np.allclose(en, ana, rtol=1e-3)


def test_harmonic_ground_state_wavefunction():
    x, en, wf = s.solve_tise(s.harmonic_oscillator(1.0), -8.0, 8.0, 400, n_states=1)
    ana = s.harmonic_wavefunction_analytical(x, 1.0)
    assert np.allclose(np.abs(wf[:, 0]), ana, atol=1e-3)


def test_harmonic_wavefunctions_normalized():
    x, en, wf = s.solve_tise(s.harmonic_oscillator(1.0), -8.0, 8.0, 400, n_states=3)
    for k in range(3):
        norm = np.trapezoid(np.abs(wf[:, k]) ** 2, x)
        assert np.isclose(norm, 1.0, atol=1e-4), k


def test_finite_well_bound_states():
    depth, width = 10.0, 2.0
    x, en, wf = s.solve_tise(s.finite_well(depth, width), -8.0, 8.0, 400, n_states=6)
    assert len(en) == 6
    # three bound states, ordered by increasing energy, all negative
    assert en[0] < en[1] < en[2] < 0.0
    assert en[0] > -depth
    # fourth state is above the well (unbound)
    assert en[3] > 0.0


def test_solve_tise_invalid_points_raises():
    with pytest.raises(ValueError):
        s.solve_tise(s.infinite_well, 0.0, 1.0, 1)


def test_plot_states_writes_file(tmp_path):
    pytest.importorskip("matplotlib")
    x, en, wf = s.solve_tise(s.infinite_well, 0.0, 1.0, 200, n_states=3)
    out = tmp_path / "wf.png"
    s.plot_states(x, s.infinite_well, en, wf, n_states=3, filename=str(out))
    assert out.exists() and out.stat().st_size > 0
