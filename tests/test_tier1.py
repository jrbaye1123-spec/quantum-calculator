import numpy as np
import pytest

import quantum_calc as qc


def test_photon_energy_wavelength():
    assert np.isclose(qc.photon_energy_from_wavelength(500.0), 2.4797, rtol=1e-3)


def test_photon_energy_frequency():
    assert np.isclose(qc.photon_energy_from_frequency(6.0e14), 2.4814, rtol=1e-3)


def test_photon_wavelength_negative_raises():
    with pytest.raises(ValueError):
        qc.photon_energy_from_wavelength(-1.0)


def test_de_broglie():
    lam = qc.de_broglie_wavelength(qc.M_ELECTRON, 1.0e6)
    assert np.isclose(lam * 1e9, 0.72739, rtol=1e-3)


def test_hydrogen_level():
    assert np.isclose(qc.hydrogen_level(1), -13.605693122994, rtol=1e-6)
    assert np.isclose(qc.hydrogen_level(2), -3.4014232807485, rtol=1e-6)


def test_hydrogen_level_invalid_n_raises():
    with pytest.raises(ValueError):
        qc.hydrogen_level(0)


def test_hydrogen_transition():
    assert np.isclose(qc.hydrogen_transition(3, 2), 1.8897, rtol=1e-3)


def test_uncertainty():
    assert np.isclose(qc.min_momentum_uncertainty(1.0e-9), 5.2729e-26, rtol=1e-3)


def test_photoelectric():
    assert np.isclose(qc.photoelectric_ke(1.2e15, 4.5), 0.4628, rtol=1e-3)


def test_photoelectric_below_threshold_raises():
    with pytest.raises(ValueError):
        qc.photoelectric_ke(1.0e15, 4.5)


def test_harmonic_oscillator():
    e_j, e_ev = qc.harmonic_oscillator_energy(0, 1.0e15)
    assert np.isclose(e_ev, 0.32911, rtol=1e-3)
