import pytest


np = pytest.importorskip("numpy")


def _geom_from_atoms(atoms, charge, multi):
    from src.geometry import Cartesian

    geom = Cartesian()
    geom.geometry = (atoms, [[0.0, 0.0, 0.0] for _ in atoms], charge, multi)
    return geom


def test_rhf_accepts_closed_shell_singlet_and_even_electrons():
    from src.hartree_fock import RestrictedHartreeFock

    # Water: 1 + 8 + 1 = 10 electrons, singlet
    geom = _geom_from_atoms(["H", "O", "H"], charge=0, multi=1)

    rhf = RestrictedHartreeFock(geom)
    energy = rhf.compute()

    assert rhf.nelectrons == 10
    assert rhf.nalpha == 5
    assert rhf.nbeta == 5
    assert rhf.converged is True
    assert energy == 0.0


def test_rhf_rejects_open_shell_or_odd_electron_system():
    from src.hartree_fock import RestrictedHartreeFock

    # Methyl radical: 6 + 1 + 1 + 1 = 9 electrons, doublet (open-shell)
    geom = _geom_from_atoms(["C", "H", "H", "H"], charge=0, multi=2)

    with pytest.raises(ValueError):
        RestrictedHartreeFock(geom)

