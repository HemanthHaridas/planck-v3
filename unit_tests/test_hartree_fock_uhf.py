import pytest


np = pytest.importorskip("numpy")


def _geom_from_atoms(atoms, charge, multi):
    from src.geometry import Cartesian

    geom = Cartesian()
    geom.geometry = (atoms, [[0.0, 0.0, 0.0] for _ in atoms], charge, multi)
    return geom


def test_uhf_accepts_doublet_odd_electron_system_and_spin_excess_matches():
    from src.hartree_fock import UnrestrictedHartreeFock

    # Methyl radical: 9 electrons, doublet -> nalpha - nbeta must be 1
    geom = _geom_from_atoms(["C", "H", "H", "H"], charge=0, multi=2)

    uhf = UnrestrictedHartreeFock(geom)
    energy = uhf.compute()

    assert uhf.nelectrons == 9
    assert (uhf.nalpha - uhf.nbeta) == (geom.multi - 1) == 1
    assert uhf.nalpha == 5
    assert uhf.nbeta == 4
    assert uhf.converged is True
    assert energy == 0.0


def test_uhf_accepts_triplet_even_electron_system_and_spin_excess_matches():
    from src.hartree_fock import UnrestrictedHartreeFock

    # Oxygen atom: 8 electrons, triplet -> nalpha - nbeta must be 2
    geom = _geom_from_atoms(["O"], charge=0, multi=3)

    uhf = UnrestrictedHartreeFock(geom)
    uhf.compute()

    assert uhf.nelectrons == 8
    assert (uhf.nalpha - uhf.nbeta) == 2
    assert uhf.nalpha == 5
    assert uhf.nbeta == 3


def test_uhf_rejects_inconsistent_nelectrons_and_multiplicity():
    from src.hartree_fock import UnrestrictedHartreeFock

    # Nitrogen atom: 7 electrons, triplet would require nalpha=4.5, nbeta=2.5 (impossible)
    geom = _geom_from_atoms(["N"], charge=0, multi=3)

    with pytest.raises(ValueError):
        UnrestrictedHartreeFock(geom)

