import pytest


np = pytest.importorskip("numpy")


def test_cartesian_tuple_sets_atomic_numbers_and_masses():
    from src.geometry import Cartesian

    geom = Cartesian()
    geom.geometry = (
        ["H", "O", "H"],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        0,
        1,
    )

    assert geom.natoms == 3
    assert geom.charge == 0
    assert geom.multi == 1
    assert geom.atoms == ["H", "O", "H"]
    assert geom.coords.shape == (3, 3)

    assert geom.atomicnumbers.tolist() == [1, 8, 1]
    assert geom.atomicmasses.shape == (3,)
    assert geom.atomicmasses[0] > 0.0


def test_cartesian_xyz_string_parses_header_and_atoms():
    from src.geometry import Cartesian

    xyz = """3
0 1
H 0.0 0.0 0.0
O 0.0 0.0 1.0
H 1.0 0.0 0.0
"""

    geom = Cartesian()
    geom.geometry = xyz

    assert geom.natoms == 3
    assert geom.charge == 0
    assert geom.multi == 1
    assert geom.atomicnumbers.tolist() == [1, 8, 1]

