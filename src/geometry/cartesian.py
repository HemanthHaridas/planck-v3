import numpy
import typing
from src.geometry import base


class Cartesian(base.BaseGeometry):
    """
    Concrete implementation of BaseGeometry using Cartesian coordinates.

    This class represents molecular geometry in terms of absolute Cartesian
    coordinates (x, y, z) for each atom. It supports initialization from
    either:
      1. A string in XYZ-like format (commonly used in computational chemistry).
      2. A tuple containing (atoms, coords, charge, multiplicity).

    Attributes:
        atoms (list[str]):
            List of atomic symbols (e.g., ["H", "O", "H"]).
        coords (numpy.ndarray):
            Array of Cartesian coordinates with shape (N, 3), where N is the
            number of atoms.
        coords_internal (numpy.ndarray):
            Flattened 1D representation of `coords`, useful for internal
            computations.
        natoms (int):
            Number of atoms in the system.
        charge (int):
            Total molecular charge.
        multi (int):
            Spin multiplicity of the system.

    Example (XYZ-like string input):
        >>> xyz_data = \"\"\"3
        ... 0 1
        ... H 0.0 0.0 0.0
        ... O 0.0 0.0 1.0
        ... H 1.0 0.0 0.0\"\"\"
        >>> geom = Cartesian()
        >>> geom.geometry = xyz_data
        >>> geom.geometry
        (['H', 'O', 'H'],
         array([[0., 0., 0.],
                [0., 0., 1.],
                [1., 0., 0.]]))

    Example (tuple input):
        >>> atoms = ["H", "O", "H"]
        >>> coords = [[0.0, 0.0, 0.0],
        ...           [0.0, 0.0, 1.0],
        ...           [1.0, 0.0, 0.0]]
        >>> charge, multi = 0, 1
        >>> geom = Cartesian()
        >>> geom.geometry = (atoms, coords, charge, multi)
        >>> geom.geometry
        (['H', 'O', 'H'],
         array([[0., 0., 0.],
                [0., 0., 1.],
                [1., 0., 0.]]))
    """

    def __init__(self):
        """
        Initialize a Cartesian geometry object.

        Calls the BaseGeometry initializer to set up default attributes
        such as atoms, coords, atomic numbers, charge, and multiplicity.
        """
        super().__init__()

    @property
    def geometry(self) -> dict:
        """
        Get the molecular geometry in Cartesian representation.

        Returns:
            tuple:
                A tuple containing:
                - atoms (list[str]): Atomic symbols.
                - coords (numpy.ndarray): Cartesian coordinates (N, 3).
        """
        return {
            "atoms": self.atoms, 
            "coords": self.coords,
            "charge": self.charge,
            "multi": self.multi,
            "atomicnumbers": self.atomicnumbers,
            "atomicmasses": self.atomicmass
        }

    @geometry.setter
    def geometry(self, value) -> typing.Any:
        """
        Set the molecular geometry in Cartesian representation.

        Accepts either:
        - A string in XYZ-like format:
            * First line: number of atoms (natoms).
            * Second line: charge and multiplicity.
            * Remaining lines: atom symbol followed by x, y, z coordinates.
        - A tuple of length 4:
            (atoms, coords, charge, multiplicity)

        Args:
            value (str or tuple):
                Input geometry data.

        Raises:
            ValueError:
                - If atom lines are malformed (not 4 tokens).
                - If the number of atoms does not match the header.
            TypeError:
                - If atoms are not strings.
                - If the input type is unsupported.
        """
        _coords = []
        if isinstance(value, str):
            _structure = [_line for _line in value.split("\n") if _line.strip()]
            self.natoms = int(_structure[0])
            self.charge, self.multi = map(int, _structure[1].split())

            for _line in _structure[2:]:
                _data = _line.split()
                if len(_data) != 4:
                    raise ValueError(f"Malformed atom line: {_line}")
                self.atoms.append(_data[0])
                _coords.append([float(x) for x in _data[1:]])

            self.coords = numpy.array(_coords)
            self.coords_internal = self.coords.flatten()

            if self.natoms != len(self.atoms):
                raise ValueError("Mismatch between header and atom lines.")

        elif isinstance(value, tuple) and len(value) == 4:
            self.atoms, self.coords, self.charge, self.multi = value

            if not all(isinstance(a, str) for a in self.atoms):
                raise TypeError("Atoms must be a list of strings.")

            self.coords = numpy.array(self.coords, dtype=float).reshape(len(self.atoms), 3)
            self.coords_internal = self.coords.flatten()
            self.charge = int(self.charge)
            self.multi = int(self.multi)

        else:
            raise TypeError("Geometry must be defined either as an xyz block or (atoms, coords, charge, multiplicity).")

    def __repr__(self):
        return (
            f"Cartesian(natoms={self.natoms}), ",
            f"Charge = {self.charge}, ",
            f"Multiplicity = {self.multi}"
        )
