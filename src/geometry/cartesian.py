"""Cartesian coordinate representation of molecular geometry."""

from typing import Any, Dict, List, Tuple, Union

import numpy as np

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
        atoms (List[str]): List of atomic symbols (e.g., ["H", "O", "H"]).
        coords (np.ndarray): Array of Cartesian coordinates with shape (N, 3).
        coords_internal (np.ndarray): Flattened 1D representation of coords.
        natoms (int): Number of atoms in the system.
        charge (int): Total molecular charge.
        multi (int): Spin multiplicity of the system.
        atomicnumbers (np.ndarray): Array of atomic numbers.
        atomicmasses (np.ndarray): Array of atomic masses.

    Example (XYZ-like string input):
        >>> xyz_data = \"\"\"3
        ... 0 1
        ... H 0.0 0.0 0.0
        ... O 0.0 0.0 1.0
        ... H 1.0 0.0 0.0\"\"\"
        >>> geom = Cartesian()
        >>> geom.geometry = xyz_data
        >>> geom.geometry["atoms"]
        ['H', 'O', 'H']

    Example (tuple input):
        >>> atoms = ["H", "O", "H"]
        >>> coords = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
        >>> charge, multi = 0, 1
        >>> geom = Cartesian()
        >>> geom.geometry = (atoms, coords, charge, multi)
        >>> geom.natoms
        3
    """

    def __init__(self) -> None:
        """
        Initialize a Cartesian geometry object.

        Calls the BaseGeometry initializer to set up default attributes
        such as atoms, coords, atomic numbers, charge, and multiplicity.
        """
        super().__init__()

    @property
    def geometry(self) -> Dict[str, Any]:
        """
        Get the molecular geometry in Cartesian representation.

        Returns:
            dict: A dictionary containing:
                - atoms (List[str]): Atomic symbols.
                - coords (np.ndarray): Cartesian coordinates (N, 3).
                - charge (int): Molecular charge.
                - multi (int): Spin multiplicity.
                - atomicnumbers (np.ndarray): Atomic numbers.
                - atomicmasses (np.ndarray): Atomic masses.
        """
        return {
            "atoms": self.atoms,
            "coords": self.coords,
            "charge": self.charge,
            "multi": self.multi,
            "atomicnumbers": self.atomicnumbers,
            "atomicmasses": self.atomicmasses,
        }

    @geometry.setter
    def geometry(self, value: Union[str, Tuple[List[str], Any, int, int]]) -> None:
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
            value: Input geometry data as string or tuple.

        Raises:
            ValueError:
                - If atom lines are malformed (not 4 tokens).
                - If the number of atoms does not match the header.
                - If coordinates cannot be parsed as floats.
            TypeError:
                - If atoms are not strings.
                - If the input type is unsupported.
        """
        if isinstance(value, str):
            self._from_xyz_string(value)
        elif isinstance(value, tuple) and len(value) == 4:
            self._from_tuple(value)
        else:
            raise TypeError(
                "Geometry must be defined either as an xyz block string or "
                "a tuple (atoms, coords, charge, multiplicity)."
            )

    def _from_xyz_string(self, xyz_string: str) -> None:
        """Parse geometry from XYZ-like string format."""
        lines = [line.strip() for line in xyz_string.strip().split("\n") if line.strip()]
        
        if len(lines) < 2:
            raise ValueError("XYZ string must contain at least 2 lines (natoms, charge/multi).")

        # Parse header
        try:
            self.natoms = int(lines[0])
        except ValueError as e:
            raise ValueError(f"First line must be an integer (number of atoms): {lines[0]}") from e

        # Parse charge and multiplicity
        try:
            charge_multi = lines[1].split()
            if len(charge_multi) != 2:
                raise ValueError("Second line must contain charge and multiplicity.")
            self.charge = int(charge_multi[0])
            self.multi = int(charge_multi[1])
        except (ValueError, IndexError) as e:
            raise ValueError(f"Invalid charge/multiplicity line: {lines[1]}") from e

        # Parse atom lines
        if len(lines) - 2 != self.natoms:
            raise ValueError(
                f"Mismatch between header ({self.natoms} atoms) and "
                f"atom lines ({len(lines) - 2} lines)."
            )

        atoms = []
        coords = []
        
        for i, line in enumerate(lines[2:], start=3):
            tokens = line.split()
            if len(tokens) != 4:
                raise ValueError(
                    f"Malformed atom line {i}: expected 4 tokens "
                    f"(symbol x y z), got {len(tokens)}: {line}"
                )
            
            symbol = tokens[0]
            try:
                x, y, z = map(float, tokens[1:])
            except ValueError as e:
                raise ValueError(
                    f"Invalid coordinates in line {i}: {tokens[1:]}"
                ) from e
            
            atoms.append(symbol)
            coords.append([x, y, z])

        self.atoms = atoms
        self.coords = np.array(coords, dtype=float)
        self.coords_internal = self.coords.flatten()
        self._update_atomic_properties()

    def _from_tuple(self, value: Tuple[List[str], Any, int, int]) -> None:
        """Parse geometry from tuple format."""
        atoms, coords, charge, multi = value

        # Validate atoms
        if not isinstance(atoms, (list, tuple)):
            raise TypeError("Atoms must be a list or tuple.")
        
        if not all(isinstance(atom, str) for atom in atoms):
            raise TypeError("All atoms must be strings.")

        if len(atoms) == 0:
            raise ValueError("Atoms list cannot be empty.")

        # Validate and convert coordinates
        coords_array = np.array(coords, dtype=float)
        
        if coords_array.ndim != 2:
            raise ValueError(f"Coordinates must be 2D array, got {coords_array.ndim}D.")
        
        if coords_array.shape[1] != 3:
            raise ValueError(
                f"Coordinates must have 3 columns (x, y, z), got {coords_array.shape[1]}."
            )
        
        if len(atoms) != coords_array.shape[0]:
            raise ValueError(
                f"Mismatch: {len(atoms)} atoms but {coords_array.shape[0]} coordinate rows."
            )

        # Validate charge and multiplicity
        self.charge = int(charge)
        self.multi = int(multi)
        
        if self.multi < 1:
            raise ValueError(f"Multiplicity must be >= 1, got {self.multi}.")

        self.atoms = list(atoms)
        self.coords = coords_array
        self.coords_internal = self.coords.flatten()
        self.natoms = len(self.atoms)
        self._update_atomic_properties()

    def __repr__(self) -> str:
        """Return string representation of the Cartesian geometry."""
        return (
            f"Cartesian(natoms={self.natoms}, "
            f"charge={self.charge}, "
            f"multiplicity={self.multi})"
        )
