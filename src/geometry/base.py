"""Base geometry class for molecular geometry representations."""

import abc
from typing import Any, List, Optional

import numpy as np


class BaseGeometry(abc.ABC):
    """
    Abstract Base Class (ABC) for molecular geometry representations.
    
    This class enforces the implementation of a `geometry` property in subclasses.
    Subclasses must define how molecular geometry information is stored and accessed.
    """

    # Periodic table data: atomic number -> (symbol, atomic_mass)
    _PERIODIC_TABLE = {
        1: ("H", 1.008), 2: ("He", 4.003), 3: ("Li", 6.941), 4: ("Be", 9.012),
        5: ("B", 10.81), 6: ("C", 12.01), 7: ("N", 14.01), 8: ("O", 16.00),
        9: ("F", 19.00), 10: ("Ne", 20.18), 11: ("Na", 22.99), 12: ("Mg", 24.31),
        13: ("Al", 26.98), 14: ("Si", 28.09), 15: ("P", 30.97), 16: ("S", 32.07),
        17: ("Cl", 35.45), 18: ("Ar", 39.95), 19: ("K", 39.10), 20: ("Ca", 40.08),
        21: ("Sc", 44.96), 22: ("Ti", 47.87), 23: ("V", 50.94), 24: ("Cr", 52.00),
        25: ("Mn", 54.94), 26: ("Fe", 55.85), 27: ("Co", 58.93), 28: ("Ni", 58.69),
        29: ("Cu", 63.55), 30: ("Zn", 65.38), 31: ("Ga", 69.72), 32: ("Ge", 72.64),
        33: ("As", 74.92), 34: ("Se", 78.96), 35: ("Br", 79.90), 36: ("Kr", 83.80),
    }

    # Reverse lookup: symbol -> atomic number
    _SYMBOL_TO_NUMBER = {symbol: num for num, (symbol, _) in _PERIODIC_TABLE.items()}

    def __init__(self) -> None:
        """
        Initialize the base geometry object with default molecular attributes.

        Attributes:
            atoms (List[str]): List of atomic symbols.
            coords (np.ndarray): Array of Cartesian coordinates with shape (N, 3).
            atomicnumbers (np.ndarray): 1D array of atomic numbers.
            atomicmasses (np.ndarray): 1D array of atomic masses.
            coords_internal (np.ndarray): Flattened representation of coords.
            natoms (Optional[int]): Number of atoms in the system.
            charge (int): Total molecular charge (default: 0).
            multi (int): Spin multiplicity (default: 1).
        """
        self.atoms: List[str] = []
        self.coords: np.ndarray = np.zeros((0, 3), dtype=float)
        self.atomicnumbers: np.ndarray = np.array([], dtype=int)
        self.atomicmasses: np.ndarray = np.array([], dtype=float)
        self.coords_internal: np.ndarray = np.array([], dtype=float)
        self.natoms: Optional[int] = None
        self.charge: int = 0
        self.multi: int = 1

    def _update_atomic_properties(self) -> None:
        """
        Update atomic numbers and masses based on atom symbols.
        
        This method should be called whenever the atoms list is modified.
        """
        if not self.atoms:
            self.atomicnumbers = np.array([], dtype=int)
            self.atomicmasses = np.array([], dtype=float)
            return

        atomic_numbers = []
        atomic_masses = []
        
        for symbol in self.atoms:
            atomic_num = self._SYMBOL_TO_NUMBER.get(symbol)
            if atomic_num is None:
                raise ValueError(
                    f"Unknown atomic symbol: {symbol}. "
                    f"Supported elements: {sorted(set(self._SYMBOL_TO_NUMBER.keys()))}"
                )
            atomic_numbers.append(atomic_num)
            atomic_masses.append(self._PERIODIC_TABLE[atomic_num][1])
        
        self.atomicnumbers = np.array(atomic_numbers, dtype=int)
        self.atomicmasses = np.array(atomic_masses, dtype=float)

    @property
    @abc.abstractmethod
    def geometry(self) -> Any:
        """
        Abstract property representing the molecular geometry.
        
        Returns:
            Any: The internal representation of the geometry.
        """
        pass

    @geometry.setter
    @abc.abstractmethod
    def geometry(self, value: Any) -> None:
        """
        Abstract setter for the molecular geometry.
        
        Args:
            value: The new geometry representation to assign.
        """
        pass
