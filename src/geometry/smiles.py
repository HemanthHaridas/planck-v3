import numpy
import typing
import rdkit
from src.geometry import base


class CartesianSmiles(base.BaseGeometry):
    def __init__(self):
        super().__init__()

    @property
    def geometry(self) -> dict:
        return {
            "atoms": self.atoms, 
            "coords": self.coords,
            "charge": self.charge,
            "multi": self.multi,
            "atomicnumbers": self.atomicnumbers,
            "atomicmasses": self.atomicmass
        }

    @geometry.setter
    def geometry(self, value):
        if not isinstance(value, str):
            raise TypeError("SMILES must be string. Please check your input.")
