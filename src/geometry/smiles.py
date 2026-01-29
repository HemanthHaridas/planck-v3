"""SMILES string representation of molecular geometry."""

from typing import Any, Dict

import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

from src.geometry import base


class CartesianSmiles(base.BaseGeometry):
    """
    Concrete implementation of BaseGeometry using SMILES strings.
    
    This class converts SMILES strings to 3D Cartesian coordinates using RDKit.
    The SMILES string is parsed and a 3D geometry is generated using RDKit's
    conformer generation capabilities.
    
    Attributes:
        smiles (str): The SMILES string representation.
        atoms (List[str]): List of atomic symbols.
        coords (np.ndarray): Array of Cartesian coordinates with shape (N, 3).
        coords_internal (np.ndarray): Flattened 1D representation of coords.
        natoms (int): Number of atoms in the system.
        charge (int): Total molecular charge.
        multi (int): Spin multiplicity of the system.
        atomicnumbers (np.ndarray): Array of atomic numbers.
        atomicmasses (np.ndarray): Array of atomic masses.
    
    Example:
        >>> geom = CartesianSmiles()
        >>> geom.geometry = "CCO"  # Ethanol
        >>> geom.natoms
        9
    """
    
    def __init__(self) -> None:
        """
        Initialize a CartesianSmiles geometry object.
        
        Raises:
            ImportError: If RDKit is not installed.
        """
        if not RDKIT_AVAILABLE:
            raise ImportError(
                "RDKit is required for SMILES parsing. "
                "Install it with: pip install rdkit"
            )
        super().__init__()
        self.smiles: str = ""

    @property
    def geometry(self) -> Dict[str, Any]:
        """
        Get the molecular geometry in Cartesian representation.
        
        Returns:
            dict: A dictionary containing:
                - smiles (str): The SMILES string.
                - atoms (List[str]): Atomic symbols.
                - coords (np.ndarray): Cartesian coordinates (N, 3).
                - charge (int): Molecular charge.
                - multi (int): Spin multiplicity.
                - atomicnumbers (np.ndarray): Atomic numbers.
                - atomicmasses (np.ndarray): Atomic masses.
        """
        return {
            "smiles": self.smiles,
            "atoms": self.atoms,
            "coords": self.coords,
            "charge": self.charge,
            "multi": self.multi,
            "atomicnumbers": self.atomicnumbers,
            "atomicmasses": self.atomicmasses,
        }

    @geometry.setter
    def geometry(self, value: str) -> None:
        """
        Set the molecular geometry from a SMILES string.
        
        The SMILES string is parsed using RDKit, and a 3D conformer is generated.
        The charge is computed from the formal charges in the molecule, and
        multiplicity defaults to 1 (singlet).
        
        Args:
            value: SMILES string representation of the molecule.
            
        Raises:
            TypeError: If value is not a string.
            ValueError: If SMILES string cannot be parsed or 3D geometry cannot be generated.
        """
        if not isinstance(value, str):
            raise TypeError("SMILES must be a string. Please check your input.")
        
        if not value.strip():
            raise ValueError("SMILES string cannot be empty.")
        
        # Parse SMILES string
        mol = Chem.MolFromSmiles(value)
        if mol is None:
            raise ValueError(f"Failed to parse SMILES string: {value}")
        
        # Add hydrogens
        mol = Chem.AddHs(mol)
        
        # Generate 3D coordinates
        try:
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
        except Exception as e:
            raise ValueError(
                f"Failed to generate 3D coordinates from SMILES: {value}. "
                f"Error: {e}"
            ) from e
        
        # Extract atoms and coordinates
        conf = mol.GetConformer()
        atoms = []
        coords = []
        
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            atoms.append(symbol)
        
        for i in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            coords.append([pos.x, pos.y, pos.z])
        
        # Calculate formal charge
        formal_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
        
        self.smiles = value
        self.atoms = atoms
        self.coords = np.array(coords, dtype=float)
        self.coords_internal = self.coords.flatten()
        self.natoms = len(self.atoms)
        self.charge = formal_charge
        self.multi = 1  # Default to singlet
        self._update_atomic_properties()

    def __repr__(self) -> str:
        """Return string representation of the SMILES geometry."""
        return (
            f"CartesianSmiles(smiles='{self.smiles}', "
            f"natoms={self.natoms}, "
            f"charge={self.charge}, "
            f"multiplicity={self.multi})"
        )
