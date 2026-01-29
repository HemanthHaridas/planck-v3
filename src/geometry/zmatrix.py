"""Z-matrix representation of molecular geometry."""

from typing import Any, Dict, Union

import numpy as np

from src.geometry import base


class ZMatrix(base.BaseGeometry):
    """
    Concrete implementation of BaseGeometry using Z-matrix (internal coordinates).
    
    This class represents molecular geometry using internal coordinates:
    - Bond distances
    - Bond angles
    - Dihedral angles
    
    The Z-matrix is converted to Cartesian coordinates for storage and use.
    
    Attributes:
        atoms (List[str]): List of atomic symbols.
        coords (np.ndarray): Array of Cartesian coordinates with shape (N, 3).
        coords_internal (np.ndarray): Flattened 1D representation of coords.
        natoms (int): Number of atoms in the system.
        charge (int): Total molecular charge.
        multi (int): Spin multiplicity of the system.
        atomicnumbers (np.ndarray): Array of atomic numbers.
        atomicmasses (np.ndarray): Array of atomic masses.
    
    Z-matrix format:
        Line 1: natoms charge multiplicity
        Line 2: atom1_symbol
        Line 3: atom2_symbol bonded_to distance
        Line 4: atom3_symbol bonded_to angle_from distance angle
        Line 5+: atom_symbol bonded_to angle_from dihedral_from distance angle dihedral
    """
    
    def __init__(self) -> None:
        """Initialize a ZMatrix geometry object."""
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
    def geometry(self, value: str) -> None:
        """
        Set the molecular geometry from a Z-matrix string.
        
        Args:
            value: Z-matrix string representation.
            
        Raises:
            TypeError: If value is not a string.
            ValueError: If Z-matrix format is invalid or cannot be parsed.
        """
        if not isinstance(value, str):
            raise TypeError("Z-matrix must be a string.")
        
        if not value.strip():
            raise ValueError("Z-matrix string cannot be empty.")
        
        self._from_zmatrix_string(value)

    def _from_zmatrix_string(self, zmatrix_string: str) -> None:
        """Parse geometry from Z-matrix string format."""
        lines = [line.strip() for line in zmatrix_string.strip().split("\n") if line.strip()]
        
        if len(lines) < 3:
            raise ValueError(
                "Z-matrix must contain at least 3 lines "
                "(header, atom1, atom2)."
            )

        # Parse header
        header_tokens = lines[0].split()
        if len(header_tokens) < 3:
            raise ValueError(
                f"Header line must contain at least 3 values "
                f"(natoms charge multi), got: {lines[0]}"
            )
        
        try:
            self.natoms = int(header_tokens[0])
            self.charge = int(header_tokens[1])
            self.multi = int(header_tokens[2])
        except ValueError as e:
            raise ValueError(f"Invalid header line: {lines[0]}") from e
        
        if self.multi < 1:
            raise ValueError(f"Multiplicity must be >= 1, got {self.multi}.")

        atoms = []
        coords = []

        # Process first atom (always at origin)
        if len(lines) < 2:
            raise ValueError("Z-matrix must define at least one atom.")
        
        atom1_tokens = lines[1].split()
        if not atom1_tokens:
            raise ValueError(f"Empty atom line: {lines[1]}")
        
        atoms.append(atom1_tokens[0])
        coords.append([0.0, 0.0, 0.0])

        # Process second atom (on x-axis)
        if len(lines) < 3:
            raise ValueError("Z-matrix must define at least two atoms.")
        
        atom2_tokens = lines[2].split()
        if len(atom2_tokens) < 3:
            raise ValueError(
                f"Atom 2 line must have at least 3 tokens "
                f"(symbol bonded_to distance), got: {lines[2]}"
            )
        
        try:
            atoms.append(atom2_tokens[0])
            distance = float(atom2_tokens[2])
            coords.append([distance, 0.0, 0.0])
        except (ValueError, IndexError) as e:
            raise ValueError(f"Invalid atom 2 line: {lines[2]}") from e

        # Process third atom (in xy-plane)
        if len(lines) >= 4:
            atom3_tokens = lines[3].split()
            if len(atom3_tokens) < 5:
                raise ValueError(
                    f"Atom 3 line must have at least 5 tokens "
                    f"(symbol bonded_to angle_from distance angle), got: {lines[3]}"
                )
            
            try:
                atoms.append(atom3_tokens[0])
                bonded_to_idx = int(atom3_tokens[1]) - 1
                angle_from_idx = int(atom3_tokens[3]) - 1
                distance = float(atom3_tokens[2])
                angle = float(atom3_tokens[4])
                
                if bonded_to_idx < 0 or bonded_to_idx >= len(coords):
                    raise ValueError(
                        f"Invalid bonded_to index for atom 3: {bonded_to_idx + 1}"
                    )
                if angle_from_idx < 0 or angle_from_idx >= len(coords):
                    raise ValueError(
                        f"Invalid angle_from index for atom 3: {angle_from_idx + 1}"
                    )
                
                coords.append(
                    self._place_third_atom(
                        coords[bonded_to_idx],
                        coords[angle_from_idx],
                        distance,
                        angle
                    )
                )
            except (ValueError, IndexError) as e:
                raise ValueError(f"Invalid atom 3 line: {lines[3]}") from e

        # Process remaining atoms (with dihedral)
        for i in range(4, len(lines)):
            tokens = lines[i].split()
            if len(tokens) < 7:
                raise ValueError(
                    f"Atom {i+1} line must have at least 7 tokens "
                    f"(symbol bonded_to angle_from dihedral_from distance angle dihedral), "
                    f"got: {lines[i]}"
                )
            
            try:
                atoms.append(tokens[0])
                bonded_to_idx = int(tokens[1]) - 1
                angle_from_idx = int(tokens[3]) - 1
                dihedral_from_idx = int(tokens[5]) - 1
                distance = float(tokens[2])
                angle = float(tokens[4])
                dihedral = float(tokens[6])
                
                # Validate indices
                for idx, name in [
                    (bonded_to_idx, "bonded_to"),
                    (angle_from_idx, "angle_from"),
                    (dihedral_from_idx, "dihedral_from"),
                ]:
                    if idx < 0 or idx >= len(coords):
                        raise ValueError(
                            f"Invalid {name} index for atom {i+1}: {idx + 1}"
                        )
                
                coords.append(
                    self._place_atom_with_dihedral(
                        coords[bonded_to_idx],
                        coords[angle_from_idx],
                        coords[dihedral_from_idx],
                        distance,
                        angle,
                        dihedral
                    )
                )
            except (ValueError, IndexError) as e:
                raise ValueError(f"Invalid atom {i+1} line: {lines[i]}") from e

        # Validate atom count
        if len(atoms) != self.natoms:
            raise ValueError(
                f"Mismatch: header specifies {self.natoms} atoms, "
                f"but {len(atoms)} atoms were defined."
            )

        self.atoms = atoms
        self.coords = np.array(coords, dtype=float)
        self.coords_internal = self.coords.flatten()
        self._update_atomic_properties()

    def _place_third_atom(
        self,
        bonded_to: np.ndarray,
        angle_from: np.ndarray,
        distance: float,
        angle_deg: float,
    ) -> np.ndarray:
        """
        Place third atom in xy-plane using bond distance and angle.
        
        Args:
            bonded_to: Coordinates of atom to bond to.
            angle_from: Coordinates of atom to measure angle from.
            distance: Bond distance.
            angle_deg: Bond angle in degrees.
            
        Returns:
            Coordinates of the new atom.
        """
        # Compute bond vector
        bond_vector = np.array(bonded_to) - np.array(angle_from)
        bond_norm = np.linalg.norm(bond_vector)
        
        if bond_norm < 1e-10:
            raise ValueError("Bonded atom and angle reference atom are too close.")
        
        bond_vector = bond_vector / bond_norm
        
        # Construct perpendicular vector in xy-plane
        perp = np.array([bond_vector[1], -bond_vector[0], 0.0])
        perp_norm = np.linalg.norm(perp)
        
        if perp_norm < 1e-10:
            # If bond is along z-axis, use different perpendicular
            perp = np.array([1.0, 0.0, 0.0])
        else:
            perp = perp / perp_norm
        
        # Rotate bond vector by angle
        angle_rad = np.radians(angle_deg)
        rotated = (
            np.cos(angle_rad) * bond_vector +
            np.sin(angle_rad) * perp
        )
        
        # Place new atom
        return np.array(bonded_to) + distance * rotated

    def _place_atom_with_dihedral(
        self,
        bonded_to: np.ndarray,
        angle_from: np.ndarray,
        dihedral_from: np.ndarray,
        distance: float,
        angle_deg: float,
        dihedral_deg: float,
    ) -> np.ndarray:
        """
        Place atom using bond distance, angle, and dihedral.
        
        Args:
            bonded_to: Coordinates of atom to bond to.
            angle_from: Coordinates of atom to measure angle from.
            dihedral_from: Coordinates of atom to measure dihedral from.
            distance: Bond distance.
            angle_deg: Bond angle in degrees.
            dihedral_deg: Dihedral angle in degrees.
            
        Returns:
            Coordinates of the new atom.
        """
        # Define local coordinate system
        p_bond = np.array(bonded_to)
        p_angle = np.array(angle_from)
        p_dihedral = np.array(dihedral_from)
        
        v1 = p_bond - p_angle
        v2 = p_dihedral - p_angle
        
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        if v1_norm < 1e-10:
            raise ValueError("Bonded atom and angle reference atom are too close.")
        if v2_norm < 1e-10:
            raise ValueError("Dihedral reference atom and angle reference atom are too close.")
        
        v1 = v1 / v1_norm
        v2 = v2 / v2_norm
        
        # Normal to plane defined by v1 and v2
        n = np.cross(v1, v2)
        n_norm = np.linalg.norm(n)
        
        if n_norm < 1e-10:
            # v1 and v2 are parallel, use alternative method
            perp = np.array([1.0, 0.0, 0.0]) if abs(v1[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            n = np.cross(v1, perp)
            n = n / np.linalg.norm(n)
        else:
            n = n / n_norm
        
        # Perpendicular vector in plane
        m = np.cross(n, v1)
        
        # Convert angles to radians
        theta = np.radians(angle_deg)
        phi = np.radians(dihedral_deg)
        
        # Build new coordinate relative to bonded atom
        direction = (
            np.cos(theta) * (-v1) +
            np.sin(theta) * (np.cos(phi) * m + np.sin(phi) * n)
        )
        
        return p_bond + distance * direction

    def __repr__(self) -> str:
        """Return string representation of the ZMatrix geometry."""
        return (
            f"ZMatrix(natoms={self.natoms}, "
            f"charge={self.charge}, "
            f"multiplicity={self.multi})"
        )
