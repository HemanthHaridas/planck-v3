"""Unrestricted Hartree-Fock (UHF) calculations."""

from typing import Optional, Union

from src.geometry import BaseGeometry
from src.hartree_fock.base import BaseHartreeFock
from src.hartree_fock.integrals import IntegralMethod


class UnrestrictedHartreeFock(BaseHartreeFock):
    """
    Unrestricted Hartree-Fock (UHF) calculation.
    
    UHF is applicable to both closed-shell and open-shell systems:
    - Any valid multiplicity (>= 1)
    - Any number of electrons (even or odd)
    - Alpha and beta orbitals can differ (unrestricted)
    
    In UHF, alpha and beta orbitals are solved separately, allowing for
    different spatial orbitals for alpha and beta electrons. This makes
    UHF suitable for:
    - Open-shell systems (radicals, triplets, etc.)
    - Systems with odd number of electrons
    - Systems where spin polarization is important
    
    Attributes:
        geometry (BaseGeometry): Molecular geometry object.
        nelectrons (int): Total number of electrons.
        nalpha (int): Number of alpha electrons.
        nbeta (int): Number of beta electrons.
        converged (bool): Whether the HF calculation has converged.
        energy (Optional[float]): Total electronic energy.
    
    Example (Open-shell radical):
        >>> from src.geometry import Cartesian
        >>> from src.hartree_fock import UnrestrictedHartreeFock, IntegralMethod
        >>> 
        >>> # Create geometry for methyl radical (CH3)
        >>> geom = Cartesian()
        >>> geom.geometry = (["C", "H", "H", "H"], 
        ...                  [[0.0, 0.0, 0.0], 
        ...                   [1.0, 0.0, 0.0], 
        ...                   [-0.5, 0.866, 0.0],
        ...                   [-0.5, -0.866, 0.0]], 
        ...                  0, 2)  # charge=0, multi=2 (doublet)
        >>> 
        >>> # Create UHF calculator with Huzinaga integrals (default)
        >>> uhf = UnrestrictedHartreeFock(geom)
        >>> # uhf.compute()  # Would perform actual UHF calculation
        >>> 
        >>> # Or use Hermite integrals
        >>> uhf_hermite = UnrestrictedHartreeFock(geom, integral_method=IntegralMethod.HERMITE)
    
    Example (Closed-shell, same as RHF):
        >>> # Water molecule (can also use RHF)
        >>> geom = Cartesian()
        >>> geom.geometry = (["H", "O", "H"], 
        ...                  [[0.0, 0.0, 0.0], 
        ...                   [0.0, 0.0, 0.957], 
        ...                   [0.957, 0.0, 0.0]], 
        ...                  0, 1)  # charge=0, multi=1 (singlet)
        >>> 
        >>> uhf = UnrestrictedHartreeFock(geom)  # Uses Huzinaga by default
        >>> # For closed-shell singlet, UHF should give same result as RHF
    """
    
    def __init__(
        self,
        geometry: BaseGeometry,
        integral_method: Union[IntegralMethod, str] = IntegralMethod.HUZINAGA,
    ) -> None:
        """
        Initialize Unrestricted Hartree-Fock calculation.
        
        Args:
            geometry: Molecular geometry object. UHF accepts any valid
                     geometry with multiplicity >= 1 and any electron count.
            integral_method: Integral method to use. Can be "hermite" or "huzinaga".
                           Defaults to "huzinaga".
        
        Note:
            Unlike RHF, UHF does not enforce restrictions on multiplicity
            or electron count in __init__. Validation happens in compute().
        """
        super().__init__(geometry, integral_method=integral_method)
    
    def validate_multiplicity(self) -> None:
        """
        Validate multiplicity for UHF.
        
        UHF accepts any valid multiplicity (>= 1), but the multiplicity must
        be consistent with the electron count. The relationship is:
        - Multiplicity = 2S + 1, where S is total spin
        - nalpha - nbeta = 2S = multiplicity - 1
        
        For example:
        - Multiplicity 1 (singlet): nalpha - nbeta = 0 (closed-shell)
        - Multiplicity 2 (doublet): nalpha - nbeta = 1
        - Multiplicity 3 (triplet): nalpha - nbeta = 2
        
        Raises:
            ValueError: If multiplicity is invalid or inconsistent with electron count.
        """
        if self.geometry.multi < 1:
            raise ValueError(
                f"Multiplicity must be >= 1, got multi={self.geometry.multi}"
            )
        
        # Check that multiplicity is consistent with electron count
        # nalpha - nbeta = multiplicity - 1
        expected_spin_excess = self.geometry.multi - 1
        actual_spin_excess = self.nalpha - self.nbeta
        
        if actual_spin_excess != expected_spin_excess:
            raise ValueError(
                f"Multiplicity ({self.geometry.multi}) is inconsistent with "
                f"electron configuration. "
                f"Expected nalpha - nbeta = {expected_spin_excess} "
                f"(for multi={self.geometry.multi}), "
                f"but got nalpha={self.nalpha}, nbeta={self.nbeta} "
                f"(difference={actual_spin_excess}). "
                f"For multiplicity {self.geometry.multi}, "
                f"the number of alpha and beta electrons must differ by {expected_spin_excess}."
            )
    
    def validate_electron_count(self) -> None:
        """
        Validate electron count for UHF.
        
        UHF accepts any number of electrons (even or odd), but the electron
        count must be consistent with the multiplicity. The relationship is:
        - nalpha + nbeta = nelectrons
        - nalpha - nbeta = multiplicity - 1
        
        This means: nelectrons + (multiplicity - 1) must be even.
        
        For example:
        - Multiplicity 1: nelectrons can be any even number
        - Multiplicity 2: nelectrons can be any odd number
        - Multiplicity 3: nelectrons can be any even number
        
        Raises:
            ValueError: If electron count is invalid or inconsistent with multiplicity.
        """
        if self.nelectrons < 0:
            raise ValueError(
                f"Number of electrons must be >= 0, got nelectrons={self.nelectrons}"
            )
        
        if self.nelectrons == 0:
            raise ValueError(
                "UHF requires at least one electron. "
                "Cannot perform calculation on system with no electrons."
            )
        
        # Check consistency: nelectrons + (multi - 1) must be even
        # This ensures nalpha and nbeta are integers
        spin_excess = self.geometry.multi - 1
        if (self.nelectrons + spin_excess) % 2 != 0:
            raise ValueError(
                f"Electron count ({self.nelectrons}) is inconsistent with "
                f"multiplicity ({self.geometry.multi}). "
                f"For multiplicity {self.geometry.multi}, "
                f"nelectrons + (multi - 1) = {self.nelectrons + spin_excess} "
                f"must be even to give integer nalpha and nbeta. "
                f"This would require nalpha={self.nelectrons + spin_excess}/2, "
                f"nbeta={self.nelectrons - spin_excess}/2 (non-integer)."
            )
        
        # Verify that nalpha and nbeta match the expected relationship
        expected_nalpha = (self.nelectrons + spin_excess) // 2
        expected_nbeta = (self.nelectrons - spin_excess) // 2
        
        if self.nalpha != expected_nalpha or self.nbeta != expected_nbeta:
            raise ValueError(
                f"Electron configuration mismatch: "
                f"Expected nalpha={expected_nalpha}, nbeta={expected_nbeta} "
                f"(for nelectrons={self.nelectrons}, multi={self.geometry.multi}), "
                f"but got nalpha={self.nalpha}, nbeta={self.nbeta}."
            )
        
        # Ensure nalpha and nbeta are non-negative
        if self.nalpha < 0 or self.nbeta < 0:
            raise ValueError(
                f"Invalid electron configuration: "
                f"nalpha={self.nalpha}, nbeta={self.nbeta} "
                f"(for nelectrons={self.nelectrons}, multi={self.geometry.multi}). "
                f"Both must be non-negative."
            )
    
    def _compute_hf_energy(self) -> float:
        """
        Perform Unrestricted Hartree-Fock calculation.
        
        This implementation uses the selected integral method (Hermite or Huzinaga)
        to compute molecular integrals. The actual SCF procedure would:
        1. Build basis set from geometry
        2. Compute overlap and kinetic integrals using the selected method
        3. Build one-electron Hamiltonian (kinetic + nuclear attraction)
        4. Compute two-electron integrals
        5. Initialize guess orbitals (separate for alpha and beta)
        6. Perform UHF SCF iterations until convergence
        7. Return the total electronic energy
        
        Key difference from RHF:
        - Separate Fock matrices for alpha and beta electrons
        - Separate orbital coefficients for alpha and beta
        - Can have different number of occupied orbitals for alpha and beta
        
        Currently returns a placeholder energy. The integral calculator is
        available via self.integral_calculator for use in full implementations.
        
        Returns:
            float: Total electronic energy (placeholder: 0.0).
        
        Note:
            The integral calculator is initialized and ready to use:
            - self.integral_calculator.overlap(contractedA, contractedB)
            - self.integral_calculator.kinetic(contractedA, contractedB)
        """
        # Placeholder: mark as converged and return zero energy
        # In a real implementation, this would:
        # 1. Build contracted Gaussian basis functions from geometry
        # 2. Compute overlap matrix S using self.integral_calculator.overlap()
        # 3. Compute kinetic energy matrix T using self.integral_calculator.kinetic()
        # 4. Build nuclear attraction integrals
        # 5. Build two-electron integrals
        # 6. Perform UHF SCF iterations with separate alpha/beta Fock matrices
        
        self.converged = True
        
        # Example of how integrals would be computed:
        # from src.integrals import Contracted, Primitive
        # contracted = Contracted()
        # contracted.location_x = self.geometry.coords[0, 0]
        # ... set up contracted Gaussian ...
        # S_ij = self.integral_calculator.overlap(contracted_i, contracted_j)
        # T_ij = self.integral_calculator.kinetic(contracted_i, contracted_j)
        # Then build F_alpha and F_beta separately
        
        return 0.0
    
    def __repr__(self) -> str:
        """Return string representation of the UHF calculation."""
        return (
            f"UnrestrictedHartreeFock("
            f"nelectrons={self.nelectrons}, "
            f"multi={self.geometry.multi}, "
            f"charge={self.geometry.charge}, "
            f"nalpha={self.nalpha}, "
            f"nbeta={self.nbeta}, "
            f"converged={self.converged}"
            ")"
        )
