"""Restricted Hartree-Fock (RHF) calculations."""

from typing import Optional, Union

from src.geometry import BaseGeometry
from src.hartree_fock.base import BaseHartreeFock
from src.hartree_fock.integrals import IntegralMethod


class RestrictedHartreeFock(BaseHartreeFock):
    """
    Restricted Hartree-Fock (RHF) calculation.
    
    RHF is applicable only to closed-shell systems:
    - Multiplicity must be 1 (singlet, closed-shell)
    - Number of electrons must be even
    
    In RHF, alpha and beta orbitals are identical, so nalpha = nbeta = nelectrons / 2.
    
    Attributes:
        geometry (BaseGeometry): Molecular geometry object.
        nelectrons (int): Total number of electrons (must be even).
        nalpha (int): Number of alpha electrons (= nelectrons / 2).
        nbeta (int): Number of beta electrons (= nelectrons / 2).
        converged (bool): Whether the HF calculation has converged.
        energy (Optional[float]): Total electronic energy.
    
    Example:
        >>> from src.geometry import Cartesian
        >>> from src.hartree_fock import RestrictedHartreeFock, IntegralMethod
        >>> 
        >>> # Create geometry for water (H2O)
        >>> geom = Cartesian()
        >>> geom.geometry = (["H", "O", "H"], 
        ...                  [[0.0, 0.0, 0.0], 
        ...                   [0.0, 0.0, 0.957], 
        ...                   [0.957, 0.0, 0.0]], 
        ...                  0, 1)  # charge=0, multi=1
        >>> 
        >>> # Create RHF calculator with Huzinaga integrals (default)
        >>> rhf = RestrictedHartreeFock(geom)
        >>> # rhf.compute()  # Would perform actual HF calculation
        >>> 
        >>> # Or specify Hermite integrals explicitly
        >>> rhf_hermite = RestrictedHartreeFock(geom, integral_method="hermite")
        >>> # rhf_hermite.compute()
    """
    
    def __init__(
        self,
        geometry: BaseGeometry,
        integral_method: Union[IntegralMethod, str] = IntegralMethod.HUZINAGA,
    ) -> None:
        """
        Initialize Restricted Hartree-Fock calculation.
        
        Args:
            geometry: Molecular geometry object. Must have multiplicity=1
                     and even number of electrons for RHF to be valid.
            integral_method: Integral method to use. Can be "hermite" or "huzinaga".
                           Defaults to "huzinaga".
        
        Raises:
            ValueError: If geometry has odd number of electrons or multi != 1.
        """
        super().__init__(geometry, integral_method=integral_method)
        
        # For RHF, we can verify alpha = beta = nelectrons / 2
        if self.nalpha != self.nbeta:
            raise ValueError(
                f"RHF requires nalpha == nbeta, but got "
                f"nalpha={self.nalpha}, nbeta={self.nbeta}. "
                f"This indicates multi != 1 or odd electron count."
            )
    
    def validate_multiplicity(self) -> None:
        """
        Validate that multiplicity is 1 (singlet) for RHF.
        
        RHF is only valid for closed-shell singlet systems.
        
        Raises:
            ValueError: If multiplicity is not 1.
        """
        if self.geometry.multi != 1:
            raise ValueError(
                f"Restricted Hartree-Fock requires multiplicity=1 (singlet), "
                f"but got multi={self.geometry.multi}. "
                f"Use UHF (Unrestricted Hartree-Fock) for open-shell systems."
            )
    
    def validate_electron_count(self) -> None:
        """
        Validate that electron count is even for RHF.
        
        RHF requires an even number of electrons because it assumes
        paired electrons (closed-shell).
        
        Raises:
            ValueError: If number of electrons is odd.
        """
        if self.nelectrons % 2 != 0:
            raise ValueError(
                f"Restricted Hartree-Fock requires an even number of electrons, "
                f"but got nelectrons={self.nelectrons}. "
                f"Use UHF (Unrestricted Hartree-Fock) for odd-electron systems."
            )
        
        # Additional check: alpha and beta should be equal
        if self.nalpha != self.nbeta:
            raise ValueError(
                f"RHF requires nalpha == nbeta, but got "
                f"nalpha={self.nalpha}, nbeta={self.nbeta}. "
                f"This should not happen if multi=1 and nelectrons is even."
            )
    
    def _compute_hf_energy(self) -> float:
        """
        Perform Restricted Hartree-Fock calculation.
        
        This implementation uses the selected integral method (Hermite or Huzinaga)
        to compute molecular integrals. The actual SCF procedure would:
        1. Build basis set from geometry
        2. Compute overlap and kinetic integrals using the selected method
        3. Build one-electron Hamiltonian (kinetic + nuclear attraction)
        4. Compute two-electron integrals
        5. Initialize guess orbitals
        6. Perform SCF iterations until convergence
        7. Return the total electronic energy
        
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
        # 6. Perform SCF iterations
        
        self.converged = True
        
        # Example of how integrals would be computed:
        # from src.integrals import Contracted, Primitive
        # contracted = Contracted()
        # contracted.location_x = self.geometry.coords[0, 0]
        # ... set up contracted Gaussian ...
        # S_ij = self.integral_calculator.overlap(contracted_i, contracted_j)
        # T_ij = self.integral_calculator.kinetic(contracted_i, contracted_j)
        
        return 0.0
    
    def __repr__(self) -> str:
        """Return string representation of the RHF calculation."""
        return (
            f"RestrictedHartreeFock("
            f"nelectrons={self.nelectrons}, "
            f"multi={self.geometry.multi}, "
            f"charge={self.geometry.charge}, "
            f"nalpha={self.nalpha}, "
            f"nbeta={self.nbeta}, "
            f"converged={self.converged}"
            ")"
        )
