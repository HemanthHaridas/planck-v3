"""Base class for Hartree-Fock calculations."""

import abc
from typing import Optional, Union

import numpy as np

from src.geometry import BaseGeometry
from src.hartree_fock.integrals import IntegralCalculator, IntegralMethod


class BaseHartreeFock(abc.ABC):
    """
    Abstract base class for Hartree-Fock calculations.
    
    This class provides template methods for validating molecular properties
    (multiplicity and electron count) before performing HF calculations.
    Subclasses must implement the actual HF calculation logic and define
    their specific validation rules.
    
    Attributes:
        geometry (BaseGeometry): Molecular geometry object.
        nelectrons (int): Total number of electrons in the system.
        nalpha (int): Number of alpha electrons.
        nbeta (int): Number of beta electrons.
        converged (bool): Whether the HF calculation has converged.
        energy (Optional[float]): Total electronic energy.
        integral_calculator (IntegralCalculator): Calculator for molecular integrals.
    
    Template Method Pattern:
        The `compute()` method follows the template method pattern:
        1. Validate geometry
        2. Calculate electron count
        3. Validate multiplicity (template method)
        4. Validate electron count (template method)
        5. Perform HF calculation (abstract method)
    """
    
    def __init__(
        self,
        geometry: BaseGeometry,
        integral_method: Union[IntegralMethod, str] = IntegralMethod.HUZINAGA,
    ) -> None:
        """
        Initialize Hartree-Fock calculation with a geometry object.
        
        Args:
            geometry: Molecular geometry object containing atoms, coordinates,
                     charge, and multiplicity information.
            integral_method: Integral method to use. Can be "hermite" or "huzinaga".
                           Defaults to "huzinaga".
        
        Raises:
            TypeError: If geometry is not a BaseGeometry instance.
            ValueError: If geometry has no atoms defined.
            ImportError: If the C++ integral module is not available.
        """
        if not isinstance(geometry, BaseGeometry):
            raise TypeError(
                f"geometry must be a BaseGeometry instance, "
                f"got {type(geometry).__name__}"
            )
        
        if geometry.natoms is None or geometry.natoms == 0:
            raise ValueError("Geometry must have at least one atom defined.")
        
        self.geometry = geometry
        self.nelectrons: int = 0
        self.nalpha: int = 0
        self.nbeta: int = 0
        self.converged: bool = False
        self.energy: Optional[float] = None
        
        # Initialize integral calculator
        self.integral_calculator = IntegralCalculator(method=integral_method)
        
        # Calculate electron count from geometry
        self._calculate_electron_count()
    
    def _calculate_electron_count(self) -> None:
        """
        Calculate the total number of electrons from atomic numbers and charge.
        
        Number of electrons = sum of atomic numbers - molecular charge
        
        This method also calculates alpha and beta electron counts based on
        multiplicity: multi = 2S + 1, where S is total spin.
        For N electrons: nalpha - nbeta = 2S, and nalpha + nbeta = N
        Solving: nalpha = (N + 2S) / 2, nbeta = (N - 2S) / 2
        """
        if len(self.geometry.atomicnumbers) == 0:
            self.nelectrons = 0
            self.nalpha = 0
            self.nbeta = 0
            return
        
        # Total electrons = sum of atomic numbers - charge
        total_atomic_electrons = int(np.sum(self.geometry.atomicnumbers))
        self.nelectrons = total_atomic_electrons - self.geometry.charge
        
        if self.nelectrons < 0:
            raise ValueError(
                f"Invalid electron count: {self.nelectrons}. "
                f"Charge ({self.geometry.charge}) cannot exceed "
                f"total atomic electrons ({total_atomic_electrons})."
            )
        
        # Calculate spin from multiplicity: multi = 2S + 1
        # S = (multi - 1) / 2
        # nalpha - nbeta = 2S = multi - 1
        # nalpha + nbeta = nelectrons
        # Solving: nalpha = (nelectrons + multi - 1) / 2
        #          nbeta = (nelectrons - multi + 1) / 2
        
        spin_excess = self.geometry.multi - 1
        self.nalpha = (self.nelectrons + spin_excess) // 2
        self.nbeta = (self.nelectrons - spin_excess) // 2
        
        # Validate that alpha and beta counts are non-negative integers
        if self.nalpha < 0 or self.nbeta < 0:
            raise ValueError(
                f"Invalid electron configuration: "
                f"nelectrons={self.nelectrons}, multi={self.geometry.multi}. "
                f"This would give nalpha={self.nalpha}, nbeta={self.nbeta}."
            )
        
        # Check if counts are integers (should always be true for valid multi)
        if (self.nelectrons + spin_excess) % 2 != 0:
            raise ValueError(
                f"Invalid combination: nelectrons={self.nelectrons}, "
                f"multi={self.geometry.multi}. "
                f"nelectrons + (multi - 1) must be even."
            )
    
    @abc.abstractmethod
    def validate_multiplicity(self) -> None:
        """
        Template method to validate multiplicity for this HF method.
        
        Subclasses must implement this to enforce their specific multiplicity
        requirements (e.g., RHF requires multi=1, UHF allows any valid multi).
        
        Raises:
            ValueError: If multiplicity is invalid for this HF method.
        """
        pass
    
    @abc.abstractmethod
    def validate_electron_count(self) -> None:
        """
        Template method to validate electron count for this HF method.
        
        Subclasses must implement this to enforce their specific electron count
        requirements (e.g., RHF requires even number of electrons).
        
        Raises:
            ValueError: If electron count is invalid for this HF method.
        """
        pass
    
    def validate_geometry(self) -> None:
        """
        Validate that the geometry object is properly initialized.
        
        Raises:
            ValueError: If geometry is invalid or incomplete.
        """
        if self.geometry.natoms is None or self.geometry.natoms == 0:
            raise ValueError("Geometry must have at least one atom.")
        
        if len(self.geometry.atoms) != self.geometry.natoms:
            raise ValueError(
                f"Mismatch: natoms={self.geometry.natoms} but "
                f"len(atoms)={len(self.geometry.atoms)}"
            )
        
        if len(self.geometry.atomicnumbers) != self.geometry.natoms:
            raise ValueError(
                f"Mismatch: natoms={self.geometry.natoms} but "
                f"len(atomicnumbers)={len(self.geometry.atomicnumbers)}"
            )
        
        if self.geometry.multi < 1:
            raise ValueError(
                f"Multiplicity must be >= 1, got {self.geometry.multi}"
            )
    
    def compute(self) -> float:
        """
        Template method to perform Hartree-Fock calculation.
        
        This method follows the template method pattern:
        1. Validate geometry
        2. Calculate electron count
        3. Validate multiplicity (template method)
        4. Validate electron count (template method)
        5. Perform HF calculation (abstract method)
        
        Returns:
            float: Total electronic energy.
        
        Raises:
            ValueError: If validation fails at any step.
        """
        # Step 1: Validate geometry
        self.validate_geometry()
        
        # Step 2: Calculate electron count (already done in __init__, but recalculate)
        self._calculate_electron_count()
        
        # Step 3: Validate multiplicity (template method)
        self.validate_multiplicity()
        
        # Step 4: Validate electron count (template method)
        self.validate_electron_count()
        
        # Step 5: Perform HF calculation (abstract method)
        self.energy = self._compute_hf_energy()
        return self.energy
    
    @abc.abstractmethod
    def _compute_hf_energy(self) -> float:
        """
        Abstract method to perform the actual HF calculation.
        
        Subclasses must implement this method to perform the specific
        HF calculation (RHF, UHF, ROHF, etc.).
        
        Returns:
            float: Total electronic energy.
        """
        pass
    
    def __repr__(self) -> str:
        """Return string representation of the HF calculation."""
        return (
            f"{self.__class__.__name__}("
            f"nelectrons={self.nelectrons}, "
            f"multi={self.geometry.multi}, "
            f"charge={self.geometry.charge}, "
            f"integral_method={self.integral_calculator.method.value}, "
            f"converged={self.converged}"
            ")"
        )
