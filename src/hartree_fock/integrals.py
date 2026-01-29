"""Integral calculation interface for Hartree-Fock methods."""

from enum import Enum
from typing import Literal, Union

import numpy as np

try:
    from src.integrals import (
        Contracted,
        Primitive,
        huzinaga_kinetic_contracted,
        huzinaga_overlap_contracted,
        kinetic_contracted,
        overlap_contracted,
    )
    INTEGRALS_AVAILABLE = True
except ImportError:
    INTEGRALS_AVAILABLE = False


class IntegralMethod(str, Enum):
    """Enumeration of available integral methods."""
    
    HERMITE = "hermite"
    HUZINAGA = "huzinaga"
    
    def __str__(self) -> str:
        return self.value


class IntegralCalculator:
    """
    Calculator for molecular integrals using different methods.
    
    This class provides a unified interface for computing overlap and kinetic
    energy integrals using either the Hermite or Huzinaga method.
    
    Attributes:
        method (IntegralMethod): The integral method to use.
        available (bool): Whether the C++ integral module is available.
    """
    
    def __init__(self, method: Union[IntegralMethod, str] = IntegralMethod.HUZINAGA) -> None:
        """
        Initialize the integral calculator.
        
        Args:
            method: Integral method to use. Can be "hermite" or "huzinaga".
                   Defaults to "huzinaga".
        
        Raises:
            ImportError: If the C++ integral module is not available.
            ValueError: If an invalid method is specified.
        """
        if not INTEGRALS_AVAILABLE:
            raise ImportError(
                "C++ integral module (planck_integrals) is not available. "
                "Please ensure the module is built and installed."
            )
        
        if isinstance(method, str):
            method = method.lower()
            if method == "hermite":
                self.method = IntegralMethod.HERMITE
            elif method == "huzinaga":
                self.method = IntegralMethod.HUZINAGA
            else:
                raise ValueError(
                    f"Invalid integral method: {method}. "
                    f"Must be 'hermite' or 'huzinaga'."
                )
        else:
            self.method = method
        
        self.available = INTEGRALS_AVAILABLE
    
    def overlap(
        self,
        contractedA: Contracted,
        contractedB: Contracted,
    ) -> float:
        """
        Compute overlap integral between two contracted Gaussian functions.
        
        Args:
            contractedA: First contracted Gaussian function.
            contractedB: Second contracted Gaussian function.
        
        Returns:
            float: Overlap integral value.
        """
        if self.method == IntegralMethod.HERMITE:
            return overlap_contracted(contractedA, contractedB)
        else:  # HUZINAGA
            return huzinaga_overlap_contracted(contractedA, contractedB)
    
    def kinetic(
        self,
        contractedA: Contracted,
        contractedB: Contracted,
    ) -> float:
        """
        Compute kinetic energy integral between two contracted Gaussian functions.
        
        Args:
            contractedA: First contracted Gaussian function.
            contractedB: Second contracted Gaussian function.
        
        Returns:
            float: Kinetic energy integral value.
        """
        if self.method == IntegralMethod.HERMITE:
            return kinetic_contracted(contractedA, contractedB)
        else:  # HUZINAGA
            return huzinaga_kinetic_contracted(contractedA, contractedB)
    
    def __repr__(self) -> str:
        """Return string representation of the integral calculator."""
        return f"IntegralCalculator(method={self.method.value})"
