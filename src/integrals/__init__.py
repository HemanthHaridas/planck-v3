"""
Python-facing integrals API.

The compiled pybind11 extension is exposed as the `planck_integrals` module.
This package re-exports its symbols for convenience.
"""

from planck_integrals import (  # type: ignore
    Contracted,
    HuzinagaGaussianProduct,
    Primitive,
    huzinaga_kinetic_contracted,
    huzinaga_kinetic_primitive_3d,
    huzinaga_overlap_contracted,
    huzinaga_overlap_primitive_1d,
    huzinaga_overlap_primitive_3d,
    kinetic_contracted,
    kinetic_primitive_3d,
    overlap_contracted,
    overlap_primitive_1d,
    overlap_primitive_3d,
)

__all__ = [
    "Primitive",
    "Contracted",
    "HuzinagaGaussianProduct",
    "overlap_primitive_1d",
    "overlap_primitive_3d",
    "overlap_contracted",
    "kinetic_primitive_3d",
    "kinetic_contracted",
    "huzinaga_overlap_primitive_1d",
    "huzinaga_overlap_primitive_3d",
    "huzinaga_overlap_contracted",
    "huzinaga_kinetic_primitive_3d",
    "huzinaga_kinetic_contracted",
]

