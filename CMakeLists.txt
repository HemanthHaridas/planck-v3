cmake_minimum_required(VERSION 3.18)

project(planck_v3 LANGUAGES CXX)

# Build the pybind11 extension module.
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(planck_integrals
  src/integrals/planck_integrals.cpp
)

target_compile_features(planck_integrals PRIVATE cxx_std_17)

