"""
Build all Cython extensions.

Usage:
    python setup.py build_ext --inplace

This compiles:
  src/cy_untyped/matrix_ops_cy.pyx   -> matrix_ops_cy*.so / *.pyd
  src/cy_typed/matrix_ops_typed.pyx  -> matrix_ops_typed*.so / *.pyd
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

numpy_include = np.get_include()

# --------------------------------------------------------------------------
# Untyped Cython extension
# --------------------------------------------------------------------------
ext_untyped = Extension(
    name="src.cy_untyped.matrix_ops_cy",
    sources=["src/cy_untyped/matrix_ops_cy.pyx"],
    include_dirs=[numpy_include],
    extra_compile_args=["-O2"],
)

# --------------------------------------------------------------------------
# Typed Cython extension (with NumPy typed memoryviews)
# --------------------------------------------------------------------------
ext_typed = Extension(
    name="src.cy_typed.matrix_ops_typed",
    sources=["src/cy_typed/matrix_ops_typed.pyx"],
    include_dirs=[numpy_include],
    extra_compile_args=["-O2", "-march=native"],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
)

setup(
    name="cython_lab",
    ext_modules=cythonize(
        [ext_untyped, ext_typed],
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
        },
        annotate=True,          # generates .html annotation files
    ),
)
