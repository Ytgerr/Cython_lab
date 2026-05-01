"""
Python wrapper around the compiled C shared library (matrix_ops.so / .dll).
Uses ctypes — no Cython required.

Build the shared library first:
  Linux/macOS:
    gcc -O2 -march=native -shared -fPIC -o src/c_impl/matrix_ops.so \
        src/c_impl/matrix_ops.c -lm

  Windows (MSVC):
    cl /O2 /LD src/c_impl/matrix_ops.c /Fe:src/c_impl/matrix_ops.dll

  Windows (MinGW/GCC):
    gcc -O2 -shared -o src/c_impl/matrix_ops.dll src/c_impl/matrix_ops.c -lm
"""

import ctypes
import os
import sys
import numpy as np

# --------------------------------------------------------------------------
# Locate the shared library
# --------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))

if sys.platform == "win32":
    _LIB_NAME = "matrix_ops.dll"
elif sys.platform == "darwin":
    _LIB_NAME = "matrix_ops.dylib"
else:
    _LIB_NAME = "matrix_ops.so"

_LIB_PATH = os.path.join(_HERE, _LIB_NAME)

try:
    _lib = ctypes.CDLL(_LIB_PATH)
    _LIB_AVAILABLE = True
except OSError:
    _LIB_AVAILABLE = False
    _lib = None


def _require_lib():
    if not _LIB_AVAILABLE:
        raise RuntimeError(
            f"C shared library not found at {_LIB_PATH}.\n"
            "Build it with:\n"
            "  gcc -O2 -march=native -shared -fPIC "
            "-o src/c_impl/matrix_ops.so src/c_impl/matrix_ops.c -lm"
        )


# --------------------------------------------------------------------------
# Declare C function signatures
# --------------------------------------------------------------------------
if _LIB_AVAILABLE:
    _dbl_p = ctypes.POINTER(ctypes.c_double)

    _lib.matmul.restype = None
    _lib.matmul.argtypes = [_dbl_p, _dbl_p, _dbl_p,
                             ctypes.c_int, ctypes.c_int, ctypes.c_int]

    _lib.dot_product.restype = ctypes.c_double
    _lib.dot_product.argtypes = [_dbl_p, _dbl_p, ctypes.c_int]

    _lib.matrix_add.restype = None
    _lib.matrix_add.argtypes = [_dbl_p, _dbl_p, _dbl_p,
                                  ctypes.c_int, ctypes.c_int]

    _lib.transpose.restype = None
    _lib.transpose.argtypes = [_dbl_p, _dbl_p, ctypes.c_int, ctypes.c_int]

    _lib.frobenius_norm.restype = ctypes.c_double
    _lib.frobenius_norm.argtypes = [_dbl_p, ctypes.c_int, ctypes.c_int]


def _ptr(arr: np.ndarray):
    """Return a ctypes double* pointer to a C-contiguous NumPy array."""
    assert arr.dtype == np.float64 and arr.flags["C_CONTIGUOUS"]
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


# --------------------------------------------------------------------------
# Public API — mirrors the Python / Cython implementations
# --------------------------------------------------------------------------

def matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    _require_lib()
    n, k = A.shape
    m = B.shape[1]
    A = np.ascontiguousarray(A, dtype=np.float64)
    B = np.ascontiguousarray(B, dtype=np.float64)
    C = np.empty((n, m), dtype=np.float64)
    _lib.matmul(_ptr(A), _ptr(B), _ptr(C), n, k, m)
    return C


def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    _require_lib()
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    return _lib.dot_product(_ptr(a), _ptr(b), len(a))


def matrix_add(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    _require_lib()
    n, m = A.shape
    A = np.ascontiguousarray(A, dtype=np.float64)
    B = np.ascontiguousarray(B, dtype=np.float64)
    C = np.empty((n, m), dtype=np.float64)
    _lib.matrix_add(_ptr(A), _ptr(B), _ptr(C), n, m)
    return C


def transpose(A: np.ndarray) -> np.ndarray:
    _require_lib()
    n, m = A.shape
    A = np.ascontiguousarray(A, dtype=np.float64)
    T = np.empty((m, n), dtype=np.float64)
    _lib.transpose(_ptr(A), _ptr(T), n, m)
    return T


def frobenius_norm(A: np.ndarray) -> float:
    _require_lib()
    n, m = A.shape
    A = np.ascontiguousarray(A, dtype=np.float64)
    return _lib.frobenius_norm(_ptr(A), n, m)
