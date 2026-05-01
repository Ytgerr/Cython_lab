# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
"""
Cython matrix operations — FULLY TYPED with typed memoryviews.

Key techniques demonstrated:
  * cdef / cpdef for C-level functions (no PyObject overhead)
  * Typed memoryviews (double[:, :]) — direct pointer arithmetic, no GIL needed
  * nogil blocks — releases the GIL so the OS can schedule other threads
  * cdivision=True — skips Python ZeroDivisionError check
  * boundscheck=False / wraparound=False — removes bounds-check guards

Memory layout:
  NumPy arrays passed in must be C-contiguous (order='C') so that
  memoryview[i, j] maps to *(ptr + i*cols + j) — a single pointer add.
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt

# Tell Cython the dtype we will use everywhere
ctypedef double DTYPE_t
DTYPE = np.float64


# ---------------------------------------------------------------------------
# matmul — the hot loop
# ---------------------------------------------------------------------------
cpdef cnp.ndarray[DTYPE_t, ndim=2] matmul(
        double[:, :] A,
        double[:, :] B):
    """
    O(n^3) matrix multiplication using typed memoryviews.
    The inner triple loop runs as pure C — no Python objects touched.
    """
    cdef int n = A.shape[0]
    cdef int k = A.shape[1]
    cdef int m = B.shape[1]
    cdef int i, j, p
    cdef double s

    cdef cnp.ndarray[DTYPE_t, ndim=2] C_arr = np.zeros((n, m), dtype=DTYPE)
    cdef double[:, :] C = C_arr

    with nogil:
        for i in range(n):
            for j in range(m):
                s = 0.0
                for p in range(k):
                    s += A[i, p] * B[p, j]
                C[i, j] = s

    return C_arr


# ---------------------------------------------------------------------------
# dot_product
# ---------------------------------------------------------------------------
cpdef double dot_product(double[:] a, double[:] b):
    cdef int n = a.shape[0]
    cdef int i
    cdef double s = 0.0
    with nogil:
        for i in range(n):
            s += a[i] * b[i]
    return s


# ---------------------------------------------------------------------------
# matrix_add
# ---------------------------------------------------------------------------
cpdef cnp.ndarray[DTYPE_t, ndim=2] matrix_add(
        double[:, :] A,
        double[:, :] B):
    cdef int n = A.shape[0]
    cdef int m = A.shape[1]
    cdef int i, j

    cdef cnp.ndarray[DTYPE_t, ndim=2] C_arr = np.empty((n, m), dtype=DTYPE)
    cdef double[:, :] C = C_arr

    with nogil:
        for i in range(n):
            for j in range(m):
                C[i, j] = A[i, j] + B[i, j]

    return C_arr


# ---------------------------------------------------------------------------
# transpose
# ---------------------------------------------------------------------------
cpdef cnp.ndarray[DTYPE_t, ndim=2] transpose(double[:, :] A):
    cdef int n = A.shape[0]
    cdef int m = A.shape[1]
    cdef int i, j

    cdef cnp.ndarray[DTYPE_t, ndim=2] T_arr = np.empty((m, n), dtype=DTYPE)
    cdef double[:, :] T = T_arr

    with nogil:
        for i in range(n):
            for j in range(m):
                T[j, i] = A[i, j]

    return T_arr


# ---------------------------------------------------------------------------
# frobenius_norm
# ---------------------------------------------------------------------------
cpdef double frobenius_norm(double[:, :] A):
    cdef int n = A.shape[0]
    cdef int m = A.shape[1]
    cdef int i, j
    cdef double s = 0.0

    with nogil:
        for i in range(n):
            for j in range(m):
                s += A[i, j] * A[i, j]

    return sqrt(s)
