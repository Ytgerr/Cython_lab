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
  * libc.stdlib rand() — C-level RNG, no Python objects at all

Memory layout:
  NumPy arrays passed in must be C-contiguous (order='C') so that
  memoryview[i, j] maps to *(ptr + i*cols + j) — a single pointer add.
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt
from libc.stdlib cimport rand, srand, RAND_MAX

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
# monte_carlo_pi — typed version
# ---------------------------------------------------------------------------
cpdef double monte_carlo_pi(int n_samples):
    """
    Monte Carlo pi estimation — fully typed Cython.

    Uses libc rand() — a C-level RNG that returns int in [0, RAND_MAX].
    All variables are C types on the stack:
      cdef int i, inside
      cdef double x, y
    No PyObject* created inside the loop. No boxing. No GIL needed.

    Speedup vs Python: ~50-150x depending on n_samples.
    """
    cdef int i
    cdef int inside = 0
    cdef double x, y
    cdef double inv_rand = 1.0 / RAND_MAX

    srand(42)  # fixed seed for reproducibility

    with nogil:
        for i in range(n_samples):
            x = rand() * inv_rand
            y = rand() * inv_rand
            if x * x + y * y < 1.0:
                inside += 1

    return 4.0 * inside / n_samples
