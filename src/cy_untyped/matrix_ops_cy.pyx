# cython: language_level=3
"""
Cython matrix operations — NO static types.
This is a direct port of the Python version.
Cython compiles it to C, but every variable is still a Python object (PyObject*).
Boxing/unboxing overhead is identical to CPython; the only gain is the
removal of the interpreter dispatch loop.
"""

def zeros(int n, int m):
    return [[0.0] * m for _ in range(n)]


def matmul(A, B):
    """Naive O(n^3) matrix multiplication — untyped Cython."""
    n = len(A)
    m = len(B[0])
    k = len(B)
    C = zeros(n, m)
    for i in range(n):
        for j in range(m):
            s = 0.0
            for p in range(k):
                s += A[i][p] * B[p][j]
            C[i][j] = s
    return C


def matrix_add(A, B):
    n = len(A)
    m = len(A[0])
    C = zeros(n, m)
    for i in range(n):
        for j in range(m):
            C[i][j] = A[i][j] + B[i][j]
    return C


def monte_carlo_pi(int n_samples):
    """
    Monte Carlo pi estimation — untyped Cython.
    Variables x, y, inside are still PyObject* — boxing remains.
    Only the interpreter dispatch loop is eliminated.
    """
    import random
    inside = 0
    for _ in range(n_samples):
        x = random.random()
        y = random.random()
        if x * x + y * y < 1.0:
            inside += 1
    return 4.0 * inside / n_samples
