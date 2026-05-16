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


def simulate_dice_game(int n_rounds):
    """
    Monte Carlo dice game — untyped Cython.

    Roll 2 dice per round, win if sum >= 7.
    Variables die1, die2, wins are still PyObject* — boxing remains.
    Only the interpreter dispatch loop is eliminated.
    """
    import random
    wins = 0
    for _ in range(n_rounds):
        die1 = random.randint(1, 6)
        die2 = random.randint(1, 6)
        if die1 + die2 >= 7:
            wins += 1
    return wins / n_rounds
