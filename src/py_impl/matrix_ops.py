"""
Pure Python matrix operations — baseline implementation.
All arithmetic is done with Python lists of lists (no NumPy).
"""

from __future__ import annotations
import random
from typing import List

Matrix = List[List[float]]


def zeros(n: int, m: int) -> Matrix:
    return [[0.0] * m for _ in range(n)]


def matmul(A: Matrix, B: Matrix) -> Matrix:
    """Naive O(n^3) matrix multiplication."""
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


def matrix_add(A: Matrix, B: Matrix) -> Matrix:
    n, m = len(A), len(A[0])
    C = zeros(n, m)
    for i in range(n):
        for j in range(m):
            C[i][j] = A[i][j] + B[i][j]
    return C


def monte_carlo_pi(n_samples: int) -> float:
    """
    Estimate pi using Monte Carlo method.
    Throw n_samples random points into the unit square [0,1)^2.
    Count how many fall inside the unit circle (x^2 + y^2 < 1).
    pi ~ 4 * inside / n_samples

    This is a pure scalar loop — every variable is a PyObject*.
    Demonstrates boxing overhead perfectly.
    """
    inside = 0
    for _ in range(n_samples):
        x = random.random()
        y = random.random()
        if x * x + y * y < 1.0:
            inside += 1
    return 4.0 * inside / n_samples
