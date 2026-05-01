"""
Pure Python matrix operations — baseline implementation.
All arithmetic is done with Python lists of lists (no NumPy).
"""

from __future__ import annotations
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


def dot_product(a: List[float], b: List[float]) -> float:
    """Dot product of two 1-D vectors."""
    s = 0.0
    for x, y in zip(a, b):
        s += x * y
    return s


def matrix_add(A: Matrix, B: Matrix) -> Matrix:
    n, m = len(A), len(A[0])
    C = zeros(n, m)
    for i in range(n):
        for j in range(m):
            C[i][j] = A[i][j] + B[i][j]
    return C


def transpose(A: Matrix) -> Matrix:
    n, m = len(A), len(A[0])
    T = zeros(m, n)
    for i in range(n):
        for j in range(m):
            T[j][i] = A[i][j]
    return T


def frobenius_norm(A: Matrix) -> float:
    s = 0.0
    for row in A:
        for v in row:
            s += v * v
    return s ** 0.5
