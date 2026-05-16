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


def simulate_dice_game(n_rounds: int) -> float:
    """
    Monte Carlo dice game simulation.

    Rules (everyone understands this):
      - Roll 2 six-sided dice each round.
      - You WIN if the sum is 7 or higher (the most common outcome).
      - You LOSE if the sum is 6 or lower.

    Returns the estimated win probability after n_rounds rounds.
    True probability = 21/36 ≈ 0.5833 (21 out of 36 combinations give sum ≥ 7).

    This is a pure scalar loop — every variable is a PyObject*.
    Demonstrates boxing overhead perfectly.
    """
    wins = 0
    for _ in range(n_rounds):
        die1 = random.randint(1, 6)
        die2 = random.randint(1, 6)
        if die1 + die2 >= 7:
            wins += 1
    return wins / n_rounds
