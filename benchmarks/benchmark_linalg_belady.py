#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = ["numpy"]
# ///
"""Compare ByteDMD costs: regular LRU vs Belady (OPT) stack.

Prints a side-by-side table showing how Belady's clairvoyant cache ordering
compares to standard LRU for each algorithm at N = 2, 4, 8.

Run:
    ./benchmarks/benchmark_linalg_belady.py
    uv run benchmarks/benchmark_linalg_belady.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from bytedmd import bytedmd as bytedmd_regular
from bytedmd_belady import bytedmd as bytedmd_belady


# --- Algorithms (imported from benchmark_linalg pattern) ---

def matvec(A, x):
    n = len(x)
    y = [None] * n
    for i in range(n):
        s = A[i][0] * x[0]
        for j in range(1, n):
            s = s + A[i][j] * x[j]
        y[i] = s
    return y


def matmul_naive(A, B):
    n = len(A)
    C = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = A[i][0] * B[0][j]
            for k in range(1, n):
                s = s + A[i][k] * B[k][j]
            C[i][j] = s
    return C


def _split(M):
    n = len(M); h = n // 2
    return ([[M[i][j] for j in range(h)] for i in range(h)],
            [[M[i][j] for j in range(h, n)] for i in range(h)],
            [[M[i][j] for j in range(h)] for i in range(h, n)],
            [[M[i][j] for j in range(h, n)] for i in range(h, n)])

def _join(C11, C12, C21, C22):
    h = len(C11); n = 2 * h
    return [[C11[i][j] if j < h else C12[i][j-h] for j in range(n)] for i in range(h)] + \
           [[C21[i][j] if j < h else C22[i][j-h] for j in range(n)] for i in range(h)]

def _add(A, B):
    n = len(A)
    return [[A[i][j] + B[i][j] for j in range(n)] for i in range(n)]

def _sub(A, B):
    n = len(A)
    return [[A[i][j] - B[i][j] for j in range(n)] for i in range(n)]

def _matmul_rec(A, B):
    n = len(A)
    if n == 1:
        return [[A[0][0] * B[0][0]]]
    A11, A12, A21, A22 = _split(A)
    B11, B12, B21, B22 = _split(B)
    C11 = _add(_matmul_rec(A11, B11), _matmul_rec(A12, B21))
    C12 = _add(_matmul_rec(A11, B12), _matmul_rec(A12, B22))
    C21 = _add(_matmul_rec(A21, B11), _matmul_rec(A22, B21))
    C22 = _add(_matmul_rec(A21, B12), _matmul_rec(A22, B22))
    return _join(C11, C12, C21, C22)

def matmul_vanilla_recursive(A, B):
    return _matmul_rec(A, B)

def _matmul_strassen(A, B, leaf=1):
    n = len(A)
    if n <= leaf:
        return matmul_naive(A, B)
    A11, A12, A21, A22 = _split(A)
    B11, B12, B21, B22 = _split(B)
    M1 = _matmul_strassen(_add(A11, A22), _add(B11, B22), leaf)
    M2 = _matmul_strassen(_add(A21, A22), B11, leaf)
    M3 = _matmul_strassen(A11, _sub(B12, B22), leaf)
    M4 = _matmul_strassen(A22, _sub(B21, B11), leaf)
    M5 = _matmul_strassen(_add(A11, A12), B22, leaf)
    M6 = _matmul_strassen(_sub(A21, A11), _add(B11, B12), leaf)
    M7 = _matmul_strassen(_sub(A12, A22), _add(B21, B22), leaf)
    C11 = _add(_sub(_add(M1, M4), M5), M7)
    C12 = _add(M3, M5)
    C21 = _add(M2, M4)
    C22 = _add(_add(_sub(M1, M2), M3), M6)
    return _join(C11, C12, C21, C22)

def matmul_strassen(A, B):
    return _matmul_strassen(A, B, leaf=1)


SIZES = [2, 4, 8]

METHODS = [
    ("matvec",           'matvec',  matvec),
    ("naive matmul",     'matmul',  matmul_naive),
    ("vanilla recursive",'matmul',  matmul_vanilla_recursive),
    ("Strassen",         'matmul',  matmul_strassen),
]


def _cost(tracer, kind, fn, n):
    if kind == 'matvec':
        return tracer(fn, (np.ones((n, n)), np.ones(n)))
    return tracer(fn, (np.ones((n, n)), np.ones((n, n))))


if __name__ == '__main__':
    print("# ByteDMD: Regular LRU vs Belady (OPT)\n")

    for name, kind, fn in METHODS:
        print(f"## {name}")
        print(f"{'N':>4} {'LRU':>10} {'Belady':>10} {'ratio':>8}")
        print(f"{'--':>4} {'--':>10} {'--':>10} {'--':>8}")
        for n in SIZES:
            lru = _cost(bytedmd_regular, kind, fn, n)
            bel = _cost(bytedmd_belady, kind, fn, n)
            ratio = bel / lru if lru > 0 else 0
            print(f"{n:4d} {lru:10,} {bel:10,} {ratio:8.3f}")
        print()
