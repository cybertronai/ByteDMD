#!/usr/bin/env python3
"""ByteDMD costs for linear-algebra algorithms *with garbage collection*.

Mirrors benchmark_linalg.py but every algorithm releases its inner-loop
temporaries via plain Python `del` (or by rebinding the name so the old
value's refcount hits zero). The ByteDMD tracer's `_Tracked.__del__` hook
then removes the dead value from the LRU stack, so subsequent reads pay
less. Comparing these numbers against benchmark_linalg.py shows how much
of the cost of the original algorithms is due to dead temporaries sitting
on the stack.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from bytedmd import bytedmd


# --- Matrix-vector and vector-matrix ---

def matvec4(A, x):
    n = len(x)
    y = [None] * n
    for i in range(n):
        s = A[i][0] * x[0]
        for j in range(1, n):
            prod = A[i][j] * x[j]
            s = s + prod          # rebinding `s` drops the old accumulator
            del prod              # drop the just-consumed product
        y[i] = s
    return y


def vecmat4(A, x):
    n = len(x)
    y = [None] * n
    for j in range(n):
        s = x[0] * A[0][j]
        for i in range(1, n):
            prod = x[i] * A[i][j]
            s = s + prod
            del prod
        y[j] = s
    return y


# --- Matrix multiply variants ---

def matmul4(A, B):
    n = len(A)
    C = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = A[i][0] * B[0][j]
            for k in range(1, n):
                prod = A[i][k] * B[k][j]
                s = s + prod
                del prod
            C[i][j] = s
    return C


def matmul4_snake_j(A, B):
    n = len(A)
    C = [[None] * n for _ in range(n)]
    for i in range(n):
        js = range(n) if i % 2 == 0 else range(n - 1, -1, -1)
        for j in js:
            s = A[i][0] * B[0][j]
            for k in range(1, n):
                prod = A[i][k] * B[k][j]
                s = s + prod
                del prod
            C[i][j] = s
    return C


def matmul4_ikj(A, B):
    n = len(A)
    C = [[None] * n for _ in range(n)]
    for i in range(n):
        Ai = A[i]
        Ci = C[i]
        for k in range(n):
            aik = Ai[k]
            Bk = B[k]
            for j in range(n):
                prod = aik * Bk[j]
                if Ci[j] is None:
                    Ci[j] = prod
                else:
                    Ci[j] = Ci[j] + prod   # old Ci[j] dies here
                del prod
    return C


def matmul4_tiled(A, B):
    n = len(A)
    t = 2
    C = [[None] * n for _ in range(n)]
    for bi in range(0, n, t):
        for bj in range(0, n, t):
            for bk in range(0, n, t):
                for i in range(bi, bi + t):
                    for j in range(bj, bj + t):
                        for k in range(bk, bk + t):
                            prod = A[i][k] * B[k][j]
                            if C[i][j] is None:
                                C[i][j] = prod
                            else:
                                C[i][j] = C[i][j] + prod
                            del prod
    return C


# --- Measurements ---

def measure(name, operation, func, args):
    return name, operation, bytedmd(func, args)


if __name__ == '__main__':
    A = np.ones((4, 4))
    B = np.ones((4, 4))
    x = np.ones(4)

    results = [
        measure("matvec (i-j) gc",      "y = A @ x",    matvec4,         (A, x)),
        measure("vecmat (j-i) gc",      "y = x^T @ A",  vecmat4,         (A, x)),
        measure("matmul (i-j-k) gc",    "C = A @ B",    matmul4,         (A, B)),
        measure("matmul (i-k-j) gc",    "C = A @ B",    matmul4_ikj,     (A, B)),
        measure("matmul (snake-j) gc",  "C = A @ B",    matmul4_snake_j, (A, B)),
        measure("matmul (2x2 tiled) gc","C = A @ B",    matmul4_tiled,   (A, B)),
    ]

    # Baselines from benchmark_linalg.py (no GC).
    baselines = {
        "matvec (i-j) gc":       194,
        "vecmat (j-i) gc":       191,
        "matmul (i-j-k) gc":     948,
        "matmul (i-k-j) gc":    1016,
        "matmul (snake-j) gc":   906,
        "matmul (2x2 tiled) gc": 947,
    }

    print(f"{'Algorithm':<24} {'Operation':<14} {'GC cost':>8} {'no-GC':>8} {'savings':>9}")
    print("-" * 68)
    for name, op, cost in results:
        base = baselines[name]
        saved = f"{100 * (base - cost) / base:5.1f}%"
        print(f"{name:<24} {op:<14} {cost:>8} {base:>8} {saved:>9}")
