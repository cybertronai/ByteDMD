#!/usr/bin/env python3
"""ByteDMD cost of naive matrix multiplication across sizes.

Runs the standard i-j-k triple-loop matmul C = A @ B on square matrices
of several sizes, measuring cost under both the regular proxy-based
ByteDMD tracer and the strict AST-validated tracer.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from bytedmd import bytedmd as bytedmd_regular
from bytedmd_fx import bytedmd as bytedmd_fx
from bytedmd_bytecode import bytedmd as bytedmd_strict


def matmul(A, B):
    """Naive i-j-k matrix multiply C = A @ B for square n x n."""
    n = len(A)
    C = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = A[i][0] * B[0][j]
            for k in range(1, n):
                s = s + A[i][k] * B[k][j]
            C[i][j] = s
    return C


def matmul_snake(A, B):
    """Snake-order matmul: reverse the j-loop on alternating rows."""
    n = len(A)
    C = [[None] * n for _ in range(n)]
    for i in range(n):
        js = range(n) if i % 2 == 0 else range(n - 1, -1, -1)
        for j in js:
            s = A[i][0] * B[0][j]
            for k in range(1, n):
                s = s + A[i][k] * B[k][j]
            C[i][j] = s
    return C


def print_table(title, func, sizes):
    print(title)
    print(f"{'N':>4} {'regular':>12} {'fx':>12} {'strict':>12} {'strict/reg':>12}")
    print("-" * 56)
    for n in sizes:
        A = np.ones((n, n))
        B = np.ones((n, n))
        reg = bytedmd_regular(func, (A, B))
        fx = bytedmd_fx(func, (A, B))
        strict = bytedmd_strict(func, (A, B))
        ratio = strict / reg if reg else float('nan')
        print(f"{n:>4} {reg:>12} {fx:>12} {strict:>12} {ratio:>12.3f}")


if __name__ == '__main__':
    sizes = list(range(1, 9))

    print_table("i-j-k order", matmul, sizes)
    print()
    print_table("snake-j order", matmul_snake, sizes)
