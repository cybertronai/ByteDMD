#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = ["matplotlib", "numpy"]
# ///
"""
Visualize EDF register allocation access patterns for:
  - Naive i-j-k matrix multiply
  - Vanilla recursive (8-way D&C) matrix multiply
  - Strassen (7-way D&C) matrix multiply

Each algorithm gets a plot showing:
  - Hot accesses (W slots) in red
  - Cold misses (E slots) in blue
  - Post-read (ANY slots) in orange
  - Slot depth envelope in green
"""

import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import bytedmd_edf as be

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Algorithm definitions
# ---------------------------------------------------------------------------

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

def matmul_recursive(A, B):
    return _matmul_rec(A, B)

def matmul_tiled(A, B, t=2):
    """Tiled i-j-k matmul with block size t."""
    n = len(A)
    C = [[None] * n for _ in range(n)]
    for bi in range(0, n, t):
        for bj in range(0, n, t):
            for bk in range(0, n, t):
                for i in range(bi, min(bi + t, n)):
                    for j in range(bj, min(bj + t, n)):
                        for k in range(bk, min(bk + t, n)):
                            if C[i][j] is None:
                                C[i][j] = A[i][k] * B[k][j]
                            else:
                                C[i][j] = C[i][j] + A[i][k] * B[k][j]
    return C

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


# ---------------------------------------------------------------------------
# Trace extraction with type classification
# ---------------------------------------------------------------------------

def extract_edf_trace(func, args):
    """Run EDF tracer and extract per-read (depth, slot_type) pairs."""
    ctx = be._Context()
    wrapped_args = tuple(be._wrap(ctx, a, deferred=True) for a in args)
    res = func(*wrapped_args)

    total_cost, read_costs = be.edf_price_trace(ctx.events)

    # Replay events to build per-read trace
    depths = []
    types = []  # 'E', 'W', or 'ANY'
    t = 0
    for ev in ctx.events:
        if ev[0] == 'READ_BATCH':
            t += 1
            seen = set()
            for k in ev[1]:
                if k not in seen:
                    seen.add(k)
                    info = read_costs.get((k, t), None)
                    if info:
                        depth, typ, cost = info
                        depths.append(depth)
                        types.append(typ)
                    else:
                        depths.append(0)
                        types.append('?')

    return depths, types, total_cost


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_edf_access(ax, depths, types, title, total_cost):
    """Plot EDF access pattern on a given axes."""
    xs = np.arange(len(depths))

    # Separate by type
    w_mask = [t == 'W' for t in types]
    e_mask = [t == 'E' for t in types]
    any_mask = [t == 'ANY' for t in types]

    w_x = xs[w_mask] if any(w_mask) else []
    w_d = np.array(depths)[w_mask] if any(w_mask) else []
    e_x = xs[e_mask] if any(e_mask) else []
    e_d = np.array(depths)[e_mask] if any(e_mask) else []
    a_x = xs[any_mask] if any(any_mask) else []
    a_d = np.array(depths)[any_mask] if any(any_mask) else []

    if len(w_x): ax.scatter(w_x, w_d, c='red', s=1.0, alpha=0.5, label='W (Working)', rasterized=True, linewidths=0)
    if len(e_x): ax.scatter(e_x, e_d, c='blue', s=1.5, alpha=0.6, label='E (External/Cold)', rasterized=True, linewidths=0)
    if len(a_x): ax.scatter(a_x, a_d, c='orange', s=1.0, alpha=0.4, label='ANY (Post-read)', rasterized=True, linewidths=0)

    # Running max envelope (working slots only)
    w_depths = [d if t == 'W' else 0 for d, t in zip(depths, types)]
    running_max_w = np.maximum.accumulate(w_depths)
    ax.plot(xs, running_max_w, color='green', linewidth=1.5, alpha=0.7, label='Max W-slot envelope')

    ax.set_xlabel('Read operation index')
    ax.set_ylabel('Slot depth')
    ax.set_title(f'{title}\nTotal EDF cost = {total_cost:,}')
    ax.legend(fontsize=7, loc='upper left')


def visualize_all(n=8):
    algos = [
        ('Naive MatMul (i-j-k)', matmul_naive),
        ('Tiled MatMul (2x2 blocks)', matmul_tiled),
        ('Vanilla Recursive (8-way)', matmul_recursive),
    ]

    fig, axes = plt.subplots(len(algos), 1, figsize=(14, 5 * len(algos)))

    for ax, (name, func) in zip(axes, algos):
        A = [[1] * n for _ in range(n)]
        B = [[1] * n for _ in range(n)]
        depths, types, total_cost = extract_edf_trace(func, (A, B))
        plot_edf_access(ax, depths, types, f'{name} (N={n})', total_cost)

    plt.suptitle(f'EDF Register Allocation Access Patterns (N={n})', fontsize=14, y=1.01)
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), 'edf_access_patterns.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close()

    # Also print scaling table
    print(f"\n{'Algorithm':<30} {'N':>4} {'EDF cost':>10}")
    print("-" * 50)
    for name, func in algos:
        for sz in [2, 4, 8]:
            A = [[1] * sz for _ in range(sz)]
            B = [[1] * sz for _ in range(sz)]
            cost = be.bytedmd(func, (A, B))
            print(f"{name:<30} {sz:4d} {cost:10,}")


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    visualize_all(n)
