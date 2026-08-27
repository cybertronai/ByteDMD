#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = ["matplotlib", "numpy"]
# ///
"""
Visualize Belady (OPT) stack access patterns for:
  - Naive i-j-k matrix multiply
  - Tiled (2x2 blocks) matrix multiply
  - Vanilla recursive (8-way D&C) matrix multiply

Each plot shows hot hits (red), cold misses (blue), and working set envelope (green).
"""

import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import bytedmd_belady as bb

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# --- Algorithms ---

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

def matmul_tiled(A, B, t=2):
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
    n = len(A); return [[A[i][j] + B[i][j] for j in range(n)] for i in range(n)]

def _matmul_rec(A, B):
    n = len(A)
    if n == 1: return [[A[0][0] * B[0][0]]]
    A11, A12, A21, A22 = _split(A); B11, B12, B21, B22 = _split(B)
    C11 = _add(_matmul_rec(A11, B11), _matmul_rec(A12, B21))
    C12 = _add(_matmul_rec(A11, B12), _matmul_rec(A12, B22))
    C21 = _add(_matmul_rec(A21, B11), _matmul_rec(A22, B21))
    C22 = _add(_matmul_rec(A21, B12), _matmul_rec(A22, B22))
    return _join(C11, C12, C21, C22)

def matmul_recursive(A, B):
    return _matmul_rec(A, B)


# --- Trace extraction ---

def extract_belady_trace(func, args):
    """Run Belady tracer and classify each read as hot or cold."""
    ctx = bb._Context()
    wrapped_args = tuple(bb._wrap(ctx, a, deferred=True) for a in args)
    res = func(*wrapped_args)
    bb._pass2(ctx, res)

    trace = ctx.trace
    # Reconstruct hot/cold: replay events and track which keys are in active_keys
    # Simpler: cold misses use the monotonic input_miss_count tape,
    # so cold depths form a sequence 1,2,3,... while hot depths depend on cache position.
    # We can detect cold by checking if a depth appeared on the cold tape.
    # Actually, let's just re-run the pass2 logic with classification.

    events = ctx.events
    import bisect
    accesses = {}
    for i, ev in enumerate(events):
        if ev[0] == 'STORE':
            k = ev[1]
            if k not in accesses: accesses[k] = []
            accesses[k].append(i)
        elif ev[0] == 'READ_BATCH':
            for k in ev[1]:
                if k not in accesses: accesses[k] = []
                accesses[k].append(i)

    names = {}
    def collect_keys(val):
        if isinstance(val, bb._Tracked): names[val._key] = True
        elif isinstance(val, (list, tuple)):
            for v in val: collect_keys(v)
        elif type(val).__name__ == 'ndarray':
            for v in val.flat: collect_keys(v)
    collect_keys(res)
    for k in names:
        if k not in accesses: accesses[k] = []
        accesses[k].append(len(events))

    last_use = {k: acc[-1] for k, acc in accesses.items() if acc}

    def get_next_use(k, step):
        acc = accesses.get(k, [])
        idx = bisect.bisect_right(acc, step)
        return acc[idx] if idx < len(acc) else float('inf')

    active_keys = []
    input_miss_count = 0
    depths = []
    is_cold = []
    wss = []

    for i, ev in enumerate(events):
        if ev[0] == 'STORE':
            k = ev[1]
            if last_use.get(k, -1) > i:
                if k not in active_keys:
                    active_keys.append(k)
                active_keys.sort(key=lambda x: (get_next_use(x, i), x))

        elif ev[0] == 'READ_BATCH':
            valid = ev[1]
            unique = list(dict.fromkeys(valid))

            for k in unique:
                if k in active_keys:
                    d = active_keys.index(k) + 1
                    cold = False
                else:
                    input_miss_count += 1
                    d = input_miss_count
                    cold = True
                    active_keys.append(k)

                # Emit per-occurrence
                count = valid.count(k)
                for _ in range(count):
                    depths.append(d)
                    is_cold.append(cold)
                    wss.append(len(active_keys))

            active_keys = [x for x in active_keys if last_use.get(x, -1) > i]
            active_keys.sort(key=lambda x: (get_next_use(x, i), x))

    total_cost = sum(math.isqrt(d - 1) + 1 for d in depths)
    return depths, is_cold, wss, total_cost


# --- Plotting ---

def plot_belady_access(ax, depths, is_cold, wss, title, total_cost):
    xs = np.arange(len(depths))
    d = np.array(depths)
    cold_mask = np.array(is_cold)
    hot_mask = ~cold_mask

    if hot_mask.any():
        ax.scatter(xs[hot_mask], d[hot_mask], c='red', s=1.0, alpha=0.5,
                   label='Hot hit', rasterized=True, linewidths=0)
    if cold_mask.any():
        ax.scatter(xs[cold_mask], d[cold_mask], c='blue', s=1.5, alpha=0.6,
                   label='Cold miss', rasterized=True, linewidths=0)

    # Working set size envelope
    ax.plot(xs, wss, color='green', linewidth=1.5, alpha=0.7, label='Working set size')

    ax.set_xlabel('Read operation index')
    ax.set_ylabel('Belady depth / WSS')
    ax.set_title(f'{title}\nTotal Belady cost = {total_cost:,}')
    ax.legend(fontsize=8, loc='upper left')


def visualize_all(n=16):
    algos = [
        ('Naive MatMul (i-j-k)', matmul_naive),
        ('Tiled MatMul (2x2 blocks)', matmul_tiled),
        ('Vanilla Recursive (8-way)', matmul_recursive),
    ]

    fig, axes = plt.subplots(len(algos), 1, figsize=(16, 18))

    for ax, (name, func) in zip(axes, algos):
        print(f'Tracing {name} N={n}...')
        A = [[1] * n for _ in range(n)]
        B = [[1] * n for _ in range(n)]
        depths, is_cold, wss, total_cost = extract_belady_trace(func, (A, B))
        print(f'  cost={total_cost:,}  reads={len(depths)}')
        plot_belady_access(ax, depths, is_cold, wss, f'{name} (N={n})', total_cost)

    plt.suptitle(f'Belady (OPT) Stack Access Patterns (N={n})', fontsize=15, y=1.01)
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), 'belady_access_patterns.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved: {out}')
    plt.close()


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    visualize_all(n)
