#!/usr/bin/env python3
"""
Memory management experiment runner.

Measures ByteDMD cost for naive matmul, recursive matmul (RMM), and
Strassen (leaf=1) under three memory management strategies, across a
range of matrix sizes. Saves raw results to results.json.

Usage:
    python3 run_experiment.py [max_log2_size]

Default max_log2_size = 5 (i.e., up to N = 32). Bumping it to 6 covers
N = 64 but takes minutes (Strassen leaf=1 grows as 7^log2(N)).
"""
import json
import os
import sys
import time

from tracer import measure
from algorithms import (
    naive_matmul, rmm, strassen, make_ones,
    flops_naive, flops_rmm, flops_strassen,
)


ALGORITHMS = [
    ('naive', naive_matmul, flops_naive),
    ('rmm', rmm, flops_rmm),
    ('strassen', strassen, flops_strassen),
]

STRATEGIES = ['unmanaged', 'tombstone', 'aggressive']


def run(sizes):
    rows = []
    for N in sizes:
        A = make_ones(N)
        B = make_ones(N)
        for name, fn, flop_fn in ALGORITHMS:
            flops = flop_fn(N)
            for strat in STRATEGIES:
                t0 = time.perf_counter()
                cost, n_reads, peak = measure(fn, A, B, strat)
                wall = time.perf_counter() - t0
                rows.append({
                    'algorithm': name,
                    'N': N,
                    'strategy': strat,
                    'cost': cost,
                    'n_reads': n_reads,
                    'peak_stack': peak,
                    'flops': flops,
                    'wall_seconds': round(wall, 4),
                })
                print(f"  N={N:4d}  {name:9s} {strat:11s}  "
                      f"cost={cost:>14}  reads={n_reads:>10}  "
                      f"peak={peak:>8}  flops={flops:>10}  "
                      f"({wall:.2f}s)")
    return rows


def main():
    max_log2 = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    sizes = [2 ** k for k in range(1, max_log2 + 1)]
    print(f"Running memory management experiment for N in {sizes}")
    print()
    rows = run(sizes)
    out_path = os.path.join(os.path.dirname(__file__), 'results.json')
    with open(out_path, 'w') as f:
        json.dump(rows, f, indent=2)
    print()
    print(f"Saved {len(rows)} rows to {out_path}")


if __name__ == '__main__':
    main()
