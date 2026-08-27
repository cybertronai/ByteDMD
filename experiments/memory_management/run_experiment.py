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
    naive_matmul, rmm, strassen,
    rmm_inplace_lex, rmm_inplace_gray,
    make_ones,
    flops_naive, flops_rmm, flops_strassen,
)


# (name, function, flop_fn, inplace_flag)
ALGORITHMS = [
    ('naive',            naive_matmul,     flops_naive,    False),
    ('rmm',              rmm,              flops_rmm,      False),
    ('rmm_inplace_lex',  rmm_inplace_lex,  flops_rmm,      True),
    ('rmm_inplace_gray', rmm_inplace_gray, flops_rmm,      True),
    ('strassen',         strassen,         flops_strassen, False),
]

STRATEGIES = ['unmanaged', 'tombstone', 'aggressive']


def run(sizes, skip_unmanaged_above=None):
    rows = []
    for N in sizes:
        A = make_ones(N)
        B = make_ones(N)
        for name, fn, flop_fn, inplace in ALGORITHMS:
            flops = flop_fn(N)
            for strat in STRATEGIES:
                if (strat == 'unmanaged' and skip_unmanaged_above
                        and N > skip_unmanaged_above):
                    continue
                t0 = time.perf_counter()
                m = measure(fn, A, B, strat, inplace=inplace)
                wall = time.perf_counter() - t0
                rows.append({
                    'algorithm': name,
                    'N': N,
                    'strategy': strat,
                    'cost_discrete': m['cost_discrete'],
                    'cost_continuous': m['cost_continuous'],
                    'n_reads': m['n_reads'],
                    'peak_stack': m['peak_stack'],
                    'flops': flops,
                    'wall_seconds': round(wall, 4),
                })
                print(f"  N={N:4d}  {name:18s} {strat:11s}  "
                      f"discrete={m['cost_discrete']:>14}  "
                      f"continuous={m['cost_continuous']:>16.1f}  "
                      f"reads={m['n_reads']:>10}  peak={m['peak_stack']:>8}  "
                      f"({wall:.1f}s)")
    return rows


def main():
    max_log2 = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    sizes = [2 ** k for k in range(1, max_log2 + 1)]
    print(f"Running memory management experiment for N in {sizes}")
    # Skip unmanaged above N=32: it's O(N^3) live footprint and the
    # tracer becomes prohibitively slow at N=64.
    print()
    rows = run(sizes, skip_unmanaged_above=32)
    out_path = os.path.join(os.path.dirname(__file__), 'results.json')
    with open(out_path, 'w') as f:
        json.dump(rows, f, indent=2)
    print()
    print(f"Saved {len(rows)} rows to {out_path}")


if __name__ == '__main__':
    main()
