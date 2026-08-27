#!/usr/bin/env python3
"""
Analyze memory management experiment results.

Computes the empirical scaling exponent between consecutive size doublings
(log2 of cost ratio when N doubles), the implied asymptotic exponent, and
fits a two-term model `cost ~ a * 8^k + b * 7^k` (or similar) to compare
against the closed-form predictions in the asymptotic analysis.
"""
import json
import math
import os
import sys

import numpy as np


def load_results(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(__file__), 'results.json')
    with open(path) as f:
        return json.load(f)


def group_by_algo_strategy(rows):
    """{ (algorithm, strategy) -> [(N, cost), ...] sorted by N }"""
    out = {}
    for r in rows:
        key = (r['algorithm'], r['strategy'])
        out.setdefault(key, []).append((r['N'], r['cost']))
    for k in out:
        out[k].sort()
    return out


def empirical_exponent(series):
    """Return list of log2(cost(2N)/cost(N)) for each consecutive doubling."""
    out = []
    for i in range(len(series) - 1):
        n1, c1 = series[i]
        n2, c2 = series[i + 1]
        if c1 > 0 and n2 == 2 * n1:
            out.append(math.log2(c2 / c1))
    return out


def fit_two_term(series, base1, base2):
    """Least-squares fit cost(N) = a * base1^k + b * base2^k where N=2^k."""
    if len(series) < 2:
        return None, None, None
    Ns = np.array([s[0] for s in series], dtype=float)
    cs = np.array([s[1] for s in series], dtype=float)
    ks = np.log2(Ns)
    X = np.column_stack([base1 ** ks, base2 ** ks])
    coef, _, _, _ = np.linalg.lstsq(X, cs, rcond=None)
    fit = X @ coef
    rel_err = np.max(np.abs(fit - cs) / cs)
    return coef[0], coef[1], rel_err


def fit_n3_logn(series):
    """Fit cost(N) = a * N^3 * log2(N) + b * N^3, the analytical RMM form."""
    if len(series) < 3:
        return None, None, None
    Ns = np.array([s[0] for s in series], dtype=float)
    cs = np.array([s[1] for s in series], dtype=float)
    ks = np.log2(Ns)
    n3 = Ns ** 3
    X = np.column_stack([n3 * ks, n3])
    coef, _, _, _ = np.linalg.lstsq(X, cs, rcond=None)
    fit = X @ coef
    rel_err = np.max(np.abs(fit - cs) / cs)
    return coef[0], coef[1], rel_err


def main():
    rows = load_results(sys.argv[1] if len(sys.argv) > 1 else None)
    grouped = group_by_algo_strategy(rows)

    print("=" * 110)
    print("EMPIRICAL SCALING EXPONENTS")
    print("Each cell is log2(cost(2N) / cost(N)) — i.e., how the cost grows")
    print("when N doubles. Asymptotically this should approach the polynomial")
    print("exponent of the leading term.")
    print("=" * 110)
    sizes = sorted({r['N'] for r in rows})
    doublings = [(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]
    header = f"  {'algorithm/strategy':<26}" + "".join(
        f"{n1:>4}->{n2:<4}" for n1, n2 in doublings)
    print(header)
    print("-" * len(header))
    for algo in ('naive', 'rmm', 'strassen'):
        for strat in ('unmanaged', 'tombstone', 'aggressive'):
            series = grouped.get((algo, strat), [])
            exps = empirical_exponent(series)
            cells = "".join(f"{e:>9.3f}" for e in exps)
            print(f"  {algo:>10s}/{strat:<14s}{cells}")
    print()

    print("=" * 110)
    print("TWO-TERM FITS to cost(N) = a*base1^k + b*base2^k where N = 2^k")
    print("(testing the closed-form predictions from the asymptotic analysis)")
    print("=" * 110)

    # Fits: which two bases per (algo, strategy) according to the asymptotic
    # claim from the report. We try the predicted form and report the
    # coefficients + relative error.
    fits = [
        # (algorithm, strategy, base1, base2, rationale)
        ('naive',    'unmanaged',  16, 8,  'N^4 (~16^k) leading'),
        ('naive',    'tombstone',  16, 8,  'N^4 (constant across strategies)'),
        ('naive',    'aggressive', 16, 8,  'N^4'),
        ('rmm',      'unmanaged',  11.3137, 8, 'N^3.5 (~2^3.5 = 11.31)'),
        ('rmm',      'tombstone',  8, 7,  'N^3 logN (try 8^k vs 7^k as proxy)'),
        ('rmm',      'aggressive', 8, 7,  'N^3 logN'),
        ('strassen', 'unmanaged',  10.556, 7,  'N^3.4 (~2^3.4 = 10.56) vs 7^k'),
        ('strassen', 'tombstone',  8, 7,  'N^3 leading + 7^k subleading'),
        ('strassen', 'aggressive', 8, 7,  'N^3 leading + 7^k subleading'),
    ]
    for algo, strat, base1, base2, why in fits:
        series = grouped.get((algo, strat), [])
        a, b, rel_err = fit_two_term(series, base1, base2)
        if a is None:
            print(f"  {algo:>10s}/{strat:<11s}: not enough data")
            continue
        print(f"  {algo:>10s}/{strat:<11s}: cost ~ "
              f"{a:>14.2f} * {base1:>7.4g}^k + {b:>14.2f} * {base2:>7.4g}^k"
              f"  (max rel err {rel_err*100:.1f}%)  -- {why}")
    print()

    print("=" * 110)
    print("RMM cost(N) = a * N^3 * log2(N) + b * N^3 fit")
    print("(this is the analytical form predicted for memory-managed RMM)")
    print("=" * 110)
    for strat in ('tombstone', 'aggressive'):
        series = grouped.get(('rmm', strat), [])
        a, b, rel_err = fit_n3_logn(series)
        if a is not None:
            print(f"  rmm/{strat:11s}: cost ~ {a:>10.4f} * N^3 log2(N) "
                  f"+ {b:>10.4f} * N^3   (max rel err {rel_err*100:.1f}%)")
    print()

    print("=" * 110)
    print("STRATEGY RATIOS at each N")
    print("(how much does memory management buy us?)")
    print("=" * 110)
    by_algo_N = {}
    for r in rows:
        by_algo_N.setdefault((r['algorithm'], r['N']), {})[r['strategy']] = r['cost']
    for algo in ('naive', 'rmm', 'strassen'):
        print(f"  {algo}:")
        for n in sizes:
            d = by_algo_N.get((algo, n), {})
            u = d.get('unmanaged')
            t = d.get('tombstone')
            a = d.get('aggressive')
            parts = []
            if u is not None:
                parts.append(f"unmgd={u:>10}")
            if t is not None:
                parts.append(f"tomb={t:>10}")
            if a is not None:
                parts.append(f"aggr={a:>10}")
            ratio_t = f" t/u={t/u:.2f}" if (u and t) else ""
            ratio_a = f" a/t={a/t:.2f}" if (t and a) else ""
            print(f"    N={n:>3}  " + "  ".join(parts) + ratio_t + ratio_a)

    print()
    print("=" * 110)
    print("PEAK STACK FOOTPRINT")
    print("(should grow as O(N^2) for managed strategies, O(N^3) for unmanaged)")
    print("=" * 110)
    peak_data = {}
    for r in rows:
        peak_data.setdefault((r['algorithm'], r['strategy']), {})[r['N']] = r['peak_stack']
    print(f"  {'algorithm/strategy':<26}" + "".join(f"{n:>10}" for n in sizes))
    for algo in ('naive', 'rmm', 'strassen'):
        for strat in ('unmanaged', 'tombstone', 'aggressive'):
            d = peak_data.get((algo, strat), {})
            cells = "".join(f"{d.get(n, 0):>10}" for n in sizes)
            print(f"  {algo:>10s}/{strat:<14s}{cells}")


if __name__ == '__main__':
    main()
