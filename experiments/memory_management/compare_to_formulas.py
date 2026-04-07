#!/usr/bin/env python3
"""
Compare measured continuous ByteDMD costs against Gemini's analytical
closed-form predictions.

Gemini's formulas (Continuous ByteDMD model, bytes=1):
    Naive (any strategy)            = 1.0   * N^4
    RMM unmanaged                   = 25.1  * N^3.5
    RMM tombstone                   = 12.3  * N^3 * log2(N)
    RMM aggressive                  =  7.3  * N^3 * log2(N)
    Strassen unmanaged              = 74.7  * N^3.403
    Strassen tombstone              = 200.0 * N^3
    Strassen aggressive             = 140.8 * N^3

This script reports:
  - Predicted vs measured continuous cost at each N
  - Ratio (measured / predicted) — should be ~1 if the formulas match
  - The leading-coefficient implied by my measurements (so we can see
    whether the SHAPE matches even if the constant is off)
"""
import json
import math
import os

import numpy as np


def load_results(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(__file__), 'results.json')
    with open(path) as f:
        return json.load(f)


# Gemini's predicted closed-form formulas.
def predict(algo, strategy, N):
    if algo == 'naive':
        return 1.0 * N ** 4
    if algo == 'rmm':
        if strategy == 'unmanaged':
            return 25.1 * N ** 3.5
        if strategy == 'tombstone':
            return 12.3 * N ** 3 * math.log2(N) if N > 1 else 0
        if strategy == 'aggressive':
            return 7.3 * N ** 3 * math.log2(N) if N > 1 else 0
    if algo == 'strassen':
        if strategy == 'unmanaged':
            return 74.7 * N ** 3.403
        if strategy == 'tombstone':
            return 200.0 * N ** 3
        if strategy == 'aggressive':
            return 140.8 * N ** 3
    return None


def implied_constant(algo, strategy, N, measured):
    """If we believe the Gemini formula's polynomial shape, what would
    the leading constant be implied by this measurement?"""
    if algo == 'naive':
        return measured / (N ** 4)
    if algo == 'rmm':
        if strategy == 'unmanaged':
            return measured / (N ** 3.5)
        return measured / (N ** 3 * math.log2(N))
    if algo == 'strassen':
        if strategy == 'unmanaged':
            return measured / (N ** 3.403)
        return measured / (N ** 3)
    return None


def main():
    rows = load_results()
    by = {(r['algorithm'], r['N'], r['strategy']): r for r in rows}
    sizes = sorted({r['N'] for r in rows})

    print("=" * 120)
    print("CONTINUOUS COST: measured vs Gemini's analytical formula")
    print("=" * 120)
    print(f"  {'algo':<10}{'strategy':<12}{'N':>5}  "
          f"{'measured':>14}  {'predicted':>14}  "
          f"{'meas/pred':>10}  {'implied_const':>14}")
    print("-" * 120)

    for algo in ('naive', 'rmm', 'strassen'):
        for strategy in ('unmanaged', 'tombstone', 'aggressive'):
            for N in sizes:
                r = by.get((algo, N, strategy))
                if r is None:
                    continue
                measured = r['cost_continuous']
                predicted = predict(algo, strategy, N)
                ratio = measured / predicted if predicted else float('nan')
                ic = implied_constant(algo, strategy, N, measured)
                print(f"  {algo:<10}{strategy:<12}{N:>5}  "
                      f"{measured:>14.1f}  {predicted:>14.1f}  "
                      f"{ratio:>10.3f}  {ic:>14.3f}")
            print()

    print()
    print("=" * 120)
    print("RATIO TRENDS: does meas/pred converge to a constant as N grows?")
    print("(if it does, the polynomial shape is right but Gemini's constant is wrong)")
    print("=" * 120)
    for algo in ('naive', 'rmm', 'strassen'):
        for strategy in ('unmanaged', 'tombstone', 'aggressive'):
            ratios = []
            for N in sizes:
                r = by.get((algo, N, strategy))
                if r is None:
                    continue
                p = predict(algo, strategy, N)
                if p:
                    ratios.append((N, r['cost_continuous'] / p))
            if ratios:
                cells = "  ".join(f"N={n}:{r:.3f}" for n, r in ratios)
                print(f"  {algo:>9s}/{strategy:<11s}: {cells}")

    print()
    print("=" * 120)
    print("EXTRACTED EXPONENT (from successive doublings of measured continuous cost)")
    print("(this is log2(cost(2N)/cost(N)) — the empirical polynomial degree)")
    print("=" * 120)
    print(f"  {'algo/strategy':<26}" + "  ".join(f"{n1}->{n2}" for n1, n2 in
        zip(sizes[:-1], sizes[1:])))
    print("-" * 90)
    for algo in ('naive', 'rmm', 'strassen'):
        for strategy in ('unmanaged', 'tombstone', 'aggressive'):
            exps = []
            for n1, n2 in zip(sizes[:-1], sizes[1:]):
                r1 = by.get((algo, n1, strategy))
                r2 = by.get((algo, n2, strategy))
                if r1 and r2 and r1['cost_continuous'] > 0:
                    exps.append(math.log2(r2['cost_continuous'] / r1['cost_continuous']))
                else:
                    exps.append(None)
            cells = "  ".join((f"{e:>6.3f}" if e is not None else "   -- ") for e in exps)
            print(f"  {algo:>10s}/{strategy:<14s} {cells}")

    print()
    print("=" * 120)
    print("AGGRESSIVE / TOMBSTONE RATIO (continuous cost)")
    print("Gemini predicts:  RMM = 7.3/12.3 = 0.593,  Strassen = 140.8/200 = 0.704")
    print("=" * 120)
    for algo in ('naive', 'rmm', 'strassen'):
        for N in sizes:
            t = by.get((algo, N, 'tombstone'))
            a = by.get((algo, N, 'aggressive'))
            if t and a and t['cost_continuous'] > 0:
                ratio = a['cost_continuous'] / t['cost_continuous']
                print(f"  {algo:>10s} N={N:>3} : aggressive/tombstone = {ratio:.3f}")
        print()


if __name__ == '__main__':
    main()
