#!/Users/yaroslavvb/.local/bin/uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = ["matplotlib", "numpy"]
# ///
"""Compute and plot every locality measure from gemini/locality-measures.md
for naive vs tiled matmul, on the AB^T workload at N=64, T=16.

Measures, in paper order:

  - Frequency locality: hotness n/m.
  - Access locality:
      * Reuse Interval (RI) histogram.
      * Reuse Distance (RD) histogram (LRU stack depth at reuse).
  - Timescale locality:
      * Footprint fp(x) — avg distinct addresses in windows of length x.
      * Working set s(x) — equivalent under sliding windows, plotted too.
  - Cache locality:
      * Miss Ratio Curve mr(c) — fraction of accesses with RD > c.
      * Fill / Eviction time: avg reuse interval of accesses with RD == c
        (Denning's "expected eviction time" proxy).

Usage:
    uv run --script visualize_locality.py
Writes a family of `locality_*.svg` files next to this script.
"""

from __future__ import annotations

import os
from collections import deque
from typing import Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from visualize_tiling import trace_naive_matmul, trace_tiled_matmul


# ---------------------------------------------------------------------------
# Trace preparation
# ---------------------------------------------------------------------------

def interleave(A: np.ndarray, B: np.ndarray, offset_B: int) -> np.ndarray:
    out = np.empty(2 * len(A), dtype=np.int64)
    out[0::2] = A
    out[1::2] = B + offset_B
    return out


# ---------------------------------------------------------------------------
# Reuse interval / reuse distance in O(n log n) via Fenwick tree
# ---------------------------------------------------------------------------

class _Fenwick:
    __slots__ = ('n', 'bit')

    def __init__(self, n: int) -> None:
        self.n = n
        self.bit = [0] * (n + 2)

    def add(self, i: int, d: int) -> None:
        n = self.n
        while i <= n:
            self.bit[i] += d
            i += i & -i

    def prefix(self, i: int) -> int:
        s = 0
        while i > 0:
            s += self.bit[i]
            i -= i & -i
        return s


def reuse_sequences(addrs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (RI, RD) arrays of the same length as addrs.

    Cold accesses (address seen for the first time) get -1 in both
    arrays, to be treated as infinity by downstream histogramming.
    """
    n = len(addrs)
    bit = _Fenwick(n + 1)
    ts = {}
    last_t = {}
    ri = np.empty(n, dtype=np.int64)
    rd = np.empty(n, dtype=np.int64)
    next_ts = 0
    for t in range(n):
        a = int(addrs[t])
        if a in ts:
            old_ts = ts[a]
            total_live = bit.prefix(n + 1)
            depth = total_live - bit.prefix(old_ts)
            rd[t] = depth
            ri[t] = t - last_t[a]
            bit.add(old_ts, -1)
        else:
            rd[t] = -1
            ri[t] = -1
        next_ts += 1
        ts[a] = next_ts
        bit.add(next_ts, 1)
        last_t[a] = t
    return ri, rd


# ---------------------------------------------------------------------------
# Footprint fp(x) / working set s(x) — sliding window, exact
# ---------------------------------------------------------------------------

def footprint_curve(addrs: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """fp(x) = average distinct addresses across all length-x sliding windows.

    Evaluated exactly at each x in `xs` using an O(n) multiset-deque
    sliding window. Total cost: O(len(xs) * n).
    """
    n = len(addrs)
    out = np.empty(len(xs), dtype=np.float64)
    for i, x in enumerate(xs):
        x = int(x)
        if x <= 0:
            out[i] = 0
            continue
        if x >= n:
            out[i] = len(set(int(a) for a in addrs))
            continue
        counts = {}
        window = deque()
        acc = 0
        nw = 0
        for t in range(n):
            a = int(addrs[t])
            window.append(a)
            counts[a] = counts.get(a, 0) + 1
            while len(window) > x:
                old = window.popleft()
                counts[old] -= 1
                if counts[old] == 0:
                    del counts[old]
            if len(window) == x:
                acc += len(counts)
                nw += 1
        out[i] = acc / max(1, nw)
    return out


# ---------------------------------------------------------------------------
# Miss-ratio curve from RD histogram
# ---------------------------------------------------------------------------

def mrc_from_rd(rd: np.ndarray, cache_sizes: np.ndarray) -> np.ndarray:
    """mr(c) = P(rd > c) + P(cold); computed via a sorted RD array."""
    finite = rd[rd >= 0]
    cold = int(np.sum(rd < 0))
    finite_sorted = np.sort(finite)
    n = len(rd)
    mr = np.empty(len(cache_sizes), dtype=np.float64)
    for i, c in enumerate(cache_sizes):
        above = len(finite_sorted) - int(np.searchsorted(finite_sorted, c,
                                                          side='right'))
        mr[i] = (above + cold) / n
    return mr


# ---------------------------------------------------------------------------
# Fill / eviction time proxy from RD histogram
# ---------------------------------------------------------------------------

def eviction_time(ri: np.ndarray, rd: np.ndarray,
                  cache_sizes: np.ndarray) -> np.ndarray:
    """For each cache size c: average RI among accesses with RD ~ c.

    At RD = c the access is right on the LRU cliff for capacity c; its
    RI is the time an entry of age c waited before being evicted under
    LRU. Since RD is discrete we bin accesses into ``[c_prev, c]``
    buckets on a log-spaced grid and return the mean RI per bucket.
    """
    finite = (rd >= 0) & (ri >= 0)
    rd_f = rd[finite]
    ri_f = ri[finite]
    out = np.empty(len(cache_sizes), dtype=np.float64)
    out[:] = np.nan
    edges = np.concatenate([[0], cache_sizes]).astype(np.int64)
    for i in range(len(cache_sizes)):
        lo, hi = edges[i], edges[i + 1]
        mask = (rd_f > lo) & (rd_f <= hi)
        if np.any(mask):
            out[i] = float(np.mean(ri_f[mask]))
    return out


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _save(fig, name: str) -> str:
    out = os.path.join(os.path.dirname(__file__), name)
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    return out


def plot_frequency(n_n: int, m_n: int, n_t: int, m_t: int) -> str:
    """Hotness comparison bar chart (paper notes these are identical)."""
    fig, ax = plt.subplots(figsize=(6, 3.6))
    labels = ['naive', 'tiled']
    hotness = [n_n / m_n, n_t / m_t]
    accesses = [n_n, n_t]
    distinct = [m_n, m_t]
    x = np.arange(len(labels))
    bars = ax.bar(x, hotness, color=['tab:red', 'tab:blue'],
                  alpha=0.7, edgecolor='black')
    for rect, n_acc, m_acc in zip(bars, accesses, distinct):
        ax.text(rect.get_x() + rect.get_width() / 2,
                rect.get_height() + 0.5,
                f'n={n_acc:,}\nm={m_acc:,}',
                ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Hotness  n/m  (avg reuses per address)')
    ax.set_title('Frequency locality — identical for naive vs tiled',
                 fontsize=11, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(0, max(hotness) * 1.25)
    return _save(fig, 'locality_frequency.svg')


def _hist_bin_edges_log() -> np.ndarray:
    """Log-spaced bin edges for RI / RD histograms. Bins of width ~10^{1/8}."""
    return np.unique(np.round(np.logspace(0, 7, 57)).astype(int))


def plot_reuse_histogram(values_n, values_t, *, kind: str, title: str,
                         xlabel: str, filename: str) -> str:
    bins = _hist_bin_edges_log()
    v_n = values_n[values_n > 0]
    v_t = values_t[values_t > 0]
    cold_n = int(np.sum(values_n < 0))
    cold_t = int(np.sum(values_t < 0))

    hist_n, _ = np.histogram(v_n, bins=bins)
    hist_t, _ = np.histogram(v_t, bins=bins)
    centers = np.sqrt(bins[:-1].astype(float) * bins[1:])

    denom = max(1, len(values_n))
    frac_n = hist_n / denom
    frac_t = hist_t / denom
    floor = 1.0 / denom / 10  # below-one bucket for log scale

    def _safe(y):
        y = y.astype(float).copy()
        y[y <= 0] = floor
        return y

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.step(centers, _safe(frac_n), where='mid', color='tab:red',
            linewidth=1.1,
            label=f'naive (cold {cold_n / denom:.2%})')
    ax.fill_between(centers, floor, _safe(frac_n), step='mid',
                    color='tab:red', alpha=0.18)
    ax.step(centers, _safe(frac_t), where='mid', color='tab:blue',
            linewidth=1.1,
            label=f'tiled (cold {cold_t / denom:.2%})')
    ax.fill_between(centers, floor, _safe(frac_t), step='mid',
                    color='tab:blue', alpha=0.22)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(floor, 1.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('fraction of accesses')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)
    return _save(fig, filename)


def plot_footprint(xs, fp_n, fp_t, N, T) -> str:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(xs, fp_n, color='tab:red', marker='o', ms=4, lw=1.2, label='naive')
    ax.plot(xs, fp_t, color='tab:blue', marker='s', ms=4, lw=1.2, label='tiled')
    ax.axhline(2 * T * T, color='gray', ls='--', lw=0.8,
               label=f'$2T^2 = {2 * T * T}$ (one tile)')
    ax.axhline(2 * N * N, color='black', ls=':', lw=0.8,
               label=f'$2N^2 = {2 * N * N}$ (full A+B)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'window size $x$ (reads)')
    ax.set_ylabel(r'$fp(x)$ — mean distinct addresses in window')
    ax.set_title('Footprint / working-set curve '
                 r'$fp(x) \approx s(x)$',
                 fontsize=11, fontweight='bold')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='lower right', fontsize=9)
    return _save(fig, 'locality_footprint.svg')


def plot_mrc(cs, mr_n, mr_t, N, T) -> str:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(cs, mr_n, color='tab:red', marker='o', ms=4, lw=1.2, label='naive')
    ax.plot(cs, mr_t, color='tab:blue', marker='s', ms=4, lw=1.2, label='tiled')
    ax.axvline(2 * T * T, color='gray', ls='--', lw=0.8,
               label=f'$c = 2T^2 = {2 * T * T}$')
    ax.axvline(2 * N * N, color='black', ls=':', lw=0.8,
               label=f'$c = 2N^2 = {2 * N * N}$')
    ax.set_xscale('log')
    ax.set_xlabel('cache capacity $c$ (entries)')
    ax.set_ylabel(r'miss ratio $mr(c)$')
    ax.set_title('Miss-ratio curve — LRU, fully associative',
                 fontsize=11, fontweight='bold')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(-0.02, 1.02)
    return _save(fig, 'locality_mrc.svg')


def plot_eviction_time(cs, ev_n, ev_t, N, T) -> str:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    mn = np.isfinite(ev_n)
    mt = np.isfinite(ev_t)
    ax.plot(cs[mn], ev_n[mn], color='tab:red', marker='o', ms=4,
            lw=1.2, label='naive')
    ax.plot(cs[mt], ev_t[mt], color='tab:blue', marker='s', ms=4,
            lw=1.2, label='tiled')
    ax.axvline(2 * T * T, color='gray', ls='--', lw=0.8,
               label=f'$c = 2T^2 = {2 * T * T}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('cache capacity $c$ (entries)')
    ax.set_ylabel('avg reuse interval in bin (reads)')
    ax.set_title('Eviction time — mean RI of accesses whose RD falls into '
                 'the bucket ending at $c$',
                 fontsize=11, fontweight='bold')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='lower right', fontsize=9)
    return _save(fig, 'locality_eviction.svg')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    N = 64
    T = 16
    offset_B = N * N

    print(f'Tracing naive & tiled matmul (N={N}, T={T})...')
    A_n, B_n = trace_naive_matmul(N)
    A_t, B_t = trace_tiled_matmul(N, T)
    tr_n = interleave(A_n, B_n, offset_B)
    tr_t = interleave(A_t, B_t, offset_B)
    n = len(tr_n)
    m_n = len(set(int(a) for a in tr_n))
    m_t = len(set(int(a) for a in tr_t))

    print('Computing reuse intervals / reuse distances...')
    ri_n, rd_n = reuse_sequences(tr_n)
    ri_t, rd_t = reuse_sequences(tr_t)

    print('Plotting: frequency, RI histogram, RD histogram...')
    p_freq = plot_frequency(n, m_n, n, m_t)
    p_ri = plot_reuse_histogram(
        ri_n, ri_t,
        kind='ri', title='Reuse Interval histogram',
        xlabel='reuse interval (reads since last access to the same address)',
        filename='locality_reuse_interval.svg')
    p_rd = plot_reuse_histogram(
        rd_n, rd_t,
        kind='rd', title='Reuse Distance histogram (LRU stack depth)',
        xlabel='reuse distance (distinct addresses since last access)',
        filename='locality_reuse_distance.svg')

    # Footprint and MRC use a log-spaced grid.
    print('Computing footprint fp(x)...')
    xs = np.unique(np.round(np.logspace(0, np.log10(n - 1), 28)).astype(int))
    xs = xs[xs >= 1]
    fp_n = footprint_curve(tr_n, xs)
    fp_t = footprint_curve(tr_t, xs)
    p_fp = plot_footprint(xs, fp_n, fp_t, N, T)

    print('Computing miss-ratio curve...')
    cs = np.unique(np.round(np.logspace(0, np.log10(n), 50)).astype(int))
    cs = cs[cs >= 1]
    mr_n = mrc_from_rd(rd_n, cs)
    mr_t = mrc_from_rd(rd_t, cs)
    p_mrc = plot_mrc(cs, mr_n, mr_t, N, T)

    print('Computing eviction-time curve...')
    cs_ev = np.unique(np.round(np.logspace(0, np.log10(n), 30)).astype(int))
    cs_ev = cs_ev[cs_ev >= 1]
    ev_n = eviction_time(ri_n, rd_n, cs_ev)
    ev_t = eviction_time(ri_t, rd_t, cs_ev)
    p_ev = plot_eviction_time(cs_ev, ev_n, ev_t, N, T)

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print('\nSummary (N=%d, T=%d, trace length = %d interleaved reads):' % (
        N, T, n))
    print(f'  naive: n/m = {n / m_n:.2f}   cold misses = '
          f'{int(np.sum(rd_n < 0)):,}')
    print(f'  tiled: n/m = {n / m_t:.2f}   cold misses = '
          f'{int(np.sum(rd_t < 0)):,}')
    for label, mr in [('naive', mr_n), ('tiled', mr_t)]:
        c_knee = cs[np.argmin(np.abs(mr - 0.05))]
        print(f'  {label}: mr(2T²={2*T*T}) = '
              f'{mrc_from_rd(rd_n if label == "naive" else rd_t, [2 * T * T])[0]:.3f}, '
              f'mr(2N²={2*N*N}) = '
              f'{mrc_from_rd(rd_n if label == "naive" else rd_t, [2 * N * N])[0]:.3f}, '
              f'knee(mr≈5%) at c ≈ {int(c_knee):,}')
    for p in (p_freq, p_ri, p_rd, p_fp, p_mrc, p_ev):
        print(f'  wrote {p}')


if __name__ == '__main__':
    main()
