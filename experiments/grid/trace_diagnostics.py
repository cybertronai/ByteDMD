#!/Users/yaroslavvb/.local/bin/uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = ["matplotlib"]
# ///
"""Three ByteDMD-live + two-stack diagnostic plots per algorithm in
run_grid.ALGOS:

  <slug>_liveset.png           — live working-set size over time
  <slug>_reuse_distance.png    — LRU depth at each L2Load
  <slug>_wss.png               — Denning-style working-set size over
                                 time (sliding fixed-τ window)

The first two walk the L2 trace with the same Fenwick-tree semantics
as bytedmd_ir._lru_cost (compaction + arg-stack first-touch promotion).
The WSS plot slides a fixed-τ window across the reference stream and
counts distinct variables touched in each [t-τ+1, t] window — the
cache-size lens on memory behaviour. Per-algorithm τ is picked via
pick_wss_window(n_events) and printed in the caption."""
from __future__ import annotations

import os
import re
import sys
from typing import List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bytedmd_ir import (
    L2Load,
    L2Store,
    opt_reuse_distances,
    static_opt_floor_curve,
    static_opt_lb,
    trace,
)
import run_grid as rg


def slugify(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()


def walk_live_and_reuse(events, input_vars) -> Tuple[List[int], List[int],
                                                     List[int], List[int]]:
    """Single pass: returns (ls_times, ls_sizes, rd_times, rd_distances).

    ls_ series sample the live-set size after every event (Store or Load).
    rd_ series sample the reuse distance at each L2Load.
    Live-compaction semantics match bytedmd_ir._lru_cost(compact=True):
      * never-loaded stores are skipped
      * last-load of a var drops it from the geom stack
      * first-load of an input promotes it; cost = arg-stack position
    """
    input_arg_idx = {v: i + 1 for i, v in enumerate(input_vars)}
    pending = set(input_arg_idx)

    last_load = {}
    for i, ev in enumerate(events):
        if isinstance(ev, L2Load):
            last_load[ev.var] = i

    T = len(events) + len(input_arg_idx) + 2
    bit = [0] * (T + 1)
    def bit_add(i, d):
        while i <= T:
            bit[i] += d
            i += i & -i
    def bit_prefix(i):
        s = 0
        while i > 0:
            s += bit[i]; i -= i & -i
        return s

    ts_of = {}
    next_ts = 0
    live_count = 0

    ls_times, ls_sizes = [], []
    rd_times, rd_distances = [], []

    for i, ev in enumerate(events):
        if isinstance(ev, L2Store):
            if ev.var in last_load:
                next_ts += 1
                ts_of[ev.var] = next_ts
                bit_add(next_ts, 1)
                live_count += 1
        elif isinstance(ev, L2Load):
            if ev.var in pending:
                pending.discard(ev.var)
                d = input_arg_idx[ev.var]
                rd_times.append(i); rd_distances.append(d)
                if last_load.get(ev.var) != i:
                    next_ts += 1
                    ts_of[ev.var] = next_ts
                    bit_add(next_ts, 1)
                    live_count += 1
            else:
                t = ts_of[ev.var]
                total_live = bit_prefix(T)
                depth = total_live - bit_prefix(t - 1)
                rd_times.append(i); rd_distances.append(depth)
                bit_add(t, -1)
                if last_load[ev.var] == i:
                    del ts_of[ev.var]
                    live_count -= 1
                else:
                    next_ts += 1
                    ts_of[ev.var] = next_ts
                    bit_add(next_ts, 1)
        ls_times.append(i); ls_sizes.append(live_count)
    return ls_times, ls_sizes, rd_times, rd_distances


def plot_liveset(times, sizes, title, out_path):
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.plot(times, sizes, color="tab:blue", linewidth=0.8,
            drawstyle="steps-post", rasterized=True)
    ax.fill_between(times, 0, sizes, color="tab:blue", alpha=0.18,
                    linewidth=0, step="post", rasterized=True)
    ax.set_xlabel("Access index (time)")
    ax.set_ylabel("Live variables on geom stack")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if times:
        ax.set_xlim(0, times[-1] + 1)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def pick_wss_window(reuse_distances, n_events: int) -> int:
    """Pick a sliding-window size τ that represents a realistic cache.

    The window is set to the 90th-percentile reuse distance: a cache of
    exactly τ slots would capture roughly 90% of this algorithm's reads
    (the remaining 10% are the long-reuse tail). Algorithms with tight
    locality (tiled / blocked / recursive) get a small τ that reveals
    their steady-state working set; algorithms that churn through cold
    data get a τ closer to the full reference stream.

    Clamped to [32, max(32, n_events // 2)] so every plot still has
    meaningful horizontal variation.
    """
    import math as _math
    if not reuse_distances:
        return max(32, round(_math.sqrt(max(1, n_events))))
    rd = sorted(reuse_distances)
    idx = min(len(rd) - 1, int(0.90 * len(rd)))
    p90 = rd[idx]
    tau = max(32, int(p90))
    if n_events > 0:
        tau = min(tau, max(32, n_events // 2))
    return tau


def working_set_over_time(events, window):
    """Denning's working-set: for each index t, report how many distinct
    variables were referenced (L2Load or L2Store) in the inclusive
    window [max(0, t - window + 1), t]. Each variable identity is one
    'page'. Returns (times, sizes) sampled at every reference event."""
    refs = [getattr(ev, "var", None) for ev in events]
    counts = {}
    current = 0
    times, sizes = [], []
    for i, v in enumerate(refs):
        if v is not None:
            if counts.get(v, 0) == 0:
                current += 1
            counts[v] = counts.get(v, 0) + 1
        if i >= window:
            old = refs[i - window]
            if old is not None:
                counts[old] -= 1
                if counts[old] == 0:
                    current -= 1
                    del counts[old]
        times.append(i); sizes.append(current)
    return times, sizes


def plot_wss(times, sizes, window, title, out_path):
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.plot(times, sizes, color="tab:orange", linewidth=0.8,
            drawstyle="steps-post", rasterized=True)
    ax.fill_between(times, 0, sizes, color="tab:orange", alpha=0.18,
                    linewidth=0, step="post", rasterized=True)
    ax.set_xlabel(f"Access index (time, τ = {window:,}-event window)")
    ax.set_ylabel("Distinct vars touched in last τ events")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if times:
        ax.set_xlim(0, times[-1] + 1)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_reuse_distance(times, distances, title, out_path):
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.scatter(times, distances, s=0.8, c="tab:purple", alpha=0.35,
               linewidths=0, rasterized=True)
    ax.set_xlabel("Access index (time)")
    ax.set_ylabel("Reuse distance (LRU depth at read)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if times:
        ax.set_xlim(0, times[-1] + 1)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_opt_reuse_distance(times, distances, title, out_path):
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.scatter(times, distances, s=0.8, c="tab:green", alpha=0.35,
               linewidths=0, rasterized=True)
    ax.set_xlabel("Access index (time)")
    ax.set_ylabel("OPT reuse distance (Bélády max-rank at read)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if times:
        ax.set_xlim(0, times[-1] + 1)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _miss_curve(distances, max_cap):
    """Return M(c) for c = 0..max_cap, where M(c) = # of distances > c.

    Bin distances into a counts array, prefix-sum it, then M(c) = N − prefix(c).
    O(max_cap + N).
    """
    if not distances:
        return [0] * (max_cap + 1)
    n = len(distances)
    bins = [0] * (max_cap + 2)
    for d in distances:
        di = d if d <= max_cap else max_cap + 1
        bins[di] += 1
    cum = 0
    misses = [0] * (max_cap + 1)
    for c in range(max_cap + 1):
        cum += bins[c]
        misses[c] = n - cum
    return misses


def plot_static_opt_floor(times, floors, total, n_events, title, out_path):
    """Step plot of the per-tick TU LP floor Σ_i ρ_{(i)} · √i over live
    vars (gemini/optimal-static-floor.md). The shaded area equals
    static_opt_lb. A dashed horizontal line marks the time-average
    floor — multiplying it by the trace length recovers the bound.
    """
    fig, ax = plt.subplots(figsize=(11, 3.4))
    if not times:
        plt.close(fig)
        return
    ax.plot(times, floors, drawstyle="steps-post", color="tab:orange",
            linewidth=1.0, rasterized=True, label="Σ ρ_(i) · √i")
    ax.fill_between(times, 0, floors, step="post", color="tab:orange",
                    alpha=0.15, linewidth=0, rasterized=True)
    avg = total / n_events if n_events else 0
    ax.axhline(avg, color="tab:red", linestyle="--", linewidth=0.8,
               alpha=0.7, label=f"average = {avg:,.2f}")
    ax.set_xlabel("Access index (time)")
    ax.set_ylabel("Per-tick floor: Σ ρ_(i) · √i")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, n_events if n_events else times[-1])
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_mrc_combined(lru_d, opt_d, title, out_path):
    """Miss-ratio curves for LRU and Bélády OPT on the same axes.

    For each cache capacity c (= number of slots), plot the number of
    loads that miss — i.e., loads with reuse distance > c. By Mattson
    inclusion the OPT curve sits at or below the LRU curve at every c;
    the area between them weighted by the marginal cost
    Δ_c = ceil(sqrt(c+1)) − ceil(sqrt(c)) is exactly the locality slack
    that bytedmd_opt extracts (energy decomposition in
    gemini/belady-min-lower-bound.md §1).
    """
    if not (lru_d and opt_d):
        return
    max_cap = max(max(lru_d), max(opt_d))
    lru_m = _miss_curve(lru_d, max_cap)
    opt_m = _miss_curve(opt_d, max_cap)
    caps = list(range(max_cap + 1))

    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(caps, lru_m, color="tab:purple", linewidth=1.1,
            rasterized=True, label="LRU misses")
    ax.plot(caps, opt_m, color="tab:green", linewidth=1.1,
            rasterized=True, label="Bélády OPT misses")
    ax.set_xlabel("Cache capacity c (slots)")
    ax.set_ylabel("Misses M(c) = # loads with reuse distance > c")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_cap)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_reuse_distance_combined(times, lru_d, opt_d, title, out_path):
    """Overlay LRU and Bélády OPT reuse distances on the same axes.

    LRU depth is upper-bounded by Mattson inclusion at OPT max-rank,
    so OPT (green) sits at or below LRU (purple) at every load — the
    visible green/purple gap is the locality slack that an offline
    oracle would extract over LRU on this trace.
    """
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.scatter(times, lru_d, s=0.8, c="tab:purple", alpha=0.35,
               linewidths=0, rasterized=True, label="LRU reuse distance")
    ax.scatter(times, opt_d, s=0.8, c="tab:green", alpha=0.55,
               linewidths=0, rasterized=True,
               label="Bélády OPT reuse distance")
    ax.set_xlabel("Access index (time)")
    ax.set_ylabel("Reuse distance (rank at read)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if times:
        ax.set_xlim(0, times[-1] + 1)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", markerscale=8, fontsize=8,
              framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    traces_dir = os.path.join(HERE, "traces")
    os.makedirs(traces_dir, exist_ok=True)

    print(f"{'algorithm':<42} {'events':>8} {'peak_live':>10} "
          f"{'max_rd':>8} {'med_rd':>8} {'wss_tau':>8} {'wss_max':>8}")
    print("-" * 100)
    summary = []
    for name, fn, args, _ in rg.ALGOS:
        events, input_vars = trace(fn, args)
        ls_t, ls_s, rd_t, rd_d = walk_live_and_reuse(events, input_vars)
        peak = max(ls_s) if ls_s else 0
        mx = max(rd_d) if rd_d else 0
        med = sorted(rd_d)[len(rd_d) // 2] if rd_d else 0

        FIXED_TAU = 100
        window = FIXED_TAU
        wss_t, wss_s = working_set_over_time(events, window)
        wss_max = max(wss_s) if wss_s else 0

        slug = slugify(name)
        plot_liveset(ls_t, ls_s,
                     f"{name} — live working-set size (peak = {peak:,})",
                     os.path.join(traces_dir, f"{slug}_liveset.png"))

        # Bélády OPT reuse distance per load for every algorithm.
        iidx = {v: i + 1 for i, v in enumerate(input_vars)}
        opt_t, opt_d = opt_reuse_distances(events, iidx)
        opt_mx = max(opt_d) if opt_d else 0

        # Combined LRU + OPT scatter on a single panel so the
        # locality-slack gap is visible at a glance. (rd_t and opt_t
        # both run over every L2Load in trace order, so they align.)
        plot_reuse_distance_combined(
            rd_t, rd_d, opt_d,
            f"{name} — reuse distance per load "
            f"(LRU max = {mx:,}; OPT max = {opt_mx:,})",
            os.path.join(traces_dir, f"{slug}_reuse_distance.png"))

        # Miss-ratio curves (LRU and OPT) across all capacities c.
        plot_mrc_combined(
            rd_d, opt_d,
            f"{name} — miss-ratio curve (LRU vs Bélády OPT)",
            os.path.join(traces_dir, f"{slug}_mrc.png"))

        # Per-tick TU LP floor: integrand of static_opt_lb.
        sof_t, sof_v = static_opt_floor_curve(events, iidx)
        sof_total = static_opt_lb(events, iidx)
        plot_static_opt_floor(
            sof_t, sof_v, sof_total, len(events),
            f"{name} — per-tick TU LP floor "
            f"(static_opt_lb = {sof_total:,.0f})",
            os.path.join(traces_dir, f"{slug}_static_opt_floor.png"))

        plot_wss(
            wss_t, wss_s, window,
            f"{name} — WSS over time (τ = {window:,}, max = {wss_max:,})",
            os.path.join(traces_dir, f"{slug}_wss.png"))
        summary.append((name, slug, peak, mx, med, window, wss_max))
        print(f"{name:<42} {len(events):>8,} {peak:>10,} "
              f"{mx:>8,} {med:>8,} {window:>8,} {wss_max:>8,}")

    summary_path = os.path.join(HERE, "trace_diagnostics_summary.tsv")
    with open(summary_path, "w") as f:
        f.write("name\tslug\tpeak_live\tmax_reuse\tmedian_reuse"
                "\twss_window\twss_max\n")
        for name, slug, peak, mx, med, window, wss_max in summary:
            f.write(f"{name}\t{slug}\t{peak}\t{mx}\t{med}"
                    f"\t{window}\t{wss_max}\n")
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
