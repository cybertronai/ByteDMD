#!/Users/yaroslavvb/.local/bin/uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = ["matplotlib", "numpy"]
# ///
"""Working-set size over time for naive vs tiled matmul, one graph.

Matches the style of ``experiments/grid/traces/*_liveset.png``: a line
chart of the live working-set size over time. Both algorithms are
overlaid so the locality story is visible at a glance.

Two complementary working-set notions are plotted (stacked panels):

  1. **Live addresses** (top panel) — an address is live on
     ``[first_access, last_access]``. This is what a liveness-aware
     allocator (``bytedmd_live``) would pin on-chip. On the *raw*
     compute trace, tiling the compute order alone does **not** reduce
     this: every input is revisited across every block, so naive and
     tiled live-set curves are similar in magnitude (tiled is actually
     slightly higher because its bi-bj-bk ordering spreads B's last
     touch later).

  2. **Sliding-window working set** W(t, τ) (bottom panel) — the number
     of distinct addresses in the trailing τ reads (Denning 1968). With
     τ sized to one tiled block (``2T^3`` interleaved reads), this
     exposes the real locality benefit: tiled's window sees only one
     ``T×T`` tile of A and B (≲ 2T²), while naive's window sweeps
     across most of B (≳ N²).

Usage:
    uv run --script visualize_working_set.py
Produces `working_set_over_time.svg`.
"""

import os
from collections import deque

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from visualize_tiling import trace_naive_matmul, trace_tiled_matmul


def interleave(A, B, offset_B):
    """Fold per-step (A_read, B_read) pairs into a single time sequence."""
    out = np.empty(2 * len(A), dtype=np.int64)
    out[0::2] = A
    out[1::2] = B + offset_B
    return out


def live_working_set(addrs):
    """|{a : first_access[a] <= t <= last_access[a]}| at each t.

    The minimum scratchpad capacity a liveness-aware allocator would
    require at time t (every address is "pinned" until its final read).
    """
    n = len(addrs)
    first, last = {}, {}
    for t, a in enumerate(addrs):
        a = int(a)
        if a not in first:
            first[a] = t
        last[a] = t
    births = np.zeros(n + 1, dtype=np.int32)
    deaths = np.zeros(n + 1, dtype=np.int32)
    for a, t0 in first.items():
        births[t0] += 1
        deaths[last[a] + 1] += 1
    return np.cumsum(births - deaths)[:n]


def sliding_working_set(addrs, tau):
    """Denning's W(t, τ): distinct addresses in the trailing τ reads.

    O(n) sliding window: a multiset of addresses currently inside the
    window; W is its set-cardinality at each step.
    """
    counts = {}
    window = deque()
    ws = np.empty(len(addrs), dtype=np.int32)
    for t, a in enumerate(addrs):
        a = int(a)
        window.append(a)
        counts[a] = counts.get(a, 0) + 1
        while len(window) > tau:
            old = window.popleft()
            counts[old] -= 1
            if counts[old] == 0:
                del counts[old]
        ws[t] = len(counts)
    return ws


def main():
    N = 64
    T = 16
    offset_B = N * N

    print(f"Tracing Naive (N={N}) and Tiled (N={N}, T={T})...")
    A_n, B_n = trace_naive_matmul(N)
    A_t, B_t = trace_tiled_matmul(N, T)
    trace_n = interleave(A_n, B_n, offset_B)
    trace_t = interleave(A_t, B_t, offset_B)

    live_n = live_working_set(trace_n)
    live_t = live_working_set(trace_t)

    tau = 2 * T ** 3  # one tiled bk-block's worth of interleaved reads
    wss_n = sliding_working_set(trace_n, tau)
    wss_t = sliding_working_set(trace_t, tau)

    footprint = 2 * N * N
    tile = 2 * T * T
    T_len = len(trace_n)

    print(f"  live working set: naive peak={live_n.max():,}, "
          f"tiled peak={live_t.max():,}  (full A+B = {footprint:,})")
    print(f"  sliding W(τ={tau}): naive max={wss_n.max():,}, "
          f"tiled max={wss_t.max():,}  (one tile 2T² = {tile})")

    fig, (ax_live, ax_slide) = plt.subplots(
        2, 1, figsize=(12, 7.5), sharex=True,
        gridspec_kw={'hspace': 0.32})

    # ----- Top: live working set (grid-style) -----
    ax_live.fill_between(np.arange(T_len), 0, live_n, color='tab:red',
                         alpha=0.18, step='post', linewidth=0, rasterized=True)
    ax_live.plot(live_n, color='tab:red', linewidth=0.9, drawstyle='steps-post',
                 rasterized=True,
                 label=f'naive (peak = {live_n.max():,})')
    ax_live.fill_between(np.arange(T_len), 0, live_t, color='tab:blue',
                         alpha=0.25, step='post', linewidth=0, rasterized=True)
    ax_live.plot(live_t, color='tab:blue', linewidth=0.9, drawstyle='steps-post',
                 rasterized=True,
                 label=f'tiled, T={T} (peak = {live_t.max():,})')
    ax_live.axhline(footprint, color='black', linestyle=':', linewidth=0.8,
                    label=f'full $A{{+}}B$ footprint $2N^2 = {footprint:,}$')
    ax_live.set_ylabel('Live variables on geom stack')
    ax_live.set_title(
        'Live working-set size over time '
        '(address is live on $[$first-use, last-use$]$)',
        fontsize=11, fontweight='bold', pad=6)
    ax_live.set_ylim(0, footprint * 1.05)
    ax_live.grid(True, alpha=0.3)
    ax_live.legend(loc='center right', fontsize=8, framealpha=0.95)

    # ----- Bottom: sliding-window working set at fixed τ -----
    ax_slide.fill_between(np.arange(T_len), 0, wss_n, color='tab:red',
                          alpha=0.18, step='post', linewidth=0,
                          rasterized=True)
    ax_slide.plot(wss_n, color='tab:red', linewidth=0.9,
                  drawstyle='steps-post', rasterized=True,
                  label=f'naive (peak = {wss_n.max():,})')
    ax_slide.fill_between(np.arange(T_len), 0, wss_t, color='tab:blue',
                          alpha=0.25, step='post', linewidth=0,
                          rasterized=True)
    ax_slide.plot(wss_t, color='tab:blue', linewidth=0.9,
                  drawstyle='steps-post', rasterized=True,
                  label=f'tiled, T={T} (peak = {wss_t.max():,})')
    ax_slide.axhline(tile, color='gray', linestyle='--', linewidth=0.8,
                     label=f'one-tile capacity $2T^2 = {tile}$')
    ax_slide.axhline(footprint, color='black', linestyle=':', linewidth=0.8)
    ax_slide.set_xlabel('Access index (time, A and B interleaved)')
    ax_slide.set_ylabel('distinct addrs in window')
    ax_slide.set_title(
        fr'Sliding-window working set $W(t,\tau)$ — '
        fr'$\tau = 2T^3 = {tau:,}$ reads (one bk block)',
        fontsize=11, fontweight='bold', pad=6)
    ax_slide.set_ylim(0, footprint * 1.05)
    ax_slide.grid(True, alpha=0.3)
    ax_slide.legend(loc='center right', fontsize=8, framealpha=0.95)
    ax_slide.set_xlim(0, T_len)

    fig.suptitle(
        f"Working-set size over time — naive vs tiled matmul "
        f"(N={N}, T={T}, $A \\times B^T$)",
        fontsize=13, fontweight='bold', y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(os.path.dirname(__file__), 'working_set_over_time.svg')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
