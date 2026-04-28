#!/usr/bin/env python3
"""
LRU vs Belady OPT cache eviction stack visualization.

Classic "LRU killer" example: reference string A B C D repeated 3 times,
cache size 3.  LRU evicts the least-recently-used item and thrashes
(12/12 misses).  Belady's OPT evicts the item with the furthest next use
and achieves 6 hits.

Each colored line traces one variable's slot position over time.
Slot 0 (bottom) = safest (MRU for LRU / soonest-next-use for OPT).
Slot 2 (top of cache) = eviction target.
Items above the dashed boundary have been evicted.

Usage:
    uv run lru_vs_opt_viz.py
    python lru_vs_opt_viz.py
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# ── parameters ────────────────────────────────────────────────────────────────
REFS       = list("ABCDABCDABCD")
CACHE_SIZE = 3
VARS       = ["A", "B", "C", "D"]
COLORS     = {"A": "#1f77b4", "B": "#e07b0d", "C": "#2ca02c", "D": "#c9392b"}
EVICT_Y    = CACHE_SIZE + 0.85   # y-level for evicted items
OUT_DIR    = Path(__file__).parent


# ── LRU simulator ─────────────────────────────────────────────────────────────
def simulate_lru(refs: list[str], k: int) -> list[tuple]:
    """Return (cache_ordered_MRU_first, hit, evicted) per step."""
    cache: list[str] = []
    out = []
    for ref in refs:
        hit = ref in cache
        evicted = None
        if hit:
            cache.remove(ref)
        elif len(cache) == k:
            evicted = cache.pop()          # evict LRU (tail)
        cache.insert(0, ref)               # insert as MRU (head)
        out.append((list(cache), hit, evicted))
    return out


# ── Belady OPT simulator ──────────────────────────────────────────────────────
def simulate_opt(refs: list[str], k: int) -> list[tuple]:
    """Return (cache_ordered_soonest_first, hit, evicted) per step."""
    n = len(refs)
    cache: dict[str, float] = {}          # var → next-use index

    out = []
    for i, ref in enumerate(refs):
        # Next-use index for each variable from position i+1 onward
        future: dict[str, float] = {}
        for j in range(i + 1, n):
            if refs[j] not in future:
                future[refs[j]] = float(j)

        hit = ref in cache
        evicted = None
        if hit:
            cache[ref] = future.get(ref, math.inf)
        else:
            if len(cache) == k:
                # Evict item with furthest next use (tie-break: alphabetical)
                victim = max(cache, key=lambda v: (cache[v], v))
                del cache[victim]
                evicted = victim
            cache[ref] = future.get(ref, math.inf)

        # Order ascending by next-use (soonest first = safest = slot 0)
        ordered = sorted(cache, key=lambda v: (cache[v], v))
        out.append((ordered, hit, evicted))
    return out


# ── position extractor ────────────────────────────────────────────────────────
def extract_positions(results: list[tuple]) -> dict[str, list[float]]:
    """y-position for each variable at each time step (after the access)."""
    pos: dict[str, list[float]] = {v: [] for v in VARS}
    for cache_ordered, *_ in results:
        for v in VARS:
            if v in cache_ordered:
                pos[v].append(float(cache_ordered.index(v)))
            else:
                pos[v].append(EVICT_Y)
    return pos


# ── step-function path builder ────────────────────────────────────────────────
DELTA = 0.20   # half-width of diagonal transition (fraction of one time unit)

def build_step_path(ys: list[float]) -> tuple[list[float], list[float]]:
    """Convert per-tick y-values into a path with diagonal transitions.

    Each tick t is rendered as a horizontal band over [t+DELTA, (t+1)-DELTA].
    When y changes between consecutive ticks, a diagonal segment connects the
    band endpoints, spanning [(t+1)-DELTA, (t+1)+DELTA] centered on the tick
    boundary.  Multiple simultaneous transitions therefore produce distinct
    crossing diagonals rather than coincident verticals.
    """
    T = len(ys)
    px: list[float] = []
    py: list[float] = []

    for t in range(T):
        y = ys[t]

        if t == 0:
            # First tick: no preceding diagonal; start flush at x=0
            px.append(0.0)
            py.append(y)
        # else: path already ends at (t+DELTA, y) from the previous diagonal

        # End of horizontal band for tick t
        x_end = ((t + 1) - DELTA) if t < T - 1 else (t + 0.82)
        px.append(x_end)
        py.append(y)

        # Diagonal to the next tick's y-value (skip for the last tick)
        if t < T - 1:
            px.append((t + 1) + DELTA)
            py.append(ys[t + 1])

    return px, py


# ── draw one panel ─────────────────────────────────────────────────────────────
def draw_panel(
    ax: matplotlib.axes.Axes,
    results: list[tuple],
    pos: dict[str, list[float]],
    policy_name: str,
    policy_note: str,
) -> None:
    T      = len(REFS)
    bnd    = CACHE_SIZE - 0.5      # cache-boundary y

    # Evicted zone shading + boundary line
    ax.axhspan(bnd, EVICT_Y + 0.38, color="#aaaaaa", alpha=0.10, zorder=0)
    ax.axhline(bnd, color="#444444", lw=1.2, ls="--", zorder=2, alpha=0.75)
    ax.text(T - 0.12, bnd + 0.07, "eviction boundary",
            ha="right", va="bottom", fontsize=7.5, color="#555555")

    # Slot ordering note (bottom-left inside plot)
    ax.text(0.01, 0.01, policy_note, transform=ax.transAxes,
            fontsize=7.5, color="#444", va="bottom", style="italic")

    for v in VARS:
        color = COLORS[v]
        ys    = pos[v]
        px, py = build_step_path(ys)

        # Draw line in two passes: solid (in-cache) and dashed (evicted)
        # Split px/py into segments that are entirely one or the other
        def _draw_segs(xs, ys_seg, in_cache: bool) -> None:
            ls  = "-"   if in_cache else "--"
            lw  = 2.3   if in_cache else 1.3
            alp = 1.0   if in_cache else 0.45
            ax.plot(xs, ys_seg, color=color, ls=ls, lw=lw, alpha=alp,
                    solid_capstyle="round", zorder=3)

        seg_x: list[float] = []
        seg_y_: list[float] = []
        cur_in: bool | None = None
        prev_x: float | None = None
        prev_y_: float | None = None

        for xi, yi in zip(px, py):
            in_c = yi < CACHE_SIZE
            if cur_in is None:
                cur_in = in_c
            if in_c != cur_in:
                # Transition: flush current segment, start new with overlap
                _draw_segs(seg_x, seg_y_, cur_in)
                seg_x  = [prev_x, xi] if prev_x is not None else [xi]
                seg_y_ = [prev_y_, yi] if prev_y_ is not None else [yi]
                cur_in = in_c
            else:
                seg_x.append(xi)
                seg_y_.append(yi)
            prev_x = xi
            prev_y_ = yi
        if seg_x:
            _draw_segs(seg_x, seg_y_, cur_in)

        # Hit / miss markers at centre of each accessed tick's horizontal band
        for t, (_, hit, _) in enumerate(results):
            if REFS[t] == v:
                xm = t + 0.5
                ym = ys[t]
                if hit:
                    ax.plot(xm, ym, "o", ms=8, color=color,
                            mec="white", mew=1.2, zorder=6)
                else:
                    ax.plot(xm, ym, "x", ms=10, color=color,
                            mew=2.4, zorder=6)

    # Axis labels and ticks
    hits   = sum(h for _, h, _ in results)
    misses = T - hits
    ax.set_title(
        f"{policy_name}\n{hits} hit{'s' if hits != 1 else ''},  "
        f"{misses} miss{'es' if misses != 1 else ''}",
        fontsize=12, fontweight="bold", pad=9,
    )

    yticks = list(range(CACHE_SIZE)) + [EVICT_Y]
    ylabs  = (
        ["Slot 0  (safest)"]
        + [f"Slot {i}" for i in range(1, CACHE_SIZE - 1)]
        + [f"Slot {CACHE_SIZE - 1}  (eviction target)"]
        + ["Evicted"]
    )
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabs, fontsize=8.5)
    ax.set_ylim(-0.6, EVICT_Y + 0.42)

    ax.set_xticks(range(T))
    ax.set_xticklabels([f"t={t}\n{REFS[t]}" for t in range(T)], fontsize=7.8)
    ax.set_xlim(-0.3, T - 0.18)

    ax.grid(axis="x", alpha=0.25, zorder=0)


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    lru_res = simulate_lru(REFS, CACHE_SIZE)
    opt_res = simulate_opt(REFS, CACHE_SIZE)
    lru_pos = extract_positions(lru_res)
    opt_pos = extract_positions(opt_res)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(15, 6.6), sharey=True)
    fig.subplots_adjust(wspace=0.06)

    draw_panel(ax_l, lru_res, lru_pos,
               "LRU  (Least Recently Used)",
               "slot 0 = most recently used")
    draw_panel(ax_r, opt_res, opt_pos,
               "Belady OPT  (Furthest Next Use)",
               "slot 0 = soonest next use")
    ax_r.set_yticklabels([])          # shared y-axis; hide duplicate labels

    fig.text(0.025, 0.5, "Cache slot position", va="center",
             rotation="vertical", fontsize=10)

    fig.suptitle(
        f"LRU vs Belady OPT — reference string  A·B·C·D × 3,  cache size {CACHE_SIZE}",
        fontsize=13, fontweight="bold", y=1.04,
    )

    # Legend
    handles = [
        Line2D([0], [0], color=COLORS[v], lw=2.5, label=f"Variable {v}")
        for v in VARS
    ] + [
        Line2D([0], [0], marker="o", color="#555", ls="", ms=8,
               mec="white", mew=1, label="Cache hit  (●)"),
        Line2D([0], [0], marker="x", color="#555", ls="", ms=9,
               markeredgewidth=2.4, label="Cache miss  (✕)"),
        Line2D([0], [0], color="#555", lw=1.3, ls="--", alpha=0.45,
               label="Evicted (dashed)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(VARS) + 3,
               fontsize=9, framealpha=0.92, bbox_to_anchor=(0.5, 1.02))

    caption = (
        "Reference string  A B C D · A B C D · A B C D  (length 12, cache size 3). "
        "LRU never retains the next needed item on this length-4 cycle — every access "
        "misses (12/12). "
        "Belady's OPT knows future accesses and always evicts the item with the furthest "
        "next use, achieving 6/12 hits. "
        "Lines trace each variable's cache slot over time. "
        "Solid lines = in cache; dashed = evicted. "
        "Dashed horizontal = cache boundary."
    )
    fig.text(0.5, -0.025, caption, ha="center", va="top",
             fontsize=8.3, style="italic",
             bbox=dict(boxstyle="round,pad=0.5", fc="#f5f5f5", alpha=0.9))

    plt.tight_layout(rect=[0.04, 0.06, 1.0, 1.0])

    png = OUT_DIR / "lru_vs_opt.png"
    svg = OUT_DIR / "lru_vs_opt.svg"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    print(f"Saved  {png}")
    print(f"Saved  {svg}")


if __name__ == "__main__":
    main()
