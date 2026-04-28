#!/usr/bin/env python3
"""
LRU vs MRU vs Belady OPT cache eviction stack visualization.

Classic "LRU killer" trace: A B C D repeated 3 times, cache size 3.

  LRU: 0/12 hits  — thrashes on every access of the length-4 cycle
  MRU: 6/12 hits  — matches OPT (on pure cyclic traces, MRU item = furthest next use)
  OPT: 6/12 hits  — theoretical optimum (Belady's furthest-next-use rule)

Each colored line traces one variable's cache slot across time steps.
Slot 0 (bottom) = safest under that policy; Slot k-1 = eviction target.
Lines connect per-tick centers with straight segments (diagonal when the
slot changes, horizontal when it stays).  Solid = in cache; dashed = evicted.

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
    """LRU: evict least recently used.
    Returns (cache ordered MRU-first, hit, evicted) per step.
    Slot 0 = MRU (safest for LRU); Slot k-1 = LRU (eviction target)."""
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


# ── MRU simulator ─────────────────────────────────────────────────────────────
def simulate_mru(refs: list[str], k: int) -> list[tuple]:
    """MRU: evict most recently used.
    Returns (cache ordered LRU-first, hit, evicted) per step.
    Slot 0 = LRU (safest for MRU); Slot k-1 = MRU (eviction target)."""
    cache: list[str] = []
    out = []
    for ref in refs:
        hit = ref in cache
        evicted = None
        if hit:
            cache.remove(ref)
        elif len(cache) == k:
            evicted = cache.pop()          # evict MRU (tail)
        cache.append(ref)                  # insert as MRU (tail)
        out.append((list(cache), hit, evicted))
    return out


# ── Belady OPT simulator ──────────────────────────────────────────────────────
def simulate_opt(refs: list[str], k: int) -> list[tuple]:
    """OPT: evict item with furthest next use (Belady's algorithm).
    Returns (cache ordered soonest-first, hit, evicted) per step.
    Slot 0 = soonest next use (safest); Slot k-1 = furthest (eviction target)."""
    n = len(refs)
    cache: dict[str, float] = {}

    out = []
    for i, ref in enumerate(refs):
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
                victim = max(cache, key=lambda v: (cache[v], v))
                del cache[victim]
                evicted = victim
            cache[ref] = future.get(ref, math.inf)

        ordered = sorted(cache, key=lambda v: (cache[v], v))
        out.append((ordered, hit, evicted))
    return out


# ── position extractor ────────────────────────────────────────────────────────
def extract_positions(results: list[tuple]) -> dict[str, list[float]]:
    """y-position for each variable at each time step (after the access).
    Index in the cache's ordered list maps directly to slot number."""
    pos: dict[str, list[float]] = {v: [] for v in VARS}
    for cache_ordered, *_ in results:
        for v in VARS:
            if v in cache_ordered:
                pos[v].append(float(cache_ordered.index(v)))
            else:
                pos[v].append(EVICT_Y)
    return pos


# ── draw one panel ─────────────────────────────────────────────────────────────
def draw_panel(
    ax: matplotlib.axes.Axes,
    results: list[tuple],
    pos: dict[str, list[float]],
    policy_name: str,
    policy_note: str,
    show_yticks: bool = False,
) -> None:
    T      = len(REFS)
    bnd    = CACHE_SIZE - 0.5      # cache-boundary y

    # Evicted zone shading + boundary line
    ax.axhspan(bnd, EVICT_Y + 0.38, color="#aaaaaa", alpha=0.10, zorder=0)
    ax.axhline(bnd, color="#444444", lw=1.2, ls="--", zorder=2, alpha=0.75)
    ax.text(T - 1.0, bnd + 0.07, "eviction boundary",
            ha="right", va="bottom", fontsize=7.5, color="#555555")

    # Slot ordering note (bottom-left inside plot)
    ax.text(0.01, 0.01, policy_note, transform=ax.transAxes,
            fontsize=7.5, color="#444", va="bottom", style="italic")

    xs = list(range(T))  # one integer x-coordinate per tick

    for v in VARS:
        color = COLORS[v]
        ys    = pos[v]

        # Draw piecewise-linear path, split into solid (in-cache) and dashed
        # (evicted) segments.  A transition between states draws the connecting
        # diagonal in the destination style.
        def _draw(sx: list[float], sy: list[float], in_cache: bool) -> None:
            if len(sx) < 2:
                return
            ls  = "-"   if in_cache else "--"
            lw  = 2.2   if in_cache else 1.3
            alp = 1.0   if in_cache else 0.45
            ax.plot(sx, sy, color=color, ls=ls, lw=lw, alpha=alp,
                    solid_capstyle="round", zorder=3)

        seg_x: list[float] = []
        seg_y: list[float] = []
        cur_in: bool | None = None
        prev_x: float | None = None
        prev_y: float | None = None

        for xi, yi in zip(xs, ys):
            in_c = yi < CACHE_SIZE
            if cur_in is None:
                cur_in = in_c
            if in_c != cur_in:
                _draw(seg_x, seg_y, cur_in)
                # Carry last point into new segment so the diagonal is drawn
                seg_x  = [prev_x, xi] if prev_x is not None else [xi]
                seg_y  = [prev_y, yi] if prev_y is not None else [yi]
                cur_in = in_c
            else:
                seg_x.append(xi)
                seg_y.append(yi)
            prev_x, prev_y = xi, yi
        _draw(seg_x, seg_y, cur_in)

        # Hit / miss markers at each tick center where this variable is accessed
        for t, (_, hit, _) in enumerate(results):
            if REFS[t] == v:
                if hit:
                    ax.plot(t, ys[t], "o", ms=8, color=color,
                            mec="white", mew=1.2, zorder=6)
                else:
                    ax.plot(t, ys[t], "x", ms=10, color=color,
                            mew=2.4, zorder=6)

    # Axis cosmetics
    hits   = sum(h for _, h, _ in results)
    misses = T - hits
    ax.set_title(
        f"{policy_name}\n{hits} hit{'s' if hits != 1 else ''},  "
        f"{misses} miss{'es' if misses != 1 else ''}",
        fontsize=11.5, fontweight="bold", pad=9,
    )

    yticks = list(range(CACHE_SIZE)) + [EVICT_Y]
    ylabs  = (
        ["Slot 0  (safest)"]
        + [f"Slot {i}" for i in range(1, CACHE_SIZE - 1)]
        + [f"Slot {CACHE_SIZE - 1}  (eviction target)"]
        + ["Evicted"]
    )
    ax.set_yticks(yticks)
    if show_yticks:
        ax.set_yticklabels(ylabs, fontsize=8.5)
    else:
        ax.set_yticklabels([])
    ax.set_ylim(-0.6, EVICT_Y + 0.42)

    ax.set_xticks(range(T))
    ax.set_xticklabels([f"t={t}\n{REFS[t]}" for t in range(T)], fontsize=7.8)
    ax.set_xlim(-0.5, T - 0.5)

    ax.grid(axis="x", alpha=0.25, zorder=0)


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    lru_res = simulate_lru(REFS, CACHE_SIZE)
    mru_res = simulate_mru(REFS, CACHE_SIZE)
    opt_res = simulate_opt(REFS, CACHE_SIZE)
    lru_pos = extract_positions(lru_res)
    mru_pos = extract_positions(mru_res)
    opt_pos = extract_positions(opt_res)

    fig, (ax_l, ax_m, ax_r) = plt.subplots(
        1, 3, figsize=(21, 6.6), sharey=True
    )
    fig.subplots_adjust(wspace=0.06)

    draw_panel(ax_l, lru_res, lru_pos,
               "LRU  (Least Recently Used)",
               "slot 0 = most recently used",
               show_yticks=True)
    draw_panel(ax_m, mru_res, mru_pos,
               "MRU  (Most Recently Used)",
               "slot 0 = least recently used")
    draw_panel(ax_r, opt_res, opt_pos,
               "Belady OPT  (Furthest Next Use)",
               "slot 0 = soonest next use")

    fig.text(0.025, 0.5, "Cache slot position", va="center",
             rotation="vertical", fontsize=10)

    fig.suptitle(
        f"LRU vs MRU vs Belady OPT — reference string  A·B·C·D × 3,"
        f"  cache size {CACHE_SIZE}",
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
        "LRU evicts the least-recently-used item and thrashes on this length-4 cycle (0/12 hits). "
        "MRU evicts the most-recently-used item — counterintuitively, this achieves "
        "the same 6/12 hits as Belady's OPT. "
        "On a pure cyclic trace the MRU item is always the one with the furthest next use, "
        "so MRU and OPT make identical eviction decisions (the two right panels are the same). "
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
