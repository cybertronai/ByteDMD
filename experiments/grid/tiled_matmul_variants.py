#!/Users/yaroslavvb/.local/bin/uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = ["matplotlib", "numpy"]
# ///
"""Render tiled_matmul's trace under several rendering strategies so we
can pick the cleanest anti-aliased output. Saves into traces/variants/."""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import manual as man


def harvest() -> tuple:
    logged = man.Allocator(logging=True)
    man.set_allocator(logged)
    try:
        man.manual_tiled_matmul(16)
    finally:
        man.set_allocator(None)

    arg_t, arg_y, scr_t, scr_y, out_t, out_y = [], [], [], [], [], []
    for t, (space, addr) in enumerate(logged.log):
        if space == "arg":
            arg_t.append(t); arg_y.append(-addr)
        elif space == "output":
            out_t.append(t); out_y.append(addr)
        else:
            scr_t.append(t); scr_y.append(addr)
    w_t, w_y = (list(zip(*logged.writes))[0], list(zip(*logged.writes))[1]) \
        if logged.writes else ([], [])
    return arg_t, arg_y, scr_t, scr_y, out_t, out_y, w_t, w_y


def render(variant: str, data: tuple, out_path: str, **kw) -> None:
    arg_t, arg_y, scr_t, scr_y, out_t, out_y, w_t, w_y = data

    figsize = kw.get("figsize", (11, 3.8))
    dpi = kw.get("dpi", 120)
    rasterized = kw.get("rasterized", True)
    scatter_s = kw.get("scatter_s", 0.6)
    scatter_alpha = kw.get("scatter_alpha", 0.45)
    out_s = kw.get("out_s", 3.0)
    backend_hint = kw.get("title_suffix", "")

    fig, ax = plt.subplots(figsize=figsize)
    # antialiased is True by default but pass explicitly for clarity.
    common = dict(rasterized=rasterized, linewidths=0)
    ax.scatter(scr_t, scr_y, s=scatter_s, c="tab:blue",
               alpha=scatter_alpha, label="scratch read", **common)
    ax.scatter(arg_t, arg_y, s=scatter_s, c="tab:green",
               alpha=scatter_alpha, label="arg read (shifted -addr)", **common)
    if w_t:
        ax.scatter(w_t, w_y, s=0.8, c="tab:orange",
                   alpha=0.55, label="scratch write", **common)
    if out_t:
        ax.scatter(out_t, out_y, s=out_s, c="#8B008B",
                   alpha=0.9, zorder=5, label="output read (epilogue)",
                   **common)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_xlabel("Access index (time)")
    ax.set_ylabel("Physical address (scratch positive / arg negative)")
    ax.set_title(f"tiled_matmul(n=16) — variant: {variant}{backend_hint}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", markerscale=8, fontsize=8, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def render_hexbin(data: tuple, out_path: str) -> None:
    """Density plot: replaces each dense band with a smoothly anti-aliased
    hexbin heat-map. Reads become one hexbin per stack; writes stay as
    markers. Best for algorithms with many co-located reads."""
    arg_t, arg_y, scr_t, scr_y, out_t, out_y, w_t, w_y = data
    fig, axes = plt.subplots(3, 1, figsize=(11, 7.5),
                             sharex=True, gridspec_kw={"hspace": 0.12})
    ax1, ax2, ax3 = axes

    if scr_t:
        ax1.hexbin(scr_t, scr_y, gridsize=(120, 40),
                   cmap="Blues", mincnt=1, linewidths=0)
    ax1.set_ylabel("scratch read addr")
    ax1.grid(True, alpha=0.3)

    if arg_t:
        ax2.hexbin(arg_t, [-y for y in arg_y], gridsize=(120, 40),
                   cmap="Greens", mincnt=1, linewidths=0)
    ax2.set_ylabel("arg read addr")
    ax2.grid(True, alpha=0.3)

    if w_t:
        ax3.scatter(w_t, w_y, s=3.0, c="tab:orange", alpha=0.8,
                    label="scratch write", linewidths=0)
    if out_t:
        ax3.scatter(out_t, out_y, s=6.0, c="#8B008B", alpha=0.95,
                    label="output read (epilogue)", linewidths=0, zorder=5)
    ax3.set_ylabel("write / output-read addr")
    ax3.set_xlabel("Access index (time)")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper right", fontsize=8, framealpha=0.85)

    fig.suptitle("tiled_matmul(n=16) — variant: hexbin_panels", y=0.995)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    out_dir = os.path.join(HERE, "traces", "variants")
    os.makedirs(out_dir, exist_ok=True)
    data = harvest()

    # 1. Baseline (same as generate_traces): rasterized, dpi=120.
    render("baseline_raster_dpi120", data,
           os.path.join(out_dir, "v1_baseline_raster_dpi120.png"),
           dpi=120, rasterized=True, scatter_s=0.6, scatter_alpha=0.45)

    # 2. Vector scatter (rasterized=False) at dpi=120: matplotlib draws
    #    each marker as an antialiased vector primitive.
    render("vector_markers_dpi120", data,
           os.path.join(out_dir, "v2_vector_dpi120.png"),
           dpi=120, rasterized=False, scatter_s=0.8, scatter_alpha=0.55)

    # 3. High-DPI raster (dpi=300).
    render("raster_dpi300", data,
           os.path.join(out_dir, "v3_raster_dpi300.png"),
           dpi=300, rasterized=True, scatter_s=0.6, scatter_alpha=0.45)

    # 4. High-DPI vector markers + bigger figure.
    render("vector_dpi300_bigfig", data,
           os.path.join(out_dir, "v4_vector_dpi300_bigfig.png"),
           dpi=300, rasterized=False, scatter_s=1.0, scatter_alpha=0.55,
           figsize=(14, 4.8))

    # 5. SVG (pure vector — no rasterization anywhere).
    render("svg_vector", data,
           os.path.join(out_dir, "v5_vector.svg"),
           dpi=120, rasterized=False, scatter_s=0.8, scatter_alpha=0.55)

    # 6. Larger markers for clarity.
    render("chunky_markers_dpi200", data,
           os.path.join(out_dir, "v6_chunky_dpi200.png"),
           dpi=200, rasterized=False, scatter_s=2.0, scatter_alpha=0.45,
           out_s=5.0, figsize=(12, 4.2))

    # 7. Hexbin density panels.
    render_hexbin(data, os.path.join(out_dir, "v7_hexbin_panels.png"))

    print("Wrote variants to", out_dir)
    for fn in sorted(os.listdir(out_dir)):
        path = os.path.join(out_dir, fn)
        size = os.path.getsize(path)
        print(f"  {fn:<40} {size/1024:>8.1f} KB")


if __name__ == "__main__":
    main()
