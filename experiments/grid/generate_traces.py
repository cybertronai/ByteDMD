#!/Users/yaroslavvb/.local/bin/uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = ["matplotlib", "numpy"]
# ///
"""Emit a memory-trace PNG for every algorithm in run_grid.ALGOS.

For each algorithm, swap in a logging Allocator, call the manual impl,
harvest the full address sequence, and plot access_index (x) vs
address (y). Writes traces/<slug>.png and prints a one-line summary.
"""
from __future__ import annotations

import os
import re
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
import run_grid as rg


def slugify(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()


def plot_trace(log: list[int],
               writes: list[tuple[int, int]],
               output_writes: list[tuple[int, int]],
               peak: int,
               title: str,
               out_path: str) -> None:
    """Reads (blue) and scratch/temp writes (orange) plot at their real
    addresses. Output writes (red) are shifted above `peak` so they
    occupy a dedicated band above the highest used byte — the
    "outputs above peak" visualization the user asked for."""
    n = len(log)
    fig, ax = plt.subplots(figsize=(11, 3.4))
    if n > 0:
        ax.scatter(
            np.arange(n), np.asarray(log),
            s=0.6, c="tab:blue", alpha=0.45,
            rasterized=True, linewidths=0,
            label="read",
        )
    if writes:
        wt, wa = zip(*writes)
        ax.scatter(
            wt, wa,
            s=0.8, c="tab:orange", alpha=0.55,
            rasterized=True, linewidths=0,
            label="scratch write",
        )
    if output_writes:
        wt, wa = zip(*output_writes)
        wa_shifted = [peak + addr for addr in wa]
        ax.scatter(
            wt, wa_shifted,
            s=0.8, c="tab:red", alpha=0.7,
            rasterized=True, linewidths=0,
            label="output write (shifted +peak)",
        )
        ax.axhline(peak, color="gray", linestyle="--",
                   linewidth=0.6, alpha=0.5)
    ax.set_xlabel("Access index (time)")
    ax.set_ylabel("Physical address")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    if n > 0 or writes or output_writes:
        ax.legend(loc="upper left", markerscale=8, fontsize=8, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    traces_dir = os.path.join(HERE, "traces")
    os.makedirs(traces_dir, exist_ok=True)

    print(f"{'algorithm':<40} {'reads':>10} {'scratch_w':>10} {'out_w':>10} {'cost':>12}  file")
    print("-" * 112)
    for name, _fn, _args, manual_fn in rg.ALGOS:
        slug = slugify(name)
        logged = man.Allocator(logging=True)
        man.set_allocator(logged)
        try:
            manual_fn()
        finally:
            man.set_allocator(None)

        out_path = os.path.join(traces_dir, f"{slug}.png")
        title = f"{name}  —  cost = {logged.cost:,}"
        plot_trace(logged.log, logged.writes, logged.output_writes,
                   logged.peak, title, out_path)
        rel = os.path.relpath(out_path, HERE)
        print(f"{name:<40} {len(logged.log):>10,} "
              f"{len(logged.writes):>10,} {len(logged.output_writes):>10,} "
              f"{logged.cost:>12,}  {rel}")


if __name__ == "__main__":
    main()
