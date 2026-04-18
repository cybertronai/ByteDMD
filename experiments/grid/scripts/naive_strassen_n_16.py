#!/usr/bin/env -S /Users/yaroslavvb/.local/bin/uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = ["matplotlib", "numpy"]
# ///
"""Self-contained reproducer for naive_strassen(n=16).

What this script does:
  1. Runs the manual (hand-placed) implementation under a logging
     Allocator to harvest the full access log.
  2. Traces the Python reference implementation via bytedmd_ir.trace(),
     which emits an L2 event stream under two-stack argument-promotion
     semantics (inputs live on the arg stack; first read promotes to
     the geometric stack).
  3. Evaluates three trace-based cost heuristics on the event stream:
     - space_dmd       — density-ranked static allocator
     - bytedmd_live    — LRU stack depth with liveness compaction
     - bytedmd_classic — LRU stack depth without compaction
  4. Walks the trace to derive working-set-size-over-time and the
     per-load LRU-depth (reuse distance) timelines.
  5. Saves three PNG plots into ../traces/ and prints the summary.

Run:
    ./scripts/naive_strassen_n_16.py
"""
from __future__ import annotations
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GRID = os.path.abspath(os.path.join(HERE, ".."))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, GRID)
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")

# Pull every helper and size constant used by run_grid.ALGOS entries
# into this namespace so the FN / ARGS / MANUAL expressions below read
# exactly as they do in run_grid.py.
from run_grid import *                                       # noqa: E402,F401,F403
from bytedmd_ir import bytedmd_classic, bytedmd_live, trace  # noqa: E402
from spacedmd import space_dmd                               # noqa: E402
from generate_traces import plot_trace                       # noqa: E402
from trace_diagnostics import (                              # noqa: E402
    walk_live_and_reuse, plot_liveset, plot_reuse_distance,
)

NAME   = 'naive_strassen(n=16)'
SLUG   = 'naive_strassen_n_16'
FN     = alg.matmul_strassen
ARGS   = (mat(N_MM), mat(N_MM))
MANUAL = lambda: man.manual_strassen(N_MM, T=4)


def main() -> None:
    # ---- Trace + heuristic costs -------------------------------------
    events, input_vars = trace(FN, ARGS)
    input_idx = {v: i + 1 for i, v in enumerate(input_vars)}
    costs = {
        "space_dmd":       space_dmd(events, input_idx),
        "bytedmd_live":    bytedmd_live(events, input_idx),
        "manual":          MANUAL(),
        "bytedmd_classic": bytedmd_classic(events, input_idx),
    }

    # ---- Diagnostics: live working set + reuse distance --------------
    ls_t, ls_s, rd_t, rd_d = walk_live_and_reuse(events, input_vars)
    peak_live    = max(ls_s) if ls_s else 0
    max_reuse    = max(rd_d) if rd_d else 0
    median_reuse = sorted(rd_d)[len(rd_d) // 2] if rd_d else 0

    # ---- Access-pattern plot: harvest log from a logging manual run --
    logged = man.Allocator(logging=True)
    man.set_allocator(logged)
    try:
        MANUAL()
    finally:
        man.set_allocator(None)

    traces_dir = os.path.join(GRID, "traces")
    os.makedirs(traces_dir, exist_ok=True)
    plot_trace(
        logged.log, logged.writes, logged.output_writes,
        logged.peak, logged.arg_peak,
        f"{NAME}  —  cost = {logged.cost:,}",
        os.path.join(traces_dir, f"{SLUG}.png"),
    )
    plot_liveset(
        ls_t, ls_s,
        f"{NAME} — live working-set size (peak = {peak_live:,})",
        os.path.join(traces_dir, f"{SLUG}_liveset.png"),
    )
    plot_reuse_distance(
        rd_t, rd_d,
        f"{NAME} — reuse distance per load (max = {max_reuse:,})",
        os.path.join(traces_dir, f"{SLUG}_reuse_distance.png"),
    )

    # ---- Report ------------------------------------------------------
    print(f"{NAME}")
    print(f"  events          {len(events):>12,}")
    for k in ("space_dmd", "bytedmd_live", "manual", "bytedmd_classic"):
        print(f"  {k:<15} {costs[k]:>12,}")
    print(f"  peak_live       {peak_live:>12,}")
    print(f"  max_reuse       {max_reuse:>12,}")
    print(f"  median_reuse    {median_reuse:>12,}")


if __name__ == "__main__":
    main()
