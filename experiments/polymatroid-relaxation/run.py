#!/usr/bin/env -S /Users/yaroslavvb/.local/bin/uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy"]
# ///
"""Run polymatroid_lower_bound on a representative subset of the grid
algorithms and emit a comparison table against the other lower bounds
and achievable cost columns from `experiments/grid/grid.csv`.

Usage:
    ./run.py                        # all representative algos
    POLY_TIMEOUT=120 ./run.py       # per-algo timeout in seconds
"""
from __future__ import annotations

import csv
import os
import sys
import time
from typing import Callable, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
GRID = os.path.join(ROOT, "experiments", "grid")
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, GRID)

from bytedmd_ir import (  # noqa: E402
    bytedmd_opt,
    matmul_rmm,
    matmul_tiled,
    splitting_lower_bound,
    static_opt_lb,
    trace,
)
import algorithms as alg  # noqa: E402
from polymatroid_lb import polymatroid_lower_bound  # noqa: E402

# Representative subset of grid ALGOS — one or two per family. Sized to
# finish in a few minutes on a workstation; large traces (e.g.,
# regular_conv, naive_attn at the grid's tile sizes) are skipped because
# the LP solver scales super-linearly in the number of intervals.
N_MM = 16
N_MV = 64
N_LU = 32

REPRESENTATIVE: List[Tuple[str, Callable, Tuple]] = [
    ("naive_matmul(n=16)",
        alg.matmul_naive_abt, (alg.mat(N_MM) if hasattr(alg, "mat") else
                                [[1.0] * N_MM for _ in range(N_MM)],
                                [[1.0] * N_MM for _ in range(N_MM)])),
    ("naive_2d_tiled_matmul(n=16,T=4)",
        lambda A, B: alg.matmul_naive_2d_tiled(A, B, tile=4),
        ([[1.0] * N_MM for _ in range(N_MM)],
         [[1.0] * N_MM for _ in range(N_MM)])),
    ("tiled_matmul(n=16)",
        matmul_tiled, ([[1.0] * N_MM for _ in range(N_MM)],
                       [[1.0] * N_MM for _ in range(N_MM)])),
    ("rmm(n=16)",
        matmul_rmm, ([[1.0] * N_MM for _ in range(N_MM)],
                     [[1.0] * N_MM for _ in range(N_MM)])),
    ("naive_strassen(n=16)",
        alg.matmul_strassen, ([[1.0] * N_MM for _ in range(N_MM)],
                              [[1.0] * N_MM for _ in range(N_MM)])),
    ("matvec_row(n=64)",
        alg.matvec_row, ([[1.0] * N_MV for _ in range(N_MV)],
                         [1.0] * N_MV)),
    ("matvec_col(n=64)",
        alg.matvec_col, ([[1.0] * N_MV for _ in range(N_MV)],
                         [1.0] * N_MV)),
    ("matvec_blocked(n=64,B=8)",
        lambda A, x: alg.matvec_blocked(A, x, B=8),
        ([[1.0] * N_MV for _ in range(N_MV)], [1.0] * N_MV)),
    ("fft_iterative(N=256)",
        alg.fft_iterative, ([1.0] * 256,)),
    ("fft_recursive(N=256)",
        alg.fft_recursive, ([1.0] * 256,)),
    ("lu_no_pivot(n=32)",
        alg.lu_no_pivot, ([[1.0] * N_LU for _ in range(N_LU)],)),
    ("blocked_lu(n=32,NB=8)",
        lambda A: alg.blocked_lu(A, NB=8),
        ([[1.0] * N_LU for _ in range(N_LU)],)),
    ("cholesky(n=32)",
        alg.cholesky, ([[1.0] * N_LU for _ in range(N_LU)],)),
    ("transpose_naive(n=32)",
        alg.transpose_naive, ([[1.0] * N_LU for _ in range(N_LU)],)),
    ("transpose_blocked(n=32)",
        alg.transpose_blocked, ([[1.0] * N_LU for _ in range(N_LU)],)),
    ("quicksort(N=64)",
        alg.quicksort, ([1.0] * 64,)),
    ("mergesort(N=64)",
        alg.mergesort, ([1.0] * 64,)),
    ("lcs_dp(32x32)",
        alg.lcs_dp, ([1.0] * 32, [1.0] * 32)),
]


def _load_grid_metrics():
    """Read `experiments/grid/grid.csv` keyed by algorithm name."""
    path = os.path.join(GRID, "grid.csv")
    by_name = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            by_name[row["algorithm"]] = row
    return by_name


def main():
    timeout = float(os.environ.get("POLY_TIMEOUT", "300"))
    grid = _load_grid_metrics()

    out = []
    for name, fn, args in REPRESENTATIVE:
        events, input_vars = trace(fn, args)
        iidx = {v: i + 1 for i, v in enumerate(input_vars)}

        t0 = time.perf_counter()
        try:
            pm = polymatroid_lower_bound(events, iidx)
            dt = time.perf_counter() - t0
            status = "ok"
        except Exception as e:
            pm = None
            dt = time.perf_counter() - t0
            status = f"err:{e}"

        # Cross-check against the existing grid columns.
        g = grid.get(name, {})
        sob = static_opt_lb(events, iidx)
        sl = splitting_lower_bound(events, iidx)
        bo = bytedmd_opt(events, iidx)

        row = {
            "algorithm": name,
            "polymatroid_lb": "" if pm is None else pm,
            "split_lb": int(sl),
            "static_opt_lb": int(sob),
            "bytedmd_opt": bo,
            "space_dmd": int(g.get("space_dmd", 0) or 0),
            "bytedmd_live": int(g.get("bytedmd_live", 0) or 0),
            "manual": int(g.get("manual", 0) or 0),
            "bytedmd_classic": int(g.get("bytedmd_classic", 0) or 0),
            "n_events": len(events),
            "poly_seconds": round(dt, 2),
            "poly_status": status,
        }
        out.append(row)
        print(
            f"  {name:<36} polymatroid={'-' if pm is None else f'{pm:>10,}'}  "
            f"split={int(sl):>10,}  static={int(sob):>10,}  "
            f"manual={int(g.get('manual', 0) or 0):>10,}  "
            f"t={dt:.2f}s"
        )

    csv_path = os.path.join(HERE, "results.csv")
    cols = [
        "algorithm",
        "polymatroid_lb",
        "split_lb",
        "static_opt_lb",
        "bytedmd_opt",
        "space_dmd",
        "bytedmd_live",
        "manual",
        "bytedmd_classic",
        "n_events",
        "poly_seconds",
        "poly_status",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in out:
            w.writerow(r)
    print(f"\nSaved {csv_path}")

    # Also emit a markdown table for the README.
    md_path = os.path.join(HERE, "results.md")

    def _fmt(v):
        return f"{v:,}" if isinstance(v, int) else (str(v) if v else "—")

    cols_md = [
        ("algorithm", "algorithm"),
        ("polymatroid_lb", "polymatroid_lb"),
        ("split_lb", "split_lb"),
        ("static_opt_lb", "static_opt_lb"),
        ("bytedmd_opt", "bytedmd_opt"),
        ("space_dmd", "space_dmd"),
        ("bytedmd_live", "bytedmd_live"),
        ("manual", "manual"),
    ]
    widths = {c: max(len(label), max(len(_fmt(r.get(c, "")))
                                     for r in out))
              for c, label in cols_md}
    header = "| " + " | ".join(label.ljust(widths[c])
                               for c, label in cols_md) + " |"
    sep = "|" + "|".join("-" * (widths[c] + 1) +
                          (":" if c != "algorithm" else "")
                          for c, label in cols_md) + "|"
    lines = [header, sep]
    for r in out:
        cells = [r["algorithm"].ljust(widths["algorithm"])]
        for c, _ in cols_md[1:]:
            cells.append(_fmt(r.get(c, "")).rjust(widths[c]))
        lines.append("| " + " | ".join(cells) + " |")
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()
