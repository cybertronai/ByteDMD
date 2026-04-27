"""Discrete polymatroid-relaxation lower bound on optimal static-allocator
cost — see gemini/polymatroid-relaxation.md.

Combines two ideas:

  1. **Discrete-calculus identity.** Decompose the per-read fetch cost
     `C(d) = ⌈√d⌉` into telescoping unit jumps:

         C(d) = 1 + Σ_{c ≥ 1, c² ≤ d−1} 1.

     So total cost = Σ_loads C(d_load) =
         R_total + Σ_{c ≥ 1} #{loads at depth > c²}.

  2. **Polymatroid LP.** For each capacity `k`, the maximum total reads
     that can be packed into `k` distinct physical addresses is the
     LP solution to

         max  Σ_v reads_v · x_v
         s.t. Σ_{v ∈ K} x_v ≤ k     (for every maximal clique K)
              0 ≤ x_v ≤ 1.

     Interval graphs have the consecutive-ones property → the
     constraint matrix is totally unimodular → the fractional LP has
     an integer optimum. Interval graphs are perfect → max-clique = c
     guarantees a valid c-coloring (= valid c-address packing).

  Combining: `LB = R_total + Σ_{c=1..⌊√(ω−1)⌋} (R_total − M[c²])`
  where M[k] is the LP value at capacity k and ω is the peak live
  size of the trace. Only square capacities matter (the ceil-sqrt
  step jumps once per square boundary), which collapses the LP-solve
  count from O(ω) to O(√ω).

  This is a lower bound on any *static* allocator under ⌈√addr⌉
  pricing. The bound is tighter than `mwis_lower_bound` (which uses
  only the single MWIS weight via water-pouring) and is the discrete
  cousin of `lp_lower_bound` (continuous-sqrt MWIS layering).
"""
from __future__ import annotations

import math
import os
import sys
from typing import Dict, List, Optional, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
from scipy.optimize import linprog  # noqa: E402
from scipy.sparse import lil_matrix  # noqa: E402

from bytedmd_ir import (  # noqa: E402
    L2Event,
    L2Load,
    L2Store,
    _Interval,
    _extract_cliques,
    _extract_intervals,
)


def _extract_intervals_two_stack(
    events: Sequence[L2Event],
    input_arg_idx: Dict[int, int],
) -> List[_Interval]:
    """Like `_extract_intervals` but also produces an interval for every
    input variable, scoped [first L2Load, last L2Load] with `reads`
    excluding the compulsory first read (charged separately to the arg
    stack). Matches the Two-Stack convention used by `global_density` and
    `local_density`.
    """
    starts: Dict[int, int] = {}
    ends: Dict[int, int] = {}
    reads: Dict[int, int] = {}
    is_input: Dict[int, bool] = {}
    for i, ev in enumerate(events):
        if isinstance(ev, L2Store):
            starts[ev.var] = i
            if ev.var not in ends:
                ends[ev.var] = i
            is_input[ev.var] = False
        elif isinstance(ev, L2Load):
            if ev.var not in starts:
                # First mention is a Load → input promoted to geom stack.
                starts[ev.var] = i
                is_input[ev.var] = ev.var in input_arg_idx
            ends[ev.var] = i
            reads[ev.var] = reads.get(ev.var, 0) + 1
    out: List[_Interval] = []
    for var, start in starts.items():
        r = reads.get(var, 0)
        if is_input.get(var):
            r -= 1  # first read paid against arg stack
        if r > 0:
            out.append(_Interval(var, start, ends[var], r))
    return out


def _arg_stack_first_load_cost(
    events: Sequence[L2Event],
    input_arg_idx: Dict[int, int],
) -> int:
    """Sum of ⌈√(arg_idx)⌉ for the first load of every input — the
    compulsory cold-miss cost paid on promotion from the arg stack."""
    cost = 0
    seen: set = set()
    for ev in events:
        if isinstance(ev, L2Load) and ev.var in input_arg_idx \
                and ev.var not in seen:
            arg_idx = input_arg_idx[ev.var]
            cost += math.isqrt(max(0, arg_idx - 1)) + 1
            seen.add(ev.var)
    return cost


def _polymatroid_solve(
    events: Sequence[L2Event],
    input_arg_idx: Optional[Dict[int, int]],
    time_budget: Optional[float],
):
    """Run the LP sweep once and return the data both
    `polymatroid_lower_bound` and `polymatroid_floor_curve` need:
    `(arg_cost, intervals, M, depth)` where M[c²] is the LP optimum
    at capacity c² and depth[i] is the smallest c² at which interval
    i was selected (or last_cap+1 if it never was).  Returns `None`
    if the time budget is exceeded mid-sweep.
    """
    import time as _time

    input_arg_idx = input_arg_idx or {}
    deadline = (_time.perf_counter() + time_budget
                if time_budget is not None else None)

    arg_cost = _arg_stack_first_load_cost(events, input_arg_idx)
    intervals = _extract_intervals_two_stack(events, input_arg_idx)
    if not intervals:
        return arg_cost, [], {}, []

    cliques = _extract_cliques(events, intervals)
    omega = max((len(c) for c in cliques), default=0)
    if omega == 0:
        return arg_cost, intervals, {}, [1] * len(intervals)

    N = len(intervals)
    var_to_idx = {iv.var_id: i for i, iv in enumerate(intervals)}
    c_obj = np.array([-iv.reads for iv in intervals], dtype=float)

    A = lil_matrix((len(cliques), N))
    for i, clique in enumerate(cliques):
        for v in clique:
            j = var_to_idx.get(v)
            if j is not None:
                A[i, j] = 1
    A = A.tocsr()
    bounds = [(0.0, 1.0)] * N

    # Only square capacities cause the ceil-sqrt step to advance.
    max_c = math.isqrt(omega - 1) if omega > 1 else 0
    capacities = [c * c for c in range(1, max_c + 1)]

    M: Dict[int, int] = {}
    last_cap = capacities[-1] if capacities else 1
    depth: List[int] = [last_cap + 1] * N
    for cap in capacities:
        if deadline is not None and _time.perf_counter() > deadline:
            return None
        b = np.full(len(cliques), float(cap))
        res = linprog(
            c_obj, A_ub=A, b_ub=b, bounds=bounds, method="highs",
        )
        if not res.success:
            raise RuntimeError(f"LP failed at capacity={cap}: {res.message}")
        M[cap] = int(round(-res.fun))
        x = res.x
        for i in range(N):
            if depth[i] > cap and x[i] > 0.5:
                depth[i] = cap
    return arg_cost, intervals, M, depth


def polymatroid_lower_bound(
    events: Sequence[L2Event],
    input_arg_idx: Optional[Dict[int, int]] = None,
    time_budget: Optional[float] = None,
) -> Optional[int]:
    """Discrete polymatroid LP lower bound (see module docstring).

    Two-Stack semantics (matching global_density / local_density):
    inputs sit on the free arg stack until first load. The compulsory
    `⌈√(arg_idx)⌉` first-read cost is added on top of the polymatroid LP
    bound, and inputs enter the polymatroid LP with `reads = k − 1`
    (one read removed; charged via arg stack).

    If `time_budget` (seconds) is given, the function aborts before
    starting any LP solve once the cumulative wall time exceeds the
    budget, returning `None`.  Useful for the grid driver, which needs
    a hard cap per cell.
    """
    out = _polymatroid_solve(events, input_arg_idx, time_budget)
    if out is None:
        return None
    arg_cost, intervals, M, _depth = out
    if not intervals:
        return arg_cost
    R_total = sum(iv.reads for iv in intervals)
    if not M:
        return int(arg_cost + R_total)
    lb = R_total
    for cap in M:
        lb += R_total - M[cap]
    return int(lb) + arg_cost


def polymatroid_floor_curve(
    events: Sequence[L2Event],
    input_arg_idx: Optional[Dict[int, int]] = None,
    time_budget: Optional[float] = None,
) -> Optional[tuple]:
    """Per-tick polymatroid floor curve, mirroring `global_density_floor_curve`
    / `local_density_floor_curve` for the discrete polymatroid LP.

    For each square capacity c² (c=1..⌊√(ω-1)⌋) we solve the same TU LP
    used by `polymatroid_lower_bound` and read off the LP-implied
    placement: variable v's depth d_v = smallest c² where v gets
    selected (x_v(c²) = 1), or ω+1 if it never fits.  Each load of v is
    then charged ⌈√(d_v)⌉, distributed evenly across v's lifespan as a
    per-tick density ρ̃_v = reads(v) · ⌈√(d_v)⌉ / lifespan(v).

    Returns (times, floors) suitable for a `drawstyle="steps-post"` plot
    where floors[k] is held over [times[k], times[k+1]).  The integral
    of the curve equals the geometric portion of the polymatroid LP
    bound (the compulsory arg-stack first-load cost is reported
    separately by `polymatroid_lower_bound`).

    Returns `None` if the LP sweep exceeds `time_budget` seconds.
    """
    out = _polymatroid_solve(events, input_arg_idx, time_budget)
    if out is None:
        return None
    _arg_cost, intervals, _M, depth = out
    if not intervals:
        return [], []
    return _curve_from_depth(intervals, depth)


def _curve_from_depth(intervals, depth):
    """Build the per-tick polymatroid floor curve from already-solved
    LP outputs.  Pure post-processing — no LP solve."""
    # Per-interval polymatroid density:
    #   ρ̃_v = reads(v) · ⌈√d_v⌉ / lifespan(v)
    # Floor(t) = Σ_{v live at t} ρ̃_v.  Sweep deaths before births at
    # equal times (births=1, deaths=0 as the secondary sort key) so
    # peak-overlap is captured one tick early — matching the convention
    # of `global_density_floor_curve`.
    DEATH, BIRTH = 0, 1
    sweep: List[tuple] = []
    for i, iv in enumerate(intervals):
        c_v = math.isqrt(max(0, depth[i] - 1)) + 1
        rho = iv.reads * c_v / max(1, iv.end - iv.start)
        sweep.append((iv.start, BIRTH, rho))
        sweep.append((iv.end,   DEATH, rho))
    sweep.sort(key=lambda e: (e[0], e[1]))

    times: List[int] = []
    floors: List[float] = []
    cur = 0.0
    last_t: Optional[int] = None
    for t, kind, rho in sweep:
        if last_t is not None and t != last_t:
            if not floors or floors[-1] != cur:
                times.append(last_t)
                floors.append(cur)
        cur += rho if kind == BIRTH else -rho
        last_t = t
    if last_t is not None and (not times or times[-1] != last_t):
        times.append(last_t)
        floors.append(0.0)  # Drops to zero after the last death.
    return times, floors


__all__ = [
    "polymatroid_lower_bound",
    "polymatroid_floor_curve",
    "_polymatroid_solve",
    "_curve_from_depth",
]
