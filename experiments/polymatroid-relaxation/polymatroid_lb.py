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
    stack). Matches the Two-Stack convention used by `static_opt_lb` and
    `splitting_lower_bound`.
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


def polymatroid_lower_bound(
    events: Sequence[L2Event],
    input_arg_idx: Optional[Dict[int, int]] = None,
) -> int:
    """Discrete polymatroid LP lower bound (see module docstring).

    Two-Stack semantics (matching static_opt_lb / splitting_lower_bound):
    inputs sit on the free arg stack until first load. The compulsory
    `⌈√(arg_idx)⌉` first-read cost is added on top of the polymatroid LP
    bound, and inputs enter the polymatroid LP with `reads = k − 1`
    (one read removed; charged via arg stack).
    """
    input_arg_idx = input_arg_idx or {}

    arg_cost = _arg_stack_first_load_cost(events, input_arg_idx)
    intervals = _extract_intervals_two_stack(events, input_arg_idx)
    if not intervals:
        return arg_cost
    cliques = _extract_cliques(events, intervals)
    omega = max((len(c) for c in cliques), default=0)
    if omega == 0:
        return arg_cost + sum(iv.reads for iv in intervals)

    R_total = sum(iv.reads for iv in intervals)
    N = len(intervals)
    var_to_idx = {iv.var_id: i for i, iv in enumerate(intervals)}
    c_obj = np.array([-iv.reads for iv in intervals], dtype=float)

    # Constraint rows = maximal cliques, columns = intervals.
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
    for cap in capacities:
        b = np.full(len(cliques), float(cap))
        res = linprog(
            c_obj,
            A_ub=A,
            b_ub=b,
            bounds=bounds,
            method="highs",
        )
        if not res.success:
            raise RuntimeError(f"LP failed at capacity={cap}: {res.message}")
        M[cap] = int(round(-res.fun))

    # Discrete-calculus identity.
    lb = R_total
    for cap in capacities:
        lb += R_total - M[cap]
    return int(lb) + arg_cost


__all__ = ["polymatroid_lower_bound"]
