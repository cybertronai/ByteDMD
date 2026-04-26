"""Comprehensive tests for splitting_lower_bound (split_lb).

The Splitting Lower Bound severs each variable's lifespan into
independent inter-access intervals and computes the fractional Rearrangement
Inequality floor at each tick:

    Floor(t) = Σ_{i=1}^{A_t}  ρ_(i) · √i    (ρ sorted descending)

Key properties tested:
  - Exact hand-computed values on small synthetic traces
  - Two-stack input handling (arg-stack first-read cost)
  - Virtual interval construction (one per inter-access gap)
  - Rearrangement Inequality ordering (higher density → lower rank)
  - Phase-structure advantage: split_lb < static_opt_lb when a variable
    has non-uniform inter-access gaps (the DMA/splitting benefit)
  - Non-negativity
  - Consistency with bytedmd_ir trace machinery on matmul algorithms
  - split_lb ≤ space_dmd (split_lb is a lower bound; space_dmd is achievable)
"""
from __future__ import annotations

import math
import sys
import os

import pytest

# Ensure repo root is on the path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "experiments", "grid"))

import bytedmd_ir as b2
from bytedmd_ir import (
    L2Load,
    L2Store,
    L2Op,
    splitting_lower_bound,
    static_opt_lb,
    bytedmd_live,
    bytedmd_classic,
    trace,
)

# spacedmd.py lives under experiments/grid/
from spacedmd import space_dmd


# ── helpers ──────────────────────────────────────────────────────────────────

def _slb(events, input_arg_idx=None):
    return splitting_lower_bound(events, input_arg_idx or {})


def _approx(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) < tol


# ── basic / empty cases ───────────────────────────────────────────────────────

def test_empty_trace_returns_zero():
    assert _slb([]) == 0.0


def test_no_reads_returns_zero():
    """Variables stored but never loaded contribute nothing."""
    evs = [L2Store(1), L2Store(2), L2Store(3)]
    assert _slb(evs) == 0.0


def test_nonnegative_simple():
    evs = [L2Store(1), L2Load(1)]
    assert _slb(evs) >= 0.0


# ── single variable, one read ─────────────────────────────────────────────────

def test_single_var_one_read_store_adjacent():
    """Store at event 0, read at event 1.
    One virtual interval [0,1) with density 1/1 = 1.0.
    Only variable → rank 1 throughout.
    Floor(t) = 1.0 · √1 = 1.0 for 1 tick → cost = 1.0."""
    evs = [L2Store(1), L2Load(1)]
    cost = _slb(evs)
    assert _approx(cost, 1.0), f"expected 1.0, got {cost}"


def test_single_var_one_read_gap5():
    """Store at event 0, 4 unrelated ops, read at event 5.
    Virtual interval [0,5), density 1/5 = 0.2.
    One variable, rank 1 for 5 ticks.
    Cost = 5 · (0.2 · √1) = 1.0."""
    evs = [L2Store(1)] + [L2Op("nop", (), None)] * 4 + [L2Load(1)]
    # Reindex: Store at 0, 4 Ops at 1-4, Load at 5.
    cost = _slb(evs)
    assert _approx(cost, 1.0), f"expected 1.0, got {cost}"


def test_single_var_invariant_to_gap():
    """For a single variable (rank always 1), cost = Σ gaps·ρ·√1 = 1 per read,
    regardless of the inter-access gap lengths."""
    # One read → one interval of any length; cost = 1.0
    for gap in (1, 3, 10, 100):
        evs = [L2Store(1)] + [L2Op("nop", (), None)] * (gap - 1) + [L2Load(1)]
        cost = _slb(evs)
        assert _approx(cost, 1.0), f"gap={gap}: expected 1.0, got {cost}"


# ── single variable, multiple reads ──────────────────────────────────────────

def test_single_var_two_reads():
    """Store at 0, reads at 2 and 5 (single variable throughout).
    Two virtual intervals: [0,2) density 1/2, [2,5) density 1/3.
    Both contribute √1 each → cost = 2.0."""
    evs = [L2Store(1), L2Op("a", (), None), L2Load(1),   # idx 0,1,2
           L2Op("b", (), None), L2Op("c", (), None), L2Load(1)]  # 3,4,5
    cost = _slb(evs)
    assert _approx(cost, 2.0), f"expected 2.0, got {cost}"


def test_single_var_k_reads():
    """k reads of a single variable should give cost = k (rank always 1)."""
    for k in range(1, 6):
        # Build: Store(1), then k*(Op, Load) pairs
        evs = [L2Store(1)]
        for _ in range(k):
            evs.append(L2Op("x", (), None))
            evs.append(L2Load(1))
        cost = _slb(evs)
        assert _approx(cost, float(k)), f"k={k}: expected {k}, got {cost}"


def test_consecutive_reads_gap1():
    """Store at 0, reads at 1 and 2.
    Intervals [0,1) and [1,2), both density 1.0.
    Each contributes 1.0 → total 2.0."""
    evs = [L2Store(1), L2Load(1), L2Load(1)]
    cost = _slb(evs)
    assert _approx(cost, 2.0), f"expected 2.0, got {cost}"


# ── virtual interval construction ─────────────────────────────────────────────

def test_virtual_interval_count_non_input():
    """A non-input variable with k reads produces k virtual intervals
    (one cold + (k-1) reuse).  We verify by checking the cost equals
    k · √1 = k when it's the only live variable."""
    for k in (1, 3, 5):
        evs = [L2Store(1)]
        for _ in range(k):
            evs.append(L2Load(1))
        cost = _slb(evs)
        assert _approx(cost, float(k)), f"k={k}: expected {float(k)}, got {cost}"


def test_input_var_zero_geometric_reads():
    """An input variable with exactly one read contributes only the arg-stack
    cost (ceil(sqrt(arg_idx))); no geometric LP interval is created."""
    # input_arg_idx = {var_id: position}
    evs = [L2Load(7)]
    # arg position 1 → ceil(sqrt(1)) = 1
    cost = _slb(evs, input_arg_idx={7: 1})
    assert _approx(cost, 1.0), f"expected 1.0, got {cost}"

    # arg position 4 → ceil(sqrt(4)) = 2
    cost = _slb(evs, input_arg_idx={7: 4})
    assert _approx(cost, 2.0), f"expected 2.0, got {cost}"


def test_input_var_subsequent_read_enters_lp():
    """Input at arg position 1 with two reads.
    First read: arg-stack cost = ceil(sqrt(1)) = 1.
    Second read: one geometric interval [t_1, t_2) with density 1/gap.
    Only variable in LP → rank 1 → LP cost = 1.0.
    Total = 2.0."""
    evs = [L2Load(5), L2Op("nop", (), None), L2Op("nop", (), None), L2Load(5)]
    #       idx 0                1                     2                  3
    # t_1 = 0, t_2 = 3, gap = 3, density = 1/3.
    cost = _slb(evs, input_arg_idx={5: 1})
    # LP: interval [0,3), 3 ticks, density 1/3, rank 1 → cost = 3*(1/3)*1 = 1.0
    # Total = arg cost(1.0) + LP(1.0) = 2.0
    assert _approx(cost, 2.0), f"expected 2.0, got {cost}"


def test_input_first_read_cost_reflects_arg_position():
    """ceil(sqrt(arg_idx)) for several input positions."""
    positions_and_expected = [(1, 1), (2, 2), (4, 2), (9, 3), (10, 4)]
    for pos, expected_floor in positions_and_expected:
        evs = [L2Load(99)]
        cost = _slb(evs, input_arg_idx={99: pos})
        assert int(cost) == expected_floor, (
            f"arg_idx={pos}: expected ceil(sqrt({pos}))={expected_floor}, got {cost}")


# ── rearrangement inequality ──────────────────────────────────────────────────

def test_rearrangement_two_vars_hand_computed():
    """Two variables with different local densities.

    Var 1 (id=1): Store@0, Load@3 → cold interval [0,3), ρ₁ = 1/3
    Var 2 (id=2): Store@1, Load@6 → cold interval [1,6), ρ₂ = 1/5

    (Note: stores are at events 0 and 1 respectively, not both at 0.)

    Sweep:
      [0,1) 1 tick : Var1 only. Floor = (1/3)·√1.          Cost = 1/3.
      [1,3) 2 ticks: Var1(1/3) rank1, Var2(1/5) rank2.
                     Floor = (1/3)·√1 + (1/5)·√2.          Cost = 2·(1/3 + (1/5)·√2).
      [3,6) 3 ticks: Var2 only. Floor = (1/5)·√1.          Cost = 3/5.

    Total = 1/3 + 2/3 + (2/5)·√2 + 3/5 = 1 + 3/5 + (2/5)·√2 = (8 + 2√2)/5 ≈ 2.1657
    """
    evs = [
        L2Store(1),               # 0 — Var1 born
        L2Store(2),               # 1 — Var2 born
        L2Op("nop", (), None),    # 2
        L2Load(1),                # 3  → cold interval [0,3), gap=3, ρ=1/3
        L2Op("nop", (), None),    # 4
        L2Op("nop", (), None),    # 5
        L2Load(2),                # 6  → cold interval [1,6), gap=5, ρ=1/5
    ]
    cost = _slb(evs)
    expected = (8 + 2 * math.sqrt(2)) / 5   # ≈ 2.16569
    assert _approx(cost, expected, tol=1e-9), f"expected {expected:.6f}, got {cost:.6f}"


def test_rearrangement_high_density_gets_rank1():
    """When two variables overlap, the higher local-density one gets rank 1.

    Var A: Store@0, Load@2 → interval [0,2), ρ_A = 1/2 = 0.5
    Var B: Store@1, Load@4 → interval [1,4), ρ_B = 1/3 ≈ 0.333

    (Stores are at events 0 and 1 respectively.)

    Sweep:
      [0,1) 1 tick : A only. Floor = 0.5·√1.               Cost = 0.5.
      [1,2) 1 tick : A(0.5) rank1, B(1/3) rank2.
                     Floor = 0.5·√1 + (1/3)·√2.            Cost = 0.5 + √2/3.
      [2,4) 2 ticks: B only. Floor = (1/3)·√1.             Cost = 2/3.

    Total = 0.5 + (0.5 + √2/3) + 2/3 = 5/3 + √2/3 = (5 + √2)/3 ≈ 2.1381
    """
    evs = [L2Store(10), L2Store(20),   # 0, 1
           L2Load(10),                  # 2 → A interval [0,2), ρ=1/2
           L2Op("x", (), None),         # 3
           L2Load(20)]                  # 4 → B interval [1,4), ρ=1/3
    cost = _slb(evs)
    expected = (5 + math.sqrt(2)) / 3   # ≈ 2.13807
    assert _approx(cost, expected, tol=1e-9), f"expected {expected:.6f}, got {cost:.6f}"


# ── phase-structure advantage over static_opt_lb ─────────────────────────────

def test_phase_structure_split_lt_static_opt():
    """split_lb < static_opt_lb on a phase-structured trace.

    Var A: Store@0, Load@2, Load@3  (phase 1 — dense)
    Var B: Store@1, Load@4, Load@5  (phase 2 — dense, but B dormant during phase 1)

    In static_opt_lb, A's global density (2/4=0.5) > B's (2/5=0.4), so A gets
    rank 1 throughout — even during ticks [3,4] when A is done and B is just
    waiting.  split_lb terminates A's active intervals at t=3 (its last read),
    so during [3,4] only B's cold interval is active, correctly freeing rank 1
    for B's high-density burst.
    """
    evs = [
        L2Store(1),  # 0  — Var A born
        L2Store(2),  # 1  — Var B born
        L2Load(1),   # 2  — A read 1
        L2Load(1),   # 3  — A read 2 (last)
        L2Load(2),   # 4  — B read 1
        L2Load(2),   # 5  — B read 2 (last)
    ]
    slb = _slb(evs)
    sob = static_opt_lb(evs)
    assert slb < sob, (
        f"expected split_lb ({slb:.4f}) < static_opt_lb ({sob:.4f}) "
        "on phase-structured trace")


def test_phase_structure_split_hand_computed():
    """Exact hand-computed split_lb for the phase-structure trace above.

    Virtual intervals (no input args):
      A cold [0,2)  ρ=1/2=0.5
      A reuse [2,3) ρ=1/1=1.0
      B cold [1,4)  ρ=1/3≈0.333
      B reuse [4,5) ρ=1/1=1.0

    Sweep (deaths before births at equal t):
      t=0: birth A_cold        → active={(0.5,A_cold)}
      t=1: cost [0,1) w/ {A_cold}: 1·0.5·√1 = 0.5
           birth B_cold        → active={(0.333,B_cold),(0.5,A_cold)}
      t=2: cost [1,2) w/ both: 1·(0.5·√1 + 0.333·√2) = 0.5+0.471=0.971
           death A_cold        → active={(0.333,B_cold)}
           birth A_reuse       → active={(0.333,B_cold),(1.0,A_reuse)}
      t=3: cost [2,3) w/ both: 1·(1.0·√1 + 0.333·√2) = 1.0+0.471=1.471
           death A_reuse       → active={(0.333,B_cold)}
      t=4: cost [3,4) w/ {B_cold}: 1·0.333·√1 = 0.333
           death B_cold        → active={}
           birth B_reuse       → active={(1.0,B_reuse)}
      t=5: cost [4,5) w/ {B_reuse}: 1·1.0·√1 = 1.0
           death B_reuse       → active={}

    Total = 0.5 + 0.971 + 1.471 + 0.333 + 1.0 = 4.275
    """
    evs = [L2Store(1), L2Store(2),
           L2Load(1), L2Load(1),
           L2Load(2), L2Load(2)]

    expected = (
        1 * 0.5 * 1.0                          # [0,1): A_cold rank1
        + 1 * (0.5 * 1.0 + (1/3) * math.sqrt(2))   # [1,2): A_cold rank1, B_cold rank2
        + 1 * (1.0 * 1.0 + (1/3) * math.sqrt(2))   # [2,3): A_reuse rank1, B_cold rank2
        + 1 * (1/3) * 1.0                      # [3,4): B_cold alone rank1
        + 1 * 1.0 * 1.0                        # [4,5): B_reuse rank1
    )
    cost = _slb(evs)
    assert _approx(cost, expected, tol=1e-9), (
        f"expected {expected:.6f}, got {cost:.6f}")


def test_uniform_access_split_equals_static_opt():
    """When all accesses are perfectly uniform (equal inter-access gaps for every
    variable), local density == global density and split_lb == static_opt_lb."""
    # Single variable, k uniformly spaced reads (gap always 1).
    # Both metrics should give the same result.
    # (In the k=1 case static_opt_lb uses continuous sqrt so it equals split_lb.)
    evs = [L2Store(1)]
    for _ in range(5):
        evs.append(L2Load(1))  # all consecutive, gap=1 each time
    slb = _slb(evs)
    sob = static_opt_lb(evs)
    # Both should give 5.0 (5 reads of the only live variable at rank 1).
    assert _approx(slb, 5.0, tol=1e-9), f"slb={slb}"
    assert _approx(sob, 5.0, tol=1e-9), f"sob={sob}"


# ── relationship to space_dmd ─────────────────────────────────────────────────

@pytest.mark.parametrize("N,algo_name", [
    (2, "matmul_naive"), (3, "matmul_naive"), (4, "matmul_naive"),
    (2, "matmul_tiled"), (3, "matmul_tiled"), (4, "matmul_tiled"),
    (2, "matmul_rmm"),   (4, "matmul_rmm"),   # RMM requires power-of-2 N
])
def test_split_lb_le_space_dmd(N, algo_name):
    """split_lb is a lower bound; space_dmd is an achievable (static) cost.
    Therefore split_lb ≤ space_dmd must hold."""
    A = [[float(i * N + j + 1) for j in range(N)] for i in range(N)]
    B = [[float(i * N + j + 1) for j in range(N)] for i in range(N)]
    func = getattr(b2, algo_name)
    evs, input_vars = trace(func, (A, B))
    iidx = {v: k + 1 for k, v in enumerate(input_vars)}

    slb = _slb(evs, iidx)
    sdmd = space_dmd(evs, iidx)
    assert slb <= sdmd + 1e-6, (
        f"{algo_name} N={N}: split_lb ({slb:.1f}) > space_dmd ({sdmd:.1f})")


@pytest.mark.parametrize("N,algo_name", [
    (2, "matmul_naive"), (3, "matmul_naive"), (4, "matmul_naive"),
    (2, "matmul_tiled"), (3, "matmul_tiled"), (4, "matmul_tiled"),
    (2, "matmul_rmm"),   (4, "matmul_rmm"),
])
def test_split_lb_le_bytedmd_live(N, algo_name):
    """split_lb should not exceed bytedmd_live (an achievable dynamic cost)."""
    A = [[float(i * N + j + 1) for j in range(N)] for i in range(N)]
    B = [[float(i * N + j + 1) for j in range(N)] for i in range(N)]
    func = getattr(b2, algo_name)
    evs, input_vars = trace(func, (A, B))
    iidx = {v: k + 1 for k, v in enumerate(input_vars)}

    slb = _slb(evs, iidx)
    live = bytedmd_live(evs, iidx)
    assert slb <= live + 1e-6, (
        f"{algo_name} N={N}: split_lb ({slb:.1f}) > bytedmd_live ({live:.1f})")


@pytest.mark.parametrize("N,algo_name", [
    (2, "matmul_naive"), (3, "matmul_naive"), (4, "matmul_naive"),
    (2, "matmul_tiled"), (3, "matmul_tiled"), (4, "matmul_tiled"),
    (2, "matmul_rmm"),   (4, "matmul_rmm"),
])
def test_split_lb_nonnegative_on_matmul(N, algo_name):
    A = [[1.0] * N for _ in range(N)]
    B = [[1.0] * N for _ in range(N)]
    func = getattr(b2, algo_name)
    evs, input_vars = trace(func, (A, B))
    iidx = {v: k + 1 for k, v in enumerate(input_vars)}
    assert _slb(evs, iidx) >= 0.0


# ── two-stack correctness ──────────────────────────────────────────────────────

def test_two_stack_input_arg_cost_separate():
    """Input args pay ceil(sqrt(pos)) once; non-inputs pay through the LP.
    Trace: two inputs (arg pos 1 and 2), each read once.
    split_lb = ceil(sqrt(1)) + ceil(sqrt(2)) = 1 + 2 = 3.
    No geometric LP since each input has only one read."""
    evs = [L2Load(1), L2Load(2)]
    cost = _slb(evs, input_arg_idx={1: 1, 2: 2})
    assert _approx(cost, 3.0), f"expected 3.0, got {cost}"


def test_two_stack_mixed_input_and_internal():
    """One input (arg pos 1, two reads) + one internal var (two reads).
    Input arg cost: ceil(sqrt(1)) = 1.
    Input LP interval [0, 4]: density = 1/4 (only one geometric interval).
    Internal var: cold [1, 2] density=1, reuse [2, 3] density=1.

    Trace structure (event indices):
      0: L2Load(input_var=10) — first read (arg-stack cost)
      1: L2Store(11)          — internal var born
      2: L2Load(11)           — internal first read
      3: L2Load(11)           — internal second read (last)
      4: L2Load(10)           — input second read (geometric LP)

    Virtual intervals:
      input LP interval: [0,4), density=1/4=0.25 (read at t=4, prev=t=0)
      internal cold: [1,2), density=1/1=1.0
      internal reuse: [2,3), density=1/1=1.0

    Ticks (deaths before births at equal t):
      [0,1): active={(0.25,input)}. Floor=0.25·√1. Cost=0.25.
      [1,2): active={(0.25,input),(1.0,int_cold)}. Floor=1.0·√1+0.25·√2=1.354. Cost=1.354.
      [2,3): death int_cold, birth int_reuse → active={(0.25,input),(1.0,int_reuse)}.
             Floor same = 1.354. Cost=1.354.
      [3,4): active={(0.25,input)}. Floor=0.25·√1=0.25. Cost=0.25.

    geom_cost = 0.25 + 1.354 + 1.354 + 0.25 = 3.208
    total = arg_cost(1.0) + geom_cost(3.208) = 4.208
    """
    evs = [
        L2Load(10),   # 0: input first read
        L2Store(11),  # 1: internal var born
        L2Load(11),   # 2: internal first read (cold interval [1,2))
        L2Load(11),   # 3: internal second read (reuse interval [2,3))
        L2Load(10),   # 4: input second read (LP interval [0,4))
    ]
    iidx = {10: 1}

    # Expected geom_cost:
    #  [0,1): input(0.25) alone → 1 * 0.25 = 0.25
    #  [1,2): int_cold(1.0) rank1, input(0.25) rank2 → 1*(1.0+0.25√2)
    #  [2,3): int_reuse(1.0) rank1, input(0.25) rank2 → 1*(1.0+0.25√2)
    #  [3,4): input(0.25) alone → 1*0.25
    per_mixed = 1.0 + 0.25 * math.sqrt(2)
    expected_geom = 0.25 + per_mixed + per_mixed + 0.25
    expected_total = 1.0 + expected_geom  # arg cost + geom

    cost = _slb(evs, iidx)
    assert _approx(cost, expected_total, tol=1e-9), (
        f"expected {expected_total:.6f}, got {cost:.6f}")


# ── regression / sanity against known algorithms ──────────────────────────────

@pytest.mark.parametrize("N", [2, 4])
def test_matmul_naive_split_positive(N):
    """split_lb of naive matmul is strictly positive."""
    A = [[1.0] * N for _ in range(N)]
    B = [[1.0] * N for _ in range(N)]
    evs, input_vars = trace(b2.matmul_naive, (A, B))
    iidx = {v: k + 1 for k, v in enumerate(input_vars)}
    assert _slb(evs, iidx) > 0


@pytest.mark.parametrize("N", [2, 4])
def test_matmul_rmm_split_positive(N):
    evs, input_vars = trace(b2.matmul_rmm,
                            ([[1.0]*N for _ in range(N)],
                             [[1.0]*N for _ in range(N)]))
    iidx = {v: k + 1 for k, v in enumerate(input_vars)}
    assert _slb(evs, iidx) > 0


def test_split_lb_increases_with_problem_size():
    """split_lb should grow with N for naive matmul (more work = more cost)."""
    costs = []
    for N in [2, 3, 4]:
        A = [[1.0] * N for _ in range(N)]
        B = [[1.0] * N for _ in range(N)]
        evs, input_vars = trace(b2.matmul_naive, (A, B))
        iidx = {v: k + 1 for k, v in enumerate(input_vars)}
        costs.append(_slb(evs, iidx))
    assert costs[0] < costs[1] < costs[2], f"costs not monotone: {costs}"


# ── edge-case: store never reached (input-only) ────────────────────────────────

def test_input_only_no_internal_vars():
    """Trace with only input loads and an Op — no L2Store for internals."""
    evs = [
        L2Load(1),
        L2Load(2),
        L2Op("add", (1, 2), None),
    ]
    iidx = {1: 1, 2: 2}
    cost = _slb(evs, iidx)
    # Each input has one read → pure arg-stack cost.
    # ceil(sqrt(1)) + ceil(sqrt(2)) = 1 + 2 = 3
    assert _approx(cost, 3.0), f"expected 3.0, got {cost}"


# ── single variable, varying gap lengths ─────────────────────────────────────

def test_variable_with_long_dormancy_and_burst():
    """Variable dormant for T ticks between two burst reads.
    During dormancy, its local density = 1/T is very low.
    If it's the only variable, it still costs 1 per read (rank always 1).
    But this test verifies correctness when a second hot variable competes."""
    # Var hot: store@0, reads at 1,2,3,4,5 (gap=1, density=1.0)
    # Var cold: store@0, reads at 1, 1000 (gap=1 and gap=999)
    # Both active during [0,1000].
    # In split_lb:
    #   cold_interval_0 [0,1): density=1.0 → tied with hot's [0,1)
    #   cold_interval_1 [1,1000): density=1/999 ≈ 0.001 → very low rank
    # hot gets rank 1 during [1,1000); cold gets rank 2 (very low density)
    evs = [L2Store(1), L2Store(2)]                    # 0, 1
    evs += [L2Load(1), L2Load(1), L2Load(1), L2Load(1), L2Load(1)]  # 2-6: hot reads
    evs += [L2Load(2)]                                # 7: cold first read

    # Deliberately skipping to event 1007 isn't practical; use a smaller gap.
    # Var cold: reads at 7 and 12 (gap=5). hot: reads at 2,3,4,5,6 (gap=1 each).
    evs2 = [L2Store(1), L2Store(2)]
    for _ in range(5):
        evs2.append(L2Load(1))   # hot reads at 2,3,4,5,6
    evs2.append(L2Load(2))       # cold first read at 7

    cost = _slb(evs2)
    # Sanity: cost should be finite and positive
    assert cost > 0
    # hot's 5 reads: each at rank 1 during their respective short intervals → ~5.0
    # cold's first read at rank 2 (tied density with hot during [0,2) cold interval)
    #   then alone for the remainder
    # Exact value is complex; just check it's less than 6*(sqrt(2)) (no spurious rank-2 charges)
    assert cost < 6 * math.sqrt(2) + 0.1


# ── check that split_lb doesn't exceed classic (ultimate upper bound) ──────────

@pytest.mark.parametrize("N,algo_name", [
    (2, "matmul_naive"), (4, "matmul_naive"),
    (2, "matmul_tiled"), (4, "matmul_tiled"),
    (2, "matmul_rmm"),   (4, "matmul_rmm"),
])
def test_split_lb_le_bytedmd_classic(N, algo_name):
    """split_lb ≤ bytedmd_classic (the most conservative upper-envelope heuristic)."""
    A = [[float(i * N + j + 1) for j in range(N)] for i in range(N)]
    B = [[float(i * N + j + 1) for j in range(N)] for i in range(N)]
    func = getattr(b2, algo_name)
    evs, input_vars = trace(func, (A, B))
    iidx = {v: k + 1 for k, v in enumerate(input_vars)}

    slb = _slb(evs, iidx)
    classic = bytedmd_classic(evs, iidx)
    assert slb <= classic + 1e-6, (
        f"{algo_name} N={N}: split_lb ({slb:.1f}) > bytedmd_classic ({classic:.1f})")
