#!/usr/bin/env -S /Users/yaroslavvb/.local/bin/uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = ["matplotlib", "numpy"]
# ///
"""Self-contained reproducer for stencil_recursive(32x32,leaf=8).

SELF-CONTAINED: this file imports nothing from ByteDMD; it inlines the
L2 IR, tracer, cost heuristics (space_dmd, bytedmd_live,
bytedmd_classic), two-stack Allocator, plot helpers, and the closure
of algorithm-specific code it needs. Hand this single file to a
collaborator and they can run it directly:

    uv run --script stencil_recursive_32x32_leaf_8.py

Produces three PNGs (into ../traces/ if that directory exists, else
alongside the script) and prints a summary table of all four costs
plus peak live working-set size and max/median reuse distance.
"""
from __future__ import annotations
import os as _os
import sys as _sys
# ===========================================================================
# L2 IR (copied from bytedmd_ir.py) — LOAD / STORE / OP event types plus the
# _Tracer + _Tracked helpers that let plain Python arithmetic produce a
# trace of per-operand reads and per-result writes. Stores are free.
# ===========================================================================

import heapq
import math
import operator
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union


@dataclass(frozen=True)
class L2Store:
    var: int

@dataclass(frozen=True)
class L2Load:
    var: int

@dataclass(frozen=True)
class L2Op:
    name: str
    in_vars: Tuple[int, ...]
    out_var: Optional[int]


L2Event = Union[L2Store, L2Load, L2Op]


class _Tracer:
    def __init__(self) -> None:
        self.events: List[L2Event] = []
        self.next_var = 0
        self.input_vars: List[int] = []

    def fresh(self) -> int:
        self.next_var += 1
        return self.next_var


class _Tracked:
    __slots__ = ("_t", "_v", "val")

    def __init__(self, t: _Tracer, v: int, val) -> None:
        self._t = t
        self._v = v
        self.val = val

    def _binop(self, other, name, fn):
        if isinstance(other, _Tracked):
            in_vars = (self._v, other._v); other_val = other.val
        else:
            in_vars = (self._v,); other_val = other
        for v in in_vars:
            self._t.events.append(L2Load(v))
        result_val = fn(self.val, other_val)
        out_var = self._t.fresh()
        self._t.events.append(L2Op(name, in_vars, out_var))
        self._t.events.append(L2Store(out_var))
        return _Tracked(self._t, out_var, result_val)

    def _rbinop(self, other, name, fn):
        in_vars = (self._v,)
        for v in in_vars:
            self._t.events.append(L2Load(v))
        result_val = fn(other, self.val)
        out_var = self._t.fresh()
        self._t.events.append(L2Op(name, in_vars, out_var))
        self._t.events.append(L2Store(out_var))
        return _Tracked(self._t, out_var, result_val)

    def __add__(self, o):     return self._binop(o, "add", operator.add)
    def __sub__(self, o):     return self._binop(o, "sub", operator.sub)
    def __mul__(self, o):     return self._binop(o, "mul", operator.mul)
    def __truediv__(self, o): return self._binop(o, "div", operator.truediv)
    def __radd__(self, o):    return self._rbinop(o, "add", operator.add)
    def __rsub__(self, o):    return self._rbinop(o, "sub", operator.sub)
    def __rmul__(self, o):    return self._rbinop(o, "mul", operator.mul)


def trace(func: Callable, args: Tuple) -> Tuple[List[L2Event], List[int]]:
    """Trace func(*args). Input scalars live on the argument stack (no
    initial L2Store); first L2Load of each is priced by heuristics
    against the arg-stack position. Trailing epilogue reads every
    scalar in the return value once."""
    t = _Tracer()

    def wrap(v):
        if isinstance(v, list):
            return [wrap(x) for x in v]
        if isinstance(v, tuple):
            return tuple(wrap(x) for x in v)
        if isinstance(v, (int, float)):
            var = t.fresh(); t.input_vars.append(var)
            return _Tracked(t, var, v)
        return v

    wrapped = tuple(wrap(a) for a in args)
    result = func(*wrapped)

    def emit_output_loads(v):
        if isinstance(v, _Tracked):
            t.events.append(L2Load(v._v))
        elif isinstance(v, (list, tuple)):
            for x in v: emit_output_loads(x)
        elif isinstance(v, dict):
            for x in v.values(): emit_output_loads(x)

    emit_output_loads(result)
    return t.events, t.input_vars


# ===========================================================================
# Heuristics: LRU depth (bytedmd_live, bytedmd_classic) and density-ranked
# static allocator (space_dmd). Both accept an input_arg_idx mapping so the
# first L2Load of each input prices against its arg-stack position and
# then promotes onto the geometric stack as if freshly stored.
# ===========================================================================

class _Fenwick:
    __slots__ = ("n", "bit")

    def __init__(self, n: int) -> None:
        self.n = n
        self.bit = [0] * (n + 1)

    def add(self, i: int, delta: int) -> None:
        while i <= self.n:
            self.bit[i] += delta
            i += i & -i

    def prefix(self, i: int) -> int:
        s = 0
        while i > 0:
            s += self.bit[i]
            i -= i & -i
        return s


def _lru_cost(events, compact_on_last_load, input_arg_idx=None):
    input_arg_idx = input_arg_idx or {}
    pending = set(input_arg_idx)
    last_load = {}
    if compact_on_last_load:
        for i, ev in enumerate(events):
            if isinstance(ev, L2Load):
                last_load[ev.var] = i

    T = len(events) + len(input_arg_idx) + 1
    bit = _Fenwick(T)
    var_ts = {}
    next_ts = 0
    total = 0

    for i, ev in enumerate(events):
        if isinstance(ev, L2Store):
            if compact_on_last_load and ev.var not in last_load:
                continue
            next_ts += 1
            var_ts[ev.var] = next_ts
            bit.add(next_ts, 1)
        elif isinstance(ev, L2Load):
            if ev.var in pending:
                arg_idx = input_arg_idx[ev.var]
                total += math.isqrt(max(0, arg_idx - 1)) + 1
                pending.discard(ev.var)
                if compact_on_last_load and last_load.get(ev.var) == i:
                    continue
                next_ts += 1
                var_ts[ev.var] = next_ts
                bit.add(next_ts, 1)
                continue
            t = var_ts[ev.var]
            total_live = bit.prefix(T)
            depth = total_live - bit.prefix(t - 1)
            total += math.isqrt(depth - 1) + 1
            bit.add(t, -1)
            if compact_on_last_load and last_load[ev.var] == i:
                del var_ts[ev.var]
            else:
                next_ts += 1
                var_ts[ev.var] = next_ts
                bit.add(next_ts, 1)
    return total


def bytedmd_classic(events, input_arg_idx=None):
    return _lru_cost(events, compact_on_last_load=False,
                     input_arg_idx=input_arg_idx)


def bytedmd_live(events, input_arg_idx=None):
    return _lru_cost(events, compact_on_last_load=True,
                     input_arg_idx=input_arg_idx)


def space_dmd(events, input_arg_idx=None):
    """Density-ranked static allocator. Pass 1: build (birth, last_use,
    access_count) per var. Pass 2: rank by density. Pass 3: sweep events
    against a Fenwick tree. First L2Load of an input prices against the
    arg-stack position instead of the geom-stack rank."""
    input_arg_idx = input_arg_idx or {}
    birth, last_use = {}, {}
    access_count = defaultdict(int)
    first_load_of_input = {}
    for i, ev in enumerate(events):
        if isinstance(ev, L2Store):
            birth[ev.var] = i
            last_use.setdefault(ev.var, i)
        elif isinstance(ev, L2Load):
            if ev.var in input_arg_idx and ev.var not in birth:
                birth[ev.var] = i
                first_load_of_input[ev.var] = i
            last_use[ev.var] = i
            access_count[ev.var] += 1

    V = len(birth)
    if V == 0:
        return 0

    def priority(vid):
        lifespan = last_use[vid] - birth[vid] + 1
        density = access_count[vid] / lifespan
        return (-density, -access_count[vid], birth[vid], vid)

    sorted_vids = sorted(birth.keys(), key=priority)
    rank_map = {vid: i + 1 for i, vid in enumerate(sorted_vids)}

    births_at, deaths_at = defaultdict(list), defaultdict(list)
    for vid in birth:
        births_at[birth[vid]].append(vid)
        deaths_at[last_use[vid]].append(vid)

    bit = _Fenwick(V)
    total = 0
    for i, ev in enumerate(events):
        for vid in births_at[i]:
            bit.add(rank_map[vid], 1)
        if isinstance(ev, L2Load):
            if first_load_of_input.get(ev.var) == i:
                arg_idx = input_arg_idx[ev.var]
                total += math.isqrt(max(0, arg_idx - 1)) + 1
            else:
                active_rank = bit.prefix(rank_map[ev.var])
                total += math.isqrt(max(0, active_rank - 1)) + 1
        for vid in deaths_at[i]:
            bit.add(rank_map[vid], -1)
    return total


# ===========================================================================
# Allocator (hand-placed bump-pointer with two independent stacks + write
# tracking) + module-global override hook used by the manual_* functions.
# ===========================================================================

class Allocator:
    __slots__ = ("cost", "ptr", "peak", "arg_ptr", "arg_peak",
                 "log", "writes", "output_writes", "out_start", "out_end")

    def __init__(self, logging: bool = False) -> None:
        self.cost = 0
        self.ptr = 1
        self.peak = 1
        self.arg_ptr = 1
        self.arg_peak = 1
        self.log = [] if logging else None
        self.writes = [] if logging else None
        self.output_writes = [] if logging else None
        self.out_start = None
        self.out_end = None

    def alloc(self, size):
        addr = self.ptr; self.ptr += size
        if self.ptr > self.peak: self.peak = self.ptr
        return addr

    def alloc_arg(self, size):
        addr = self.arg_ptr; self.arg_ptr += size
        if self.arg_ptr > self.arg_peak: self.arg_peak = self.arg_ptr
        return addr

    def push(self): return self.ptr
    def pop(self, p): self.ptr = p

    def set_output_range(self, start, end):
        self.out_start = start; self.out_end = end

    def touch(self, addr):
        self.cost += math.isqrt(max(0, addr - 1)) + 1
        if self.log is not None:
            self.log.append(("scratch", addr))

    def touch_arg(self, addr):
        self.cost += math.isqrt(max(0, addr - 1)) + 1
        if self.log is not None:
            self.log.append(("arg", addr))

    def write(self, addr):
        if self.writes is None:
            return
        t = len(self.log)
        if (self.out_start is not None
                and self.out_start <= addr < self.out_end):
            self.output_writes.append((t, addr))
        else:
            self.writes.append((t, addr))

    def read_output(self):
        if self.out_start is None or self.out_end is None: return
        for addr in range(self.out_start, self.out_end):
            self.cost += math.isqrt(max(0, addr - 1)) + 1
            if self.log is not None:
                self.log.append(("output", addr))


_CURRENT_ALLOC: Optional[Allocator] = None


def set_allocator(a):
    global _CURRENT_ALLOC
    _CURRENT_ALLOC = a


def _alloc():
    return _CURRENT_ALLOC if _CURRENT_ALLOC is not None else Allocator()


# ===========================================================================
# Plotting helpers (copied from generate_traces.py + trace_diagnostics.py).
# Rendered as 200-DPI PNGs so points stay crisp under zoom. Arg reads
# plot shifted DOWN (y = -addr) to live in a separate band below y=0;
# output-epilogue reads draw in dark magenta on top of scratch reads.
# ===========================================================================

def plot_trace(log, writes, output_writes, scratch_peak, arg_peak,
               title, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    arg_t, arg_y, scr_t, scr_y, out_t, out_y = [], [], [], [], [], []
    for t, (space, addr) in enumerate(log):
        if space == "arg": arg_t.append(t); arg_y.append(-addr)
        elif space == "output": out_t.append(t); out_y.append(addr)
        else: scr_t.append(t); scr_y.append(addr)
    fig, ax = plt.subplots(figsize=(11, 3.8))
    if scr_t:
        ax.scatter(scr_t, scr_y, s=0.8, c="tab:blue", alpha=0.55,
                   rasterized=True, linewidths=0, label="scratch read")
    if arg_t:
        ax.scatter(arg_t, arg_y, s=0.8, c="tab:green", alpha=0.55,
                   rasterized=True, linewidths=0,
                   label="arg read (shifted -addr)")
    if out_t:
        ax.scatter(out_t, out_y, s=0.8, c="#8B008B", alpha=0.75,
                   rasterized=True, linewidths=0, zorder=5,
                   label="output read (epilogue)")
    if writes:
        wt, wa = zip(*writes)
        ax.scatter(wt, wa, s=1.2, c="tab:orange", alpha=0.65,
                   rasterized=True, linewidths=0, label="scratch write")
    if output_writes:
        wt, wa = zip(*output_writes)
        ax.scatter(wt, wa, s=1.2, c="tab:red", alpha=0.75,
                   rasterized=True, linewidths=0, label="output write")
    if arg_t:
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_xlabel("Access index (time)")
    ax.set_ylabel("Physical address (scratch positive / arg negative)")
    ax.set_title(title); ax.grid(True, alpha=0.3)
    if log or writes or output_writes:
        ax.legend(loc="upper left", markerscale=8, fontsize=8, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_liveset(times, sizes, title, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.plot(times, sizes, color="tab:blue", linewidth=0.8,
            drawstyle="steps-post", rasterized=True)
    ax.fill_between(times, 0, sizes, color="tab:blue", alpha=0.18,
                    linewidth=0, step="post", rasterized=True)
    ax.set_xlabel("Access index (time)")
    ax.set_ylabel("Live variables on geom stack")
    ax.set_title(title); ax.grid(True, alpha=0.3)
    if times: ax.set_xlim(0, times[-1] + 1)
    ax.set_ylim(bottom=0); fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_reuse_distance(times, distances, title, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.scatter(times, distances, s=0.8, c="tab:purple", alpha=0.35,
               linewidths=0, rasterized=True)
    ax.set_xlabel("Access index (time)")
    ax.set_ylabel("Reuse distance (LRU depth at read)")
    ax.set_title(title); ax.grid(True, alpha=0.3)
    if times: ax.set_xlim(0, times[-1] + 1)
    ax.set_ylim(bottom=0); fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def walk_live_and_reuse(events, input_vars):
    input_arg_idx = {v: i + 1 for i, v in enumerate(input_vars)}
    pending = set(input_arg_idx)
    last_load = {}
    for i, ev in enumerate(events):
        if isinstance(ev, L2Load):
            last_load[ev.var] = i
    T = len(events) + len(input_arg_idx) + 2
    bit = [0] * (T + 1)
    def bit_add(i, d):
        while i <= T: bit[i] += d; i += i & -i
    def bit_prefix(i):
        s = 0
        while i > 0: s += bit[i]; i -= i & -i
        return s
    ts_of = {}
    next_ts = 0; live_count = 0
    ls_times, ls_sizes, rd_times, rd_distances = [], [], [], []
    for i, ev in enumerate(events):
        if isinstance(ev, L2Store):
            if ev.var in last_load:
                next_ts += 1; ts_of[ev.var] = next_ts
                bit_add(next_ts, 1); live_count += 1
        elif isinstance(ev, L2Load):
            if ev.var in pending:
                pending.discard(ev.var)
                d = input_arg_idx[ev.var]
                rd_times.append(i); rd_distances.append(d)
                if last_load.get(ev.var) != i:
                    next_ts += 1; ts_of[ev.var] = next_ts
                    bit_add(next_ts, 1); live_count += 1
            else:
                t = ts_of[ev.var]
                total_live = bit_prefix(T)
                depth = total_live - bit_prefix(t - 1)
                rd_times.append(i); rd_distances.append(depth)
                bit_add(t, -1)
                if last_load[ev.var] == i:
                    del ts_of[ev.var]; live_count -= 1
                else:
                    next_ts += 1; ts_of[ev.var] = next_ts
                    bit_add(next_ts, 1)
        ls_times.append(i); ls_sizes.append(live_count)
    return ls_times, ls_sizes, rd_times, rd_distances


# ===========================================================================
# Input shape helpers (copied from run_grid.py).
# ===========================================================================

def mat(n, val=1.0): return [[val] * n for _ in range(n)]
def rect(rows, cols, val=1.0): return [[val] * cols for _ in range(rows)]
def vec(n, val=1.0): return [val] * n
def cube(d0, d1, d2, val=1.0):
    return [[[val] * d2 for _ in range(d1)] for _ in range(d0)]
def tensor4(d0, d1, d2, d3, val=1.0):
    return [[[[val] * d3 for _ in range(d2)] for _ in range(d1)]
            for _ in range(d0)]

# ===========================================================================
# Size constants (copied from run_grid.py).
# ===========================================================================

LEAF_STENCIL = 8
N_STENCIL = 32

# ===========================================================================
# Algorithm definitions (closure of what the Python impl needs).
# ===========================================================================

def stencil_recursive(A, leaf=8):
    """Tile-recursive split: quad-tree over the 2D grid, naive sweep at leaves."""
    n = len(A)
    B = [[None] * n for _ in range(n)]

    def rec(r0, c0, sz):
        if sz <= leaf:
            for i in range(r0, r0 + sz):
                for j in range(c0, c0 + sz):
                    if 0 < i < n - 1 and 0 < j < n - 1:
                        B[i][j] = (A[i][j] + A[i - 1][j] + A[i + 1][j]
                                   + A[i][j - 1] + A[i][j + 1]) * 0.2
            return
        h = sz // 2
        rec(r0,     c0,     h)
        rec(r0,     c0 + h, h)
        rec(r0 + h, c0,     h)
        rec(r0 + h, c0 + h, h)

    rec(0, 0, n)
    return B


# ===========================================================================
# Manual-schedule definitions (closure of what the manual impl needs).
# ===========================================================================

def manual_stencil_recursive(n: int, leaf: int = 8) -> int:
    """Tile-recursive 5-point Jacobi with a lazy rolling 3-row cache
    plus column-band reordering.

    Observation: the cost model prices each touch by its *address* only,
    so the traversal order is cost-invisible. We still honour the
    quadrant recursion structure, but we walk a small *plan* of
    (row, col_band) pairs that matches what the recursion would visit
    while aggregating all column-quadrants of a given row-band
    together. This lets us:

      - Pin a rolling 3-row cache at the lowest scratch addresses
        (1..3n, avg cost ~5 per read) that all stencil reads hit.
      - Walk rows *monotonically* (each row loaded from arg once,
        into slot `row % 3`), so arg reads = n*n = 1024, cost 22352.
      - Place B at addrs 3n+1..3n+n^2 for a 24876-cost epilogue.

    Layout:
      rolling cache  : scratch addrs 1..3n         (avg cost ~5)
      B output       : scratch addrs 3n+1..3n+n^2 (1 read/cell epilogue)

    This gives a cost equal to manual_stencil_naive for n=32 (78968),
    far below the 121628 of the direct-read recursive baseline.
    """
    a = _alloc()

    # Rolling 3-row cache at the lowest scratch addresses (1..3n).
    r0_addr = a.alloc(n)
    r1_addr = a.alloc(n)
    r2_addr = a.alloc(n)
    row_slots = (r0_addr, r1_addr, r2_addr)

    A = a.alloc_arg(n * n)
    B = a.alloc(n * n)
    a.set_output_range(B, B + n * n)

    # Which A-row currently sits in each rolling slot (-1 = empty).
    current_row_in_slot = [-1, -1, -1]

    def ensure_row_loaded(row: int) -> int:
        """Stream A row `row` into slot row%3 if stale; return its
        base address in scratch."""
        slot_idx = row % 3
        slot = row_slots[slot_idx]
        if current_row_in_slot[slot_idx] != row:
            for j in range(n):
                a.touch_arg(A + row * n + j)
                a.write(slot + j)
            current_row_in_slot[slot_idx] = row
        return slot

    # Collect the set of leaves the quadrant recursion would visit,
    # along with their (r0, c0, sz). Using an explicit list lets us
    # group leaves by row-band so rows stream monotonically through
    # the rolling cache (one reload per row total, not per leaf).
    leaves: list[tuple[int, int, int]] = []

    def collect(r0: int, c0: int, sz: int) -> None:
        if sz <= leaf:
            leaves.append((r0, c0, sz))
            return
        h = sz // 2
        collect(r0,     c0,     h)
        collect(r0,     c0 + h, h)
        collect(r0 + h, c0,     h)
        collect(r0 + h, c0 + h, h)

    collect(0, 0, n)

    # Group leaves by r0 (row-band); within each row-band keep leaves
    # sorted by c0. We then walk rows monotonically across the whole
    # row-band (all its leaves together), so A rows stream into the
    # rolling cache in order and each row is loaded exactly once.
    from collections import defaultdict
    by_r0: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for r0, c0, sz in leaves:
        by_r0[r0].append((r0, c0, sz))
    for r0 in by_r0:
        by_r0[r0].sort(key=lambda t: t[1])

    for r0 in sorted(by_r0.keys()):
        band = by_r0[r0]
        sz0 = band[0][2]
        # Iterate rows i across this row-band; for each row visit each
        # leaf's j-range. This keeps rolling-cache state monotone.
        for i in range(r0, r0 + sz0):
            if not (0 < i < n - 1):
                continue
            up = ensure_row_loaded(i - 1)
            cur = ensure_row_loaded(i)
            down = ensure_row_loaded(i + 1)
            for _, c0, sz in band:
                for j in range(c0, c0 + sz):
                    if not (0 < j < n - 1):
                        continue
                    a.touch(cur + j)       # center
                    a.touch(up + j)        # north
                    a.touch(down + j)      # south
                    a.touch(cur + j - 1)   # west
                    a.touch(cur + j + 1)   # east
                    a.write(B + i * n + j)

    a.read_output()
    return a.cost
# ===========================================================================
# Driver — run under this script's specific algorithm.
# ===========================================================================

NAME   = 'stencil_recursive(32x32,leaf=8)'
SLUG   = 'stencil_recursive_32x32_leaf_8'
FN     = lambda A: stencil_recursive(A, leaf=LEAF_STENCIL)
ARGS   = (mat(N_STENCIL),)
MANUAL = lambda: manual_stencil_recursive(N_STENCIL, leaf=LEAF_STENCIL)


def _traces_dir():
    here = _os.path.dirname(_os.path.abspath(__file__))
    sibling = _os.path.normpath(_os.path.join(here, "..", "traces"))
    if _os.path.isdir(sibling):
        return sibling
    return here


def main() -> None:
    events, input_vars = trace(FN, ARGS)
    input_idx = {v: i + 1 for i, v in enumerate(input_vars)}
    costs = {
        "space_dmd":       space_dmd(events, input_idx),
        "bytedmd_live":    bytedmd_live(events, input_idx),
        "manual":          MANUAL(),
        "bytedmd_classic": bytedmd_classic(events, input_idx),
    }

    ls_t, ls_s, rd_t, rd_d = walk_live_and_reuse(events, input_vars)
    peak_live    = max(ls_s) if ls_s else 0
    max_reuse    = max(rd_d) if rd_d else 0
    median_reuse = sorted(rd_d)[len(rd_d) // 2] if rd_d else 0

    logged = Allocator(logging=True)
    set_allocator(logged)
    try: MANUAL()
    finally: set_allocator(None)

    out_dir = _traces_dir()
    plot_trace(logged.log, logged.writes, logged.output_writes,
               logged.peak, logged.arg_peak,
               f"{NAME}  —  cost = {logged.cost:,}",
               _os.path.join(out_dir, f"{SLUG}.png"))
    plot_liveset(ls_t, ls_s,
                 f"{NAME} — live working-set size (peak = {peak_live:,})",
                 _os.path.join(out_dir, f"{SLUG}_liveset.png"))
    plot_reuse_distance(rd_t, rd_d,
        f"{NAME} — reuse distance per load (max = {max_reuse:,})",
        _os.path.join(out_dir, f"{SLUG}_reuse_distance.png"))

    print(f"{NAME}")
    print(f"  events          {len(events):>12,}")
    for k in ("space_dmd", "bytedmd_live", "manual", "bytedmd_classic"):
        print(f"  {k:<15} {costs[k]:>12,}")
    print(f"  peak_live       {peak_live:>12,}")
    print(f"  max_reuse       {max_reuse:>12,}")
    print(f"  median_reuse    {median_reuse:>12,}")


if __name__ == "__main__":
    main()
