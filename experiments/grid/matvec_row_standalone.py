#!/usr/bin/env python3
"""
Self-contained reproduction of the matvec_row(n=64) row from the grid table:

    | algorithm        | space_dmd | bytedmd_live | manual  | bytedmd_classic |
    | matvec_row(n=64) |    72,775 |      229,199 | 238,853 |         450,939 |

Four different cost models, all pricing memory reads by ceil(sqrt(depth_or_addr)):

  1. space_dmd      — "Optimal static compiler" (e.g. TPU scratchpad).
                      Assigns physical addresses once, up-front, based on access
                      density = access_count / lifespan. High-density variables
                      (inner-loop temporaries) get the lowest addresses. Cost of
                      a LOAD = ceil(sqrt(rank among currently-live variables)).
                      Uses a Fenwick tree for O(log V) rank queries.

  2. bytedmd_live   — "LRU cache with garbage collection". Tracks an LRU stack
                      of all live variables. When variable X is LOADed, its cost
                      is ceil(sqrt(depth)) where depth = #live vars above it in
                      the stack. After its LAST LOAD, X is removed ("compacted")
                      so dead vars don't inflate depth for others. Models an
                      ideal hardware LRU cache with perfect liveness info.

  3. manual         — "Hand-placed bump allocator". A physical 1-D address space
                      where alloc() returns the next address. Hot temporaries
                      (accumulator s, multiply temp tmp) at addresses 1-2,
                      output vector y at 3..n+2, input vector x at n+3..2n+2,
                      then the cold bulk matrix A at 2n+3..2n+2+n^2.
                      Reading address d costs ceil(sqrt(d)). Writes are free.

  4. bytedmd_classic — "LRU cache, no GC" (the Mattson stack-distance model).
                      Same LRU stack as bytedmd_live, but dead variables are
                      NEVER removed. They stay in the stack forever, pushing
                      live data deeper. This is the worst-case: the "infinite
                      graveyard" where every past allocation pollutes the stack.

The algorithm being measured:
  matvec_row(A, x) — row-major matrix-vector product y = A @ x for a 64×64
  matrix. The outer loop iterates over rows i of A; the inner loop computes
  the dot product y[i] = sum_j A[i][j] * x[j].

  Access pattern: A is read row-major (contiguous within each row), x is
  swept entirely for every row (N full passes over x), and each y[i] is
  written once. The key data movement challenge is that x (64 elements) is
  re-read 64 times, but under LRU it gets buried by the intervening row
  of A (64 elements) between consecutive passes.

Usage:
    python3 matvec_row_standalone.py
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union
import operator


# ============================================================================
# IR event types — the "bytecode" that all cost models operate on
# ============================================================================

@dataclass(frozen=True)
class L2Store:
    """A new value is written to variable `var`."""
    var: int

@dataclass(frozen=True)
class L2Load:
    """Variable `var` is read as an operand."""
    var: int

@dataclass(frozen=True)
class L2Op:
    """An arithmetic operation (add, mul, etc.) is performed."""
    name: str
    in_vars: Tuple[int, ...]
    out_var: Optional[int]


L2Event = Union[L2Store, L2Load, L2Op]


# ============================================================================
# Tracer — instruments plain Python arithmetic into L2 events
# ============================================================================

class _Tracer:
    """Global event recorder. Each _Tracked operation appends events here."""
    def __init__(self):
        self.events: List[L2Event] = []
        self.next_var = 0

    def fresh(self) -> int:
        self.next_var += 1
        return self.next_var


class _Tracked:
    """Wraps a Python float/int so that every +, * records L2 events.

    When you write `a + b` where both are _Tracked:
      - L2Load(a.var), L2Load(b.var)   — both operands are "read from memory"
      - L2Op("add", (a.var, b.var), result_var)
      - L2Store(result_var)            — result is "written to memory"
    The result is a new _Tracked wrapping the arithmetic result.

    When one operand is a plain float (e.g. `tracked * 2.0`), only the
    tracked operand generates a LOAD — the constant is "free".
    """
    __slots__ = ("_t", "_v", "val")

    def __init__(self, t: _Tracer, v: int, val):
        self._t = t
        self._v = v
        self.val = val

    def _binop(self, other, name, fn):
        if isinstance(other, _Tracked):
            in_vars = (self._v, other._v)
            other_val = other.val
        else:
            in_vars = (self._v,)
            other_val = other
        for v in in_vars:
            self._t.events.append(L2Load(v))
        result_val = fn(self.val, other_val)
        out_var = self._t.fresh()
        self._t.events.append(L2Op(name, in_vars, out_var))
        self._t.events.append(L2Store(out_var))
        return _Tracked(self._t, out_var, result_val)

    def __add__(self, o): return self._binop(o, "add", operator.add)
    def __mul__(self, o): return self._binop(o, "mul", operator.mul)
    def __radd__(self, o):
        for v in (self._v,):
            self._t.events.append(L2Load(v))
        out = self._t.fresh()
        self._t.events.append(L2Op("add", (self._v,), out))
        self._t.events.append(L2Store(out))
        return _Tracked(self._t, out, operator.add(o, self.val))


def trace(func, args):
    """Run func(*args) with tracing. Returns list of L2 events.

    Each scalar in `args` (nested lists of floats) is wrapped in _Tracked.
    The function runs normally, but every arithmetic op is recorded.
    """
    t = _Tracer()
    def wrap(v):
        if isinstance(v, list):
            return [wrap(x) for x in v]
        if isinstance(v, (int, float)):
            var = t.fresh()
            t.events.append(L2Store(var))
            return _Tracked(t, var, v)
        return v
    wrapped = tuple(wrap(a) for a in args)
    func(*wrapped)
    return t.events


# ============================================================================
# Algorithm: row-major matrix-vector product (the function being measured)
# ============================================================================

def matvec_row(A, x):
    """y = A @ x using row-major access: y[i] = sum_j A[i][j] * x[j].

    Outer loop over rows i of A. Inner loop computes the dot product of
    row i of A with vector x. A is accessed row-major (contiguous), while
    x is re-read in full for every row — N complete passes over x.
    """
    n = len(A)
    y = [None] * n
    for i in range(n):
        s = A[i][0] * x[0]
        for j in range(1, n):
            s = s + A[i][j] * x[j]
        y[i] = s
    return y


# ============================================================================
# Fenwick tree — O(log n) prefix sums, used by all LRU/rank-based models
# ============================================================================

class _Fenwick:
    """1-indexed Binary Indexed Tree. Supports point updates and prefix queries.

    Used to efficiently compute "how many items have index <= k" in O(log n),
    which translates to LRU stack depth or spatial rank queries.
    """
    __slots__ = ("n", "bit")

    def __init__(self, n: int):
        self.n = n
        self.bit = [0] * (n + 1)

    def add(self, i: int, delta: int):
        while i <= self.n:
            self.bit[i] += delta
            i += i & -i

    def prefix(self, i: int) -> int:
        s = 0
        while i > 0:
            s += self.bit[i]
            i -= i & -i
        return s


# ============================================================================
# Cost model 1: SpaceDMD (density-ranked static allocation)
# ============================================================================

def space_dmd(events: Sequence[L2Event]) -> int:
    """Models an optimal ahead-of-time (AOT) static allocator like a TPU.

    Pass 1: Scan events to find each variable's birth time, last use time,
            and total access count.
    Pass 2: Compute density = access_count / lifespan for each variable.
            Sort all variables by density (descending). Assign rank 1 to
            the densest variable, rank 2 to the next, etc.
    Pass 3: Sweep through time. On each LOAD, query the variable's
            "active rank" (rank among currently-live variables) via Fenwick
            prefix sum. Cost = ceil(sqrt(active_rank)).
    """
    birth: Dict[int, int] = {}
    last_use: Dict[int, int] = {}
    access_count: Dict[int, int] = defaultdict(int)
    for i, ev in enumerate(events):
        if isinstance(ev, L2Store):
            birth[ev.var] = i
            last_use.setdefault(ev.var, i)
        elif isinstance(ev, L2Load):
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

    births_at: Dict[int, List[int]] = defaultdict(list)
    deaths_at: Dict[int, List[int]] = defaultdict(list)
    for vid in birth:
        births_at[birth[vid]].append(vid)
        deaths_at[last_use[vid]].append(vid)

    bit = _Fenwick(V)
    total = 0
    for i, ev in enumerate(events):
        for vid in births_at[i]:
            bit.add(rank_map[vid], 1)
        if isinstance(ev, L2Load):
            active_rank = bit.prefix(rank_map[ev.var])
            total += math.isqrt(max(0, active_rank - 1)) + 1
        for vid in deaths_at[i]:
            bit.add(rank_map[vid], -1)
    return total


# ============================================================================
# Cost models 2 & 4: LRU stack (bytedmd_live and bytedmd_classic)
# ============================================================================

def _lru_cost(events: Sequence[L2Event], compact_on_last_load: bool) -> int:
    """Shared LRU stack engine for both bytedmd_live and bytedmd_classic.

    Maintains an LRU stack using timestamps and a Fenwick tree:
    - Each variable gets a timestamp when STOREd (pushed to top of stack).
    - On LOAD, the variable's depth = #vars with timestamp >= its timestamp.
      Cost = ceil(sqrt(depth)). Then its timestamp is refreshed (LRU bump).
    - If compact_on_last_load=True (bytedmd_live): on a variable's LAST LOAD,
      it is removed from the stack entirely. Dead vars free up space.
    - If compact_on_last_load=False (bytedmd_classic): dead vars stay forever.
    """
    last_load: Dict[int, int] = {}
    if compact_on_last_load:
        for i, ev in enumerate(events):
            if isinstance(ev, L2Load):
                last_load[ev.var] = i

    T = len(events) + 1
    bit = _Fenwick(T)
    var_ts: Dict[int, int] = {}
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


def bytedmd_live(events):
    """LRU stack WITH liveness compaction. Dead vars are removed on last LOAD."""
    return _lru_cost(events, compact_on_last_load=True)


def bytedmd_classic(events):
    """LRU stack WITHOUT liveness compaction. Dead vars stay forever."""
    return _lru_cost(events, compact_on_last_load=False)


# ============================================================================
# Cost model 3: Manual (hand-placed bump allocator)
# ============================================================================

def manual_matvec_row(n: int) -> int:
    """Physical cost of row-major matvec with hand-placed memory layout.

    Memory layout (1-D bump-allocated, address 1 onwards):
      Address 1         : accumulator s (hottest — read every inner iteration)
      Address 2         : multiply temp tmp (intermediate product A[i][j]*x[j])
      Addresses 3..n+2  : output vector y (written once per row, read for final
                          accumulation in the tracer's add chain)
      Addresses n+3..2n+2 : input vector x (re-read N times, once per row)
      Addresses 2n+3..2n+2+n^2 : matrix A (cold bulk, row-major)

    For each row i:
      - Read A[i][0] and x[0] (first mul, creates initial s)
      - For j = 1..n-1: read A[i][j] and x[j] (mul), then read s and tmp (add)
      - Read s one final time to write y[i] (write is free)

    Cost per read at address d = ceil(sqrt(d)) = isqrt(d-1) + 1.
    Writes are free.
    """
    cost = 0
    # Allocate: hot temporaries first, then vectors, then bulk matrix
    s = 1;   ptr = 2
    tmp = 2; ptr = 3
    y = ptr; ptr += n         # y at addrs 3..n+2
    x = ptr; ptr += n         # x at addrs n+3..2n+2
    A = ptr; ptr += n * n     # A at addrs 2n+3..2n+2+n^2

    def touch(addr):
        nonlocal cost
        cost += math.isqrt(max(0, addr - 1)) + 1

    for i in range(n):
        # First element: s = A[i][0] * x[0]
        touch(A + i * n + 0)
        touch(x + 0)
        # Remaining elements: s = s + A[i][j] * x[j]
        for j in range(1, n):
            touch(A + i * n + j)  # read A[i][j]
            touch(x + j)          # read x[j]
            touch(s)              # read accumulator for add
            touch(tmp)            # read mul result for add
        # Write y[i] = s (read s to get the value, write is free)
        touch(s)
    return cost


# ============================================================================
# Main — generate and print all four numbers
# ============================================================================

def main():
    n = 64

    # Step 1: Trace matvec_row to get L2 events
    A = [[1.0] * n for _ in range(n)]
    x = [1.0] * n
    events = trace(matvec_row, (A, x))
    print(f"Traced matvec_row(n={n}): {len(events):,} L2 events\n")

    # Step 2: Evaluate all four cost models on the same trace
    sd = space_dmd(events)
    bl = bytedmd_live(events)
    mn = manual_matvec_row(n)
    bc = bytedmd_classic(events)

    print(f"| {'algorithm':<20} | {'space_dmd':>10} | {'bytedmd_live':>12} | {'manual':>8} | {'bytedmd_classic':>15} |")
    print(f"|{'-'*22}|{'-'*12}|{'-'*14}|{'-'*10}|{'-'*17}|")
    print(f"| {'matvec_row(n=64)':<20} | {sd:>10,} | {bl:>12,} | {mn:>8,} | {bc:>15,} |")

    print(f"\nExpected:              |     72,775 |      229,199 |  238,853 |         450,939 |")


if __name__ == "__main__":
    main()
