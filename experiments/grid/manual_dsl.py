"""Correctness-preserving DSL for writing manual schedules.

The raw `Allocator` from `manual.py` exposes `touch`/`touch_arg`/`write` at
the per-access level — so every binary op in an inner loop is an
opportunity to forget a read (the intermediate product of a multiply,
the re-read of an accumulator, the second op of a butterfly …). The
six classes of MAC-pricing bug we've caught in the grid so far all
share that pattern.

This module wraps Allocator with two abstractions:

  Cell  — a logical storage slot (arg-stack or scratch-stack) that
          knows its own address. First read from an arg `Cell` auto-
          promotes it onto the geometric stack (matching the
          two-stack cost model applied by the trace-based heuristics).

  Sched — a scheduler whose primitives correspond one-to-one with
          ByteDMD binary ops. Each primitive prices itself correctly:

            mul(a, b, out)        — 2 reads, free write
            add(a, b, out)        — 2 reads, free write
            sub(a, b, out)        — 2 reads, free write
            mac(acc, a, b, tmp)   — 4 reads (mul then add)
            butterfly(a, b, tmp)  — 5 reads (sum + diff via swap)
            swap(a, b, tmp)       — 3 reads (3-move through tmp)
            assign(src, dst)      — 1 read (copy via free write)
            zero_init(cell)       — 0 reads (writes are free)

          Used consistently, it becomes impossible to accidentally
          mis-price a binary op or skip an intermediate.

Usage:

    from manual_dsl import Sched

    def manual_naive_matmul(n):
        s = Sched()
        A = s.arg_buffer(n * n)
        B = s.arg_buffer(n * n)
        tmp = s.scalar()
        acc = s.scalar()
        c_A_row = s.buffer(n)
        C = s.output_buffer(n * n)

        for i in range(n):
            for k in range(n):
                s.assign(A[i*n + k], c_A_row[k])
            for j in range(n):
                for k in range(n):
                    if k == 0:
                        s.mul(c_A_row[k], B[j*n + k], acc)
                    else:
                        s.mac(acc, c_A_row[k], B[j*n + k], tmp)
                s.assign(acc, C[i*n + j])
        return s.finalize()
"""
from __future__ import annotations

from typing import List, Optional

# Re-use the Allocator from the main manual module so the DSL sits ON
# TOP of the existing cost engine rather than duplicating it.
from manual import Allocator, set_allocator, _alloc


# ---------------------------------------------------------------------------
# Cell — a logical storage slot with an address and an origin.
# ---------------------------------------------------------------------------

class Cell:
    """A single-address logical cell.

    `origin="arg"` means the cell lives on the argument stack. First
    read charges the arg-stack cost; subsequent reads come from a
    promoted scratch-stack slot (the `Sched` assigns one lazily).

    `origin="scratch"` is a normal scratch cell — every read charges
    the scratch-stack cost.
    """

    __slots__ = ("sched", "addr", "origin", "arg_addr", "_promoted_addr")

    def __init__(self, sched: "Sched", addr: int, origin: str,
                 arg_addr: Optional[int] = None) -> None:
        self.sched = sched
        self.addr = addr                  # scratch address if origin="scratch"
        self.origin = origin              # "arg" | "scratch"
        self.arg_addr = arg_addr          # arg addr if origin="arg"
        self._promoted_addr: Optional[int] = None

    # ----- pricing hooks — called by Sched, not by user code directly.

    def _read(self) -> None:
        a = self.sched._a
        if self.origin == "scratch":
            a.touch(self.addr)
        else:
            if self._promoted_addr is None:
                # First-touch of an arg cell: pay arg-stack cost and
                # auto-allocate a promoted scratch slot for subsequent
                # reads (mirrors the two-stack promotion the tracer
                # applies to input_vars).
                a.touch_arg(self.arg_addr)
                self._promoted_addr = a.alloc(1)
                a.write(self._promoted_addr)
            else:
                a.touch(self._promoted_addr)

    def _write(self) -> None:
        """Writes are free in ByteDMD; we still record the target for
        visualization and downstream promotion bookkeeping."""
        a = self.sched._a
        if self.origin == "scratch":
            a.write(self.addr)
        else:
            # Writing to an arg cell doesn't make sense — treat as a
            # write to its promoted scratch slot (which may still be
            # None; allocate if needed).
            if self._promoted_addr is None:
                self._promoted_addr = a.alloc(1)
            a.write(self._promoted_addr)

    @property
    def effective_addr(self) -> int:
        if self.origin == "scratch":
            return self.addr
        return self._promoted_addr if self._promoted_addr is not None else -1


# ---------------------------------------------------------------------------
# Sched — the DSL entry point.
# ---------------------------------------------------------------------------

class Sched:
    """A scheduler that tracks allocations and prices binary ops
    correctly. Wraps a single Allocator instance."""

    def __init__(self, allocator: Optional[Allocator] = None) -> None:
        self._a = allocator if allocator is not None else _alloc()
        self._output_cells: Optional[List[Cell]] = None

    # ----- Cell constructors -------------------------------------------------

    def scalar(self) -> Cell:
        """Allocate a 1-cell scratch scalar (hot register-style)."""
        return Cell(self, self._a.alloc(1), origin="scratch")

    def buffer(self, size: int) -> List[Cell]:
        """Allocate a contiguous run of scratch cells; return them as Cells."""
        base = self._a.alloc(size)
        return [Cell(self, base + i, origin="scratch") for i in range(size)]

    def arg_buffer(self, size: int) -> List[Cell]:
        """Allocate an arg-stack buffer of `size` cells. Returned cells
        auto-promote to scratch on first read."""
        base = self._a.alloc_arg(size)
        return [Cell(self, -1, origin="arg", arg_addr=base + i)
                for i in range(size)]

    def output_buffer(self, size: int) -> List[Cell]:
        """Allocate a scratch buffer designated as algorithm output.
        The epilogue read (a.read_output()) will price each cell."""
        base = self._a.alloc(size)
        cells = [Cell(self, base + i, origin="scratch") for i in range(size)]
        self._a.set_output_range(base, base + size)
        self._output_cells = cells
        return cells

    # ----- Primitive binary ops ---------------------------------------------

    def mul(self, a: Cell, b: Cell, out: Cell) -> None:
        """out = a * b   (2 reads, free write)"""
        a._read(); b._read(); out._write()

    def add(self, a: Cell, b: Cell, out: Cell) -> None:
        """out = a + b   (2 reads, free write)"""
        a._read(); b._read(); out._write()

    def sub(self, a: Cell, b: Cell, out: Cell) -> None:
        """out = a - b   (2 reads, free write)"""
        a._read(); b._read(); out._write()

    def mac(self, acc: Cell, a: Cell, b: Cell, tmp: Cell) -> None:
        """acc += a * b.

        Decomposes to two binary ops = 4 priced reads:
          tmp = a * b     # 2 reads, free write
          acc = acc + tmp # 2 reads, free write
        """
        self.mul(a, b, tmp)
        self.add(acc, tmp, acc)

    def mac_into(self, dst: Cell, a: Cell, b: Cell, tmp: Cell) -> None:
        """Explicit first-MAC pattern: dst = a * b (overwrites, no acc
        read). Use this for the first iteration of an accumulation
        loop to avoid the spurious zero-init read."""
        self.mul(a, b, tmp)
        self.assign(tmp, dst)

    def butterfly(self, a: Cell, b: Cell, tmp: Cell) -> None:
        """In-place butterfly:
          (a, b)  →  (a + b, a - b)
        Decomposes to 5 priced reads via a sum+diff pattern with one tmp:
          tmp = a - b           # 2 reads
          a   = a + b           # 2 reads (in-place)
          b   = tmp             # 1 read (assign)
        """
        self.sub(a, b, tmp)
        self.add(a, b, a)
        self.assign(tmp, b)

    def swap(self, a: Cell, b: Cell, tmp: Cell) -> None:
        """Swap a and b via 3 reads through tmp (assignments are binary
        moves)."""
        self.assign(a, tmp)
        self.assign(b, a)
        self.assign(tmp, b)

    def assign(self, src: Cell, dst: Cell) -> None:
        """dst = src  (1 read, free write).

        Fine-grained for when you literally want a copy. For arrays,
        use `copy` or `load_block`."""
        src._read(); dst._write()

    def zero_init(self, dst: Cell) -> None:
        """Initialize dst to zero. Writes are free — this costs 0
        reads. Use explicitly to document intent (and avoid accidentally
        touching an uninitialized cell)."""
        dst._write()

    # ----- Array-oriented helpers -------------------------------------------

    def copy_block(self, src: List[Cell], dst: List[Cell]) -> None:
        """dst[i] = src[i] for each cell (1 read per cell)."""
        assert len(src) == len(dst)
        for s, d in zip(src, dst):
            self.assign(s, d)

    def read(self, cell: Cell) -> None:
        """Bare read of a cell. Use sparingly — the binary-op primitives
        should cover most cases. Useful for non-MAC touches (e.g. a
        pivot scalar read once outside an inner loop)."""
        cell._read()

    def write(self, cell: Cell) -> None:
        cell._write()

    # ----- Finalize ---------------------------------------------------------

    def finalize(self) -> int:
        """Emit the output-read epilogue and return total cost."""
        self._a.read_output()
        return self._a.cost


# ---------------------------------------------------------------------------
# Convenience: run an Sched-using function under a logging allocator.
# ---------------------------------------------------------------------------

def run_logged(fn) -> Allocator:
    """Install a logging Allocator, call `fn()`, restore, return the
    Allocator so callers can inspect `.log`, `.writes`, `.output_writes`,
    and `.cost`."""
    a = Allocator(logging=True)
    set_allocator(a)
    try:
        fn()
    finally:
        set_allocator(None)
    return a
