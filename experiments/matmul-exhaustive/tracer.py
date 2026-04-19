"""
Flat Manhattan-distance cost model with manual allocation.

Matches the `manual` column of experiments/grid/:

    cost = sum over each read of ceil(sqrt(addr))

where `addr` is the 1-indexed address of the cell in its stack.
Writes are free. Two independent address spaces:

    arg stack:      read-only inputs   (A, B for matmul)
    scratch stack:  scratch + outputs  (tmp, partial products, C)

Addresses come from a bump-pointer allocator with LIFO push/pop.
Cells are never silently relocated. Under this model the *order* of
independent operations does not affect the cost — only the *layout*
(which cell lives at which address) and the *read profile* (how many
times each cell is read) matter. Lifetime interactions can force a
cell to sit at a higher address than an unconstrained optimum.
"""

import math


def addr_cost(addr):
    """ceil(sqrt(addr)) for addr >= 1."""
    return math.isqrt(addr - 1) + 1


class Allocator:
    """Two-space bump-pointer allocator with cost accounting + push/pop.

    `alloc(size)` and `alloc_arg(size)` return the base address of the
    allocated range. `push()` / `pop(saved_ptr)` implement LIFO
    deallocation on the scratch stack.
    """

    __slots__ = (
        'cost', 'ptr', 'arg_ptr',
        'peak', 'arg_peak',
        'log',
    )

    def __init__(self):
        self.cost = 0
        self.ptr = 1
        self.arg_ptr = 1
        self.peak = 0
        self.arg_peak = 0
        self.log = []

    def alloc(self, size=1):
        base = self.ptr
        self.ptr += size
        if self.ptr - 1 > self.peak:
            self.peak = self.ptr - 1
        return base

    def alloc_arg(self, size=1):
        base = self.arg_ptr
        self.arg_ptr += size
        if self.arg_ptr - 1 > self.arg_peak:
            self.arg_peak = self.arg_ptr - 1
        return base

    def push(self):
        return self.ptr

    def pop(self, saved_ptr):
        assert saved_ptr <= self.ptr
        self.ptr = saved_ptr

    def read(self, addr):
        self.cost += addr_cost(addr)
        self.log.append(('scratch', addr))

    def read_arg(self, addr):
        self.cost += addr_cost(addr)
        self.log.append(('arg', addr))

    def write(self, addr):
        pass   # free under this model


def evaluate_layout(read_counts_scratch, read_counts_arg):
    """Cost under *optimal unconstrained placement* — sort cells by
    read count descending and assign addresses 1, 2, 3, .... Achievable
    for strategies whose cells are mutually live throughout (no push/
    pop reuse); an optimistic lower bound otherwise.
    """
    def place(counts):
        return sum(n * addr_cost(a + 1)
                   for a, n in enumerate(sorted(counts, reverse=True)))
    return place(read_counts_scratch) + place(read_counts_arg)
