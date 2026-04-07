"""
ByteDMD tracer with three pluggable memory management strategies.

Strategy 1: Unmanaged
    Free is a no-op. Temporaries pile up forever, pushing live data deeper.
    This matches the current bytedmd.py behavior — abstract addresses are
    never recycled. Models a memory-leaking allocator.

Strategy 2: Tombstone (Traditional GC / Hole-fill)
    Freed slots become tombstones. New allocations reuse the most recently
    freed tombstone in place (LIFO of holes). Old data BELOW a freed slot
    still sees the same depth (depth includes the dead slot, because in
    real hardware the cache line still occupies a physical address).
    Models a standard malloc/free or a stop-the-world GC that doesn't
    compact.

Strategy 3: Aggressive (Instant compaction)
    Freed slots are removed from the stack immediately. Items below slide
    UP toward MRU. Models an idealized compacting GC where dead bytes get
    instantly returned to faster cache.

The cost model is element-level (bytes_per_element=1). Reading at depth d
costs ceil(sqrt(d)). Reuses _Context semantics from bytedmd.py.
"""

import math


def usqrt(d):
    """ceil(sqrt(d)) for d > 0."""
    return math.isqrt(d - 1) + 1 if d > 0 else 0


def trace_to_cost(trace):
    """Total ByteDMD cost = sum of ceil(sqrt(depth)) over all reads."""
    return sum(usqrt(d) for d in trace)


class Context:
    """LRU stack with pluggable free() semantics.

    Implementation: a doubly-linked list of nodes, each holding a key
    and alive flag. We keep a dict `slot[key] -> node` for O(1) lookup
    and a `top_index` count to compute depth in O(distance from top).
    For correctness we use a simpler approach: a Python list with a
    parallel `key_to_index` dict that we keep in sync after each
    mutation. This makes index() O(1) but mutations (insert/delete)
    require updating the index dict for shifted entries.

    To keep mutations cheap, we represent the LRU stack as a Python
    list of (key, alive) but locate keys via a dict. We rebuild the
    dict-of-positions lazily.
    """

    def __init__(self, strategy='unmanaged'):
        assert strategy in ('unmanaged', 'tombstone', 'aggressive')
        self.strategy = strategy
        self.stack = []        # entries: positive int (alive key) or -1 (dead)
        self.pos = {}          # key -> current index in self.stack
        self.trace = []
        self.counter = 0

    def _refresh_pos(self):
        """Recompute pos dict after a stack mutation that shifts indices."""
        self.pos = {k: i for i, k in enumerate(self.stack) if k != -1}

    def allocate(self):
        self.counter += 1
        key = self.counter
        if self.strategy == 'tombstone':
            # Scan from top down for the most recent tombstone.
            for i in range(len(self.stack) - 1, -1, -1):
                if self.stack[i] == -1:
                    self.stack[i] = key
                    self.pos[key] = i
                    return key
        self.stack.append(key)
        self.pos[key] = len(self.stack) - 1
        return key

    def read(self, key):
        idx = self.pos[key]
        depth = len(self.stack) - idx
        self.trace.append(depth)
        # Move key from idx to top. The entries between idx+1 and end
        # all shift down by 1.
        del self.stack[idx]
        self.stack.append(key)
        # Update pos for everything that shifted: entries previously at
        # indices > idx now sit at idx, idx+1, ...
        for j in range(idx, len(self.stack) - 1):
            k = self.stack[j]
            if k != -1:
                self.pos[k] = j
        self.pos[key] = len(self.stack) - 1
        return depth

    def free(self, key):
        if self.strategy == 'unmanaged':
            return
        idx = self.pos.get(key)
        if idx is None:
            return
        if self.strategy == 'tombstone':
            self.stack[idx] = -1
            del self.pos[key]
        elif self.strategy == 'aggressive':
            del self.stack[idx]
            del self.pos[key]
            for j in range(idx, len(self.stack)):
                k = self.stack[j]
                if k != -1:
                    self.pos[k] = j


class Tracked:
    """Element wrapper. Records reads via dunder methods, frees on __del__."""

    def __init__(self, ctx, key, val):
        self._ctx = ctx
        self._key = key
        self.val = val

    def __del__(self):
        # Hook into Python GC to simulate memory reclamation.
        # Wrapped in try/except because __del__ may run during interpreter
        # shutdown when self._ctx may be gone.
        try:
            self._ctx.free(self._key)
        except Exception:
            pass

    def _do_binop(self, other, op):
        if isinstance(other, Tracked):
            self._ctx.read(self._key)
            self._ctx.read(other._key)
            v = op(self.val, other.val)
        else:
            self._ctx.read(self._key)
            v = op(self.val, other)
        return Tracked(self._ctx, self._ctx.allocate(), v)

    def __add__(self, other): return self._do_binop(other, lambda a, b: a + b)
    def __sub__(self, other): return self._do_binop(other, lambda a, b: a - b)
    def __mul__(self, other): return self._do_binop(other, lambda a, b: a * b)
    def __radd__(self, other): return self._do_binop(other, lambda a, b: b + a)
    def __rsub__(self, other): return self._do_binop(other, lambda a, b: b - a)
    def __rmul__(self, other): return self._do_binop(other, lambda a, b: b * a)


def wrap_matrix(ctx, mat):
    """Convert a list-of-lists matrix into Tracked objects."""
    return [[Tracked(ctx, ctx.allocate(), v) for v in row] for row in mat]


def measure(matmul_fn, A, B, strategy):
    """Run matmul_fn(A, B) under the given strategy and return (cost, n_reads, peak_stack)."""
    ctx = Context(strategy=strategy)
    A_w = wrap_matrix(ctx, A)
    B_w = wrap_matrix(ctx, B)
    # Track the peak live stack length so we can correlate with footprint
    # bounds in the asymptotic analysis.
    peak = [len(ctx.stack)]
    orig_alloc = ctx.allocate
    def tracking_alloc():
        k = orig_alloc()
        peak[0] = max(peak[0], len(ctx.stack))
        return k
    ctx.allocate = tracking_alloc

    result = matmul_fn(A_w, B_w)
    # Force the result to drop so its temporaries are reclaimed before we
    # snapshot the trace. Without this, GC of the result happens after we
    # already returned from this function — the trace is unaffected, but
    # peak stack would not include the result-cleanup steps.
    del result
    del A_w
    del B_w

    return trace_to_cost(ctx.trace), len(ctx.trace), peak[0]
