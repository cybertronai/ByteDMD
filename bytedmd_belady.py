"""
ByteDMD tracer — Belady (OPT) Stack + Separate DRAM Hierarchy

Features implemented:
  1. Two-pass liveness analysis: Pass 1 logs all accesses; Pass 2 simulates the cache.
  2. Simultaneous Pricing: All inputs are priced against the pre-instruction
     stack state before the cache is updated, guaranteeing commutativity.
  3. Belady Algorithm (OPT Stack): The active SRAM is continuously maintained sorted
     by Next-Use time. Variables needed soonest float to the top (Depth 1).
  4. Aggressive Compaction (Slot Reuse): Variables vaporize the exact step they
     are read for the last time, instantly shrinking the active cache.
  5. Separate Cache Hierarchy for Cold Misses: The first access to an unmapped input
     is priced on an independent, monotonic tape (DRAM), rather than the local SRAM size.
  6. Scalar Tracking Granularity: Tracks primitive elements independently.
"""

import math
import operator
import bisect

class _Context:
    __slots__ = ('stack', 'trace', 'sync', 'memo', 'counter', 'ir', 'events', 'last_use')

    def __init__(self):
        self.stack, self.trace, self.sync, self.ir, self.events = [], [], [], [], []
        self.memo = {}
        self.counter = 0

    def allocate(self, deferred=False):
        """Allocate a tracking ID. Deferred pushing until second pass via STORE event."""
        self.counter += 1
        if not deferred:
            self.events.append(('STORE', self.counter))
        return self.counter

    def read(self, keys):
        """Emits a READ_BATCH logical event mapped to be priced during pass 2."""
        valid = [k for k in keys if k is not None]
        if not valid: return []
        self.events.append(('READ_BATCH', valid))
        return [0] * len(valid)

class _Tracked:
    __slots__ = ('_ctx', '_key', 'val')
    def __init__(self, ctx, key, val):
        self._ctx, self._key, self.val = ctx, key, val

    def _rd(self):
        self._ctx.read([self._key])
        return self.val

    def __str__(self): return str(self.val)
    def __repr__(self): return f"Tracked({self.val})"
    def __bool__(self): return bool(self._rd())
    def __int__(self): return int(self._rd())
    def __float__(self): return float(self._rd())
    def __complex__(self): return complex(self._rd())
    def __index__(self): return operator.index(self._rd())
    def __hash__(self): return hash(self._rd())

def _make_op(op, rev=False):
    name = op.__name__
    def method(self, *args):
        keys = [getattr(a, '_key', None) for a in args]
        vals = [getattr(a, 'val', a) for a in args]

        read_keys = [keys[0], self._key] + keys[1:] if rev else [self._key] + keys
        res = op(vals[0], self.val, *vals[1:]) if rev else op(self.val, *vals)

        depths = self._ctx.read(read_keys)
        valid_keys = [k for k in read_keys if k is not None]

        if res is NotImplemented:
            self._ctx.events.append(('OP', name, valid_keys, None))
            return res

        self._ctx.events.append(('OP_START', name, valid_keys))
        wrapped = _wrap(self._ctx, res)
        out_key = getattr(wrapped, '_key', None)

        self._ctx.events.append(('OP_END', name, valid_keys, out_key))
        return wrapped
    return method

_OPS = {
    **{k: getattr(operator, k) for k in 'add sub mul truediv floordiv mod lshift rshift xor matmul neg pos abs invert eq ne lt le gt ge'.split()},
    'and': operator.and_, 'or': operator.or_, 'divmod': divmod, 'pow': pow,
    'trunc': math.trunc, 'ceil': math.ceil, 'floor': math.floor, 'round': round
}

for n, f in _OPS.items():
    setattr(_Tracked, f'__{n}__', _make_op(f))
    if n in 'add sub mul truediv floordiv mod divmod pow lshift rshift and xor or matmul'.split():
        setattr(_Tracked, f'__r{n}__', _make_op(f, rev=True))

def _wrap(ctx, val, deferred=False):
    if isinstance(val, _Tracked): return val
    vid = id(val)
    if vid in ctx.memo: return ctx.memo[vid]

    typ = type(val)
    is_prim = typ in (int, float, bool, complex, str)

    if typ.__name__ == 'ndarray':
        import numpy as np
        res = np.empty_like(val, dtype=object)
        if not is_prim:
            ctx.memo[vid] = res
            ctx.sync.append((val, res))
        for idx in np.ndindex(val.shape):
            v = val[idx]
            res[idx] = _wrap(ctx, v.item() if hasattr(v, 'item') and not isinstance(v, np.ndarray) else v, deferred)
        return res

    if isinstance(val, (list, tuple)):
        res = typ(_wrap(ctx, v, deferred) for v in val)
        if not is_prim:
            ctx.memo[vid] = res
            if typ is list: ctx.sync.append((val, res))
        return res

    key = ctx.allocate(deferred)
    res = _Tracked(ctx, key, val)
    if not is_prim: ctx.memo[vid] = res
    return res

def _unwrap(val, memo=None):
    if memo is None: memo = {}
    vid = id(val)
    if vid in memo: return memo[vid]

    typ = type(val)
    is_prim = typ in (int, float, bool, complex, str)

    if isinstance(val, (list, tuple)):
        res = typ(_unwrap(v, memo) for v in val)
        if not is_prim: memo[vid] = res
        return res

    if typ.__name__ == 'ndarray':
        import numpy as np
        if val.dtype == object:
            flat = [_unwrap(x, memo) for x in val.flat]
            try: res = np.array(flat).reshape(val.shape)
            except Exception:
                res = np.empty_like(val, dtype=object)
                for i, x in enumerate(flat): res.flat[i] = x
        else: res = val
        if not is_prim: memo[vid] = res
        return res

    res = getattr(val, 'val', val)
    if not is_prim: memo[vid] = res
    return res

def _pass2(ctx, res):
    """Pass 2: Computes liveness, applies Belady OPT caching, and tracks cold misses separately."""
    events = ctx.events
    accesses = {}

    # --- 1. Liveness Analysis ---
    for i, ev in enumerate(events):
        if ev[0] == 'STORE':
            k = ev[1]
            if k not in accesses:
                accesses[k] = []
            accesses[k].append(i)
        elif ev[0] == 'READ_BATCH':
            for k in ev[1]:
                if k not in accesses:
                    accesses[k] = []
                accesses[k].append(i)

    names = {}
    def collect_keys(val):
        if isinstance(val, _Tracked):
            names[val._key] = True
        elif isinstance(val, (list, tuple)):
            for v in val: collect_keys(v)
        elif type(val).__name__ == 'ndarray':
            import numpy as np
            for v in val.flat: collect_keys(v)

    collect_keys(res)
    # Results are marked alive until the end of the program
    for k in names:
        if k not in accesses: accesses[k] = []
        accesses[k].append(len(events))

    last_use = {k: acc[-1] for k, acc in accesses.items() if acc}

    def get_next_use(k, current_step):
        """Binary search to find the very next time a variable is used strictly > current_step."""
        acc = accesses.get(k, [])
        idx = bisect.bisect_right(acc, current_step)
        if idx < len(acc):
            return acc[idx]
        return float('inf')

    # --- 2. Cache Simulation ---
    active_keys = []
    input_miss_count = 0  # Separate DRAM cache hierarchy tape
    trace = []
    ir = []
    op_start_stack = []
    last_depths_map = {}

    for i, ev in enumerate(events):
        if ev[0] == 'STORE':
            k = ev[1]
            # Only store if it will be used again in the future
            if last_use.get(k, -1) > i:
                if k not in active_keys:
                    active_keys.append(k)
                # Sort SRAM via Belady MIN (Earliest Next Use -> Top of Stack)
                active_keys.sort(key=lambda x: (get_next_use(x, i), x))
            ir.append(('STORE', k))

        elif ev[0] == 'READ_BATCH':
            valid = ev[1]
            unique = list(dict.fromkeys(valid))
            depths_map = {}

            # Simultaneous pricing against the pre-instruction OPT stack
            for k in unique:
                if k in active_keys:
                    # Hot Hit: Depth is 1-based index in the Belady-sorted SRAM stack
                    depths_map[k] = active_keys.index(k) + 1
                else:
                    # Cold Miss: Billed on the separate independent DRAM hierarchy
                    input_miss_count += 1
                    depths_map[k] = input_miss_count
                    active_keys.append(k)

            trace.extend(depths_map[k] for k in valid)
            for k in valid:
                ir.append(('READ', k, depths_map[k]))

            last_depths_map = depths_map

            # Aggressive Compaction: Vaporize variables that just experienced their last read
            active_keys = [x for x in active_keys if last_use.get(x, -1) > i]

            # Re-sort active working set using Belady's OPT stack order for upcoming instructions
            active_keys.sort(key=lambda x: (get_next_use(x, i), x))

        elif ev[0] == 'OP_START':
            op_start_stack.append(len(ir))

        elif ev[0] == 'OP_END':
            name, valid_keys, out_key = ev[1], ev[2], ev[3]
            depths = [last_depths_map.get(k, 0) for k in valid_keys]
            idx = op_start_stack.pop()
            ir.insert(idx, ('OP', name, valid_keys, depths, out_key))

        elif ev[0] == 'OP':
            # Fallback for NotImplemented returns
            name, valid_keys, out_key = ev[1], ev[2], ev[3]
            depths = [last_depths_map.get(k, 0) for k in valid_keys]
            ir.append(('OP', name, valid_keys, depths, out_key))

    ctx.trace = trace
    ctx.ir = ir
    ctx.last_use = last_use

def _sum_usqrt(N):
    """Closed form of Sum[ceil(sqrt(i))] from 1 to N."""
    if N <= 0: return 0
    M = math.isqrt(N - 1) + 1
    return M * (6 * N - 2 * M * M + 3 * M - 1) // 6

def traced_eval(func, args):
    ctx = _Context()
    wrapped_args = tuple(_wrap(ctx, a, deferred=True) for a in args)
    res = func(*wrapped_args)

    _pass2(ctx, res)

    memo = {}
    for orig, wrapped in ctx.sync:
        if isinstance(orig, list): orig[:] = _unwrap(wrapped, memo)
        elif type(orig).__name__ == 'ndarray': orig[...] = _unwrap(wrapped, memo)

    return ctx.trace, _unwrap(res, memo)

def trace_to_bytedmd(trace, bytes_per_element=1):
    """Apply the 2D spatial physical penalty (sqrt) to the depth traces."""
    if bytes_per_element == 1:
        return sum(math.isqrt(d - 1) + 1 for d in trace)
    bpe = bytes_per_element
    return sum(_sum_usqrt(d * bpe) - _sum_usqrt((d - 1) * bpe) for d in trace)

def inspect_ir(func, args):
    ctx = _Context()
    wrapped_args = tuple(_wrap(ctx, a, deferred=True) for a in args)
    res = func(*wrapped_args)
    _pass2(ctx, res)
    return ctx.ir

def format_ir(ir):
    out, total = [], 0
    for ev in ir:
        if ev[0] == 'STORE':
            out.append(f"STORE v{ev[1]}")
        elif ev[0] == 'READ':
            cost = math.isqrt(ev[2] - 1) + 1
            out.append(f"  READ v{ev[1]}@{ev[2]}  cost={cost}")
        else:
            _, name, keys, depths, _ok = ev
            cost = sum(math.isqrt(d - 1) + 1 for d in depths)
            total += cost
            rd = ", ".join(f"v{k}@{d}" for k, d in zip(keys, depths))
            out.append(f"OP    {name}({rd})  cost={cost}")
    out.append(f"# total cost = {total}")
    return "\n".join(out)

def bytedmd(func, args, bytes_per_element=1):
    trace, _ = traced_eval(func, args)
    return trace_to_bytedmd(trace, bytes_per_element)
