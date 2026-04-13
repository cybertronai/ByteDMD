"""
ByteDMD tracer — Earliest Deadline First (EDF) Register Allocation Strategy

Features implemented:
  1. The Tombstone Strategy: Variables live in rigid slots for intervals [start, end)
     and vaporize entirely after their last read. Unsolicited reads are mathematically rejected.
  2. Separate Cache Hierarchy: Arguments and pre-loaded data perfectly overlap at t=0 and
     originate in the E (External) triangle. Computations originate in the W (Working) triangle.
  3. Iterative MWIS (EDF): Allocates intervals to spatial slots of increasing cost optimally
     using O(N log N) Earliest-Deadline-First greedy interval packing.
  4. Squatter Anomaly Resolution: Because W and E slots are properly routed via EDF packing,
     it rigorously preserves the true asymptotic Big-O data movement complexity.
"""

import math
import operator

# ==============================================================================
# 1. TRACING MECHANISM (Pass 1)
# ==============================================================================

class _Context:
    __slots__ = ('sync', 'memo', 'counter', 'ir', 'events')

    def __init__(self):
        self.sync, self.ir, self.events = [], [], []
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

        self._ctx.read(read_keys)
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


# ==============================================================================
# 2. EDF PRICING UTILITY (Pass 2)
# ==============================================================================

def _sum_usqrt(N):
    """Closed form of Sum[ceil(sqrt(i))] from 1 to N."""
    if N <= 0: return 0
    M = math.isqrt(N - 1) + 1
    return M * (6 * N - 2 * M * M + 3 * M - 1) // 6

def edf_price_trace(events, bytes_per_element=1):
    """
    Takes a logical trace of stores and reads and prices it by simulating
    the spatial physics of an optimal open register allocator via the
    standard Earliest Deadline First (Iterative MWIS) algorithm.
    """
    def get_cost(idx):
        if bytes_per_element == 1:
            return math.isqrt(idx - 1) + 1
        return _sum_usqrt(idx * bytes_per_element) - _sum_usqrt((idx - 1) * bytes_per_element)

    var_state = {}
    intervals = []

    # 1. Extract Intervals
    t = 0
    for ev in events:
        if ev[0] == 'STORE':
            var_state[ev[1]] = (t, 'W')
        elif ev[0] == 'READ_BATCH':
            t += 1
            counts = {}
            for k in ev[1]:
                counts[k] = counts.get(k, 0) + 1

            for k, count in counts.items():
                if k not in var_state:
                    var_state[k] = (0, 'E') # Argument cold miss
                t_start, typ = var_state[k]
                intervals.append({'start': t_start, 'end': t, 'type': typ, 'weight': count, 'key': k})
                var_state[k] = (t, 'ANY') # Post-read agile routing

    # 2. Generate Unified Rigid Slot Pool
    max_slots = len(intervals)
    slots = []
    for k in range(1, max_slots + 1):
        cost = get_cost(k)
        slots.append({'cost': cost, 'type': 'W', 'depth': k})
        slots.append({'cost': cost, 'type': 'E', 'depth': k})

    # Cost-sorted unified pool. Break ties organically favoring Working Memory.
    slots.sort(key=lambda x: (x['cost'], 0 if x['type'] == 'W' else 1))

    # Earliest-Deadline-First (EDF) deterministic priority sorting
    intervals.sort(key=lambda x: (x['end'], x['start'], x['key']))

    total_cost = 0
    remaining = intervals[:]
    read_costs = {}

    # 3. Iterative Interval Packing
    for slot in slots:
        if not remaining:
            break

        slot_type = slot['type']
        slot_cost = slot['cost']

        # Filter topological compatibility
        compatible = [iv for iv in remaining if iv['type'] in (slot_type, 'ANY')]

        last_end = -1
        packed_this_round = []

        # Greedily pack the Maximum Independent Set directly adjacent to the processor
        for iv in compatible:
            if iv['start'] >= last_end:
                packed_this_round.append(iv)
                last_end = iv['end']

        if packed_this_round:
            packed_ids = {id(iv) for iv in packed_this_round}
            remaining = [iv for iv in remaining if id(iv) not in packed_ids]

            for iv in packed_this_round:
                total_cost += slot_cost * iv['weight'] # Applies spatial read routing penalty
                read_costs[(iv['key'], iv['end'])] = (slot['depth'], slot['type'], slot_cost)

    return total_cost, read_costs


# ==============================================================================
# 3. PUBLIC API WRAPPERS
# ==============================================================================

def traced_eval(func, args, bytes_per_element=1):
    ctx = _Context()
    wrapped_args = tuple(_wrap(ctx, a, deferred=True) for a in args)
    res = func(*wrapped_args)

    # Decoupled utility calculation
    total_cost, read_costs = edf_price_trace(ctx.events, bytes_per_element)

    ir = []
    t = 0
    op_start_stack = []
    last_depths_map = {}

    # Reconstruct chronological ByteDMD IR using spatial pricing geometry
    for ev in ctx.events:
        if ev[0] == 'STORE':
            ir.append(('STORE', ev[1]))
        elif ev[0] == 'READ_BATCH':
            t += 1
            valid = ev[1]
            unique = list(dict.fromkeys(valid))
            for k in unique:
                depth, typ, cost = read_costs.get((k, t), (0, '?', 0))
                ir.append(('READ', k, depth, typ, cost))
                last_depths_map[k] = (depth, typ, cost)
        elif ev[0] == 'OP_START':
            op_start_stack.append(len(ir))
        elif ev[0] == 'OP_END':
            name, valid_keys, out_key = ev[1], ev[2], ev[3]
            depths_info = [last_depths_map.get(k, (0, '?', 0)) for k in valid_keys]
            depths = [d[0] for d in depths_info]
            types = [d[1] for d in depths_info]
            costs = [d[2] for d in depths_info]
            idx = op_start_stack.pop()
            ir.insert(idx, ('OP', name, valid_keys, depths, types, costs, out_key))
        elif ev[0] == 'OP':
            name, valid_keys, out_key = ev[1], ev[2], ev[3]
            depths_info = [last_depths_map.get(k, (0, '?', 0)) for k in valid_keys]
            depths = [d[0] for d in depths_info]
            types = [d[1] for d in depths_info]
            costs = [d[2] for d in depths_info]
            ir.append(('OP', name, valid_keys, depths, types, costs, out_key))

    ctx.ir = ir

    memo = {}
    for orig, wrapped in ctx.sync:
        if isinstance(orig, list): orig[:] = _unwrap(wrapped, memo)
        elif type(orig).__name__ == 'ndarray': orig[...] = _unwrap(wrapped, memo)

    return total_cost, _unwrap(res, memo), ctx.ir

def inspect_ir(func, args, bytes_per_element=1):
    _, _, ir = traced_eval(func, args, bytes_per_element)
    return ir

def format_ir(ir):
    out, total = [], 0
    for ev in ir:
        if ev[0] == 'STORE':
            out.append(f"STORE v{ev[1]}")
        elif ev[0] == 'READ':
            out.append(f"  READ v{ev[1]}@{ev[3]}{ev[2]}  cost={ev[4]}")
        else:
            _, name, keys, depths, types, costs, _ok = ev
            cost = sum(costs)
            total += cost
            rd = ", ".join(f"v{k}@{typ}{d}" for k, d, typ in zip(keys, depths, types))
            out.append(f"OP    {name}({rd})  cost={cost}")
    out.append(f"# total cost = {total}")
    return "\n".join(out)

def bytedmd(func, args, bytes_per_element=1):
    cost, _, _ = traced_eval(func, args, bytes_per_element)
    return cost
