"""
ByteDMD tracer — stationary slots with demand-paged initialization.

  1. Stationary scratchpad: SRAM is an array of fixed slots numbered
     1, 2, 3, .... The cost to access slot d is ceil(sqrt(d)).
     Variables do not move once placed.
  2. Min-heap slot recycling: Dead variables' slots are returned to a
     min-heap free-list so the next allocation reuses the lowest
     available position.
  3. Global Peak & Demand-paged initialization: Arguments start in "DRAM".
     First access triggers a cold miss priced on a monotonically increasing
     tape strictly beyond the peak working set.
  4. Liveness analysis: Two-pass analysis determines when each variable's
     last use occurs; dead variables' slots are freed immediately.
"""

import heapq
import math
import operator

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
    """Pass 2: Computes liveness over the event log, vaporizes dead variables, and constructs the trace."""
    events = ctx.events
    last_use = {}
    for i, ev in enumerate(events):
        if ev[0] == 'READ_BATCH':
            for k in ev[1]:
                last_use[k] = i
        elif ev[0] == 'STORE':
            k = ev[1]
            if k not in last_use:
                last_use[k] = i

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

    # Results are marked alive until the end
    for k in names:
        last_use[k] = len(events)

    # --- Pass 1.5 - Calculate Global Peak Working Set ---
    sim_stack = []
    peak_working_set = 0

    def kill_dead_sim(current_idx):
        nonlocal sim_stack
        new_stack = []
        for k in sim_stack:
            if last_use.get(k, -1) > current_idx:
                new_stack.append(k)
        sim_stack = new_stack

    for i, ev in enumerate(events):
        if ev[0] == 'STORE':
            sim_stack.append(ev[1])
            if len(sim_stack) > peak_working_set:
                peak_working_set = len(sim_stack)
            kill_dead_sim(i)

        elif ev[0] == 'READ_BATCH':
            unique = list(dict.fromkeys(ev[1]))
            for k in unique:
                if k not in sim_stack:
                    sim_stack.append(k)
            # Record the peak footprint the moment data is fully paged into L1
            if len(sim_stack) > peak_working_set:
                peak_working_set = len(sim_stack)
            kill_dead_sim(i)
    # ---------------------------------------------------------

    # --- Pass 3: Stationary Slots + Min-Heap Free-List ---
    # Variables occupy fixed memory slots.  Accessing slot d costs sqrt(d).
    # Dead variables' slots are recycled via a min-heap so the next
    # allocation reuses the lowest available position.
    # Cold misses are priced on a monotonic tape past the peak working set.
    slot_of = {}            # key -> assigned slot number
    free_heap = list(range(1, peak_working_set + 1))  # slots 1..peak_working_set
    heapq.heapify(free_heap)
    live_keys = set()       # keys currently occupying a slot

    trace = []
    ir = []
    last_depths_map = {}
    op_start_stack = []

    global_cold_counter = 0  # Monotonically increasing spatial tape for DRAM misses

    def free_dead_slots(current_idx):
        dead = [k for k in live_keys if last_use.get(k, -1) <= current_idx]
        for k in dead:
            live_keys.discard(k)
            heapq.heappush(free_heap, slot_of[k])
            del slot_of[k]

    def alloc_slot(k):
        """Assign the lowest available slot to key k."""
        if free_heap:
            s = heapq.heappop(free_heap)
        else:
            # Overflow: allocate beyond peak_working_set (should be rare)
            s = peak_working_set + len(live_keys) + 1
        slot_of[k] = s
        live_keys.add(k)

    for i, ev in enumerate(events):
        if ev[0] == 'STORE':
            k = ev[1]
            alloc_slot(k)
            ir.append(('STORE', k))
            free_dead_slots(i)

        elif ev[0] == 'READ_BATCH':
            valid = ev[1]
            unique = list(dict.fromkeys(valid))
            depths_map = {}

            # Price simultaneously against current slot assignments
            for k in unique:
                if k in slot_of:
                    # Hot hit: cost = assigned slot number (stationary)
                    depths_map[k] = slot_of[k]
                else:
                    # Cold miss: priced on monotonic tape past peak working set
                    global_cold_counter += 1
                    depths_map[k] = peak_working_set + global_cold_counter
                    # After cold miss, assign a stationary SRAM slot
                    alloc_slot(k)

            trace.extend(depths_map[k] for k in valid)
            for k in valid:
                ir.append(('READ', k, depths_map[k]))

            last_depths_map = depths_map
            free_dead_slots(i)

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

def trace_to_bytedmd(trace, bytes_per_element):
    if bytes_per_element == 1: return sum(math.isqrt(d - 1) + 1 for d in trace)
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

def _collect_keys(wrapped, name, names):
    """Walk a wrapped structure to map tracking keys to human-readable names."""
    if isinstance(wrapped, _Tracked):
        names[wrapped._key] = name
    elif isinstance(wrapped, (list, tuple)):
        for i, v in enumerate(wrapped):
            _collect_keys(v, f"{name}[{i}]", names)
    elif type(wrapped).__name__ == 'ndarray':
        import numpy as np
        for idx in np.ndindex(wrapped.shape):
            sub = name + ''.join(f'[{i}]' for i in idx)
            _collect_keys(wrapped[idx], sub, names)

_OP_SYMBOLS = {
    'add': '+', 'sub': '-', 'mul': '*', 'truediv': '/', 'floordiv': '//',
    'mod': '%', 'pow': '**', 'matmul': '@', 'lshift': '<<', 'rshift': '>>',
    'and_': '&', 'or_': '|', 'xor': '^', 'eq': '==', 'ne': '!=',
    'lt': '<', 'le': '<=', 'gt': '>', 'ge': '>=', 'neg': '-', 'pos': '+',
    'abs': 'abs', 'invert': '~',
}

def trace_ir(func, args):
    """Replay an IR step-by-step with variable names and the compacted logical stack state."""
    import inspect
    ctx = _Context()
    wrapped_args = tuple(_wrap(ctx, a, deferred=True) for a in args)
    res = func(*wrapped_args)

    _pass2(ctx, res)
    ir = ctx.ir
    last_use = ctx.last_use

    # Re-calculate a mapped local `last_use` map dict to tie directly to `ir` iteration formatting
    last_use_ir = {}
    for i, ev in enumerate(ir):
        if ev[0] == 'STORE':
            if ev[1] not in last_use_ir:
                last_use_ir[ev[1]] = i
        elif ev[0] == 'READ':
            last_use_ir[ev[1]] = i
        elif ev[0] == 'OP':
            for k in ev[2]:
                last_use_ir[k] = i

    names = {}
    def collect_keys_val(val):
        if isinstance(val, _Tracked):
            names[val._key] = True
        elif isinstance(val, (list, tuple)):
            for v in val: collect_keys_val(v)
        elif type(val).__name__ == 'ndarray':
            import numpy as np
            for v in val.flat: collect_keys_val(v)
    collect_keys_val(res)
    for k in names:
        last_use_ir[k] = len(ir)

    names = {}
    try:
        param_names = list(inspect.signature(func).parameters.keys())
    except (ValueError, TypeError):
        param_names = [f'arg{i}' for i in range(len(args))]
    for pname, warg in zip(param_names, wrapped_args):
        _collect_keys(warg, pname, names)

    def n(key):
        return names.get(key, f'v{key}')

    stack = []
    out = []
    total = 0

    def fmt_stack():
        return '[' + ', '.join(n(k) for k in stack) + ']'

    def compact(current_idx):
        new_stack = []
        for k in stack:
            if k in last_use_ir and last_use_ir[k] <= current_idx:
                pass # Vaporize on death
            else:
                new_stack.append(k)
        stack[:] = new_stack

    for i, ev in enumerate(ir):
        tag = ev[0]
        if tag == 'STORE':
            key = ev[1]
            stack.append(key)
            compact(i)
            out.append(f"STORE {n(key):<20}              stack={fmt_stack()}")
        elif tag == 'READ':
            key, depth = ev[1], ev[2]
            cost = math.isqrt(depth - 1) + 1
            if key not in stack:
                stack.append(key)
            stack.remove(key)
            stack.append(key)

            is_last_in_read_block = True
            if i + 1 < len(ir) and ir[i+1][0] in ('READ', 'OP'):
                is_last_in_read_block = False

            out.append(f"  READ {n(key)}@{depth:<3} cost={cost:<3}          stack={fmt_stack()}")

            if is_last_in_read_block:
                compact(i)
        elif tag == 'OP':
            _, opname, keys, depths, out_key = ev
            total += sum(math.isqrt(d - 1) + 1 for d in depths)
            sym = _OP_SYMBOLS.get(opname, opname)
            if out_key is not None:
                if len(keys) == 2:
                    names[out_key] = f"({n(keys[0])}{sym}{n(keys[1])})"
                elif len(keys) == 1:
                    names[out_key] = f"{sym}({n(keys[0])})"
                else:
                    names[out_key] = f"{opname}({', '.join(n(k) for k in keys)})"

            # The operation natively completes read requirements, enabling stack clearance
            compact(i)

    out.append(f"# total cost = {total}")
    result = "\n".join(out)
    print(result)
    return result

def bytedmd(func, args, bytes_per_element=1):
    trace, _ = traced_eval(func, args)
    return trace_to_bytedmd(trace, bytes_per_element)
