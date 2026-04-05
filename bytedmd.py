#!/usr/bin/env python3
"""
Implements ByteDMD cost model robustly resolving Python architectural Gotchas.
"""

import math
import operator

class _TrackedContext:
    """Shared LRU context used during traced execution."""
    __slots__ = ('stack', 'trace', '_counter', 'memo', 'sync', 'const_keys')

    def __init__(self):
        self.stack = []     
        self.trace = []     
        self._counter = 0
        self.memo = {}          # Preserves topology referencing (aliasing)
        self.sync = []          # Maps proxies back to originals for mutation sync
        self.const_keys = {}    # Caches literal primitives 

    def allocate(self):
        """Allocate a new entry at the top of the stack and return its key."""
        self._counter += 1
        self.stack.append(self._counter)
        return self._counter

    def read_all_then_move(self, keys):
        """Read all operands at current distances, then sequentially move all to top."""
        valid_keys = [k for k in keys if k is not None]
        if not valid_keys:
            return
            
        n = len(self.stack)
        self.trace.extend(n - self.stack.index(k) for k in valid_keys)
        
        for k in valid_keys:
            self.stack.remove(k)
            self.stack.append(k)


class _TrackedValue:
    """A scalar value that records read operations on a shared LRU stack."""
    __slots__ = ('_ctx', '_key', 'value')

    def __init__(self, ctx, key, value):
        self._ctx = ctx
        self._key = key
        self.value = value
        
    def __str__(self): return str(self.value)
    def __repr__(self): return f"Tracked({self.value})"

    # Native control flow triggers. Logs the read locally avoiding premature untracking 
    def __bool__(self):
        self._ctx.read_all_then_move([self._key])
        return bool(self.value)
        
    def __int__(self):
        self._ctx.read_all_then_move([self._key])
        return int(self.value)
        
    def __float__(self):
        self._ctx.read_all_then_move([self._key])
        return float(self.value)
        
    def __complex__(self):
        self._ctx.read_all_then_move([self._key])
        return complex(self.value)
        
    def __index__(self):
        self._ctx.read_all_then_move([self._key])
        return operator.index(self.value)
        
    def __hash__(self):
        self._ctx.read_all_then_move([self._key])
        return hash(self.value)


def _make_method(op_func, is_rev=False):
    """Generic factory evaluating bin/unary/cmp operations dynamically parsing *args boundaries."""
    def method(self, *args):
        keys, vals = [], []
        for a in args:
            if isinstance(a, _TrackedValue):
                keys.append(a._key)
                vals.append(a.value)
            elif type(a) in (int, float, bool, complex, str):
                # Address Constants Gotcha: Inject literals implicitly as cached Immediate Loads
                if a not in self._ctx.const_keys:
                    self._ctx.const_keys[a] = self._ctx.allocate()
                keys.append(self._ctx.const_keys[a])
                vals.append(a)
            else:
                keys.append(None)
                vals.append(a)
                
        # Appropriately re-order the evaluation based on sequence (for e.g. __radd__) 
        if is_rev:
            read_keys = [keys[0], self._key] + keys[1:]
            res = op_func(vals[0], self.value, *vals[1:])
        else:
            read_keys = [self._key] + keys
            res = op_func(self.value, *vals)
            
        self._ctx.read_all_then_move(read_keys)
            
        if res is NotImplemented:
            return res
        return _wrap(self._ctx, res)
    return method


# Map Python magic methods properly linking Variadic and Binary implementations
_OPS = {
    **{k: getattr(operator, k) for k in 'add sub mul truediv floordiv mod lshift rshift xor matmul neg pos abs invert eq ne lt le gt ge'.split()},
    'and': operator.and_, 'or': operator.or_, 'divmod': divmod, 'pow': pow,
    'trunc': math.trunc, 'ceil': math.ceil, 'floor': math.floor, 'round': round
}
_BIN = {'add', 'sub', 'mul', 'truediv', 'floordiv', 'mod', 'divmod', 'pow', 'lshift', 'rshift', 'and', 'xor', 'or', 'matmul'}

for _name, _func in _OPS.items():
    setattr(_TrackedValue, f'__{_name}__', _make_method(_func))
    if _name in _BIN:
        setattr(_TrackedValue, f'__r{_name}__', _make_method(_func, is_rev=True))


class _TrackedList(list):
    """List proxy intercepting pure memory migration and actively preserving aliased mutations."""
    __slots__ = ('_ctx',)
    def __init__(self, ctx, iterable):
        super().__init__(iterable)
        self._ctx = ctx

    def __getitem__(self, key):
        item = super().__getitem__(key)
        if isinstance(key, slice):
            res = _TrackedList(self._ctx, [])
            res.extend(item)
            return res
        if isinstance(item, _TrackedValue):
            self._ctx.read_all_then_move([item._key])
        return item

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            super().__setitem__(key, [_wrap(self._ctx, v) for v in value])
        else:
            super().__setitem__(key, _wrap(self._ctx, value))

    def __iter__(self):
        for item in super().__iter__():
            if isinstance(item, _TrackedValue):
                self._ctx.read_all_then_move([item._key])
            yield item

    def append(self, value):
        super().append(_wrap(self._ctx, value))

    def extend(self, iterable):
        super().extend(_wrap(self._ctx, v) for v in iterable)

    def insert(self, index, value):
        super().insert(index, _wrap(self._ctx, value))

    def pop(self, index=-1):
        item = super().pop(index)
        if isinstance(item, _TrackedValue):
            self._ctx.read_all_then_move([item._key])
        return item


class _TrackedTuple(tuple):
    """Safely preserves read logic bounds across pure structural tuple boundaries."""
    def __new__(cls, ctx, iterable):
        obj = super().__new__(cls, iterable)
        obj._ctx = ctx
        return obj

    def __getitem__(self, key):
        item = super().__getitem__(key)
        if isinstance(key, slice):
            return _TrackedTuple(self._ctx, item)
        if isinstance(item, _TrackedValue):
            self._ctx.read_all_then_move([item._key])
        return item

    def __iter__(self):
        for item in super().__iter__():
            if isinstance(item, _TrackedValue):
                self._ctx.read_all_then_move([item._key])
            yield item


def _wrap(ctx, val):
    """Recursively convert primitives/arrays mapping aliasing pointer constraints securely."""
    if isinstance(val, _TrackedValue): return val

    # Avoid aggressively caching unique primitives (int cache protection limits)
    is_prim = type(val) in (int, float, bool, complex, str)
    if not is_prim:
        vid = id(val)
        if vid in ctx.memo: return ctx.memo[vid]

    if type(val).__name__ == 'ndarray':
        import numpy as np
        res = np.empty_like(val, dtype=object)
        if not is_prim:
            ctx.memo[vid] = res
            ctx.sync.append((val, res))
        for idx in np.ndindex(val.shape):
            v = val[idx]
            # Safely cast standard numpy scalars so primitives do not cause massive memory footprints
            if hasattr(v, 'item') and not isinstance(v, np.ndarray):
                v = v.item()
            res[idx] = _wrap(ctx, v)
        return res

    if isinstance(val, list):
        res = _TrackedList(ctx, [])
        if not is_prim:
            ctx.memo[vid] = res
            ctx.sync.append((val, res))
        res.extend(_wrap(ctx, v) for v in val)
        return res

    if isinstance(val, tuple):
        wrapped_items = tuple(_wrap(ctx, v) for v in val)
        res = _TrackedTuple(ctx, wrapped_items)
        if not is_prim: ctx.memo[vid] = res
        return res

    res = _TrackedValue(ctx, ctx.allocate(), val)
    if not is_prim: ctx.memo[vid] = res
    return res


def _unwrap(val, memo=None):
    """Safely flatten nested Tracked primitives without array leakage limits."""
    if memo is None: memo = {}

    is_prim = type(val) in (int, float, bool, complex, str)
    if not is_prim:
        vid = id(val)
        if vid in memo: return memo[vid]

    if isinstance(val, list):
        res = []
        if not is_prim: memo[vid] = res
        res.extend(_unwrap(v, memo) for v in val)
        return res

    if isinstance(val, tuple):
        res = tuple(_unwrap(v, memo) for v in val)
        if not is_prim: memo[vid] = res
        return res

    if type(val).__name__ == 'ndarray':
        import numpy as np
        if val.dtype == object:
            flat = [_unwrap(x, memo) for x in val.flat]
            try:
                res = np.array(flat).reshape(val.shape)
            except Exception:
                res = np.empty_like(val, dtype=object)
                for i, x in enumerate(flat):
                    res.flat[i] = x
        else:
            res = val
        if not is_prim: memo[vid] = res
        return res

    res = getattr(val, 'value', val)
    if not is_prim: memo[vid] = res
    return res


def traced_eval(func, args):
    """Run func with internally aliased tracked arguments. Returns (trace, result)."""
    ctx = _TrackedContext()
    wrapped_args = tuple(_wrap(ctx, val) for val in args)
    
    result = func(*wrapped_args)
    
    # Safely synchronize complex mutations natively back to original container instances for caller
    unwrap_memo = {}
    for orig, wrapped in ctx.sync:
        if isinstance(orig, list):
            orig[:] = _unwrap(wrapped, unwrap_memo)
        elif type(orig).__name__ == 'ndarray':
            orig[...] = _unwrap(wrapped, unwrap_memo)
            
    return ctx.trace, _unwrap(result, unwrap_memo)


def _sum_usqrt(N):
    """Closed-form mathematically exact sum of ceil(sqrt(k)) computationally optimized to O(1)."""
    if N <= 0:
        return 0
    M = math.isqrt(N - 1) + 1
    # Bypasses definite step-summation limits
    return M * (6 * N - 2 * M * M + 3 * M - 1) // 6


def trace_to_bytedmd(trace, bytes_per_element):
    """Convert a trace (list of element depths) to ByteDMD cost."""
    # Instanced short circuit bypass explicitly dropping nested subtraction iterations 
    if bytes_per_element == 1:
        return sum(math.isqrt(d - 1) + 1 for d in trace)
        
    bpe = bytes_per_element
    return sum(_sum_usqrt(d * bpe) - _sum_usqrt((d - 1) * bpe) for d in trace)


def bytedmd(func, args, bytes_per_element=1):
    """Evaluate ByteDMD cost of running func with args."""
    trace, _ = traced_eval(func, args)
    return trace_to_bytedmd(trace, bytes_per_element)
