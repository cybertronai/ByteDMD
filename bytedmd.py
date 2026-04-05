import math
import operator

def assert_noescape(func, args):
    """
    Performs a 'dry-run' of func(args) in a strict tracking sandbox.
    Asserts if any data movement escapes the ByteDMD tracking model.
    """
    def _block(msg):
        def method(self, *a, **kw): assert False, msg
        return method

    class StrictTracked:
        __slots__ = ('_val',) 
        def __init__(self, val): 
            object.__setattr__(self, '_val', val)

        def __getattribute__(self, name):
            # 1. Block direct extraction of tracked state
            if name in ('val', '_ctx', '_key', 'trace'):
                assert False, f"ByteDMD Leak: Direct access to internal attribute '{name}' bypasses tracking."
            try: 
                return object.__getattribute__(self, name)
            except AttributeError:
                # 6. Block untracked object methods like a.conjugate()
                assert False, f"ByteDMD Leak: Untracked object method or attribute '{name}' accessed."

        # 2 & 3. Block implicit unboxing to primitives & C-extensions (math, etc.)
        __int__ = _block("ByteDMD Leak: Primitive unboxing via __int__ (e.g., int() or math module).")
        __float__ = _block("ByteDMD Leak: Primitive unboxing via __float__ (e.g., float() or C-extensions).")
        __bool__ = _block("ByteDMD Leak: Primitive unboxing via __bool__ (e.g., 'if' statements bypass tracking).")
        __index__ = _block("ByteDMD Leak: Primitive unboxing via __index__.")
        __complex__ = _block("ByteDMD Leak: Primitive unboxing via __complex__.")
        __iter__ = _block("ByteDMD Leak: Implicit iteration unboxing.")
        __array__ = _block("ByteDMD Leak: Implicit conversion to NumPy array.")
        __array_ufunc__ = _block("ByteDMD Leak: NumPy ufunc execution unboxes values.")
        
        # 4. Block string coercion bypassing cost checks
        __str__ = _block("ByteDMD Leak: Untracked string coercion via __str__.")
        __repr__ = _block("ByteDMD Leak: Untracked string coercion via __repr__.")
        __format__ = _block("ByteDMD Leak: Untracked string coercion via f-string.")
        __hash__ = _block("ByteDMD Leak: Hashing unboxes value (e.g., dict keys).")

    def _make_strict_op(op, rev=False):
        def method(self, *args_op):
            vals = [object.__getattribute__(a, '_val') if isinstance(a, StrictTracked) else a for a in args_op]
            my_val = object.__getattribute__(self, '_val')
            try:
                res = op(vals[0], my_val, *vals[1:]) if rev else op(my_val, *vals)
            except AssertionError:
                raise
            except Exception as e:
                # 5. Block Exception short-circuiting that bypasses trace logging
                assert False, f"ByteDMD Leak: Exception {type(e).__name__} during math allows short-circuiting read costs."
            return res if res is NotImplemented else _strict_wrap(res)
        return method

    _STRICT_OPS = {
        **{k: getattr(operator, k) for k in 'add sub mul truediv floordiv mod lshift rshift xor matmul neg pos abs invert eq ne lt le gt ge'.split()},
        'and': operator.and_, 'or': operator.or_, 'divmod': divmod, 'pow': pow,
        'trunc': math.trunc, 'ceil': math.ceil, 'floor': math.floor, 'round': round
    }

    for n, f in _STRICT_OPS.items():
        setattr(StrictTracked, f'__{n}__', _make_strict_op(f))
        if n in 'add sub mul truediv floordiv mod divmod pow lshift rshift and xor or matmul'.split():
            setattr(StrictTracked, f'__r{n}__', _make_strict_op(f, rev=True))

    class StrictList(list):
        # 6. Block container mutations
        def _mut(self, *a, **kw): assert False, "ByteDMD Leak: Container mutation can insert untracked primitives."
        append = extend = insert = remove = pop = clear = __setitem__ = __delitem__ = __iadd__ = _mut

    def _strict_wrap(val, memo=None):
        if memo is None: memo = {}
        if isinstance(val, StrictTracked): return val
        vid = id(val)
        if vid in memo: return memo[vid]
        
        is_prim = type(val) in (int, float, bool, complex, str)
        if isinstance(val, dict):
            assert False, "ByteDMD Leak: Dictionaries are skipped by wrap, hiding contents inside an opaque proxy."
            
        if type(val).__name__ == 'ndarray':
            import numpy as np
            res = np.empty_like(val, dtype=object)
            if not is_prim: memo[vid] = res
            for idx in np.ndindex(val.shape):
                v = val[idx]
                res[idx] = _strict_wrap(v.item() if hasattr(v, 'item') and not isinstance(v, np.ndarray) else v, memo)
            res.flags.writeable = False # Hardware-lock numpy arrays to prevent mutation
            return res

        if isinstance(val, list):
            res = StrictList()
            if not is_prim: memo[vid] = res
            list.extend(res, (_strict_wrap(v, memo) for v in val))
            return res

        if isinstance(val, tuple):
            res = tuple(_strict_wrap(v, memo) for v in val)
            if not is_prim: memo[vid] = res
            return res

        res = StrictTracked(val)
        if not is_prim: memo[vid] = res
        return res

    try:
        func(*[_strict_wrap(a) for a in args])
    except AssertionError:
        raise
    except ValueError as e:
        if "read-only" in str(e):
            assert False, "ByteDMD Leak: Array mutation can insert untracked primitives."
    except Exception:
        pass # Ignore standard unhandled native Exceptions, let them naturally fail in the main tracking run


class _Context:
    __slots__ = ('stack', 'trace', 'sync', 'memo', 'counter')
    def __init__(self):
        self.stack, self.trace, self.sync, self.memo, self.counter = [], [], [], {}, 0

    def allocate(self):
        self.counter += 1
        self.stack.append(self.counter)
        return self.counter

    def read(self, keys):
        valid = [k for k in keys if k is not None]
        if not valid: return
        self.trace.extend(len(self.stack) - self.stack.index(k) for k in valid)
        for k in valid:
            self.stack.remove(k)
            self.stack.append(k)


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
    def method(self, *args):
        keys = [a._key if isinstance(a, _Tracked) else None for a in args]
        vals = [a.val if isinstance(a, _Tracked) else a for a in args]

        read_keys = [keys[0], self._key] + keys[1:] if rev else [self._key] + keys
        res = op(vals[0], self.val, *vals[1:]) if rev else op(self.val, *vals)
        self._ctx.read(read_keys)

        return res if res is NotImplemented else _wrap(self._ctx, res)
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


def _wrap(ctx, val):
    if isinstance(val, _Tracked): return val
    vid = id(val)
    if vid in ctx.memo: return ctx.memo[vid]

    is_prim = type(val) in (int, float, bool, complex, str)
    if type(val).__name__ == 'ndarray':
        import numpy as np
        res = np.empty_like(val, dtype=object)
        if not is_prim:
            ctx.memo[vid] = res
            ctx.sync.append((val, res))
        for idx in np.ndindex(val.shape):
            v = val[idx]
            res[idx] = _wrap(ctx, v.item() if hasattr(v, 'item') and not isinstance(v, np.ndarray) else v)
        return res

    if isinstance(val, list):
        res = []
        if not is_prim:
            ctx.memo[vid] = res
            ctx.sync.append((val, res))
        res.extend(_wrap(ctx, v) for v in val)
        return res

    if isinstance(val, tuple):
        res = tuple(_wrap(ctx, v) for v in val)
        if not is_prim: ctx.memo[vid] = res
        return res

    res = _Tracked(ctx, ctx.allocate(), val)
    if not is_prim: ctx.memo[vid] = res
    return res


def _unwrap(val, memo=None):
    if memo is None: memo = {}
    vid = id(val)
    if vid in memo: return memo[vid]

    is_prim = type(val) in (int, float, bool, complex, str)
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


def _sum_usqrt(N):
    if N <= 0: return 0
    M = math.isqrt(N - 1) + 1
    return M * (6 * N - 2 * M * M + 3 * M - 1) // 6


def traced_eval(func, args):
    """Run func with tracked arguments. Returns (trace, result)."""
    ctx = _Context()
    res = func(*(_wrap(ctx, a) for a in args))
    memo = {}
    for orig, wrapped in ctx.sync:
        if isinstance(orig, list): orig[:] = _unwrap(wrapped, memo)
        elif type(orig).__name__ == 'ndarray': orig[...] = _unwrap(wrapped, memo)
    return ctx.trace, _unwrap(res, memo)


def trace_to_bytedmd(trace, bytes_per_element):
    """Convert a trace (list of element depths) to ByteDMD cost."""
    if bytes_per_element == 1: return sum(math.isqrt(d - 1) + 1 for d in trace)
    bpe = bytes_per_element
    return sum(_sum_usqrt(d * bpe) - _sum_usqrt((d - 1) * bpe) for d in trace)


def bytedmd(func, args, bytes_per_element=1):
    """Evaluate ByteDMD cost of running func with args."""
    assert_noescape(func, args)
    trace, _ = traced_eval(func, args)
    return trace_to_bytedmd(trace, bytes_per_element)