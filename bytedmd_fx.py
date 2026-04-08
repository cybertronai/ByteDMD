#!/usr/bin/env python3
"""
Strict ByteDMD tracer — rejects functions with UNTRACKED COMPUTATION.

This module mirrors bytedmd_fx.py but adds a strict validation pass that
detects any computation which bypasses the ByteDMD cost model.

The strict tracer DOES allow data-dependent behavior:
  - if/while branching on tracked values is fine (whichever branch executes
    has its operations counted correctly)
  - indexing with tracked values (range(proxy), list[proxy]) is fine
  - dictionary lookups with tracked keys are fine
  - the cost may differ for different inputs — that is OK

The strict tracer REJECTS only operations that perform untracked numeric
computation:
  - __float__  : triggered by C extensions like math.sin(proxy), which
                 perform floating-point math invisibly to the tracer
  - __complex__: same issue with complex C extensions
  - __array__  : numpy operations on tracked values would compute outside
                 the tracer
  - __array_ufunc__: numpy ufunc dispatch

These escapes are what allow someone to "cheat" — to insert computation
that ByteDMD doesn't see and thus gets for free.  Everything else is just
control flow or structural access, which doesn't count as hidden compute.

The contract: every function the strict tracer accepts has every numeric
operation accounted for in its ByteDMD cost.  The cost may vary with input,
but it is never an undercount.

API (same as bytedmd_fx.py plus strict_eval):
    traced_eval(func, args)                    → (FxContext, result)
    get_trace(func, args)                      → (trace_list, result)
    bytedmd(func, args, bytes_per_element=1)   → int
    trace_to_bytedmd(trace, bytes_per_element)  → int
    format_ir(ctx, annotate=True)              → str
"""

import ast
import math
import operator
import inspect
import textwrap

import torch.fx


# ────────────────── AST validator for escape hatch detection ─────────────
# Catches escape hatches that no proxy override can prevent:
#   - try/except blocks (catches tracer assertions, exception side-channels)
#   - is / is not on non-None operands (C-level identity bypass)
#   - id/type/isinstance/issubclass/vars/dir/hasattr/getattr/... (introspection)
#   - str/repr/format/ascii/round/f-strings (stringification leaks)
#   - math.trunc/ceil/floor calls (coercion backdoors)
#   - list literals containing non-None constants (unproxied local data)
#   - access to __class__ attribute
#
# The validator runs before traced_eval executes the function. Lambdas
# (func.__name__ == '<lambda>') are skipped because inspect.getsource returns
# the surrounding line, which makes precise validation unreliable.

_BANNED_BUILTINS = frozenset({
    'id', 'type', 'isinstance', 'issubclass', 'vars', 'dir',
    'hasattr', 'getattr', 'setattr', 'delattr',
    'str', 'repr', 'format', 'ascii', 'round',
})

_BANNED_MATH_FUNCS = frozenset({'trunc', 'ceil', 'floor'})


class _StrictASTValidator(ast.NodeVisitor):
    """Walk a function's AST and collect escape hatch violations."""

    def __init__(self):
        self.violations = []

    def _add(self, msg):
        if msg not in self.violations:
            self.violations.append(msg)

    def visit_Try(self, node):
        self._add(
            "try/except block — can catch tracer assertions or use exception "
            "side-channels (e.g. ZeroDivisionError) to probe values without "
            "invoking proxy methods."
        )
        self.generic_visit(node)

    def visit_TryStar(self, node):  # Python 3.11+
        self._add("try/except* block — same risk as try/except.")
        self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        self._add("except handler — can swallow tracer enforcement assertions.")
        self.generic_visit(node)

    def visit_Compare(self, node):
        operands = [node.left] + list(node.comparators)
        for i, op in enumerate(node.ops):
            if isinstance(op, (ast.Is, ast.IsNot)):
                left = operands[i]
                right = operands[i + 1]
                # Allow `x is None` / `x is not None` (canonical Python idiom)
                if isinstance(left, ast.Constant) and left.value is None:
                    continue
                if isinstance(right, ast.Constant) and right.value is None:
                    continue
                op_name = "'is'" if isinstance(op, ast.Is) else "'is not'"
                self._add(
                    f"{op_name} on non-None operands — evaluates at C-level, "
                    f"bypasses tracing."
                )
                break
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            if node.func.id in _BANNED_BUILTINS:
                self._add(
                    f"call to {node.func.id}() — introspection/coercion "
                    f"escape hatch."
                )
        elif isinstance(node.func, ast.Attribute):
            if (isinstance(node.func.value, ast.Name)
                    and node.func.value.id == 'math'
                    and node.func.attr in _BANNED_MATH_FUNCS):
                self._add(
                    f"call to math.{node.func.attr}() — coercion backdoor "
                    f"that unwraps proxy to a native int."
                )
        self.generic_visit(node)

    def visit_JoinedStr(self, node):
        self._add(
            "f-string — leaks proxy values as native strings via __format__."
        )
        self.generic_visit(node)

    def visit_List(self, node):
        for elt in node.elts:
            if isinstance(elt, ast.Constant) and elt.value is not None:
                self._add(
                    "list literal with non-None constant element — creates "
                    "an untracked data structure that can be iterated for "
                    "free reads."
                )
                break
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if node.attr == '__class__':
            self._add(
                "access to __class__ — bypasses tracing via type introspection."
            )
        self.generic_visit(node)


def _validate_function_strict(func):
    """Parse func source and reject if it contains escape hatches."""
    if getattr(func, '__name__', None) == '<lambda>':
        return  # Skip lambda validation (source context unreliable)
    try:
        src = inspect.getsource(func)
    except (TypeError, OSError):
        return  # Can't get source — skip validation
    src = textwrap.dedent(src)
    try:
        tree = ast.parse(src)
    except (SyntaxError, IndentationError, ValueError):
        return  # Skip if unparseable

    target = None
    for node in ast.walk(tree):
        if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == func.__name__):
            target = node
            break
    if target is None:
        return  # Couldn't locate the specific function — skip

    validator = _StrictASTValidator()
    # Visit only the function body (not the def signature) to avoid flagging
    # default arguments. We walk each statement.
    for stmt in target.body:
        validator.visit(stmt)

    if validator.violations:
        msg = (
            "StrictByteDMD: function contains untracked-computation escape "
            "hatches:\n"
            + "\n".join(f"  - {v}" for v in validator.violations)
            + "\nFix: rewrite the function to avoid these patterns."
        )
        raise AssertionError(msg)



# ─────────────── Named wrappers for escape operations ────────────────────

def to_bool(x):   return bool(x)
def to_int(x):    return int(x)
def to_float(x):  return float(x)
def to_complex(x): return complex(x)
def to_index(x):  return operator.index(x)
def to_hash(x):   return hash(x)


# ──────────────────────────────── Context ────────────────────────────────

class FxContext:
    __slots__ = ('graph', 'stack', 'trace', 'sync', 'memo', 'counter',
                 '_names', '_node_info', '_key_to_name')

    def __init__(self):
        self.graph = torch.fx.Graph()
        self.stack = []
        self.trace = []
        self.sync = []
        self.memo = {}
        self.counter = 0
        self._names = {}
        self._node_info = {}
        self._key_to_name = {}

    def _name(self, prefix):
        c = self._names.get(prefix, 0)
        self._names[prefix] = c + 1
        return f'{prefix}_{c}' if c else prefix

    def allocate(self):
        self.counter += 1
        self.stack.append(self.counter)
        return self.counter

    def read(self, keys, node=None):
        valid = [k for k in keys if k is not None]
        if not valid:
            return
        depths = [len(self.stack) - self.stack.index(k) for k in valid]
        self.trace.extend(depths)
        if node is not None:
            self._node_info[node] = {
                'depths': depths,
                'input_names': [self._key_to_name.get(k, f'?{k}') for k in valid],
            }
        for k in valid:
            self.stack.remove(k)
            self.stack.append(k)


# ──────────────────────────────── Proxy ──────────────────────────────────

class FxProxy:
    """Strict proxy: rejects only operations that allow untracked computation."""
    __slots__ = ('_ctx', '_key', '_node', 'val')

    def __init__(self, ctx, key, node, val):
        self._ctx, self._key, self._node, self.val = ctx, key, node, val

    def __str__(self):
        return str(self.val)

    def __repr__(self):
        return f'FxProxy({self.val})'

    # ── ALLOWED escapes (control flow + structural; the read is tracked) ──
    # These return native Python values so they can be used in if/range/etc.
    # The read is recorded in the trace, so the cost is counted correctly.

    def __bool__(self):
        node = self._ctx.graph.create_node(
            'call_function', to_bool, (self._node,),
            name=self._ctx._name('to_bool'))
        self._ctx.read([self._key], node)
        return bool(self.val)

    def __int__(self):
        node = self._ctx.graph.create_node(
            'call_function', to_int, (self._node,),
            name=self._ctx._name('to_int'))
        self._ctx.read([self._key], node)
        return int(self.val)

    def __index__(self):
        node = self._ctx.graph.create_node(
            'call_function', to_index, (self._node,),
            name=self._ctx._name('to_index'))
        self._ctx.read([self._key], node)
        return operator.index(self.val)

    def __hash__(self):
        node = self._ctx.graph.create_node(
            'call_function', to_hash, (self._node,),
            name=self._ctx._name('to_hash'))
        self._ctx.read([self._key], node)
        return hash(self.val)

    # ── REJECTED escapes (allow untracked numeric computation) ──
    # These are the only paths through which a function can perform
    # arithmetic that ByteDMD does not see.

    def __float__(self):
        raise AssertionError(
            "StrictByteDMD: __float__ on tracked value enables untracked computation.\n"
            "  C-level functions like math.sin/cos/exp/sqrt call __float__\n"
            "  on the proxy, then perform floating-point arithmetic in C\n"
            "  without going through any tracked operation.  The numeric\n"
            "  work is invisible to the ByteDMD cost model.\n"
            "  Fix: avoid C math functions on tracked values.\n"
            "  Use proxy-level arithmetic (a*a, a+b, etc.) instead."
        )

    def __complex__(self):
        raise AssertionError(
            "StrictByteDMD: __complex__ on tracked value enables untracked computation.\n"
            "  C-level complex math functions would perform arithmetic\n"
            "  invisibly to the tracer.\n"
            "  Fix: avoid complex() and complex C extensions on tracked values."
        )

    def __array__(self):
        raise AssertionError(
            "StrictByteDMD: __array__ conversion enables untracked numpy computation.\n"
            "  Once a tracked value enters numpy, all downstream arithmetic\n"
            "  happens inside numpy's C kernels and is invisible to the tracer.\n"
            "  Fix: avoid numpy operations on tracked values."
        )

    def __array_ufunc__(self, *args, **kwargs):
        raise AssertionError(
            "StrictByteDMD: numpy ufunc on tracked value enables untracked computation.\n"
            "  Fix: avoid numpy ufuncs on tracked values."
        )


# ────────────────────── Operator method generation ───────────────────────

def _make_fx_op(op_func, rev=False):
    def method(self, *args):
        ctx = self._ctx
        keys = [a._key if isinstance(a, FxProxy) else None for a in args]
        vals = [a.val if isinstance(a, FxProxy) else a for a in args]
        nodes = [a._node if isinstance(a, FxProxy) else a for a in args]
        read_keys = ([keys[0], self._key] + keys[1:] if rev
                     else [self._key] + keys)
        result_val = (op_func(vals[0], self.val, *vals[1:]) if rev
                      else op_func(self.val, *vals))
        graph_args = ((nodes[0], self._node, *nodes[1:]) if rev
                      else (self._node, *nodes))
        op_name = getattr(op_func, '__name__', str(op_func))
        name = ctx._name(op_name)
        node = ctx.graph.create_node('call_function', op_func, tuple(graph_args),
                                     name=name)
        ctx.read(read_keys, node)
        if result_val is NotImplemented:
            return NotImplemented
        return _wrap_result(ctx, result_val, node)
    return method


_OPS = {
    **{k: getattr(operator, k) for k in
       'add sub mul truediv floordiv mod lshift rshift xor matmul '
       'neg pos abs invert eq ne lt le gt ge'.split()},
    'and': operator.and_, 'or': operator.or_,
    'divmod': divmod, 'pow': pow,
    'trunc': math.trunc, 'ceil': math.ceil, 'floor': math.floor, 'round': round,
}

for _n, _f in _OPS.items():
    setattr(FxProxy, f'__{_n}__', _make_fx_op(_f))
    if _n in ('add sub mul truediv floordiv mod divmod pow '
              'lshift rshift and xor or matmul').split():
        setattr(FxProxy, f'__r{_n}__', _make_fx_op(_f, rev=True))


# ─────────── Defense in depth: block coercion dunders ────────────────────
# math.trunc/ceil/floor/round() and the round() builtin call __trunc__,
# __ceil__, __floor__, __round__ on the proxy.  These would otherwise
# return native Python ints, providing a backdoor that lets the user
# extract a value and perform untracked arithmetic on it.
# These overrides run after the _OPS loop installs the tracked versions.

def _block_coercion(name):
    def method(self, *args):
        raise AssertionError(
            f"StrictByteDMD: __{name}__ on tracked value enables "
            f"untracked computation.\n"
            f"  This dunder is invoked by math.{name}() (or round()) which "
            f"returns a native int that can then be used in arithmetic\n"
            f"  outside the tracer.\n"
            f"  Fix: avoid math.{name}() / round() on tracked values."
        )
    method.__name__ = f'__{name}__'
    return method


for _n in ('trunc', 'ceil', 'floor', 'round'):
    setattr(FxProxy, f'__{_n}__', _block_coercion(_n))


# ──────────────────────── Wrapping / unwrapping ──────────────────────────

def _wrap_arg(ctx, val, name='arg'):
    if isinstance(val, FxProxy):
        return val
    vid = id(val)
    if vid in ctx.memo:
        return ctx.memo[vid]
    is_prim = type(val) in (int, float, bool, complex, str)
    if type(val).__name__ == 'ndarray':
        import numpy as np
        res = np.empty_like(val, dtype=object)
        if not is_prim:
            ctx.memo[vid] = res
            ctx.sync.append((val, res))
        for idx in np.ndindex(val.shape):
            v = val[idx]
            suffix = '_'.join(str(i) for i in idx)
            actual = v.item() if hasattr(v, 'item') and not isinstance(v, np.ndarray) else v
            res[idx] = _wrap_arg(ctx, actual, f'{name}_{suffix}')
        return res
    if isinstance(val, list):
        res = []
        if not is_prim:
            ctx.memo[vid] = res
            ctx.sync.append((val, res))
        for i, v in enumerate(val):
            res.append(_wrap_arg(ctx, v, f'{name}_{i}'))
        return res
    if isinstance(val, tuple):
        items = tuple(_wrap_arg(ctx, v, f'{name}_{i}') for i, v in enumerate(val))
        if not is_prim:
            ctx.memo[vid] = items
        return items
    key = ctx.allocate()
    node = ctx.graph.create_node('placeholder', name, name=name)
    ctx._key_to_name[key] = name
    return FxProxy(ctx, key, node, val)


def _wrap_result(ctx, val, parent_node):
    if isinstance(val, FxProxy):
        return val
    if isinstance(val, tuple):
        items = []
        for i, v in enumerate(val):
            item_name = ctx._name('item')
            item_node = ctx.graph.create_node(
                'call_function', operator.getitem, (parent_node, i),
                name=item_name)
            items.append(_wrap_result(ctx, v, item_node))
        return tuple(items)
    if isinstance(val, list):
        result = []
        for i, v in enumerate(val):
            item_name = ctx._name('item')
            item_node = ctx.graph.create_node(
                'call_function', operator.getitem, (parent_node, i),
                name=item_name)
            result.append(_wrap_result(ctx, v, item_node))
        return result
    key = ctx.allocate()
    ctx._key_to_name[key] = parent_node.name
    return FxProxy(ctx, key, parent_node, val)


def _unwrap(val, memo=None):
    if memo is None:
        memo = {}
    if isinstance(val, type):
        return val
    vid = id(val)
    if vid in memo:
        return memo[vid]
    is_prim = type(val) in (int, float, bool, complex, str)
    if isinstance(val, list):
        res = []
        if not is_prim:
            memo[vid] = res
        res.extend(_unwrap(v, memo) for v in val)
        return res
    if isinstance(val, tuple):
        res = tuple(_unwrap(v, memo) for v in val)
        if not is_prim:
            memo[vid] = res
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
        if not is_prim:
            memo[vid] = res
        return res
    res = getattr(val, 'val', val)
    if not is_prim:
        memo[vid] = res
    return res


# ──────────────────────────── Public API ─────────────────────────────────

def _sum_usqrt(N):
    if N <= 0:
        return 0
    M = math.isqrt(N - 1) + 1
    return M * (6 * N - 2 * M * M + 3 * M - 1) // 6


def traced_eval(func, args):
    """Trace func in strict mode — raises on any escape."""
    _validate_function_strict(func)
    ctx = FxContext()
    try:
        sig = inspect.signature(func)
        param_names = list(sig.parameters.keys())
    except (ValueError, TypeError):
        param_names = [f'arg{i}' for i in range(len(args))]
    wrapped_args = []
    for i, a in enumerate(args):
        name = param_names[i] if i < len(param_names) else f'arg{i}'
        wrapped_args.append(_wrap_arg(ctx, a, name))
    res = func(*wrapped_args)

    def _collect_output(v):
        if isinstance(v, FxProxy):
            return v._node
        if isinstance(v, list):
            return [_collect_output(x) for x in v]
        if isinstance(v, tuple):
            return tuple(_collect_output(x) for x in v)
        return v
    ctx.graph.output(_collect_output(res))

    memo = {}
    for orig, wrapped in ctx.sync:
        if isinstance(orig, list):
            orig[:] = _unwrap(wrapped, memo)
        elif type(orig).__name__ == 'ndarray':
            orig[...] = _unwrap(wrapped, memo)
    return ctx, _unwrap(res, memo)


def get_trace(func, args):
    ctx, result = traced_eval(func, args)
    return ctx.trace, result


def trace_to_bytedmd(trace, bytes_per_element):
    if bytes_per_element == 1:
        return sum(math.isqrt(d - 1) + 1 for d in trace)
    bpe = bytes_per_element
    return sum(_sum_usqrt(d * bpe) - _sum_usqrt((d - 1) * bpe) for d in trace)


def bytedmd(func, args, bytes_per_element=1):
    ctx, _ = traced_eval(func, args)
    return trace_to_bytedmd(ctx.trace, bytes_per_element)


def format_ir(ctx, annotate=True):
    lines = []
    for node in ctx.graph.nodes:
        if node.op == 'placeholder':
            lines.append(f'  %{node.name} = placeholder({node.target})')
        elif node.op == 'call_function':
            target_name = getattr(node.target, '__name__', str(node.target))
            parts = []
            for a in node.args:
                if isinstance(a, torch.fx.Node):
                    parts.append(f'%{a.name}')
                else:
                    parts.append(repr(a))
            args_str = ', '.join(parts)
            line = f'  %{node.name} = {target_name}({args_str})'
            if annotate and node in ctx._node_info:
                info = ctx._node_info[node]
                reads = [f'{inp}@{d}' for inp, d in zip(info['input_names'], info['depths'])]
                cost = sum(math.isqrt(d - 1) + 1 for d in info['depths'])
                line += f'    # read {", ".join(reads)} → cost={cost}'
            lines.append(line)
        elif node.op == 'output':
            out = node.args[0]
            if isinstance(out, torch.fx.Node):
                lines.append(f'  return %{out.name}')
            elif isinstance(out, (list, tuple)):
                def _fmt(v):
                    if isinstance(v, torch.fx.Node): return f'%{v.name}'
                    if isinstance(v, (list, tuple)):
                        inner = ', '.join(_fmt(x) for x in v)
                        return f'[{inner}]' if isinstance(v, list) else f'({inner})'
                    return repr(v)
                lines.append(f'  return {_fmt(out)}')
            else:
                lines.append(f'  return {repr(out)}')
    total = trace_to_bytedmd(ctx.trace, 1) if ctx.trace else 0
    lines.append(f'  # Total ByteDMD cost: {total}')
    return '\n'.join(lines)
