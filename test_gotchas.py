#!/usr/bin/env python3
"""
Python gotchas: Because we implement ByteDMD in Python by wrapping Python objects,
our Python framework deviates from idealized description in the README
for certain cases, illustrated by tests below.

Future variants of ByteDMD metric may choose to implement this behavior differently.
"""
import math
import numpy as np
import pytest

from bytedmd import traced_eval


def test_gotcha_constant_ops():
    """
    Limitation of Python model: constants are not tracked
    """
    def f(a):
        return 10 - a * 10

    trace, _ = traced_eval(f, (5,))
    assert trace == [1, 1]


def test_gotcha_pure_memory_movement_is_free_trace():
    """
    Limitation of Python model: pure list index without computation does not trigger math magic methods,
    hence generating no read trace.
    """
    def transpose(A):
        n = len(A)
        return [[A[j][i] for j in range(n)] for i in range(n)]

    A = [[1, 2], [3, 4]]
    trace, result = traced_eval(transpose, (A,))

    assert trace == []
    assert result == [[1, 3], [2, 4]]

def test_gotcha_implicit_boolean_bypass_trace():
    """
    WEAKNESS 1: Implicit truthiness evaluation bypasses the trace.
    Python evaluates an object's truthiness using `__bool__`. Because `_TrackedValue`
    does not override it, Python treats ALL wrapped values as `True` (since they are objects).
    This silently executes the wrong branch AND completely bypasses the read trace.
    """
    def implicit_branch(a):
        if a:  # Wrongly evaluates to True for a=0, generates NO trace!
            return a + 10
        return a

    def explicit_branch(a):
        if a != 0:  # Correctly evaluates to False for a=0, generates trace!
            return a + 10
        return a

    trace_implicit, result_implicit = traced_eval(implicit_branch, (0,))
    # Misses the condition check entirely, only tracking the `a + 10` execution!
    assert trace_implicit == [1]
    assert result_implicit == 10

    trace_explicit, result_explicit = traced_eval(explicit_branch, (0,))
    # Correctly traces the explicit `a != 0` check
    assert trace_explicit == [1]
    assert result_explicit == 0

def test_gotcha_short_circuit_logic_bypass_trace():
    """
    WEAKNESS 2: Python's `and` / `or` keywords do not invoke magic methods.
    They rely on implicit truthiness, completely circumventing read tracing and
    returning incorrect mathematical results.
    """
    def logical_and(a, b):
        return a and b

    trace, result = traced_eval(logical_and, (0, 5))
    assert trace == []
    assert result == 5

def test_gotcha_not_uses_implicit_truthiness():
    """
    Missing from current gotchas: `not a` also bypasses tracing because `_TrackedValue`
    does not implement `__bool__`.
    """
    trace, result = traced_eval(lambda a: not a, (0,))
    assert trace == []
    assert result is False  # raw Python would return True


def test_gotcha_or_uses_implicit_truthiness():
    """Current suite checks `and`, but `or` is wrong for the same reason."""
    trace, result = traced_eval(lambda a, b: a or b, (0, 5))
    assert trace == []
    assert result == 0  # raw Python would return 5


def test_gotcha_comparison_untracking_trace():
    """
    WEAKNESS 3: Native Booleans escape the LRU Stack.
    To allow standard Python control flow (`if a > b:`), comparison operators
    intentionally evaluate directly to native Python booleans. If an algorithm
    subsequently uses these booleans mathematically, they are completely untracked.
    """
    def compare_and_use(a, b):
        c = a > b  # 'c' escapes the tracker and becomes a raw Python boolean (True)
        return c + a # 'c' is mathematically used, but its read generates no cost!

    trace, result = traced_eval(compare_and_use, (5, 3))

    # Trace logic:
    # 1. `a > b` triggers trace [2, 1] (a is depth 2, b is depth 1)
    # 2. `c + a` reads 'a' at depth 2 ('c' generates NO trace because it is a raw boolean)
    assert trace == [2, 1, 2]

    # Result is 6 (True + 5)
    assert result == 6


def test_gotcha_missing_index_protocol_breaks_range_and_indexing():
    """
    `_TrackedValue` has no `__index__`, so algorithms using input-derived loop bounds or
    indices fail instead of being traced.
    """
    with pytest.raises(TypeError):
        traced_eval(lambda n: [i for i in range(n)], (3,))

    with pytest.raises(TypeError):
        traced_eval(lambda xs, i: xs[i], ([10, 20, 30], 1))


def test_gotcha_math_module_functions_are_not_supported():
    """
    Only a small set of numeric magic methods are implemented; `math.*` calls that expect
    real numbers fail on `_TrackedValue`.
    """
    with pytest.raises(TypeError):
        traced_eval(lambda a: math.sqrt(a), (4,))


def test_gotcha_tuple_inputs_are_coerced_to_lists():
    """
    `_wrap` recursively converts tuples/ndarrays into Python lists, so tuple-specific
    behavior is lost.
    """
    trace, result = traced_eval(lambda t: isinstance(t, tuple), ((1, 2),))
    assert trace == []
    assert result is False


def test_gotcha_numpy_outputs_are_not_fully_unwrapped():
    """
    `_unwrap` handles only lists/tuples/_TrackedValue, so ndarray outputs can leak wrapped
    scalars back to the caller.
    """
    trace, result = traced_eval(
        lambda a, b: np.add(a, b),
        (np.array([1, 2]), np.array([3, 4])),
    )
    assert isinstance(result, np.ndarray)
    assert result.dtype == object
    assert [x.value for x in result.tolist()] == [4, 6]


def test_gotcha_argument_aliasing_and_mutation_are_not_preserved():
    """
    `_wrap` rebuilds containers recursively, so shared-object aliasing disappears and
    in-place mutation does not affect the caller's objects.
    """
    def mutate_alias(a, b):
        a[0] = 7
        return b[0]

    raw = [1]
    assert mutate_alias(raw, raw) == 7
    assert raw == [7]

    wrapped = [1]
    trace, result = traced_eval(mutate_alias, (wrapped, wrapped))
    assert trace == []
    assert result == 1
    assert wrapped == [1]


@pytest.mark.xfail(reason="_make_method ignores the 3rd argument to __pow__")
def test_pow_with_modulo_should_be_traced_and_match_python():
    """
    This is a real semantic bug, not just a gotcha: `pow(a, b, m)` currently ignores `m`,
    returns the wrong value, and never charges a read for the modulo operand.
    """
    trace, result = traced_eval(lambda a, b, m: pow(a, b, m), (2, 5, 3))
    assert trace == [3, 2, 1]
    assert result == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
