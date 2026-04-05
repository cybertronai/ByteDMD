#!/usr/bin/env python3
"""
Python gotchas: Because we implement ByteDMD in Python by wrapping Python objects,
our Python framework deviates from idealized description in the README
for certain cases, illustrated by tests below.

Future release of ByteDMD metric may fix this to be more faithful to the README.md
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



def test_short_circuit_gotcha():
    """
    Python short-circuit operation means that we may only have 1 operand that is traced rather than both. 
    """
    def logical_and(a, b):
        return a and b

    trace, result = traced_eval(logical_and, (0, 5))
    assert trace == [2]
    assert result == 0

    trace, result = traced_eval(lambda a, b: a or b, (0, 5))
    assert trace == [2]
    assert result == 5


def test_gotcha_comparison_untracking_trace():
    """
    Native Booleans escape the LRU Stack.
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




def test_math_module_functions_work():
    """
    __float__ is now implemented, so math.* functions that call float() work correctly.
    The trace records the read but the result is an unwrapped float (escapes tracking).
    """
    trace, result = traced_eval(lambda a: math.sqrt(a), (4,))
    assert trace == [1]
    assert result == 2.0


def test_tuple_inputs_are_preserved():
    """
    _wrap now preserves tuples as tuples instead of converting to lists.
    """
    trace, result = traced_eval(lambda t: isinstance(t, tuple), ((1, 2),))
    assert trace == []
    assert result is True


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
