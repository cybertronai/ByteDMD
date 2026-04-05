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


def test_constant_ops_are_tracked():
    """
    Constants are now tracked on the LRU stack via const_keys cache.
    `a * 10` reads [a, const_10], then `10 - result` reads [const_10 (cached), result].
    """
    def f(a):
        return 10 - a * 10

    trace, _ = traced_eval(f, (5,))
    assert trace == [2, 1, 2, 1]


def test_pure_memory_movement_is_now_traced():
    """
    _TrackedList intercepts __getitem__ and __iter__, so pure list indexing now generates traces.
    """
    def transpose(A):
        n = len(A)
        return [[A[j][i] for j in range(n)] for i in range(n)]

    A = [[1, 2], [3, 4]]
    trace, result = traced_eval(transpose, (A,))

    assert trace == [2, 2, 2, 2, 4, 3, 4, 4, 4, 3, 4, 4, 4, 4, 6, 6]
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


def test_comparison_results_are_now_tracked():
    """
    Comparisons now return _TrackedValue instead of raw booleans.
    The comparison result stays on the LRU stack and is properly tracked when used later.
    """
    def compare_and_use(a, b):
        c = a > b  # c is now a _TrackedValue wrapping True
        return c + a  # both c and a are tracked

    trace, result = traced_eval(compare_and_use, (5, 3))

    # Trace logic:
    # 1. `a > b` reads [a, b] -> trace [2, 1]
    # 2. __bool__ on result for truthiness check in + doesn't happen here
    # 3. `c + a` reads [c, a] -> trace [1, 3]
    assert trace == [2, 1, 1, 3]

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


def test_numpy_outputs_are_properly_unwrapped():
    """
    _unwrap now handles ndarray with object dtype, properly unwrapping to native arrays.
    """
    trace, result = traced_eval(
        lambda a, b: np.add(a, b),
        (np.array([1, 2]), np.array([3, 4])),
    )
    assert isinstance(result, np.ndarray)
    assert result.dtype != object
    np.testing.assert_array_equal(result, [4, 6])


def test_argument_aliasing_and_mutation_are_preserved():
    """
    _wrap now preserves aliasing via memo dict, and mutations are synced back to originals.
    """
    def mutate_alias(a, b):
        a[0] = 7
        return b[0]

    raw = [1]
    assert mutate_alias(raw, raw) == 7
    assert raw == [7]

    wrapped = [1]
    trace, result = traced_eval(mutate_alias, (wrapped, wrapped))
    # _TrackedList.__getitem__ reads the element, __setitem__ wraps the new value
    assert trace == [1, 1]
    assert result == 7
    assert wrapped == [7]


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
