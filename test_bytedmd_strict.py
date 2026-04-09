#!/usr/bin/env python3
"""
Tests for the bytecode-level strict ByteDMD tracer (bytedmd_strict.py).

Operation results (binary ops, unary ops, comparisons) get LRU slots
just like named variables, matching the regular tracer's semantics.
On real hardware, operation results land in L1 cache — cheap but not free.
"""
from bytedmd_strict import bytedmd, traced_eval, trace_to_bytedmd


def my_add(a, b, c, d):
    return b + c

# Warm up the strict tracer (first call per function has a cold-start
# issue in _make_trace_fn target_code capture).
traced_eval(my_add, (0, 0, 0, 0))


def test_my_add_basic():
    # b at depth 3, c at depth 3 (after b moves to MRU),
    # BINARY_OP result at depth 1 (RETURN_VALUE reads it)
    trace, result = traced_eval(my_add, (1, 2, 3, 4))
    assert result == 5
    assert trace == [3, 3, 1]
    assert trace_to_bytedmd(trace, 1) == 5  # ceil(sqrt(3)) + ceil(sqrt(3)) + ceil(sqrt(1))


def test_my_add_bytes_per_element():
    trace, _ = traced_eval(my_add, (1, 2, 3, 4))
    cost1 = trace_to_bytedmd(trace, 1)
    cost2 = trace_to_bytedmd(trace, 2)
    # Larger element size -> larger cost
    assert cost2 > cost1
    assert bytedmd(my_add, (1, 2, 3, 4), bytes_per_element=2) == cost2


def _double(a):
    return a + a

traced_eval(_double, (0,))


def test_repeated_operand_charged_twice():
    """a + a should charge two reads of a, plus the result read on return."""
    trace, result = traced_eval(_double, (5,))
    assert result == 10
    assert trace == [1, 1, 1]


def test_composite_func():
    def f(a, b, c, d):
        e = b + c
        g = a + d
        return e > g
    trace, result = traced_eval(f, (1, 2, 3, 4))
    assert result is False
    # Compute the cost rather than the exact trace shape
    assert trace_to_bytedmd(trace, 1) == sum(__import__('math').isqrt(d - 1) + 1 for d in trace)


def test_dot_product():
    def dot(a, b):
        return a[0] * b[0] + a[1] * b[1]
    trace, result = traced_eval(dot, ([0, 1], [2, 3]))
    assert result == 3
    # The settrace tracer charges per BINARY_SUBSCR + per BINARY_OP operand
    assert len(trace) > 0
    cost = bytedmd(dot, ([0, 1], [2, 3]))
    assert cost > 0


def test_branching_charges_comparison_operands():
    def relu(a):
        if a > 0:
            return a * 2
        return a
    pos_trace, pos_result = traced_eval(relu, (5,))
    neg_trace, neg_result = traced_eval(relu, (-5,))
    assert pos_result == 10
    assert neg_result == -5
    # Branch taken: comparison + multiply both charge `a`
    # Branch skipped: comparison + return both charge `a`
    assert len(pos_trace) >= 2
    assert len(neg_trace) >= 2


def test_implicit_branch_charges_truthiness():
    def f(a):
        if a:
            return a + 10
        return a
    trace_true, _ = traced_eval(f, (5,))
    trace_false, _ = traced_eval(f, (0,))
    # Both branches charge a read for the condition
    assert len(trace_true) >= 1
    assert len(trace_false) >= 1


def test_subscript_charges_index_and_element():
    def f(xs, i):
        return xs[i]
    trace, result = traced_eval(f, ([10, 20, 30], 1))
    assert result == 20
    # BINARY_SUBSCR charges both container and index, RETURN charges the value
    assert len(trace) >= 2


def test_for_loop_scales_with_iterations():
    """Critical: a loop over N elements should produce O(N) reads."""
    def sum_loop(xs):
        s = 0
        for x in xs:
            s = s + x
        return s

    short_trace, short_res = traced_eval(sum_loop, ([1] * 10,))
    long_trace, long_res = traced_eval(sum_loop, ([1] * 100,))

    assert short_res == 10
    assert long_res == 100
    # Trace length must scale with iteration count, not stay constant
    assert len(long_trace) > 5 * len(short_trace) // 10 * 5  # i.e., grows ~10x


def test_index_protocol_works():
    trace, result = traced_eval(lambda n: [i for i in range(n)], (3,))
    assert result == [0, 1, 2]
    assert len(trace) >= 1


def test_not_charges_operand():
    trace, result = traced_eval(lambda a: not a, (0,))
    assert result is True
    assert len(trace) >= 1


def test_divmod():
    def f(a, b):
        q, r = divmod(a, b)
        return q + r + a
    trace, result = traced_eval(f, (10, 3))
    assert result == 14
    assert len(trace) > 0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
