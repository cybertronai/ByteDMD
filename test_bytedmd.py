#!/usr/bin/env python3
import numpy as np
from bytedmd import bytedmd, traced_eval, trace_to_bytedmd


def my_add(a, b, c, d):
    return b + c


def test_my_add():
    cost = bytedmd(my_add, (1, 2, 3, 4))
    assert cost == 4

    # trace counts depth in terms of number of elements, not bytes
    trace, _ = traced_eval(my_add, (1, 2, 3, 4))
    assert trace == [3, 2]

    assert trace_to_bytedmd(trace, bytes_per_element=1) == 4
    assert trace_to_bytedmd(trace, bytes_per_element=2) == 10

    assert bytedmd(my_add, (1, 2, 3, 4), bytes_per_element=2) == 10


def my_composite_func(a, b, c, d):
    e = b + c
    f = a + d
    return e > f


def test_repeated_operand_is_charged_twice():
    """README says a+a should charge two reads against the same pre-instruction stack."""
    trace, _ = traced_eval(lambda a: a + a, (5,))
    assert trace == [1, 1]


def test_my_composite_func():
    trace, result = traced_eval(my_composite_func, (1, 2, 3, 4))
    assert trace == [3, 2, 5, 4, 4, 1]
    cost = bytedmd(my_composite_func, (1, 2, 3, 4))
    assert cost == 12

def test_dot_product():
    def dot(a, b):
        return sum(i1 * i2 for i1, i2 in zip(a, b))
        
    a, b = [0, 1], [2, 3]
    trace, result = traced_eval(dot, (a, b))

    assert trace == [4, 2, 1, 6, 5, 4, 1]
    assert result == 3
    assert bytedmd(dot, (a, b)) == 14


def test_branching_and_comparisons_trace():
    def my_relu(a):
        if a > 0:
            return a * 2
        return a
        
    # Branch taken: traces reading `a` twice (`a > 0` and `a * 2`)
    trace_pos, _ = traced_eval(my_relu, (5,))
    assert trace_pos == [1, 1]
    
    # Branch skipped: traces reading `a` once (`a > 0`)
    trace_neg, _ = traced_eval(my_relu, (-5,))
    assert trace_neg == [1]


def test_divmod_tuple_allocation_trace():
    """
    6. Tests operations natively returning multiple tracked values.
    divmod(a, b) evaluates to a tuple (q, r), sequentially triggering 
    multiple allocations on the LRU stack.
    """
    def my_divmod(a, b):
        q, r = divmod(a, b)
        return q + r + a
        
    trace, result = traced_eval(my_divmod, (10, 3))
    assert trace == [2, 1, 2, 1, 1, 5]


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
