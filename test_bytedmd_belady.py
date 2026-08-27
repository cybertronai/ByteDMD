"""Tests for bytedmd_belady — Belady (OPT) stack with separate DRAM cold-miss
pricing and aggressive compaction.

Belady sorts the active cache by next-use time, so variables needed soonest
sit at depth 1. This gives a strict lower bound on data movement cost for
any online cache policy.
"""

import pytest

import bytedmd
import bytedmd_belady as bb


# ---------------------------------------------------------------------------
# Simple scalar functions
# ---------------------------------------------------------------------------

def test_simple_add():
    """a + b: cold misses #1 and #2 on the DRAM tape."""
    def f(a, b):
        return a + b
    trace, result = bb.traced_eval(f, (1, 2))
    assert trace == [1, 2]
    assert result == 3
    assert bb.bytedmd(f, (1, 2)) == 3


def test_my_add_bc():
    """f(a,b,c,d) = b+c — only b and c are accessed."""
    def f(a, b, c, d):
        return b + c
    trace, result = bb.traced_eval(f, (1, 2, 3, 4))
    assert trace == [1, 2]
    assert result == 5
    assert bb.bytedmd(f, (1, 2, 3, 4)) == 3


def test_unused_args_not_charged():
    """Unused arguments don't appear on the DRAM tape."""
    def f4(a, b, c, d):
        return b + c
    def f6(a, b, c, d, e, f):
        return b + c
    assert bb.bytedmd(f4, (1, 2, 3, 4)) == bb.bytedmd(f6, (1, 2, 3, 4, 5, 6))


def test_repeated_operand():
    """a + a: single cold miss, both reads see depth 1."""
    def f(a):
        return a + a
    trace, result = bb.traced_eval(f, (5,))
    assert trace == [1, 1]
    assert result == 10
    assert bb.bytedmd(f, (5,)) == 2


def test_left_associative_chain():
    """(a + b) + c: Belady knows the intermediate is needed immediately."""
    def f(a, b, c):
        return (a + b) + c
    trace, result = bb.traced_eval(f, (1, 2, 3))
    assert trace == [1, 2, 1, 3]
    assert result == 6
    assert bb.bytedmd(f, (1, 2, 3)) == 6


# ---------------------------------------------------------------------------
# Dot product
# ---------------------------------------------------------------------------

def test_dot_product():
    def dot(a, b):
        return sum(i * j for i, j in zip(a, b))
    trace, result = bb.traced_eval(dot, ([1, 2], [3, 4]))
    assert trace == [1, 2, 1, 3, 4, 1, 2]
    assert result == 11
    assert bb.bytedmd(dot, ([1, 2], [3, 4])) == 11


# ---------------------------------------------------------------------------
# Matrix-vector
# ---------------------------------------------------------------------------

def _matvec(A, x):
    n = len(x)
    y = [None] * n
    for i in range(n):
        s = A[i][0] * x[0]
        for j in range(1, n):
            s = s + A[i][j] * x[j]
        y[i] = s
    return y


def test_matvec_2x2_full_trace():
    A = [[1, 2], [3, 4]]
    x = [5, 6]
    trace, result = bb.traced_eval(_matvec, (A, x))
    assert trace == [1, 2, 3, 4, 1, 2, 5, 1, 6, 1, 1, 2]
    assert result == [17, 39]
    assert bb.bytedmd(_matvec, (A, x)) == 21


def test_matvec_3x3_cost():
    A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    x = [1, 1, 1]
    assert bb.bytedmd(_matvec, (A, x)) == 58


def test_matvec_4x4_cost():
    A = [[1] * 4 for _ in range(4)]
    x = [1] * 4
    assert bb.bytedmd(_matvec, (A, x)) == 118


# ---------------------------------------------------------------------------
# Belady is a lower bound on regular LRU
# ---------------------------------------------------------------------------

def test_belady_cheaper_on_large_inputs():
    """On larger inputs where hot-hit savings dominate cold-miss differences,
    Belady's OPT ordering should be cheaper than regular LRU.

    Note: Belady is NOT a strict lower bound on regular LRU because the
    cold-miss pricing models differ (Belady uses a separate monotonic DRAM
    tape; regular LRU prices cold misses relative to current stack size).
    On small inputs, the DRAM tape can be more expensive than the small LRU
    stack, so Belady can exceed regular LRU cost."""
    belady_cost = bb.bytedmd(_matvec, ([[1]*4 for _ in range(4)], [1]*4))
    reg_cost = bytedmd.bytedmd(_matvec, ([[1]*4 for _ in range(4)], [1]*4))
    assert belady_cost < reg_cost


def test_belady_strictly_cheaper_on_matvec():
    """Belady should be strictly cheaper than LRU on larger inputs."""
    A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    x = [1, 1, 1]
    assert bb.bytedmd(_matvec, (A, x)) < bytedmd.bytedmd(_matvec, (A, x))


# ---------------------------------------------------------------------------
# Matmul: Belady significantly cheaper than LRU
# ---------------------------------------------------------------------------

def test_matmul_belady_cheaper():
    """Belady reorders the cache optimally, giving large savings on matmul."""
    def matmul(A, B):
        n = len(A)
        C = [[None] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                s = A[i][0] * B[0][j]
                for k in range(1, n):
                    s = s + A[i][k] * B[k][j]
                C[i][j] = s
        return C

    A = [[1] * 4 for _ in range(4)]
    B = [[1] * 4 for _ in range(4)]
    belady_cost = bb.bytedmd(matmul, (A, B))
    reg_cost = bytedmd.bytedmd(matmul, (A, B))
    assert belady_cost == 413
    assert belady_cost < reg_cost


# ---------------------------------------------------------------------------
# Bytes-per-element scaling
# ---------------------------------------------------------------------------

def test_bytes_per_element_scaling():
    def f(a, b, c, d):
        return b + c
    assert bb.bytedmd(f, (1, 2, 3, 4), bytes_per_element=1) == 3
    assert bb.bytedmd(f, (1, 2, 3, 4), bytes_per_element=2) == 7


def test_bytes_per_element_dot():
    def dot(a, b):
        return sum(i * j for i, j in zip(a, b))
    bpe1 = bb.bytedmd(dot, ([1, 2], [3, 4]), bytes_per_element=1)
    bpe2 = bb.bytedmd(dot, ([1, 2], [3, 4]), bytes_per_element=2)
    assert bpe2 > bpe1


# ---------------------------------------------------------------------------
# IR / introspection
# ---------------------------------------------------------------------------

def test_inspect_ir_returns_list():
    def f(a, b):
        return a + b
    ir = bb.inspect_ir(f, (1, 2))
    assert isinstance(ir, list)
    assert len(ir) > 0
    kinds = {ev[0] for ev in ir}
    assert 'READ' in kinds
    assert 'OP' in kinds


def test_format_ir_contains_total():
    def f(a, b):
        return a + b
    ir = bb.inspect_ir(f, (1, 2))
    txt = bb.format_ir(ir)
    assert 'total cost' in txt
    assert 'add' in txt


def test_traced_eval_preserves_result():
    def dot(a, b):
        return sum(i * j for i, j in zip(a, b))
    a, b = [1, 2, 3], [4, 5, 6]
    _, traced_result = bb.traced_eval(dot, (a, b))
    assert traced_result == dot(a, b)


# ---------------------------------------------------------------------------
# Sanity: trace_to_bytedmd agrees with bytedmd()
# ---------------------------------------------------------------------------

def test_trace_to_bytedmd_consistency():
    def f(a, b, c):
        return (a + b) * c
    trace, _ = bb.traced_eval(f, (2, 3, 4))
    direct = bb.bytedmd(f, (2, 3, 4))
    via_trace = bb.trace_to_bytedmd(trace, bytes_per_element=1)
    assert direct == via_trace
