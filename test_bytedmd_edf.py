"""Tests for bytedmd_edf — Earliest Deadline First register allocation.

EDF packs variable lifetimes into spatial slots of increasing cost using
greedy interval scheduling. Arguments (cold misses) are routed to E slots;
computation results to W slots; post-read intervals go to ANY.
"""

import pytest

import bytedmd
import bytedmd_edf as be


# ---------------------------------------------------------------------------
# Simple scalar functions
# ---------------------------------------------------------------------------

def test_simple_add():
    def f(a, b):
        return a + b
    cost, result, _ = be.traced_eval(f, (1, 2))
    assert result == 3
    assert cost == 3


def test_my_add_bc():
    def f(a, b, c, d):
        return b + c
    cost, result, _ = be.traced_eval(f, (1, 2, 3, 4))
    assert result == 5
    assert cost == 3


def test_unused_args_not_charged():
    """Unused arguments produce no intervals and incur no cost."""
    def f4(a, b, c, d):
        return b + c
    def f6(a, b, c, d, e, f):
        return b + c
    assert be.bytedmd(f4, (1, 2, 3, 4)) == be.bytedmd(f6, (1, 2, 3, 4, 5, 6))


def test_repeated_operand():
    def f(a):
        return a + a
    cost, result, _ = be.traced_eval(f, (5,))
    assert result == 10
    assert cost == 2


def test_left_associative_chain():
    def f(a, b, c):
        return (a + b) + c
    cost, result, _ = be.traced_eval(f, (1, 2, 3))
    assert result == 6
    assert cost == 6


# ---------------------------------------------------------------------------
# Dot product
# ---------------------------------------------------------------------------

def test_dot_product():
    def dot(a, b):
        return sum(i * j for i, j in zip(a, b))
    cost, result, _ = be.traced_eval(dot, ([1, 2], [3, 4]))
    assert result == 11
    assert cost == 11


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


def test_matvec_2x2_cost():
    A = [[1, 2], [3, 4]]
    x = [5, 6]
    cost, result, _ = be.traced_eval(_matvec, (A, x))
    assert result == [17, 39]
    assert cost == 22


def test_matvec_3x3_cost():
    A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    x = [1, 1, 1]
    assert be.bytedmd(_matvec, (A, x)) == 62


def test_matvec_4x4_cost():
    A = [[1] * 4 for _ in range(4)]
    x = [1] * 4
    assert be.bytedmd(_matvec, (A, x)) == 127


# ---------------------------------------------------------------------------
# Matmul
# ---------------------------------------------------------------------------

def _matmul(A, B):
    n = len(A)
    C = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = A[i][0] * B[0][j]
            for k in range(1, n):
                s = s + A[i][k] * B[k][j]
            C[i][j] = s
    return C


def test_matmul_2x2():
    A = [[1] * 2 for _ in range(2)]
    B = [[1] * 2 for _ in range(2)]
    assert be.bytedmd(_matmul, (A, B)) == 45


def test_matmul_4x4():
    A = [[1] * 4 for _ in range(4)]
    B = [[1] * 4 for _ in range(4)]
    assert be.bytedmd(_matmul, (A, B)) == 515


def test_matmul_edf_cheaper_than_lru():
    """EDF optimal allocation should be cheaper than LRU on matmul."""
    A = [[1] * 4 for _ in range(4)]
    B = [[1] * 4 for _ in range(4)]
    edf_cost = be.bytedmd(_matmul, (A, B))
    lru_cost = bytedmd.bytedmd(_matmul, (A, B))
    assert edf_cost < lru_cost


# ---------------------------------------------------------------------------
# Bytes-per-element scaling
# ---------------------------------------------------------------------------

def test_bytes_per_element_scaling():
    def f(a, b, c, d):
        return b + c
    assert be.bytedmd(f, (1, 2, 3, 4), bytes_per_element=1) == 3
    assert be.bytedmd(f, (1, 2, 3, 4), bytes_per_element=2) == 7


def test_bytes_per_element_dot():
    def dot(a, b):
        return sum(i * j for i, j in zip(a, b))
    bpe1 = be.bytedmd(dot, ([1, 2], [3, 4]), bytes_per_element=1)
    bpe2 = be.bytedmd(dot, ([1, 2], [3, 4]), bytes_per_element=2)
    assert bpe2 > bpe1


# ---------------------------------------------------------------------------
# IR / introspection
# ---------------------------------------------------------------------------

def test_inspect_ir_returns_list():
    def f(a, b):
        return a + b
    ir = be.inspect_ir(f, (1, 2))
    assert isinstance(ir, list)
    assert len(ir) > 0
    kinds = {ev[0] for ev in ir}
    assert 'READ' in kinds
    assert 'OP' in kinds


def test_format_ir_contains_total():
    def f(a, b):
        return a + b
    ir = be.inspect_ir(f, (1, 2))
    txt = be.format_ir(ir)
    assert 'total cost' in txt
    assert 'add' in txt


def test_traced_eval_preserves_result():
    def dot(a, b):
        return sum(i * j for i, j in zip(a, b))
    a, b = [1, 2, 3], [4, 5, 6]
    _, traced_result, _ = be.traced_eval(dot, (a, b))
    assert traced_result == dot(a, b)


# ---------------------------------------------------------------------------
# EDF should never be worse than cold (stationary slots) on matmul
# ---------------------------------------------------------------------------

def test_edf_vs_cold_matmul():
    """EDF resolves the Squatter Anomaly, so it should generally
    be competitive with or better than greedy min-heap allocation."""
    import bytedmd_cold as bc
    A = [[1] * 4 for _ in range(4)]
    B = [[1] * 4 for _ in range(4)]
    edf_cost = be.bytedmd(_matmul, (A, B))
    cold_cost = bc.bytedmd(_matmul, (A, B))
    # EDF should be at most as expensive as cold (stationary slots)
    assert edf_cost <= cold_cost, f"edf={edf_cost} > cold={cold_cost}"
