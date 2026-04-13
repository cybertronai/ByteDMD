"""Tests for bytedmd_cold — the demand-paged variant using stationary slots
with min-heap recycling.  Cold misses are priced on a monotonic tape past the
global peak working set; hot hits cost sqrt(slot_number) where the slot is
fixed once assigned.

All expected values are calibrated by running the tracer.
"""

import pytest

import bytedmd
import bytedmd_cold as bc


# ---------------------------------------------------------------------------
# Simple scalar functions
# ---------------------------------------------------------------------------

def test_simple_add():
    """a + b: cold misses on both operands, priced past peak working set."""
    def f(a, b):
        return a + b
    trace, result = bc.traced_eval(f, (1, 2))
    assert trace == [3, 4]
    assert result == 3
    assert bc.bytedmd(f, (1, 2)) == 4


def test_my_add_bc():
    """f(a,b,c,d) = b+c — only b and c are ever accessed."""
    def f(a, b, c, d):
        return b + c
    trace, result = bc.traced_eval(f, (1, 2, 3, 4))
    assert trace == [3, 4]
    assert result == 5
    assert bc.bytedmd(f, (1, 2, 3, 4)) == 4


def test_unused_args_not_charged():
    """Unused arguments don't inflate the peak working set, so cold-miss
    pricing is identical regardless of how many unused args are passed."""
    def f4(a, b, c, d):
        return b + c
    def f6(a, b, c, d, e, f):
        return b + c
    assert bc.bytedmd(f4, (1, 2, 3, 4)) == bc.bytedmd(f6, (1, 2, 3, 4, 5, 6))


def test_repeated_operand():
    """a + a: peak_ws=1, cold miss #1 prices a at depth 2, both reads
    see the same depth since pricing is simultaneous."""
    def f(a):
        return a + a
    trace, result = bc.traced_eval(f, (5,))
    assert trace == [2, 2]
    assert result == 10


def test_left_associative_chain():
    """(a + b) + c: one cold batch for (a,b), then one hot+cold batch."""
    def f(a, b, c):
        return (a + b) + c
    trace, result = bc.traced_eval(f, (1, 2, 3))
    assert trace == [3, 4, 1, 5]
    assert result == 6
    assert bc.bytedmd(f, (1, 2, 3)) == 8


# ---------------------------------------------------------------------------
# Dot product
# ---------------------------------------------------------------------------

def test_dot_product():
    def dot(a, b):
        return sum(i * j for i, j in zip(a, b))
    trace, result = bc.traced_eval(dot, ([1, 2], [3, 4]))
    assert trace == [4, 5, 1, 6, 7, 1, 2]
    assert result == 11
    assert bc.bytedmd(dot, ([1, 2], [3, 4])) == 15


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
    trace, result = bc.traced_eval(_matvec, (A, x))
    assert trace == [5, 6, 7, 8, 1, 3, 9, 2, 10, 4, 2, 3]
    assert result == [17, 39]
    assert bc.bytedmd(_matvec, (A, x)) == 30


def test_matvec_3x3_cost():
    A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    x = [1, 1, 1]
    assert bc.bytedmd(_matvec, (A, x)) == 85


def test_matvec_4x4_cost():
    A = [[1] * 4 for _ in range(4)]
    x = [1] * 4
    assert bc.bytedmd(_matvec, (A, x)) == 178


# ---------------------------------------------------------------------------
# Cross-check against the regular bytedmd tracer
# ---------------------------------------------------------------------------

def test_cold_cost_meets_or_exceeds_regular():
    """The stationary-slot model never underprices relative to regular bytedmd,
    because slots are fixed and don't benefit from free compaction."""
    def dot(a, b):
        return sum(i * j for i, j in zip(a, b))

    cases = [
        (dot, ([1, 2], [3, 4])),
        (dot, ([1, 2, 3], [4, 5, 6])),
        (_matvec, ([[1, 2], [3, 4]], [5, 6])),
        (_matvec, ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [1, 1, 1])),
    ]
    for func, args in cases:
        cold_cost = bc.bytedmd(func, args)
        reg_cost = bytedmd.bytedmd(func, args)
        assert cold_cost >= reg_cost, (
            f"{func.__name__}{args}: cold={cold_cost} < regular={reg_cost}"
        )


def test_cold_strictly_penalizes_matvec():
    """Stationary slots are strictly more expensive than regular LRU."""
    A = [[1, 2], [3, 4]]
    x = [5, 6]
    assert bc.bytedmd(_matvec, (A, x)) > bytedmd.bytedmd(_matvec, (A, x))


# ---------------------------------------------------------------------------
# Naive matmul scales as O(N^4) under the stationary-slot model
# ---------------------------------------------------------------------------

def test_matmul_scaling_n4():
    """The document's key prediction: naive matmul must scale as O(N^4)
    under a stationary-slot model, not the O(N^3.5) that free LRU
    compaction would suggest."""
    import math

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

    costs = []
    for n in [4, 8, 16]:
        A = [[1] * n for _ in range(n)]
        B = [[1] * n for _ in range(n)]
        costs.append(bc.bytedmd(matmul, (A, B)))

    # Check that cost(N) / N^4 converges (ratio should be roughly constant)
    ratios = [c / n**4 for c, n in zip(costs, [4, 8, 16])]
    # The ratio should not diverge — assert it stays within 2x of itself
    assert ratios[-1] / ratios[0] < 2.0, (
        f"N^4 ratios diverging: {ratios} — not O(N^4)"
    )
    # And check effective exponent between N=8 and N=16 is close to 4
    exponent = math.log2(costs[2] / costs[1]) / math.log2(2)
    assert 3.8 <= exponent <= 4.2, f"Effective exponent {exponent} not near 4"


# ---------------------------------------------------------------------------
# Bytes-per-element scaling
# ---------------------------------------------------------------------------

def test_bytes_per_element_scaling():
    """bytes_per_element=2 uses the closed-form sum-of-usqrts formula."""
    def f(a, b, c, d):
        return b + c
    assert bc.bytedmd(f, (1, 2, 3, 4), bytes_per_element=1) == 4
    assert bc.bytedmd(f, (1, 2, 3, 4), bytes_per_element=2) == 12


def test_bytes_per_element_dot():
    def dot(a, b):
        return sum(i * j for i, j in zip(a, b))
    bpe1 = bc.bytedmd(dot, ([1, 2], [3, 4]), bytes_per_element=1)
    bpe2 = bc.bytedmd(dot, ([1, 2], [3, 4]), bytes_per_element=2)
    assert bpe2 > bpe1


# ---------------------------------------------------------------------------
# IR / introspection
# ---------------------------------------------------------------------------

def test_inspect_ir_returns_list():
    def f(a, b):
        return a + b
    ir = bc.inspect_ir(f, (1, 2))
    assert isinstance(ir, list)
    assert len(ir) > 0
    kinds = {ev[0] for ev in ir}
    assert 'READ' in kinds
    assert 'OP' in kinds


def test_format_ir_contains_total():
    def f(a, b):
        return a + b
    ir = bc.inspect_ir(f, (1, 2))
    txt = bc.format_ir(ir)
    assert 'total cost' in txt
    assert 'add' in txt


def test_traced_eval_preserves_result():
    """The traced evaluation must return the same numerical result."""
    def dot(a, b):
        return sum(i * j for i, j in zip(a, b))
    a, b = [1, 2, 3], [4, 5, 6]
    _, traced_result = bc.traced_eval(dot, (a, b))
    assert traced_result == dot(a, b)


# ---------------------------------------------------------------------------
# Sanity: trace_to_bytedmd agrees with bytedmd()
# ---------------------------------------------------------------------------

def test_trace_to_bytedmd_consistency():
    def f(a, b, c):
        return (a + b) * c
    trace, _ = bc.traced_eval(f, (2, 3, 4))
    direct = bc.bytedmd(f, (2, 3, 4))
    via_trace = bc.trace_to_bytedmd(trace, bytes_per_element=1)
    assert direct == via_trace
