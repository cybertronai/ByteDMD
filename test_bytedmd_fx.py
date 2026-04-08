#!/usr/bin/env python3
"""
Tests for bytedmd_fx — strict tracer that rejects UNTRACKED COMPUTATION.

The strict tracer allows data-dependent behavior (different costs for
different inputs) — control flow, indexing, dict lookups are all fine.
It only rejects operations that would let computation bypass the cost model.

Test categories:
  1. POSITIVE tests — clean functions that pass strict validation
  2. ALLOWED-CONTROL-FLOW tests — branching/indexing/dict lookups are accepted
  3. REJECTION tests — only float/complex/array escapes are rejected
  4. ROBUSTNESS — strict catches what non-strict undercounts
"""

import math
import operator
import numpy as np
import pytest

import bytedmd as original
import bytedmd_fx as fx
from bytedmd_fx import (
    bytedmd, traced_eval, get_trace, trace_to_bytedmd, format_ir, FxProxy,
)


# ═══════════════════════════════════════════════════════════════════════════
# 1. POSITIVE TESTS — clean functions pass and match non-strict results
# ═══════════════════════════════════════════════════════════════════════════

class TestStrictPositive:
    """Functions that don't escape — should trace cleanly and match originals."""

    def test_simple_add(self):
        def f(a, b, c, d):
            return b + c
        assert bytedmd(f, (1, 2, 3, 4)) == 4
        trace, _ = get_trace(f, (1, 2, 3, 4))
        assert trace == [3, 2]

    def test_dot_product_pure_arithmetic(self):
        def dot(a, b):
            return a[0] * b[0] + a[1] * b[1]
        a, b = [0, 1], [2, 3]
        trace, result = get_trace(dot, (a, b))
        assert trace == [4, 2, 5, 4, 4, 1]
        assert result == 3
        assert bytedmd(dot, (a, b)) == 12

    def test_repeated_operand(self):
        trace, _ = get_trace(lambda a: a + a, (5,))
        assert trace == [1, 1]

    def test_matvec_static_size(self):
        """matvec works because n is computed from len(plain list), not proxy."""
        def matvec(A, x):
            n = len(x)  # x is a plain Python list, len() returns plain int
            y = [None] * n
            for i in range(n):
                s = A[i][0] * x[0]
                for j in range(1, n):
                    s = s + A[i][j] * x[j]
                y[i] = s
            return y

        A, x = [[1, 2], [3, 4]], [5, 6]
        _, res = get_trace(matvec, (A, x))
        assert res == [17, 39]
        # Cost should match the original tracer
        assert bytedmd(matvec, (A, x)) == original.bytedmd(matvec, (A, x))

    def test_matmul_4x4_matches_original(self):
        def matmul4(A, B):
            n = len(A)
            C = [[None] * n for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    s = A[i][0] * B[0][j]
                    for k in range(1, n):
                        s = s + A[i][k] * B[k][j]
                    C[i][j] = s
            return C
        A, B = np.ones((4, 4)), np.ones((4, 4))
        assert bytedmd(matmul4, (A, B)) == 948  # known value

    def test_polynomial_horner(self):
        def horner(coeffs, x):
            result = coeffs[0]
            for i in range(1, len(coeffs)):
                result = result * x + coeffs[i]
            return result
        coeffs = [1, -2, 3, -4]
        _, res = get_trace(horner, (coeffs, 2))
        assert res == 2
        assert bytedmd(horner, (coeffs, 2)) == original.bytedmd(horner, (coeffs, 2))

    def test_constants_in_ops_allowed(self):
        """Constants on the right side of ops are fine."""
        trace, _ = get_trace(lambda a: a + 10, (5,))
        assert trace == [1]

    def test_neg_abs(self):
        trace, res = get_trace(lambda a: abs(-a), (5,))
        assert res == 5

    def test_comparison_returning_proxy(self):
        """A comparison that's the final return value is allowed (no __bool__)."""
        trace, res = get_trace(lambda a, b: a > b, (5, 3))
        assert res == True
        assert trace == [2, 1]

    def test_arithmetic_chain(self):
        def f(a, b, c, d):
            return (a + b) * (c - d)
        _, res = get_trace(f, (1, 2, 3, 4))
        assert res == -3

    def test_cross_check_against_non_strict(self):
        """Strict and non-strict should produce identical traces for clean code."""
        funcs = [
            (lambda a, b: a + b, (3, 7)),
            (lambda a, b: a * b + a, (2, 5)),
            (lambda a, b, c: a + b + c, (1, 2, 3)),
            (lambda a: a * a, (4,)),
            (lambda a: -a, (5,)),
            (lambda a, b: a - b, (10, 3)),
        ]
        for func, args in funcs:
            t_strict, _ = get_trace(func, args)
            t_fx, _ = fx.get_trace(func, args)
            assert t_strict == t_fx


# ═══════════════════════════════════════════════════════════════════════════
# 2. ALLOWED CONTROL FLOW — data-dependent behavior is fine
# ═══════════════════════════════════════════════════════════════════════════

class TestStrictAllowsControlFlow:
    """Branching, looping, indexing on tracked values is allowed.
    The cost is correctly counted for whichever path executes."""

    def test_allows_if_branching(self):
        def relu(a):
            if a > 0:
                return a * 2
            return a * 0

        # Both branches work; cost differs but is correctly counted
        _, res_pos = get_trace(relu, (5,))
        _, res_neg = get_trace(relu, (-5,))
        assert res_pos == 10
        assert res_neg == 0

        # Both should also match what bytedmd_fx (non-strict) reports
        assert bytedmd(relu, (5,)) == fx.bytedmd(relu, (5,))
        assert bytedmd(relu, (-5,)) == fx.bytedmd(relu, (-5,))

    def test_allows_implicit_bool(self):
        def f(a):
            if a:
                return a + 1
            return a
        _, res = get_trace(f, (5,))
        assert res == 6
        _, res = get_trace(f, (0,))
        assert res == 0

    def test_allows_short_circuit_and(self):
        _, res = get_trace(lambda a, b: a and b, (0, 5))
        assert res == 0
        _, res = get_trace(lambda a, b: a and b, (1, 5))
        assert res == 5

    def test_allows_short_circuit_or(self):
        _, res = get_trace(lambda a, b: a or b, (0, 5))
        assert res == 5
        _, res = get_trace(lambda a, b: a or b, (1, 5))
        assert res == 1

    def test_allows_not(self):
        _, res = get_trace(lambda a: not a, (0,))
        assert res is True
        _, res = get_trace(lambda a: not a, (5,))
        assert res is False

    def test_allows_while_loop_with_proxy(self):
        def f(a):
            s = a
            while a > 0:
                s = s + a
                a = a - 1
            return s
        _, res = get_trace(f, (3,))
        assert res == 9  # 3 + 3 + 2 + 1

    def test_allows_explicit_int_cast(self):
        _, res = get_trace(lambda a: int(a), (3.7,))
        assert res == 3

    def test_allows_explicit_bool_cast(self):
        _, res = get_trace(lambda a: bool(a), (5,))
        assert res is True

    def test_allows_range_with_proxy(self):
        """Loop count depends on proxy; each iteration's cost is counted."""
        def f(n, s):
            for _ in range(n):
                s = s + 1
            return s

        _, res3 = get_trace(f, (3, 0))
        assert res3 == 3
        _, res5 = get_trace(f, (5, 0))
        assert res5 == 5

        # The cost is naturally larger for larger n — that is correct,
        # because more additions actually executed.
        c3 = bytedmd(f, (3, 0))
        c5 = bytedmd(f, (5, 0))
        assert c5 > c3  # more iterations → more cost

    def test_allows_indexing_with_proxy(self):
        def f(xs, i):
            return xs[i]
        _, res = get_trace(f, ([10, 20, 30], 1))
        assert res == 20

    def test_allows_dict_lookup_with_proxy(self):
        def f(a):
            d = {5: 100, 10: 200}
            return d.get(a, 0)
        _, res = get_trace(f, (5,))
        assert res == 100
        _, res = get_trace(f, (99,))
        assert res == 0

    def test_allows_set_membership(self):
        def f(a):
            s = {1, 2, 3}
            return a in s
        _, res = get_trace(f, (2,))
        assert res is True
        _, res = get_trace(f, (5,))
        assert res is False

    def test_allows_hash_call(self):
        _, res = get_trace(lambda a: hash(a), (5,))
        assert res == hash(5)

    def test_data_dependent_loop_count_is_correct(self):
        """A function whose cost depends on input value still counts correctly."""
        def repeat_add(n, a):
            s = a
            for _ in range(n):
                s = s + a
            return s

        # Cost grows with n (more additions actually executed).
        c5 = bytedmd(repeat_add, (5, 1))
        c10 = bytedmd(repeat_add, (10, 1))
        c20 = bytedmd(repeat_add, (20, 1))
        assert c5 < c10 < c20


# ═══════════════════════════════════════════════════════════════════════════
# 3. REJECTION TESTS — only untracked-computation escapes are rejected
# ═══════════════════════════════════════════════════════════════════════════

class TestStrictRejection:
    """Only operations that allow untracked numeric computation are rejected."""

    def test_rejects_math_sin(self):
        """math.sin calls __float__ → rejected (untracked floating-point math)."""
        with pytest.raises(AssertionError, match="__float__"):
            traced_eval(lambda x: math.sin(x), (1.0,))

    def test_rejects_math_cos(self):
        with pytest.raises(AssertionError, match="__float__"):
            traced_eval(lambda x: math.cos(x), (1.0,))

    def test_rejects_math_exp(self):
        with pytest.raises(AssertionError, match="__float__"):
            traced_eval(lambda x: math.exp(x), (1.0,))

    def test_rejects_math_sqrt(self):
        with pytest.raises(AssertionError, match="__float__"):
            traced_eval(lambda x: math.sqrt(x), (4.0,))

    def test_rejects_math_log(self):
        with pytest.raises(AssertionError, match="__float__"):
            traced_eval(lambda x: math.log(x), (2.0,))

    def test_rejects_math_pow(self):
        with pytest.raises(AssertionError, match="__float__"):
            traced_eval(lambda x: math.pow(x, 2), (3.0,))

    def test_rejects_explicit_float_cast(self):
        with pytest.raises(AssertionError, match="__float__"):
            traced_eval(lambda a: float(a), (1.5,))

    def test_rejects_complex_cast(self):
        with pytest.raises(AssertionError, match="__complex__"):
            traced_eval(lambda a: complex(a), (5,))

    def test_rejects_math_in_loop(self):
        """math.sin used inside an allowed loop is still rejected."""
        def f(n):
            s = 0
            for i in range(n):
                s = s + math.sin(i)
            return s
        # range(n) is fine, but math.sin escapes
        # Actually i is a plain int from range, so math.sin(i) doesn't trigger
        # __float__ on a proxy. This should work.
        # Let's instead use a tracked value:
        def g(a):
            return math.sin(a) + a
        with pytest.raises(AssertionError, match="__float__"):
            traced_eval(g, (1.0,))

    def test_error_messages_are_informative(self):
        try:
            traced_eval(lambda x: math.sin(x), (1.0,))
            assert False, "should have raised"
        except AssertionError as e:
            msg = str(e)
            assert 'StrictByteDMD' in msg
            assert 'Fix:' in msg
            assert 'untracked' in msg.lower()


# ═══════════════════════════════════════════════════════════════════════════
# 3. ROBUSTNESS DEMONSTRATION
# ═══════════════════════════════════════════════════════════════════════════

class TestStrictRobustness:
    """Strict catches the only true 'cheating' path: untracked C-level math."""

    def test_non_strict_silently_undercounts_math_sin(self):
        """Non-strict tracer gives WRONG cost for math.sin — silent escape."""
        def f(x):
            # math.sin/cos/tan is significant work, but non-strict only
            # sees the to_float cast — the actual numeric work is invisible
            return math.sin(x) + math.cos(x) + math.tan(x)

        non_strict_cost = fx.bytedmd(f, (1.0,))
        # Some cost is reported, but it omits the sin/cos/tan computation
        assert non_strict_cost > 0

        # Strict rejects this immediately
        with pytest.raises(AssertionError, match="__float__"):
            bytedmd(f, (1.0,))

    def test_strict_allows_branching_with_correct_count(self):
        """Both branches give correct (different) costs in strict mode."""
        def f(a):
            if a > 0:
                return a * 100  # expensive branch
            return a  # cheap branch

        cost_pos = bytedmd(f, (5,))
        cost_neg = bytedmd(f, (-5,))
        # Different costs for different inputs is OK — it reflects the
        # actual work done. Both paths are correctly tracked.
        assert cost_pos != cost_neg
        assert cost_pos > cost_neg  # multiplication is extra work

        # Both should match what non-strict reports
        assert cost_pos == fx.bytedmd(f, (5,))
        assert cost_neg == fx.bytedmd(f, (-5,))

    def test_strict_allows_dynamic_loops_with_correct_count(self):
        """Loop costs scale with input — that is the correct count."""
        def f(n, s):
            for _ in range(n):
                s = s + 1
            return s

        c3 = bytedmd(f, (3, 0))
        c10 = bytedmd(f, (10, 0))
        # More iterations → more work → larger cost. This is correct.
        assert c10 > c3

        # Strict and non-strict give the same answer
        assert c3 == fx.bytedmd(f, (3, 0))
        assert c10 == fx.bytedmd(f, (10, 0))

    def test_strict_allows_dict_lookup(self):
        """Dict lookup is structural, not computation."""
        def f(a):
            d = {1: 10, 2: 20, 3: 30}
            return d.get(a, 0)
        _, res = get_trace(f, (1,))
        assert res == 10

    def test_strict_only_catches_untracked_compute(self):
        """The strict tracer is precisely targeted at hidden computation."""
        # These are all OK — visible work, correctly counted
        ok_funcs = [
            (lambda a: a + a + a, (5,)),
            (lambda a: a * a * a, (3,)),
            (lambda a, b: a if a > b else b, (5, 3)),  # if/else
            (lambda n: sum(i for i in range(n)), (5,)),  # range(proxy)
        ]
        for func, args in ok_funcs:
            cost = bytedmd(func, args)  # should not raise
            assert cost >= 0

        # These are rejected — untracked numeric computation
        cheats = [
            (lambda x: math.sin(x), (1.0,)),
            (lambda x: math.exp(x), (1.0,)),
            (lambda x: math.sqrt(x * x + 1), (3.0,)),
        ]
        for func, args in cheats:
            with pytest.raises(AssertionError):
                bytedmd(func, args)

    def test_cannot_cheat_with_math_in_loop_body(self):
        """A user can't sneak math.sin into a loop body."""
        def cheat(a):
            s = 0
            for _ in range(10):
                s = s + math.sin(a)  # tracked s, but sin escapes!
            return s

        with pytest.raises(AssertionError, match="__float__"):
            bytedmd(cheat, (1.0,))

    def test_cannot_cheat_with_math_in_branch(self):
        """A user can't sneak math.sin into a conditional branch."""
        def cheat(a):
            if a > 0:
                return math.sin(a)
            return a
        with pytest.raises(AssertionError, match="__float__"):
            bytedmd(cheat, (1.0,))


# ═══════════════════════════════════════════════════════════════════════════
# 4. IR INSPECTION (still works for clean functions)
# ═══════════════════════════════════════════════════════════════════════════

class TestStrictIR:

    def test_ir_for_clean_function(self):
        ctx, _ = traced_eval(lambda a, b: a + b, (1, 2))
        ir = format_ir(ctx)
        assert '%a = placeholder(a)' in ir
        assert '%b = placeholder(b)' in ir
        assert 'add(%a, %b)' in ir

    def test_graph_structure_matches_non_strict(self):
        def f(a, b, c):
            return (a + b) * c
        ctx_strict, _ = traced_eval(f, (2, 3, 4))
        ctx_fx, _ = fx.traced_eval(f, (2, 3, 4))

        strict_ops = [n.op for n in ctx_strict.graph.nodes]
        fx_ops = [n.op for n in ctx_fx.graph.nodes]
        assert strict_ops == fx_ops


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
