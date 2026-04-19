"""pytest harness: lints manual.py + runs the parity check + verifies
DSL-ported examples match their hand-rolled counterparts.

Run:
    python3 -m pytest experiments/grid/test_manuals.py -v

Or inline:
    python3 experiments/grid/test_manuals.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import lint_manual
import check_parity
import manual as man
from manual_dsl_examples import (
    manual_naive_matmul_dsl,
    manual_fft_iterative_dsl,
    manual_bitonic_sort_dsl,
    manual_matvec_row_dsl,
)


# ---------------------------------------------------------------------------
# Lint
# ---------------------------------------------------------------------------

def test_lint_clean() -> None:
    """manual.py should have no unsuppressed lint hits."""
    path = os.path.join(HERE, "manual.py")
    hits = lint_manual.run_lint(path)
    if hits:
        msg = [f"{h.rule}: line {h.line}  {h.snippet}  ({h.explanation})"
               for h in hits]
        raise AssertionError("manual.py has lint hits:\n  " + "\n  ".join(msg))


# ---------------------------------------------------------------------------
# Parity check — flags suspicious undercharges
# ---------------------------------------------------------------------------

def test_parity() -> None:
    """Every manual's read count should be ≥ 80% of its trace's load
    count (unless exempted in check_parity.FUSION_EXCEPTIONS)."""
    import run_grid as rg
    problems = []
    for name, fn, args, manual_fn in rg.ALGOS:
        m, t = check_parity.parity_check(name, fn, args, manual_fn)
        ratio = m / t if t else float("inf")
        if ratio < 0.8 and name not in check_parity.FUSION_EXCEPTIONS:
            problems.append(
                f"{name}: manual={m:,} trace={t:,} ratio={ratio:.2f}"
            )
    if problems:
        raise AssertionError(
            "parity check failed for:\n  " + "\n  ".join(problems))


# ---------------------------------------------------------------------------
# DSL parity — ported DSL versions must match their hand-rolled counterparts.
# ---------------------------------------------------------------------------

def test_dsl_matches_naive_matmul() -> None:
    # DSL may differ slightly from hand-rolled (arg promotion allocates
    # tmp slots whose exact addresses differ from the hand-rolled layout).
    # Within 5% is acceptable.
    dsl_cost = manual_naive_matmul_dsl(16)
    hand_cost = man.manual_naive_matmul(16)
    ratio = dsl_cost / hand_cost
    assert 0.95 <= ratio <= 1.05, (
        f"dsl={dsl_cost}  hand={hand_cost}  ratio={ratio:.3f}"
    )


def test_dsl_matches_fft_iterative() -> None:
    # The hand-rolled FFT may differ slightly because it writes explicit
    # sub+add+assign using the SAME tmp addr repeatedly (no cost
    # difference at the inner loop, but small address-layout shift).
    # Assert within 5% tolerance rather than exact match.
    dsl_cost = manual_fft_iterative_dsl(256)
    hand_cost = man.manual_fft_iterative(256)
    ratio = dsl_cost / hand_cost
    assert 0.95 <= ratio <= 1.05, (
        f"dsl={dsl_cost}  hand={hand_cost}  ratio={ratio:.3f}"
    )


def test_dsl_matches_bitonic_sort() -> None:
    dsl_cost = manual_bitonic_sort_dsl(64)
    hand_cost = man.manual_bitonic_sort(64)
    ratio = dsl_cost / hand_cost
    assert 0.95 <= ratio <= 1.05, (
        f"dsl={dsl_cost}  hand={hand_cost}  ratio={ratio:.3f}"
    )


def test_dsl_matches_matvec_row() -> None:
    dsl_cost = manual_matvec_row_dsl(64)
    hand_cost = man.manual_matvec_row(64)
    ratio = dsl_cost / hand_cost
    assert 0.95 <= ratio <= 1.05, (
        f"dsl={dsl_cost}  hand={hand_cost}  ratio={ratio:.3f}"
    )


if __name__ == "__main__":
    # Run all tests manually (no pytest dependency needed).
    failures = []
    for name in list(globals()):
        if name.startswith("test_"):
            try:
                globals()[name]()
                print(f"PASS  {name}")
            except AssertionError as e:
                print(f"FAIL  {name}\n      {e}")
                failures.append(name)
    if failures:
        print(f"\n{len(failures)} test(s) failed.")
        sys.exit(1)
    print("\nAll tests passed.")
