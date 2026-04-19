"""Parity check — for every algorithm in run_grid.ALGOS, compare the
manual's read count to the trace's L2Load count.

Rationale: under strict ByteDMD semantics, the manual should do at
least as many reads as the ops in the Python trace (the manual is a
different schedule, but it can't skip *fundamental* operations like
'read both operands of an add'). If the manual reports fewer reads
than the trace, it almost certainly has a mis-priced inner body.

The rule has documented exceptions — algorithms where a legitimate
fusion optimization genuinely eliminates whole intermediates:

  fused_strassen — 7 materialized M-intermediates are fused directly
                   into C fan-outs.
  flash_attn     — online-softmax blockwise state replaces full N×N S.
  naive_attn     — row-fused schedule matches flash's online pattern.
  tiled_matmul_* — register-blocked outer product with blocks=2
                   legitimately halves B arg reads.
  stencil_time_diamond — lazy-load + diamond pruning skips halo cells
                   outside the dependence cone.

For those, manual_reads < trace_loads is expected. For everything
else, a ratio below 0.8 is flagged for review.
"""
from __future__ import annotations

import os
import sys
from typing import List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

from bytedmd_ir import L2Load, trace
import manual as man
import run_grid as rg


# Algorithms where the manual *legitimately* runs fewer reads than the
# naive Python trace would emit. Each gets a short one-line explanation.
FUSION_EXCEPTIONS = {
    "fused_strassen(n=16)":
        "M-intermediates fused into C fan-outs, no materialization.",
    "flash_attn(N=32,d=2,Bk=8)":
        "online-softmax blockwise state replaces full N×N S.",
    "naive_attn(N=32,d=2)":
        "row-fused schedule streams S one row at a time.",
    "tiled_matmul(n=16)":
        "register-blocked outer product (blocks=2) halves B arg reads.",
    "tiled_matmul_explicit(n=16,T=4)":
        "shares the tiled_matmul manual schedule.",
    "stencil_time_diamond(16x16,T=4)":
        "lazy load + diamond pruning skips cells outside the dependence cone.",
    "blocked_lu(n=32,NB=8)":
        "lazy arg-load skips cells never needed at kb==0 boundary.",
    "mergesort(N=64)":
        "in-place oblivious merge + L1 scratchpad avoids temp copy-back.",
    "fft_iterative(N=256)":
        "in-place butterflies, no fan-out materialization.",
    "fft_recursive(N=256)":
        "strided in-place recursion eliminates even/odd temps.",
    "lcs_dp(32x32)":
        "rolling 2-row buffer (full DP table not materialized).",
    "stencil_naive(32x32)":
        "rolling 3-row buffer (full A not re-read each neighbor).",
    "naive_strassen(n=16)":
        "2D tile-MAC — fewer reads than the Strassen trace's 4-input "
        "M-intermediate assembly phase.",
    "matvec_col(n=64)":
        "column-major in-place accumulation — skips a tmp per inner op "
        "vs the trace's explicit s=s+tmp chain.",
    "stencil_recursive(32x32,leaf=8)":
        "rolling-row buffer via recursion leaf-reorder.",
    "lu_partial_pivot(n=32)":
        "scratchpad-cached pivot row avoids per-row trace tmp reads.",
    "householder_qr(32x32)":
        "c_V reflector cache + accumulator-pinned dot product halves "
        "trace's reflector-apply reads.",
    "blocked_qr(32x32,NB=8)":
        "same c_V + frequency-remapped A layout as householder_qr.",
    "stencil_time_naive(16x16,T=4)":
        "rolling 3-row buffer in time loop — skips T-1 re-read passes.",
    "layernorm_unfused(N=256)":
        "scalar accumulators held hot across variance + normalize; "
        "trace re-reads x-differences as explicit vars.",
    "tsqr(64x16,br=8)":
        "L1 tile cache + frequency-remapped layout; fewer reads than "
        "the trace's naive Householder emit.",
    "recursive_lu(n=32)":
        "c_C row buffer + frequency-remapped layout.",
}


def parity_check(name: str, fn, args, manual_fn) -> Tuple[int, int]:
    """Return (manual_reads, trace_loads)."""
    events, _ = trace(fn, args)
    trace_loads = sum(1 for e in events if isinstance(e, L2Load))

    logged = man.Allocator(logging=True)
    man.set_allocator(logged)
    try:
        manual_fn()
    finally:
        man.set_allocator(None)
    manual_reads = len(logged.log)
    return manual_reads, trace_loads


def main() -> int:
    print(f"{'algorithm':<42} {'manual':>10} {'trace':>10} "
          f"{'ratio':>6}  status")
    print("-" * 86)
    problems: List[str] = []
    for name, fn, args, manual_fn in rg.ALGOS:
        try:
            m, t = parity_check(name, fn, args, manual_fn)
        except Exception as e:
            print(f"{name:<42} EXCEPTION  {e}")
            problems.append(f"{name}: {e}")
            continue
        ratio = m / t if t else float("inf")
        status = "OK"
        if ratio < 0.8:
            if name in FUSION_EXCEPTIONS:
                status = f"EXEMPT ({FUSION_EXCEPTIONS[name]})"
            else:
                status = "LOW  (audit for missing reads)"
                problems.append(
                    f"{name}: manual={m:,}  trace={t:,}  ratio={ratio:.2f}"
                )
        print(f"{name:<42} {m:>10,} {t:>10,} {ratio:>6.2f}  {status}")
    print()
    if problems:
        print(f"{len(problems)} suspicious entries:")
        for p in problems:
            print(f"  {p}")
        return 1
    print("All manuals price at least 80% of the trace's reads "
          "(or are declared fusion exceptions).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
