"""Worked-example ports of manual schedules to `manual_dsl`.

These mirror the hand-rolled `manual_*` functions in `manual.py` but
expressed in terms of Sched primitives. Use them as templates when
migrating additional algorithms — the DSL's MAC / butterfly / swap
primitives make it essentially impossible to get the binary-op read
counts wrong.

Each function returns the same cost number as its `manual.py`
counterpart (verified by `test_manuals.py`), but the source is
half the length and typo-proof for the common MAC patterns.
"""
from __future__ import annotations

from manual_dsl import Sched


# ---------------------------------------------------------------------------
# naive_matmul — C = A @ Bᵀ with a hoisted A row.
# ---------------------------------------------------------------------------

def manual_naive_matmul_dsl(n: int) -> int:
    s = Sched()
    A = s.arg_buffer(n * n)
    B = s.arg_buffer(n * n)
    tmp = s.scalar()
    acc = s.scalar()
    c_A_row = s.buffer(n)
    C = s.output_buffer(n * n)

    for i in range(n):
        for k in range(n):
            s.assign(A[i * n + k], c_A_row[k])
        for j in range(n):
            s.mul(c_A_row[0], B[j * n + 0], acc)     # first MAC: just multiply
            for k in range(1, n):
                s.mac(acc, c_A_row[k], B[j * n + k], tmp)
            s.assign(acc, C[i * n + j])
    return s.finalize()


# ---------------------------------------------------------------------------
# fft_iterative — in-place radix-2 Cooley-Tukey with priced butterflies.
# ---------------------------------------------------------------------------

def manual_fft_iterative_dsl(N: int) -> int:
    s = Sched()
    x_in = s.arg_buffer(N)
    tmp = s.scalar()
    x = s.output_buffer(N)

    # Preload input → output buffer (1 read per cell).
    for i in range(N):
        s.assign(x_in[i], x[i])

    # Bit-reverse permutation (each swap = 3 reads through tmp).
    j = 0
    for i in range(1, N):
        bit = N >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            s.swap(x[i], x[j], tmp)

    # Butterflies (each = 5 reads via Sched.butterfly).
    m = 1
    while m < N:
        for k in range(0, N, m * 2):
            for jj in range(m):
                s.butterfly(x[k + jj], x[k + jj + m], tmp)
        m *= 2
    return s.finalize()


# ---------------------------------------------------------------------------
# bitonic_sort — data-oblivious sorting network via `butterfly`.
# ---------------------------------------------------------------------------

def manual_bitonic_sort_dsl(N: int) -> int:
    s = Sched()
    arr_in = s.arg_buffer(N)
    tmp = s.scalar()
    arr = s.output_buffer(N)
    for i in range(N):
        s.assign(arr_in[i], arr[i])
    k = 2
    while k <= N:
        j = k // 2
        while j > 0:
            for i in range(N):
                l = i ^ j
                if l > i:
                    s.butterfly(arr[i], arr[l], tmp)
            j //= 2
        k *= 2
    return s.finalize()


# ---------------------------------------------------------------------------
# matvec_row — y = A · x with x preloaded into a hot scratch buffer.
# ---------------------------------------------------------------------------

def manual_matvec_row_dsl(n: int) -> int:
    sch = Sched()
    A = sch.arg_buffer(n * n)
    x = sch.arg_buffer(n)
    tmp = sch.scalar()
    acc = sch.scalar()
    c_X = sch.buffer(n)
    y = sch.output_buffer(n)
    for j in range(n):
        sch.assign(x[j], c_X[j])
    for i in range(n):
        sch.mul(A[i * n + 0], c_X[0], acc)
        for j in range(1, n):
            sch.mac(acc, A[i * n + j], c_X[j], tmp)
        sch.assign(acc, y[i])
    return sch.finalize()
