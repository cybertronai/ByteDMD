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


# ---------------------------------------------------------------------------
# matvec_col — column-major accumulator
# ---------------------------------------------------------------------------

def manual_matvec_col_dsl(n: int) -> int:
    sch = Sched()
    A = sch.arg_buffer(n * n)
    x = sch.arg_buffer(n)
    tmp = sch.scalar()
    c_xj = sch.scalar()
    y = sch.output_buffer(n)
    for j in range(n):
        sch.assign(x[j], c_xj)        # preload x[j] into hot scalar
        if j == 0:
            for i in range(n):
                # First column: y[i] = A[i][0] * x[0]  (no accumulation)
                sch.mul(A[i * n + 0], c_xj, y[i])
        else:
            for i in range(n):
                sch.mac(y[i], A[i * n + j], c_xj, tmp)
    return sch.finalize()


# ---------------------------------------------------------------------------
# matvec_blocked — B×B tile with x-tile scratchpad reuse.
# ---------------------------------------------------------------------------

def manual_matvec_blocked_dsl(n: int, B: int = 4) -> int:
    sch = Sched()
    A = sch.arg_buffer(n * n)
    x_main = sch.arg_buffer(n)
    s = [sch.scalar() for _ in range(B)]
    x_tile = sch.buffer(B)
    tmp = sch.scalar()
    y = sch.output_buffer(n)

    for i_out in range(0, n, B):
        for j_out in range(0, n, B):
            # Copy the current x-slice into the tile.
            for j in range(B):
                sch.assign(x_main[j_out + j], x_tile[j])
            for i in range(B):
                if j_out == 0:
                    # First contribution: s[i] = A[i][j_out] * x_tile[0]
                    sch.mul(A[(i_out + i) * n + (j_out + 0)], x_tile[0], s[i])
                    for j in range(1, B):
                        sch.mac(s[i], A[(i_out + i) * n + (j_out + j)],
                                x_tile[j], tmp)
                else:
                    for j in range(B):
                        sch.mac(s[i], A[(i_out + i) * n + (j_out + j)],
                                x_tile[j], tmp)
        for i in range(B):
            sch.assign(s[i], y[i_out + i])
    return sch.finalize()


# ---------------------------------------------------------------------------
# rmm — 2D leaf tile MAC via c_A scalar + c_B_row vector (same B-row
# stationary pattern as the optimized tiled_matmul).
# ---------------------------------------------------------------------------

def manual_rmm_dsl(n: int, T: int = 4) -> int:
    sch = Sched()
    A = sch.arg_buffer(n * n)
    B_in = sch.arg_buffer(n * n)
    c_A = sch.scalar()
    c_B = sch.buffer(T)
    tmp = sch.scalar()
    sC = sch.buffer(T * T)
    C = sch.output_buffer(n * n)

    def compute_tile(rA: int, cA: int, rB: int, cB: int, rC: int, cC: int,
                     is_first: bool) -> None:
        for kk in range(T):
            # Stream a single B-row into c_B.
            for jj in range(T):
                sch.assign(B_in[(rB + kk) * n + cB + jj], c_B[jj])
            for ii in range(T):
                # Broadcast a single A element into c_A.
                sch.assign(A[(rA + ii) * n + cA + kk], c_A)
                for jj in range(T):
                    if is_first and kk == 0:
                        sch.mul(c_A, c_B[jj], sC[ii * T + jj])
                    else:
                        sch.mac(sC[ii * T + jj], c_A, c_B[jj], tmp)
        # Flush sC → C.
        for ii in range(T):
            for jj in range(T):
                sch.assign(sC[ii * T + jj], C[(rC + ii) * n + cC + jj])

    last_C: list = [None]

    def recurse(rA: int, cA: int, rB: int, cB: int, rC: int, cC: int,
                sz: int) -> None:
        if sz <= T:
            is_first = (last_C[0] != (rC, cC))
            last_C[0] = (rC, cC)
            compute_tile(rA, cA, rB, cB, rC, cC, is_first)
            return
        h = sz // 2
        for dr, dc, erb, ecb, frc, fcc in [
            (0, 0, 0, 0, 0, 0), (0, 0, 0, h, 0, h),
            (h, 0, 0, h, h, h), (h, 0, 0, 0, h, 0),
            (h, h, h, 0, h, 0), (h, h, h, h, h, h),
            (0, h, h, h, 0, h), (0, h, h, 0, 0, 0),
        ]:
            recurse(rA + dr, cA + dc, rB + erb, cB + ecb,
                    rC + frc, cC + fcc, h)

    recurse(0, 0, 0, 0, 0, 0, n)
    return sch.finalize()


# ---------------------------------------------------------------------------
# tiled_matmul — B-row stationary outer product with blocks=2 reuse.
# ---------------------------------------------------------------------------

def manual_tiled_matmul_dsl(n: int, T: int | None = None) -> int:
    if T is None:
        T = max(1, int(round(n ** 0.5)))
    sch = Sched()
    A = sch.arg_buffer(n * n)
    B = sch.arg_buffer(n * n)
    tmp = sch.scalar()
    c_A = sch.scalar()
    c_B = sch.buffer(T)
    blocks = 2
    sC = sch.buffer(blocks * T * T)
    C = sch.output_buffer(n * n)

    for bj in range(0, n, T):
        for bi_start in range(0, n, blocks * T):
            for bk in range(0, n, T):
                for kk in range(min(T, n - bk)):
                    for jj in range(min(T, n - bj)):
                        sch.assign(B[(bk + kk) * n + (bj + jj)], c_B[jj])
                    for bi in range(bi_start,
                                    min(n, bi_start + blocks * T), T):
                        local_bi = (bi - bi_start) // T
                        for ii in range(min(T, n - bi)):
                            sch.assign(A[(bi + ii) * n + (bk + kk)], c_A)
                            for jj in range(min(T, n - bj)):
                                slot = sC[local_bi * T * T + ii * T + jj]
                                if bk == 0 and kk == 0:
                                    sch.mul(c_A, c_B[jj], slot)
                                else:
                                    sch.mac(slot, c_A, c_B[jj], tmp)
            # Flush fully-computed C tiles to output.
            for bi in range(bi_start,
                            min(n, bi_start + blocks * T), T):
                local_bi = (bi - bi_start) // T
                for ii in range(min(T, n - bi)):
                    for jj in range(min(T, n - bj)):
                        sch.assign(
                            sC[local_bi * T * T + ii * T + jj],
                            C[(bi + ii) * n + (bj + jj)])
    return sch.finalize()
