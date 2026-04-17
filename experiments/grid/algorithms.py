"""Traceable Python algorithms for the grid experiment.

Every algorithm performs its work through overloaded arithmetic so that
bytedmd_ir.trace() can convert it to an L2 event sequence. Helpers _max2,
_exp, _inv stand in for max/exp/1-over in attention — they have the same
read/write pattern as the real ops.
"""
from __future__ import annotations

from typing import List


# ============================================================================
# Matmul variants not already in bytedmd_ir
# ============================================================================

def _split(M):
    n = len(M); h = n // 2
    return ([[M[i][j] for j in range(h)] for i in range(h)],
            [[M[i][j] for j in range(h, n)] for i in range(h)],
            [[M[i][j] for j in range(h)] for i in range(h, n)],
            [[M[i][j] for j in range(h, n)] for i in range(h, n)])

def _join(C11, C12, C21, C22):
    h = len(C11); n = 2 * h
    return [[C11[i][j] if j < h else C12[i][j-h] for j in range(n)] for i in range(h)] + \
           [[C21[i][j] if j < h else C22[i][j-h] for j in range(n)] for i in range(h)]

def _addm(A, B):
    n = len(A)
    return [[A[i][j] + B[i][j] for j in range(n)] for i in range(n)]

def _subm(A, B):
    n = len(A)
    return [[A[i][j] - B[i][j] for j in range(n)] for i in range(n)]


def matmul_strassen(A, B):
    n = len(A)
    if n == 1:
        return [[A[0][0] * B[0][0]]]
    A11, A12, A21, A22 = _split(A)
    B11, B12, B21, B22 = _split(B)
    M1 = matmul_strassen(_addm(A11, A22), _addm(B11, B22))
    M2 = matmul_strassen(_addm(A21, A22), B11)
    M3 = matmul_strassen(A11, _subm(B12, B22))
    M4 = matmul_strassen(A22, _subm(B21, B11))
    M5 = matmul_strassen(_addm(A11, A12), B22)
    M6 = matmul_strassen(_subm(A21, A11), _addm(B11, B12))
    M7 = matmul_strassen(_subm(A12, A22), _addm(B21, B22))
    C11 = _addm(_subm(_addm(M1, M4), M5), M7)
    C12 = _addm(M3, M5)
    C21 = _addm(M2, M4)
    C22 = _addm(_subm(_addm(M1, M3), M2), M6)
    return _join(C11, C12, C21, C22)


# ============================================================================
# Attention
# ============================================================================

def _max2(a, b): return a + b      # trace-safe stand-in for max
def _exp(x):    return x * x       # trace-safe stand-in for exp
def _inv(x):    return x * x       # trace-safe stand-in for 1/x


def naive_attention(Q, K, V):
    N = len(Q); d = len(Q[0]); scale = d ** -0.5
    S = [[None] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            acc = Q[i][0] * K[j][0]
            for dd in range(1, d):
                acc = acc + Q[i][dd] * K[j][dd]
            S[i][j] = acc * scale
    P = [[None] * N for _ in range(N)]
    for i in range(N):
        mx = S[i][0]
        for j in range(1, N):
            mx = _max2(mx, S[i][j])
        row_sum = None
        for j in range(N):
            P[i][j] = _exp(S[i][j] - mx)
            row_sum = P[i][j] if row_sum is None else row_sum + P[i][j]
        inv_sum = _inv(row_sum)
        for j in range(N):
            P[i][j] = P[i][j] * inv_sum
    O = [[None] * d for _ in range(N)]
    for i in range(N):
        for dd in range(d):
            acc = P[i][0] * V[0][dd]
            for j in range(1, N):
                acc = acc + P[i][j] * V[j][dd]
            O[i][dd] = acc
    return O


def flash_attention(Q, K, V, Bk=2):
    N = len(Q); d = len(Q[0]); scale = d ** -0.5
    num_blocks = (N + Bk - 1) // Bk
    O = [[None] * d for _ in range(N)]
    for i in range(N):
        m_prev = None; l_prev = None
        o_acc = [None] * d
        for kb in range(num_blocks):
            k0 = kb * Bk
            k1 = min(k0 + Bk, N)
            bs = k1 - k0
            s_block = [None] * bs
            for j in range(bs):
                kj = k0 + j
                acc = Q[i][0] * K[kj][0]
                for dd in range(1, d):
                    acc = acc + Q[i][dd] * K[kj][dd]
                s_block[j] = acc * scale
            m_block = s_block[0]
            for j in range(1, bs):
                m_block = _max2(m_block, s_block[j])
            p_block = [None] * bs
            l_block = None
            for j in range(bs):
                p_block[j] = _exp(s_block[j] - m_block)
                l_block = p_block[j] if l_block is None else l_block + p_block[j]
            o_block = [None] * d
            for dd in range(d):
                acc = p_block[0] * V[k0][dd]
                for j in range(1, bs):
                    acc = acc + p_block[j] * V[k0 + j][dd]
                o_block[dd] = acc
            if m_prev is None:
                m_prev = m_block; l_prev = l_block
                for dd in range(d):
                    o_acc[dd] = o_block[dd]
            else:
                m_new = _max2(m_prev, m_block)
                alpha = _exp(m_prev - m_new)
                beta = _exp(m_block - m_new)
                l_prev = alpha * l_prev + beta * l_block
                for dd in range(d):
                    o_acc[dd] = alpha * o_acc[dd] + beta * o_block[dd]
                m_prev = m_new
        inv_l = _inv(l_prev)
        for dd in range(d):
            O[i][dd] = o_acc[dd] * inv_l
    return O


# ============================================================================
# Transpose
# ============================================================================

def transpose_naive(A):
    """B[i][j] = A[j][i]. Reads A in column-major, writes B in row-major."""
    n = len(A)
    B = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            # force a load via "+ 0"
            B[i][j] = A[j][i] + 0
    return B


def transpose_blocked(A, T=None):
    n = len(A)
    if T is None:
        T = max(1, int(round(n ** 0.5)))
    B = [[None] * n for _ in range(n)]
    for bi in range(0, n, T):
        for bj in range(0, n, T):
            for i in range(bi, min(bi + T, n)):
                for j in range(bj, min(bj + T, n)):
                    B[i][j] = A[j][i] + 0
    return B


def transpose_recursive(A):
    """Cache-oblivious transpose via 4-way split."""
    n = len(A)
    B = [[None] * n for _ in range(n)]

    def rec(ar, ac, br, bc, sz):
        if sz == 1:
            B[br][bc] = A[ar][ac] + 0
            return
        h = sz // 2
        rec(ar,     ac,     br,     bc,     h)
        rec(ar + h, ac,     br,     bc + h, h)
        rec(ar,     ac + h, br + h, bc,     h)
        rec(ar + h, ac + h, br + h, bc + h, h)

    rec(0, 0, 0, 0, n)
    return B


# ============================================================================
# Matrix-vector
# ============================================================================

def matvec_row(A, x):
    """Row-major matvec: y[i] = sum_j A[i][j] * x[j]."""
    n = len(A)
    y = [None] * n
    for i in range(n):
        s = A[i][0] * x[0]
        for j in range(1, n):
            s = s + A[i][j] * x[j]
        y[i] = s
    return y


def matvec_col(A, x):
    """Column-major matvec: accumulate y column-by-column — strided reads of A."""
    n = len(A)
    y = [None] * n
    for i in range(n):
        y[i] = A[i][0] * x[0]
    for j in range(1, n):
        for i in range(n):
            y[i] = y[i] + A[i][j] * x[j]
    return y


# ============================================================================
# FFT (radix-2 Cooley–Tukey, real twiddle stand-in — we only care about the
# read/write pattern, not the numeric result)
# ============================================================================

def fft_iterative(x_in):
    """In-place iterative radix-2 Cooley-Tukey on a length-N array.
    Uses a constant real factor in place of the complex twiddle — the
    load pattern is identical."""
    x = [v + 0 for v in x_in]  # force per-element load into fresh vars
    N = len(x)
    # Bit-reverse permutation
    j = 0
    for i in range(1, N):
        bit = N >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            t = x[i] + 0
            x[i] = x[j] + 0
            x[j] = t
    # Butterflies
    m = 1
    while m < N:
        for k in range(0, N, m * 2):
            for jj in range(m):
                t = x[k + jj + m] * 1.5   # twiddle stand-in
                u = x[k + jj]
                x[k + jj] = u + t
                x[k + jj + m] = u - t
        m *= 2
    return x


def fft_recursive(x_in):
    """Out-of-place recursive radix-2 Cooley-Tukey."""
    N = len(x_in)
    if N == 1:
        return [x_in[0] + 0]
    even = fft_recursive([x_in[2 * i] + 0 for i in range(N // 2)])
    odd  = fft_recursive([x_in[2 * i + 1] + 0 for i in range(N // 2)])
    out = [None] * N
    for k in range(N // 2):
        t = odd[k] * 1.5
        out[k] = even[k] + t
        out[k + N // 2] = even[k] - t
    return out


# ============================================================================
# 2D Jacobi stencil (5-point, one sweep)
# ============================================================================

def stencil_naive(A):
    """Row-major sweep: B[i][j] = 0.2 * (A[i][j] + A[i-1][j] + A[i+1][j]
    + A[i][j-1] + A[i][j+1]). Boundary cells are left None."""
    n = len(A)
    B = [[None] * n for _ in range(n)]
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            B[i][j] = (A[i][j] + A[i - 1][j] + A[i + 1][j]
                       + A[i][j - 1] + A[i][j + 1]) * 0.2
    return B


def stencil_recursive(A, leaf=8):
    """Tile-recursive split: quad-tree over the 2D grid, naive sweep at leaves."""
    n = len(A)
    B = [[None] * n for _ in range(n)]

    def rec(r0, c0, sz):
        if sz <= leaf:
            for i in range(r0, r0 + sz):
                for j in range(c0, c0 + sz):
                    if 0 < i < n - 1 and 0 < j < n - 1:
                        B[i][j] = (A[i][j] + A[i - 1][j] + A[i + 1][j]
                                   + A[i][j - 1] + A[i][j + 1]) * 0.2
            return
        h = sz // 2
        rec(r0,     c0,     h)
        rec(r0,     c0 + h, h)
        rec(r0 + h, c0,     h)
        rec(r0 + h, c0 + h, h)

    rec(0, 0, n)
    return B
