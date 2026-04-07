"""
Three matmul algorithms for the memory management experiment.

All take and return list-of-lists of scalars (Tracked or plain).
Sizes must be powers of 2 for rmm and strassen.
Strassen uses leaf size 1 (full recursion) so the data movement effects of
its 18 intermediate additions show clearly.
"""


def split(M):
    """Split N x N into four (N/2) x (N/2) quadrants."""
    n = len(M)
    h = n // 2
    A11 = [[M[i][j] for j in range(h)] for i in range(h)]
    A12 = [[M[i][j] for j in range(h, n)] for i in range(h)]
    A21 = [[M[i][j] for j in range(h)] for i in range(h, n)]
    A22 = [[M[i][j] for j in range(h, n)] for i in range(h, n)]
    return A11, A12, A21, A22


def join(C11, C12, C21, C22):
    h = len(C11)
    n = 2 * h
    C = [[None] * n for _ in range(n)]
    for i in range(h):
        for j in range(h):
            C[i][j] = C11[i][j]
            C[i][j + h] = C12[i][j]
            C[i + h][j] = C21[i][j]
            C[i + h][j + h] = C22[i][j]
    return C


def add_mat(A, B):
    n = len(A)
    return [[A[i][j] + B[i][j] for j in range(n)] for i in range(n)]


def sub_mat(A, B):
    n = len(A)
    return [[A[i][j] - B[i][j] for j in range(n)] for i in range(n)]


# ─────────────────────────── Naive (i-j-k) ────────────────────────────────

def naive_matmul(A, B):
    """Standard O(N^3) matmul, i-j-k loop order."""
    n = len(A)
    C = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = A[i][0] * B[0][j]
            for k in range(1, n):
                s = s + A[i][k] * B[k][j]
            C[i][j] = s
    return C


# ────────────────── Cache-oblivious recursive matmul (RMM) ────────────────

def rmm(A, B):
    """8-way recursive matmul. Leaf size 1 to expose the recurrence cleanly."""
    n = len(A)
    if n == 1:
        return [[A[0][0] * B[0][0]]]
    A11, A12, A21, A22 = split(A)
    B11, B12, B21, B22 = split(B)

    P1 = rmm(A11, B11); P2 = rmm(A12, B21)
    C11 = add_mat(P1, P2); del P1, P2
    P3 = rmm(A11, B12); P4 = rmm(A12, B22)
    C12 = add_mat(P3, P4); del P3, P4
    P5 = rmm(A21, B11); P6 = rmm(A22, B21)
    C21 = add_mat(P5, P6); del P5, P6
    P7 = rmm(A21, B12); P8 = rmm(A22, B22)
    C22 = add_mat(P7, P8); del P7, P8
    return join(C11, C12, C21, C22)


# ─────────────────────────── Strassen (leaf=1) ────────────────────────────

def strassen(A, B):
    """Strassen with leaf size 1.

    Standard recombination:
        C11 = M1 + M4 - M5 + M7
        C12 = M3 + M5
        C21 = M2 + M4
        C22 = M1 - M2 + M3 + M6

    Local temps are scoped tightly so the tombstone/aggressive memory
    strategies can reclaim them in sequence.
    """
    n = len(A)
    if n == 1:
        return [[A[0][0] * B[0][0]]]

    A11, A12, A21, A22 = split(A)
    B11, B12, B21, B22 = split(B)

    T1 = add_mat(A11, A22); T2 = add_mat(B11, B22)
    M1 = strassen(T1, T2); del T1, T2

    T3 = add_mat(A21, A22)
    M2 = strassen(T3, B11); del T3

    T4 = sub_mat(B12, B22)
    M3 = strassen(A11, T4); del T4

    T5 = sub_mat(B21, B11)
    M4 = strassen(A22, T5); del T5

    T6 = add_mat(A11, A12)
    M5 = strassen(T6, B22); del T6

    T7 = sub_mat(A21, A11); T8 = add_mat(B11, B12)
    M6 = strassen(T7, T8); del T7, T8

    T9 = sub_mat(A12, A22); T10 = add_mat(B21, B22)
    M7 = strassen(T9, T10); del T9, T10

    t = add_mat(M1, M4); t = sub_mat(t, M5)
    C11 = add_mat(t, M7); del t, M7

    C12 = add_mat(M3, M5); del M5

    C21 = add_mat(M2, M4)

    t = sub_mat(M1, M2); t = add_mat(t, M3)
    C22 = add_mat(t, M6); del t, M1, M2, M3, M4, M6

    return join(C11, C12, C21, C22)


# ─────────────────────────── FLOP counts ──────────────────────────────────

def flops_naive(N):
    """Naive matmul: N^2 dot products of length N. Each dot product is
    N multiplications + (N - 1) additions."""
    return N * N * (2 * N - 1)


def flops_rmm(N):
    """Recursive matmul (leaf=1) does the same arithmetic as naive."""
    return flops_naive(N)


def flops_strassen(N):
    """Strassen leaf=1 FLOP count.

    At each level S, performs 7 recursive multiplications of size S/2 and
    18 matrix additions of size S/2 (each addition has (S/2)^2 element
    operations). Recurrence: F(S) = 7 F(S/2) + 18 (S/2)^2 with F(1) = 1.
    """
    if N == 1:
        return 1
    k = N.bit_length() - 1  # log2(N)
    n_mults = 7 ** k
    n_adds = 0
    branches = 1
    s = N
    for _ in range(k):
        n_adds += branches * 18 * (s // 2) ** 2
        branches *= 7
        s //= 2
    return n_mults + n_adds


def make_ones(n):
    return [[1] * n for _ in range(n)]


def make_zeros(n):
    return [[0] * n for _ in range(n)]


# ─────────────── In-place Recursive Matmul (no temporaries) ───────────────
# C += A @ B implemented by passing index offsets into the original matrices.
# Allocates zero temporaries (only the per-leaf scalar product is allocated
# and immediately discarded). The 8 sub-products are dispatched in lex order
# (i, j, k) which forces every transition between calls to swap one or
# more sub-quadrants out of the working set.

def rmm_inplace_lex(A, B, C, ai=0, aj=0, bi=0, bj=0, ci=0, cj=0, n=None):
    """In-place 8-way recursive matmul, dispatched in lex (i,j,k) order."""
    if n is None:
        n = len(A)
    if n == 1:
        C[ci][cj] = C[ci][cj] + A[ai][aj] * B[bi][bj]
        return
    h = n // 2
    # Lex order: (i, j, k) where k is the shared dim, ci_off=i, cj_off=j.
    LEX_ORDER = [
        (0, 0, 0), (0, 0, 1),
        (0, 1, 0), (0, 1, 1),
        (1, 0, 0), (1, 0, 1),
        (1, 1, 0), (1, 1, 1),
    ]
    for i_off, j_off, k_off in LEX_ORDER:
        rmm_inplace_lex(
            A, B, C,
            ai + i_off * h, aj + k_off * h,
            bi + k_off * h, bj + j_off * h,
            ci + i_off * h, cj + j_off * h,
            h,
        )


# Gray-code (Morton Z-curve) order: only one of (i, j, k) changes per step.
# When the changing coordinate is X, the one matrix that does not depend on
# X (C if k changes; B if i changes; A if j changes) stays perfectly hot in
# the working set, halving the per-transition fetch cost.

GRAY_ORDER = [
    (0, 0, 0),  # 000
    (0, 0, 1),  # 001  k flipped
    (0, 1, 1),  # 011  j flipped → C is fresh, A stays hot
    (0, 1, 0),  # 010  k flipped → C, A stay hot
    (1, 1, 0),  # 110  i flipped → A, B stay hot... wait, A depends on i
    (1, 1, 1),  # 111  k flipped → A, C stay hot
    (1, 0, 1),  # 101  j flipped → A stays hot
    (1, 0, 0),  # 100  k flipped → A, C stay hot
]


def rmm_inplace_gray(A, B, C, ai=0, aj=0, bi=0, bj=0, ci=0, cj=0, n=None):
    """In-place 8-way recursive matmul, dispatched in Gray-code order."""
    if n is None:
        n = len(A)
    if n == 1:
        C[ci][cj] = C[ci][cj] + A[ai][aj] * B[bi][bj]
        return
    h = n // 2
    for i_off, j_off, k_off in GRAY_ORDER:
        rmm_inplace_gray(
            A, B, C,
            ai + i_off * h, aj + k_off * h,
            bi + k_off * h, bj + j_off * h,
            ci + i_off * h, cj + j_off * h,
            h,
        )
