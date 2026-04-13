"""
Optimal O(N^3) Strassen Matrix Multiplication Tracker.
Employs an Inverted Stack Arena mapped physically to theoretical access routing.

Strassen reduces recursive branches from 8 to 7, shifting the recurrence from
D(N) = 8D(N/2) + O(N^3)  =>  O(N^3 log N)    [standard RMM]
to
D(N) = 7D(N/2) + O(N^3)  =>  O(N^3)           [Strassen]

Source: gemini/optimal-managed-strassen.md
"""

import math


def generate_strassen_traces(n: int):
    """
    Simulate Strassen matrix multiplication with inverted stack arenas.

    Returns (ext_reads, wm_reads, wm_writes) — three lists of 1-indexed
    memory addresses accessed during execution.
    """
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError("n must be a power of 2 for Strassen arenas.")

    D = int(math.log2(n)) if n > 0 else 0
    ext_reads, wm_reads, wm_writes = [], [], []

    # 1. Build the Inverted Stack Layout for Working Memory
    # At depth d, allocate 3 matrices (X, Y, Z) of size (M/2) x (M/2)
    arena_size = {}
    for d in range(1, D + 1):
        m_half = n // (2**d)
        arena_size[d] = 3 * (m_half ** 2)

    arena_start = {}
    current_addr = 1

    # Deepest allocations get lowest indices
    for d in range(D, 0, -1):
        arena_start[d] = current_addr
        current_addr += arena_size[d]

    arena_start[0] = current_addr

    # 2. Memory operations
    def make_input(src1_info, src2_info, dst_info, size, op='add'):
        """Stage operands into local tightly-bounded arenas."""
        sp1, base1, stride1 = src1_info
        sp_dst, base_dst, stride_dst = dst_info

        for i in range(size):
            for j in range(size):
                a1 = base1 + i * stride1 + j
                if sp1 == 'EXT': ext_reads.append(a1)
                else: wm_reads.append(a1)

                if src2_info:
                    sp2, base2, stride2 = src2_info
                    a2 = base2 + i * stride2 + j
                    if sp2 == 'EXT': ext_reads.append(a2)
                    else: wm_reads.append(a2)

                a_dst = base_dst + i * stride_dst + j
                wm_writes.append(a_dst)

    def accumulate(src_info, dst_ops, size):
        """Dispatch finished child Z into target parent C quadrants."""
        sp_src, base_src, stride_src = src_info
        for i in range(size):
            for j in range(size):
                a_src = base_src + i * stride_src + j
                wm_reads.append(a_src)

                for dst_info, op in dst_ops:
                    sp_dst, base_dst, stride_dst = dst_info
                    a_dst = base_dst + i * stride_dst + j
                    if op != 'assign':
                        wm_reads.append(a_dst)
                    wm_writes.append(a_dst)

    def quad(info, r_off, c_off, child_m):
        """O(1) logical stride slicing."""
        sp, base, stride = info
        return (sp, base + r_off * child_m * stride + c_off * child_m, stride)

    # 3. Core Recursive Strassen
    def strassen(M, d, A_info, B_info, C_info):
        if M == 1:
            spA, bA, _ = A_info
            spB, bB, _ = B_info
            spC, bC, _ = C_info

            if spA == 'EXT': ext_reads.append(bA)
            else: wm_reads.append(bA)
            if spB == 'EXT': ext_reads.append(bB)
            else: wm_reads.append(bB)

            wm_writes.append(bC)
            return

        child_m = M // 2

        # Parent quadrants
        A11 = quad(A_info, 0, 0, child_m); A12 = quad(A_info, 0, 1, child_m)
        A21 = quad(A_info, 1, 0, child_m); A22 = quad(A_info, 1, 1, child_m)

        B11 = quad(B_info, 0, 0, child_m); B12 = quad(B_info, 0, 1, child_m)
        B21 = quad(B_info, 1, 0, child_m); B22 = quad(B_info, 1, 1, child_m)

        C11 = quad(C_info, 0, 0, child_m); C12 = quad(C_info, 0, 1, child_m)
        C21 = quad(C_info, 1, 0, child_m); C22 = quad(C_info, 1, 1, child_m)

        # Tombstone buffers for (d+1)
        base_d1 = arena_start[d+1]
        X_info = ('WM', base_d1, child_m)
        Y_info = ('WM', base_d1 + child_m**2, child_m)
        Z_info = ('WM', base_d1 + 2 * child_m**2, child_m)

        # M1 = (A11 + A22) * (B11 + B22)
        make_input(A11, A22, X_info, child_m, 'add')
        make_input(B11, B22, Y_info, child_m, 'add')
        strassen(child_m, d+1, X_info, Y_info, Z_info)
        accumulate(Z_info, [(C11, 'assign'), (C22, 'assign')], child_m)

        # M2 = (A21 + A22) * B11
        make_input(A21, A22, X_info, child_m, 'add')
        make_input(B11, None, Y_info, child_m, 'copy')
        strassen(child_m, d+1, X_info, Y_info, Z_info)
        accumulate(Z_info, [(C21, 'assign'), (C22, 'sub')], child_m)

        # M3 = A11 * (B12 - B22)
        make_input(A11, None, X_info, child_m, 'copy')
        make_input(B12, B22, Y_info, child_m, 'sub')
        strassen(child_m, d+1, X_info, Y_info, Z_info)
        accumulate(Z_info, [(C12, 'assign'), (C22, 'add')], child_m)

        # M4 = A22 * (B21 - B11)
        make_input(A22, None, X_info, child_m, 'copy')
        make_input(B21, B11, Y_info, child_m, 'sub')
        strassen(child_m, d+1, X_info, Y_info, Z_info)
        accumulate(Z_info, [(C11, 'add'), (C21, 'add')], child_m)

        # M5 = (A11 + A12) * B22
        make_input(A11, A12, X_info, child_m, 'add')
        make_input(B22, None, Y_info, child_m, 'copy')
        strassen(child_m, d+1, X_info, Y_info, Z_info)
        accumulate(Z_info, [(C11, 'sub'), (C12, 'add')], child_m)

        # M6 = (A21 - A11) * (B11 + B12)
        make_input(A21, A11, X_info, child_m, 'sub')
        make_input(B11, B12, Y_info, child_m, 'add')
        strassen(child_m, d+1, X_info, Y_info, Z_info)
        accumulate(Z_info, [(C22, 'add')], child_m)

        # M7 = (A12 - A22) * (B21 + B22)
        make_input(A12, A22, X_info, child_m, 'sub')
        make_input(B21, B22, Y_info, child_m, 'add')
        strassen(child_m, d+1, X_info, Y_info, Z_info)
        accumulate(Z_info, [(C11, 'add')], child_m)

    # 4. Entry point
    A_ext_info = ('EXT', 1, n)
    B_ext_info = ('EXT', 1 + n**2, n)
    C_wm_info = ('WM', arena_start[0], n)

    strassen(n, 0, A_ext_info, B_ext_info, C_wm_info)

    return ext_reads, wm_reads, wm_writes


def calc_cost(trace):
    """ByteDMD cost: sum of ceil(sqrt(k)) for each address k."""
    return sum(math.ceil(math.sqrt(k)) for k in trace)


if __name__ == "__main__":
    print(f"{'N':>4} {'ext_reads':>10} {'wm_reads':>10} {'wm_writes':>10} {'cost':>10}")
    for n in [2, 4, 8, 16, 32]:
        ext_r, wm_r, wm_w = generate_strassen_traces(n)
        total_cost = calc_cost(ext_r) + calc_cost(wm_r) + calc_cost(wm_w)
        print(f"{n:4d} {len(ext_r):10d} {len(wm_r):10d} {len(wm_w):10d} {total_cost:10d}")
