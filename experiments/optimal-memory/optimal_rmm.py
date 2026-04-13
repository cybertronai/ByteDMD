"""
Optimal O(N^3 log N) Recursive Matrix Multiplication Tracker.

Uses an Inverted Stack Arena layout where the deepest recursion level
(1x1 base cases) occupies the lowest memory addresses, and parent levels
occupy progressively higher addresses. This ensures base-case operations
cost O(1) to access, achieving O(N^3 log N) total data movement.

Source: gemini/optimal-managed-rmm.md
"""

import math


def generate_traces(n: int):
    """
    Simulate recursive matrix multiplication with inverted stack arenas.

    Returns (ext_reads, wm_reads, wm_writes) — three lists of 1-indexed
    memory addresses accessed during execution.
    """
    ext_reads = []
    wm_reads = []
    wm_writes = []

    # 1. Pre-compute the maximum dimensions at each recursion depth
    def get_depths_and_sizes(N):
        max_dims = {}

        def dry_run(R, K, C, depth):
            if depth not in max_dims:
                max_dims[depth] = [0, 0, 0]
            max_dims[depth][0] = max(max_dims[depth][0], R)
            max_dims[depth][1] = max(max_dims[depth][1], K)
            max_dims[depth][2] = max(max_dims[depth][2], C)

            if R == 1 and K == 1 and C == 1:
                return

            r1 = (R + 1) // 2; r2 = R - r1
            k1 = (K + 1) // 2; k2 = K - k1
            c1 = (C + 1) // 2; c2 = C - c1

            for r in [r1, r2]:
                for k in [k1, k2]:
                    for c in [c1, c2]:
                        if r > 0 and k > 0 and c > 0:
                            dry_run(r, k, c, depth + 1)

        dry_run(N, N, N, 0)
        return max_dims

    max_dims = get_depths_and_sizes(n)
    D = max(max_dims.keys())

    # 2. Build the Inverted Stack Layout for Working Memory
    arena_size = {}
    for d in range(1, D + 1):
        mR, mK, mC = max_dims[d]
        arena_size[d] = mR * mK + mK * mC + mR * mC

    arena_start = {}
    current_addr = 1
    for d in range(D, 0, -1):
        arena_start[d] = current_addr
        current_addr += arena_size[d]

    # Top-level target C matrix after all recursion buffers
    arena_start[0] = current_addr

    def copy_matrix(src_sp, src_base, src_stride,
                    dst_sp, dst_base, dst_stride,
                    R, C):
        """Trace block copying between External/Working space."""
        for i in range(R):
            for j in range(C):
                src_addr = src_base + i * src_stride + j
                dst_addr = dst_base + i * dst_stride + j

                if src_sp == 'EXT':
                    ext_reads.append(src_addr)
                elif src_sp == 'WM':
                    wm_reads.append(src_addr)

                if dst_sp == 'WM':
                    wm_writes.append(dst_addr)

    # 3. Core Recursive MatMul with arena-based allocation
    def matmul(R, K, C_dim, d, A_info, B_info, C_info):
        # Base Case
        if d == D:
            A_sp, A_base, _ = A_info
            B_sp, B_base, _ = B_info
            C_sp, C_base, _ = C_info

            # C += A * B
            if A_sp == 'EXT': ext_reads.append(A_base)
            else: wm_reads.append(A_base)

            if B_sp == 'EXT': ext_reads.append(B_base)
            else: wm_reads.append(B_base)

            if C_sp == 'WM': wm_reads.append(C_base)
            if C_sp == 'WM': wm_writes.append(C_base)
            return

        r1 = (R + 1) // 2; r2 = R - r1
        k1 = (K + 1) // 2; k2 = K - k1
        c1 = (C_dim + 1) // 2; c2 = C_dim - c1

        r_splits = [(0, r1), (r1, r2)]
        k_splits = [(0, k1), (k1, k2)]
        c_splits = [(0, c1), (c1, c2)]

        # Child depth arena layout
        A_prime_base = arena_start[d+1]
        B_prime_base = arena_start[d+1] + max_dims[d+1][0] * max_dims[d+1][1]
        C_prime_base = B_prime_base + max_dims[d+1][1] * max_dims[d+1][2]

        # 8-way quadrant sub-multiplications
        for r_idx in range(2):
            r_off, sub_r = r_splits[r_idx]
            if sub_r == 0: continue

            for c_idx in range(2):
                c_off, sub_c = c_splits[c_idx]
                if sub_c == 0: continue

                # Pull parent C block down into child arena
                C_sp, C_base, C_stride = C_info
                C_sub_base = C_base + r_off * C_stride + c_off

                copy_matrix(C_sp, C_sub_base, C_stride,
                            'WM', C_prime_base, sub_c,
                            sub_r, sub_c)

                for k_idx in range(2):
                    k_off, sub_k = k_splits[k_idx]
                    if sub_k == 0: continue

                    # Copy A quadrant down
                    A_sp, A_base, A_stride = A_info
                    A_sub_base = A_base + r_off * A_stride + k_off
                    copy_matrix(A_sp, A_sub_base, A_stride,
                                'WM', A_prime_base, sub_k,
                                sub_r, sub_k)

                    # Copy B quadrant down
                    B_sp, B_base, B_stride = B_info
                    B_sub_base = B_base + k_off * B_stride + c_off
                    copy_matrix(B_sp, B_sub_base, B_stride,
                                'WM', B_prime_base, sub_c,
                                sub_k, sub_c)

                    A_prime_info = ('WM', A_prime_base, sub_k)
                    B_prime_info = ('WM', B_prime_base, sub_c)
                    C_prime_info = ('WM', C_prime_base, sub_c)

                    matmul(sub_r, sub_k, sub_c, d+1, A_prime_info, B_prime_info, C_prime_info)

                # Push resolved C back up to parent
                copy_matrix('WM', C_prime_base, sub_c,
                            C_sp, C_sub_base, C_stride,
                            sub_r, sub_c)

    # 4. Entry Point
    A_ext_info = ('EXT', 1, n)
    B_ext_info = ('EXT', 1 + n**2, n)
    C_wm_info = ('WM', arena_start[0], n)

    # Initialize top-level C to zeros
    for i in range(n * n):
        wm_writes.append(arena_start[0] + i)

    matmul(n, n, n, 0, A_ext_info, B_ext_info, C_wm_info)

    return ext_reads, wm_reads, wm_writes


def calc_cost(trace):
    """ByteDMD cost: sum of ceil(sqrt(k)) for each address k."""
    return sum(math.ceil(math.sqrt(k)) for k in trace)


if __name__ == "__main__":
    for n in [2, 4, 8, 16]:
        ext_r, wm_r, wm_w = generate_traces(n)
        total_cost = calc_cost(ext_r) + calc_cost(wm_r) + calc_cost(wm_w)
        print(f"N={n:>2}  ext_reads={len(ext_r):>8}  wm_reads={len(wm_r):>8}  "
              f"wm_writes={len(wm_w):>8}  cost={total_cost:>10}")
