"""
Visualize memory access patterns for optimal Strassen matrix multiply.

Produces two figures:
  1. Memory access timeline — (time step, address) colored by access type,
     with arena boundaries showing the inverted stack layout.
  2. Cumulative cost — shows how total ByteDMD cost accumulates over time.

Also generates a side-by-side comparison with standard RMM.
"""

import math
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from optimal_strassen import generate_strassen_traces, calc_cost
from optimal_rmm import generate_traces as generate_rmm_traces

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def generate_interleaved_strassen(n):
    """Re-run Strassen with interleaved event recording."""
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError("n must be a power of 2")

    D = int(math.log2(n)) if n > 0 else 0
    all_ev = []

    arena_size = {}
    for d in range(1, D + 1):
        m_half = n // (2**d)
        arena_size[d] = 3 * (m_half ** 2)

    arena_start = {}
    current_addr = 1
    for d in range(D, 0, -1):
        arena_start[d] = current_addr
        current_addr += arena_size[d]
    arena_start[0] = current_addr

    def make_input(src1_info, src2_info, dst_info, size, op='add'):
        sp1, base1, stride1 = src1_info
        sp_dst, base_dst, stride_dst = dst_info
        for i in range(size):
            for j in range(size):
                a1 = base1 + i * stride1 + j
                if sp1 == 'EXT': all_ev.append((a1, 'ext_read'))
                else: all_ev.append((a1, 'wm_read'))
                if src2_info:
                    sp2, base2, stride2 = src2_info
                    a2 = base2 + i * stride2 + j
                    if sp2 == 'EXT': all_ev.append((a2, 'ext_read'))
                    else: all_ev.append((a2, 'wm_read'))
                a_dst = base_dst + i * stride_dst + j
                all_ev.append((a_dst, 'wm_write'))

    def accumulate(src_info, dst_ops, size):
        sp_src, base_src, stride_src = src_info
        for i in range(size):
            for j in range(size):
                a_src = base_src + i * stride_src + j
                all_ev.append((a_src, 'wm_read'))
                for dst_info, op in dst_ops:
                    sp_dst, base_dst, stride_dst = dst_info
                    a_dst = base_dst + i * stride_dst + j
                    if op != 'assign':
                        all_ev.append((a_dst, 'wm_read'))
                    all_ev.append((a_dst, 'wm_write'))

    def quad(info, r_off, c_off, child_m):
        sp, base, stride = info
        return (sp, base + r_off * child_m * stride + c_off * child_m, stride)

    def strassen(M, d, A_info, B_info, C_info):
        if M == 1:
            spA, bA, _ = A_info; spB, bB, _ = B_info; spC, bC, _ = C_info
            if spA == 'EXT': all_ev.append((bA, 'ext_read'))
            else: all_ev.append((bA, 'wm_read'))
            if spB == 'EXT': all_ev.append((bB, 'ext_read'))
            else: all_ev.append((bB, 'wm_read'))
            all_ev.append((bC, 'wm_write'))
            return

        child_m = M // 2
        A11 = quad(A_info, 0, 0, child_m); A12 = quad(A_info, 0, 1, child_m)
        A21 = quad(A_info, 1, 0, child_m); A22 = quad(A_info, 1, 1, child_m)
        B11 = quad(B_info, 0, 0, child_m); B12 = quad(B_info, 0, 1, child_m)
        B21 = quad(B_info, 1, 0, child_m); B22 = quad(B_info, 1, 1, child_m)
        C11 = quad(C_info, 0, 0, child_m); C12 = quad(C_info, 0, 1, child_m)
        C21 = quad(C_info, 1, 0, child_m); C22 = quad(C_info, 1, 1, child_m)

        base_d1 = arena_start[d+1]
        X_info = ('WM', base_d1, child_m)
        Y_info = ('WM', base_d1 + child_m**2, child_m)
        Z_info = ('WM', base_d1 + 2 * child_m**2, child_m)

        make_input(A11, A22, X_info, child_m); make_input(B11, B22, Y_info, child_m)
        strassen(child_m, d+1, X_info, Y_info, Z_info)
        accumulate(Z_info, [(C11, 'assign'), (C22, 'assign')], child_m)

        make_input(A21, A22, X_info, child_m); make_input(B11, None, Y_info, child_m)
        strassen(child_m, d+1, X_info, Y_info, Z_info)
        accumulate(Z_info, [(C21, 'assign'), (C22, 'sub')], child_m)

        make_input(A11, None, X_info, child_m); make_input(B12, B22, Y_info, child_m)
        strassen(child_m, d+1, X_info, Y_info, Z_info)
        accumulate(Z_info, [(C12, 'assign'), (C22, 'add')], child_m)

        make_input(A22, None, X_info, child_m); make_input(B21, B11, Y_info, child_m)
        strassen(child_m, d+1, X_info, Y_info, Z_info)
        accumulate(Z_info, [(C11, 'add'), (C21, 'add')], child_m)

        make_input(A11, A12, X_info, child_m); make_input(B22, None, Y_info, child_m)
        strassen(child_m, d+1, X_info, Y_info, Z_info)
        accumulate(Z_info, [(C11, 'sub'), (C12, 'add')], child_m)

        make_input(A21, A11, X_info, child_m); make_input(B11, B12, Y_info, child_m)
        strassen(child_m, d+1, X_info, Y_info, Z_info)
        accumulate(Z_info, [(C22, 'add')], child_m)

        make_input(A12, A22, X_info, child_m); make_input(B21, B22, Y_info, child_m)
        strassen(child_m, d+1, X_info, Y_info, Z_info)
        accumulate(Z_info, [(C11, 'add')], child_m)

    strassen(n, 0, ('EXT', 1, n), ('EXT', 1 + n**2, n), ('WM', arena_start[0], n))
    return all_ev, arena_start, arena_size, D


def generate_interleaved_rmm(n):
    """Re-run standard RMM with interleaved event recording."""
    all_ev = []

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
    arena_size = {}
    for d in range(1, D + 1):
        mR, mK, mC = max_dims[d]
        arena_size[d] = mR * mK + mK * mC + mR * mC
    arena_start = {}
    current_addr = 1
    for d in range(D, 0, -1):
        arena_start[d] = current_addr
        current_addr += arena_size[d]
    arena_start[0] = current_addr

    def copy_matrix(src_sp, src_base, src_stride,
                    dst_sp, dst_base, dst_stride, R, C):
        for i in range(R):
            for j in range(C):
                src_addr = src_base + i * src_stride + j
                dst_addr = dst_base + i * dst_stride + j
                if src_sp == 'EXT': all_ev.append((src_addr, 'ext_read'))
                elif src_sp == 'WM': all_ev.append((src_addr, 'wm_read'))
                if dst_sp == 'WM': all_ev.append((dst_addr, 'wm_write'))

    def matmul(R, K, C_dim, d, A_info, B_info, C_info):
        if d == D:
            A_sp, A_base, _ = A_info; B_sp, B_base, _ = B_info
            C_sp, C_base, _ = C_info
            if A_sp == 'EXT': all_ev.append((A_base, 'ext_read'))
            else: all_ev.append((A_base, 'wm_read'))
            if B_sp == 'EXT': all_ev.append((B_base, 'ext_read'))
            else: all_ev.append((B_base, 'wm_read'))
            if C_sp == 'WM': all_ev.append((C_base, 'wm_read'))
            if C_sp == 'WM': all_ev.append((C_base, 'wm_write'))
            return
        r1 = (R + 1) // 2; r2 = R - r1
        k1 = (K + 1) // 2; k2 = K - k1
        c1 = (C_dim + 1) // 2; c2 = C_dim - c1
        r_splits = [(0, r1), (r1, r2)]
        k_splits = [(0, k1), (k1, k2)]
        c_splits = [(0, c1), (c1, c2)]
        A_prime_base = arena_start[d+1]
        B_prime_base = arena_start[d+1] + max_dims[d+1][0] * max_dims[d+1][1]
        C_prime_base = B_prime_base + max_dims[d+1][1] * max_dims[d+1][2]
        for r_idx in range(2):
            r_off, sub_r = r_splits[r_idx]
            if sub_r == 0: continue
            for c_idx in range(2):
                c_off, sub_c = c_splits[c_idx]
                if sub_c == 0: continue
                C_sp, C_base, C_stride = C_info
                C_sub_base = C_base + r_off * C_stride + c_off
                copy_matrix(C_sp, C_sub_base, C_stride,
                            'WM', C_prime_base, sub_c, sub_r, sub_c)
                for k_idx in range(2):
                    k_off, sub_k = k_splits[k_idx]
                    if sub_k == 0: continue
                    A_sp, A_base, A_stride = A_info
                    copy_matrix(A_sp, A_base + r_off * A_stride + k_off, A_stride,
                                'WM', A_prime_base, sub_k, sub_r, sub_k)
                    B_sp, B_base, B_stride = B_info
                    copy_matrix(B_sp, B_base + k_off * B_stride + c_off, B_stride,
                                'WM', B_prime_base, sub_c, sub_k, sub_c)
                    matmul(sub_r, sub_k, sub_c, d+1,
                           ('WM', A_prime_base, sub_k),
                           ('WM', B_prime_base, sub_c),
                           ('WM', C_prime_base, sub_c))
                copy_matrix('WM', C_prime_base, sub_c,
                            C_sp, C_sub_base, C_stride, sub_r, sub_c)

    for i in range(n * n):
        all_ev.append((arena_start[0] + i, 'wm_write'))
    matmul(n, n, n, 0, ('EXT', 1, n), ('EXT', 1 + n**2, n), ('WM', arena_start[0], n))
    return all_ev, arena_start, arena_size, D


def visualize_strassen(n=8):
    all_ev, arena_start, arena_size, D = generate_interleaved_strassen(n)

    ext_r, wm_r, wm_w = generate_strassen_traces(n)
    total_cost = calc_cost(ext_r) + calc_cost(wm_r) + calc_cost(wm_w)

    colors = {'ext_read': '#e74c3c', 'wm_read': '#3498db', 'wm_write': '#2ecc71'}
    labels = {'ext_read': 'External read (DRAM)',
              'wm_read': 'Working mem read',
              'wm_write': 'Working mem write'}

    # --- Figure 1: Access timeline + histogram ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10),
                                    gridspec_kw={'height_ratios': [3, 1]})

    for typ in ['wm_write', 'wm_read', 'ext_read']:
        ts = [i for i, (_, t) in enumerate(all_ev) if t == typ]
        addrs = [all_ev[i][0] for i in ts]
        ax1.scatter(ts, addrs, c=colors[typ], s=0.5, alpha=0.6,
                    label=labels[typ], rasterized=True)

    for d in range(D, 0, -1):
        y = arena_start[d]
        ax1.axhline(y=y, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
        ax1.text(len(all_ev) * 0.01, y + 0.5, f'arena d={d}', fontsize=7, color='gray')

    ax1.axhline(y=arena_start[0], color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax1.text(len(all_ev) * 0.01, arena_start[0] + 0.5, 'C output (d=0)', fontsize=7, color='gray')

    ax1.set_xlabel('Time step')
    ax1.set_ylabel('Memory address')
    ax1.set_title(f'Optimal Strassen Memory Access Pattern (N={n})\n'
                  f'7 sub-problems per level, inverted stack arenas\n'
                  f'Total cost = {total_cost:,}')
    ax1.legend(loc='upper left', markerscale=10, fontsize=8)

    max_addr = max(a for a, _ in all_ev)
    bins = np.linspace(0, max_addr + 1, min(100, max_addr))
    for typ in ['ext_read', 'wm_read', 'wm_write']:
        addrs = [a for a, t in all_ev if t == typ]
        if addrs:
            ax2.hist(addrs, bins=bins, alpha=0.5, color=colors[typ], label=labels[typ])
    for d in range(D, 0, -1):
        ax2.axvline(x=arena_start[d], color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax2.set_xlabel('Memory address')
    ax2.set_ylabel('Access count')
    ax2.set_title('Address distribution by access type')
    ax2.legend(fontsize=8)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), 'strassen_access_pattern.png')
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close()

    # --- Figure 2: Cumulative cost ---
    fig2, ax3 = plt.subplots(1, 1, figsize=(14, 5))
    cum_cost = {'ext_read': [], 'wm_read': [], 'wm_write': []}
    running = {'ext_read': 0, 'wm_read': 0, 'wm_write': 0}
    for addr, typ in all_ev:
        running[typ] += math.ceil(math.sqrt(addr))
        for t in cum_cost:
            cum_cost[t].append(running[t])

    x = np.arange(len(all_ev))
    for typ in ['ext_read', 'wm_read', 'wm_write']:
        ax3.plot(x, cum_cost[typ], color=colors[typ], label=labels[typ], linewidth=1)
    total_cum = [sum(cum_cost[t][i] for t in cum_cost) for i in range(len(all_ev))]
    ax3.plot(x, total_cum, color='black', label='Total', linewidth=1.5, linestyle='--')
    ax3.set_xlabel('Time step')
    ax3.set_ylabel('Cumulative cost (ceil(sqrt(addr)))')
    ax3.set_title(f'Cumulative ByteDMD Cost Over Time — Strassen (N={n})')
    ax3.legend(fontsize=8)

    plt.tight_layout()
    out2 = os.path.join(os.path.dirname(__file__), 'strassen_cumulative_cost.png')
    plt.savefig(out2, dpi=150)
    print(f"Saved: {out2}")
    plt.close()

    # --- Figure 3: Side-by-side comparison with standard RMM ---
    fig3, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Strassen
    ax_s = axes[0]
    for typ in ['wm_write', 'wm_read', 'ext_read']:
        ts = [i for i, (_, t) in enumerate(all_ev) if t == typ]
        addrs = [all_ev[i][0] for i in ts]
        ax_s.scatter(ts, addrs, c=colors[typ], s=0.3, alpha=0.5, rasterized=True)
    ax_s.set_xlabel('Time step')
    ax_s.set_ylabel('Memory address')
    ax_s.set_title(f'Strassen (N={n})\ncost={total_cost:,}')

    # Standard RMM
    rmm_ev, rmm_arena_start, _, rmm_D = generate_interleaved_rmm(n)
    rmm_ext, rmm_wm_r, rmm_wm_w = generate_rmm_traces(n)
    rmm_cost = calc_cost(rmm_ext) + calc_cost(rmm_wm_r) + calc_cost(rmm_wm_w)

    ax_r = axes[1]
    for typ in ['wm_write', 'wm_read', 'ext_read']:
        ts = [i for i, (_, t) in enumerate(rmm_ev) if t == typ]
        addrs = [rmm_ev[i][0] for i in ts]
        ax_r.scatter(ts, addrs, c=colors[typ], s=0.3, alpha=0.5,
                     label=labels[typ], rasterized=True)
    ax_r.set_xlabel('Time step')
    ax_r.set_ylabel('Memory address')
    ax_r.set_title(f'Standard RMM (N={n})\ncost={rmm_cost:,}')
    ax_r.legend(loc='upper left', markerscale=10, fontsize=7)

    plt.suptitle(f'Memory Access Patterns: Strassen vs Standard RMM (N={n})', fontsize=13, y=1.02)
    plt.tight_layout()
    out3 = os.path.join(os.path.dirname(__file__), 'strassen_vs_rmm.png')
    plt.savefig(out3, dpi=150, bbox_inches='tight')
    print(f"Saved: {out3}")
    plt.close()

    # Print scaling comparison
    print(f"\n{'N':>4} {'Strassen':>10} {'Std RMM':>10} {'ratio':>8}")
    for sz in [2, 4, 8, 16]:
        s_ext, s_wm_r, s_wm_w = generate_strassen_traces(sz)
        s_cost = calc_cost(s_ext) + calc_cost(s_wm_r) + calc_cost(s_wm_w)
        r_ext, r_wm_r, r_wm_w = generate_rmm_traces(sz)
        r_cost = calc_cost(r_ext) + calc_cost(r_wm_r) + calc_cost(r_wm_w)
        print(f"{sz:4d} {s_cost:10d} {r_cost:10d} {r_cost/s_cost:8.2f}x")


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    visualize_strassen(n)
