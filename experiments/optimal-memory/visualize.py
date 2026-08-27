"""
Visualize memory access patterns for the optimal recursive matrix multiply.

Produces two figures:
  1. Memory access timeline — every read/write plotted as (time step, address),
     colored by access type (external read, working read, working write).
     Shows how the inverted stack arenas keep base-case accesses at low addresses.
  2. Address histogram — distribution of accessed addresses by type,
     showing the arena boundaries.
"""

import math
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from optimal_rmm import generate_traces, calc_cost

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def visualize(n=8):
    ext_r, wm_r, wm_w = generate_traces(n)
    total_cost = calc_cost(ext_r) + calc_cost(wm_r) + calc_cost(wm_w)

    # Build unified timeline: (time_step, address, type)
    timeline = []
    t = 0
    # Reconstruct interleaved order by re-running (simpler: just merge the
    # three traces in the order they were appended)
    # Since the traces are appended in execution order within each list,
    # we need to interleave them.  Re-run to get a single ordered trace.
    events = []  # (address, type_str)

    # Re-generate with interleaved events
    ext_reads2 = []
    wm_reads2 = []
    wm_writes2 = []
    all_events = []  # (addr, 'ext_read' | 'wm_read' | 'wm_write')

    def generate_interleaved(n):
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
                    if src_sp == 'EXT':
                        all_ev.append((src_addr, 'ext_read'))
                    elif src_sp == 'WM':
                        all_ev.append((src_addr, 'wm_read'))
                    if dst_sp == 'WM':
                        all_ev.append((dst_addr, 'wm_write'))

        def matmul(R, K, C_dim, d, A_info, B_info, C_info):
            if d == D:
                A_sp, A_base, _ = A_info
                B_sp, B_base, _ = B_info
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
                        A_sub_base = A_base + r_off * A_stride + k_off
                        copy_matrix(A_sp, A_sub_base, A_stride,
                                    'WM', A_prime_base, sub_k, sub_r, sub_k)
                        B_sp, B_base, B_stride = B_info
                        B_sub_base = B_base + k_off * B_stride + c_off
                        copy_matrix(B_sp, B_sub_base, B_stride,
                                    'WM', B_prime_base, sub_c, sub_k, sub_c)
                        matmul(sub_r, sub_k, sub_c, d+1,
                               ('WM', A_prime_base, sub_k),
                               ('WM', B_prime_base, sub_c),
                               ('WM', C_prime_base, sub_c))
                    copy_matrix('WM', C_prime_base, sub_c,
                                C_sp, C_sub_base, C_stride, sub_r, sub_c)

        A_ext_info = ('EXT', 1, n)
        B_ext_info = ('EXT', 1 + n**2, n)
        C_wm_info = ('WM', arena_start[0], n)
        for i in range(n * n):
            all_ev.append((arena_start[0] + i, 'wm_write'))
        matmul(n, n, n, 0, A_ext_info, B_ext_info, C_wm_info)

        return all_ev, arena_start, arena_size, D

    all_ev, arena_start, arena_size, D = generate_interleaved(n)

    # --- Figure 1: Memory access timeline ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10),
                                    gridspec_kw={'height_ratios': [3, 1]})

    colors = {'ext_read': '#e74c3c', 'wm_read': '#3498db', 'wm_write': '#2ecc71'}
    labels = {'ext_read': 'External read (DRAM)',
              'wm_read': 'Working mem read',
              'wm_write': 'Working mem write'}

    for typ in ['wm_write', 'wm_read', 'ext_read']:
        ts = [i for i, (_, t) in enumerate(all_ev) if t == typ]
        addrs = [all_ev[i][0] for i in ts]
        ax1.scatter(ts, addrs, c=colors[typ], s=0.5, alpha=0.6, label=labels[typ], rasterized=True)

    # Draw arena boundaries
    for d in range(D, 0, -1):
        y = arena_start[d]
        ax1.axhline(y=y, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
        ax1.text(len(all_ev) * 0.01, y + 1, f'arena d={d}', fontsize=7, color='gray')

    ax1.axhline(y=arena_start[0], color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax1.text(len(all_ev) * 0.01, arena_start[0] + 1, 'C output (d=0)', fontsize=7, color='gray')

    ax1.set_xlabel('Time step')
    ax1.set_ylabel('Memory address')
    ax1.set_title(f'Optimal RMM Memory Access Pattern (N={n})\n'
                  f'Inverted stack: deepest recursion at lowest addresses\n'
                  f'Total cost = {total_cost:,}')
    ax1.legend(loc='upper left', markerscale=10, fontsize=8)

    # --- Figure 2: Address histogram ---
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
    out_path = os.path.join(os.path.dirname(__file__), 'memory_access_pattern.png')
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close()

    # --- Figure 3: Cost-weighted access pattern ---
    fig2, ax3 = plt.subplots(1, 1, figsize=(14, 5))

    # Show cumulative cost over time, broken down by type
    cum_cost = {'ext_read': [], 'wm_read': [], 'wm_write': []}
    running = {'ext_read': 0, 'wm_read': 0, 'wm_write': 0}
    for addr, typ in all_ev:
        running[typ] += math.ceil(math.sqrt(addr))
        for t in cum_cost:
            cum_cost[t].append(running[t])

    x = np.arange(len(all_ev))
    for typ in ['ext_read', 'wm_read', 'wm_write']:
        ax3.plot(x, cum_cost[typ], color=colors[typ], label=labels[typ], linewidth=1)

    total_cum = [cum_cost['ext_read'][i] + cum_cost['wm_read'][i] + cum_cost['wm_write'][i]
                 for i in range(len(all_ev))]
    ax3.plot(x, total_cum, color='black', label='Total', linewidth=1.5, linestyle='--')

    ax3.set_xlabel('Time step')
    ax3.set_ylabel('Cumulative cost (ceil(sqrt(addr)))')
    ax3.set_title(f'Cumulative ByteDMD Cost Over Time (N={n})')
    ax3.legend(fontsize=8)

    plt.tight_layout()
    out_path2 = os.path.join(os.path.dirname(__file__), 'cumulative_cost.png')
    plt.savefig(out_path2, dpi=150)
    print(f"Saved: {out_path2}")
    plt.close()


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    visualize(n)
