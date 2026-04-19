#!/Users/yaroslavvb/.local/bin/uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = ["matplotlib", "numpy"]
# ///
"""Naive vs Tiled matmul with manual (bump-allocated) memory and scratchpad.

Uses small matrices (N=4, T=2) so every access is individually visible.
Both reads (cost = ceil(sqrt(addr))) and writes (cost = 0) are traced.

Memory layout for tiled version:
  Addresses 1..T^2      : scratchpad tile sA  (fast, low sqrt cost)
  Addresses T^2+1..2T^2 : scratchpad tile sB
  Addresses 2T^2+1..3T^2: scratchpad tile sC
  Addresses 3T^2+1..     : main memory A, B, C (slow, high sqrt cost)

Memory layout for naive version:
  Address 1              : accumulator s
  Addresses 2..          : main memory A, B, C

Usage:
    uv run --script visualize_manual.py
"""

import math
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


REGION_COLORS = {
    'scratch': 'tab:cyan',
    'main_A': 'tab:blue',
    'main_B': 'tab:red',
    'main_C': 'tab:purple',
    'accum': 'tab:green',
}


def trace_naive(N):
    """Naive triple-loop C += A @ B^T with bump-allocated memory.

    Layout: s (accumulator) at addr 1, then A[N^2], B[N^2], C[N^2].
    For each output C[i][j]: read C[i][j] to initialize accumulator,
    then MAC loop over k, then write back (free).
    Returns (addrs, is_write, annotations, regions, cost).
    """
    addrs = []
    writes = []
    annotations = []
    cost = 0
    ptr = 1
    s = ptr; ptr += 1
    A = ptr; ptr += N * N
    B = ptr; ptr += N * N
    C = ptr; ptr += N * N

    def read(addr, desc):
        nonlocal cost
        c = math.isqrt(max(0, addr - 1)) + 1
        addrs.append(addr)
        writes.append(False)
        annotations.append((addr, c, 'R', desc))
        cost += c

    def write(addr, desc):
        addrs.append(addr)
        writes.append(True)
        annotations.append((addr, 0, 'W', desc))

    for i in range(N):
        for j in range(N):
            read(C + i*N + j, f"C[{i}][{j}] (init)")
            read(A + i*N + 0, f"A[{i}][0]")
            read(B + j*N + 0, f"B[{j}][0]")
            write(s, f"s = A[{i}][0]*B[{j}][0]")
            for k in range(1, N):
                read(A + i*N + k, f"A[{i}][{k}]")
                read(B + j*N + k, f"B[{j}][{k}]")
                read(s, f"s (acc for C[{i}][{j}])")
                write(s, f"s += A[{i}][{k}]*B[{j}][{k}]")
            read(s, f"s final")
            write(C + i*N + j, f"C[{i}][{j}] = s")

    regions = {
        'accum': (s, s),
        'main_A': (A, A + N*N - 1),
        'main_B': (B, B + N*N - 1),
        'main_C': (C, C + N*N - 1),
    }
    return addrs, writes, annotations, regions, cost


def trace_tiled(N, T):
    """Tiled matmul with scratchpad at lowest addresses.

    Layout: sA[T^2], sB[T^2], sC[T^2] at addrs 1..3T^2 (fast),
    then A[N^2], B[N^2], C[N^2] in main memory (slow).
    Returns (addrs, is_write, annotations, regions, cost).
    """
    addrs = []
    writes = []
    annotations = []
    cost = 0
    ptr = 1
    sA = ptr; ptr += T * T
    sB = ptr; ptr += T * T
    sC = ptr; ptr += T * T
    A = ptr;  ptr += N * N
    B = ptr;  ptr += N * N
    C = ptr;  ptr += N * N

    def read(addr, desc):
        nonlocal cost
        c = math.isqrt(max(0, addr - 1)) + 1
        addrs.append(addr)
        writes.append(False)
        annotations.append((addr, c, 'R', desc))
        cost += c

    def write(addr, desc):
        addrs.append(addr)
        writes.append(True)
        annotations.append((addr, 0, 'W', desc))

    for bi in range(0, N, T):
        for bj in range(0, N, T):
            # DMA: load C tile into scratchpad sC
            for ii in range(T):
                for jj in range(T):
                    read(C + (bi+ii)*N + (bj+jj),
                         f"DMA C[{bi+ii}][{bj+jj}]->sC")
                    write(sC + ii*T + jj,
                          f"sC[{ii}][{jj}] = C[{bi+ii}][{bj+jj}]")

            for bk in range(0, N, T):
                # DMA: load A tile into sA
                for ii in range(T):
                    for kk in range(T):
                        read(A + (bi+ii)*N + (bk+kk),
                             f"DMA A[{bi+ii}][{bk+kk}]->sA")
                        write(sA + ii*T + kk,
                              f"sA[{ii}][{kk}] = A[{bi+ii}][{bk+kk}]")
                # DMA: load B tile into sB
                for kk in range(T):
                    for jj in range(T):
                        read(B + (bk+kk)*N + (bj+jj),
                             f"DMA B[{bk+kk}][{bj+jj}]->sB")
                        write(sB + kk*T + jj,
                              f"sB[{kk}][{jj}] = B[{bk+kk}][{bj+jj}]")
                # MAC loop: all reads from scratchpad (low addresses)
                for ii in range(T):
                    for jj in range(T):
                        read(sC + ii*T + jj,
                             f"sC[{ii}][{jj}] (acc)")
                        for kk in range(T):
                            read(sA + ii*T + kk,
                                 f"sA[{ii}][{kk}]")
                            read(sB + kk*T + jj,
                                 f"sB[{kk}][{jj}]")
                        write(sC + ii*T + jj,
                              f"sC[{ii}][{jj}] += ...")

            # Flush: read sC, write back to C
            for ii in range(T):
                for jj in range(T):
                    read(sC + ii*T + jj,
                         f"flush sC[{ii}][{jj}]")
                    write(C + (bi+ii)*N + (bj+jj),
                          f"C[{bi+ii}][{bj+jj}] = sC[{ii}][{jj}]")

    regions = {
        'scratch': (sA, sC + T*T - 1),
        'main_A': (A, A + N*N - 1),
        'main_B': (B, B + N*N - 1),
        'main_C': (C, C + N*N - 1),
    }
    return addrs, writes, annotations, regions, cost


def print_trace(name, annotations, max_lines=None):
    """Print a human-readable trace of every memory access."""
    print(f"\n{'='*72}")
    print(f" {name} — Detailed Access Trace")
    print(f"{'='*72}")
    print(f"{'#':>4}  {'R/W':>3}  {'addr':>4}  {'cost':>4}  description")
    print(f"{'-'*4}  {'-'*3}  {'-'*4}  {'-'*4}  {'-'*40}")
    total = 0
    n = len(annotations) if max_lines is None else min(max_lines, len(annotations))
    for i in range(n):
        addr, c, rw, desc = annotations[i]
        total += c
        print(f"{i:>4}    {rw}  {addr:>4}  {c:>4}  {desc}")
    if max_lines is not None and max_lines < len(annotations):
        remaining_cost = sum(a[1] for a in annotations[max_lines:])
        total += remaining_cost
        print(f" ...  ({len(annotations) - max_lines} more accesses, "
              f"cost={remaining_cost:,})")
    print(f"\nTotal: {len(annotations)} accesses "
          f"({sum(1 for a in annotations if a[2] == 'R')} reads, "
          f"{sum(1 for a in annotations if a[2] == 'W')} writes), "
          f"cost = {total:,}")


def plot_panel(ax, addrs, is_write, regions, title, cost, y_max):
    ys = np.array(addrs)
    ws = np.array(is_write)
    xs = np.arange(len(ys))

    # Plot reads as filled circles, writes as hollow triangles
    for name, color in REGION_COLORS.items():
        if name not in regions:
            continue
        lo, hi = regions[name]
        mask_region = (ys >= lo) & (ys <= hi)
        mask_r = mask_region & ~ws
        mask_w = mask_region & ws
        if mask_r.any():
            ax.scatter(xs[mask_r], ys[mask_r], s=12, alpha=0.7, c=color,
                       label=f"{name} R ({lo}..{hi})", rasterized=True,
                       linewidths=0, marker='o')
        if mask_w.any():
            ax.scatter(xs[mask_w], ys[mask_w], s=30, alpha=0.7,
                       facecolors='none', edgecolors=color,
                       label=f"{name} W", rasterized=True,
                       linewidths=0.8, marker='v')

    n_reads = int((~ws).sum())
    n_writes = int(ws.sum())
    ax.set_ylabel('Physical address', fontsize=11)
    ax.set_ylim(0, y_max)
    ax.set_title(f'{title}\n{n_reads} reads + {n_writes} writes, '
                 f'read cost = {cost:,}', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc='center left', bbox_to_anchor=(1.01, 0.5),
              framealpha=0.95)


def main():
    N = 4
    T = 2

    naive_addrs, naive_ws, naive_ann, naive_reg, naive_cost = trace_naive(N)
    tiled_addrs, tiled_ws, tiled_ann, tiled_reg, tiled_cost = trace_tiled(N, T)

    # Print detailed traces
    print_trace(f"Naive Matmul (N={N})", naive_ann)
    print_trace(f"Tiled Matmul with Scratchpad (N={N}, T={T})", tiled_ann)

    # Summary
    print(f'\n{"="*72}')
    print(f' Summary: N={N}, T={T}')
    print(f'{"="*72}')
    nr = sum(1 for w in naive_ws if not w)
    nw = sum(1 for w in naive_ws if w)
    tr = sum(1 for w in tiled_ws if not w)
    tw = sum(1 for w in tiled_ws if w)
    print(f'  Naive:  {nr} reads + {nw} writes   read cost = {naive_cost:>6,}')
    print(f'  Tiled:  {tr} reads + {tw} writes   read cost = {tiled_cost:>6,}')
    print(f'  Speedup: {naive_cost/tiled_cost:.2f}x')

    # Plot
    y_max = max(max(naive_addrs), max(tiled_addrs)) + 3

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    plot_panel(axes[0], naive_addrs, naive_ws, naive_reg,
              f'Naive Matmul (N={N}) — all reads from main memory',
              naive_cost, y_max)
    plot_panel(axes[1], tiled_addrs, tiled_ws, tiled_reg,
              f'Tiled with Scratchpad (N={N}, T={T}) — inner loop from low addresses',
              tiled_cost, y_max)
    axes[1].set_xlabel('Access index', fontsize=11)

    fig.suptitle(
        f'Manual Memory: Naive vs Tiled (N={N})  '
        f'[filled = read, hollow triangle = write]\n'
        f'Naive read cost={naive_cost:,}  |  Tiled read cost={tiled_cost:,}  |  '
        f'Speedup={naive_cost/tiled_cost:.2f}x',
        fontsize=12, y=0.99)
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), 'manual_access_pattern.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nSaved: {out}')


if __name__ == '__main__':
    main()
