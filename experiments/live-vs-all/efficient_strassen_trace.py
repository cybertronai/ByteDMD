#!/Users/yaroslavvb/.local/bin/uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = ["matplotlib", "numpy"]
# ///
"""Three-way comparison: RMM vs Standard Strassen vs Zero-Allocation Fused Strassen.

Shows how eliminating temporary matrix allocations via virtual matrices
dramatically reduces Strassen's data movement cost.

Based on: gemini/strassen-manual.md (standard) and gemini/efficient-strassen.md (ZAFS).
"""

import math
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


class Allocator:
    """Bump-allocated physical memory with stack semantics and cost logging."""
    def __init__(self):
        self.cost = 0
        self.accesses = 0
        self.ptr = 1
        self.max_ptr = 1
        self.log = []

    def alloc(self, size: int) -> int:
        addr = self.ptr
        self.ptr += size
        if self.ptr > self.max_ptr:
            self.max_ptr = self.ptr
        return addr

    def push(self): return self.ptr
    def pop(self, ptr): self.ptr = ptr

    def read(self, addr: int):
        self.accesses += 1
        self.log.append(addr)
        self.cost += math.isqrt(max(0, addr - 1)) + 1


class Scratchpad:
    """Software-managed L1 scratchpad for 3 sub-tiles."""
    def __init__(self, alloc: Allocator, T: int):
        self.alloc, self.T = alloc, T
        self.fast = {'A': alloc.alloc(T*T), 'B': alloc.alloc(T*T), 'C': alloc.alloc(T*T)}
        self.loaded = {'A': None, 'B': None, 'C': None}
        self.dirty_C = False

    def sync(self, name, ptr, stride):
        if self.loaded[name] == (ptr, stride): return
        if name == 'C': self.flush_C()
        for i in range(self.T):
            for j in range(self.T):
                self.alloc.read(ptr + i * stride + j)
        self.loaded[name] = (ptr, stride)
        if name == 'C': self.dirty_C = False

    def flush_C(self):
        if self.loaded['C'] is not None and self.dirty_C:
            for i in range(self.T * self.T):
                self.alloc.read(self.fast['C'] + i)
            self.dirty_C = False

    def compute_tile(self, pA, sA, pB, sB, pC, sC):
        self.sync('A', pA, sA)
        self.sync('B', pB, sB)
        self.sync('C', pC, sC)
        T = self.T
        for i in range(T):
            for j in range(T):
                self.alloc.read(self.fast['C'] + i * T + j)
                for k in range(T):
                    self.alloc.read(self.fast['A'] + i * T + k)
                    self.alloc.read(self.fast['B'] + k * T + j)
        self.dirty_C = True


# ============================================================================
# 1. RMM (Hamiltonian 8-way recursive)
# ============================================================================

def run_rmm(N, T=4):
    alloc = Allocator()
    sp = Scratchpad(alloc, T)
    pA, pB, pC = alloc.alloc(N*N), alloc.alloc(N*N), alloc.alloc(N*N)

    def recurse(rA, cA, rB, cB, rC, cC, sz):
        if sz <= T:
            sp.compute_tile(pA + rA*N + cA, N, pB + rB*N + cB, N, pC + rC*N + cC, N)
            return
        h = sz // 2
        for drA, dcA, drB, dcB, drC, dcC in [
            (0,0,0,0,0,0), (0,0,0,h,0,h), (h,0,0,h,h,h), (h,0,0,0,h,0),
            (h,h,h,0,h,0), (h,h,h,h,h,h), (0,h,h,h,0,h), (0,h,h,0,0,0)]:
            recurse(rA+drA, cA+dcA, rB+drB, cB+dcB, rC+drC, cC+dcC, h)

    recurse(0, 0, 0, 0, 0, 0, N)
    sp.flush_C()
    regions = {
        'fast_A': (sp.fast['A'], sp.fast['A'] + T*T - 1),
        'fast_B': (sp.fast['B'], sp.fast['B'] + T*T - 1),
        'fast_C': (sp.fast['C'], sp.fast['C'] + T*T - 1),
        'main_A': (pA, pA + N*N - 1),
        'main_B': (pB, pB + N*N - 1),
        'main_C': (pC, pC + N*N - 1),
    }
    return alloc.log, regions, alloc.accesses, alloc.cost


# ============================================================================
# 2. Standard Strassen (stack-allocated temporaries)
# ============================================================================

def run_strassen(N, T=4):
    alloc = Allocator()
    sp = Scratchpad(alloc, T)
    pA, pB, pC = alloc.alloc(N*N), alloc.alloc(N*N), alloc.alloc(N*N)

    def add_mats(p1, s1, p2, s2, h):
        for i in range(h):
            for j in range(h):
                alloc.read(p1 + i*s1 + j)
                alloc.read(p2 + i*s2 + j)

    def recurse(pA_, sA, pB_, sB, pC_, sC, sz):
        if sz <= T:
            sp.compute_tile(pA_, sA, pB_, sB, pC_, sC)
            return
        h = sz // 2
        ckpt = alloc.push()
        SA, SB = alloc.alloc(h*h), alloc.alloc(h*h)
        M = [alloc.alloc(h*h) for _ in range(7)]

        A11, A12 = pA_, pA_ + h
        A21, A22 = pA_ + h*sA, pA_ + h*sA + h
        B11, B12 = pB_, pB_ + h
        B21, B22 = pB_ + h*sB, pB_ + h*sB + h

        add_mats(A11, sA, A22, sA, h); add_mats(B11, sB, B22, sB, h)
        recurse(SA, h, SB, h, M[0], h, h)
        add_mats(A21, sA, A22, sA, h)
        recurse(SA, h, B11, sB, M[1], h, h)
        add_mats(B12, sB, B22, sB, h)
        recurse(A11, sA, SB, h, M[2], h, h)
        add_mats(B21, sB, B11, sB, h)
        recurse(A22, sA, SB, h, M[3], h, h)
        add_mats(A11, sA, A12, sA, h)
        recurse(SA, h, B22, sB, M[4], h, h)
        add_mats(A21, sA, A11, sA, h); add_mats(B11, sB, B12, sB, h)
        recurse(SA, h, SB, h, M[5], h, h)
        add_mats(A12, sA, A22, sA, h); add_mats(B21, sB, B22, sB, h)
        recurse(SA, h, SB, h, M[6], h, h)

        sp.flush_C()
        def read_M(*indices):
            for i in range(h):
                for j in range(h):
                    for idx in indices:
                        alloc.read(M[idx] + i*h + j)
        read_M(0, 3, 4, 6)
        read_M(2, 4)
        read_M(1, 3)
        read_M(0, 1, 2, 5)
        alloc.pop(ckpt)

    recurse(pA, N, pB, N, pC, N, N)
    sp.flush_C()
    regions = {
        'fast_A': (sp.fast['A'], sp.fast['A'] + T*T - 1),
        'fast_B': (sp.fast['B'], sp.fast['B'] + T*T - 1),
        'fast_C': (sp.fast['C'], sp.fast['C'] + T*T - 1),
        'main_A': (pA, pA + N*N - 1),
        'main_B': (pB, pB + N*N - 1),
        'main_C': (pC, pC + N*N - 1),
    }
    if alloc.max_ptr > pC + N*N:
        regions['stack_tmp'] = (pC + N*N, alloc.max_ptr - 1)
    return alloc.log, regions, alloc.accesses, alloc.cost


# ============================================================================
# 3. Zero-Allocation Fused Strassen (ZAFS)
#    From gemini/efficient-strassen.md — virtual matrices resolved in L1
# ============================================================================

class VirtualScratchpad:
    """Resolves matrix addition DAGs on-the-fly during L1 fetch."""
    def __init__(self, alloc, T):
        self.alloc, self.T = alloc, T
        self.fast_A = alloc.alloc(T*T)
        self.fast_B = alloc.alloc(T*T)
        self.fast_C = alloc.alloc(T*T)

    def compute_fused_tile(self, pA, pB, pC, N, ops_A, ops_B, ops_C, r, c, k_off):
        T = self.T
        for i in range(T):
            for j in range(T):
                for sgn, rb, cb in ops_A:
                    self.alloc.read(pA + (rb + r + i)*N + cb + k_off + j)
        for i in range(T):
            for j in range(T):
                for sgn, rb, cb in ops_B:
                    self.alloc.read(pB + (rb + k_off + i)*N + cb + c + j)
        for i in range(T*T):
            self.alloc.read(self.fast_C + i)
        for i in range(T):
            for j in range(T):
                self.alloc.read(self.fast_C + i*T + j)
                for k in range(T):
                    self.alloc.read(self.fast_A + i*T + k)
                    self.alloc.read(self.fast_B + k*T + j)
        for sgn, rb, cb, is_first in ops_C:
            for i in range(T):
                for j in range(T):
                    self.alloc.read(self.fast_C + i*T + j)
                    if not is_first:
                        self.alloc.read(pC + (rb + r + i)*N + cb + c + j)


def run_fused_strassen(N, T=4):
    alloc = Allocator()
    sp = VirtualScratchpad(alloc, T)
    pA, pB, pC = alloc.alloc(N*N), alloc.alloc(N*N), alloc.alloc(N*N)

    h = N // 2
    q11, q12, q21, q22 = (0, 0), (0, h), (h, 0), (h, h)

    recipes = [
        ([(1, *q11), (1, *q22)], [(1, *q11), (1, *q22)], [(1, *q11, True), (1, *q22, True)]),
        ([(1, *q21), (1, *q22)], [(1, *q11)], [(1, *q21, True), (-1, *q22, False)]),
        ([(1, *q11)], [(1, *q12), (-1, *q22)], [(1, *q12, True), (1, *q22, False)]),
        ([(1, *q22)], [(1, *q21), (-1, *q11)], [(1, *q11, False), (1, *q21, False)]),
        ([(1, *q11), (1, *q12)], [(1, *q22)], [(-1, *q11, False), (1, *q12, False)]),
        ([(1, *q21), (-1, *q11)], [(1, *q11), (1, *q12)], [(1, *q22, False)]),
        ([(1, *q12), (-1, *q22)], [(1, *q21), (1, *q22)], [(1, *q11, False)]),
    ]

    for A_ops, B_ops, C_ops in recipes:
        for r, c in [(0,0), (0,T), (T,0), (T,T)]:
            sp.compute_fused_tile(pA, pB, pC, N, A_ops, B_ops, C_ops, r, c, k_off=0)
            C_ops_accum = [(sgn, rb, cb, False) for sgn, rb, cb, _ in C_ops]
            sp.compute_fused_tile(pA, pB, pC, N, A_ops, B_ops, C_ops_accum, r, c, k_off=T)

    regions = {
        'scratch': (1, 3*T*T),
        'main_A': (pA, pA + N*N - 1),
        'main_B': (pB, pB + N*N - 1),
        'main_C': (pC, pC + N*N - 1),
    }
    return alloc.log, regions, alloc.accesses, alloc.cost


# ============================================================================
# Plotting
# ============================================================================

REGION_COLORS = {
    'fast_A': 'tab:green', 'fast_B': 'tab:olive', 'fast_C': 'tab:cyan',
    'scratch': 'tab:cyan',
    'main_A': 'tab:red', 'main_B': 'tab:orange', 'main_C': 'tab:purple',
    'stack_tmp': 'tab:gray',
}


def plot_panel(ax, addrs, regions, algo_label, accesses, cost, y_max):
    if not addrs: return
    ys, xs = np.array(addrs), np.arange(len(addrs))
    for name, color in REGION_COLORS.items():
        if name not in regions:
            continue
        lo, hi = regions[name]
        mask = (ys >= lo) & (ys <= hi)
        if mask.any():
            ax.scatter(xs[mask], ys[mask], s=5, alpha=0.5, c=color,
                       label=f"{name} ({lo}..{hi})", rasterized=True, linewidths=0)
    ax.set_ylabel('Physical address', fontsize=10)
    ax.set_ylim(0, y_max)
    ax.set_title(f'{algo_label}\n{accesses:,} accesses, cost = {cost:,}', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc='center left', bbox_to_anchor=(1.01, 0.5), framealpha=0.95)


def main():
    N, T = 16, 4

    r_log, r_reg, r_acc, r_cost = run_rmm(N, T)
    s_log, s_reg, s_acc, s_cost = run_strassen(N, T)
    f_log, f_reg, f_acc, f_cost = run_fused_strassen(N, T)

    y_max = max(max(r_log), max(s_log), max(f_log)) + 10

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=False)
    plot_panel(axes[0], r_log, r_reg,
              f'RMM + Scratchpad (N={N}, tile={T})', r_acc, r_cost, y_max)
    plot_panel(axes[1], s_log, s_reg,
              f'Standard Strassen + Stack Alloc (N={N}, tile={T})', s_acc, s_cost, y_max)
    plot_panel(axes[2], f_log, f_reg,
              f'Zero-Allocation Fused Strassen (N={N}, tile={T})', f_acc, f_cost, y_max)
    axes[2].set_xlabel('Access index', fontsize=11)

    fig.suptitle(
        f'Memory Access Traces: RMM vs Strassen vs Fused Strassen  (N={N})\n'
        f'Cost: RMM={r_cost:,}  |  Strassen={s_cost:,}  |  Fused={f_cost:,}',
        fontsize=13, y=0.99)
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), 'efficient_strassen_trace_n16.png')
    plt.savefig(out, dpi=140, bbox_inches='tight')
    plt.close()

    print(f'Saved: {out}')
    print(f'\nN={N}, tile={T}:')
    print(f'  RMM:               {r_acc:>8,} accesses   cost = {r_cost:>10,}')
    print(f'  Standard Strassen: {s_acc:>8,} accesses   cost = {s_cost:>10,}')
    print(f'  Fused Strassen:    {f_acc:>8,} accesses   cost = {f_cost:>10,}')
    print(f'  Fused / Standard Strassen: {f_cost/s_cost:.2f}x')
    print(f'  RMM / Fused:               {r_cost/f_cost:.2f}x')


if __name__ == '__main__':
    main()
