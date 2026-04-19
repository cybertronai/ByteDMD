#!/Users/yaroslavvb/.local/bin/uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = ["matplotlib", "numpy"]
# ///
"""Self-contained script comparing CAQR vs Classical Householder QR.

Motivated by Demmel et al. (2008) "Communication-optimal parallel and
sequential QR and LU factorizations".

Generates caqr_vs_classical.png demonstrating the reduction in continuous
data movement (bytedmd cost) when utilizing TSQR panels and CAQR trailing updates.

Run:
    uv run --script caqr_vs_classical.py
"""

import bisect
import math
import os
from typing import List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# 1. Profile-Guided Oracle Allocator (Approximating Continuous LP Bounds)
# ============================================================================

class OracleAllocator:
    """
    Solves the Interval Packing Problem via Greedy Coloring.
    Phase 1: Records exact lifetimes and access frequencies of variables.
    Phase 2: Ranks variables by read-density and packs them into the optimal
             non-overlapping fixed physical addresses.
    Phase 3: Executes using the perfectly optimized static physical map.
    """
    def __init__(self):
        self.mode = 'trace'
        self.tick = 0
        self.intervals = {}
        self.next_v_addr = 1

        self.v_to_p = {}
        self.memory = {}
        self.cost = 0
        self.log = []

    def alloc(self, size: int) -> int:
        v = self.next_v_addr
        self.next_v_addr += size
        if self.mode == 'trace':
            for i in range(size):
                self.intervals[v + i] = [self.tick, self.tick, 0]
        self.tick += 1
        return v

    def write(self, v_addr: int, val: float) -> None:
        self.tick += 1
        if self.mode == 'trace':
            if v_addr not in self.intervals:
                self.intervals[v_addr] = [self.tick, self.tick, 0]
            else:
                self.intervals[v_addr][1] = self.tick
        else:
            p_addr = self.v_to_p.get(v_addr, v_addr)
            self.memory[p_addr] = val

    def read(self, v_addr: int) -> float:
        self.tick += 1
        if self.mode == 'trace':
            if v_addr not in self.intervals:
                self.intervals[v_addr] = [self.tick, self.tick, 0]
            self.intervals[v_addr][1] = self.tick
            self.intervals[v_addr][2] += 1
            return 0.0
        else:
            p_addr = self.v_to_p.get(v_addr, v_addr)
            self.log.append(p_addr)
            self.cost += math.isqrt(max(0, p_addr - 1)) + 1
            return self.memory.get(p_addr, 0.0)

    def compile(self):
        valid_addrs = [a for a, info in self.intervals.items() if info[2] > 0]

        def sort_key(a):
            start, end, reads = self.intervals[a]
            return (-reads, end - start)

        valid_addrs.sort(key=sort_key)

        tracks = []
        for a in valid_addrs:
            start, end, _ = self.intervals[a]
            assigned_p = -1
            for p, track in enumerate(tracks):
                idx = bisect.bisect_right(track, (start, float('inf')))
                overlap = False
                if idx > 0 and track[idx-1][1] >= start: overlap = True
                if idx < len(track) and track[idx][0] <= end: overlap = True

                if not overlap:
                    track.insert(idx, (start, end))
                    assigned_p = p + 1
                    break

            if assigned_p == -1:
                tracks.append([(start, end)])
                assigned_p = len(tracks)

            self.v_to_p[a] = assigned_p

        self.mode = 'execute'
        self.tick = 0
        self.next_v_addr = 1
        self.memory.clear()
        self.cost = 0
        self.log.clear()


def load_matrix(alloc, M, N):
    ptr = alloc.alloc(M * N)
    for i in range(M):
        for j in range(N):
            alloc.write(ptr + i * N + j, 1.0)
    return ptr

# ============================================================================
# 2. Base Math: Householder QR Components
# ============================================================================

def standard_qr(alloc, pA, pTau, M, N, strideA):
    """Standard column-by-column Householder QR."""
    for j in range(min(M, N)):
        # 1. Compute Norm
        norm_sq = 0.0
        for i in range(j, M):
            val = alloc.read(pA + i * strideA + j)
            norm_sq += val * val

        # 2. Householder vector v & tau
        a_jj = alloc.read(pA + j * strideA + j)
        u0 = a_jj + math.sqrt(max(0.0, norm_sq))
        alloc.write(pA + j * strideA + j, u0)
        alloc.write(pTau + j, 2.0)

        # 3. Update trailing matrix
        for k in range(j + 1, N):
            dot = 0.0
            for i in range(j, M):
                dot += alloc.read(pA + i * strideA + j) * alloc.read(pA + i * strideA + k)

            tau = alloc.read(pTau + j)
            scale = tau * dot

            for i in range(j, M):
                val = alloc.read(pA + i * strideA + k)
                v_i = alloc.read(pA + i * strideA + j)
                alloc.write(pA + i * strideA + k, val - scale * v_i)


def apply_Q_T(alloc, pV, pTau, M, N_V, strideV, pC, N_C, strideC):
    """Applies Q^T (from pV, pTau) to trailing matrix C."""
    for j in range(N_V):
        for k in range(N_C):
            dot = 0.0
            for i in range(j, M):
                dot += alloc.read(pV + i * strideV + j) * alloc.read(pC + i * strideC + k)

            tau = alloc.read(pTau + j)
            scale = tau * dot

            for i in range(j, M):
                val = alloc.read(pC + i * strideC + k)
                v_i = alloc.read(pV + i * strideV + j)
                alloc.write(pC + i * strideC + k, val - scale * v_i)

# ============================================================================
# 3. Communication-Avoiding QR (Sequential TSQR / CAQR)
# ============================================================================

def caqr_panel_step(alloc, pA, start_col, M, N, b, B_row, strideA):
    """Sequential CAQR on a panel of width b, and trailing matrix update."""
    P = M // B_row
    if P == 0: return

    pR_prev = alloc.alloc(b * b)
    C_N = N - start_col - b  # Trailing matrix width

    # === Step 1: Factor first block ===
    pBlock = alloc.alloc(B_row * b)
    pTau = alloc.alloc(b)

    for i in range(B_row):
        for j in range(b):
            alloc.write(pBlock + i * b + j, alloc.read(pA + i * strideA + start_col + j))

    standard_qr(alloc, pBlock, pTau, B_row, b, b)

    for i in range(b):
        for j in range(i, b):
            alloc.write(pR_prev + i * b + j, alloc.read(pBlock + i * b + j))
        for j in range(0, i):
            alloc.write(pR_prev + i * b + j, 0.0)

    # Trailing matrix update for first block
    if C_N > 0:
        for c_start in range(0, C_N, b):
            c_width = min(b, C_N - c_start)
            pCBlock = alloc.alloc(B_row * c_width)
            for i in range(B_row):
                for k in range(c_width):
                    alloc.write(pCBlock + i * c_width + k, alloc.read(pA + i * strideA + start_col + b + c_start + k))

            apply_Q_T(alloc, pBlock, pTau, B_row, b, b, pCBlock, c_width, c_width)

            for i in range(B_row):
                for k in range(c_width):
                    alloc.write(pA + i * strideA + start_col + b + c_start + k, alloc.read(pCBlock + i * c_width + k))

    # Write Q factor back to A
    for i in range(B_row):
        for j in range(b):
            alloc.write(pA + i * strideA + start_col + j, alloc.read(pBlock + i * b + j))

    # === Step 2: Flat tree reduction for remaining blocks ===
    for p in range(1, P):
        start_row = p * B_row

        # 2b x b matrix: top is R_prev, bottom is next block from A
        pTemp = alloc.alloc((b + B_row) * b)
        pTempTau = alloc.alloc(b)

        for i in range(b):
            for j in range(b):
                alloc.write(pTemp + i * b + j, alloc.read(pR_prev + i * b + j))
        for i in range(B_row):
            for j in range(b):
                alloc.write(pTemp + (b + i) * b + j, alloc.read(pA + (start_row + i) * strideA + start_col + j))

        standard_qr(alloc, pTemp, pTempTau, b + B_row, b, b)

        for i in range(b):
            for j in range(i, b):
                alloc.write(pR_prev + i * b + j, alloc.read(pTemp + i * b + j))
            for j in range(0, i):
                alloc.write(pR_prev + i * b + j, 0.0)

        for i in range(B_row):
            for j in range(b):
                alloc.write(pA + (start_row + i) * strideA + start_col + j, alloc.read(pTemp + (b + i) * b + j))

        # Trailing matrix update
        if C_N > 0:
            for c_start in range(0, C_N, b):
                c_width = min(b, C_N - c_start)
                pCTemp = alloc.alloc((b + B_row) * c_width)

                # Load C top part (updated by previous R)
                for i in range(b):
                    for k in range(c_width):
                        alloc.write(pCTemp + i * c_width + k, alloc.read(pA + i * strideA + start_col + b + c_start + k))
                # Load C bottom part (current block)
                for i in range(B_row):
                    for k in range(c_width):
                        alloc.write(pCTemp + (b + i) * c_width + k, alloc.read(pA + (start_row + i) * strideA + start_col + b + c_start + k))

                apply_Q_T(alloc, pTemp, pTempTau, b + B_row, b, b, pCTemp, c_width, c_width)

                # Write back to C
                for i in range(b):
                    for k in range(c_width):
                        alloc.write(pA + i * strideA + start_col + b + c_start + k, alloc.read(pCTemp + i * c_width + k))
                for i in range(B_row):
                    for k in range(c_width):
                        alloc.write(pA + (start_row + i) * strideA + start_col + b + c_start + k, alloc.read(pCTemp + (b + i) * c_width + k))

    # Write the very last R factor to the top b rows of A
    for i in range(b):
        for j in range(b):
            alloc.write(pA + i * strideA + start_col + j, alloc.read(pR_prev + i * b + j))

# ============================================================================
# 4. Execution Wrappers
# ============================================================================

def run_classical_qr(M, N):
    alloc = OracleAllocator()
    def execute():
        pA = load_matrix(alloc, M, N)
        pTau = alloc.alloc(N)
        standard_qr(alloc, pA, pTau, M, N, N)

        for i in range(M):
            for j in range(N):
                alloc.read(pA + i * N + j)
    execute()
    alloc.compile()
    execute()
    return alloc.log, alloc.cost


def run_caqr(M, N, b, B_row):
    alloc = OracleAllocator()
    def execute():
        pA = load_matrix(alloc, M, N)
        for start_col in range(0, N, b):
            width = min(b, N - start_col)
            caqr_panel_step(alloc, pA, start_col, M, N, width, B_row, N)

        for i in range(M):
            for j in range(N):
                alloc.read(pA + i * N + j)
    execute()
    alloc.compile()
    execute()
    return alloc.log, alloc.cost

# ============================================================================
# 5. Plotting
# ============================================================================

REGION_COLORS = {
    'L1 / Fast Cache (1..1024)': 'tab:green',
    'L2 / Med Cache (1025..4096)': 'tab:orange',
    'RAM / Slow Memory (4097+)': 'tab:red',
}

def classify(addr, regions):
    for label, (lo, hi) in regions.items():
        if lo <= addr <= hi: return label
    return 'other'

def plot_panel(ax, addrs, regions, algo_label, cost, y_max):
    xs = np.arange(len(addrs))
    ys = np.array(addrs)
    labels = np.array([classify(int(a), regions) for a in ys])

    for region, color in REGION_COLORS.items():
        if region not in regions: continue
        mask = labels == region
        if mask.any():
            ax.scatter(xs[mask], ys[mask], s=6, alpha=0.55, c=color,
                       label=region, rasterized=True, linewidths=0)

    ax.set_ylabel('Physical address', fontsize=11)
    ax.set_ylim(0, y_max)
    ax.set_title(f'{algo_label}  —  {len(addrs):,} reads, cost ∑⌈√addr⌉ = {cost:,}', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='center left', bbox_to_anchor=(1.01, 0.5), framealpha=0.95)

def main():
    M, N = 128, 32
    b, B_row = 8, 16  # Panel width=8, Row Block=16

    print(f"Tracing Classical QR (M={M}, N={N})...")
    class_log, class_cost = run_classical_qr(M, N)

    print(f"Tracing Sequential CAQR (M={M}, N={N}, b={b}, B_row={B_row})...")
    caqr_log, caqr_cost = run_caqr(M, N, b, B_row)

    regions = {
        'L1 / Fast Cache (1..1024)': (1, 1024),
        'L2 / Med Cache (1025..4096)': (1025, 4096),
        'RAM / Slow Memory (4097+)': (4097, float('inf'))
    }

    y_max = max(max(class_log), max(caqr_log)) + 10

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

    plot_panel(axes[0], class_log, regions, 'Classical Right-Looking Householder QR', class_cost, y_max)
    plot_panel(axes[1], caqr_log, regions, 'Communication-Avoiding QR (CAQR / Sequential TSQR)', caqr_cost, y_max)

    axes[1].set_xlabel('Access index (Time)', fontsize=11)
    fig.suptitle(f'Oracle LP-Bound Memory Allocation  —  Classical QR vs CAQR\n'
                 f'Energy Ratio (Classical / CAQR) = {class_cost / caqr_cost:.2f}×',
                 fontsize=14, y=1.02)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), 'caqr_vs_classical.png')
    plt.savefig(out, dpi=140, bbox_inches='tight')

    print(f'\nSaved: {out}')
    print(f'Classical QR — {len(class_log):,} accesses, cost {class_cost:,}')
    print(f'CAQR         — {len(caqr_log):,} accesses, cost {caqr_cost:,}')
    print(f'Energy ratio (Classical / CAQR)  = {class_cost / caqr_cost:.2f}×')

if __name__ == '__main__':
    main()
