# A cost model of complexity for the 21st century: ByteDMD

Data movement matters more than FLOPs. Recently accessed bytes can be cached, penalize non-local reads using the following cost model:

$$C=\sum_{b \in bytes} \sqrt{D(b)}$$

where $D(b)$ is the depth of byte $b$ in the LRU stack. Square-root is motivated by VLSI routing cost in 2D.

## Usage

```python
from bytedmd import bytedmd

def dot(a, b):
    return sum(i1*i2 for (i1,i2) in zip(a,b))

a = [0, 1]
b = [2, 3]

# dot product
assert dot(a,b) == 3

# ByteDMD cost of dot product
assert bytedmd(dot, (a, b)) == 12
```

## Motivation


Modern architectures spend more energy moving data than doing arithmetic, making FLOP counts an outdated cost metric. Bill Dally ([ACM Opinion](https://cacm.acm.org/opinion/on-the-model-of-computation-point/)) proposed penalizing data movement based on 2D spatial distance to the processor. To avoid manual spatial mapping, Ding and Smith ([Beyond Time Complexity, 2022](https://arxiv.org/abs/2203.02536)) automated this via Data Movement Distance (DMD): a rule treating memory as an LRU stack where reading a byte at depth $d$ costs $\sqrt{d}$, modeling a cache laid out in 2D.

To avoid floating point issues, we round up to the nearest integer.

![ByteDMD](docs/ceil_figure.svg)

This rounding corresponds to routing wire length on a 2D grid with LRU stack arranged in the following order.

![ByteDMD](docs/manhattan_figure.svg)

## Computation Model

An idealized processor operates directly on an element-level LRU stack. **Computations and writes are free; only memory reads incur a cost.**

- **Stack State:** Ordered from least recently used (bottom) to most recently used (top). Depth is measured in bytes from the top (topmost byte = depth 1). Multi-byte scalars are treated as contiguous blocks of bytes.
- **Eager initialization:** Arguments are loaded onto the stack left to right — the first argument sits at the top (depth 1). All input elements are live and addressable from the start.
- **Read Cost:** Reading a byte at depth $d$ costs $\lceil\sqrt{d}\rceil$.
- **Simultaneous pricing:** All inputs to an instruction are priced against the stack state *before* any LRU bumping. This guarantees commutativity: `Cost(a+b) == Cost(b+a)`.
- **Only live contribute to depth of the stack:** Any value that's dead (no longer used) is immediately removed from the stack and remaining elements slide up to close the gap. This models an optimal compiler that keeps the stack clamped to the active working set.

### Instruction Semantics

See [Instruction Set](docs/instruction_set.md) for the complete list of supported instructions.

For an instruction with inputs $x_1, \dots, x_m$ and outputs $y_1, \dots, y_n$ with $m\ge 1, n\ge 0$

1. **Price reads:** Evaluate $\sum C(x_j)$ simultaneously against the stack state *before* the instruction begins. All inputs see the same pre-instruction snapshot. Repeated inputs are charged per occurrence at the same depth (e.g., `a + a` charges `⌈√d⌉` twice where `d` is `a`'s pre-instruction depth).
2. **Update LRU:** Batch-move unique inputs to the top of the stack in read order. `b + c` and `c + b` yield the same cost (commutativity) but may differ in final stack order.
3. **Push outputs:** Allocate new output blocks and push them to the top at zero cost.

## Example Walkthrough

Consider the following function with three scalar arguments:

```python
def my_add(a, b, c):
    return (a + b) + c
```

**1. Initial Stack (left = top, right = bottom)** 
Arguments are loaded left to right — first argument at the top:
```text
[a, b, c]    ← a at depth 1, b at depth 2, c at depth 3
```

**2. First operation: `a + b`**  
Both operands are priced simultaneously against the initial stack:

$$C(a) + C(b) = \lceil\sqrt{1}\rceil + \lceil\sqrt{2}\rceil = 1 + 2 = 3$$

After LRU bumping and pushing the result `t = a + b`:
```text
[t, b, a, c]    ← t at depth 1, b at depth 2, a at depth 3, c at depth 4
```
Liveness analysis evicts `a` and `b` (their last use just happened):
```text
[t, c]    ← t at depth 1, c at depth 2
```

**3. Second operation: `t + c`**  
$$C(t) + C(c) = \lceil\sqrt{1}\rceil + \lceil\sqrt{2}\rceil = 1 + 2 = 3$$

**Total cost:** $3 + 3 = 6$. Trace: `[1, 2, 1, 2]`.


## Inspecting the IR

The tracer also emits a small **intermediate representation** that makes the
LRU stack lifecycle explicit. Three event types: `STORE k` (allocate vk on
top), `READ k@d` (read vk at depth d and LRU-bump), `OP name(vk@d, …)`
(summary of the preceding reads — this is what incurs cost). Op results are
materialized by the `STORE` that immediately follows the `OP`.

```python
from bytedmd import inspect_ir, format_ir, bytedmd

def matvec2(A, x):
    y0 = A[0][0]*x[0] + A[0][1]*x[1]
    y1 = A[1][0]*x[0] + A[1][1]*x[1]
    return [y0, y1]

print(format_ir(inspect_ir(matvec2, ([[1,2],[3,4]], [5,6]))))
```

```text
STORE v1                                # x[0] loaded first (deepest)
STORE v2                                # x[1]
STORE v3                                # A[0][0]
STORE v4                                # A[0][1]
STORE v5                                # A[1][0]
STORE v6                                # A[1][1]
  READ v3@4  cost=2                     # A[0][0] (left-to-right: A at top)
  READ v1@6  cost=3                     # x[0] at bottom
OP    mul(v3@4, v1@6)  cost=5           # A[0][0]*x[0]
STORE v7
  READ v4@5  cost=3                     # A[0][1] (v3 evicted after last use)
  READ v2@6  cost=3                     # x[1]
OP    mul(v4@5, v2@6)  cost=6           # A[0][1]*x[1]
STORE v8
  READ v7@3  cost=2                     # hot hit: v7 sank as v4, v2 entered
  READ v8@1  cost=1                     # hot hit: v8 still at top
OP    add(v7@3, v8@1)  cost=3           # y0
STORE v9
  READ v5@5  cost=3                     # A[1][0] (dead temps evicted)
  READ v1@3  cost=2                     # hot hit: x[0] still on stack
OP    mul(v5@5, v1@3)  cost=5
STORE v10
  READ v6@4  cost=2                     # A[1][1]
  READ v2@3  cost=2                     # hot hit: x[1]
OP    mul(v6@4, v2@3)  cost=4
STORE v11
  READ v10@2  cost=2
  READ v11@1  cost=1
OP    add(v10@2, v11@1)  cost=3         # y1
STORE v12
# total cost = 26
```

Note the left-to-right initialization: `A` elements (the first argument) sit
at the top of the stack, while `x` elements (the second argument) are
deeper. Liveness analysis aggressively evicts dead variables: after `v3`'s
single read, it is removed and remaining elements slide up. This keeps the
stack clamped to the active working set.

## ByteDMD benchmarks

See "benchmarks/" folder

### Matrix-vector (4x4 matrix, 4-vector)

| Algorithm | Operation | ByteDMD Cost |
|-----------|-----------|-------------|
| matvec (i-j) | y = A @ x | 157 |
| vecmat (j-i) | y = x^T @ A | 150 |

### Matrix multiply (4x4)

| Algorithm | Operation | ByteDMD Cost |
|-----------|-----------|-------------|
| naive matmul (i-j-k) | C = A @ B | 720 |

### microGPT single-token forward pass

Architecture: `vocab=4, embd=4, heads=2, head_dim=2, 1 layer, block_size=4`.
Based on [Karpathy's microGPT](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95).

| Algorithm | Operation | ByteDMD Cost |
|-----------|-----------|-------------|
| microGPT (1 layer, embd=4) | single token forward | 3214 |

# Reports

In-depth reports applying ByteDMD to specific algorithms and design questions:

- [Strassen vs naive matmul](docs/report-strassen-benchmarks/report.md) — at what matrix size does Strassen's recursive algorithm beat naive matmul under ByteDMD? Includes a crossover-point experiment.
- [Modern flash attention vs naive attention](docs/report-modern-flash-attention/report.md) — full sweep across sequence length, head dim, and block size showing flash attention's advantage growing as O(sqrt(N/Bk)) under ByteDMD while FLOPs see no benefit. Uses an optimised tracer (`bytedmd_fast.py`).
- [Antigravity flash attention experiments](docs/report-antigravity-flash-attention/report.md) — alternative flash attention implementations and their ByteDMD costs.
- [Attention benchmark notes](benchmarks/attention_report.md) — the small-scale flash vs naive results that motivated the modern-attention deep dive.

<!-- GRID-REPORT-START -->

## Heuristic Grid for ByteDMD-Style Metrics

This experiment compares a concrete no-free-compaction 2D cost against SpaceDMD and the two abstract ByteDMD heuristics on a small suite of workloads.

Every traced metric cell finished under 4.790 seconds on this run.

### Algorithms

Rows are grouped to follow the dev-branch-style ordering: matmul, attention/vector fusion, matvec/traversal/sparse, FFT, stencil, convolution, sorting/DP/APSP, dense solve, LU, Cholesky, and QR.

| Algorithm | Workload | Implementation |
| --- | --- | --- |
| Naive Matmul | 16x16 | standard i-j-k triple loop |
| Tiled Matmul | 16x16, tile=4 | one explicit blocking level |
| Recursive Matmul | 16x16 | 8-way cache-oblivious recursion |
| Recursive In-Place (Lex) | 16x16 | manual in-place schedule, lexicographic order |
| Recursive In-Place (Gray) | 16x16 | manual in-place schedule, Gray-code order |
| Strassen | 16x16 | leaf size 1 to expose temporary traffic |
| Fused Strassen | 16x16, leaf=8 | zero-allocation virtual sums with direct accumulation into C |
| Naive Attention (d=2) | N=32, d=2 | materializes the full score matrix |
| Flash Attention (Bk=8) | N=32, d=2, Bq=8, Bk=8 | double-tiled Q/KV blocks with wider KV tiles |
| Naive Attention (d=4) | N=32, d=4 | materializes the full score matrix |
| Flash Attention | N=32, d=4, Bq=8, Bk=4 | double-tiled Q/KV blocks with snake KV order |
| LayerNorm (Unfused) | N=1024 | three full vector passes for mean, variance, and normalize |
| LayerNorm (Fused) | N=1024 | Welford statistics fused into two vector passes |
| Matvec | 32x32 by 32 | row-wise matrix-vector baseline |
| Vecmat | 32 by 32x32 | column-oriented access order |
| Matvec Row | 64x64 by 64 | row-major matrix-vector multiply |
| Matvec Column | 64 by 64x64 | column-major vector-matrix multiply |
| Matrix Powers (Naive) | 32x32, s=4 | re-reads A for each successive dense matvec |
| Matrix Powers (CA) | 32x32, s=4, block=4 | row-blocked communication-avoiding proxy for chained matvecs |
| SpMV CSR (Banded) | N=64, bandwidth=3 | sparse matrix-vector multiply with clustered indirect reads |
| SpMV CSR (Random) | N=64, nnz/row=7 | sparse matrix-vector multiply with randomized indirect reads |
| Row Scan | 64x64 | row-major traversal sum |
| Column Scan | 64x64 | column-major traversal sum |
| Transpose (Naive) | 32x32 | direct row-major transpose copy |
| Transpose (Blocked) | 32x32, block=8 | blocked transpose copy |
| Transpose (Recursive) | 32x32, leaf=8 | cache-oblivious recursive transpose |
| FFT (Iterative) | N=1024 | iterative radix-2 Cooley-Tukey |
| FFT (Recursive) | N=1024 | recursive radix-2 Cooley-Tukey |
| Stencil (Naive) | 32x32, one sweep | row-major Jacobi stencil |
| Stencil (Recursive) | 32x32, one sweep, leaf=8 | tile-recursive Jacobi stencil |
| Stencil (Time-Naive) | 32x32, T=4 | four full Jacobi sweeps with fresh intermediate buffers |
| Stencil (Time-Diamond) | 32x32, T=4, block=8 | space-time tiled Jacobi proxy with per-tile halo reuse |
| Spatial Conv (2D, 16x16) | 16x16, kernel=5x5 | same-size zero-padded spatial convolution |
| Spatial Conv (2D, 32x32) | 32x32, kernel=5x5 | same-size zero-padded spatial convolution |
| Regular Conv | 16x16, kernel=3x3, Cin=4, Cout=4 | direct same-size convolution over 4 input/output channels |
| FFT Conv (1D) | N=32 | circular 1D convolution via recursive FFT |
| FFT Conv (2D) | 16x16, kernel=5x5, pad=32x32 | same-size convolution via zero-padded recursive 2D FFT |
| Mergesort | N=64 | top-down mergesort with tracked comparisons |
| Bitonic Sort | N=64 | data-oblivious sorting network with butterfly compare-swaps |
| LCS DP | 32x32 | dynamic programming longest common subsequence |
| Floyd-Warshall (Naive) | V=32 | standard k-i-j all-pairs shortest paths |
| Floyd-Warshall (Recursive) | V=32, leaf=8 | recursive blocked APSP over k/i/j ranges |
| Gaussian Elimination | N=24 | dense solve without pivoting |
| Gauss-Jordan Inverse | N=16 | dense matrix inverse without pivoting |
| LU (No Pivot) | N=24 | Doolittle LU without row swaps |
| LU (Blocked) | N=24, block=4 | panel/TRSM/trailing-update LU without pivoting |
| LU (Recursive) | N=24, leaf=6 | recursive block LU without pivoting |
| LU (Partial Pivot) | N=24 | partial pivoting with row-copy traffic |
| Cholesky | N=24 | lower-triangular Cholesky factorization |
| Cholesky (Blocked) | N=24, block=4 | tile-oriented Cholesky factorization |
| Cholesky (Recursive) | N=24, leaf=6 | recursive block Cholesky factorization |
| Cholesky (Right-Looking) | N=24 | eager trailing-update Cholesky for read/write asymmetry comparison |
| Householder QR | 48x12 | unblocked Householder QR returning R |
| Blocked QR | 48x12, block=4 | panel-blocked Householder QR with delayed trailing updates |
| TSQR | 48x12, leaf_rows=12 | tall-skinny recursive QR returning the final R |

### Measures

- `SpaceDMD`: density-ranked spatial liveness, now with inputs first read from a separate argument stack and only later re-read from the geometric stack.
- `ByteDMD-live`: aggressive live-only compaction on the geometric stack, with the same separate argument-stack first-touch rule.
- `Manual-2D`: hand-scheduled fixed-address implementations with separate scratch and argument/output regions under the 2D `ceil(sqrt(addr))` cost model.
- `ByteDMD-classic`: graveyard model with no reclamation on the geometric stack, again after the first-touch argument-stack read.

All four columns now include a terminal readback of the full returned value, so the table prices both computation and the final result extraction.

SpaceDMD globally ranks geometric-stack variables by access density (`access_count / lifespan`) and then charges each read by that variable's rank among the currently live variables; untouched inputs are priced separately on the argument stack until their first promotion.

### Interpretation Notes

- The trace models now have an explicit first-touch boundary: inputs are priced on an argument stack on first use, then promoted into the geometric stack for later re-use. Manual kernels mirror this with separate scratch and argument/output regions.
- SpaceDMD is intentionally order-blind once data is in the geometric stack: pure permutations with the same multiset of reads, such as `Matvec` vs `Vecmat` or `Row Scan` vs `Column Scan`, can collapse to identical SpaceDMD costs even when `Manual-2D` separates them strongly.
- Single-touch kernels such as the transpose trio are a deliberate failure mode for SpaceDMD. When every cell is read once, the metric collapses to the read count (`n^2` here) rather than the physical `ceil(sqrt(addr))` placement cost.
- The blocked LU and blocked QR rows are panel-update variants, not cosmetic loop chunking. If they still land close to their unblocked counterparts, that should be read as an empirical result rather than a placeholder implementation.
- `Recursive LU` and `Recursive Cholesky` here are copy-based block decompositions built out of `_slice_copy`, triangular solves, and Schur complements. Their costs therefore include explicit materialization traffic and should not be read as in-place communication-optimal factorizations.
- `Matrix Powers (CA)` and `Stencil (Time-Diamond)` are locality proxies rather than full communication-optimal solvers. They preserve the intended block-local dataflow but should be read as stress tests for the heuristics, not numerically tuned production kernels.
- These numbers are implementation-specific to this branch. Comparing them directly to other branches that use different schedules, such as right-looking versus left-looking factorizations or different Strassen fusions, can change the measured locality substantially even when the math is the same.
- SpaceDMD can mis-rank virtual/intermediate-heavy traces such as `Strassen` versus `Fused Strassen`, because it scores density-ranked liveness rather than concrete placement.
- The ranking table has a split verdict: `ByteDMD-live` has the best rank correlation while `ByteDMD-classic` has the best scaled MAPE. In other words, the heuristic that orders rows best is not the same one that matches magnitudes best.

Attention uses proxy `max`, `exp`, and reciprocal operators with the same read arity as the real kernels, so the table focuses on data movement rather than numerical fidelity.

### Results Grid

| Algorithm | SpaceDMD | ByteDMD-live | Manual-2D | ByteDMD-classic |
| --- | --- | --- | --- | --- |
| Naive Matmul | 75,573 | 119,088 | 138,486 | 178,319 |
| Tiled Matmul | 90,626 | 89,445 | 82,574 | 141,169 |
| Recursive Matmul | 104,162 | 95,371 | 102,056 | 149,081 |
| Recursive In-Place (Lex) | 99,017 | 83,216 | 239,777 | 131,240 |
| Recursive In-Place (Gray) | 87,471 | 78,313 | 239,777 | 124,441 |
| Strassen | 129,252 | 202,785 | 210,953 | 341,157 |
| Fused Strassen | 177,235 | 183,970 | 147,360 | 310,614 |
| Naive Attention (d=2) | 121,175 | 182,223 | 419,566 | 273,037 |
| Flash Attention (Bk=8) | 79,155 | 94,106 | 117,651 | 143,791 |
| Naive Attention (d=4) | 227,596 | 298,016 | 804,056 | 462,791 |
| Flash Attention | 183,674 | 195,469 | 286,628 | 317,851 |
| LayerNorm (Unfused) | 81,500 | 153,618 | 122,384 | 206,144 |
| LayerNorm (Fused) | 95,319 | 134,068 | 123,398 | 178,185 |
| Matvec | 29,911 | 36,134 | 29,890 | 39,572 |
| Vecmat | 23,195 | 29,418 | 29,890 | 32,856 |
| Matvec Row | 213,297 | 244,553 | 213,156 | 262,597 |
| Matvec Column | 157,421 | 188,677 | 213,156 | 206,721 |
| Matrix Powers (Naive) | 107,592 | 178,637 | 237,120 | 263,206 |
| Matrix Powers (CA) | 116,600 | 144,919 | 211,932 | 197,497 |
| SpMV CSR (Banded) | 9,602 | 11,410 | 17,869 | 12,270 |
| SpMV CSR (Random) | 11,005 | 14,116 | 18,518 | 15,979 |
| Row Scan | 180,896 | 180,896 | 180,960 | 180,896 |
| Column Scan | 125,024 | 125,024 | 180,960 | 125,024 |
| Transpose (Naive) | 32,980 | 58,989 | 62,813 | 58,989 |
| Transpose (Blocked) | 32,324 | 58,522 | 62,813 | 58,522 |
| Transpose (Recursive) | 32,268 | 58,464 | 62,813 | 58,464 |
| FFT (Iterative) | 266,902 | 410,458 | 467,423 | 602,976 |
| FFT (Recursive) | 136,833 | 213,586 | 467,423 | 311,419 |
| Stencil (Naive) | 64,584 | 91,862 | 142,215 | 118,313 |
| Stencil (Recursive) | 53,942 | 83,339 | 142,215 | 108,221 |
| Stencil (Time-Naive) | 290,346 | 512,219 | 877,793 | 719,588 |
| Stencil (Time-Diamond) | 510,495 | 854,037 | 2,442,625 | 1,441,257 |
| Spatial Conv (2D, 16x16) | 83,083 | 112,862 | 165,557 | 162,940 |
| Spatial Conv (2D, 32x32) | 421,216 | 542,936 | 1,367,491 | 849,599 |
| Regular Conv | 893,512 | 994,160 | 1,958,775 | 1,548,723 |
| FFT Conv (1D) | 3,148 | 5,071 | 5,924 | 6,263 |
| FFT Conv (2D) | 193,770 | 395,662 | 2,812,578 | 644,543 |
| Mergesort | 2,157 | 3,410 | 8,574 | 3,977 |
| Bitonic Sort | 10,283 | 13,095 | 8,842 | 15,899 |
| LCS DP | 23,238 | 30,392 | 138,668 | 30,392 |
| Floyd-Warshall (Naive) | 1,296,932 | 1,600,250 | 2,208,605 | 2,467,170 |
| Floyd-Warshall (Recursive) | 1,289,599 | 1,528,176 | 2,208,605 | 2,371,997 |
| Gaussian Elimination | 144,330 | 174,839 | 149,098 | 264,399 |
| Gauss-Jordan Inverse | 138,241 | 291,004 | 197,639 | 447,568 |
| LU (No Pivot) | 176,646 | 183,748 | 152,609 | 285,388 |
| LU (Blocked) | 168,823 | 205,254 | 152,320 | 292,112 |
| LU (Recursive) | 143,081 | 154,745 | 170,273 | 242,879 |
| LU (Partial Pivot) | 211,252 | 211,749 | 196,063 | 333,458 |
| Cholesky | 58,449 | 60,233 | 103,898 | 87,161 |
| Cholesky (Blocked) | 59,135 | 65,048 | 113,956 | 99,660 |
| Cholesky (Recursive) | 81,590 | 97,705 | 106,249 | 149,742 |
| Cholesky (Right-Looking) | 70,155 | 76,858 | 246,125 | 111,975 |
| Householder QR | 191,725 | 235,920 | 210,511 | 372,359 |
| Blocked QR | 199,250 | 231,744 | 210,847 | 370,913 |
| TSQR | 200,413 | 251,200 | 308,395 | 393,651 |

### Heuristic Ranking Against Manual-2D

| Heuristic | Spearman rho | Scaled MAPE |
| --- | --- | --- |
| SpaceDMD | 0.848 | 51.4% |
| ByteDMD-live | 0.853 | 55.7% |
| ByteDMD-classic | 0.846 | 46.3% |

### Trace Diagnostics

These follow the dev-branch style plots for the current `ByteDMD-live` path: every algorithm gets an inline reuse-distance-per-load scatter plot and a working-set-size-over-time step plot on this page.

A tab-separated summary is also saved as [`diagnostics/diagnostics_summary.tsv`](./experiments/grid/diagnostics/diagnostics_summary.tsv).

| Algorithm | Peak live | Max reuse | Median reuse |
| --- | --- | --- | --- |
| Naive Matmul | 770 | 768 | 34 |
| Tiled Matmul | 771 | 768 | 11 |
| Recursive Matmul | 896 | 768 | 11 |
| Recursive In-Place (Lex) | 770 | 768 | 6 |
| Recursive In-Place (Gray) | 770 | 768 | 7 |
| Strassen | 1,194 | 1,023 | 13 |
| Fused Strassen | 774 | 773 | 8 |
| Naive Attention (d=2) | 2,309 | 2,082 | 4 |
| Flash Attention (Bk=8) | 409 | 343 | 4 |
| Naive Attention (d=4) | 2,565 | 2,146 | 4 |
| Flash Attention | 723 | 593 | 5 |
| LayerNorm (Unfused) | 2,057 | 2,054 | 4 |
| LayerNorm (Fused) | 2,058 | 2,055 | 4 |
| Matvec | 1,090 | 1,056 | 25 |
| Vecmat | 1,090 | 1,056 | 13 |
| Matvec Row | 4,226 | 4,160 | 49 |
| Matvec Column | 4,226 | 4,160 | 23 |
| Matrix Powers (Naive) | 1,122 | 1,088 | 65 |
| Matrix Powers (CA) | 1,473 | 1,024 | 66 |
| SpMV CSR (Banded) | 567 | 560 | 14 |
| SpMV CSR (Random) | 579 | 569 | 28 |
| Row Scan | 4,098 | 4,096 | 1 |
| Column Scan | 4,098 | 4,096 | 1 |
| Transpose (Naive) | 2,048 | 2,047 | 836 |
| Transpose (Blocked) | 2,048 | 2,047 | 821 |
| Transpose (Recursive) | 2,048 | 2,047 | 832 |
| FFT (Iterative) | 2,051 | 2,048 | 3 |
| FFT (Recursive) | 3,073 | 2,559 | 3 |
| Stencil (Naive) | 2,049 | 2,047 | 9 |
| Stencil (Recursive) | 2,049 | 2,047 | 9 |
| Stencil (Time-Naive) | 3,074 | 2,048 | 9 |
| Stencil (Time-Diamond) | 2,817 | 2,385 | 9 |
| Spatial Conv (2D, 16x16) | 539 | 537 | 33 |
| Spatial Conv (2D, 32x32) | 2,075 | 2,073 | 34 |
| Regular Conv | 2,194 | 2,192 | 51 |
| FFT Conv (1D) | 225 | 127 | 3 |
| FFT Conv (2D) | 5,715 | 3,096 | 3 |
| Mergesort | 192 | 159 | 3 |
| Bitonic Sort | 130 | 127 | 61 |
| LCS DP | 1,155 | 1,152 | 4 |
| Floyd-Warshall (Naive) | 2,050 | 2,047 | 4 |
| Floyd-Warshall (Recursive) | 2,050 | 2,047 | 4 |
| Gaussian Elimination | 1,228 | 1,200 | 25 |
| Gauss-Jordan Inverse | 772 | 767 | 44 |
| LU (No Pivot) | 1,431 | 1,152 | 29 |
| LU (Blocked) | 2,710 | 1,409 | 10 |
| LU (Recursive) | 2,115 | 1,325 | 15 |
| LU (Partial Pivot) | 1,431 | 1,151 | 37 |
| Cholesky | 604 | 599 | 4 |
| Cholesky (Blocked) | 1,266 | 1,151 | 9 |
| Cholesky (Recursive) | 1,467 | 905 | 13 |
| Cholesky (Right-Looking) | 1,454 | 1,151 | 13 |
| Householder QR | 1,256 | 1,201 | 4 |
| Blocked QR | 1,349 | 1,338 | 4 |
| TSQR | 1,208 | 719 | 4 |

### Diagnostic Gallery

#### Naive Matmul

Peak live = 770. Max reuse = 768. Median reuse = 34.

<p align="center">
  <img src="experiments/grid/diagnostics/naive-matmul-16_liveset.png" alt="Naive Matmul working-set size over time" width="49%" />
  <img src="experiments/grid/diagnostics/naive-matmul-16_reuse_distance.png" alt="Naive Matmul reuse distance per load" width="49%" />
</p>

#### Tiled Matmul

Peak live = 771. Max reuse = 768. Median reuse = 11.

<p align="center">
  <img src="experiments/grid/diagnostics/tiled-matmul-16_liveset.png" alt="Tiled Matmul working-set size over time" width="49%" />
  <img src="experiments/grid/diagnostics/tiled-matmul-16_reuse_distance.png" alt="Tiled Matmul reuse distance per load" width="49%" />
</p>

#### Recursive Matmul

Peak live = 896. Max reuse = 768. Median reuse = 11.

<p align="center">
  <img src="experiments/grid/diagnostics/rmm-16_liveset.png" alt="Recursive Matmul working-set size over time" width="49%" />
  <img src="experiments/grid/diagnostics/rmm-16_reuse_distance.png" alt="Recursive Matmul reuse distance per load" width="49%" />
</p>

#### Recursive In-Place (Lex)

Peak live = 770. Max reuse = 768. Median reuse = 6.

<p align="center">
  <img src="experiments/grid/diagnostics/rmm-lex-16_liveset.png" alt="Recursive In-Place (Lex) working-set size over time" width="49%" />
  <img src="experiments/grid/diagnostics/rmm-lex-16_reuse_distance.png" alt="Recursive In-Place (Lex) reuse distance per load" width="49%" />
</p>

#### Recursive In-Place (Gray)

Peak live = 770. Max reuse = 768. Median reuse = 7.

<p align="center">
  <img src="experiments/grid/diagnostics/rmm-gray-16_liveset.png" alt="Recursive In-Place (Gray) working-set size over time" width="49%" />
  <img src="experiments/grid/diagnostics/rmm-gray-16_reuse_distance.png" alt="Recursive In-Place (Gray) reuse distance per load" width="49%" />
</p>

#### Strassen

Peak live = 1,194. Max reuse = 1,023. Median reuse = 13.

<p align="center">
  <img src="experiments/grid/diagnostics/strassen-16_liveset.png" alt="Strassen working-set size over time" width="49%" />
  <img src="experiments/grid/diagnostics/strassen-16_reuse_distance.png" alt="Strassen reuse distance per load" width="49%" />
</p>

#### Fused Strassen

Peak live = 774. Max reuse = 773. Median reuse = 8.

<p align="center">
  <img src="experiments/grid/diagnostics/fused-strassen-16_liveset.png" alt="Fused Strassen working-set size over time" width="49%" />
  <img src="experiments/grid/diagnostics/fused-strassen-16_reuse_distance.png" alt="Fused Strassen reuse distance per load" width="49%" />
</p>

#### Naive Attention (d=2)

Peak live = 2,309. Max reuse = 2,082. Median reuse = 4.

<p align="center">
  <img src="experiments/grid/diagnostics/naive-attention-32x2_liveset.png" alt="Naive Attention (d=2) working-set size over time" width="49%" />
  <img src="experiments/grid/diagnostics/naive-attention-32x2_reuse_distance.png" alt="Naive Attention (d=2) reuse distance per load" width="49%" />
</p>

#### Flash Attention (Bk=8)

Peak live = 409. Max reuse = 343. Median reuse = 4.

<p align="center">
  <img src="experiments/grid/diagnostics/flash-attention-32x2-b8_liveset.png" alt="Flash Attention (Bk=8) working-set size over time" width="49%" />
  <img src="experiments/grid/diagnostics/flash-attention-32x2-b8_reuse_distance.png" alt="Flash Attention (Bk=8) reuse distance per load" width="49%" />
</p>

#### Naive Attention (d=4)

Peak live = 2,565. Max reuse = 2,146. Median reuse = 4.

<p align="center">
  <img src="experiments/grid/diagnostics/regular-attention-32x4_liveset.png" alt="Naive Attention (d=4) working-set size over time" width="49%" />
  <img src="experiments/grid/diagnostics/regular-attention-32x4_reuse_distance.png" alt="Naive Attention (d=4) reuse distance per load" width="49%" />
</p>

#### Flash Attention

Peak live = 723. Max reuse = 593. Median reuse = 5.

<p align="center">
  <img src="experiments/grid/diagnostics/flash-attention-32x4_liveset.png" alt="Flash Attention working-set size over time" width="49%" />
  <img src="experiments/grid/diagnostics/flash-attention-32x4_reuse_distance.png" alt="Flash Attention reuse distance per load" width="49%" />
</p>

#### LayerNorm (Unfused)

Peak live = 2,057. Max reuse = 2,054. Median reuse = 4.

<p align="center">
  <img src="experiments/grid/diagnostics/layernorm-unfused-1024_liveset.png" alt="LayerNorm (Unfused) working-set size over time" width="49%" />
  <img src="experiments/grid/diagnostics/layernorm-unfused-1024_reuse_distance.png" alt="LayerNorm (Unfused) reuse distance per load" width="49%" />
</p>

#### LayerNorm (Fused)

Peak live = 2,058. Max reuse = 2,055. Median reuse = 4.

<p align="center">
  <img src="experiments/grid/diagnostics/layernorm-fused-1024_liveset.png" alt="LayerNorm (Fused) working-set size over time" width="49%" />
  <img src="experiments/grid/diagnostics/layernorm-fused-1024_reuse_distance.png" alt="LayerNorm (Fused) reuse distance per load" width="49%" />
</p>

#### Matvec

Peak live = 1,090. Max reuse = 1,056. Median reuse = 25.

<p align="center">
  <img src="experiments/grid/diagnostics/matvec-32_liveset.png" alt="Matvec working-set size over time" width="49%" />
  <img src="experiments/grid/diagnostics/matvec-32_reuse_distance.png" alt="Matvec reuse distance per load" width="49%" />
</p>

#### Vecmat

Peak live = 1,090. Max reuse = 1,056. Median reuse = 13.

<p align="center">
  <img src="experiments/grid/diagnostics/vecmat-32_liveset.png" alt="Vecmat working-set size over time" width="49%" />
  <img src="experiments/grid/diagnostics/vecmat-32_reuse_distance.png" alt="Vecmat reuse distance per load" width="49%" />
</p>

#### Matvec Row

Peak live = 4,226. Max reuse = 4,160. Median reuse = 49.

<p align="center">
  <img src="experiments/grid/diagnostics/matvec-row-64_liveset.png" alt="Matvec Row working-set size over time" width="49%" />
  <img src="experiments/grid/diagnostics/matvec-row-64_reuse_distance.png" alt="Matvec Row reuse distance per load" width="49%" />
</p>

#### Matvec Column

Peak live = 4,226. Max reuse = 4,160. Median reuse = 23.

<p align="center">
  <img src="experiments/grid/diagnostics/matvec-col-64_liveset.png" alt="Matvec Column working-set size over time" width="49%" />
  <img src="experiments/grid/diagnostics/matvec-col-64_reuse_distance.png" alt="Matvec Column reuse distance per load" width="49%" />
</p>

#### Matrix Powers (Naive)

Peak live = 1,122. Max reuse = 1,088. Median reuse = 65.

<p align="center">
  <img src="experiments/grid/diagnostics/matrix-powers-naive-32-s4_liveset.png" alt="Matrix Powers (Naive) working-set size over time" width="49%" />
  <img src="experiments/grid/diagnostics/matrix-powers-naive-32-s4_reuse_distance.png" alt="Matrix Powers (Naive) reuse distance per load" width="49%" />
</p>

#### Matrix Powers (CA)

Peak live = 1,473. Max reuse = 1,024. Median reuse = 66.

<p align="center">
  <img src="experiments/grid/diagnostics/matrix-powers-ca-32-s4_liveset.png" alt="Matrix Powers (CA) working-set size over time" width="49%" />
  <img src="experiments/grid/diagnostics/matrix-powers-ca-32-s4_reuse_distance.png" alt="Matrix Powers (CA) reuse distance per load" width="49%" />
</p>

#### SpMV CSR (Banded)

Peak live = 567. Max reuse = 560. Median reuse = 14.

<p align="center">
  <img src="experiments/grid/diagnostics/spmv-csr-banded-64_liveset.png" alt="SpMV CSR (Banded) working-set size over time" width="49%" />
  <img src="experiments/grid/diagnostics/spmv-csr-banded-64_reuse_distance.png" alt="SpMV CSR (Banded) reuse distance per load" width="49%" />
</p>

#### SpMV CSR (Random)

Peak live = 579. Max reuse = 569. Median reuse = 28.

<p align="center">
  <img src="experiments/grid/diagnostics/spmv-csr-random-64_liveset.png" alt="SpMV CSR (Random) working-set size over time" width="49%" />
  <img src="experiments/grid/diagnostics/spmv-csr-random-64_reuse_distance.png" alt="SpMV CSR (Random) reuse distance per load" width="49%" />
</p>

#### Row Scan

Peak live = 4,098. Max reuse = 4,096. Median reuse = 1.

<p align="center">
  <img src="experiments/grid/diagnostics/scan-row-64_liveset.png" alt="Row Scan working-set size over time" width="49%" />
  <img src="experiments/grid/diagnostics/scan-row-64_reuse_distance.png" alt="Row Scan reuse distance per load" width="49%" />
</p>

#### Column Scan

Peak live = 4,098. Max reuse = 4,096. Median reuse = 1.

<p align="center">
  <img src="experiments/grid/diagnostics/scan-column-64_liveset.png" alt="Column Scan working-set size over time" width="49%" />
  <img src="experiments/grid/diagnostics/scan-column-64_reuse_distance.png" alt="Column Scan reuse distance per load" width="49%" />
</p>

#### Transpose (Naive)

Peak live = 2,048. Max reuse = 2,047. Median reuse = 836.

<p align="center">
  <img src="experiments/grid/diagnostics/transpose-naive-32_liveset.png" alt="Transpose (Naive) working-set size over time" width="49%" />
  <img src="experiments/grid/diagnostics/transpose-naive-32_reuse_distance.png" alt="Transpose (Naive) reuse distance per load" width="49%" />
</p>

#### Transpose (Blocked)

Peak live = 2,048. Max reuse = 2,047. Median reuse = 821.

<p align="center">
  <img src="experiments/grid/diagnostics/transpose-blocked-32_liveset.png" alt="Transpose (Blocked) working-set size over time" width="49%" />
  <img src="experiments/grid/diagnostics/transpose-blocked-32_reuse_distance.png" alt="Transpose (Blocked) reuse distance per load" width="49%" />
</p>

#### Transpose (Recursive)

Peak live = 2,048. Max reuse = 2,047. Median reuse = 832.

<p align="center">
  <img src="experiments/grid/diagnostics/transpose-recursive-32_liveset.png" alt="Transpose (Recursive) working-set size over time" width="49%" />
  <img src="experiments/grid/diagnostics/transpose-recursive-32_reuse_distance.png" alt="Transpose (Recursive) reuse distance per load" width="49%" />
</p>

#### FFT (Iterative)

Peak live = 2,051. Max reuse = 2,048. Median reuse = 3.

<p align="center">
  <img src="experiments/grid/diagnostics/fft-iterative-1024_liveset.png" alt="FFT (Iterative) working-set size over time" width="49%" />
  <img src="experiments/grid/diagnostics/fft-iterative-1024_reuse_distance.png" alt="FFT (Iterative) reuse distance per load" width="49%" />
</p>

#### FFT (Recursive)

Peak live = 3,073. Max reuse = 2,559. Median reuse = 3.

<p align="center">
  <img src="experiments/grid/diagnostics/fft-recursive-1024_liveset.png" alt="FFT (Recursive) working-set size over time" width="49%" />
  <img src="experiments/grid/diagnostics/fft-recursive-1024_reuse_distance.png" alt="FFT (Recursive) reuse distance per load" width="49%" />
</p>

#### Stencil (Naive)

Peak live = 2,049. Max reuse = 2,047. Median reuse = 9.

<p align="center">
  <img src="experiments/grid/diagnostics/jacobi-naive-32_liveset.png" alt="Stencil (Naive) working-set size over time" width="49%" />
  <img src="experiments/grid/diagnostics/jacobi-naive-32_reuse_distance.png" alt="Stencil (Naive) reuse distance per load" width="49%" />
</p>

#### Stencil (Recursive)

Peak live = 2,049. Max reuse = 2,047. Median reuse = 9.

<p align="center">
  <img src="experiments/grid/diagnostics/jacobi-recursive-32_liveset.png" alt="Stencil (Recursive) working-set size over time" width="49%" />
  <img src="experiments/grid/diagnostics/jacobi-recursive-32_reuse_distance.png" alt="Stencil (Recursive) reuse distance per load" width="49%" />
</p>

#### Stencil (Time-Naive)

Peak live = 3,074. Max reuse = 2,048. Median reuse = 9.

<p align="center">
  <img src="experiments/grid/diagnostics/stencil-time-naive-32-t4_liveset.png" alt="Stencil (Time-Naive) working-set size over time" width="49%" />
  <img src="experiments/grid/diagnostics/stencil-time-naive-32-t4_reuse_distance.png" alt="Stencil (Time-Naive) reuse distance per load" width="49%" />
</p>

#### Stencil (Time-Diamond)

Peak live = 2,817. Max reuse = 2,385. Median reuse = 9.

<p align="center">
  <img src="experiments/grid/diagnostics/stencil-time-diamond-32-t4_liveset.png" alt="Stencil (Time-Diamond) working-set size over time" width="49%" />
  <img src="experiments/grid/diagnostics/stencil-time-diamond-32-t4_reuse_distance.png" alt="Stencil (Time-Diamond) reuse distance per load" width="49%" />
</p>

#### Spatial Conv (2D, 16x16)

Peak live = 539. Max reuse = 537. Median reuse = 33.

<p align="center">
  <img src="experiments/grid/diagnostics/conv2d-spatial-16x16-k5_liveset.png" alt="Spatial Conv (2D, 16x16) working-set size over time" width="49%" />
  <img src="experiments/grid/diagnostics/conv2d-spatial-16x16-k5_reuse_distance.png" alt="Spatial Conv (2D, 16x16) reuse distance per load" width="49%" />
</p>

#### Spatial Conv (2D, 32x32)

Peak live = 2,075. Max reuse = 2,073. Median reuse = 34.

<p align="center">
  <img src="experiments/grid/diagnostics/spatial-conv-32x32-k5_liveset.png" alt="Spatial Conv (2D, 32x32) working-set size over time" width="49%" />
  <img src="experiments/grid/diagnostics/spatial-conv-32x32-k5_reuse_distance.png" alt="Spatial Conv (2D, 32x32) reuse distance per load" width="49%" />
</p>

#### Regular Conv

Peak live = 2,194. Max reuse = 2,192. Median reuse = 51.

<p align="center">
  <img src="experiments/grid/diagnostics/regular-conv-16x16-k3-c4_liveset.png" alt="Regular Conv working-set size over time" width="49%" />
  <img src="experiments/grid/diagnostics/regular-conv-16x16-k3-c4_reuse_distance.png" alt="Regular Conv reuse distance per load" width="49%" />
</p>

#### FFT Conv (1D)

Peak live = 225. Max reuse = 127. Median reuse = 3.

<p align="center">
  <img src="experiments/grid/diagnostics/fft-conv-32_liveset.png" alt="FFT Conv (1D) working-set size over time" width="49%" />
  <img src="experiments/grid/diagnostics/fft-conv-32_reuse_distance.png" alt="FFT Conv (1D) reuse distance per load" width="49%" />
</p>

#### FFT Conv (2D)

Peak live = 5,715. Max reuse = 3,096. Median reuse = 3.

<p align="center">
  <img src="experiments/grid/diagnostics/conv2d-fft-16x16-k5_liveset.png" alt="FFT Conv (2D) working-set size over time" width="49%" />
  <img src="experiments/grid/diagnostics/conv2d-fft-16x16-k5_reuse_distance.png" alt="FFT Conv (2D) reuse distance per load" width="49%" />
</p>

#### Mergesort

Peak live = 192. Max reuse = 159. Median reuse = 3.

<p align="center">
  <img src="experiments/grid/diagnostics/mergesort-64_liveset.png" alt="Mergesort working-set size over time" width="49%" />
  <img src="experiments/grid/diagnostics/mergesort-64_reuse_distance.png" alt="Mergesort reuse distance per load" width="49%" />
</p>

#### Bitonic Sort

Peak live = 130. Max reuse = 127. Median reuse = 61.

<p align="center">
  <img src="experiments/grid/diagnostics/bitonic-sort-64_liveset.png" alt="Bitonic Sort working-set size over time" width="49%" />
  <img src="experiments/grid/diagnostics/bitonic-sort-64_reuse_distance.png" alt="Bitonic Sort reuse distance per load" width="49%" />
</p>

#### LCS DP

Peak live = 1,155. Max reuse = 1,152. Median reuse = 4.

<p align="center">
  <img src="experiments/grid/diagnostics/lcs-dp-32x32_liveset.png" alt="LCS DP working-set size over time" width="49%" />
  <img src="experiments/grid/diagnostics/lcs-dp-32x32_reuse_distance.png" alt="LCS DP reuse distance per load" width="49%" />
</p>

#### Floyd-Warshall (Naive)

Peak live = 2,050. Max reuse = 2,047. Median reuse = 4.

<p align="center">
  <img src="experiments/grid/diagnostics/floyd-warshall-naive-32_liveset.png" alt="Floyd-Warshall (Naive) working-set size over time" width="49%" />
  <img src="experiments/grid/diagnostics/floyd-warshall-naive-32_reuse_distance.png" alt="Floyd-Warshall (Naive) reuse distance per load" width="49%" />
</p>

#### Floyd-Warshall (Recursive)

Peak live = 2,050. Max reuse = 2,047. Median reuse = 4.

<p align="center">
  <img src="experiments/grid/diagnostics/floyd-warshall-recursive-32_liveset.png" alt="Floyd-Warshall (Recursive) working-set size over time" width="49%" />
  <img src="experiments/grid/diagnostics/floyd-warshall-recursive-32_reuse_distance.png" alt="Floyd-Warshall (Recursive) reuse distance per load" width="49%" />
</p>

#### Gaussian Elimination

Peak live = 1,228. Max reuse = 1,200. Median reuse = 25.

<p align="center">
  <img src="experiments/grid/diagnostics/gaussian-elimination-24_liveset.png" alt="Gaussian Elimination working-set size over time" width="49%" />
  <img src="experiments/grid/diagnostics/gaussian-elimination-24_reuse_distance.png" alt="Gaussian Elimination reuse distance per load" width="49%" />
</p>

#### Gauss-Jordan Inverse

Peak live = 772. Max reuse = 767. Median reuse = 44.

<p align="center">
  <img src="experiments/grid/diagnostics/gauss-jordan-inverse-16_liveset.png" alt="Gauss-Jordan Inverse working-set size over time" width="49%" />
  <img src="experiments/grid/diagnostics/gauss-jordan-inverse-16_reuse_distance.png" alt="Gauss-Jordan Inverse reuse distance per load" width="49%" />
</p>

#### LU (No Pivot)

Peak live = 1,431. Max reuse = 1,152. Median reuse = 29.

<p align="center">
  <img src="experiments/grid/diagnostics/lu-no-pivot-24_liveset.png" alt="LU (No Pivot) working-set size over time" width="49%" />
  <img src="experiments/grid/diagnostics/lu-no-pivot-24_reuse_distance.png" alt="LU (No Pivot) reuse distance per load" width="49%" />
</p>

#### LU (Blocked)

Peak live = 2,710. Max reuse = 1,409. Median reuse = 10.

<p align="center">
  <img src="experiments/grid/diagnostics/blocked-lu-24_liveset.png" alt="LU (Blocked) working-set size over time" width="49%" />
  <img src="experiments/grid/diagnostics/blocked-lu-24_reuse_distance.png" alt="LU (Blocked) reuse distance per load" width="49%" />
</p>

#### LU (Recursive)

Peak live = 2,115. Max reuse = 1,325. Median reuse = 15.

<p align="center">
  <img src="experiments/grid/diagnostics/recursive-lu-24_liveset.png" alt="LU (Recursive) working-set size over time" width="49%" />
  <img src="experiments/grid/diagnostics/recursive-lu-24_reuse_distance.png" alt="LU (Recursive) reuse distance per load" width="49%" />
</p>

#### LU (Partial Pivot)

Peak live = 1,431. Max reuse = 1,151. Median reuse = 37.

<p align="center">
  <img src="experiments/grid/diagnostics/lu-partial-pivot-24_liveset.png" alt="LU (Partial Pivot) working-set size over time" width="49%" />
  <img src="experiments/grid/diagnostics/lu-partial-pivot-24_reuse_distance.png" alt="LU (Partial Pivot) reuse distance per load" width="49%" />
</p>

#### Cholesky

Peak live = 604. Max reuse = 599. Median reuse = 4.

<p align="center">
  <img src="experiments/grid/diagnostics/cholesky-24_liveset.png" alt="Cholesky working-set size over time" width="49%" />
  <img src="experiments/grid/diagnostics/cholesky-24_reuse_distance.png" alt="Cholesky reuse distance per load" width="49%" />
</p>

#### Cholesky (Blocked)

Peak live = 1,266. Max reuse = 1,151. Median reuse = 9.

<p align="center">
  <img src="experiments/grid/diagnostics/blocked-cholesky-24_liveset.png" alt="Cholesky (Blocked) working-set size over time" width="49%" />
  <img src="experiments/grid/diagnostics/blocked-cholesky-24_reuse_distance.png" alt="Cholesky (Blocked) reuse distance per load" width="49%" />
</p>

#### Cholesky (Recursive)

Peak live = 1,467. Max reuse = 905. Median reuse = 13.

<p align="center">
  <img src="experiments/grid/diagnostics/recursive-cholesky-24_liveset.png" alt="Cholesky (Recursive) working-set size over time" width="49%" />
  <img src="experiments/grid/diagnostics/recursive-cholesky-24_reuse_distance.png" alt="Cholesky (Recursive) reuse distance per load" width="49%" />
</p>

#### Cholesky (Right-Looking)

Peak live = 1,454. Max reuse = 1,151. Median reuse = 13.

<p align="center">
  <img src="experiments/grid/diagnostics/cholesky-right-looking-24_liveset.png" alt="Cholesky (Right-Looking) working-set size over time" width="49%" />
  <img src="experiments/grid/diagnostics/cholesky-right-looking-24_reuse_distance.png" alt="Cholesky (Right-Looking) reuse distance per load" width="49%" />
</p>

#### Householder QR

Peak live = 1,256. Max reuse = 1,201. Median reuse = 4.

<p align="center">
  <img src="experiments/grid/diagnostics/householder-qr-48x12_liveset.png" alt="Householder QR working-set size over time" width="49%" />
  <img src="experiments/grid/diagnostics/householder-qr-48x12_reuse_distance.png" alt="Householder QR reuse distance per load" width="49%" />
</p>

#### Blocked QR

Peak live = 1,349. Max reuse = 1,338. Median reuse = 4.

<p align="center">
  <img src="experiments/grid/diagnostics/blocked-qr-48x12_liveset.png" alt="Blocked QR working-set size over time" width="49%" />
  <img src="experiments/grid/diagnostics/blocked-qr-48x12_reuse_distance.png" alt="Blocked QR reuse distance per load" width="49%" />
</p>

#### TSQR

Peak live = 1,208. Max reuse = 719. Median reuse = 4.

<p align="center">
  <img src="experiments/grid/diagnostics/tsqr-48x12_liveset.png" alt="TSQR working-set size over time" width="49%" />
  <img src="experiments/grid/diagnostics/tsqr-48x12_reuse_distance.png" alt="TSQR reuse distance per load" width="49%" />
</p>

### Runtime

| Algorithm | Max traced cell (s) | Total traced time (s) |
| --- | --- | --- |
| Naive Matmul | 0.225 | 0.372 |
| Tiled Matmul | 0.187 | 0.302 |
| Recursive Matmul | 0.191 | 0.319 |
| Recursive In-Place (Lex) | 0.148 | 0.266 |
| Recursive In-Place (Gray) | 0.147 | 0.260 |
| Strassen | 0.546 | 0.791 |
| Fused Strassen | 0.548 | 0.795 |
| Naive Attention (d=2) | 0.532 | 0.858 |
| Flash Attention (Bk=8) | 0.155 | 0.287 |
| Naive Attention (d=4) | 0.873 | 1.343 |
| Flash Attention | 0.512 | 0.776 |
| LayerNorm (Unfused) | 0.453 | 0.735 |
| LayerNorm (Fused) | 0.395 | 0.657 |
| Matvec | 0.035 | 0.100 |
| Vecmat | 0.032 | 0.079 |
| Matvec Row | 0.420 | 1.185 |
| Matvec Column | 0.425 | 0.939 |
| Matrix Powers (Naive) | 0.505 | 0.794 |
| Matrix Powers (CA) | 0.192 | 0.388 |
| SpMV CSR (Banded) | 0.009 | 0.027 |
| SpMV CSR (Random) | 0.014 | 0.037 |
| Row Scan | 0.355 | 1.040 |
| Column Scan | 0.418 | 0.786 |
| Transpose (Naive) | 0.094 | 0.203 |
| Transpose (Blocked) | 0.117 | 0.269 |
| Transpose (Recursive) | 0.088 | 0.204 |
| FFT (Iterative) | 1.354 | 2.068 |
| FFT (Recursive) | 0.644 | 1.075 |
| Stencil (Naive) | 0.197 | 0.352 |
| Stencil (Recursive) | 0.190 | 0.400 |
| Stencil (Time-Naive) | 1.729 | 3.154 |
| Stencil (Time-Diamond) | 4.287 | 6.105 |
| Spatial Conv (2D, 16x16) | 0.153 | 0.272 |
| Spatial Conv (2D, 32x32) | 1.804 | 2.496 |
| Regular Conv | 2.947 | 4.090 |
| FFT Conv (1D) | 0.005 | 0.014 |
| FFT Conv (2D) | 1.865 | 2.922 |
| Mergesort | 0.004 | 0.010 |
| Bitonic Sort | 0.010 | 0.027 |
| LCS DP | 0.074 | 0.113 |
| Floyd-Warshall (Naive) | 4.790 | 7.218 |
| Floyd-Warshall (Recursive) | 4.744 | 7.157 |
| Gaussian Elimination | 0.369 | 0.599 |
| Gauss-Jordan Inverse | 0.633 | 1.000 |
| LU (No Pivot) | 0.502 | 0.742 |
| LU (Blocked) | 0.525 | 0.913 |
| LU (Recursive) | 0.427 | 0.643 |
| LU (Partial Pivot) | 0.555 | 0.822 |
| Cholesky | 0.082 | 0.154 |
| Cholesky (Blocked) | 0.115 | 0.214 |
| Cholesky (Recursive) | 0.192 | 0.324 |
| Cholesky (Right-Looking) | 0.115 | 0.223 |
| Householder QR | 0.503 | 0.792 |
| Blocked QR | 0.498 | 0.768 |
| TSQR | 0.451 | 0.752 |

Run the experiment with:

```bash
uv run experiments/grid/run_experiment.py
```

<!-- GRID-REPORT-END -->

# Python Gotcha's
The tracer implements ByteDMD by wrapping Python objects. This means that the "Instruction Set" of this metric corresponds to Python built-ins, documented under [docs/instruction_set.md](docs/instruction_set.md).

Python behavior means this implementation occasionally doesn't match README semantics and it is possible to escape the wrapping mechanism (local arrays, exception side-channels, identity ops, type introspection, f-strings, math.trunc/ceil/floor on tracked values, etc.). Known failure cases are documented in `test_gotchas.py` — avoid those patterns when writing code you want measured.


[Original Google Doc](https://docs.google.com/document/d/1sj5NqOg6Yqh10bXzGVEF5uIzSjFWAnqqTE75AMng2-s/edit?tab=t.0#heading=h.ujy6ygk7sjmb)
