# Heuristic Grid for ByteDMD-Style Metrics

This experiment compares a concrete no-free-compaction 2D cost against the two abstract ByteDMD heuristics on a small suite of workloads.

Every traced metric cell finished under 1.114 seconds on this run.

## Algorithms

| Algorithm | Workload | Implementation |
| --- | --- | --- |
| Matvec | 32x32 by 32 | row-wise matrix-vector baseline |
| Vecmat | 32 by 32x32 | column-oriented access order |
| Transpose (Naive) | 32x32 | direct row-major transpose copy |
| Transpose (Blocked) | 32x32, block=8 | blocked transpose copy |
| Transpose (Recursive) | 32x32, leaf=8 | cache-oblivious recursive transpose |
| Row Scan | 64x64 | row-major traversal sum |
| Column Scan | 64x64 | column-major traversal sum |
| Naive Matmul | 16x16 | standard i-j-k triple loop |
| Tiled Matmul | 16x16, tile=4 | one explicit blocking level |
| Recursive Matmul | 16x16 | 8-way cache-oblivious recursion |
| Recursive In-Place (Lex) | 16x16 | manual in-place schedule, lexicographic order |
| Recursive In-Place (Gray) | 16x16 | manual in-place schedule, Gray-code order |
| Strassen | 16x16 | leaf size 1 to expose temporary traffic |
| FFT (Iterative) | N=32 | iterative radix-2 Cooley-Tukey |
| FFT (Recursive) | N=32 | recursive radix-2 Cooley-Tukey |
| Stencil (Naive) | 32x32, one sweep | row-major Jacobi stencil |
| Stencil (Recursive) | 32x32, one sweep, leaf=8 | tile-recursive Jacobi stencil |
| Regular Attention | N=32, d=4 | materializes the full score matrix |
| Flash Attention | N=32, d=4, Bq=8, Bk=4 | double-tiled Q/KV blocks with snake KV order |

## Measures

- `Manual-2D`: the concrete tombstone/no-compaction 2D cost used as the target.
- `ByteDMD-classic`: graveyard model with no reclamation.
- `ByteDMD-live`: aggressive live-only compaction.

Attention uses proxy `max`, `exp`, and reciprocal operators with the same read arity as the real kernels, so the table focuses on data movement rather than numerical fidelity.

## Results Grid

| Algorithm | Manual-2D | ByteDMD-classic | ByteDMD-live |
| --- | --- | --- | --- |
| Matvec | 47,951 | 62,694 | 46,926 |
| Vecmat | 43,860 | 59,331 | 42,795 |
| Transpose (Naive) | 40,447 | 40,447 | 40,447 |
| Transpose (Blocked) | 39,806 | 39,806 | 39,806 |
| Transpose (Recursive) | 39,737 | 39,737 | 39,737 |
| Row Scan | 274,427 | 325,675 | 270,334 |
| Column Scan | 235,441 | 294,101 | 231,311 |
| Naive Matmul | 121,869 | 178,324 | 117,935 |
| Tiled Matmul | 96,306 | 143,280 | 88,687 |
| Recursive Matmul | 106,395 | 154,251 | 95,462 |
| Recursive In-Place (Lex) | 96,130 | 162,049 | 91,212 |
| Recursive In-Place (Gray) | 89,378 | 155,454 | 86,402 |
| Strassen | 250,051 | 353,207 | 204,752 |
| FFT (Iterative) | 1,773 | 2,139 | 1,691 |
| FFT (Recursive) | 1,522 | 1,708 | 1,366 |
| Stencil (Naive) | 73,287 | 94,490 | 68,807 |
| Stencil (Recursive) | 66,583 | 86,431 | 62,206 |
| Regular Attention | 320,318 | 474,581 | 303,850 |
| Flash Attention | 234,485 | 335,704 | 197,629 |

## Heuristic Ranking Against Manual-2D

| Heuristic | Spearman rho | Scaled MAPE |
| --- | --- | --- |
| ByteDMD-live | 0.996 | 4.7% |
| ByteDMD-classic | 0.977 | 12.5% |

## Runtime

| Algorithm | Max traced cell (s) | Total traced time (s) |
| --- | --- | --- |
| Matvec | 0.100 | 0.224 |
| Vecmat | 0.090 | 0.204 |
| Transpose (Naive) | 0.115 | 0.287 |
| Transpose (Blocked) | 0.113 | 0.281 |
| Transpose (Recursive) | 0.121 | 0.293 |
| Row Scan | 1.114 | 2.745 |
| Column Scan | 0.992 | 2.504 |
| Naive Matmul | 0.218 | 0.404 |
| Tiled Matmul | 0.187 | 0.314 |
| Recursive Matmul | 0.206 | 0.360 |
| Recursive In-Place (Lex) | 0.250 | 0.395 |
| Recursive In-Place (Gray) | 0.243 | 0.381 |
| Strassen | 0.605 | 0.975 |
| FFT (Iterative) | 0.002 | 0.004 |
| FFT (Recursive) | 0.001 | 0.004 |
| Stencil (Naive) | 0.171 | 0.417 |
| Stencil (Recursive) | 0.161 | 0.413 |
| Regular Attention | 0.944 | 1.719 |
| Flash Attention | 0.506 | 0.770 |

Run the experiment with:

```bash
uv run experiments/heuristic_grid/run_experiment.py
```