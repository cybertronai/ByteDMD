# Heuristic Grid for ByteDMD-Style Metrics

This experiment compares a concrete no-free-compaction 2D cost against several fast heuristics on a small suite of workloads.

Every traced metric cell finished under 0.943 seconds on this run.

## Algorithms

| Algorithm | Workload | Implementation |
| --- | --- | --- |
| Matvec | 32x32 by 32 | row-wise matrix-vector baseline |
| Vecmat | 32 by 32x32 | column-oriented access order |
| Naive Matmul | 16x16 | standard i-j-k triple loop |
| Tiled Matmul | 16x16, tile=4 | one explicit blocking level |
| Recursive Matmul | 16x16 | 8-way cache-oblivious recursion |
| Recursive In-Place (Lex) | 16x16 | manual in-place schedule, lexicographic order |
| Recursive In-Place (Gray) | 16x16 | manual in-place schedule, Gray-code order |
| Strassen | 16x16 | leaf size 1 to expose temporary traffic |
| Regular Attention | N=32, d=4 | materializes the full score matrix |
| Flash Attention | N=32, d=4, Bq=8, Bk=4 | double-tiled Q/KV blocks with snake KV order |

## Measures

- `Manual-2D`: the concrete tombstone/no-compaction 2D cost used as the target.
- `ByteDMD-classic`: graveyard model with no reclamation.
- `ByteDMD-live`: aggressive live-only compaction.
- `Reads×sqrt(Peak)`: `reads * ceil(sqrt(peak_live))`, a bandwidth-times-footprint proxy.
- `Reads`: total tracked reads.
- `Peak live slots`: peak active footprint under the live policy.
- `FLOPs`: arithmetic work count.

Attention uses proxy `max`, `exp`, and reciprocal operators with the same read arity as the real kernels, so the table focuses on data movement rather than numerical fidelity.

## Results Grid

| Algorithm | Manual-2D | ByteDMD-classic | ByteDMD-live | Reads×sqrt(Peak) | Reads | Peak live slots | FLOPs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Matvec | 47,951 | 62,694 | 46,926 | 137,088 | 4,032 | 1,090 | 2,016 |
| Vecmat | 43,860 | 59,331 | 42,795 | 137,088 | 4,032 | 1,090 | 2,016 |
| Naive Matmul | 121,869 | 178,324 | 117,935 | 444,416 | 15,872 | 770 | 7,936 |
| Tiled Matmul | 96,306 | 143,280 | 88,687 | 444,416 | 15,872 | 771 | 7,936 |
| Recursive Matmul | 106,395 | 154,251 | 95,462 | 476,160 | 15,872 | 896 | 7,936 |
| Recursive In-Place (Lex) | 96,130 | 162,049 | 91,212 | 458,752 | 16,384 | 770 | 7,936 |
| Recursive In-Place (Gray) | 89,378 | 155,454 | 86,402 | 458,752 | 16,384 | 770 | 7,936 |
| Strassen | 250,051 | 353,207 | 204,752 | 1,068,970 | 30,542 | 1,194 | 15,271 |
| Regular Attention | 320,318 | 474,581 | 303,850 | 2,072,640 | 40,640 | 2,565 | 20,320 |
| Flash Attention | 234,485 | 335,704 | 197,629 | 1,218,240 | 45,120 | 723 | 22,560 |

## Heuristic Ranking Against Manual-2D

| Heuristic | Spearman rho | Scaled MAPE |
| --- | --- | --- |
| ByteDMD-live | 0.988 | 6.0% |
| ByteDMD-classic | 0.903 | 6.2% |
| FLOPs | 0.895 | 19.7% |
| Reads×sqrt(Peak) | 0.838 | 22.2% |
| Reads | 0.759 | 20.4% |
| Peak live slots | 0.148 | 55.0% |

## Runtime

| Algorithm | Max traced cell (s) | Total traced time (s) |
| --- | --- | --- |
| Matvec | 0.111 | 0.239 |
| Vecmat | 0.101 | 0.239 |
| Naive Matmul | 0.222 | 0.443 |
| Tiled Matmul | 0.244 | 0.385 |
| Recursive Matmul | 0.226 | 0.395 |
| Recursive In-Place (Lex) | 0.279 | 0.426 |
| Recursive In-Place (Gray) | 0.242 | 0.381 |
| Strassen | 0.616 | 0.982 |
| Regular Attention | 0.943 | 1.706 |
| Flash Attention | 0.486 | 0.738 |

Run the experiment with:

```bash
uv run experiments/heuristic_grid/run_experiment.py
```