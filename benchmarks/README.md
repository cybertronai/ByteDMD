# Benchmarks

A folder of self-contained scripts that print ByteDMD cost tables for different
algorithm families. Each benchmark imports `bytedmd.py` from the repo root and
self-verifies its numbers via `assert` statements.

## How to run

From the repo root:

```bash
python3 benchmarks/benchmark_linalg.py
python3 benchmarks/benchmark_microgpt.py
python3 benchmarks/benchmark_attention.py
```

Each script is standalone and prints its table to stdout.

## `benchmark_linalg.py` — 4x4 linear algebra

Nine linear-algebra routines on 4x4 inputs (matvec, vecmat, five naive matmul
schedules, Strassen, and Winograd). Shows how loop order, tiling, and
Strassen/Winograd recursion affect data movement even though FLOP counts are
identical within each operation group.

```
Algorithm                 Operation       ByteDMD Cost
-------------------------------------------------------
matvec (i-j)              y = A @ x                194
vecmat (j-i)              y = x^T @ A              191
matmul (i-j-k)            C = A @ B                948
matmul (i-k-j)            C = A @ B               1016
matmul (snake-j)          C = A @ B                906
matmul (2x2 tiled)        C = A @ B                947
matmul (TSP)              C = A @ B                895
Strassen (leaf=1)         C = A @ B               2435
Winograd                  C = A @ B               2178
```

Note that Strassen and Winograd are *more expensive* than naive matmul at N=4
— their FLOP savings are swamped by the extra intermediate matrices they
materialize.

## `benchmark_microgpt.py` — tiny GPT forward pass

A single forward step (one token) through a minimal 1-layer transformer in the
style of Karpathy's microGPT: embeddings, 2-head self-attention, MLP, and LM
head. Config is `vocab=4, embd=4, heads=2, head_dim=2, block_size=4`.

```
microGPT (1 layer, embd=4)    single token forward    7047
```

This is the smallest end-to-end transformer whose ByteDMD cost is still
interesting, and is useful as a canary when changing the tracer.

## `benchmark_attention.py` — naive vs flash attention

Sweeps sequence length N ∈ {4, 8, 16, 32, 64, 128} at head dimension d=2,
comparing naive attention (which materializes the full NxN score matrix) to a
tiled flash-attention implementation that streams K/V blocks with online
softmax merging. For each N the script picks the best block size Bk.

Headline result (from `attention_results.json`, full analysis in
[`attention_report.md`](attention_report.md)):

| N   | Naive ByteDMD | Flash ByteDMD  | ByteDMD ratio | FLOP ratio |
|-----|---------------|----------------|---------------|------------|
| 4   | 1,406         | 1,498 (Bk=2)   | 0.94x         | 0.70x      |
| 32  | 293,648       | 163,643 (Bk=8) | 1.79x         | 0.86x      |
| 128 | 13,705,802    | 4,221,808 (Bk=8) | **3.25x**   | 0.96x      |

FLOPs stay essentially flat between the two variants, but flash attention's
ByteDMD advantage grows from 0.94x at N=4 to 3.25x at N=128 — exactly the
memory-hierarchy effect that motivates flash attention in practice, and that
FLOP counting cannot see.
