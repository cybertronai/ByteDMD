# Experiments

Stand-alone experiments that go beyond the canonical benchmarks in `benchmarks/`. Each experiment lives in its own subdirectory with its own runner, raw results, and a written report.

## Index

- [`memory_management/`](memory_management/report.md) — Three memory management strategies (unmanaged, tombstone GC, aggressive compaction) tested on naive matmul, recursive matmul, and Strassen at sizes N=2..64. Confirms the asymptotic bounds from the closed-form analysis (Strassen tombstone fits `146.96·8^k − 146.57·7^k` with 4.7% error) and explains why the empirical exponent at small N looks closer to Wesley Smith's `N^3.23` than to the asymptotic `N³`.
