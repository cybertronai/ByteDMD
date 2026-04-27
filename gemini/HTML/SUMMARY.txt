# ByteDMD Research Reports — Summary

These 10 reports document a research effort to develop **ByteDMD** (Byte-level Data Movement Distance), a concrete metric for quantifying data movement cost of algorithms. The metric models an idealized LRU stack where each memory read costs ceil(sqrt(depth)), motivated by 2D VLSI wire-length routing on a spatial chip.

---

## Reports Extracted (8 of 10)

### 1. matvec-vecmat
Derives exact discrete closed-form summation formulas for matrix-vector and vector-matrix multiplication under ByteDMD. Key finding: vecmat (column-major) is ~8% cheaper than matvec (row-major) under the naive model (1.2869 N^3 vs 1.3987 N^3), because column-by-column fetching acts as a pseudo-eviction buffer. With a "smart compiler" performing perfect liveness analysis, the gap widens to 1.5x (0.444 N^3 vs 0.666 N^3) via a phenomenon called "cache hollowing."

### 2. matmul-just-snake
Exact ByteDMD cost for naive matrix multiplication using snake (boustrophedon) loop ordering. The cost decomposes into six closed-form summations giving exactly 906 for N=4. The asymptotic leading term is (sqrt(5)/(3*sqrt(2))) N^4 ~ 1.4907 N^4, confirming Bill Dally's thesis that untiled naive matmul forces O(N^4) data movement despite O(N^3) FLOPs.

### 3. renaming-brainstorming
Brainstorms alternative names for ByteDMD to better emphasize the metric as a concrete count rather than an asymptotic bound. Top recommendations: **ExactDMD/TraceDMD** (emphasizes concrete execution), **ElementDMD** (fixes "byte" generality), **DMC** (Data Movement Cost — clean benchmarking name).

### 4. bytedmd-instruction-list
Catalogs 24 canonical instruction families (38 methods with reverse forms) implemented by the ByteDMD Python tracer. Includes binary ops (2 in, 1 out), divmod (2 in, 2 out), unary ops (1 in, 1 out), comparisons (2 in, 0 tracked out). Also proposes extending coverage to built-in reductions (sum, any, all, min, max) and math functions (abs, round).

### 5. matmul-optimal
Analyzes N x N matrix-vector multiplication on the Thompson 2D VLSI / Spatial Computer model. Proves the energy lower bound is Omega(N^2) and shows it is achievable via a 2D systolic wavefront pipeline where the matrix stays stationary, x streams vertically, and partial sums stream horizontally. All messages travel distance 1, achieving total energy 2N^2 - 2N = Theta(N^2) and depth Theta(N).

### 6. matmul-regular-snake
Comprehensive analysis covering three algorithms: normal order (i-j-k), snake order, and recursive matmul (without reuse). Normal order scales at sqrt(3) N^4 ~ 1.732 N^4; snake drops to (sqrt(5)/(3*sqrt(2))) N^4 ~ 1.491 N^4 (14% reduction). Recursive matmul avoids catastrophic N^4 penalties entirely, scaling at O(N^3.5). Also demonstrates that garbage collection drops the normal-order routing constant from 1.732 to 1.0 (42% improvement). Includes Mathematica code for LRU depth visualization showing the "sawtooth wave" pattern of snake order.

### 7. bytedmd-naive-formulas
Focuses specifically on exact formulas for naive matmul, identifying the "phantom zero bypass" where NumPy initializes accumulators with the first product rather than 0, producing the benchmark score of exactly 948 for N=4. Provides both exact piecewise Mathematica formulas and clean closed-form polynomial bounds: M(N) = sqrt(3)*N^4 + alpha*N^3.5 with alpha = 2 + (2*sqrt(2))/3 ~ 2.943.

### 8. tombstone-analysis
Analyzes the liveness/tombstone-based ByteDMD variant (scores: matvec=187, vecmat=181 for N=4). Identifies four critical model flaws: sequential pricing violating simultaneous rules, a "leapfrog anomaly" rewarding cache thrashing, argument-order bias, and tombstone teleportation. Proposes four fixes: simultaneous pricing, cache-line chunking (block size B), demand-paged initialization, and natural LRU aging. Also develops the theory of **cold miss pricing** with a monotonic DRAM frontier to prevent nesting/streaming exploits, using the sub-additivity of sqrt to prove that function abstraction boundaries cannot reduce total data movement cost.

---

## Not Extracted (no public share links available)

### 9. cache-eviction-visualization
Topic: How to propagate more smartly. Only an /app/ link was provided (private, not accessible).

### 10. bytedmd-strassen
Topic: Strassen algorithm analysis under ByteDMD. Only an /app/ link was provided (private, not accessible).

---

## Cross-Cutting Themes

**Recurring implementation discrepancies** identified across multiple reports: (1) unbounded stack growth / memory leak from never popping dead temporaries, (2) Python AST left-to-right evaluation breaking simultaneous pricing semantics, (3) untracked array indexing making pointer traversal free, (4) zero-accumulator initialization artifacts.

**Key scaling results:**
| Algorithm | Leading ByteDMD Term | Notes |
|-----------|---------------------|-------|
| Naive matmul (normal) | sqrt(3) N^4 ~ 1.732 N^4 | Dead temporaries cause O(N^4) |
| Naive matmul (snake) | 1.491 N^4 | 14% improvement over normal |
| Naive matmul (normal + GC) | 1.0 N^4 | GC removes 42% of penalty |
| Recursive matmul | O(N^3.5) | Morton-curve bounds cache distances |
| Matvec | ~N^3 | Achievable at Theta(N^2) energy on spatial computer |
| Vecmat | ~0.854 N^3 | "Cache hollowing" advantage |

**Model evolution:** The reports trace a progression from a basic Python proxy tracer toward a more physically grounded model with cache-line chunking, demand paging, simultaneous instruction pricing, and dual-tracked cold/hot memory routing.
