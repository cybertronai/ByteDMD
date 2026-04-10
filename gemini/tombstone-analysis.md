# tombstone-analysis

**Source:** https://gemini.google.com/share/6eacee1a7249
**Created with:** Gemini 3 Deep Think, April 9, 2026
**Topic:** ByteDMD formulas with tombstone/liveness-based LRU stack, model fixes, cold miss pricing

---

## Prompt 1
Exact ByteDMD formulas for matvec and vecmat with liveness-based implementation.

Reference scores: matvec=187, vecmat=181 for N=4. Requires fully unrolled left-to-right AST evaluation (not standard accumulator loop which would give 193/201).

### Exact Formulas (N >= 4)

Uses S(M) = sum_{k=1}^{M} ceil(sqrt(k)) = K(6M - 2K^2 + 3K - 1)/6 where K = floor(sqrt(M-1)) + 1

**Matvec (i-j loop):**
C_matvec(N) = N^3 + 5N^2 - 4N + S(4N-1) - S(N) + ceil(sqrt(3N)) + (N-3)*ceil(sqrt(4N-1)) + (N^2-4N+3)*ceil(sqrt(4N))

Verification: N=4 gives 80 + 48 + 59 = **187**

**Vecmat (j-i loop):**
Matrix A cost involves double sum over ceil(sqrt(N^2 + N - i(N-1-j))), reflecting column-major stride through row-major stack.

Verification: N=4 gives 76 + 48 + 57 = **181**

### Tight Continuous Bounds

**Matvec:** N^3 + 2N^{5/2} + 5N^2 - (10/3)N^{3/2} - 4N <= C <= N^3 + 2N^{5/2} + 6N^2 - (10/3)N^{3/2}

**Vecmat:** K = 16/9 - (4/3)ln2 ~ 0.85358
KN^3 + 2N^{5/2} + 4N^2 - (10/3)N^{3/2} - 4N <= C <= KN^3 + 2N^{5/2} + 6N^2

Vecmat scales structurally better (~0.853N^3 vs 1.0N^3).

### Discrepancies
1. **Sequential Pricing Violates "Simultaneous" Rule:** a+a prices first read at 5, bumps to top, second read at 1. Should price both at 5. Breaks commutativity.
2. **"Leapfrog" Anomaly:** Scalar-level tracking rewards column-major cache thrashing (~15% cheaper than row-major), contradicting physical hardware behavior.
3. **Argument Ordering Hack:** vecmat(A,x) gives 181 but vecmat(x,A) gives 189. Benchmark favored vecmat by preserving matvec's optimal push order.
4. **Tombstone Teleportation:** STORE allocations .remove(None) from bottom, instantly teleporting dead memory to top without physically shifting live variables.

---

## Prompt 2
Suggestions to fix the model while keeping formulas analytically compact.

### Four Architectural Fixes

**1. Fix Simultaneous Pricing:** Price all inputs against pre-instruction stack, then batch-move. Restores commutativity.

**2. Cache-Line Chunking:** Group contiguous memory into blocks of size B. LRU tracks block_ids. Fixes locality anomaly.

**3. Demand-Paged Initialization:** Start with empty stack. First access = cold miss appended organically. Removes argument-order signature bias.

**4. Natural LRU Aging:** Delete Pass 2 liveness logic. Let dead variables sink naturally. Removes tombstone teleportation.

### Resulting Compact Formulas (with block size B)

**Matvec:** N^2 stream hits at depth 1, x working set at depth N/B. Leading order O(N^{2.5} / sqrt(B))

**Vecmat:** Strides vertically, accessing N distinct blocks per read. Thrashes at distance N. Leading order O(2N^{2.5})

Penalty ratio vecmat/matvec ~ 2*sqrt(B). For B=16, vecmat is 8x more expensive, correctly reflecting physical spatial locality.

---

## Prompt 3
Cold miss pricing and preventing nesting exploits.

### Physical Motivation
Data lives in 2D DRAM grid. M-th element at spatial distance sqrt(M). Cold misses represent routing wires across the DRAM grid.

### The Nesting/Streaming Cheat
sqrt is sub-additive: sqrt(A+B) < sqrt(A) + sqrt(B). A programmer can stream elements through nested functions to keep LRU stack size = 1, reducing total cold miss cost from ~666M to M. This pretends a million-element matrix occupies 1 unit of physical space.

### Solution: Dual-Tracked Spatial Routing

**LRU Stack (L1 Cache):** Tracks active hot variables. Can shrink via tombstones.

**Monotonic DRAM Frontier:** Global integer counting total unique elements ever fetched. Never decreases.

Total Cold Miss Cost = sum_{k=1}^{M} ceil(sqrt(k)) ~ (2/3) M^{1.5}

Nested function boundaries chain integrals seamlessly: integral from 0 to A + integral from A to A+B = integral from 0 to A+B. Zero abstraction cost, but no cheating.

### Implementation Rules
- **Rule A:** Global singleton _Context passed through all nested functions. Never cleared on scope change.
- **Rule B:** Lazy input paging — assign IDs but leave in DRAM until instruction evaluates them.
- **Rule C:** Dual pricing — hot hits use LRU stack depth, cold misses use monotonic dram_frontier.
- Computed intermediates (c = a+b) pushed to stack at depth 1, do NOT increment dram_frontier.
