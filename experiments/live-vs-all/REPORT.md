# Classic DMD · DMD-live · Tombstone — report

## Three cost models on one trace

Every measure below is priced on the **same L2 trace** (the same algorithm,
the same load/store events). The three differ only in how they assign
depth to each read. The Gemini analysis in
`gemini/15apr26-dmdlive-analysis.md` derives each regime from the physics
of a 2-D continuous cache whose d-th concentric ring costs `⌈√d⌉`.

### Classic DMD — "Infinite Graveyard"

LRU stack with **no** liveness compaction. Every STORE pushes to the top;
every LOAD bumps the variable to the top; **dead variables never leave
the stack**. The depth of a LOAD of `X` equals the number of distinct
variables (live or dead) referenced since `X`'s previous LOAD.

Physical picture: every new temporary is placed at the cache center,
permanently shoving all older variables one ring outward. On RMM, the
addition step at the root of an 8-way recursion wades through an
`O(N^3)` graveyard of dead temporaries produced by sibling subcalls, so
each read charges `√(N^3) = N^{1.5}` across `N^2` reads →
**`Θ(N^{3.5})`**.

### DMD-live — "Teleporting Cache"

LRU stack **with** liveness compaction. A variable is dropped from the
stack the instant its last LOAD executes; everything above it slides
inward for free. The depth of a LOAD of `X` equals the number of **live
bytes** referenced between `X`'s previous LOAD and the current one.

Physical picture: whenever a temporary dies, the entire cache
magically telescopes inward at zero energy cost. This is a mathematical
idealisation — real hardware cannot move arbitrary rings inward without
physical work — but it tightly clamps the cache radius to the
*instantaneous* live working set `O(N^2)`. Each read charges `O(N)`
across `O(N^3)` reads per level × `log N` levels →
**`Θ(N^3 log N)`**.

### Tombstone — concrete stationary allocator

When a variable dies it leaves a *hole* (tombstone) at its physical
address; older live variables **do not slide inward**. Every new
variable is allocated to the **closest available hole** (smallest free
address). Implemented in `bytedmd_ir.compile_min_heap` — a stationary
min-heap of freed addresses with `ceil(sqrt(addr))` per LOAD.

Physical picture: matches what a real silicon address map does —
recycle freed slots without moving live data. On RMM this preserves a
peak footprint of `O(N^2)` but never reduces the cache radius below the
high-water mark. Reads therefore charge `√(O(N^2)) = O(N)` for every
live access, regardless of which recursion level is currently executing.
Summed over `O(N^3)` reads this gives **`Θ(N^4)`** empirically — strictly
worse than DMD-live because the Tombstone cache cannot telescope.

## The three IR levels

| Level | Name          | Contents                                                |
|-------|---------------|---------------------------------------------------------|
| L1    | Python source | Algorithm as a plain function.                          |
| L2    | Abstract IR   | `LOAD(var)` / `STORE(var)` / `OP(name, in, out)` — no addresses. |
| L3    | Concrete IR   | Same events + physical `addr` per variable (via allocator). |

Classic DMD and DMD-live are computed **directly on L2** with an LRU
walk (O(log T) per event via a Fenwick tree). Tombstone lowers L2 to L3
with `compile_min_heap` and then applies the shared `ceil(sqrt(addr))`
cost.

## Results

Tracing two algorithms at `N ∈ {4, 8, 16, 32, 64}`.

### Cache-oblivious RMM (8-way recursive)

|  N  | Classic DMD  | DMD-live    | Tombstone     | classic / live | tombstone / live |
|----:|-------------:|------------:|--------------:|---------------:|-----------------:|
|   4 |        1,043 |         689 |           985 |          1.51× |            1.43× |
|   8 |       13,047 |       7,773 |        16,315 |          1.68× |            2.10× |
|  16 |      154,251 |      80,716 |       266,593 |          1.91× |            3.30× |
|  32 |    1,779,356 |     794,969 |     4,320,478 |          2.24× |            5.44× |
|  64 |   20,291,116 |   7,554,413 |    69,716,078 |          2.69× |            9.23× |

### One-level tiled matmul (tile = ⌈√N⌉)

|  N  | Classic DMD  | DMD-live    | Tombstone     | classic / live | tombstone / live |
|----:|-------------:|------------:|--------------:|---------------:|-----------------:|
|   4 |        1,000 |         644 |           961 |          1.55× |            1.49× |
|   8 |       12,368 |       7,210 |        15,128 |          1.72× |            2.10× |
|  16 |      143,280 |      74,560 |       233,811 |          1.92× |            3.14× |
|  32 |    1,740,310 |     790,183 |     3,683,154 |          2.20× |            4.66× |
|  64 |   19,737,581 |   7,917,595 |    57,162,017 |          2.49× |            7.22× |

## Asymptotic verification

Fitting `cost / N^α`:

- **Classic DMD**: normalising by `N^{3.5}` gives 8–10 across N; steady
  → **`Θ(N^{3.5})`** confirmed.
- **DMD-live**: normalising by `N^3 log₂ N` gives ≈ 5 across N; steady
  → **`Θ(N^3 log N)`** confirmed.
- **Tombstone**: normalising by `N^4` gives ≈ 4 across N; steady
  → **`Θ(N^4)`** empirically. This is strictly *worse* than DMD-live's
  `N^3 log N`, contradicting the prediction in
  `gemini/15apr26-dmdlive-analysis.md` that Tombstone preserves the
  optimal asymptotic. The gap is the extra `N / log N` factor of paying
  `√(N^2) = N` per read at the global high-water mark instead of the
  subproblem-local `√(S^2) = S` that DMDlive's magic sliding provides.

## Interpretation

- **Classic DMD** is the pessimistic upper bound: what you get if dead
  temporaries cannot be evicted at all.
- **DMD-live** is the optimistic lower bound: the cost only of
  *referencing* live data, ignoring the physical work of compacting.
- **Tombstone** is the realistic middle: stationary slots you can
  actually implement in hardware. For matmul it lies above both
  Classic DMD and DMD-live once `N` is large enough that the loss from
  stationary placement outweighs the savings from having no graveyard.

The crossover point (RMM) where Tombstone overtakes Classic DMD is
`N ≈ 6`; for N ≥ 8 Tombstone is strictly above both DMD curves. This
makes the three measures an informative trio for any cost-model
discussion: each captures a distinct trade-off between realism and
optimality.

## Reproducibility

```bash
uv run pytest test_bytedmd_ir.py                        # 27 tests
uv run --script experiments/live-vs-all/envelope.py 4,8,16,32,64
```

Outputs `envelope.png` (three curves per algorithm, with `N^3 log N`,
`N^{3.5}`, and `N^4` reference lines) and `envelope_ratio.png`
(classic / live vs N).
