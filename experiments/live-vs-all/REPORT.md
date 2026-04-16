# Classic DMD · DMD-live · Tombstone · Ripple Shift · Manual — report

## Five cost models on one trace

Every measure below is priced on the **same L2 trace** (the same algorithm,
the same load/store events). They differ in how they assign depth to each
read. Refs: `gemini/15apr26-dmdlive-analysis.md`,
`gemini/ripple-shift.md`.

### Classic DMD — "Infinite Graveyard"

LRU stack with **no** liveness compaction. Dead variables never leave.
Depth of a LOAD of X = number of distinct variables (live or dead)
referenced since X's previous LOAD. RMM: **`Θ(N^{3.5})`**.

### DMD-live — "Teleporting Cache"

LRU stack **with** liveness compaction. A variable is dropped on its
last LOAD; everything above slides inward for free. Depth of a LOAD of X
= number of **live bytes** between X's prev and current LOAD. The free
sliding is unphysical, but clamps the cache to the live working set.
RMM: **`Θ(N^3 log N)`**.

### Tombstone — mobile LRU with permanent holes

A stack where dead variables leave permanent tombstones. Has a known
**inflation bug**: a LOAD/STORE with no hole above appends to a new top,
pushing every dormant variable outward by 1. At N = 32 RMM the live
working set is 2,645 but the tombstone footprint bloats to 31,624
slots — 12× inflation. Cost drifts toward Classic DMD.

### Ripple Shift — cascaded eviction (fixes the inflation bug)

Real hardware cascaded-eviction caches (shift registers / systolic
arrays) propagate evictions outward until absorbed by a hole. On
STORE/LOAD: target at addr 1; dormant variables shift `addr → addr + 1`;
**the cascade stops at the first hole**. Unlike DMD-live, inward movement
is only performed as the natural consequence of outward shift absorbing
at a hole — no free global compaction.

Footprint stays clamped to the live high-water mark (0 % inflation at
every N up to 128). Implementation: `bytedmd_ir.compile_ripple`, Fenwick
tree over timestamps, `O(E log E)` total.

### Manual — hand-written RMM on a bump-allocated address space

A physically grounded concrete implementation that ignores all the
metrics above and just builds a cost by running the algorithm:

- `ManualAllocator` is a bump allocator: `alloc(n)` returns the next
  base address, writes are free, reads from address `d` charge
  `⌈√d⌉`.
- `ScratchpadRMM` pins three `T × T` tiles (A, B, C) to the lowest
  addresses (1…3T²). Reads to those addresses cost ≤ ⌈√(3T²)⌉ = 7 for
  `T = 4`.
- `matmul_rmm_manual` runs an 8-way recursive RMM with a Hamiltonian
  traversal that keeps exactly one (A, B, or C) tile in the scratchpad
  across every consecutive pair of recursive calls. At `size ==
  tile_size`, `compute_tile` DMAs missing tiles into the scratchpad and
  performs the inner multiply-accumulate entirely on low addresses.

Implemented in `manual_matmul.py`. This is the "no-magic" point of
comparison: every load goes through a real address, no liveness oracle,
no tracer, no implicit movement.

## The three IR levels

| Level | Name          | Contents                                                |
|-------|---------------|---------------------------------------------------------|
| L1    | Python source | Algorithm as a plain function.                          |
| L2    | Abstract IR   | `LOAD(var)` / `STORE(var)` / `OP(name, in, out)` — no addresses. |
| L3    | Concrete IR   | Same events + physical `addr` per variable.             |

Classic DMD and DMD-live run on L2; Tombstone and Ripple on L3 (via an
allocator); Manual is a separate implementation that never touches the
L2 stream.

## Results

Two algorithms traced at `N ∈ {4, 8, 16, 32, 64}`. The Manual column is
the same `matmul_rmm_manual(tile_size=4)` cost in both subplots — it
serves as a "realistic implementation" reference point regardless of
which L2 trace is being priced.

### Cache-oblivious RMM (8-way recursive)

|   N  | Classic DMD   | Tombstone     | Manual         | Ripple Shift | DMD-live    |
|-----:|--------------:|--------------:|---------------:|-------------:|------------:|
|    4 |         1,043 |           912 |          1,194 |          781 |         689 |
|    8 |        13,047 |        11,582 |          9,465 |        8,568 |       7,773 |
|   16 |       154,251 |       131,742 |         94,490 |       87,463 |      80,716 |
|   32 |     1,779,356 |     1,445,402 |      1,085,426 |      852,117 |     794,969 |
|   64 |    20,291,116 |    15,768,636 |     14,088,144 |    8,038,558 |   7,554,413 |

### One-level tiled matmul (tile = ⌈√N⌉)

|   N  | Classic DMD   | Tombstone     | Manual         | Ripple Shift | DMD-live    |
|-----:|--------------:|--------------:|---------------:|-------------:|------------:|
|    4 |         1,000 |           902 |          1,194 |          735 |         644 |
|    8 |        12,368 |        11,250 |          9,465 |        7,906 |       7,210 |
|   16 |       143,280 |       122,699 |         94,490 |       81,010 |      74,560 |
|   32 |     1,740,310 |     1,500,333 |      1,085,426 |      829,469 |     790,183 |
|   64 |    19,737,581 |    17,264,621 |     14,088,144 |    8,237,471 |   7,917,595 |

## Ordering

At every `N ≥ 8`:

`DMD-live  <  Ripple Shift  <  Manual  <  Tombstone  <  Classic DMD`

Interpretation:

- **DMD-live** is the theoretical floor (unphysical teleportation).
- **Ripple Shift** lands within ~5 % of DMD-live — it is physically
  realizable and practically tracks DMD-live as a cost predictor.
- **Manual** is a real hand-written implementation with a software
  scratchpad; at `N = 64` RMM it's about 1.9× above Ripple Shift and
  1.86× above DMD-live. The gap is what a hand-written implementation
  "leaves on the table" relative to the offline-optimal allocator that
  Ripple Shift approximates.
- **Tombstone** pays the inflation tax (12× stack bloat at N = 32).
- **Classic DMD** is the memory-leak upper bound.

## Asymptotic verification

- **Classic DMD**: `/N^{3.5}` ≈ 8–10 across N → `Θ(N^{3.5})`.
- **DMD-live**: `/(N^3 log₂ N)` ≈ 5 across N → `Θ(N^3 log N)`.
- **Ripple Shift**: `/(N^3 log₂ N)` converges to ≈ 5 as N grows →
  `Θ(N^3 log N)`, matching DMD-live to within ~5 %.
- **Manual**: `/(N^3 log₂ N)` ≈ 6–9 across N. Same asymptotic class;
  constant larger than Ripple because the hand-written implementation
  does all its reads from concrete addresses (no free compaction) and
  the DMA transfers touch the large bump-allocated region.
- **Tombstone**: `/(N^3 log₂ N)` grows slowly 7 → 10; sub-polynomial
  drift from the inflation bug, but still `Θ(N^3 log N)`-class.

## Footprint comparison (RMM)

|  N  | Live high-water | Tombstone peak | Ripple peak |
|----:|----------------:|---------------:|------------:|
|   8 |             165 |            520 |         165 |
|  16 |             661 |          4,008 |         661 |
|  32 |           2,645 |         31,624 |       2,645 |
|  64 |          10,581 |      (~250 k+) |      10,581 |

Ripple's footprint exactly matches the live working set. Tombstone
inflates 3–12×.

## Reproducibility

```bash
uv run pytest test_bytedmd_ir.py                         # 37 tests
uv run --script experiments/live-vs-all/envelope.py      # default N up to 64, ~20 s
uv run --script experiments/live-vs-all/envelope.py 4,8,16,32,64,128   # ~3 min
```

## References

- `gemini/15apr26-dmdlive-analysis.md` — three-regime framing.
- `gemini/ripple-shift.md` — cascaded-eviction allocator and
  inflation-bug diagnosis.
- `manual_matmul.py` — hand-written RMM + scratchpad reference.
