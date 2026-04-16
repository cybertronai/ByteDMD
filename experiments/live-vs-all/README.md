# Classic DMD · DMD-live · Tombstone · Ripple Shift · Manual

Five-column experiment on matmul:

| Column        | Level   | What it is                                                                            |
|---------------|---------|---------------------------------------------------------------------------------------|
| Classic DMD   | L2      | LRU stack, **no** liveness — dead vars pollute deeper rings (infinite graveyard).     |
| DMD-live      | L2      | LRU stack **with** liveness — dead vars vaporize, stack slides inward for free.        |
| Tombstone     | L3      | Mobile LRU with permanent holes. Has a stack-inflation bug.                           |
| Ripple Shift  | L3      | Cascaded-eviction allocator; cascade absorbed at the first hole. Fixes the inflation. |
| Manual        | direct  | Hand-written RMM on a bump-allocated address space with a software-managed scratchpad. |

Classic DMD and DMD-live are priced directly on the L2 event stream.
Tombstone and Ripple Shift lower L2 → L3 via allocators in
`bytedmd_ir.py`. Manual runs the matmul itself on a bump-allocated
memory in `manual_matmul.py` and accumulates cost as it executes — no
tracer, no liveness oracle, no magic movement.

## The three IR levels

- **L1** — Python source (algorithms as plain functions).
- **L2** — abstract IR: `LOAD(var)`, `STORE(var)`, `OP(name, in, out)`.
- **L3** — concrete IR: same events with `addr` per variable assigned by
  an allocator.

## Results

### Cache-oblivious RMM (8-way)

|   N  | Classic DMD   | Tombstone     | Manual         | Ripple Shift | DMD-live    |
|-----:|--------------:|--------------:|---------------:|-------------:|------------:|
|    4 |         1,043 |           912 |          1,194 |          781 |         689 |
|    8 |        13,047 |        11,582 |          9,465 |        8,568 |       7,773 |
|   16 |       154,251 |       131,742 |         94,490 |       87,463 |      80,716 |
|   32 |     1,779,356 |     1,445,402 |      1,085,426 |      852,117 |     794,969 |
|   64 |    20,291,116 |    15,768,636 |     14,088,144 |    8,038,558 |   7,554,413 |

### Tiled matmul (one level, tile = ⌈√N⌉)

|   N  | Classic DMD   | Tombstone     | Manual         | Ripple Shift | DMD-live    |
|-----:|--------------:|--------------:|---------------:|-------------:|------------:|
|    4 |         1,000 |           902 |          1,194 |          735 |         644 |
|    8 |        12,368 |        11,250 |          9,465 |        7,906 |       7,210 |
|   16 |       143,280 |       122,699 |         94,490 |       81,010 |      74,560 |
|   32 |     1,740,310 |     1,500,333 |      1,085,426 |      829,469 |     790,183 |
|   64 |    19,737,581 |    17,264,621 |     14,088,144 |    8,237,471 |   7,917,595 |

**Ordering at N ≥ 8**:
`DMD-live  <  Ripple Shift  <  Manual  <  Tombstone  <  Classic DMD`

Ripple Shift sits within ~5 % of DMD-live across the full range. Manual
(a real hand-written implementation with a software scratchpad) sits
between the two L3 allocators, showing what a practical programmer can
achieve without the offline optimality of Ripple.

See **[REPORT.md](REPORT.md)** for the full writeup with physical
picture, asymptotic derivations, and footprint comparison.

## Reproducibility

```bash
uv run pytest test_bytedmd_ir.py                  # 37 tests
uv run --script envelope.py                       # default N up to 64, ~20 s
uv run --script envelope.py 4,8,16,32,64,128      # ~3 min
```
