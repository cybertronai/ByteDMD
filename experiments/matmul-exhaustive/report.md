# matmul-exhaustive — every n^3 strategy for 2x2

## Summary

For 2x2 matrix multiplication restricted to the (*, +) semiring (8
scalar products + 4 scalar sums — Strassen and its cousins are out of
scope), this experiment enumerates every strategy that can use a
scratchpad under manual allocation, and ranks them by DMD cost.

**Cost model.** Flat Manhattan distance: every read at address `a`
costs `ceil(sqrt(a))`; writes are free. Two bump-pointer allocators
run in parallel — one for arg inputs (A, B), one for scratch
(products + output C). Addresses are fixed at allocation time; nothing
is relocated. This is the same `manual` model used in
`experiments/grid/`.

**83 strategies sweep.** For each of the four output cells C[i][j] we
pick one of three assembly modes, and we include two additional
"concurrent-batched" variants that keep multiple pairs' products alive
simultaneously:

| mode       | how C[i][j] is assembled                                     | scratch reads this pair |
|------------|---------------------------------------------------------------|-----------------------:|
| `direct`   | MUL1 writes **directly to C[i][j]**; MUL2 writes to P0; ADD reads C + P0 → C | 1 P0 + 1 C = 2 |
| `indirect` | MUL1 writes to P0; ASSIGN copies P0 → C[i][j]; MUL2 overwrites P0; ADD reads C + P0 → C | 2 P0 + 1 C = 3 |
| `batched`  | MUL1 writes to P0; MUL2 writes to P1; ADD reads P0 + P1 → C   | 1 P0 + 1 P1 = 2 |

P0 and P1 are single scratch slots reused across every pair. Each C
cell is also read once in the epilogue (the caller reading the
output). 3^4 = 81 mode combinations plus 2 concurrent-batched variants
(`k_live=2`, `k_live=4`) gives 83 strategies.

**Arg layout.** A[0][0], A[0][1], A[1][0], A[1][1] at arg addrs 1..4;
B[0][0], B[0][1], B[1][0], B[1][1] at arg addrs 5..8. Every arg cell
is read exactly twice under any n^3 schedule, so arg order is
cost-neutral — total arg-read cost is always
`2 * (1+2+2+2+3+3+3+3) = 38`.

**Scratch layout.** Under optimal placement — sorting cells by read
count descending and assigning addresses 1, 2, 3, ... — which is
achievable for every strategy here because all allocated cells are
mutually live throughout the computation (no push/pop reuse is
required to hit the optimum).

## Ranked table (grouped by cost)

| rank | cost | count | example | cells + read counts |
|-----:|-----:|------:|---------|---------------------|
|   1  |  60  |  16   | `naive:bbbb` | P0×4, P1×4, C×1 ×4 |
|   2  |  61  |  32   | `naive:bbbi` | P0×5, P1×3, three C×1 and one C×2 |
|   3  |  62  |  24   | `naive:bbii` | P0×6, P1×2, two C×1 and two C×2 |
|   4  |  63  |   8   | `naive:biii` | P0×7, P1×1, one C×1 and three C×2 |
|   5  |  64  |   2   | `naive:iiii`, `batched_parallel:k_live=2` | see full table |
|   6  |  72  |   1   | `batched_parallel:k_live=4` | 8× P×1 and 4× C×1 |

Full sorted list: [`ranked_full.md`](ranked_full.md). Raw results:
[`results.json`](results.json).

## Findings

1. **16 strategies tie at the optimum, cost = 60.** They are exactly
   the 2^4 combinations where every pair is either `direct` or
   `batched` (no `indirect`). The degenerate `naive:dddd` (all direct,
   5 scratch cells) and `naive:bbbb` (all batched, 6 scratch cells)
   both land on 60.

2. **Every `indirect` pair costs exactly +1.** The read-count
   contribution an `indirect` pair adds to P0 is 2 (vs. 1 for direct
   or batched), and under optimal placement P0 sits at address 1, so
   each extra indirect pair costs 1 × (⌈√1⌉) extra. This produces the
   clean +1-per-indirect progression 60 → 61 → 62 → 63 → 64 as the
   indirect count goes 0 → 1 → 2 → 3 → 4.

3. **Keeping more products alive concurrently never helps.**
   `batched_parallel:k_live=2` costs 64 (tie with all-indirect) and
   `k_live=4` costs 72. The extra P cells dilute addresses to higher
   rings without reducing any individual read count.

4. **Arg layout is irrelevant under n^3.** Every arg cell is read
   exactly twice in any legal schedule, so permuting A and B on the
   arg stack rearranges cost contributions among identical-count
   cells and the total stays at 38.

5. **"Direct-first" is never worse than "indirect."** Since both
   modes have the same lifetime footprint and identical C-cell reads,
   but direct saves the ASSIGN's P0 read, direct strictly dominates
   indirect per pair. Any optimal strategy avoids indirect.

## Reproduce

```
python3 run_experiment.py
```

Writes `results.json`, `ranked_grouped.md`, `ranked_full.md`.

## Files

- `tracer.py` — flat-addr Manhattan cost model + `evaluate_layout`.
- `algorithms.py` — mode profiles + the 83-strategy enumerator.
- `run_experiment.py` — sweep runner + table writer.
- `ranked_full.md` — all 83 strategies sorted by cost.
- `ranked_grouped.md` — 6 rows, one per distinct cost value.
- `results.json` — raw output for downstream analysis.
