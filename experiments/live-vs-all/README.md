# Classic DMD · DMD-live · Tombstone

Three-column experiment: two ByteDMD measures priced directly on the L2
trace (Classic DMD, DMD-live) plus one concrete stationary-slot allocator
(Tombstone). Ref: `gemini/15apr26-dmdlive-analysis.md`.

## The three measures

| Column        | What it is                                                                | RMM asymptotic     |
|---------------|---------------------------------------------------------------------------|--------------------|
| Classic DMD   | LRU stack, **no** liveness — dead vars pollute deeper rings ("graveyard") | `Θ(N^{3.5})`       |
| DMD-live      | LRU stack **with** liveness — dead vars vaporize and everything above slides inward for free ("teleporting cache") | `Θ(N^3 log N)`     |
| Tombstone     | Stationary slots with tombstone reuse — dead vars leave holes, new vars take the closest (smallest-addr) free slot; live vars don't move | `Θ(N^4)` empirical |

Classic DMD and DMD-live are computed directly on the L2 event stream
with an LRU walk (Fenwick-tree indexed, O(log T) per op). Tombstone
lowers L2 to L3 via `bytedmd_ir.compile_min_heap` and then applies
`sum ceil(sqrt(addr))` on L3 LOADs.

## The three IR levels

- **L1** — Python source (algorithms as plain functions).
- **L2** — abstract IR: `LOAD(var)`, `STORE(var)`, `OP(name, in, out)`.
- **L3** — concrete IR: same events, each `var` carries a physical `addr`
  assigned by an allocator.

## Results

### Cache-oblivious RMM (8-way)

|  N  | Classic DMD  | DMD-live    | Tombstone     |
|----:|-------------:|------------:|--------------:|
|   4 |        1,043 |         689 |           985 |
|   8 |       13,047 |       7,773 |        16,315 |
|  16 |      154,251 |      80,716 |       266,593 |
|  32 |    1,779,356 |     794,969 |     4,320,478 |
|  64 |   20,291,116 |   7,554,413 |    69,716,078 |

### Tiled matmul (one level, tile = ⌈√N⌉)

|  N  | Classic DMD  | DMD-live    | Tombstone     |
|----:|-------------:|------------:|--------------:|
|   4 |        1,000 |         644 |           961 |
|   8 |       12,368 |       7,210 |        15,128 |
|  16 |      143,280 |      74,560 |       233,811 |
|  32 |    1,740,310 |     790,183 |     3,683,154 |
|  64 |   19,737,581 |   7,917,595 |    57,162,017 |

**Asymptotics** (fit across N = 4…64):
`Classic ≈ 9.6 · N^{3.5}`, `Live ≈ 4.8 · N^3 log₂ N`, `Tombstone ≈ 4 · N^4`.

For RMM, Tombstone sits *above* Classic DMD for `N ≥ 8` — the
stationary-slot penalty (`√(O(N^2)) = O(N)` per read) grows faster with
N than the graveyard penalty that LRU bumping tames. This is the central
quantitative finding of the experiment: DMD-live's magic sliding saves
a full `N / log N` factor over realistic stationary reuse.

See **[REPORT.md](REPORT.md)** for the full writeup with physics
picture and asymptotic derivation.

## Reproducibility

```bash
uv run pytest test_bytedmd_ir.py
uv run --script envelope.py 4,8,16,32,64
```
