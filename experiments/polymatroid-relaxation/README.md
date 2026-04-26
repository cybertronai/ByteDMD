# polymatroid-relaxation — discrete LP lower bound on static allocators

Implements the discrete polymatroid relaxation from
[`gemini/polymatroid-relaxation.md`](../../gemini/polymatroid-relaxation.md):
a sequence of totally-unimodular LPs over the maximal cliques of the
interval graph, combined with the discrete-calculus (Abel-summation)
identity for `⌈√d⌉`.

## How it works

1. Extract per-variable liveness intervals (Two-Stack convention:
   inputs scoped `[first L2Load, last L2Load]` with their compulsory
   first read charged separately to the arg stack).
2. Find the maximal cliques of the interval graph — each clique is a
   set of variables alive simultaneously.
3. For each capacity `k`, the LP

       max  Σ_v reads_v · x_v
       s.t. Σ_{v ∈ K} x_v ≤ k    for every maximal clique K
            0 ≤ x_v ≤ 1

   solves for `M[k]` = max total reads packable into `k` distinct
   physical addresses. Two miracles make this exact:
   - **Consecutive-Ones Property → Total Unimodularity.** Each
     interval graph variable's column is a contiguous block of 1s
     when cliques are sorted chronologically. Any TU LP has integer
     vertex solutions — HiGHS finds the integer optimum directly.
   - **Interval graphs are perfect graphs.** Max-clique = chromatic
     number, so a `k`-bounded clique selection always corresponds to
     a valid `k`-address packing.
4. **Discrete-calculus identity.** Decompose `⌈√d⌉ = 1 + Σ_{c ≥ 1, c² ≤ d−1} 1`
   so the per-load fetch cost is a sum of unit step jumps at
   `j ∈ {1, 2, 5, 10, 17, …}` = `{c² + 1 : c ≥ 0}`. Total cost then
   collapses to

       LB = R_total + Σ_{c=1..⌊√(ω−1)⌋} (R_total − M[c²])

   so we only need to solve the LP at *square* capacities — `O(√ω)`
   LPs instead of `O(ω)`. ω is the peak live size.

The compulsory arg-stack first-load cost `Σ_inputs ⌈√(arg_idx)⌉` is
added on top so the bound is directly comparable to `static_opt_lb`,
`splitting_lower_bound`, etc.

## Run

    ./run.py        # solves LPs for the representative algos and
                    # writes results.csv + results.md

`scipy` is the only extra dependency (declared via uv-script meta).

## Comparison against existing bounds

LP totals (all are *lower bounds* on the cost a static allocator must
pay, except `bytedmd_live` and `manual` which are achievable costs).

| algorithm                       | polymatroid_lb | split_lb | static_opt_lb | bytedmd_opt | space_dmd | bytedmd_live | manual  |
|--------------------------------|---------------:|---------:|--------------:|------------:|----------:|-------------:|--------:|
| naive_matmul(n=16)              |         30,052 |   75,234 |        75,671 |     111,132 |    80,501 |      109,217 | 177,744 |
| naive_2d_tiled_matmul(n=16,T=4) |         30,052 |   65,571 |        88,791 |      95,315 |    91,253 |       95,634 | 177,744 |
| tiled_matmul(n=16)              |         31,124 |   53,900 |        94,537 |      79,329 |    95,487 |       78,708 |  67,758 |
| rmm(n=16)                       |         39,952 |   57,840 |       110,833 |      84,470 |   110,373 |       83,196 | 106,835 |
| naive_strassen(n=16)            |        142,584 |  125,078 |       145,059 |     184,965 |   138,024 |      175,157 | 251,486 |
| matvec_row(n=64)                |        197,467 |  211,973 |       212,304 |     229,752 |   217,272 |      229,527 | 218,552 |
| matvec_col(n=64)                |        213,336 |  211,904 |       212,183 |     229,667 |   213,631 |      229,716 | 217,952 |
| matvec_blocked(n=64,B=8)        |        206,865 |  202,963 |       203,485 |     215,068 |   207,429 |      214,377 | 208,832 |
| fft_iterative(N=256)            |         42,627 |   31,631 |        39,049 |      46,033 |    40,200 |       47,088 |  55,516 |
| fft_recursive(N=256)            |         31,316 |   23,077 |        27,237 |      31,665 |    29,450 |       33,110 |  52,704 |
| lu_no_pivot(n=32)               |        690,040 |  271,413 |       600,081 |     407,900 |   482,165 |      407,042 | 405,592 |
| blocked_lu(n=32,NB=8)           |        507,915 |  189,462 |       420,494 |     283,615 |   366,551 |      283,294 | 250,767 |
| cholesky(n=32)                  |        292,337 |  121,746 |       245,659 |     177,026 |   186,117 |      176,313 | 251,039 |
| transpose_naive(n=32)           |         44,704 |   38,611 |        38,614 |      44,704 |    44,704 |       44,704 |  44,704 |
| transpose_blocked(n=32)         |         44,704 |   38,571 |        38,573 |      44,704 |    43,298 |       43,873 |  44,704 |
| quicksort(N=64)                 |          1,386 |    1,987 |         2,791 |       2,767 |     2,470 |        2,852 |   4,718 |
| mergesort(N=64)                 |          2,786 |    2,110 |         2,261 |       3,030 |     2,572 |        3,148 |   3,386 |
| lcs_dp(32x32)                   |         20,168 |   19,992 |        25,098 |      30,878 |    26,583 |       29,980 |  27,192 |

(`results.csv` adds the `bytedmd_classic`, `n_events`, and per-row LP
runtime columns.)

## Observations

- **`polymatroid_lb` and `static_opt_lb` are different relaxations**
  of the same physical model (no splitting, no compaction). They are
  not generally ordered:
  - `polymatroid_lb < static_opt_lb` on read-uniform traces with
    long-lived inputs (matmul family, matvec). The discrete clique
    LP only constrains how reads can be *packed*, while
    `static_opt_lb`'s density-weighted time integral charges every
    var for every tick of its lifespan — the latter is much larger
    when inputs are alive throughout a long compute phase.
  - `polymatroid_lb > static_opt_lb` on phase-structured traces
    (LU, Cholesky, blocked_lu). The discrete clique LP pushes
    against the peak live set which is large in these traces (the
    full submatrix or panel), while the density-weighted integral
    benefits from low-density vars dropping to back ranks.
- **`polymatroid_lb` exactly equals `bytedmd_opt`** for the two
  transpose variants and `naive_matmul`-shaped traces with one read
  per cell. In that regime `M[c²] = c²` for every `c ≤ √ω`, the LP
  collapses to the simple address ladder, and the discrete identity
  recovers `Σ_{cells} ⌈√(rank)⌉` directly.
- **`split_lb` ≤ `polymatroid_lb` is not universal.** They bound
  different optima (a splitting / DMA allocator vs. a pure static
  allocator). On matmul-style traces `polymatroid_lb < split_lb`
  because polymatroid's purely discrete relaxation drops the
  density information that splitting's per-burst integration
  captures; on LU/Cholesky the order flips.
- **`manual < polymatroid_lb` happens** on `lu_no_pivot`,
  `blocked_lu`, and `cholesky` (and on three other rows already
  flagged in the grid README). That's the same well-documented
  phenomenon: the manual schedule is implementing a fused /
  in-place / streaming variant whose effective trace is much
  smaller than the abstract DAG the LP is bounding.

## Notes / scope

- LP runtime scales super-linearly in the number of intervals; the
  representative subset is sized so the full sweep finishes in ~9
  minutes on a workstation. Larger traces (`naive_attn`,
  `regular_conv`, `lu_partial_pivot`, `householder_qr`) are deferred —
  the LP at peak `ω ≈ 1k` grows to a few minutes per capacity.
- The current bound uses Two-Stack semantics for inputs (compulsory
  arg-stack first-load cost added on top of the LP). Drop that
  prefix to recover the pure interval-LP value comparable to
  `mwis_lower_bound` and `lp_lower_bound`.
