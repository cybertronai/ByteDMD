"""
Strategies for 2x2 matrix multiplication under manual allocation.

Scope
-----
Only the n^3, (*, +)-semiring algorithm. Eight scalar products

    P_ijk = A[i][k] * B[k][j],   i, j, k in {0, 1}

four scalar sums

    C[i][j] = P_ij0 + P_ij1.

No Strassen / Winograd / other bilinear-reduction schemes — those use
cancellations outside (*, +).

Cost model
----------
Flat Manhattan distance. Each read of a cell at address `a` costs
ceil(sqrt(a)); writes are free. Two independent address spaces:

    arg stack:      A @ 1..4, B @ 5..8   (read-only inputs)
    scratch stack:  P0 / P1 / C cells    (bump-pointer allocated)

Addresses come from an explicit bump-pointer allocator — nothing is
relocated. Under this model the *order* of independent reads does not
change the total cost; only the *layout* (which cell lives at which
address) and the *read profile* (how many times each cell is read) do.
When all cells are mutually live, optimal placement sorts them by read
count descending and assigns addresses 1, 2, 3, ...

Strategy space (the exhaustive sweep)
-------------------------------------
For each of the four output cells C[i][j] we pick how its two products
get assembled:

    'direct'   MUL1 writes directly to C[i][j]; MUL2 writes to the
               shared scratch slot P0; then the ADD reads C[i][j] and
               P0 and writes back to C[i][j].
               Scratch reads for this pair: 1 P0 + 1 C = 2.
    'indirect' MUL1 writes to P0; ASSIGN copies P0 to C[i][j]; MUL2
               writes to P0 (overwrite); ADD reads C[i][j] and P0.
               Scratch reads for this pair: 2 P0 + 1 C = 3.
    'batched'  MUL1 writes to P0; MUL2 writes to P1; ADD reads P0
               and P1 and writes C[i][j].
               Scratch reads for this pair: 1 P0 + 1 P1 = 2.

Each C cell is also read once at the end (epilogue — the caller
reads the output).

Sharing. P0 is reused across every pair regardless of mode (it's the
"first scratch product slot"). P1 is allocated only when at least one
pair is batched and is reused across batched pairs. With sequential
processing of pairs, one P0 and at most one P1 suffice.

Concurrent-batched variants keep multiple pairs' P0/P1 cells alive at
the same time so each P cell is read exactly once: k_live=2 splits the
four pairs into two waves of two; k_live=4 is the maximally-stored
variant with all 8 products live simultaneously. These are sampled
alongside the 3^4 = 81 mode combinations.

Total = 3^4 + 2 = 83 strategies.
"""

from itertools import product

from tracer import addr_cost, evaluate_layout


MODES = ('direct', 'indirect', 'batched')

# Every arg cell is read exactly twice under any n^3 schedule: A[i][k]
# once per j (2 values of j), B[k][j] once per i.
NAIVE_ARG_COUNTS = [2] * 8


# ---------------------------------------------------------------------------
# Mode profiles
# ---------------------------------------------------------------------------

def naive_profile(modes):
    """Return (scratch_counts, arg_counts, cells_label) for a 4-tuple
    of modes (one per C cell in lex order (0,0), (0,1), (1,0), (1,1)).
    """
    n_d = modes.count('direct')
    n_i = modes.count('indirect')
    n_b = modes.count('batched')
    assert n_d + n_i + n_b == 4

    cells = []   # (name, read_count)

    # P0 is shared across all modes: direct contributes 1 read (the
    # ADD step's P0 operand), indirect 2 (ASSIGN + ADD), batched 1.
    p0_reads = n_d + 2 * n_i + n_b
    cells.append(('P0', p0_reads))

    if n_b > 0:
        cells.append(('P1', n_b))

    for idx, m in enumerate(modes):
        c_mid = 1 if m in ('direct', 'indirect') else 0
        c_epi = 1
        cells.append((f'C{idx}', c_mid + c_epi))

    scratch = [c for _, c in cells]
    return scratch, list(NAIVE_ARG_COUNTS), cells


def naive_cost(modes):
    scratch, argc, _ = naive_profile(modes)
    return evaluate_layout(scratch, argc)


# ---------------------------------------------------------------------------
# Concurrent-batched variants
# ---------------------------------------------------------------------------
# k_live batched pairs keep their P0/P1 cells live at once. Since P0/P1
# are allocated afresh for each wave, k_live=1 is redundant with the
# sequential all-batched case and is omitted.

def batched_concurrent_profile(k_live):
    assert k_live in (2, 4) and 4 % k_live == 0
    n_P = 2 * k_live
    reads_per_P = 4 // k_live   # how many waves share each P cell
    cells = [(f'P{p}', reads_per_P) for p in range(n_P)]
    cells += [(f'C{idx}', 1) for idx in range(4)]   # epilogue only
    scratch = [c for _, c in cells]
    return scratch, list(NAIVE_ARG_COUNTS), cells


def batched_concurrent_cost(k_live):
    scratch, argc, _ = batched_concurrent_profile(k_live)
    return evaluate_layout(scratch, argc)


# ---------------------------------------------------------------------------
# Strategy enumerator
# ---------------------------------------------------------------------------

def strategy_label(modes):
    return ''.join(m[0] for m in modes)   # e.g. 'dddd', 'dibi'


def all_strategies():
    """Yield dicts describing every strategy in the sweep."""
    for modes in product(MODES, repeat=4):
        modes_t = tuple(modes)
        scratch, argc, cells = naive_profile(modes_t)
        yield {
            'family': 'naive',
            'name': f'naive:{strategy_label(modes_t)}',
            'modes': modes_t,
            'cells': cells,
            'scratch_reads': scratch,
            'arg_reads': argc,
            'cost': evaluate_layout(scratch, argc),
        }

    for k in (2, 4):
        scratch, argc, cells = batched_concurrent_profile(k)
        yield {
            'family': 'batched_parallel',
            'name': f'batched_parallel:k_live={k}',
            'modes': ('batched',) * 4,
            'cells': cells,
            'scratch_reads': scratch,
            'arg_reads': argc,
            'cost': evaluate_layout(scratch, argc),
        }
