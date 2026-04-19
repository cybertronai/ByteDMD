#!/usr/bin/env python3
"""
Enumerate every 2x2 n^3 matmul strategy under manual allocation,
rank by DMD cost, and emit results.json + a sorted markdown table.
"""
import json
import os
from collections import Counter, defaultdict

from algorithms import all_strategies


HERE = os.path.dirname(__file__)


def main():
    strategies = list(all_strategies())

    # Serialize every strategy with enough detail to reconstruct the
    # layout and read profile.
    rows = []
    for s in strategies:
        rows.append({
            'name': s['name'],
            'family': s['family'],
            'modes': list(s['modes']),
            'cells': [[name, n] for name, n in s['cells']],
            'scratch_reads': s['scratch_reads'],
            'arg_reads': s['arg_reads'],
            'cost': s['cost'],
        })

    rows.sort(key=lambda r: (r['cost'], r['name']))

    out_json = os.path.join(HERE, 'results.json')
    with open(out_json, 'w') as f:
        json.dump(rows, f, indent=2)

    # Grouped table: one row per unique cost.
    groups = defaultdict(list)
    for r in rows:
        groups[r['cost']].append(r)

    counts = Counter(r['cost'] for r in rows)
    table_lines = [
        '| rank | cost | count | example | cells + read counts |',
        '|-----:|-----:|------:|---------|---------------------|',
    ]
    for rank, (cost, rs) in enumerate(sorted(groups.items()), start=1):
        ex = rs[0]
        cells_desc = ', '.join(f'{n}×{c}' for n, c in ex['cells'])
        table_lines.append(
            f"| {rank} | {cost} | {counts[cost]} | "
            f"`{ex['name']}` | {cells_desc} |"
        )
    grouped = '\n'.join(table_lines)

    # Full table (all 83 strategies, sorted by cost).
    full_lines = [
        '| rank | strategy | cost | peak_cells | scratch profile |',
        '|-----:|----------|-----:|-----------:|-----------------|',
    ]
    for rank, r in enumerate(rows, start=1):
        profile = '[' + ', '.join(str(x) for x in sorted(
            r['scratch_reads'], reverse=True)) + ']'
        full_lines.append(
            f"| {rank} | `{r['name']}` | {r['cost']} | "
            f"{len(r['scratch_reads'])} | {profile} |"
        )
    full = '\n'.join(full_lines)

    out_grouped = os.path.join(HERE, 'ranked_grouped.md')
    with open(out_grouped, 'w') as f:
        f.write(grouped + '\n')
    out_full = os.path.join(HERE, 'ranked_full.md')
    with open(out_full, 'w') as f:
        f.write(full + '\n')

    # Console summary.
    print(f"Enumerated {len(rows)} strategies.\n")
    print("Distinct cost values and their populations:")
    for cost, n in sorted(counts.items()):
        print(f"  cost={cost}: {n} strategies")
    print(f"\nWrote: results.json, ranked_grouped.md, ranked_full.md")


if __name__ == '__main__':
    main()
