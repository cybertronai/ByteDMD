# optimal-cache-visualization

Side-by-side visualization of LRU, MRU, and Belady OPT cache eviction on a
mixed-reuse reference string that separates all three policies.

## What it shows

Reference string **A B C A D A B C D** (length 9, cache size 3).

| Policy | Hits | Misses | Why |
|---|---|---|---|
| LRU (Least Recently Used) | 2 | 7 | At t=4 evicts B (LRU item), missing it again at t=6 |
| MRU (Most Recently Used) | 3 | 6 | At t=4 evicts A (MRU item); keeps B and C for hits at t=6–7 |
| Belady OPT (Furthest Next Use) | 4 | 5 | At t=4 evicts C (next use t=7 > B's t=6 > A's t=5) |

### The key decision: t=4 (D miss, cache full with {A, B, C})

- **LRU** evicts **B** — B was loaded at t=1 and not used since, so it sits at the LRU
  position. But B is needed again at t=6, so evicting it now forces an extra miss.
- **MRU** evicts **A** — A was just used at t=3, making it the MRU item. But A is needed
  again at t=5, so evicting it costs a miss there. However, MRU keeps B and C (both
  accessed at t=1 and t=2, "old" by MRU's ordering), which pay off with hits at t=6 and t=7.
- **OPT** evicts **C** — C's next use is t=7, further than B's t=6 and A's t=5. By keeping
  the two soonest-needed items (A and B) and discarding the furthest one (C), OPT adds
  hits at t=5 (A) and t=6 (B) while only paying one extra miss when C is needed at t=7.

This trace uses a mix of short reuse (A: gap of 2), medium reuse (B, C, D: gap of 5),
and a single "hot" item (A appears 3×). That mixture prevents MRU from matching OPT
(as it would on a pure cyclic trace) and gives each policy a distinct eviction sequence.

## Visualization

Each colored line traces one variable's **cache slot position** over time
(X = time / access number, Y = cache slot).

- **Slot 0 (bottom)**: safest — most recently used (LRU) / least recently used (MRU) / soonest next use (OPT)
- **Slot k−1 (top of cache)**: eviction target
- **Above dashed line**: evicted (variable not in cache)
- **Solid line**: in cache; **dashed line**: evicted
- **Filled circle (●)**: cache hit; **✕**: cache miss

Lines connect per-tick centers with straight segments — diagonal when a
variable changes slot, horizontal when it stays.

![LRU vs MRU vs OPT](lru_vs_opt.png)

## Run

```bash
uv run lru_vs_opt_viz.py
# or
python lru_vs_opt_viz.py
```

Produces `lru_vs_opt.png` and `lru_vs_opt.svg` in this directory.
