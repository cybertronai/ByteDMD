# Simplified Explicit Communication Model

Bill Dally ([*On the Model of Computation*, CACM
2022](https://cacm.acm.org/opinion/on-the-model-of-computation-point/))
proposed modeling algorithm data movement explicitly on the Manhattan
grid.

This is a simplified implementation of that model for a single
processor, designed to price a single function call.

![Upper-half-plane Manhattan layout](simplified_explicit_communication_model.svg)

- Processor is at the origin, memory is arranged as a 2D grid in the
  upper half-plane around it.
- Every cell is linearly indexed; `ceil(sqrt(idx))` gives the Manhattan
  distance from the core.

## Cost model (what is priced)

- **Reads are priced.** The cost of a read is the Manhattan distance
  from the core to the cell being read.
- **Writes are free.**
- **Arithmetic is free.**

## Function semantics

- **At the start of a call**, the location of every input byte is
  specified by the caller. Inputs are placed on the grid for free
- **At the end of a call**, the location of every output byte is
  specified by the caller, these incur standard read cost.

## Worked example

```python
def myfunc(a, b, c, d, e):
    return a*b + c*d + e
```

Five input bytes (`a, b, c, d, e`) and one output byte. The
computation reads each operand of each arithmetic op separately —
arithmetic and intermediate writes are free, only operand reads are
priced.

**Caller-chosen layout.** The caller picks where each input byte
lives. With five inputs, at most four can sit at distance ≤ 2 (the
disc of radius 2 holds 4 cells); one input has to land at distance 3.
The cheapest assignment is to put one input at addr 1 (the only
distance-1 cell), three at addrs 2..4 (distance 2), and the fifth at
addr 5 (distance 3). For this example we put `a→1, b→2, c→3, d→4,
e→5`. Initial placement is free.

| addr | cell | `⌈√addr⌉` |
|-----:|------|----------:|
| 1    | `a`  | 1 |
| 2    | `b`  | 2 |
| 3    | `c`  | 2 |
| 4    | `d`  | 2 |
| 5    | `e`  | 3 |

**Execution.** The function evaluates `(a*b) + (c*d) + e` left-to-right
and reuses low addresses for intermediates as soon as their previous
contents become dead.

| step | action                                  | reads             | cost |
|-----:|-----------------------------------------|-------------------|-----:|
| 1    | `t1 = a * b`, write `t1 → 1`            | `a@1`, `b@2`      | 1+2  |
| 2    | `t2 = c * d`, write `t2 → 2`            | `c@3`, `d@4`      | 2+2  |
| 3    | `s  = t1 + t2`, write `s  → 1`          | `t1@1`, `t2@2`    | 1+2  |
| 4    | `r  = s + e`, write `r  → 1`            | `s@1`, `e@5`      | 1+3  |

In step 1 the multiply consumes `a` (addr 1) and `b` (addr 2) and
writes `t1` back to addr 1 — `a` is dead after this read, so addr 1
is free for reuse. Step 2 does the same for `c, d` at addr 2. Steps 3
and 4 fold the sums into addr 1.

**Exit.** The caller asks for the result at addr 1, so the function
leaves `r` there. The exit read costs `⌈√1⌉ = 1`.

**Total cost:**
`(1+2) + (2+2) + (1+2) + (1+3) + 1 = 15`.
