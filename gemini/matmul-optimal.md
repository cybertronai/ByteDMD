# VLSI Matrix-Vector Multiplication Analysis

**Source:** https://gemini.google.com/share/30594364a3ce
**Created with:** Gemini 3 Deep Think, April 9, 2026
**Topic:** Systolic array to optimally do matmul and matvec using Thompson 2D VLSI model

---

## Prompt
Using the Thompson 2D VLSI model (spatial computer) to analyze the complexity of N*N matrix-vector multiplication. What is the energy lower bound, is it possible to achieve it, what is a low energy way to implement it, give detailed specification of a spatial computer that implements it.

## 1. Energy Lower Bound

The absolute energy lower bound is **Omega(N^2)**.

To store the N^2 elements of dense matrix A, they must be distributed across at least N x N processors (diameter Theta(N)). Routing each x_j to all N matrix elements in column j requires Omega(N) energy. For all N elements: N * Omega(N) = Omega(N^2). A symmetric Omega(N^2) cost applies for gathering partial products.

## 2. Is it Achievable?

**Yes.** Matrix-vector multiplication avoids the Omega(N^1.5) permutation bottleneck that affects matmul. The matrix stays completely stationary; only vectors are routed, keeping communication strictly localized to hit Theta(N^2).

Note: Using a 1D Broadcast tree for O(log N) depth increases energy to sub-optimal O(N^2 log N). Optimal energy requires accepting O(N) depth.

## 3. Low-Energy Implementation: 2D Systolic Wavefront Pipeline

Strategy: Matrix A remains stationary. Vector x is streamed vertically downward column-by-column. As x elements pass over A elements, they are locally multiplied. Output sums for y are streamed horizontally left row-by-row. All messages travel to immediate neighbors only (energy cost = 1 per message).

## 4. Detailed Spatial Computer Specification

### Hardware Architecture per Node
- **Grid:** N x N subgrid, processor p_{i,j} at coordinates (i,j)
- **Local Memory (O(1)):** Four registers:
  - A_val: permanently stores A_{i,j}
  - x_val: temporarily stores streaming vector element x_j
  - prod: locally computed scalar product
  - sum_val: accumulating horizontal sum
- **Queues:** Constant-sized receive queue (max 2 messages per time-step)
- **ALU:** O(1) arithmetic per cycle (fused multiply-add)

### Initial State
- p_{i,j} holds A_{i,j} in A_val
- Top row p_{0,j} holds input x_j
- Output materializes at leftmost column p_{i,0}

### Execution Protocol (Synchronous)
Diagonal wavefront. Each clock cycle t, all processors run:

**Phase 1: Vector Shift & Local Multiply** (at t == i)
- Receive x_j from North (if i > 0)
- prod = A_val * x_val
- Send x_val to South (if i < N-1)

**Phase 2: Row Reduction** (at t == i + (N-1-j))
- Receive partial sum from East (or start at 0 if rightmost)
- sum_val = sum_val + prod
- Send sum_val to West (or terminate if j == 0)

### Final Complexity
- **Total Energy:** 2N^2 - 2N = Theta(N^2) — matches the lower bound
- **Depth:** 2N - 2 = Theta(N)
- **Wire-Depth:** Theta(N) — no message travels distance > 1
