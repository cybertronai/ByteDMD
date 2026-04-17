# Hardware-Agnostic Cache-Friendliness Heuristics

To build heuristics that evaluate the intrinsic "cache-friendliness" of an algorithm independently of any specific hardware, physical cache model (like ByteDMD), or dynamic allocation strategy, you must analyze its **1D Execution Trace and Dataflow Topology**.

When comparing these four algorithms, your heuristics must capture two completely different types of computational flaws:

- **Schedule Flaws** (Naive vs. Recursive Matmul): Both execute the exact same $\mathcal{O}(N^3)$ operations. The difference is purely the order in which they visit the data.
- **Materialization Flaws** (Regular vs. Flash Attention): Flash Attention mathematically alters the dataflow graph to eliminate massive intermediate variables that Regular Attention creates.

Here are five powerful, hardware-agnostic heuristics you can calculate directly from a linear execution trace to definitively rank these algorithms.

---

## 1. Mean Unique Reuse Distance (Mattson's Stack Distance)

**The Concept:** When the algorithm accesses a variable $X$, how much other stuff does it touch before it accesses $X$ again? For every memory access in your trace, count the number of unique memory addresses accessed since the last time that specific address was touched. The heuristic is the mean distance across the trace.

**Why it ranks them correctly (Temporal Locality):**

- **Naive Matmul:** To compute $C_{i,j}$, the inner loop reads a column of $B$. The next time the algorithm needs $B_{0,j}$, it has moved on to computing the next row $C_{i+1, j}$. In between, it touched the entire length of the previous row and column. The average reuse distance scales with $\mathcal{O}(N)$.
- **Recursive Matmul:** Because it fully processes sub-blocks before moving on, data is pounded with operations and then abandoned. The reuse distance never exceeds the block size $\mathcal{O}(K^2)$.
- **Regular vs. Flash Attention:** Regular Attention computes the $N \times N$ matrix $S = QK^T$. It then does $\mathcal{O}(N^2)$ other operations before reading $S_{0,0}$ again to apply the Softmax. Flash Attention calculates a tiny block of $S$, immediately applies Softmax, and consumes it. The intermediate reuse distance drops to practically zero.

---

## 2. Sliding-Window Arithmetic Intensity (Dynamic Surface-to-Volume)

**The Concept:** A cache-friendly algorithm is one that can do a massive amount of math while trapped inside a very small working set of data. Slide a window of $W$ sequential operations (e.g., $W = 1{,}000$ FLOPs) across your trace. For each window, count the number of unique memory addresses $U(W)$ accessed. The score is the average ratio of $\frac{W}{U(W)}$.

**Why it ranks them correctly (The Hong-Kung Lower Bound):**

- **Naive Matmul:** In a window of $W$ operations, the algorithm sweeps linearly across rows and columns. It touches an amount of data strictly proportional to $W$. The math-to-memory ratio is a flat $\mathcal{O}(1)$.
- **Recursive Matmul:** Because the trace forms a 3D computational cube (a Hamiltonian path), a window of $W$ operations is trapped inside a local sub-block. To do $W$ operations, it only needs to touch the 2D surface area of that block: $U(W) \approx W^{2/3}$. The intensity grows as $\mathcal{O}(W^{1/3})$. RMM gets exponentially more math done per byte fetched.
- **Flash Attention:** Similarly traps execution inside an SRAM tile, resulting in massive spikes in chunked computational density compared to the long linear sweeps of Regular Attention.

---

## 3. Peak Concurrent Liveness (Maximum Working Set)

**The Concept:** How much temporary "garbage" does the algorithm generate that must be kept alive simultaneously? A variable is "live" from the tick it is first written until the tick it is read for the absolute final time. Sweep the trace and track the maximum number of variables that are simultaneously live.

**Why it ranks them correctly (Operator Fusion):**

- **Regular Attention:** Materializes the entire $N \times N$ attention matrix $S$. It must be kept fully alive while the Softmax denominator is calculated. Peak liveness is $\Omega(N^2)$. This is a catastrophic memory wall.
- **Flash Attention:** Tiling allows the algorithm to fuse operations. It computes a block of $S$, normalizes it using online running statistics, multiplies it by $V$, and instantly discards the intermediate block. The peak liveness never exceeds the I/O vectors plus the block size: $\mathcal{O}(N \cdot d)$.

> **Note:** Naive and Recursive Matmul have roughly the same peak liveness of $3N^2$ for matrices A, B, and C, so this specific heuristic separates the Attention algorithms while relying on Heuristics 1 & 2 to separate the Matmuls.

---

## 4. The Spacetime Liveness Integral (Byte-Ticks)

**The Concept:** Imagine memory capacity is rented by the millisecond. Every byte costs \$1 for every operation it remains "alive." The heuristic is the integral (sum) of Lifespan $\times$ Size for all variables.

**Why it ranks them correctly (Data Pollution):**

- **Regular Attention:** Creates the $N \times N$ intermediate matrix and leaves it idle, doing nothing, while row-wise maxes and denominators are computed. The Spacetime Integral scales catastrophically to $\mathcal{O}(N^4)$ (an $N^2$ footprint kept alive for $N^2$ operations).
- **Flash Attention:** Instantly kills intermediates, keeping the footprint to $\mathcal{O}(Nd)$ and the spacetime integral tiny.
- **Naive vs. Recursive Matmul:** If you use a naive loop order, partial sums of the output matrix $C$ are kept "open" and alive across the entire $N^3$ execution. Recursive Matmul completes a sub-block of $C$ entirely, permanently closing its lifespan early and drastically dropping the integral score.

---

## 5. Area Under the Oracle Curve (Belady's MIN AUC)

**The Concept:** To prove how an algorithm performs under *all possible allocators*, grade it against the absolute mathematical ceiling: **Belady's MIN algorithm** (the theoretically optimal offline cache policy that always evicts the variable needed furthest in the future). Simulate the trace through MIN for cache capacities $C \in \{1, 2, 3 \ldots \text{Peak}\}$. Plot Cache Size (X) vs. Total Misses (Y), and calculate the Area Under the Curve (AUC).

**Why it ranks them correctly (The Unbreakable Bound):**

- MIN strips away all artifacts of "dumb" algorithms like LRU thrashing. If an algorithm has a high AUC under MIN, it is fundamentally hostile to any memory hierarchy in the universe.
- **Naive Matmul & Regular Attention:** Even the magical MIN oracle cannot save them; it is mathematically forced to thrash the cache, keeping the miss rate near 100% until the cache size reaches the massive $N^2$ footprints. Huge AUC.
- **RMM & Flash Attention:** The fractal nature of RMM and the blocked nature of Flash mean their miss rates collapse to near-zero incredibly early on the X-axis (the moment the cache size matches the block size), squashing the AUC leftward and downward perfectly.
