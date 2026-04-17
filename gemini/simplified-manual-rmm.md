# Simplified Manual RMM: The Pyramid and the Forklift

## The Simplified Code

By replacing the massive blocks of manually unrolled quadrant math with a clean triple-loop, the logic now perfectly mirrors standard block-matrix multiplication ($C_{i,j} += A_{i,k} \times B_{k,j}$).

```python
def dma_copy(alloc, src, src_stride, sr, sc, dst, dst_stride, dr, dc, size):
    """Software DMA: Explicitly copies a 2D block from src to dst."""
    for i in range(size):
        for j in range(size):
            val = alloc.read(src + (sr + i)*src_stride + (sc + j))
            alloc.write(dst + (dr + i)*dst_stride + (dc + j), val)

def rmm_explicit(alloc, ptrs, size, pA, pB, pC, stride):
    # BASE CASE: Math executes natively at the absolute fastest physical addresses
    if size == 1:
        a, b, c = alloc.read(pA), alloc.read(pB), alloc.read(pC)
        alloc.write(pC, c + a * b)
        return

    H = size // 2
    sA, sB, sC = ptrs[H]['A'], ptrs[H]['B'], ptrs[H]['C'] # 's' = fast scratchpad

    # Triple loop mapping directly to the 4 quadrants of the matrices
    for i in (0, H):
        for j in (0, H):
            # 1. DMA Load C quadrant from slow Parent down to fast Child
            dma_copy(alloc, pC, stride, i, j, sC, H, 0, 0, H)

            for k in (0, H):
                # 2. DMA Load A and B quadrants down to fast Child
                dma_copy(alloc, pA, stride, i, k, sA, H, 0, 0, H)
                dma_copy(alloc, pB, stride, k, j, sB, H, 0, 0, H)

                # 3. Recurse entirely within the fast Child memory!
                rmm_explicit(alloc, ptrs, H, sA, sB, sC, H)

            # 4. DMA Store the finished C quadrant back up to slow Parent
            dma_copy(alloc, sC, H, 0, 0, pC, stride, i, j, H)
```

## Walkthrough: The Pyramid and the Forklift

If you perform standard recursive matrix multiplication on a stationary $N \times N$ matrix, the tiny $1 \times 1$ base cases at the bottom of the tree are forced to fetch numbers from physical addresses as deep as $N^2$. This triggers massive continuous distance penalties, degrading the performance to a catastrophic $\mathcal{O}(N^4)$.

To fix this, the code implements a **Pyramid and Forklift** strategy:

1. **The Pyramid** (`allocate_hierarchy`): It permanently reserves the absolute lowest, fastest physical memory addresses exclusively for the $1 \times 1$ base cases. Behind them, it reserves addresses for $2 \times 2$ tasks, and so on. The massive inputs sit at the very end of memory.

2. **The Forklift** (`dma_copy`): Before the CPU works on a sub-problem, a Software DMA "forklift" physically hauls the required $A, B,$ and $C$ sub-matrices down from the slower Parent memory into the faster Child memory.

3. **Speed of Light**: By the time the recursion bottoms out, the required data has been actively relayed all the way down to addresses 1, 2, and 3. The actual computational math happens at a speed-of-light cost.

## The Master Theorem: Why is it exactly $\Theta(N^3 \log N)$?

To understand the bound, we evaluate the "overhead" of the DMA forklift at a given recursion level of size $K$.

- **The Volume:** The forklift copies sub-matrices of size $K/2 \times K/2$, touching $\Theta(K^2)$ elements.
- **The Distance Penalty:** Because of our pyramid layout, the parent data lives at an address depth proportional to $K^2$. Because your physical model penalizes reads by $\sqrt{\text{address}}$, the distance penalty is $\sqrt{K^2} = \mathbf{\Theta(K)}$.
- **The Total Transfer Cost:** $\text{Volume} \times \text{Distance Penalty} = K^2 \times K = \mathbf{\Theta(K^3)}$.

This creates the classic divide-and-conquer recurrence relation:

$$ T(K) = 8T(K/2) + \Theta(K^3) $$

**The Mathematical Resonance:** The algorithm splits into 8 branches. The volume of the matrix shrinks by a factor of $(2)^3 = \mathbf{8}$. Because the branching factor perfectly matches the shrinkage, the algorithm is in perfect resonance. Every single layer of the recursion tree does the exact same amount of data-movement work: $\Theta(N^3)$.

Because a tree halving at each step has exactly $\log_2 N$ layers, we multiply the work per layer by the depth, yielding an inescapable grand total of $\Theta(N^3 \log N)$.

## Connection to Existing Results & 2D Physics Bounds

The fact that you arrived at exactly $\Theta(N^3 \log N)$ under a $\lceil\sqrt{\text{address}}\rceil$ cost model is no coincidence. You have independently simulated one of the most profound mathematical singularities in hardware design: **The 2D Communication Boundary**.

### The Hierarchical Memory Model (1987)

In 1987, IBM researchers Alok Aggarwal, Bowen Alpern, Ashok Chandra, and Marc Snir published the landmark paper "A Model for Hierarchical Memory" (HMM). They argued that theoretical computer science was flawed for assuming RAM access was a flat $O(1)$ cost.

They evaluated algorithms under a continuous polynomial distance penalty $f(x) = x^\alpha$. They proved mathematically that when $\alpha = 0.5$, standard matrix multiplication evaluates to exactly $\Theta(N^3 \log N)$.

### Why $\alpha = 0.5$? The Physics of a 2D Microchip

If you build a central CPU on a flat silicon wafer, the physical area grows quadratically ($\text{Area} \propto r^2$). Therefore, to store $x$ bytes of data, the physical radius of your memory must grow as $r \propto \sqrt{x}$. Since routing data over wires consumes energy and time linearly with physical distance, fetching the $x$-th byte of data on a planar 2D chip mathematically costs $\Theta(\sqrt{x})$.

Your Continuous Cache simulation is a perfect proxy for 2D space. The $\Theta(N^3 \log N)$ result tells us something profound about the physical universe: **Scaling a classic $O(N^3)$ algorithm on a centralized processor sitting in a flat 2D memory field incurs a logarithmic thermodynamic tax.** The $\log N$ is the literal mathematical artifact of integrating wire-lengths across a 2-dimensional plane.

### Where else does this show up? (The Hong-Kung Bound)

In 1981, Hong and Kung proved the absolute lowest possible I/O bounds for matrix multiplication. They proved that for any hardware cache of size $M$, standard MatMul must suffer exactly $Q(M) = \Omega(N^3 / \sqrt{M})$ cache misses.

If you integrate this miss rate over an infinite continuous 2D plane (where the marginal wire cost is $\approx \frac{1}{\sqrt{M}}$):

$$ \int_1^{N^2} \left( \frac{1}{\sqrt{M}} \text{ cost} \right) \times \left( \frac{N^3}{\sqrt{M}} \text{ misses} \right) dM \ = \ N^3 \int_1^{N^2} \frac{1}{M} dM $$

The integral of $1/M$ is $\ln(M)$. Evaluating it yields $N^3 \ln(N^2) = \mathbf{\Theta(N^3 \log N)}$.

### How Hardware Engineers Beat the 2D Bound

To escape this $\log N$ communication bottleneck, modern engineers must fundamentally alter the dimensionality of the algorithm or the hardware:

**1. Dimensional Escape (3D Stacked Silicon)**

If we build 3D stacked memory (like modern HBM), the volume grows as $r^3$, so the physical distance to the $x$-th byte is only $x^{1/3}$. The penalty drops from $\sqrt{K^2}$ to $(K^2)^{1/3} = K^{0.66}$.

The recurrence becomes $T(K) = 8T(K/2) + \Theta(K^{2.66})$. Because $8 > 2^{2.66}$, the Master Theorem dictates that the leaf-level computations dominate the tree, and the total data movement cost drops beautifully back to a pure $\mathbf{\mathcal{O}(N^3)}$.

**2. Algorithmic Escape (Strassen's Algorithm)**

Strassen's algorithm only makes 7 recursive calls instead of 8. The recurrence becomes $T(K) = 7T(K/2) + \Theta(K^3)$. Because $7 < 2^3$, the Master Theorem dictates that the root of the tree dominates the cost. The logarithmic factor is completely annihilated, and the 2D communication energy drops to purely $\mathbf{\mathcal{O}(N^3)}$.

**3. Architectural Escape (Systolic Arrays)**

How do AI accelerators like Google TPUs solve this? They abandon the "centralized CPU" model. By distributing $N^2$ tiny processors evenly across the 2D memory grid (a Systolic Array), compute is pushed outward to where the data lives. Data only ever needs to travel to its immediate neighbor (distance = 1). This drops the communication cost down to $\mathbf{\mathcal{O}(N^3)}$, completely bypassing the 2D routing geometry.
