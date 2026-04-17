# RMM, simplified — the Pyramid / Forklift strategy and the $\Theta(N^3 \log N)$ bound

Below is a radically simplified, clean version of the Explicit Recursive
Matrix Multiplication (RMM) code, followed by a concise walkthrough, the
Master Theorem proof, and a deep dive into where this exact
$\Theta(N^3 \log N)$ bound lives in classical computer science
literature.

---

## 1. The simplified code

By replacing the massive blocks of manually unrolled quadrant math with
a clean triple loop, the logic now perfectly mirrors standard
block-matrix multiplication
($C_{i,j} \mathrel{+}= A_{i,k} \times B_{k,j}$).

```python
def dma_copy(alloc, src, src_stride, sr, sc, dst, dst_stride, dr, dc, size):
    """Software DMA: explicitly copies a 2D block from src to dst."""
    for i in range(size):
        for j in range(size):
            val = alloc.read(src + (sr + i) * src_stride + (sc + j))
            alloc.write(dst + (dr + i) * dst_stride + (dc + j), val)


def rmm_explicit(alloc, ptrs, size, pA, pB, pC, stride):
    # BASE CASE: math executes natively at the absolute fastest physical addresses
    if size == 1:
        a, b, c = alloc.read(pA), alloc.read(pB), alloc.read(pC)
        alloc.write(pC, c + a * b)
        return

    H = size // 2
    sA, sB, sC = ptrs[H]['A'], ptrs[H]['B'], ptrs[H]['C']  # 's' = fast scratchpad

    # Triple loop mapping directly onto the four quadrants of the matrices
    for i in (0, H):
        for j in (0, H):
            # 1. DMA-load the C quadrant from the slow parent down to the fast child
            dma_copy(alloc, pC, stride, i, j, sC, H, 0, 0, H)

            for k in (0, H):
                # 2. DMA-load A and B quadrants down to the fast child
                dma_copy(alloc, pA, stride, i, k, sA, H, 0, 0, H)
                dma_copy(alloc, pB, stride, k, j, sB, H, 0, 0, H)

                # 3. Recurse entirely inside the fast child memory
                rmm_explicit(alloc, ptrs, H, sA, sB, sC, H)

            # 4. DMA-store the finished C quadrant back up to the slow parent
            dma_copy(alloc, sC, H, 0, 0, pC, stride, i, j, H)
```

---

## 2. Walkthrough: the Pyramid and the Forklift

If you run standard recursive matrix multiplication on a stationary
$N \times N$ matrix, the tiny $1 \times 1$ base cases at the bottom of
the tree are forced to fetch numbers from physical addresses as deep as
$N^2$. That triggers massive continuous-distance penalties, degrading
performance to a catastrophic $\mathcal{O}(N^4)$.

The code above fixes this with a **Pyramid-and-Forklift** strategy:

- **The Pyramid** (`allocate_hierarchy`). It permanently reserves the
  absolute lowest, fastest physical addresses *exclusively* for the
  $1 \times 1$ base cases. Behind those sit the $2 \times 2$ scratchpads,
  then $4 \times 4$, and so on. The massive input matrices sit at the
  very end of memory.
- **The Forklift** (`dma_copy`). Before the CPU works on a sub-problem, a
  software DMA controller physically hauls the required $A$, $B$, and
  $C$ sub-matrices down from the slower parent memory into the faster
  child memory.
- **Speed of light.** By the time the recursion bottoms out, the
  required data has been actively relayed all the way down to addresses
  1, 2, and 3. The actual computational math happens at speed-of-light
  cost.

---

## 3. The Master Theorem: why is it exactly $\Theta(N^3 \log N)$?

To understand the bound, evaluate the "overhead" of the DMA forklift at
a recursion level of size $K$.

- **Volume.** The forklift copies sub-matrices of size
  $\tfrac{K}{2} \times \tfrac{K}{2}$, touching $\Theta(K^2)$ elements.
- **Distance penalty.** Because of the pyramid layout, the parent data
  lives at an address depth proportional to $K^2$. Under the physical
  model that penalizes reads by $\sqrt{\text{address}}$, the distance
  penalty is $\sqrt{K^2} = \Theta(K)$.
- **Total transfer cost.**
  $\text{Volume} \times \text{Distance penalty} = K^2 \times K = \Theta(K^3)$.

This gives the classic divide-and-conquer recurrence

$$
T(K) = 8\, T(K/2) + \Theta(K^3).
$$

**The mathematical resonance.** The algorithm splits into 8 branches.
The volume of the matrix shrinks by a factor of $(2)^3 = 8$. Because the
branching factor **perfectly matches** the shrinkage, the algorithm sits
in perfect resonance — every single layer of the recursion tree does the
exact same amount of data-movement work, $\Theta(N^3)$.

Because a tree halved at each step has exactly $\log_2 N$ layers,
multiplying work-per-layer by depth yields the inescapable grand total

$$
T(N) = \Theta(N^3 \log N).
$$
