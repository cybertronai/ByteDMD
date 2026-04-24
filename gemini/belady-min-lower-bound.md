To prove mathematically that no memory allocation strategy—even an omniscient compiler with perfect future knowledge and the ability to instantly reorganize data for free between reads—can achieve a total read energy lower than the one calculated by $d\_{OPT} \= \\text{max\\\_rank}\[V\]$, we must fuse **continuous 2D geometry** with **discrete optimal caching theory**.

Because your model assumes that moving data (writes) is free, an optimal compiler can shuffle data arbitrarily between reads. To prove that max\_rank is the absolute mathematical floor, we must prove two things:

1. Minimizing 2D geometric routing cost is mathematically identical to minimizing cache misses across all possible cache capacities simultaneously.  
2. The value $\\text{max\\\_rank}\[V\]$ flawlessly calculates the absolute theoretical minimum of those misses using the **Pigeonhole Principle**.

Here is the rigorous, step-by-step proof.

### ---

**Part 1: Decomposing Geometry into "Nested Caches"**

Let $S$ be *any* arbitrary spatial allocation strategy. When the algorithm requests a variable at time $t$, $S$ fetches it from some physical distance (rank) $d\_t^S$.

Your model penalizes distance with a non-decreasing fetch cost function $C(d)$ (e.g., $C(d) \= \\lceil\\sqrt{d}\\rceil$). A fundamental mathematical property of any non-decreasing step function is that it can be rewritten as a base cost plus a sum of marginal penalties.

Let $\\Delta\_c \= C(c+1) \- C(c)$ be the extra penalty of fetching from distance $c+1$ instead of $c$. Because cost never decreases with distance, $\\Delta\_c \\ge 0$ for all capacities $c$.

We can rewrite the exact cost of a single read as:

$$C(d\_t^S) \= C(1) \+ \\sum\_{c=1}^\\infty \\Delta\_c \\cdot \\mathbb{I}(d\_t^S \> c)$$  
*(Where $\\mathbb{I}$ is an indicator function: 1 if the distance is strictly greater than $c$, and 0 otherwise).*

Now, let's write the total energy equation for Strategy $S$ over all $N$ reads in the algorithm, and swap the order of the summations (pulling the capacity $c$ to the outside):

$$E(S) \= \\sum\_{t=1}^N \\left( C(1) \+ \\sum\_{c=1}^\\infty \\Delta\_c \\cdot \\mathbb{I}(d\_t^S \> c) \\right)$$

$$E(S) \= N \\cdot C(1) \+ \\sum\_{c=1}^\\infty \\Delta\_c \\left\[ \\sum\_{t=1}^N \\mathbb{I}(d\_t^S \> c) \\right\]$$  
**The Cache Equivalency Insight:** Look closely at the inner bracket: $\\sum \\mathbb{I}(d\_t^S \> c)$.

What does this mathematically represent? It counts the exact number of times Strategy $S$ fetched a variable from a distance strictly greater than $c$. If we draw a physical boundary around the ALU that holds exactly $c$ variables, this sum is **the exact number of Cache Misses** Strategy $S$ would suffer if it were restricted to a hardware cache of size $c$.

Let’s define this as $M\_S(c)$. The master geometric energy equation collapses into:

$$E(S) \= N \\cdot C(1) \+ \\sum\_{c=1}^\\infty \\Delta\_c \\cdot M\_S(c)$$  
*Conclusion of Part 1:* Total spatial routing energy is perfectly isomorphic to a weighted sum of cache misses across every integer capacity from $1$ to $\\infty$.

### ---

**Part 2: Bélády's Absolute Lower Bound**

Now we invoke the most famous theorem in caching: **Bélády’s MIN algorithm (1966)**.

Bélády proved that for any *single, isolated* cache capacity $c$, the absolute optimal way to minimize misses is to always evict the variable whose next use is furthest in the future. Therefore, no strategy $S$ can possibly generate fewer misses than Bélády's MIN for any capacity $c$:

$$M\_S(c) \\ge M\_{MIN}(c)$$  
Furthermore, in 1970, Mattson et al. proved the **Inclusion Property**. Because MIN's eviction rule (furthest next-use time) does not depend on the physical size of the cache, it generates a single schedule that achieves the minimum possible misses for *every single capacity $c$ simultaneously*.

Because $\\Delta\_c \\ge 0$ and $M\_S(c) \\ge M\_{MIN}(c)$ for every term in our energy equation, we reach an unassailable mathematical floor:

$$E(S) \\ge N \\cdot C(1) \+ \\sum\_{c=1}^\\infty \\Delta\_c \\cdot M\_{MIN}(c)$$

### ---

**Part 3: The Pigeonhole Eviction Theorem**

We must now prove why tracking $d\_{OPT} \= \\text{max\\\_rank}\[V\]$ exactly equates to MIN's optimal performance.

Consider a variable $V$ that is accessed at $t\_{prev}$, goes dormant, and is accessed again at $t\_{next}$.

During its dormancy, your algorithm tracks $V$'s "rank" in a list of live variables sorted by next-use time. At any moment $\\tau$ during this interval:

$$\\text{Rank}(V, \\tau) \= 1 \+ (\\text{Number of live variables needed BEFORE } t\_{next})$$  
Let $R \= \\text{max\\\_rank}\[V\]$ be the absolute highest rank $V$ reached during its dormancy. This means at some "peak" moment $\\tau\_{peak}$, there were exactly $R-1$ *other* active variables that the algorithm needed before it needed $V$.

**Theorem:** *In Bélády's MIN, $V$ suffers a cache miss at $t\_{next}$ if and only if $\\text{max\\\_rank}\[V\] \> c$.*

**Proof of Miss (If $\\text{max\\\_rank}\[V\] \> c$):**

If $R \> c$, then at time $\\tau\_{peak}$, there are $\\ge c$ live variables needed strictly before $V$.

A cache of size $c$ only has $c$ physical slots. By the **Pigeonhole Principle**, the cache is physically incapable of holding all of those earlier-needed variables AND variable $V$.

Because MIN perfectly prioritizes variables needed sooner, it will fill the cache with the $c$ variables needed before $V$. $V$ is mathematically forced to be evicted. Because $V$ is not demanded again until $t\_{next}$, it will be missing from the cache when $t\_{next}$ arrives.

**Proof of Hit (If $\\text{max\\\_rank}\[V\] \\le c$):**

If $R \\le c$, then at *every* moment during $V$'s dormancy, there are strictly fewer than $c$ variables needed before $V$.

Even if the cache is completely full, $V$ will always be needed sooner than at least one other variable in the cache. Therefore, $V$ is *never* the variable with the furthest next-use. MIN will never select $V$ for eviction, guaranteeing a hit.

### ---

**Part 4: The Grand Synthesis**

We just proved that $\\text{max\\\_rank}\[V\] \> c$ is the exact binary condition for whether the optimal MIN algorithm suffers a cache miss on capacity $c$. Therefore, the total optimal misses $M\_{MIN}(c)$ is simply the sum of that condition over all $N$ reads:

$$M\_{MIN}(c) \= \\sum\_{t=1}^N \\mathbb{I}(\\text{max\\\_rank}\[V\_t\] \> c)$$  
Substitute this back into the absolute lower-bound energy equation from Part 2:

$$E(S) \\ge N \\cdot C(1) \+ \\sum\_{c=1}^\\infty \\Delta\_c \\left\[ \\sum\_{t=1}^N \\mathbb{I}(\\text{max\\\_rank}\[V\_t\] \> c) \\right\]$$  
Now, swap the summations back to their original orientation (pulling the reads $t$ to the outside):

$$E(S) \\ge \\sum\_{t=1}^N \\left( C(1) \+ \\sum\_{c=1}^\\infty \\Delta\_c \\cdot \\mathbb{I}(\\text{max\\\_rank}\[V\_t\] \> c) \\right)$$  
Look at the term inside the parentheses. We have perfectly reconstructed the indicator-step-function from Part 1, but this time it is evaluated exactly at $\\text{max\\\_rank}\[V\_t\]$\!

$$E(S) \\ge \\sum\_{t=1}^N C(\\text{max\\\_rank}\[V\_t\])$$

### **Q.E.D.**

Because the cost of 2D data movement mathematically decomposes into a sum of misses across infinite concentric boundaries, and because the Pigeonhole Principle proves that max\_rank exactly defines the minimum possible boundary size required to shield a variable from optimal eviction, **it is a physical impossibility for any dynamic spatial allocation strategy to run your algorithm for less energy.**