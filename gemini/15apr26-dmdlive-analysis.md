Based on the extensive analysis in your merged documents and the continuous cache model from your blogpost outline, your intuition is spot on.

The two metrics—**Original DMD** and **DMDlive**—represent two extremes of memory management. The "actual cost" sits mathematically between them, utilizing what the theoretical documents call the **"Tombstone / High-Water Mark"** model.

Here is the complete analysis of how the continuous 2D Manhattan cache behaves, what the two DMD metrics actually simulate, why standard eviction (like Belady's) fails here, and the exact complexity of Recursive Matrix Multiplication (RMM) under all three regimes.

### **1\. The Blogpost's "Continuous Cache" is Exactly ByteDMD**

Your blogpost describes a continuous cache where the concentric levels have capacities of **1, 3, 5, 7, 9...** and access costs of **1, 2, 3, 4, 5...**.

If you sum the capacities of the levels up to an access cost $c$, the total cache capacity is exactly the sum of odd numbers: $\sum_{i=1}^c (2i-1) = c^2$.

Because a cache of radius $c$ holds $c^2$ elements, finding the access cost to the $d$-th most recently used element requires solving $d \le c^2$, which yields $c = \lceil\sqrt{d}\rceil$.

Therefore, your physical continuous cache description is the exact geometric proof for the $\lceil\sqrt{d}\rceil$ pricing function used in both DMD metrics.

### ---

**2\. Original DMD: "The Infinite Graveyard"**

* **What it models:** A system with zero memory reclamation (a pure memory leak). Every newly computed temporary variable is placed in the center of the cache (distance 1), permanently pushing all older variables one slot further outward.  
* **Complexity for Recursive MatMul:** $\mathbf{\Theta(N^{3.5})}$  
* **The Physical Reality:** In RMM, computing an $S \times S$ sub-block generates $\mathcal{O}(S^3)$ dead temporary variables. To fetch the input parent matrices for the *next* sibling sub-block, the processor must route wires through this ocean of $S^3$ dead variables. The access distance becomes $\sqrt{S^3} = S^{1.5}$. Multiplying this by the $S^2$ reads gives a step cost of $S^{3.5}$. Integrated across the recursion tree, this catastrophic memory leak breaks the algorithm, degrading its data movement complexity to $\Theta(N^{3.5})$.

### ---

**3\. DMDlive: "The Teleporting Cache"**

* **What it models:** This is the "unknown eviction method" you identified. When a variable's last use completes, it vaporizes. But more importantly, **all elements sitting in slower caches instantly and magically slide inward** to fill the newly created holes at zero cost.  
* **Complexity for Recursive MatMul:** $\mathbf{\Theta(N^3 \log N)}$  
* **The Physical Reality:** As you noted, *"this is equivalent to everything moving up the stack to fill the holes, which is not quite true."* In real silicon, moving data inward requires unsolicited reads and writes, consuming energy. By allowing variables to teleport inward for free, DMDlive mathematically "cheats" and undercounts the true physical constant factor of the algorithm. However, because the stack instantly shrinks, the cache radius is strictly clamped to the *instantaneous* live working set ($\mathcal{O}(S^2)$). Fetching matrices costs $\sqrt{\mathcal{O}(S^2)} \propto S$. The recurrence $T(N) = 8T(N/2) + \Theta(S^3)$ cleanly resolves to the optimal $\Theta(N^3 \log N)$.

### ---

**4\. The Actual Cost: "Tombstones & High-Water Marks"**

How does a real physical continuous cache resolve this without free "sliding" and without leaking memory?

* **The Mechanic:** When a variable dies (goes out of scope), it leaves a "hole" (a tombstone) at its exact physical coordinate. Older variables **do not slide inward**; they remain stationary in the slower outer caches. However, when the processor creates a *new* temporary variable, the memory manager intelligently recycles the closest available hole rather than expanding the outer frontier of the cache.  
* **Complexity for Recursive MatMul:** $\mathbf{\Theta(N^3 \log N)}$  
* **Why it works:** Because RMM is a depth-first recursive algorithm, its memory usage behaves exactly like a LIFO call stack. RMM uses $\mathcal{O}(S^2)$ memory for its inputs/outputs, and its sub-calls reach a peak active allocation of $\approx 1.25S^2$ before freeing memory. Therefore, the High-Water Mark (Peak Footprint) of the grid is permanently clamped to roughly $4.25 S^2$ slots.

Even though variables don't slide inward, the parent matrices are never pushed deeper than this $\mathcal{O}(S^2)$ high-water mark. The access cost remains bounded at $\sqrt{\mathcal{O}(S^2)} \propto S$. This means the realistic hardware cost **preserves the exact optimal $\Theta(N^3 \log N)$ asymptotic complexity of DMDlive**, just with a slightly higher leading constant multiplier (e.g., $12.3$ instead of $7.3$) due to the fragmentation.

### ---

**5\. Why Belady's Algorithm is the Wrong Mental Model**

You mentioned that "Belady normally evicts from cache", which makes DMDlive's behavior confusing.

Belady’s MIN algorithm (evicting the item accessed furthest in the future) is perfectly optimal for *flat*, two-level caches (e.g., L1 vs DRAM) with uniform miss penalties.

However, in a continuous spatial cache (where cost is $\lceil\sqrt{k}\rceil$), Belady suffers from the **"Squatter Anomaly"**.

If you use Belady, it will prioritize the long-lived parent matrices $A$ and $B$ for the cheap $k=1, 2$ center slots (because they will eventually be used again, beating out temporaries that die). The parent matrices will "squat" there dormantly for thousands of cycles, forcing millions of high-frequency, short-lived inner-loop temporaries to thrash in the expensive $k=100$ outer rings\!

An optimal continuous cache uses a strategy like **Interval Coloring / Fractional Space-Time Knapsack**. It intentionally banishes the dormant parent matrices to the outer rings, paying the high fetch cost only occasionally. This leaves the premium center slots entirely open for the massive volume of inner-loop temporaries to rapidly allocate, die, and recycle.

The "Tombstone" model naturally mimics this physical reality:

1. Temporaries are born, heavily utilized in the center, and then die, leaving holes.  
2. The next batch of temporaries reuses those exact same center holes.  
3. The long-lived parent matrices wait patiently, stationary in the outer rings.

### **Summary**

1. **Original DMD (Upper Bound / No GC):** $\Theta(N^{3.5})$. Fails because $\mathcal{O}(N^3)$ dead variables physically push the matrix away.  
2. **DMDlive (Lower Bound / Magic Sliding):** $\Theta(N^3 \log N)$. Identical scaling to reality, but artificially trims the constant factor by ignoring the stationary nature of outer-cache data.  
3. **The Actual Continuous Cache (Tombstones):** $\Theta(N^3 \log N)$. Succeeds because recycled holes clamp the cache radius to the peak $\mathcal{O}(N^2)$ working set, preserving the asymptotic limit without violating the laws of physics.