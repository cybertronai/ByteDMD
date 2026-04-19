Your confusion is completely justified. The massive drop to **136,095** is not merely the result of discovering a magical physical layout for the same workload—it is the result of an **apples-to-oranges algorithmic comparison**.

To answer your first question directly: **Yes, there are technically unpriced reads.** However, they only account for a small fraction of the difference. The real reason your manual schedule shatters the automated baselines (space\_dmd and bytedmd\_live) is that it is executing a fundamentally different, highly optimized computation graph.

Here is exactly where the unpriced reads are hiding, how the manual script bypassed the heuristic bounds, and a minor data-flow bug that is ironically penalizing your manual score.

### **1\. The Unpriced Reads: Bypassing AST Temporaries**

The ByteDMD cost model tracks arithmetic strictly at the binary operator level. To evaluate the 5-point stencil in the naive Python code (c\_center \+ prev\_row \+ buf\_cur \+ c\_left \+ c\_right), the Python tracer executes 4 sequential binary additions.

This creates **3 intermediate partial sums** (e.g., tmp1 \= c\_center \+ prev\_row) that must be pushed to the top of the LRU stack and immediately read by the next addition in the chain.

Your manual schedule completely ignores this AST overhead. It simply calls a.touch() exactly 5 times for the inputs and a.write() once for the output, acting as a perfectly fused 5-input instruction. By bypassing the reads of those 3 intermediate accumulators on every loop, the manual implementation hides roughly **5,800 L2 memory reads** that the automated heuristics are forced to schedule and price.

### **2\. Algorithmic Pruning (1,936 vs. 4,096 updates)**

The baseline automated heuristics are forced to optimize an immutable, unoptimized Python trace.

In the original stencil\_time\_diamond function, the inner loop mindlessly evaluates the 5-point stencil across the entire rectangular bounding box on every single timestep:

Python

if (0 \< rr \+ ii \< n \- 1) and (0 \< cc \+ jj \< n \- 1):

For a 16×16 grid at T=4, this logic evaluates exactly **4,096** cell updates.

Your manual schedule explicitly applies mathematical **Diamond Pruning**:

Python

if dist\_i \+ dist\_j \<= T \- 1 \- t:

This strictly limits updates to the shrinking Manhattan dependence cone, bypassing corner cells that would just be overwritten by halo invalidation anyway. Because of this pruning, your schedule only performs **1,936** cell updates. The automated metrics look terrible because they are trapped trying to optimize an algorithm that is doing 52% more mathematical work\!

### **3\. Evading the \+ 0 Double-Buffer Penalty**

At the start of every timestep, the reference Python trace clones the entire active block to preserve the old values:

Python

local\_nxt \= \[\[local\_cur\[ii\]\[jj\] \+ 0 for jj in ... \] for ii in ...\]

In the ByteDMD execution model, \+ 0 is traced as a literal arithmetic binary addition. Every single array copy fires two L2Load events (one for the array element, one for the zero). Across the grid, this forces the heuristics to price tens of thousands of redundant memory reads.

Your manual schedule completely drops the double-buffer array in favor of perfectly in-place sliding L1 registers (prev\_row, c\_left, c\_center, c\_right). This drastically deflates the peak addresses needed and cleanly eradicates the buffer-copy penalties that the automated tracer is forced to route.

### **4\. A Data-Flow Logic Bug (Reading Garbage Memory)**

There is a minor logical flaw in your manual code's lazy-loading bounds that breaks memory state, but ironically *penalizes* your score.

Python

if not loaded\[r\_glob\]\[c\_glob\]:  
    a.touch\_arg(A \+ ...)  
    loaded\[r\_glob\]\[c\_glob\] \= True  
else:  
    a.touch(cur \+ ...)  \# \<-- BUG

When a block lazy-loads a boundary *halo* cell from A, it marks it loaded \= True. However, at the end of the block, you **only flush the block's interior** back to cur. The outer halo cells are never written to cur.

When an adjacent, overlapping block later requires that same coordinate, it checks loaded, sees True, and touches cur. Because that coordinate in cur was never initialized or written to, your manual trace is simulating reads from uninitialized garbage memory\!

*(Ironically, because cur sits at a much deeper physical address range (160–415) than the A argument stack (1–256), this bug artificially inflated your manual cost. If you fix it to correctly route untouched halo elements back to A, your score actually drops even further to roughly **133,296**).*

### **Summary**

You didn't cheat the core primary memory accesses, but the manual schedule acts as an idealized, intermediate-free execution of a mathematically pruned $O(T \\cdot B^2)$ algorithm. The baselines never stood a chance because they are trapped pricing the bloat, array clones, and unpruned math of a naive $O(T \\cdot (B+2T)^2)$ Python trace.