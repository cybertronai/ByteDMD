Because of a beautiful mathematical property of the geometric cache you just described, using a two-pass Bélády system will not just give you a "loose lower bound"—it allows you to cheaply compute the **exact absolute mathematical minimum energy (OPT)** required by a perfect omniscient compiler.

You don't need complex linear programming or Sleator-Tarjan adversary math for this. Here is the mathematical theory behind why it works perfectly for your model, followed by the $O(N \log W)$ algorithm to code it.

### ---

**1\. The Mathematical Magic of your Geometric Cache**

Notice what happens when we accumulate the capacities of your cache tiers:

* **Cost 1:** 1 unit $\rightarrow$ total capacity \= **1**  
* **Cost 2:** 3 units $\rightarrow$ total capacity \= 1 \+ 3 \= **4**  
* **Cost 3:** 5 units $\rightarrow$ total capacity \= 4 \+ 5 \= **9**  
* **Cost 4:** 7 units $\rightarrow$ total capacity \= 9 \+ 7 \= **16**

Because the total physical capacity up to cost $c$ is exactly $c^2$, the minimum routing cost to fetch a byte at depth $d$ is exactly **$\lceil\sqrt{d}\rceil$**.

Because this fetch cost strictly increases across nested, inclusive capacities ($1, 4, 9, 16\dots$), caching theory (Mattson et al., 1970\) proves that Bélády’s MIN algorithm is a **"Stack Algorithm."** This is a profound property: it means a single, optimal offline schedule minimizes cache misses for *every single capacity size simultaneously*.

Therefore, by calculating the true **Bélády Stack Distance** for your trace, you are mathematically minimizing every single tier of your geometric cache at the exact same time. No spatial layout strategy can beat it.

### ---

**2\. The Insight: "Max-Rank" equals OPT Distance**

In a standard Live-LRU stack, a variable's distance is simply its depth just before it is accessed.

Bélády’s optimal stack works differently. By prioritizing data sorted by **Next-Use Time**, variables can shift *up and down* in the cache. A variable waiting a long time might get pushed deep by hot variables, but as its own access time approaches, those hot variables die and it naturally bubbles back up to the top.

Therefore, the true optimal cache depth of an access is not its rank when it is fetched (it will always be Rank 1 right before you need it\!). Its true OPT distance is the **maximum rank it reached** while sitting inactive in the cache.

### ---

**3\. The Cheap 2-Pass "Bélády Energy" Algorithm**

You can compute this directly without simulating a 2D grid.

#### **Pass 1: The Oracle (Right-to-Left)**

Sweep through your trace of memory accesses in reverse.

* For every read at time t, record the exact integer index of its *next* read in an array: NextUse\[t\].  
* If the variable is never read again (it dies), set NextUse\[t\] \= infinity.

#### **Pass 2: The Geometric MIN Stack (Left-to-Right)**

Maintain a dynamic list of currently "live" variables in the cache.

**Crucial Rule:** This list must *always* be kept strictly sorted ascending by their NextUse time. You will also maintain a dictionary, max\_rank, to track the deepest position each variable gets pushed to.

For each memory access to variable $V$ at time $t$:

1. **Charge the Energy:**  
   * If $V$ is a cold miss (first time seen), charge your static argument-fetch cost.  
   * If $V$ is already live (a geometric hit), its optimal geometric depth is $d_{OPT} = \text{max\-rank}\[V\]$.  
   * Add **$\lceil\sqrt{d_{OPT}}\rceil$** to your total lower-bound energy.  
2. **Remove $V$:**  
   * Remove $V$ from the active live list.  
3. **Garbage Collect & Re-insert:**  
   * Look at NextUse\[t\]. If it is infinity, **do nothing**. $V$ is dead and vanishes. This perfectly models ideal garbage collection—it instantly shrinks the stack for all remaining variables.  
   * If it is not infinity, re-insert $V$ into the active list using NextUse\[t\] as its sorting key.  
   * Let its new 1-based index in the list be $r$. Initialize its tracking: max\_rank\[V\] \= r.  
4. **Update Pushed-Down Variables:**  
   * Because $V$ was just inserted at Rank $r$, every variable currently in the list at a rank $\ge r$ just got pushed one step further from the ALU.  
   * For every variable $Y$ sitting at rank $\ge r$, update its historical maximum:  
     max\_rank\[Y\] \= max(max\_rank\[Y\], current\_rank\[Y\])

### **Why this is brilliant for your workflow**

When you finish Pass 2, your total accumulated cost is exactly the offline optimal lower bound ($E_{OPT}$).

Because it knows the future, this algorithm automatically bypasses LRU thrashing artifacts (like cyclic array scans) and flawlessly handles garbage collection.

**Computational Speed:** If you implement Pass 2 with a standard array, it runs in $O(N \times W)$ time (where $W$ is your maximum active footprint). If you want to process traces with millions of operations in seconds, you can optimize Pass 2 to **$O(N \log W)$** by using a Fenwick Tree or simply Python's sortedcontainers.SortedList to track ranks. This provides an ironclad, incredibly cheap mathematical baseline for Bill Dally's PECM model.