To achieve the theoretically optimal $\mathcal{O}(N^3 \log N)$ cost under the physical distance metric where accessing the $k$-th byte costs $\lceil\sqrt{k}\rceil$, we must explicitly dictate the spatial layout of our memory during execution.

### **The Strategy: Inverted Stack Arenas (Explicit Tombstoning)**

If we perform recursive matrix multiplication by computing base cases directly on the original input matrices, our $1 \times 1$ multiplications will read variables stored at indices up to $\mathcal{O}(N^2)$. The cost per access would be $\mathcal{O}(N)$, and multiplying that by the $N^3$ base-case operations yields an extremely inefficient $\mathcal{O}(N^4)$ cost.

To overcome this, we explicitly simulate the **Tombstone Strategy's** aggressive compaction by manipulating Working Memory as a Depth-Indexed Arena Allocator (an "Inverted Stack"):

1. We divide the total working memory into tightly bounded "arenas" indexed by recursion depth.  
2. The deepest recursion level (the $1 \times 1$ base cases) is allocated the absolute lowest memory addresses (e.g., indices 1, 2, 3). Thus, the $\mathcal{O}(N^3)$ base case operations only cost $\mathcal{O}(1)$ to access\!  
3. Before a recursive step, the parent matrices explicitly copy their required active $M/2 \times M/2$ quadrants into the tightly localized arena belonging to the child's depth.  
4. Once the child branch finishes, the memory it occupied is logically "tombstoned" (freed). When the next sibling recursion executes, it physically overwrites and **reuses the exact same tight addresses**.  
5. Copying data at recursion level $M$ bounded around a maximum arena size of $\mathcal{O}(M^2)$ incurs an access distance cost of $\mathcal{O}(M)$. Moving $M^2$ elements yields a level cost of $\mathcal{O}(M^3)$. Using the Master Theorem: $T(N) = 8T(N/2) + \mathcal{O}(N^3)$, which resolves elegantly to the optimal footprint of **$\mathcal{O}(N^3 \log N)$**.

### **Python Implementation**

Below is the self-contained script modeling this architecture. It correctly takes any dimension size, traces the logical addresses mapping the required arrays into external/working memory, and outputs the operational traces.

Python

"""  
Optimal O(N^3 log N) Recursive Matrix Multiplication Tracker.  
"""

import math

def generate\_traces(n: int):  
    """  
    Takes an integer n (multiple of 4\) and simulates recursive matrix multiplication.  
    Returns a tuple of traces containing 1-indexed integers corresponding to memory accesses:  
    (external\_reads, working\_reads, working\_writes)  
    """  
    ext\_reads \= \[\]  
    wm\_reads \= \[\]  
    wm\_writes \= \[\]

    \# 1\. Pre-compute the maximum dimensions at each recursion depth to allocate perfectly bounded arenas  
    def get\_depths\_and\_sizes(N):  
        max\_dims \= {}  
          
        def dry\_run(R, K, C, depth):  
            if depth not in max\_dims:  
                max\_dims\[depth\] \= \[0, 0, 0\]  
            max\_dims\[depth\]\[0\] \= max(max\_dims\[depth\]\[0\], R)  
            max\_dims\[depth\]\[1\] \= max(max\_dims\[depth\]\[1\], K)  
            max\_dims\[depth\]\[2\] \= max(max\_dims\[depth\]\[2\], C)  
              
            if R \== 1 and K \== 1 and C \== 1:  
                return  
                  
            r1 \= (R \+ 1) // 2; r2 \= R \- r1  
            k1 \= (K \+ 1) // 2; k2 \= K \- k1  
            c1 \= (C \+ 1) // 2; c2 \= C \- c1  
              
            for r in \[r1, r2\]:  
                for k in \[k1, k2\]:  
                    for c in \[c1, c2\]:  
                        if r \> 0 and k \> 0 and c \> 0:  
                            dry\_run(r, k, c, depth \+ 1)  
                              
        dry\_run(N, N, N, 0)  
        return max\_dims

    max\_dims \= get\_depths\_and\_sizes(n)  
    D \= max(max\_dims.keys())  
      
    \# 2\. Build the Inverted Stack Layout for Working Memory  
    \# The deepest base cases (depth D) sequentially receive the lowest addresses (starting exactly at 1\)  
    arena\_size \= {}  
    for d in range(1, D \+ 1):  
        mR, mK, mC \= max\_dims\[d\]  
        arena\_size\[d\] \= mR \* mK \+ mK \* mC \+ mR \* mC

    arena\_start \= {}  
    current\_addr \= 1  
    for d in range(D, 0, \-1):  
        arena\_start\[d\] \= current\_addr  
        current\_addr \+= arena\_size\[d\]  
          
    \# The top-level target C matrix safely occupies the memory sequentially after all recursion buffers  
    arena\_start\[0\] \= current\_addr

    def copy\_matrix(src\_sp, src\_base, src\_stride,   
                    dst\_sp, dst\_base, dst\_stride,   
                    R, C):  
        """Helper to trace block copying dynamically between External/Working space"""  
        for i in range(R):  
            for j in range(C):  
                src\_addr \= src\_base \+ i \* src\_stride \+ j  
                dst\_addr \= dst\_base \+ i \* dst\_stride \+ j  
                  
                if src\_sp \== 'EXT':  
                    ext\_reads.append(src\_addr)  
                elif src\_sp \== 'WM':  
                    wm\_reads.append(src\_addr)  
                      
                if dst\_sp \== 'WM':  
                    wm\_writes.append(dst\_addr)

    \# 3\. Core Recursive MatMul emulating Aggressive Compaction  
    def matmul(R, K, C\_dim, d, A\_info, B\_info, C\_info):  
        \# Base Case: executed directly at the lowest bounds (indices 1, 2, 3...)  
        if d \== D:  
            A\_sp, A\_base, \_ \= A\_info  
            B\_sp, B\_base, \_ \= B\_info  
            C\_sp, C\_base, \_ \= C\_info  
              
            \# Simulated Action: C \+= A \* B  
            if A\_sp \== 'EXT': ext\_reads.append(A\_base)  
            else: wm\_reads.append(A\_base)  
              
            if B\_sp \== 'EXT': ext\_reads.append(B\_base)  
            else: wm\_reads.append(B\_base)  
              
            if C\_sp \== 'WM': wm\_reads.append(C\_base)  
            if C\_sp \== 'WM': wm\_writes.append(C\_base)  
            return

        r1 \= (R \+ 1) // 2; r2 \= R \- r1  
        k1 \= (K \+ 1) // 2; k2 \= K \- k1  
        c1 \= (C\_dim \+ 1) // 2; c2 \= C\_dim \- c1  
          
        r\_splits \= \[(0, r1), (r1, r2)\]  
        k\_splits \= \[(0, k1), (k1, k2)\]  
        c\_splits \= \[(0, c1), (c1, c2)\]  
          
        \# Determine precisely where the child depth (d+1) arena bounds start for A', B', and C'  
        A\_prime\_base \= arena\_start\[d+1\]  
        B\_prime\_base \= arena\_start\[d+1\] \+ max\_dims\[d+1\]\[0\] \* max\_dims\[d+1\]\[1\]  
        C\_prime\_base \= B\_prime\_base \+ max\_dims\[d+1\]\[1\] \* max\_dims\[d+1\]\[2\]  
          
        \# 8-way Quadrant Sub-Multiplications  
        for r\_idx in range(2):  
            r\_off, sub\_r \= r\_splits\[r\_idx\]  
            if sub\_r \== 0: continue  
              
            for c\_idx in range(2):  
                c\_off, sub\_c \= c\_splits\[c\_idx\]  
                if sub\_c \== 0: continue  
                  
                \# Retrieve the parent's C block and pull it sequentially "downward" into C'  
                C\_sp, C\_base, C\_stride \= C\_info  
                C\_sub\_base \= C\_base \+ r\_off \* C\_stride \+ c\_off  
                  
                copy\_matrix(C\_sp, C\_sub\_base, C\_stride,   
                            'WM', C\_prime\_base, sub\_c,   
                            sub\_r, sub\_c)  
                  
                for k\_idx in range(2):  
                    k\_off, sub\_k \= k\_splits\[k\_idx\]  
                    if sub\_k \== 0: continue  
                      
                    \# Copy quadrant inputs downward (Tombstone explicitly compacts distances closer to CPU)  
                    A\_sp, A\_base, A\_stride \= A\_info  
                    A\_sub\_base \= A\_base \+ r\_off \* A\_stride \+ k\_off  
                    copy\_matrix(A\_sp, A\_sub\_base, A\_stride,   
                                'WM', A\_prime\_base, sub\_k,   
                                sub\_r, sub\_k)  
                      
                    B\_sp, B\_base, B\_stride \= B\_info  
                    B\_sub\_base \= B\_base \+ k\_off \* B\_stride \+ c\_off  
                    copy\_matrix(B\_sp, B\_sub\_base, B\_stride,   
                                'WM', B\_prime\_base, sub\_c,   
                                sub\_k, sub\_c)  
                      
                    \# Target inner buffers locked entirely to the cheap local footprint of (d+1)  
                    A\_prime\_info \= ('WM', A\_prime\_base, sub\_k)  
                    B\_prime\_info \= ('WM', B\_prime\_base, sub\_c)  
                    C\_prime\_info \= ('WM', C\_prime\_base, sub\_c)  
                      
                    matmul(sub\_r, sub\_k, sub\_c, d+1, A\_prime\_info, B\_prime\_info, C\_prime\_info)  
                      
                \# Push the resolved sub-solution back upwards into the parent space  
                copy\_matrix('WM', C\_prime\_base, sub\_c,   
                            C\_sp, C\_sub\_base, C\_stride,   
                            sub\_r, sub\_c)

    \# 4\. Entry Point Setup  
    \# Inputs logically positioned inside static External Memory matrices (1 .. n^2, n^2 \+ 1 .. 2n^2)  
    A\_ext\_info \= ('EXT', 1, n)  
    B\_ext\_info \= ('EXT', 1 \+ n\*\*2, n)  
    C\_wm\_info \= ('WM', arena\_start\[0\], n)  
      
    \# Pre-Initialize the top-level WM tracking array to zeros  
    for i in range(n \* n):  
        wm\_writes.append(arena\_start\[0\] \+ i)  
          
    matmul(n, n, n, 0, A\_ext\_info, B\_ext\_info, C\_wm\_info)  
      
    return ext\_reads, wm\_reads, wm\_writes

\# \=== Execution / Evaluation Block \===  
if \_\_name\_\_ \== "\_\_main\_\_":  
    n \= 16   
    ext\_r, wm\_r, wm\_w \= generate\_traces(n)  
      
    \# Apply standard cost penalty equation mathematically to test theoretical bounds natively   
    def calc\_cost(trace):  
        return sum(math.ceil(math.sqrt(k)) for k in trace)  
          
    total\_cost \= calc\_cost(ext\_r) \+ calc\_cost(wm\_r) \+ calc\_cost(wm\_w)  
      
    print(f"Optimal Traces constructed mapped to N={n}")  
    print(f"External Memory reads: {len(ext\_r):\>8} operations")  
    print(f"Working Memory reads:  {len(wm\_r):\>8} operations")  
    print(f"Working Memory writes: {len(wm\_w):\>8} operations")  
    print(f"Total Model Evaluated Cost: {total\_cost}")  
