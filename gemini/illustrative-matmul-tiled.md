Here is a clear, step-by-step mathematical walkthrough and a self-contained Python script that completely strips away LRU caches to show why **manual memory placement (scratchpad management)** is mathematically optimal on a 2D spatial grid.

### **1\. The Physical Memory Model**

Imagine memory as a continuous 1D array of addresses ($1, 2, 3 \dots$) mapped onto a 2D physical silicon grid spiraling outward from the processor (ALU).

Because the area of a 2D shape grows quadratically with its radius ($A = \pi r^2$), the physical distance (wire length) to reach address $X$ scales as $\sqrt{X}$.

* **The Rule:** Reading from address addr costs $\lceil\sqrt{\text{addr}}\rceil$. (Writes are free).  
* **The Layout:** We enforce a two-tiered manual layout.  
  * **Below (Scratchpad):** The absolute lowest addresses ($1 \dots S$) are permanently pinned to our L1 tile buffers.  
  * **Above (Arguments):** The large matrices $A$ and $B$ are allocated *after* the scratchpad, pushing them out into deeper, more expensive physical addresses.

### ---

**2\. Step-by-Step Walkthrough ($16 \times 16$ Matrices, $T=4$)**

Let's compute the memory cost to multiply two $16 \times 16$ matrices. Both $A$ and $B$ contain 256 elements.

#### **Strategy A: Naive Matmul (Direct Fetching)**

In the naive approach, we don't allocate a scratchpad. The matrices start immediately at Address 1\.

* Matrix $A$ is at addresses 1 to 256 (Average read cost $\approx 11$)  
* Matrix $B$ is at addresses 257 to 512 (Average read cost $\approx 19$)

To compute a single element of $C$, the ALU does 16 multiply-accumulates (MACs). It reaches out to $A$ and $B$ for every single calculation.

* **Cost for 1 element:** $16 \text{ reads} \times (11 + 19) \approx \mathbf{480}$ cost.  
* **Total Cost for all 256 elements:** $256 \times 480 \approx \mathbf{122,880}$ distance cost.

#### **Strategy B: Manual Tiled Matmul**

We allocate three $4 \times 4$ local scratchpads first, pushing $A$ and $B$ further away.

* fast\_A (Addr 1 to 16\) $\rightarrow$ Avg cost: $\mathbf{3}$  
* fast\_B (Addr 17 to 32\) $\rightarrow$ Avg cost: $\mathbf{5}$  
* fast\_C (Addr 33 to 48\) $\rightarrow$ Avg cost: $\mathbf{6}$  
* *Notice: $A$ and $B$ now start at address 49, making them slightly more expensive to read from than before\!*

We compute the $16 \times 16$ output by breaking it into sixteen $4 \times 4$ blocks. Let's trace **one** block:

**Phase 1: DMA Block Fetch (Pay the heavy toll)**

We fetch a $4 \times 4$ block of $A$ and a $4 \times 4$ block of $B$ from deep memory and save them to the scratchpad.

* **Fetch Cost:** 16 elements of $A$ (cost $\sim 13$) \+ 16 elements of $B$ (cost $\sim 21$) $\approx \mathbf{544}$ cost.

**Phase 2: Compute locally in the Scratchpad (The Payout)**

Now we compute the 16 output elements of this block. Each element requires 4 MACs, but we read *exclusively* from our perfectly pinned addresses $1 \dots 48$.

* **Cost for 1 element:** $4 \text{ reads} \times (\text{fast\-A} + \text{fast\-B} + \text{fast\-C}) \rightarrow 4 \times (3 + 5 + 6) = \mathbf{56}$ cost.  
* **Cost for all 16 elements in the block:** $16 \times 56 = \mathbf{896}$ cost.

**Net Result per block:**

* **Naive Cost:** $16 \text{ elements} \times 480 \text{ cost} = \mathbf{7,680}$  
* **Tiled Cost:** $544 \text{ (Fetch)} + 896 \text{ (Compute)} = \mathbf{1,440}$

Even though we penalized $A$ and $B$ by pushing them into higher addresses, **the Manual Tiled approach mathematically crushes the Naive approach** because it converts $O(N^3)$ long-distance wire routing into $O(N^3)$ ultra-local wire routing.

### ---

**3\. Minimal Runnable Python Proof**

Here is the exact implementation of the manual continuous DMD stack layout.

Python

import math

class ContinuousAllocator:  
    """A bump-allocator representing a continuous 2D physical grid."""  
    def \_\_init\_\_(self):  
        self.ptr \= 1  
        self.cost \= 0

    def alloc(self, size: int) \-\> int:  
        addr \= self.ptr  
        self.ptr \+= size  
        return addr

    def read(self, addr: int):  
        \# 2D Routing Cost: ceil(sqrt(addr))  
        self.cost \+= math.isqrt(max(0, addr \- 1)) \+ 1

def naive\_matmul(N: int):  
    mem \= ContinuousAllocator()  
      
    \# Arguments allocated immediately (Starts at Address 1\)  
    A \= mem.alloc(N \* N)  
    B \= mem.alloc(N \* N)  
    C \= mem.alloc(N \* N)

    for i in range(N):  
        for j in range(N):  
            for k in range(N):  
                \# O(N^3) expensive reads directly from distant main memory  
                mem.read(A \+ i \* N \+ k)  
                mem.read(B \+ k \* N \+ j)  
                mem.read(C \+ i \* N \+ j) \# Read accumulator  
                  
    return mem.cost

def tiled\_matmul(N: int, T: int):  
    mem \= ContinuousAllocator()  
      
    \# 1\. Scratchpad allocated BELOW (Addresses 1 to 3\*T^2)  
    \# These become permanently pinned to the cheapest physical addresses.  
    fast\_A \= mem.alloc(T \* T)  
    fast\_B \= mem.alloc(T \* T)  
    fast\_C \= mem.alloc(T \* T)  
      
    \# 2\. Arguments allocated ABOVE (pushed out to higher, more expensive addresses)  
    A \= mem.alloc(N \* N)  
    B \= mem.alloc(N \* N)  
    C \= mem.alloc(N \* N)

    for bi in range(0, N, T):  
        for bj in range(0, N, T):  
            for bk in range(0, N, T):  
                  
                \# Phase 1: Block Fetch (Move data from expensive memory to cheap scratch)  
                for i in range(T):  
                    for k in range(T):  
                        mem.read(A \+ (bi \+ i) \* N \+ (bk \+ k))  
                for k in range(T):  
                    for j in range(T):  
                        mem.read(B \+ (bk \+ k) \* N \+ (bj \+ j))

                \# Phase 2: Compute (Math using ONLY the cheap, locally pinned scratchpad)  
                for i in range(T):  
                    for j in range(T):  
                        for k in range(T):  
                            mem.read(fast\_A \+ i \* T \+ k)  
                            mem.read(fast\_B \+ k \* T \+ j)  
                            mem.read(fast\_C \+ i \* T \+ j) \# Read accumulator  
                              
    return mem.cost

if \_\_name\_\_ \== "\_\_main\_\_":  
    N, T \= 16, 4  
      
    cost\_naive \= naive\_matmul(N)  
    cost\_tiled \= tiled\_matmul(N, T)  
      
    print(f"Matrix Size: {N}x{N}, Tile Size: {T}x{T}")  
    print("-" \* 40)  
    print(f"Naive Matmul Cost: {cost\_naive:,}")  
    print(f"Tiled Matmul Cost: {cost\_tiled:,}")  
    print(f"Advantage: Tiled is {cost\_naive / cost\_tiled:.2f}x cheaper")

### **Why this beats LRU caches:**

If this code was run on an LRU cache (like a CPU), sweeping across the large matrix blocks of $A$ and $B$ during Phase 1 would routinely sink your $C$ accumulator to the bottom of the LRU stack, dynamically inflating its distance penalty.

By manually reserving addresses $1 \dots 48$, you forcefully declare that fast\_A, fast\_B, and fast\_C can **never be displaced**. The physical wires connecting them to the ALU remain short forever, mathematically enforcing optimal energy efficiency.