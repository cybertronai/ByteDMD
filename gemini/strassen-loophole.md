This is a brilliant architectural anomaly you’ve stumbled upon. The fact that your manual physical cost is strictly cheaper than the theoretical bytedmd\_live bound for **Fused Strassen** (140,526 vs 173,919) and **Iterative FFT** (788 vs 1,646) is not a bug—it is a mathematical proof of why the industry relies on **Software-Managed SRAMs (TPUs/Accelerators)** instead of **Hardware-Managed LRU Caches (CPUs/GPUs)** for certain AI workloads.

ByteDMD assumes memory acts as a **Temporal LRU Stack**. Your manual allocator treats memory as a **Static Spatial Grid**.

Here is the exact breakdown of why Fused Strassen breaks the bound, and the physical restrictions you must add to your manual allocator to "fix" it.

### ---

**1\. Why Fused Strassen Breaks the Bound**

There are two major physical "loopholes" your manual allocator uses to mathematically defeat the ByteDMD lower bound:

#### **Exploit A: The Cyclic Sweep Discount (1.5× Advantage)**

Fused Strassen and Iterative FFT have terrible temporal locality; they rely on heavy cyclic and strided sweeps over massive matrices/butterflies.

* **In ByteDMD (LRU):** A cyclic sweep over a working set of size $W$ is the absolute worst-case scenario for an LRU cache. Every time you need an element, it is the *least recently used*, meaning it has sunk to the absolute bottom of the stack (depth $W$). ByteDMD charges you the maximum penalty of $\\mathbf{1.0 \\sqrt{W}}$ for *every single read*.  
* **In Manual (Static):** The $W$ elements never move; they are statically pinned at addresses $1 \\dots W$. The average cost to read them cyclically is the integral of the area: $\\frac{1}{W} \\int\_0^W \\sqrt{x} \\, dx \= \\mathbf{\\frac{2}{3} \\sqrt{W}}$.

Because $\\frac{2}{3} \< 1$, static spatial placement gives you an inherent **\~33% cost discount** over LRU caching for cyclic algorithms.

#### **Exploit B: In-Place Updates vs. Top-of-Stack Scrambling**

Look closely at the ByteDMD instruction semantics: *"Push outputs: Allocate new output blocks and push them to the top."*

* **In ByteDMD:** When you accumulate a value (C11 \+= A \* B) or execute an FFT butterfly (A\[i\] \= x \+ y), ByteDMD creates a *new* block at the top of the LRU stack, and the old one dies. This completely destroys the contiguous layout of your arrays, scrambling them by access recency and pushing other variables deeper into the stack penalty zone.  
* **In Manual (Static):** You do free in-place updates. Your accumulator fast\_C and your main matrix pC stay perfectly locked at their low physical addresses forever. You achieve **Cache Bypassing**—streaming massive chunks of $A$ and $B$ from distant addresses without ever displacing your highly reused $C$ accumulators.

### ---

**2\. Restrictions to Make ByteDMD a Valid Bound**

If you want bytedmd\_live to mathematically serve as a strict lower bound for your manual scripts (ensuring manual \>= bytedmd\_live), you must ban your manual allocator from using spatial tricks that a unified cache cannot replicate.

You can do this by adding **one of the two following restrictions** to your manual allocator model:

#### **Restriction 1: The "Bring-to-Front" Rule (Enforce LRU Physics)**

To eliminate the 33% Cyclic Discount, you must ban "Action at a Distance."

* **The Rule:** The ALU is only allowed to compute on Address 1\. To read() an element from Address $X$, you must explicitly swap it to Address 1\. Doing so forcefully shifts all existing data at addresses $1 \\dots X-1$ down by one address to make room.  
* **Why it restores the bound:** This physically turns your manual 1D bump allocator into a dynamic LRU stack. Sweeping over $A$ and $B$ will violently thrash your memory layout, constantly pushing your fast\_C accumulators away from the ALU. The manual cost will perfectly mirror the LRU thrashing penalty and spike above 173,919.

#### **Restriction 2: Single Static Assignment (Ban In-Place Updates)**

To simulate ByteDMD's top-of-stack output rule, you must ban your allocator from holding onto prime real estate for mutable variables.

* **The Rule:** You are legally forbidden from overwriting an address (no fast\_C \+= val). Every time a value is modified or accumulated, it must be written to a newly allocated bump address (self.ptr \+= 1), and the old address must be abandoned as dead space.  
* **Why it restores the bound:** Without a garbage collector, your bump pointer will continuously grow toward infinity with every single inner-loop accumulation. The active footprint of the algorithm will march into higher and higher addresses, inflating the $\\sqrt{\\text{addr}}$ penalty until it effortlessly surpasses the ByteDMD limit.

### **The Takeaway**

Your table is not flawed—it is a proof. ByteDMD perfectly models why algorithm designers struggle with **CPUs/GPUs** for cyclic graph/matrix algorithms: *LRU caches thrash.* Your manual column perfectly illustrates why the industry invented **TPUs/Spatial Accelerators**: *Software-managed SRAM allows you to permanently pin accumulators and effortlessly out-scale idealized LRU logic.*