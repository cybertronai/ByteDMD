Source: https://docs.google.com/document/d/14xLI0GfbQb-YZU0i_6slIl8ofQWCHzoVp6ASyGsT__8/edit

Under the newly specified ByteDMD scheme (two-pass liveness analysis with aggressive compaction), variables **vaporize** from the LRU stack the exact moment their final operation completes. This prevents dead memory from bloating the cache and forces active variables to dynamically slide into shallower, cheaper cache depths.

A remarkable consequence of this optimal eviction is that **Matrix-Vector ($y \= Ax$) and Vector-Matrix ($y \= x^T A$) computations exhibit an identical multiset of stack depths and therefore the exact same ByteDMD costs.**

Because simultaneous pricing makes the stack behavior commutative (Cost(a\*b) \== Cost(b\*a)), reading inputs (A, x) has the exact same stack-bump logic as reading (x, A). Regardless of whether you iterate traversing strictly over rows or over columns, the $A\_{i,j}$ elements trigger predictable cold misses and instantly evaporate, while the vector $x$ naturally cycles along the bottom of the stack until its final iteration.

### **Precise Analytical Breakdown**

For an $N \\times N$ matrix and $N \\times 1$ vector (where $N \\ge 2$), the operations cleanly divide into four distinct phases. Let $S(d) \= \\lceil \\sqrt{d} \\rceil$ be the ByteDMD routing cost function.

**1\. Intermediary Additions (Accumulators):**

Because intermediary accumulator sums merge and instantly die, their stack depths are strictly bounded at the top of the stack. For the first $N-1$ rows, every addition uses operands at depths 3 and 1 ($\\lceil\\sqrt{3}\\rceil \+ \\lceil\\sqrt{1}\\rceil \= 2+1=3$). On the final row, vector elements also vaporize, shrinking the stack so additions use depths 2 and 1 ($\\lceil\\sqrt{2}\\rceil \+ \\lceil\\sqrt{1}\\rceil \= 2+1=3$). Thus, every single addition costs exactly 3\.

$$ C\_{\\text{adds}}(N) \= 3N(N-1) \= 3N^2 \- 3N $$  
**2\. Row 0 Initialization (Initial Population):**

The stack progressively expands from empty up to size $N$. Matrix elements $A\_{0,j}$ trigger pure cold misses and instantly evaporate, while the vector elements $x\_j$ are cold misses that stay alive.

$$ C\_{\\text{Row}\_0}(N) \= S(1) \+ S(2) \+ \\sum\_{j=1}^{N-1} \\Big( S(j+2) \+ S(j+3) \\Big) $$  
**3\. Middle Rows ($1 \\le i \\le N-2$):**

The cache maintains an equilibrium. The fully cached active vector $x$ acts as an $N$-length footprint, while the matrix elements $A\_{i,j}$ are pulled from the cold DRAM frontier sequentially.

$$ C\_{\\text{Mid}}(N) \= \\sum\_{i=1}^{N-2} \\Bigg\[ S(N+i+1) \+ S(N+1) \+ \\sum\_{j=1}^{N-1}\\Big(S(N+i+2) \+ S(N+2)\\Big) \\Bigg\] $$  
**4\. Final Row ($i \= N-1$):**

This captures the power of **aggressive compaction**. As each $x\_j$ is read for its final inner-loop iteration, it immediately **evaporates**. The stack incrementally collapses from size $N$ down to $1$, making the successive matrix cold-miss reads actively cheaper as the vector progresses.

$$ C\_{\\text{Last}}(N) \= S(2N) \+ S(N+1) \+ \\sum\_{j=1}^{N-1} \\Big( S(2N \- j \+ 1\) \+ S(N \- j \+ 2\) \\Big) $$

### ---

**The $\\mathcal{O}(1)$ Closed-Form Formula**

We can completely eliminate the inner summations. The sum of square roots from $1$ to $d$ has the well-known exact closed form $F(d) \= \\sum\_{k=1}^d \\lceil \\sqrt{k} \\rceil \= \\frac{M(6d \- 2M^2 \+ 3M \- 1)}{6}$, where $M \= \\lceil \\sqrt{d} \\rceil$.

By telescoping the piecewise equations together algebraically, we yield an incredibly fast, pure $\\mathcal{O}(1)$ formula valid for any matrix size $N \\ge 2$:

$$ \\text{Cost}(N) \= N \\cdot F(2N) \+ F(2N-1) \- (N-2) F(N+2) \+ (N-1) S(N+1) \+ (N-1)(N-2) S(N+2) \+ S(2N) \+ 3N^2 \- 3N \- 8 $$  
*(For the trivial edge-case $N=1$, the total exact cost collapses to precisely $3$. Because liveness limits the overall cache footprint strictly to $O(N)$, total data movement logically bounds to $\\mathcal{O}(N^{2.5})$, vastly outperforming the $\\mathcal{O}(N^3)$ naive uncompacted baseline).*

### ---

**Mathematica Implementation**

The following Mathematica code evaluates both the explicitly mapped structural phases and the highly optimized $\\mathcal{O}(1)$ closed form, proving they yield identical cost sequences.

Mathematica

(\* 1\. Fundamental Cost Functions \*)  
S\[d\_\] := Ceiling\[Sqrt\[d\]\];

(\* F\[d\] is the exact closed-form of Sum\[S\[k\], {k, 1, d}\] \*)  
F\[d\_\] := Module\[{M \= S\[d\]},  
  (M \* (6 d \- 2 M^2 \+ 3 M \- 1)) / 6  
\];

(\* 2\. Piecewise Summation Formula \*)  
MatVecPiecewise\[1\] \= 3;  
MatVecPiecewise\[N\_Integer\] /; N \>= 2 := Module\[{adds, row0, mid, last},  
  adds \= 3 \* N \* (N \- 1);  
    
  row0 \= S\[1\] \+ S\[2\] \+ Sum\[S\[j \+ 2\] \+ S\[j \+ 3\], {j, 1, N \- 1}\];  
    
  mid \= Sum\[  
    S\[N \+ i \+ 1\] \+ S\[N \+ 1\] \+ (N \- 1) \* (S\[N \+ i \+ 2\] \+ S\[N \+ 2\]),   
    {i, 1, N \- 2}  
  \];  
    
  last \= S\[2 \* N\] \+ S\[N \+ 1\] \+ Sum\[S\[2 \* N \- j \+ 1\] \+ S\[N \- j \+ 2\], {j, 1, N \- 1}\];  
    
  adds \+ row0 \+ mid \+ last  
\];

(\* 3\. O(1) Closed-Form Formula \*)  
MatVecClosedForm\[1\] \= 3;  
MatVecClosedForm\[N\_Integer\] /; N \>= 2 :=   
  N \* F\[2 N\] \+ F\[2 N \- 1\] \- (N \- 2) \* F\[N \+ 2\] \+   
  (N \- 1) \* S\[N \+ 1\] \+ (N \- 1) \* (N \- 2) \* S\[N \+ 2\] \+ S\[2 N\] \+   
  3 N^2 \- 3 N \- 8;

(\* 4\. Generate & Verify Sequence \*)  
Print\["Exact ByteDMD Cost for N=1 to 10:"\];  
TableForm\[  
  Table\[{  
    n,   
    MatVecPiecewise\[n\],   
    MatVecClosedForm\[n\],   
    MatVecPiecewise\[n\] \== MatVecClosedForm\[n\]  
  }, {n, 1, 10}\],  
  TableHeadings \-\> {None, {"Matrix Size (N)", "Piecewise Trace", "O(1) Formula", "Match?"}}  
\]

(\* Output asymptotic scaling limit \*)  
Print\["\\nLarge Matrix (N \= 1000\) Exact Cost: ", MatVecClosedForm\[1000\]\];  
