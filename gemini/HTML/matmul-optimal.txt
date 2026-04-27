# matmul-optimal

Created with Gemini 3 Deep Think April 9, 2026 at 09:20 PM
Published April 10, 2026 at 05:31 PM

Source: https://gemini.google.com/share/b56b8bd21a1c

---

## User

I'm interseted in using THompson 2D vlsi model or more precisely the spatial computer attached to analyze the complexity N*N matrix vector multiplication. Tell me what is the energy lower bound, is it possible to achieve it, what is a low energy way to implement matrix vector multiplication, give me detailed specification of a spatial computer that implements such thing

## Gemini

Based on the Spatial Computer model introduced by Gianinazzi et al.—which abstracts modern 2D spatial architectures by strictly pricing communication energy based on the physical Manhattan distance messages travel—here is the formal complexity analysis and detailed architectural specification for an N×N matrix-vector multiplication (y=Ax).

1. What is the Energy Lower Bound?

For an N×N matrix-vector multiplication, the absolute energy lower bound on the spatial computer is Ω(N2).

Proof of the bound:
In the Spatial Computer model, processors are restricted to a constant O(1) memory. To simply store the N2 elements of the dense matrix A, you are mathematically forced to distribute them across a 2D physical subgrid of at least N×N processors. This creates a grid with a physical diameter of Θ(N).

To compute the output vector y, every element xj​ of the N-dimensional input vector must be multiplied by every element in the corresponding j-th column of matrix A. Because the N matrix elements of that column are spatially spread across the grid, routing a single scalar xj​ to all of them requires messages to traverse a minimum physical distance (Steiner tree weight) of Ω(N). Doing this for all N elements of vector x takes N×Ω(N)=Ω(N2) energy. A symmetric Ω(N2) energy cost applies to physically gathering and summing the partial products across the rows to form the output vector y.

2. Is it Possible to Achieve This Bound?

Yes. It is perfectly possible to achieve the exact Θ(N2) energy bound.

The paper points out a heavy Ω(n1.5) energy "permutation bottleneck" for sorting and square matrix-matrix multiplication (which corresponds to Ω(N3) for an N×N matrix). Matrix-vector multiplication entirely avoids this bottleneck. Because the matrix data requires O(N2) area but the vectors only represent O(N) data, we can keep the massive matrix completely stationary and only route the vectors, keeping communication strictly localized to hit the Θ(N2) bound.

(Note: There is an energy-depth tradeoff here. If you use the paper's 1D Broadcast tree to speed up the operation to O(logN) depth, the long wires would increase the energy to a sub-optimal O(N2logN). To get optimal energy, we must accept an O(N) depth).

3. A Low-Energy Way to Implement It

To hit the absolute minimum Θ(N2) energy, we completely avoid long-distance tree routing. Instead, we implement a 2D Systolic Wave-front Pipeline (conceptually similar to Cannon's algorithm).

The Strategy:
The matrix A remains entirely stationary. The vector x is streamed vertically downward column-by-column. As elements of x pass over elements of A, they are locally multiplied. The accumulating sums for the output vector y are then streamed horizontally to the left row-by-row. Because messages only ever travel to immediate physical neighbors, the energy cost per message is precisely 1 unit, strictly minimizing the total energy and wire-depth.

4. Detailed Specification of the Spatial Computer

Below is the strict node-level and lifecycle specification mapping this optimal algorithm to the Spatial Computer rules defined in the paper.

A. Hardware Architecture per Node

Grid Allocation: An unbounded 2D Cartesian grid is used, bounding active execution to an N×N subgrid where processor pi,j​ sits at coordinates (i,j) for 0≤i,j<N.

Local Memory (O(1) constraint): Each processor utilizes exactly four constant-sized registers:

A_val: Permanently stores the matrix element Ai,j​.

x_val: Temporarily stores the streaming vector element xj​.

prod: Stores the locally computed scalar product.

sum_val: Stores the accumulating horizontal sum.

Queues: A constant-sized receive queue. To prevent undefined behavior, the execution protocol below guarantees a processor will never receive more than 2 messages in a single time-step.

ALU: Capable of O(1) arithmetic operations per cycle (a fused multiply-add).

B. Initial State

Processor pi,j​ holds its respective Ai,j​ in A_val.

The top row of processors (p0,j​) holds the input vector xj​ in their x_val registers.

Output will naturally materialize at the left-most column (pi,0​).

C. Execution Protocol (Synchronous Time-Steps)

All processors share a global clock t (starting at t=0). To respect data dependencies, execution flows as a diagonal wavefront. Every clock cycle, all active processors simultaneously run the following logic based on their local physical (i,j) coordinates:

Python
# Phase 1: Vector Shift & Local Multiply
if t == i:
    if i > 0:
        x_val = dequeue_from(North)     # Receive x_j from above
        
    prod = A_val * x_val                # O(1) Local ALU computation
    
    if i < N - 1:
        send_to(South, x_val)           # Route x_j downwards (Cost: 1 Energy)

# Phase 2: Row Reduction (Accumulating y_i)
if t == i + (N - 1 - j):
    if j < N - 1:
        sum_val = dequeue_from(East)    # Receive partial sum from the right
    else:
        sum_val = 0                     # Right-most column starts at 0
        
    sum_val = sum_val + prod            # O(1) Local ALU computation
    
    if j > 0:
        send_to(West, sum_val)          # Route partial sum leftwards (Cost: 1 Energy)
    else:
        # Final output y_i is complete and stored in sum_val at j=0
        terminate()

D. Final Complexity Output

Total Energy (E):

The vector x takes N−1 distance-1 hops down each of the N columns = N(N−1) energy.

The sums take N−1 distance-1 hops left across each of the N rows = N(N−1) energy.

Total Energy = 2N2−2N=Θ(N2). This perfectly matches the lower bound.

Depth (D): The longest critical dependency chain is the time it takes xN−1​ to reach the bottom right (N−1 steps), plus the time for yN−1​'s sum to reach the bottom left (N−1 steps). Depth = 2N−2=Θ(N).

Wire-Depth (Dw​): Because no individual message traverses a physical distance greater than 1, the maximum energy expended over any dependent chain is completely proportional to the depth. Wire-depth = Θ(N).

## User

 now give me the corresponding low-energy way to do matrix-vector multiplication.

## Gemini

In my previous response, I outlined the Systolic Wavefront approach. That method is the absolute lowest-energy way to perform Matrix-Vector multiplication, achieving the mathematical minimum Θ(N2) energy by restricting all communication to immediate physical neighbors. However, it requires a linear Θ(N) depth (latency) to execute.

Depending on what you meant by "corresponding," here are the two alternative implementations based strictly on the rules and abstractions of the Gianinazzi et al. Spatial Computer paper.

1. The "Spatial Primitives" Method (Low-Depth Matrix-Vector)

If by "corresponding" you are looking for the counterpart algorithm that optimizes for highly parallel, sub-linear speed using the specific geometric primitives defined in Section 2 of the paper, this is how you do it. It drops the latency to logarithmic time, but incurs a slight O(logN) energy penalty because messages must traverse longer physical wires.

Initial Layout: The matrix A is mapped in row-major order so processor pi,j​ holds Ai,j​. The input vector x is loaded into the top row so p0,j​ holds xj​.

Phase 1: Column-wise 1D Broadcast (O(logN) depth)
Instead of passing xj​ down the column one neighbor at a time, every processor in the top row acts as the root of a spatial binary tree, initiating a 1D Broadcast (as defined in Lemma 2.2) down its column.
Cost: The paper proves a 1D broadcast of length N takes O(NlogN) energy. Doing this across all N columns in parallel takes O(N2logN) energy.

Phase 2: Local Compute (O(1) depth)
Every processor pi,j​ now holds both Ai,j​ and xj​. In a single synchronous clock cycle, all N2 processors execute the local multiplication: prod = A_val * x_val.
Cost: 0 communication energy.

Phase 3: Row-wise 1D Reduce (O(logN) depth)
To sum the products horizontally and form the output vector y, every row executes a 1D Reduce (Corollary 2.3). The spatial binary tree is used in reverse: pairs of processors sum their values and pass them leftward, halving the active nodes at each step until the final sum yi​ lands at pi,0​.
Cost: A 1D reduce of length N takes O(NlogN) energy. Doing this for all N rows simultaneously takes O(N2logN) energy.

Final Complexity:

Energy: O(N2logN)

Depth: O(logN) (Exponentially faster than the systolic approach)

Wire-Depth: O(N)

2. Cannon's Algorithm (Low-Energy Matrix-Matrix)

If by "corresponding" you had a typo and meant the algorithm for N×N Matrix-Matrix Multiplication (C=A×B)—which is heavily analyzed in Section 4 of the provided paper—the rules and complexities change entirely.

Because you have to move an O(N2) volume of data across an O(N2) physical grid, Matrix-Matrix multiplication suffers from the permutation bottleneck (Lemma 4.1).

Lower Bound: The absolute energy lower bound mathematically jumps to Ω(N3).

The Low-Energy Implementation: To achieve this lower bound without wasting energy on long-distance communication, the paper (Section 4.1) recommends Cannon’s Algorithm.

Execution Protocol:

Initial Skew: Circularly shift the i-th row of matrix A to the left by i positions. Circularly shift the j-th column of matrix B upward by j positions.

Multiply & Shift: For N synchronous iterations, every processor does the following:

Multiplies its currently co-located elements of A and B, adding the result to a stationary local Ci,j​ register.

Circularly shifts its element of A to the left neighbor (distance 1).

Circularly shifts its element of B to the top neighbor (distance 1).

Final Complexity:

Energy: Θ(N3) (Mathematically optimal for Matrix-Matrix)

Depth: Θ(N)

Wire-Depth: Θ(N)
