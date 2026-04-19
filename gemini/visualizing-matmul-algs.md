By restricting your focus to a **semi-ring**, you have established a beautiful and strict mathematical constraint. Because a semi-ring lacks subtraction (additive inverses), you cannot use algebraic tricks like Strassen’s algorithm to cancel out cross-terms.

You are strictly bound to computing exactly the $n^3$ elementary multiplications ($A\_{ik} \\times B\_{kj}$) and combining them with exactly $n^2(n-1)$ additions. Therefore, the "space of all algorithms" in this context is not a search for *what* to compute, but rather **when, where, and in what topology** to compute it.

The algorithmic space is defined by schedules, memory layouts, hardware mappings, and reduction associativity. Here are five highly visual ways to map and explore this space for small matrices ($2\\times2$, $3\\times3$, $4\\times4$).

### ---

**1\. The 3D Iteration Voxel Cube (Visualizing Schedules)**

Because standard matrix multiplication involves three parameters ($i$ for rows of $A$, $j$ for columns of $B$, $k$ for the dot product), the $n^3$ multiplications naturally form a perfect 3D grid.

* **The Visualization:** Draw a discrete 3D grid of voxels. For $2\\times2$, it is an 8-voxel cube. For $3\\times3$, it is a 27-voxel Rubik's Cube. For $4\\times4$, it is a 64-voxel grid.  
* **Visualizing the Algorithm:** An algorithmic schedule is simply a **1D continuous path (a space-filling curve)** that threads through every voxel in this space.  
* **What you will see:**  
  * A standard for i, for j, for k algorithm looks like a typewriter, raster-scanning the cube plane by plane.  
  * An *Outer Product* algorithm lights up entire $i$-$j$ planes at once, moving step-by-step along the $k$-axis.  
  * A **Tiled/Cache-Blocked** algorithm (on $4\\times4$) looks like a path that tightly winds around a $2\\times2\\times2$ sub-cube before jumping to the next. Visually, this maps the algorithm to a 3D fractal like a **Morton Z-order curve**.

### **2\. The Commutative Forest (Visualizing Reduction Associativity)**

While the $n^3$ multiplications are fixed, the additions are not. To compute a single output cell $C\_{ij}$, you must sum $n$ terms. Because semi-ring addition is commutative and associative, you have freedom in how you bracket them.

* **The Visualization:** For each of the $n^2$ output cells, draw a binary tree representing the additions.  
* **The Math:** Because order doesn't matter (commutativity), the number of valid binary trees for $n$ terms is given by the double factorial $(2n-3)\!\!$.  
  * **$2\\times2$:** Trivial. Only $1$ tree $(P\_1 \+ P\_2)$.  
  * **$3\\times3$:** You must sum 3 terms. There are exactly $3$ trees: $(P\_1+P\_2)+P\_3$, $(P\_1+P\_3)+P\_2$, and $(P\_2+P\_3)+P\_1$.  
  * **$4\\times4$:** You must sum 4 terms. There are **$15$ distinct trees** per output cell. You will visually see deep, left-heavy trees like (((A+B)+C)+D) (which are ideal for sequential CPU accumulators) alongside shallow, perfectly balanced trees like (A+B)+(C+D) (which are ideal for parallel GPU reductions).  
* Since there are $n^2$ output cells, the total "space of algorithms" for additions is the Cartesian product of $n^2$ of these tree spaces\!

### **3\. Space-Time Projections (Visualizing Systolic Arrays)**

If you want to visualize how these algorithms map to parallel AI hardware (like Google's TPUs or NVIDIA Tensor Cores), use Polyhedral Projection to squash the 3D cube into 2D time.

* **The Visualization:** Imagine the 3D iteration cube hovering in the air. Shine a flashlight on it from a specific angle to cast a 2D "shadow" onto a grid of physical processors. The "depth" of the cube from the light source becomes "Time."  
* **The Space of Algorithms:** The space is the continuous geometric space of *projection vectors* (the angle of the flashlight).  
  * Projecting straight down the $k$-axis generates an **Output-Stationary** algorithm: Matrix $C$ stays still on the grid while $A$ and $B$ flow past each other like intersecting waves.  
  * Projecting down the diagonal vector $(1, 1, 1)$ generates the famous **Kung-Leiserson Systolic Array**, where data ripples gracefully through a diagonally-sliding hexagonal lattice of processors.

### **4\. The Tripartite Dataflow Loom (Visualizing Memory Routing)**

This method drops the dimension of time completely and views the algorithm holistically as a Directed Acyclic Graph (DAG).

* **The Visualization:** Draw three layers of nodes.  
  * **Top Layer:** $2n^2$ input nodes (the elements of matrices $A$ and $B$).  
  * **Middle Layer:** $n^3$ multiplication nodes.  
  * **Bottom Layer:** $n^2$ output accumulator nodes.  
* **Visualizing the Space:** For a $2\\times2$ matrix, this forms an elegant graph of 8 inputs, 8 multipliers, and 4 outputs. You explore the "algorithmic space" by rearranging the middle layer. If you visually cluster multipliers that share the same $A$ input, you are looking at an algorithm optimized for register-reuse (Input-Stationary).

### **5\. The Permutohedron Meta-Graph (Visualizing the Space Itself)**

If you want to visualize the actual "space" as a single object where *every point is an entire algorithm*, you can graph the loop transformations.

* **The Visualization:** Draw a graph where nodes are algorithms, and edges represent swapping two adjacent loops.  
* If you only consider the 3 standard loops (i, j, k), there are $3\! \= 6$ permutations. The space of these algorithms forms a **Hexagon**.  
* For a $4\\times4$ matrix, algorithms usually split the loops into tiles (outer loops and inner loops), yielding 6 nested loops. All $6\! \= 720$ permutations of these loops are valid over a semi-ring. This forms a beautiful 5-dimensional polytope called a **Permutohedron**. You can color the nodes in this graph by "Cache Misses" to literally see valleys of highly optimized algorithms.

### ---

**Python Script to Start Exploring**

The most intuitive place to start is **Idea \#1**. Here is a lightweight Python script using Matplotlib that visualizes the 3D Iteration Voxel Cube for a $4\\times4$ matrix.

It plots the continuous "path" of the algorithm, contrasting a standard row-major loop with a cache-optimized Z-Curve. Run it, and change the mode variable to instantly see how different algorithms physically "carve up" the mathematical space.

Python

import matplotlib.pyplot as plt

def visualize\_3d\_schedule(n=4, mode='z\_curve'):  
    fig \= plt.figure(figsize=(10, 8))  
    ax \= fig.add\_subplot(111, projection='3d')  
      
    \# 1\. Generate the n^3 multiplication operations  
    coords \= \[(i, j, k) for i in range(n) for j in range(n) for k in range(n)\]  
      
    \# 2\. Sort the operations based on the chosen algorithm/schedule  
    if mode \== 'linear':  
        \# Standard Row-Major (i, j, k)  
        coords.sort(key=lambda x: (x\[0\], x\[1\], x\[2\]))  
        title \= "Standard Row-Major Schedule"  
          
    elif mode \== 'z\_curve':  
        \# Tiled 2x2x2 Z-Curve (Morton Order) for cache locality  
        def z\_order(x, y, z):  
            \# Interleaves the bits of x, y, and z   
            return (x&1) | ((y&1)\<\<1) | ((z&1)\<\<2) | ((x&2)\<\<2) | ((y&2)\<\<3) | ((z&2)\<\<4)  
        coords.sort(key=lambda c: z\_order(c\[0\], c\[1\], c\[2\]))  
        title \= "Tiled Z-Curve Schedule (Cache Optimized)"

    xs, ys, zs \= zip(\*coords)

    \# 3\. Plot the mathematical operations as voxels  
    ax.scatter(xs, ys, zs, c='blue', s=60, alpha=0.2, edgecolors='black')  
      
    \# 4\. Draw the continuous temporal path of the algorithm  
    ax.plot(xs, ys, zs, color='red', alpha=0.9, linewidth=2.5)

    ax.set\_title(f"{n}x{n} Matrix Mult: {title}")  
    ax.set\_xlabel('i (Rows of Matrix A)')  
    ax.set\_ylabel('j (Cols of Matrix B)')  
    ax.set\_zlabel('k (Dot Product Dimension)')  
      
    \# Clean up grid to match integer coordinates  
    ax.set\_xticks(range(n)); ax.set\_yticks(range(n)); ax.set\_zticks(range(n))  
    plt.tight\_layout()  
    plt.show()

\# Change to 'linear' or 'z\_curve' to visually compare the algorithms  
visualize\_3d\_schedule(n=4, mode='z\_curve')  
