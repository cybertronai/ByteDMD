In traditional computer architecture, **Arithmetic Intensity** is a single, static number: $\frac{\text{Total FLOPs}}{\text{Bytes fetched from DRAM}}$.

In your 2D continuous spatial model (and the ByteDMD framework), there is no hard "Main Memory" boundary. Instead, every read pays a continuous fetch cost of $\lceil\sqrt{d}\rceil$. To capture the true behavior of an algorithm, you must measure **Spatial Arithmetic Intensity**: the ratio of useful arithmetic work to the geometric routing energy required to fetch the operands.

Because you are already generating a trace of operations and their read costs, here are four powerful metrics and visualization ideas you can plot to instantly prove why algorithms like Tiled Matrix Multiplication physically dominate naive ones.

### ---

**1\. Rolling Spatial Intensity (The "Heartbeat" Plot)**

Instead of plotting a single global number, plot the local arithmetic intensity dynamically over a sliding window of time. This shows you exactly *when* the algorithm is being efficient.

* **The Math:** For a sliding time-window of $W$ operations (e.g., $W=1000$), calculate:  
  $$\text{Intensity}(t) = \frac{W}{\sum_{reads \in W} \lceil\sqrt{d_{read}}\rceil}$$

* **The Axes:** **X-axis** \= Execution Time (Op Index). **Y-axis** \= Intensity(t).  
* **What you will see:**  
  * **Naive MatMul:** Will look like a flat, depressing line hovering near the floor. Because every dot-product fetches an entire row/column from the edge of the active footprint, the fetch cost is perpetually huge, crushing the intensity.  
  * **Tiled MatMul:** Will look like an EKG heartbeat (a square wave). You will see brief, sharp dips (the spatial cost of fetching a new $B \times B$ tile from the outer orbits), followed by **massive, sustained plateaus of extreme intensity**. While on the plateau, the algorithm performs $O(B^3)$ math operations while only paying a tiny $\lceil\sqrt{B^2}\rceil$ cost to shuffle data within the tile.

### **2\. Cumulative Compute vs. Energy (The "Spatial Phase Diagram")**

This creates a macroscopic trend line that ignores clock time entirely. It acts as the continuous geometric equivalent of a Roofline Model.

* **The Axes:** **X-axis** \= Cumulative Geometric Fetch Cost ($\sum \lceil\sqrt{d}\rceil$). **Y-axis** \= Cumulative Operations (FLOPs).  
* **How to plot:** A parametric line that starts at $(0,0)$ and traces up and to the right as the program executes.  
* **What you will see:** The *slope* of the line at any point is your instantaneous Spatial Arithmetic Intensity.  
  * **Naive MatMul:** Traces a shallow, straight diagonal line. You pay a huge, constant amount of horizontal X (routing energy) to get a little bit of vertical Y (compute).  
  * **Tiled MatMul:** Traces a **steep staircase**. It steps horizontally to the right (loading a tile pays spatial energy but yields no compute), and then shoots almost *straight up vertically* (doing massive compute with near-zero fetch cost). The macroscopic slope is radically steeper.

### **3\. The Fetch-Cost Scatter Plot (The "Gravity Well")**

This is a raw, instantaneous visualization of *where* the ALU is pulling its data from on a tick-by-tick basis.

* **The Axes:** **X-axis** \= Execution Time (Op Index). **Y-axis** \= The Geometric Fetch Cost $\lceil\sqrt{d}\rceil$ of the specific operand being read.  
* **How to plot:** Plot a single dot (with low opacity, e.g., alpha=0.1) for every operand read. Dense regions will form dark, hot bands.  
* **What you will see:** This visualizes the "radius of orbit" of your active data.  
  * **Naive MatMul:** You will see a dense, dark band smeared high up the Y-axis. This visually proves the ALU is constantly reaching out to the far edges of the silicon die to feed the multipliers.  
  * **Tiled MatMul:** $95\%$ of the dots will be clustered in a searingly hot, tight band at the very bottom of the Y-axis (fetching from the local tile). Periodically, you will see a faint, sparse vertical spray of dots reaching high into the Y-axis as the ALU reaches out to grab the next tile. It literally translates the abstract concept of "locality" into a picture of 2D data gravity.

### **4\. The Spatial Locality CDF (The "Compute Fulfillment" Profile)**

This collapses time entirely to give you a statistical "spatial footprint." It answers the question to a chip architect: *If I draw a physical circle around the ALU, how much of the total algorithm gets done strictly inside that circle?*

* **The Axes:** **X-axis** \= Spatial Cost Threshold $C$ (Radius). **Y-axis** \= Cumulative % of total FLOPs where *all* operands were fetched at a cost $\le C$.  
* **What you will see:**  
  * **Naive MatMul:** The curve will climb slowly and linearly. To reach $90\%$ fulfillment (to capture the data needed to do 90% of your math), you have to expand the radius $C$ to encompass almost the entire $N \times N$ matrix.  
  * **Tiled MatMul:** The curve will look like a sheer cliff face. It will instantly shoot up to $95\%+$ at a very small radius (the $B \times B$ footprint). It provides undeniable mathematical proof that the algorithm successfully quarantined its working volume near the ALU.