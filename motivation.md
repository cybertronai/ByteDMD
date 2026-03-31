# A metric of complexity for the 21st century: ByteDMD 

## Motivation

Modern costs are mainly determined by the cost of data movement hence FLOP-count is no longer a good heuristic.

The objective is to define a computation model with a cost function which is more realistic than FLOP-count.

Bill Dally (in ACM [https://cacm.acm.org/opinion/on-the-model-of-computation-point/](https://cacm.acm.org/opinion/on-the-model-of-computation-point/)) argues for a model of computation where bytes have physical location on a 2D grid and the cost of fetching data from position x1,y1 to processor located at x2,y2 should be proportional to the Manhattan distance between the byte and the processor.

![][image1]

People are not used to specifying x,y placement of bytes in their variables so we need to come up with an automatic x,y placement strategy.

An idea to address this comes from the "geometric stack" and the Data Movement Distance, Wesley Smith, Aidan Goldfarb, Chen Ding, Beyond Time Complexity: Data Movement Complexity Analysis for Matrix Multiplication", [https://arxiv.org/abs/2203.02536](https://arxiv.org/abs/2203.02536)

To motivate geometric stack, note that modern processors have a hierarchy of caches, with larger caches being more expensive to access.

![][image2]

There's a range of automatic replacement policies to determine how which cache level to use for which data, LRU lemma tells us that using LRU heuristic is competitive within a factor of 2\. (TODO: clarify this result)

For a given set of cache sizes L1, L2, L3 we can determine the cost of running the algorithm. The downside is that this defines a family of metrics, which makes it harder to compare algorithms. An algorithm may be better for one set of cache sizes but worse for another.  
![][image3]

To act as a replacement of "FLOP-count" as a measure of complexity we want a single metric rather than a family of metrics.

The solution of Ding/Smith is to introduce the idea of "geometric stack", instead of 3 layers of cache, which have an infinite number of layers, where each piece of data lives on its own level L. The cost of accessing layer L is sqrt(L).

This cost model can be seen as a continuous approximation of Bill Dally's "manhattan distance" combined with an LRU cache.

TODO(y): clean-up this diagram  
[Colab](https://colab.research.google.com/drive/1Gt6DNTmnzQsqqL0twFdmKaAKQfP-hP_G#scrollTo=2XzD3vHE4t5E)

![][image4]

The positions on the stack are indexed by bytes rather than abstract data-types because modern algorithm designs have freedom to improve the runtime of their algorithm by using a smaller data type for intermediate variables, whereas we want our metric to be sensitive to such choices.

## Computation model

We can think of our computer C being an idealized processor with infinite registers and a set of instructions. Writes are free, reads incur cost related to distance and update the stack. We assume that this computer can execute arbitrary functions, with arguments of the function preloaded onto the stack.

Suppose we run C on a function like func(arg1, arg2), this proceeds as follows

The scheduler loads arguments arg1,arg2 onto the LRU stack at 0 cost.  
It then proceeds to execute instructions sequentially. Each instruction takes a list of ordered inputs, at least 1, and produces a list of ordered outputs, at least 0\.

To execute the instruction, the scheduler  
1\. loads all instruction inputs from the LRU stack into the registers  
2\. updates position of bytes read in the LRU stack, in the order of reading  
	ie suppose we start with stack \[a,b,c,d\]  
a. running instruction(a,b) will produce stack \[c,d,a,b\]  
b. running instruction(b, a) will produce stack \[c,d,b,a\]  
3\. puts instruction output onto the stack.

For example, consider the following function add(a, b, c)  
def add(a, b, c, d):  
  return b+c

We call it this function with four int8 values.

Running this function in an idealized computer C then proceeds as follows:

Load a, b, c, d onto the stack

stack: \[a, b, c, d\]  
registers: \[\]  
cost: 0

In this stack we consider d to have distance 1 (most recently used), c has distance 2 and b has distance 3\.

Upon encountering addition instruction b+c:

Read b into the registers, incurring cost sqrt(3)  
stack: \[a, b, c, d\]  
registers: \[b\]  
cost: sqrt(3)

Read c into the registers, incurring cost sqrt(2)  
stack: \[a, b, c, d\]  
registers: \[b, c\]  
cost: sqrt(2)

Update positions of arguments b,c on the stack, in the order they were read into registers  
stack: \[a, d, b, c\]  
registers: \[b, c\]  
cost: 0

Perform addition operation, write result onto the stack, clear registers  
stack: \[a, d, b, c, b+c\]  
registers: \[\]  
cost: 0

Byte-DMD cost of running this function is sqrt(3)+sqrt(2)

Now consider calling this function with 

cost=measureDMD(function, arguments)

## Examples

### Order of reading values from the stack into registers matters

sum1(a, b, c, d):  
  return b+c

starting stack: \[a, b, c, d\]  
ending stack: \[a, d, b, c, b+c\]

sum2(a, b, c, d):  
  return c+b

starting stack: \[a, b, c, d\]  
ending stack: \[a, d, c, b, b+c\]

# Future work

## bits or bytes?

Here, the decision is to focus on byte-level boundaries because bit-level operations are not readily available. In the future, we may want our algorithm specification to work on the bit level and change the metric to count at bit-level boundaries.

## Extending to parallel 

In the example above, scheduler loads function arguments onto the LRU stack at 0 cost. For multi-processor system we may consider a location aware scheduler which incurs a cost that depends on location of original source of data.
