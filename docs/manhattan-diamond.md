# Manhattan Diamond

(aka, simplified explicit communication model)

Bill Dally ([*On the Model of Computation*, CACM
2022](https://cacm.acm.org/opinion/on-the-model-of-computation-point/)) proposed modeling algorithm data movement explicitly on the Manhattan grid.

This is a simplified implementation of this model for a single processor, designed to price a single function call.

![Manhattan-Diamond layout](manhattan_diamond.svg)

- Processor is in the center, memory is arranged on a 2D grid around it.
- Every cell is linearly indexed, `ceil(sqrt(idx))` gives the Manhattan distance from the center
- Only price reads, writes and arithmetic is free.


## Function semantics

- Arguments are pre-loaded on the read-only part of the memory (negative indices)
- Outputs and temporaries are in the writeable part of the memory (positive indices)
- Return value is assembled at the end by reading every output
