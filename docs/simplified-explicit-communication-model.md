# Simplified Explicit Communication Model

Bill Dally ([*On the Model of Computation*, CACM
2022](https://cacm.acm.org/opinion/on-the-model-of-computation-point/))
proposed modeling algorithm data movement explicitly on the Manhattan
grid.

This is a simplified implementation of that model for a single
processor, designed to price a single function call.

![Upper-half-plane Manhattan layout](simplified_explicit_communication_model.svg)

- Processor is at the origin, memory is arranged as a 2D grid in the
  upper half-plane around it.
- Every cell is linearly indexed; `ceil(sqrt(idx))` gives the Manhattan
  distance from the core.

## Cost model (what is priced)

- **Reads are priced.** The cost of a read is the Manhattan distance
  from the core to the cell being read.
- **Writes are free.**
- **Arithmetic is free.**

## Function semantics

- **At the start of a call**, the location of every input byte is
  specified by the caller. Inputs are placed on the grid for free
- **At the end of a call**, the location of every output byte is
  specified by the caller, these incur standard read cost.
