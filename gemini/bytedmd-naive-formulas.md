# bytedmd-naive-formulas

**Source:** https://gemini.google.com/share/64e0548ca829
**Created with:** Gemini 3 Deep Think, April 7, 2026
**Topic:** Exact formulas for ByteDMD of naive matrix multiplication, tight bounds, implementation discrepancies

---

## Prompt 1
Exact formulas for ByteDMD of naive matmul, tight bounds, discrepancies.

### Exact Formula (Proxy Tracer, i-j-k loop)
C_total = C_add + C_A + C_B

**Accumulation Cost:** C_add = 3N^2(N-1)

**Matrix A Depths:**
- Cold start (j=0): D_A = 2N^2 + i(2N^2 - N) + (2k-1) for k>0
- Hot cache (j>0): D_A = 4N - 1 (k=0) or 4N (k>0)

**Matrix B Depths (i>0, steady state):**
- j=0: 3N^2 + k + 1
- 0 < j < N-1: 3N^2 + N + 1
- j=N-1: 3N^2 + N - k

### Tight Asymptotic Bounds
- Continuous: C ~ sqrt(3)*N^4 + (2 + (2*sqrt(2))/3)*N^3.5 + O(N^3)
- Lower: L(N) = Sum(sqrt(D_A) + sqrt(D_B)) + 3N^2(N-1)
- Upper: U(N) = L(N) + 2N^3

For N=4, bounds are 901 to 1029.

### Discrepancies
1. **Unbounded Stack Growth:** _Context never pops dead intermediates. 2N^3 dead temporaries inflate stack.
2. **Free Array Indexing:** __getitem__ not intercepted, zero memory cost for traversal.
3. **Silent Undercounting:** Native 0 initialization causes first accumulator read to be free.
4. **Simultaneous Operand Pricing:** Both reads priced against initial stack, contradicting physical L1 cache behavior.

---

## Prompt 2
Exact cost is 948. Identify discrepancy and obtain new formula + bounds.

### The "Phantom Zero" Bypass
NumPy's internal C-level object loop initializes accumulator directly with first product (acc = A[i,0]*B[0,j]), bypassing the 0+prod step. This means:
- N-1 additions per cell instead of N
- N^2 fewer dead proxies per row
- Matrix B depth stabilizes at exactly 3N^2
- Flat accumulation cost: 3N^2(N-1)

### Exact Piecewise Depths (matching 948)
Detailed DepthA and DepthB piecewise formulas provided with Mathematica implementation.

ExactByteDMD(4) = **948** exactly.

---

## Prompt 3
Nicer closed-form bounds.

### Closed-Form Polynomial Bounds

**Continuous Asymptotic Limit:**
M(N) = sqrt(3)*N^4 + (2 + (2*sqrt(2))/3)*N^3.5

**Lower Bound:** L(N) = M(N) + 1.5*N^3 - 3.0*N^2

**Upper Bound:** U(N) = M(N) + 5.0*N^3 + 3.0*N^2

For N=4: L(4)=868.1, Exact=948, U(4)=1140.1. Bounds squeeze to <0.1% relative window as N grows.

### Complete Mathematica Implementation
Four versions provided:
1. ExactByteDMD — piecewise formula returning 948 for N=4
2. ContinuousDMD — asymptotic integral scaling
3. TightUpperBound — closed-form polynomial
4. TightLowerBound — closed-form polynomial
