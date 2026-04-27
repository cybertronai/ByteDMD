## User

Uploaded a file

Uploaded a file

I'm interested in ByteDMD numbers for naive matrix multiplication. Give me closed form formulas for discrete (original) and continuous bytedmd using naive matmul, and also naive matmul using snake order. If the discrete Formulas are too hard. Instead, give tight bounds.


---

## ChatGPT

I’m deriving the formulas from the ByteDMD stack model itself and checking them against the provided implementation and benchmark notes so the result matches the actual tracer behavior.


---

## ChatGPT

A straightforward Python `i-j-k` trace doesn’t match the README’s 4×4 cost, so I’m aligning with the repo’s exact benchmark conventions and separating standard row-major from snake-order.


---

## ChatGPT

I’ve now pinned down the exact depth patterns under the actual `bytedmd.py` semantics. Standard and snake-order naive matmul differ only in the B-access pattern; A and accumulator/product accesses are identical.


---

## ChatGPT

I’m taking the **exact benchmark kernels** from the repo: standard `i-j-k` computes each dot product as `s = A[i][0]*B[0][j]`, then for `k=1..n-1` does `s = s + A[i][k]*B[k][j]`; `snake-j` uses the same inner work but alternates the `j` direction by row. I’m also using the repo’s README / regular-tracer semantics: only reads cost, read operands are moved to the LRU top, and every arithmetic op pushes a fresh result. Those are exactly the rules behind the README’s 4×4 costs 948 (`i-j-k`) and 906 (`snake-j`). 

README

 

bytedmd

 [GitHub](https://github.com/cybertronai/ByteDMD/raw/refs/heads/main/benchmarks/benchmark_linalg.py?utm_source=chatgpt.com)

I’ll write the formulas for an $n\times n$ multiply, $n\ge 2$.

## 1) Generic depth-cost function

Let $d$ be the **element depth** in the LRU stack.

For the benchmark setting $w=1$ byte-per-element:

$$
G\_{\mathrm{disc}}(d)=\lceil \sqrt d\rceil,\qquad
G\_{\mathrm{cont}}(d)=\sqrt d.
$$

For literal $w$-byte ByteDMD, replace that by

$$
G^{(w)}\_{\mathrm{disc}}(d)=U(dw)-U((d-1)w),
$$

where

$$
U(m)=\sum\_{r=1}^m \lceil \sqrt r\rceil
=\frac{M(6m-2M^2+3M-1)}{6},\qquad M=\lceil \sqrt m\rceil,
$$

which is exactly the helper implemented in `bytedmd.py`, and

$$
G^{(w)}\_{\mathrm{cont}}(d)=\sum\_{q=1}^{w}\sqrt{(d-1)w+q}.
$$

The README benchmark numbers use the default `bytes_per_element=1`, so $w=1$ is the setting that reproduces 948 and 906. 

bytedmd

 [GitHub](https://github.com/cybertronai/ByteDMD/raw/refs/heads/main/benchmarks/benchmark_linalg.py?utm_source=chatgpt.com)

A useful simplification is that every add step

$$
s \leftarrow s + (A[i][k]B[k][j]),\qquad k\ge 1
$$

always reads depths $4$ and $1$, so the whole add contribution is

$$
\Phi(n;G)=n^2(n-1)\bigl(G(4)+G(1)\bigr).
$$

For $w=1$, both discrete and continuous give $G(4)+G(1)=2+1=3$, so

$$
\Phi(n)=3n^2(n-1).
$$

## 2) Shared $A$-side contribution

The $A$-side is the same for standard and snake order.

Define

$$
A(n;G)=
\sum\_{i=0}^{n-1}
\left[
G\!\bigl(2n^2+i\,n(2n-1)\bigr)
+
\sum\_{k=1}^{n-1}
G\!\bigl(2n^2+i\,n(2n-1)+2k-1\bigr)
\right]
$$

$$
\qquad\qquad
+n(n-1)\,G(4n-1)
+n(n-1)^2\,G(4n).
$$

Interpretation:  
the first dot product of each row is “cold” on the $A$-side; the other $n-1$ dots in that row are “warm”.

## 3) Standard `i-j-k`

The $B$-side splits into the first row and all later rows.

### First row

$$
B^{(0)}\_{\mathrm{std}}(n;G)=
G(n^2)
+
\sum\_{k=1}^{n-1} G\!\bigl(n^2-1+(4-n)k\bigr)
$$

$$
\qquad\qquad
+
\sum\_{j=1}^{n-1}
\left[
G\!\bigl(n^2+n+j(2n-1)\bigr)
+
\sum\_{k=1}^{n-1}
G\!\bigl(n^2+n+j(2n-1)-1+k(j+3-n)\bigr)
\right].
$$

### Rows $i=1,\dots,n-1$

$$
B^{(+)}\_{\mathrm{std}}(n;G)=
(n-1)\Bigg[
G(3n^2)
+
\sum\_{k=1}^{n-1} G(3n^2+k+1)
$$

$$
\qquad\qquad\qquad
+(n-2)\Bigl(G(3n^2+n)+(n-1)G(3n^2+n+1)\Bigr)
+G(3n^2+n-1)
+\sum\_{k=1}^{n-1}G(3n^2+n-k)
\Bigg].
$$

### Total

$$
C\_{\mathrm{std}}(n;G)=A(n;G)+B^{(0)}\_{\mathrm{std}}(n;G)+B^{(+)}\_{\mathrm{std}}(n;G)+\Phi(n;G).
$$

So the exact discrete and continuous costs are just

$$
C^{\mathrm{disc}}\_{\mathrm{std}}(n,w)=C\_{\mathrm{std}}\!\bigl(n;G^{(w)}\_{\mathrm{disc}}\bigr),
\qquad
C^{\mathrm{cont}}\_{\mathrm{std}}(n,w)=C\_{\mathrm{std}}\!\bigl(n;G^{(w)}\_{\mathrm{cont}}\bigr).
$$

## 4) Snake-`j`

Here the $A$-side stays the same; only the later-row $B$-side changes.

Let $r=0,1,\dots,n-1$ be the **visit rank** of the column in that row:  
first visited column has $r=0$, second has $r=1$, and so on.  
For forward rows this is $r=j$; for reversed rows this is $r=n-1-j$.

After the first row, the $B$-depth multiset depends only on $r$, not on the row index.

### First row

Same as standard:

$$
B^{(0)}\_{\mathrm{snake}}(n;G)=B^{(0)}\_{\mathrm{std}}(n;G).
$$

### Rows $i=1,\dots,n-1$

$$
B^{(+)}\_{\mathrm{snake}}(n;G)=
(n-1)\Bigg[
G(4n-2)+(n-1)G(4n-1)
$$

$$
\qquad\qquad\qquad
+\sum\_{r=1}^{n-1}
\Bigl(
G\!\bigl(r(5n-2)+5n-1\bigr)
+(n-1)G\!\bigl(r(5n-2)+5n\bigr)
\Bigr)
\Bigg].
$$

### Total

$$
C\_{\mathrm{snake}}(n;G)=A(n;G)+B^{(0)}\_{\mathrm{std}}(n;G)+B^{(+)}\_{\mathrm{snake}}(n;G)+\Phi(n;G).
$$

Thus

$$
C^{\mathrm{disc}}\_{\mathrm{snake}}(n,w)=C\_{\mathrm{snake}}\!\bigl(n;G^{(w)}\_{\mathrm{disc}}\bigr),
\qquad
C^{\mathrm{cont}}\_{\mathrm{snake}}(n,w)=C\_{\mathrm{snake}}\!\bigl(n;G^{(w)}\_{\mathrm{cont}}\bigr).
$$

## 5) Sanity checks

For the repo benchmark setting $w=1$:

$$
C^{\mathrm{disc}}\_{\mathrm{std}}(4,1)=948,\qquad
C^{\mathrm{disc}}\_{\mathrm{snake}}(4,1)=906,
$$

matching the README. 

README

The corresponding continuous / unrounded values are

$$
C^{\mathrm{cont}}\_{\mathrm{std}}(4,1)\approx 901.3407008548,
\qquad
C^{\mathrm{cont}}\_{\mathrm{snake}}(4,1)\approx 869.3761304268.
$$

## 6) Tight bounds and asymptotics

For $w=1$, the total number of scalar reads is

$$
2n^3 + 2n^2(n-1)=4n^3-2n^2,
$$

so

$$
0\le C^{\mathrm{disc}}\_{\mathrm{std}}(n,1)-C^{\mathrm{cont}}\_{\mathrm{std}}(n,1)<4n^3-2n^2,
$$

$$
0\le C^{\mathrm{disc}}\_{\mathrm{snake}}(n,1)-C^{\mathrm{cont}}\_{\mathrm{snake}}(n,1)<4n^3-2n^2.
$$

For $w$-byte scalars, multiply the RHS by $w$.

The leading asymptotics are

$$
C^{\mathrm{cont}}\_{\mathrm{std}}(n,1)=\sqrt{3}\,n^4+O(n^{7/2}),
\qquad
C^{\mathrm{cont}}\_{\mathrm{snake}}(n,1)=\frac{2\sqrt{5}}{3}\,n^4+O(n^{7/2}),
$$

and the discrete costs have the same leading terms, since the discrete-continuous gap is only $O(n^3)$.

So asymptotically, snake order improves the dominant term by the factor

$$
\frac{ \frac{2\sqrt5}{3} }{\sqrt3}
=\frac{2\sqrt5}{3\sqrt3}
\approx 0.8607.
$$

One nice takeaway: **snake changes only the $B$-side leading term; the $A$-side formula is identical to standard row-major `i-j-k`.**
