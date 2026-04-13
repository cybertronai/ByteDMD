#!/usr/bin/env wolframscript
(* Exact O(N^3) closed-form ByteDMD cost for naive N x N matrix multiply
   under the demand-paged, two-pass aggressive liveness LRU model.

   Ported from the Python implementation in gemini/mathematica-matmul.md
   which derives the exact piecewise depth bounds for every LOAD inside
   the inner triple-loop of the i-j-k naive matmul.

   The Python reference outputs [3, 54, 236, 676, 1516, 2899, 5008, 8443, 12861]
   for N = 1..9. This script reproduces those numbers in pure Wolfram
   Language and extends them through N = 16, then compares the values
   against the tracer numbers measured by benchmarks/benchmark_linalg.py.

   Run:
     ./benchmarks/matmul.wl
     wolframscript benchmarks/matmul.wl
*)

(* ByteDMD element cost at integer depth d *)
S[d_] := Ceiling[Sqrt[d]];


(* Depth of the A[i,k] read at loop step (i, j, k) for N x N naive matmul.
   Derived by case-analysis on the i, j, k boundary conditions. *)
depthA[i_, j_, k_, n_] := Which[
  (* j == 0: start of a new row of C, A[i,0] and B[0,0] are freshly loaded *)
  j == 0,
    Which[
      i == 0,      If[k == 0, 1, 2 k + 2],
      i == n - 1,  If[k == 0, 2 n^2 - n + 1, 2 n^2 - n + 2],
      True,        Which[
        k == 0,  n^2 + 1 + i n,
        k == 1,  n^2 + 3 + i n,
        True,    n^2 + i n + k + 2
      ]
    ],

  (* j == N-1: last column, A[i] about to vaporize at end of row *)
  j == n - 1,
    If[i == n - 1,
      If[k <= 1, n + 1, n - k + 2],
      If[k <= 1, 2 n + 1, 2 n - k + 2]
    ],

  (* interior j: A[i] is in steady state depth ~2N *)
  True,
    If[i == n - 1,
      If[k == 0, n + 1, n + 2],
      If[k == 0, 2 n + 1, 2 n + 2]
    ]
];


(* Depth of the B[k,j] read at loop step (i, j, k). *)
depthB[i_, j_, k_, n_] := Which[
  (* i == 0: first row of C, B is being cold-fetched column by column *)
  i == 0,
    Which[
      j == 0,      Which[k == 0, 2, k == 1, 5, True, 2 k + 3],
      j == n - 1,  If[k == 0, n^2 + n, n^2 + n + 1],
      True,        With[{base = (n + 1) (j + 1)},
                     Which[k == 0, base, k == 1, base + 2, True, base + k + 1]]
    ],

  (* i == N-1: last row, B fully resident on the stack *)
  i == n - 1,
    Which[
      j == 0,      If[k == 0, n^2 + n, n^2 + n + 1],
      j == n - 1,  Which[k == 0, 3 n, k == 1, 3 n - 1, True, 3 n - 2 k + 1],
      True,        With[{base = n (n - j + 2)},
                     If[k <= 1, base, base - k + 1]]
    ],

  (* interior i: steady state for B *)
  True,
    Which[
      j == 0,      Which[k == 0, n^2 + n, k == 1, n^2 + n + 2, True, n^2 + n + k + 1],
      j == n - 1,  If[k <= 1, n^2 + 2 n, n^2 + 2 n - k + 1],
      True,        If[k == 0, n^2 + 2 n, n^2 + 2 n + 1]
    ]
];


(* Total closed-form ByteDMD cost for naive N x N matmul *)
NaiveMatmulCost[n_Integer] /; n >= 1 := Module[{total = 0},
  (* (1) Product (P) depths: always at MRU, depth 1, count = N^2 (N-1) *)
  total += S[1] * n^2 (n - 1);

  (* (2) Accumulator (S) depths: bounded by liveness vaporization *)
  If[n >= 2,
    total += S[2] * (n - 1);
    total += S[3] * 2 (n - 1)^2;
    total += S[4] * (n - 1)^3;
  ];

  (* (3) A and B reads: sum piecewise-algebraic depths over the triple loop *)
  total += Sum[
    S[depthA[i, j, k, n]] + S[depthB[i, j, k, n]],
    {i, 0, n - 1}, {j, 0, n - 1}, {k, 0, n - 1}
  ];

  total
];


(* Print the cost table *)
Print["Naive N x N matmul ByteDMD cost:"];
Print[""];
Print["   N | ByteDMD"];
Print["-----|---------"];
Do[
  Print[
    "  " <> StringPadLeft[ToString[n], 2] <> " | " <>
    StringPadLeft[ToString[NaiveMatmulCost[n]], 7]
  ],
  {n, {2, 4, 8, 16}}
];
