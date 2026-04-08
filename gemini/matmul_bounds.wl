#!/usr/bin/env wolframscript
(* ByteDMD matmul: empirical vs Gemini's exact/continuous/tight bounds.
   Source of the formulas: https://gemini.google.com/share/d4f2325d4cdb *)

ClearAll[Allocate, ReadAllThenMove, TrackedOp, TrackedAdd, TrackedMul,
  Wrap, Matmul4, Matmul4Snake, Matmul4Tiled, Matvec4, Vecmat4, MeasureByteDMD];

(* 1. Concise Allocation *)
Allocate[size_] := ($Sizes[++$Counter] = size; AppendTo[$Stack, $Counter]; $Counter);

(* 2. Declarative Stack Search (Replaces Catch/Throw looping) *)
ReadAllThenMove[keys_List] := Module[{pos, depth, size},
  Do[
    If[KeyExistsQ[$Sizes, key],
      pos = FirstPosition[$Stack, key];
      If[!MissingQ[pos],
        (* 'Drop' takes elements *after* the index, naturally representing newer MRU keys *)
        depth = Total[Lookup[$Sizes, Drop[$Stack, pos[[1]]]]];
        size = $Sizes[key];
        $Accesses = Join[$Accesses, Range[depth + size, depth + 1, -1]];
      ]
    ]
  , {key, keys}];

  (* Update LRU stack seamlessly *)
  Do[
    If[KeyExistsQ[$Sizes, key],
      $Stack = DeleteCases[$Stack, key];
      AppendTo[$Stack, key];
    ]
  , {key, keys}];
];

(* 3. Pattern Matching & Higher-Order Functions (DRY) *)
TrackedOp[op_, {idA_, valA_, sizeA_}, {idB_, valB_, sizeB_}] := Module[{newS = Max[sizeA, sizeB]},
  ReadAllThenMove[{idA, idB}];
  {Allocate[newS], op[valA, valB], newS}
];

TrackedAdd[a_, b_] := TrackedOp[Plus, a, b];
TrackedMul[a_, b_] := TrackedOp[Times, a, b];

(* 4. Universal Tensor Wrapper (Level {-1} natively handles N-dimensional nesting) *)
Wrap[tensor_, size_ : 1] := Map[{Allocate[size], #, size} &, tensor, {-1}];

(* 5. Mathematical Operations via Functional Accumulation (Fold) *)
Matmul4[A_, B_] := Table[
  Fold[
    TrackedAdd[#1, TrackedMul[A[[i, #2]], B[[#2, j]]]] &,
    TrackedMul[A[[i, 1]], B[[1, j]]],
    Range[2, Length[A]]
  ], {i, Length[A]}, {j, Length[B[[1]]]}]

(* Snake order: reverse the j-loop direction on alternating rows *)
Matmul4Snake[A_, B_] := Module[{n = Length[A], m = Length[B[[1]]], js},
  Table[
    js = If[OddQ[i], Range[m], Range[m, 1, -1]];
    AssociationThread[js -> Table[
      Fold[
        TrackedAdd[#1, TrackedMul[A[[i, #2]], B[[#2, j]]]] &,
        TrackedMul[A[[i, 1]], B[[1, j]]],
        Range[2, n]
      ], {j, js}]]
  , {i, n}]
];

Matvec4[A_, x_] := Table[
  Fold[
    TrackedAdd[#1, TrackedMul[A[[i, #2]], x[[#2]]]] &,
    TrackedMul[A[[i, 1]], x[[1]]],
    Range[2, Length[x]]
  ], {i, Length[A]}]

Vecmat4[A_, x_] := Table[
  Fold[
    TrackedAdd[#1, TrackedMul[x[[#2]], A[[#2, j]]]] &,
    TrackedMul[x[[1]], A[[1, j]]],
    Range[2, Length[x]]
  ], {j, Length[A[[1]]]}]

(* Tiled block execution explicitly preserves procedural looping to track rigid cache bounds *)
Matmul4Tiled[A_, B_] := Module[{n = Length[A], t = 2, Cmat},
  Cmat = ConstantArray[Null, {n, n}];
  Do[
    With[{mul = TrackedMul[A[[i, k]], B[[k, j]]]},
      Cmat[[i, j]] = If[Cmat[[i, j]] === Null, mul, TrackedAdd[Cmat[[i, j]], mul]]
    ];
  , {bi, 1, n, t}, {bj, 1, n, t}, {bk, 1, n, t},
    {i, bi, bi + t - 1}, {j, bj, bj + t - 1}, {k, bk, bk + t - 1}];
  Cmat
];

(* 6. Unified Sandbox Evaluator & Listable Math *)
MeasureByteDMD[Func_, arg1_, arg2_, size_ : 1] := Block[
  {$Stack = {}, $Sizes = <||>, $Accesses = {}, $Counter = 0},
  Func[Wrap[arg1, size], Wrap[arg2, size]];
  Total[Ceiling[Sqrt[$Accesses]]]
];

(* --- Run tests --- *)
Atest = ConstantArray[1, {4, 4}];
Btest = ConstantArray[1, {4, 4}];
xtest = ConstantArray[1, 4];

$AnyFailed = False;

AssertEqual[name_String, got_, expected_] := (
  If[got =!= expected,
    Print["FAIL: " <> name <> " (expected " <> ToString[expected] <> ", got " <> ToString[got] <> ")"];
    $AnyFailed = True;
  ];
  {name, got}
);

resultsTable = {
  {"Test Operation", "ByteDMD Cost"},
  AssertEqual["test_matmul4", MeasureByteDMD[Matmul4, Atest, Btest], 948],
  AssertEqual["test_matmul4_tiled", MeasureByteDMD[Matmul4Tiled, Atest, Btest], 947],
  AssertEqual["test_matvec4", MeasureByteDMD[Matvec4, Atest, xtest], 194],
  AssertEqual["test_vecmat4", MeasureByteDMD[Vecmat4, Atest, xtest], 191]
};

(* Print a pure ASCII table *)
Print[""];
Print[StringJoin[ConstantArray["=", 50]]];
Do[Print[
   StringPadRight[ToString[resultsTable[[i, 1]]], 32] <> "| " <>
    ToString[resultsTable[[i, 2]]]];
  If[i == 1, Print[StringJoin[ConstantArray["-", 50]]]], {i, 1,
   Length[resultsTable]}];
Print[StringJoin[ConstantArray["=", 50]]];
Print[""];

If[$AnyFailed, Print["Some tests FAILED."]; Exit[1],
  Print["All tests passed."]];

(* --- Matmul cost sweep over sizes --- *)
(* MatmulSizeTable[sizes] prints a table of ByteDMD costs for naive i-j-k
   matrix multiplication on square all-ones matrices of each given size.
   Usage:  MatmulSizeTable[{2, 4, 8}]                                         *)
MatmulSizeTable[sizes_List] := Module[{n, A, B, cIJK, cSnake, rows},
  rows = Table[
    n = sz;
    A = ConstantArray[1, {n, n}];
    B = ConstantArray[1, {n, n}];
    cIJK   = MeasureByteDMD[Matmul4,      A, B];
    cSnake = MeasureByteDMD[Matmul4Snake, A, B];
    {n, cIJK, cSnake}
  , {sz, sizes}];
  Print[""];
  Print[StringJoin[ConstantArray["=", 44]]];
  Print[StringPadRight["N", 6] <> "| " <>
        StringPadRight["i-j-k", 14] <> "| " <>
        StringPadRight["snake-j", 14]];
  Print[StringJoin[ConstantArray["-", 44]]];
  Do[Print[
       StringPadRight[ToString[r[[1]]], 6] <> "| " <>
       StringPadRight[ToString[r[[2]]], 14] <> "| " <>
       StringPadRight[ToString[r[[3]]], 14]], {r, rows}];
  Print[StringJoin[ConstantArray["=", 44]]];
  rows
];

MatmulSizeTable[{2, 4, 8}];

measureByteDMDEmpirical[n_] := (
    A = ConstantArray[1, {n, n}];
    B = ConstantArray[1, {n, n}];
    MeasureByteDMD[Matmul4, A, B]
);

ClearAll[DepthA, DepthB, ExactByteDMD, ContinuousDMD, TightUpperBound,
  TightLowerBound];

(* =========================================================================*)
(* 1. EXACT FORMULAS (Piecewise depths mapped directly from the Tracer)     *)
(* =========================================================================*)
DepthA[n_, i_, j_, k_] :=
  If[j > 0, If[k == 0, 4*n - 1, 4*n],
   If[k == 0, 2*n^2 + i*(2*n^2 - n), 2*n^2 + i*(2*n^2 - n) + 2*k - 1]];

DepthB[n_, i_, j_, k_] :=
  If[i > 0,
   If[j == 0, If[k == 0, 3*n^2, 3*n^2 + k + 1],
    If[0 < j < n - 1, If[k == 0, 3*n^2 + n, 3*n^2 + n + 1],
     If[k == 0, 3*n^2 + n - 1,
      3*n^2 + n -
       k]]],(*Transient logic for moving B from initial call-
   stack order (i=0)*)
   If[j == 0, If[k == 0, n^2, n^2 - 1 + k*(4 - n)],
    With[{DB0 = n^2 + 3*n - 1 + (j - 1)*(2*n - 1),
      delta = Max[4 - n, j - n + 3]},
     If[k == 0, DB0, DB0 + k*delta - 1]]]];

ExactByteDMD[n_] :=
  3*n^2*(n - 1) +
   Sum[Ceiling[Sqrt[DepthA[n, i, j, k]]] +
     Ceiling[Sqrt[DepthB[n, i, j, k]]], {i, 0, n - 1}, {j, 0,
     n - 1}, {k, 0, n - 1}];

(* =========================================================================*)
(* 2. CONTINUOUS DMD APPROXIMATION (Integral Limit)                          *)
(* =========================================================================*)
ContinuousDMD[n_] := Sqrt[3]*n^4 + (2 + (2*Sqrt[2])/3)*n^(7/2);

(* =========================================================================*)
(* 3 & 4. EXACT TIGHT UPPER AND LOWER BOUNDS                                 *)
(* Bounding the fractional Ceiling overhead precisely across the 2N^3 reads  *)
(* =========================================================================*)
TightLowerBound[n_] :=
  3*n^2*(n - 1) +
   Sum[Sqrt[DepthA[n, i, j, k]] + Sqrt[DepthB[n, i, j, k]], {i, 0,
     n - 1}, {j, 0, n - 1}, {k, 0, n - 1}];

TightUpperBound[n_] := TightLowerBound[n] + 2*n^3;

(* =====EVALUATION=====*)
PredictionsRow[n_] := {
  n,
  measureByteDMDEmpirical[n],
  ExactByteDMD[n],
  N[ContinuousDMD[n]],
  N[TightLowerBound[n]],
  N[TightUpperBound[n]]
};

PrintPredictionsTable[sizes_List] := Module[{rows, fmt},
  rows = PredictionsRow /@ sizes;
  Print[""];
  Print[StringJoin[ConstantArray["=", 86]]];
  Print[StringPadRight["N", 4] <> "| " <>
        StringPadRight["empirical", 12] <> "| " <>
        StringPadRight["exact", 12] <> "| " <>
        StringPadRight["continuous", 14] <> "| " <>
        StringPadRight["lower", 14] <> "| " <>
        StringPadRight["upper", 14]];
  Print[StringJoin[ConstantArray["-", 86]]];
  fmt[x_] := ToString[NumberForm[x, {10, 2}]];
  Do[Print[
       StringPadRight[ToString[r[[1]]], 4] <> "| " <>
       StringPadRight[ToString[r[[2]]], 12] <> "| " <>
       StringPadRight[ToString[r[[3]]], 12] <> "| " <>
       StringPadRight[fmt[r[[4]]], 14] <> "| " <>
       StringPadRight[fmt[r[[5]]], 14] <> "| " <>
       StringPadRight[fmt[r[[6]]], 14]], {r, rows}];
  Print[StringJoin[ConstantArray["=", 86]]];
  rows
];

PrintPredictionsTable[{2, 3, 4, 5, 6, 7, 8}];
