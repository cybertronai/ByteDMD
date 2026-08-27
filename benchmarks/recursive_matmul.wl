#!/usr/bin/env wolframscript
(* Exact ByteDMD cost for vanilla recursive square matrix multiplication.

   Matched semantics:
     - lazy / demand-paged inputs
     - simultaneous operand pricing per READ event
     - move-to-top after pricing
     - STORE event for every scalar multiply and scalar add
     - two-pass aggressive liveness compaction

   Matched algorithm:
     C11 = A11.B11 + A12.B21
     C12 = A11.B12 + A12.B22
     C21 = A21.B11 + A22.B21
     C22 = A21.B12 + A22.B22
   with scalar base case N=1.

   Assumes N is a positive power of two.
   Only runs N in {2, 4, 8} because the Mathematica replay is slow.

   Source: gemini/vanilla-recursive2-11apr26.md  (adapted)

   Run:
     ./benchmarks/recursive_matmul.wl
     wolframscript benchmarks/recursive_matmul.wl
*)

ByteDMDCostRecursiveMatMul[n_Integer?Positive,
   bytesPerElement_Integer : 1] :=
  Module[{counter = 0, events = {}, outputs, splitMatrix, rec,
    addMatrices, sumUSqrt, elementCost, lastUse, stack = {},
    traceCost = 0, unique, liveBefore, dMap, coldSeen, pos, touched},
   If[BitAnd[n, n - 1] != 0,
    Return["Error: n must be a positive power of 2."]];
   sumUSqrt[x_Integer] := Module[{m}, If[x <= 0, Return[0]];
     m = IntegerPart[Sqrt[x - 1]] + 1;
     Quotient[m (6 x - 2 m^2 + 3 m - 1), 6]];
   elementCost[d_Integer] :=
    If[d <= 0, 0,
     If[bytesPerElement == 1, IntegerPart[Sqrt[d - 1]] + 1,
      sumUSqrt[d bytesPerElement] - sumUSqrt[(d - 1) bytesPerElement]]];
   splitMatrix[mat_] :=
    Module[{m = Length[mat]/2}, {mat[[1 ;; m, 1 ;; m]],
      mat[[1 ;; m, m + 1 ;; -1]], mat[[m + 1 ;; -1, 1 ;; m]],
      mat[[m + 1 ;; -1, m + 1 ;; -1]]}];
   addMatrices[m1_, m2_] := Module[{id}, MapThread[(id = ++counter;
        AppendTo[events, {"READ", {#1, #2}}];
        AppendTo[events, {"STORE", id}];
        id) &, {m1, m2}, 2]];
   rec[a_, b_] :=
    Module[{size = Length[a], a11, a12, a21, a22, b11, b12, b21, b22,
      c11, c12, c21, c22, id}, If[size == 1, id = ++counter;
      AppendTo[events, {"READ", {a[[1, 1]], b[[1, 1]]}}];
      AppendTo[events, {"STORE", id}];
      Return[{{id}}]];
     {a11, a12, a21, a22} = splitMatrix[a];
     {b11, b12, b21, b22} = splitMatrix[b];
     c11 = addMatrices[rec[a11, b11], rec[a12, b21]];
     c12 = addMatrices[rec[a11, b12], rec[a12, b22]];
     c21 = addMatrices[rec[a21, b11], rec[a22, b21]];
     c22 = addMatrices[rec[a21, b12], rec[a22, b22]];
     ArrayFlatten[{{c11, c12}, {c21, c22}}]];
   outputs = Module[{a, b}, a = Table[++counter, {n}, {n}];
     b = Table[++counter, {n}, {n}];
     rec[a, b]];
   (* Pass 1: last use *)
   lastUse = Association[];
   Do[With[{ev = events[[i]]},
     If[ev[[1]] === "READ", Scan[(lastUse[#] = i) &, ev[[2]]],
      If[! KeyExistsQ[lastUse, ev[[2]]], lastUse[ev[[2]]] = i]]], {i,
     Length[events]}];
   Scan[(lastUse[#] = Length[events] + 1) &, Flatten[outputs]];
   (* Pass 2: exact stack simulation *)
   Do[With[{ev = events[[i]]},
     If[ev[[1]] === "STORE", stack = Append[stack, ev[[2]]];
      touched = {ev[[2]]};, unique = DeleteDuplicates[ev[[2]]];
      liveBefore = Length[stack];
      dMap = Association[];
      coldSeen = 0;
      Do[pos = FirstPosition[stack, k];
       If[MissingQ[pos], coldSeen += 1;
        dMap[k] = liveBefore + coldSeen,
        dMap[k] = liveBefore - pos[[1]] + 1], {k, unique}];
      Do[traceCost += elementCost[dMap[k]], {k, ev[[2]]}];
      stack = Join[DeleteCases[stack, Alternatives @@ unique], unique];
      touched = unique;];
     stack = Select[stack, lastUse[#] > i &];], {i, Length[events]}];
   traceCost];

ByteDMDFirstValues[maxPower_Integer?NonNegative,
   bytesPerElement_Integer : 1] :=
  Table[{2^p, ByteDMDCostRecursiveMatMul[2^p, bytesPerElement]}, {p,
    0, maxPower}];


(* Print the cost table *)
Print["Vanilla recursive matmul (leaf=1) ByteDMD cost:"];
Print[""];
Print["   N | ByteDMD"];
Print["-----|---------"];
Do[
  Print[
    "  " <> StringPadLeft[ToString[n], 2] <> " | " <>
    StringPadLeft[ToString[ByteDMDCostRecursiveMatMul[n]], 7]
  ],
  {n, {2, 4, 8}}
];
