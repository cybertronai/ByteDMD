#!/usr/bin/env wolframscript

ClearAll[Allocate, ReadAllThenMove, TrackedOp, TrackedAdd, TrackedMul,
  Wrap, Matmul4, Matmul4Tiled, Matvec4, Vecmat4, MeasureByteDMD];

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
