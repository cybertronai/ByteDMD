#!/usr/bin/env wolframscript
(* Simulate the ByteDMD-cold read trace of an N x N matrix-vector multiply
   under the demand-paged, two-pass aggressive liveness model with
   global peak working set cold-miss pricing.

   Matched semantics (bytedmd_cold.py):
     - demand-paged inputs (no STORE events for A / x; cold-miss on
       first read)
     - simultaneous operand pricing per READ event
     - move-to-top after pricing
     - STORE event for every scalar multiply and scalar add
     - two-pass aggressive liveness compaction
     - Pass 1.5: compute peak working set across the full event tape
     - Cold misses priced at  peak_working_set + global_cold_counter
       (monotonically increasing, never reset)

   Returns {depths, hotcold} where hotcold is "cold" or "hot" for
   each entry in the depth trace.

   Matched algorithm (standard i-j accumulator loop):
       for i in 0..N-1:
           s = A[i,0] * x[0]
           for j in 1..N-1:
               s = s + A[i,j] * x[j]
           y[i] = s

   Run:
     ./experiments/matvec/matvec_cold_trace.wl
     wolframscript experiments/matvec/matvec_cold_trace.wl
*)

ClearAll[MatVecColdTrace];

MatVecColdTrace[n_Integer?Positive] := Module[
  {counter = 0, newID, mulOp, addOp, A, x, y, events = {},
   lastUse, stack, trace, hotcold, res, unique, liveBefore, dMap,
   coldSeen, pos, peakWS, simStack, globalCold, hcMap},

  newID[] := ++counter;

  mulOp[a_, b_] := Module[{c = newID[]},
    AppendTo[events, {"READ", {a, b}}];
    AppendTo[events, {"STORE", c}];
    c];
  addOp[a_, b_] := Module[{c = newID[]},
    AppendTo[events, {"READ", {a, b}}];
    AppendTo[events, {"STORE", c}];
    c];

  (* Logical IDs for A (row-major) and x — cold until first read. *)
  A = Table[newID[], {n}, {n}];
  x = Table[newID[], {n}];

  (* Pass 1: generate the event tape. *)
  y = Table[
    Module[{s},
      s = mulOp[A[[i, 1]], x[[1]]];
      Do[s = addOp[s, mulOp[A[[i, j]], x[[j]]]], {j, 2, n}];
      s],
    {i, n}];
  res = y;

  (* Pass 2: compute last-use index for every logical ID. *)
  lastUse = Association[];
  Do[With[{ev = events[[i]]},
    If[ev[[1]] === "READ",
      Scan[(lastUse[#] = i) &, ev[[2]]],
      If[!KeyExistsQ[lastUse, ev[[2]]], lastUse[ev[[2]]] = i]
    ]
   ], {i, Length[events]}];
  (* Output values never vaporize *)
  Scan[(lastUse[#] = Length[events] + 1) &, res];

  (* Pass 1.5: compute peak working set. *)
  simStack = {};
  peakWS = 0;
  Do[With[{ev = events[[i]]},
    If[ev[[1]] === "STORE",
      simStack = Append[simStack, ev[[2]]];
      If[Length[simStack] > peakWS, peakWS = Length[simStack]];
      simStack = Select[simStack, lastUse[#] > i &];
      ,
      unique = DeleteDuplicates[ev[[2]]];
      Do[If[!MemberQ[simStack, k], simStack = Append[simStack, k]], {k, unique}];
      If[Length[simStack] > peakWS, peakWS = Length[simStack]];
      simStack = Select[simStack, lastUse[#] > i &];
    ];
  ], {i, Length[events]}];

  (* Pass 3: replay through demand-paged LRU stack with cold-miss pricing. *)
  stack = {};
  trace = {};
  hotcold = {};
  globalCold = 0;

  Do[With[{ev = events[[i]]},
    If[ev[[1]] === "STORE",
      stack = Append[stack, ev[[2]]];
      ,
      unique = DeleteDuplicates[ev[[2]]];
      liveBefore = Length[stack];
      dMap = Association[];
      hcMap = Association[];
      Do[pos = FirstPosition[stack, k];
        If[MissingQ[pos],
          globalCold += 1;
          dMap[k] = peakWS + globalCold;
          hcMap[k] = True,
          dMap[k] = liveBefore - pos[[1]] + 1;
          hcMap[k] = False
        ],
        {k, unique}];
      (* Emit one depth per operand in READ event order. *)
      Do[
        AppendTo[trace, dMap[k]];
        AppendTo[hotcold, hcMap[k]],
        {k, ev[[2]]}];
      (* Move touched keys to top; cold keys get appended. *)
      stack = Join[
        DeleteCases[stack, Alternatives @@ unique],
        Select[unique, !MissingQ[FirstPosition[stack, #]] &],
        Select[unique, MissingQ[FirstPosition[stack, #]] &]
      ];
      (* Actually simpler: remove all unique from stack, append all unique *)
      stack = Join[DeleteCases[stack, Alternatives @@ unique], unique];
    ];
    (* Compaction: evict keys whose last-use index has passed. *)
    stack = Select[stack, lastUse[#] > i &];
  ], {i, Length[events]}];

  {trace, hotcold}
];


(* --- Run for N = 2, 3, 4 --- *)
Do[Module[{result, tr, hc},
  result = MatVecColdTrace[n];
  tr = result[[1]];
  hc = result[[2]];
  Print["MatVec ", n, "x", n, " cold depth trace (", Length[tr], " reads):"];
  Print["  depths:  ", tr];
  Print["  hot/cold: ", hc];
  Print["  total ByteDMD cost: ", Total[Ceiling[Sqrt[tr]]]];
  Print[""];
], {n, {2, 3, 4}}];
