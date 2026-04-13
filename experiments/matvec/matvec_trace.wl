#!/usr/bin/env wolframscript
(* Simulate the ByteDMD read trace of an N x N matrix-vector multiply
   under the demand-paged, two-pass aggressive liveness LRU model that
   bytedmd.py implements.

   Matched semantics:
     - demand-paged inputs (no STORE events for A / x; cold-miss on
       first read)
     - simultaneous operand pricing per READ event
     - move-to-top after pricing
     - STORE event for every scalar multiply and scalar add
     - two-pass aggressive liveness compaction: items whose last-use
       index has been reached are evicted from the stack immediately

   Matched algorithm (standard i-j accumulator loop):
       for i in 0..N-1:
           s = A[i,0] * x[0]
           for j in 1..N-1:
               s = s + A[i,j] * x[j]
           y[i] = s

   Run:
     ./experiments/matvec/matvec_trace.wl
     wolframscript experiments/matvec/matvec_trace.wl
*)

ClearAll[MatVecTrace];

MatVecTrace[n_Integer?Positive] := Module[
  {counter = 0, newID, mulOp, addOp, A, x, y, events = {},
   lastUse, stack = {}, trace = {}, res, unique, liveBefore, dMap,
   coldSeen, pos},

  newID[] := ++counter;

  (* Every scalar multiply and every scalar add emits the same two
     events: a READ with both operands, then a STORE for the result. *)
  mulOp[a_, b_] := Module[{c = newID[]},
    AppendTo[events, {"READ", {a, b}}];
    AppendTo[events, {"STORE", c}];
    c];
  addOp[a_, b_] := Module[{c = newID[]},
    AppendTo[events, {"READ", {a, b}}];
    AppendTo[events, {"STORE", c}];
    c];

  (* Logical IDs for A (row-major) and x. These are NOT materialised
     on the stack yet — they're cold until first read. *)
  A = Table[newID[], {n}, {n}];
  x = Table[newID[], {n}];

  (* Pass 1: generate the event tape by symbolically executing matvec. *)
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

  (* Pass 3: replay the event tape through a demand-paged LRU stack. *)
  Do[With[{ev = events[[i]]},
    If[ev[[1]] === "STORE",
      stack = Append[stack, ev[[2]]];
      ,
      unique = DeleteDuplicates[ev[[2]]];
      liveBefore = Length[stack];
      dMap = Association[];
      coldSeen = 0;
      Do[pos = FirstPosition[stack, k];
        If[MissingQ[pos],
          coldSeen += 1;
          dMap[k] = liveBefore + coldSeen,
          dMap[k] = liveBefore - pos[[1]] + 1
        ],
        {k, unique}];
      (* Emit one depth per operand in the order they appear in the
         READ event (duplicates are priced against the same snapshot). *)
      Do[AppendTo[trace, dMap[k]], {k, ev[[2]]}];
      (* Move the touched keys to the top of the stack (any cold keys
         that were not there before get appended here). *)
      stack = Join[DeleteCases[stack, Alternatives @@ unique], unique];
    ];
    (* Compaction: evict keys whose last-use index has passed. *)
    stack = Select[stack, lastUse[#] > i &];
  ], {i, Length[events]}];

  trace
];


(* --- Run for N = 4 --- *)
Module[{result},
  result = MatVecTrace[4];
  Print["MatVec 4x4 depth trace (", Length[result], " reads):"];
  Print[result];
  Print[""];
  Print["Total ByteDMD cost: ", Total[Ceiling[Sqrt[result]]]];
];
