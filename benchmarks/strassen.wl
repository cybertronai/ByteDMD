#!/usr/bin/env wolframscript
(* Exact ByteDMD trace cost for Strassen's algorithm (leaf size 1)
   under the demand-paged, two-pass aggressive liveness LRU model.

   Source: gemini/strassen-11apr26.md (compiled-engine variant)

   Implementation notes:
     - Pass 1 symbolically executes Strassen on matrices of logical IDs
       and records READ/STORE events as {a, b} / {-1, c} integer pairs.
     - Pass 2 (forward liveness) is a single O(E) array sweep.
     - Passes 2+3 are merged into a single O(E log E) compiled routine
       (CompileLRUByteDMD) that uses a Fenwick tree (bit array) keyed on
       an ever-increasing timestamp so move-to-top and depth queries are
       both O(log E). Items are evicted inline when their last-use index
       is reached — no post-pass compaction loop needed.
     - Pass 4 applies the integer ceil-sqrt cost to each trace entry via
       a Listable compiled function.

   The compiled engine is fast enough to run N in {2, 4, 8, 16} within
   a few seconds on modern hardware.

   Run:
     ./benchmarks/strassen.wl
     wolframscript benchmarks/strassen.wl
*)

ClearAll[CompileLRUByteDMD, ByteDMDSqrtSum, ByteDMDStrassen];

(* Extremely fast Compiled Engine merging Pass 2 & Pass 3 into an
   O(E log E) execution via a Fenwick tree keyed on timestamps. *)
CompileLRUByteDMD =
  Compile[{{events, _Integer, 2}, {lastUse, _Integer, 1}, {counter, _Integer}},
   Module[{bit, bitSize, timeCounter, lastAccess, nEvents, a, b, c,
     numElements, coldCount, depthA, depthB, resTrace, traceIdx,
     unique1, unique2, numUnique, sum, idx},
    nEvents = Length[events];
    (* Timeline max increments bounded safely at 3x ops *)
    bitSize = 3*nEvents + 10;
    bit = Table[0, {bitSize}];
    lastAccess = Table[0, {counter}];
    resTrace = Table[0, {2*nEvents}];
    traceIdx = 0;
    timeCounter = 0;
    numElements = 0;
    Do[
     If[events[[i, 1]] == -1,
      (* --- STORE OPERATION --- *)
      c = events[[i, 2]];
      timeCounter++;
      lastAccess[[c]] = timeCounter;
      idx = timeCounter;
      While[idx <= bitSize, bit[[idx]] += 1; idx += BitAnd[idx, -idx];];
      numElements++;
      (* Apply Compaction: Exclude Dead Elements instantly on limits *)
      If[lastUse[[c]] == i,
       idx = lastAccess[[c]];
       While[idx <= bitSize, bit[[idx]] -= 1; idx += BitAnd[idx, -idx];];
       lastAccess[[c]] = 0;
       numElements--;
      ];
      ,
      (* --- READ OPERATION --- *)
      a = events[[i, 1]];
      b = events[[i, 2]];
      If[a == b,
       unique1 = a; numUnique = 1; unique2 = -1;,
       unique1 = a; unique2 = b; numUnique = 2;
      ];
      coldCount = 0;
      If[lastAccess[[unique1]] > 0,
       sum = 0;
       idx = lastAccess[[unique1]];
       While[idx > 0, sum += bit[[idx]]; idx -= BitAnd[idx, -idx];];
       depthA = 1 + numElements - sum;,
       coldCount++;
       depthA = numElements + coldCount;
      ];
      If[numUnique == 2,
       If[lastAccess[[unique2]] > 0,
        sum = 0;
        idx = lastAccess[[unique2]];
        While[idx > 0, sum += bit[[idx]]; idx -= BitAnd[idx, -idx];];
        depthB = 1 + numElements - sum;,
        coldCount++;
        depthB = numElements + coldCount;
       ];,
       depthB = depthA;
      ];
      traceIdx++; resTrace[[traceIdx]] = depthA;
      traceIdx++; resTrace[[traceIdx]] = depthB;
      (* Bump all simultaneously accessed items cleanly to the LRU top
         via timestamps *)
      If[lastAccess[[unique1]] > 0,
       idx = lastAccess[[unique1]];
       While[idx <= bitSize, bit[[idx]] -= 1; idx += BitAnd[idx, -idx];];,
       numElements++;
      ];
      timeCounter++;
      lastAccess[[unique1]] = timeCounter;
      idx = timeCounter;
      While[idx <= bitSize, bit[[idx]] += 1; idx += BitAnd[idx, -idx];];
      If[numUnique == 2,
       If[lastAccess[[unique2]] > 0,
        idx = lastAccess[[unique2]];
        While[idx <= bitSize, bit[[idx]] -= 1; idx += BitAnd[idx, -idx];];,
        numElements++;
       ];
       timeCounter++;
       lastAccess[[unique2]] = timeCounter;
       idx = timeCounter;
       While[idx <= bitSize, bit[[idx]] += 1; idx += BitAnd[idx, -idx];];
      ];
      (* Re-apply Compaction: Evict dead keys natively passing
         explicit read boundaries *)
      If[lastUse[[unique1]] == i,
       idx = lastAccess[[unique1]];
       While[idx <= bitSize, bit[[idx]] -= 1; idx += BitAnd[idx, -idx];];
       lastAccess[[unique1]] = 0;
       numElements--;
      ];
      If[numUnique == 2,
       If[lastUse[[unique2]] == i,
        idx = lastAccess[[unique2]];
        While[idx <= bitSize, bit[[idx]] -= 1; idx += BitAnd[idx, -idx];];
        lastAccess[[unique2]] = 0;
        numElements--;
       ];
      ];
     ];
    , {i, 1, nEvents}];
    Take[resTrace, traceIdx]
   ],
   RuntimeOptions -> "Speed"
  ];

(* Compiled scalar evaluations strictly replacing mapped functions *)
ByteDMDSqrtSum =
  Compile[{{x, _Integer}},
   If[x <= 0, 0,
    Module[{m = Floor[Sqrt[x - 1]] + 1},
     Quotient[m*(6*x - 2*m^2 + 3*m - 1), 6]]],
   RuntimeOptions -> "Speed", RuntimeAttributes -> {Listable}
  ];


ByteDMDStrassen[Nsize_Integer, bytesPerElement_Integer : 1] :=
 Module[{counter = 0, newID, addOp, subOp, mulOp, addMat, subMat,
   strassen, A, B, res, evReap, eventList, lastUseArr, trace, nEvents},
  If[BitAnd[Nsize, Nsize - 1] != 0,
   Print["Nsize must be a power of 2."]; Return[$Failed]];
  newID[] := ++counter;
  (* Pure Integer Encoding mapping rules (Replaced strings to natively
     pack memory) *)
  addOp[a_, b_] :=
   Module[{c}, Sow[{a, b}, "ev"]; c = newID[]; Sow[{-1, c}, "ev"]; c];
  subOp[a_, b_] := addOp[a, b];
  mulOp[a_, b_] := addOp[a, b];
  addMat[X_List, Y_List] :=
   Table[addOp[X[[i, j]], Y[[i, j]]], {i, Length[X]}, {j, Length[X]}];
  subMat[X_List, Y_List] :=
   Table[subOp[X[[i, j]], Y[[i, j]]], {i, Length[X]}, {j, Length[X]}];
  strassen[X_List, Y_List] :=
   Module[{n = Length[X], mid, X11, X12, X21, X22, Y11, Y12, Y21, Y22,
      M1, M2, M3, M4, M5, M6, M7, C11, C12, C21, C22},
    If[n == 1, Return[{{mulOp[X[[1, 1]], Y[[1, 1]]]}}]];
    mid = Quotient[n, 2];
    X11 = X[[1 ;; mid, 1 ;; mid]]; X12 = X[[1 ;; mid, mid + 1 ;; n]];
    X21 = X[[mid + 1 ;; n, 1 ;; mid]];
    X22 = X[[mid + 1 ;; n, mid + 1 ;; n]];
    Y11 = Y[[1 ;; mid, 1 ;; mid]]; Y12 = Y[[1 ;; mid, mid + 1 ;; n]];
    Y21 = Y[[mid + 1 ;; n, 1 ;; mid]];
    Y22 = Y[[mid + 1 ;; n, mid + 1 ;; n]];
    M1 = strassen[addMat[X11, X22], addMat[Y11, Y22]];
    M2 = strassen[addMat[X21, X22], Y11];
    M3 = strassen[X11, subMat[Y12, Y22]];
    M4 = strassen[X22, subMat[Y21, Y11]];
    M5 = strassen[addMat[X11, X12], Y22];
    M6 = strassen[subMat[X21, X11], addMat[Y11, Y12]];
    M7 = strassen[subMat[X12, X22], addMat[Y21, Y22]];
    C11 = addMat[subMat[addMat[M1, M4], M5], M7];
    C12 = addMat[M3, M5];
    C21 = addMat[M2, M4];
    C22 = addMat[addMat[subMat[M1, M2], M3], M6];
    ArrayFlatten[{{C11, C12}, {C21, C22}}]];

  (* PASS 1: Initialize Inputs & Generate Instructions *)
  {res, evReap} =
   Reap[A = Table[newID[], {Nsize}, {Nsize}];
    B = Table[newID[], {Nsize}, {Nsize}];
    strassen[A, B], "ev"];
  eventList = If[Length[evReap] == 0, {}, evReap[[1]]];
  nEvents = Length[eventList];
  If[nEvents == 0, Return[0]];

  (* PASS 2: Establish Forward Liveness Boundaries (Pre-allocation
     Array Vectorization) *)
  lastUseArr = ConstantArray[0, counter];
  Do[If[eventList[[i, 1]] == -1, lastUseArr[[eventList[[i, 2]]]] = i,
    lastUseArr[[eventList[[i, 1]]]] = i;
    lastUseArr[[eventList[[i, 2]]]] = i], {i, 1, nEvents}];
  Scan[(lastUseArr[[#]] = nEvents + 1) &, Flatten[res]];

  (* PASS 3: Route pure operations to WVM Compiled Backend *)
  trace = CompileLRUByteDMD[eventList, lastUseArr, counter];

  (* PASS 4: Closed-Form Evaluations *)
  If[bytesPerElement == 1, Total[Floor[Sqrt[trace - 1]] + 1],
   Total[ByteDMDSqrtSum[trace*bytesPerElement] -
     ByteDMDSqrtSum[(trace - 1)*bytesPerElement]]]
 ]


(* Print the cost table *)
Print["Strassen (leaf=1) ByteDMD cost:"];
Print[""];
Print["   N | ByteDMD"];
Print["-----|---------"];
Do[
  Print[
    "  " <> StringPadLeft[ToString[n], 2] <> " | " <>
    StringPadLeft[ToString[ByteDMDStrassen[n]], 7]
  ],
  {n, {2, 4, 8, 16}}
];
