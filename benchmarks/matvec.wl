#!/usr/bin/env wolframscript
(* ::Package:: *)

(* Exact O(1) closed-form ByteDMD cost for N x N matvec under the
   demand-paged LRU model with aggressive liveness analysis.

   Source: https://gemini.google.com/app/003aa0f1bd759c2b

   Run directly:
     ./experiments/matvec_analysis/matvec_closed_form.wls
     wolframscript experiments/matvec_analysis/matvec_closed_form.wls
*)

(* 1. Fundamental Cost Functions *)
S[d_] := Ceiling[Sqrt[d]];

(* F[d] is the exact closed-form of Sum[S[k], {k, 1, d}] *)
F[d_] := Module[{M = S[d]}, (M*(6 d - 2 M^2 + 3 M - 1))/6];

(* 2. O(1) Closed-Form Formula *)
MatVecClosedForm[1] = 3;
MatVecClosedForm[N_Integer] /; N >= 2 :=
  N*F[2 N] +
   F[2 N - 1] - (N - 2)*F[N + 2] + (N - 1)*S[N + 1] + (N - 1)*(N - 2)*
    S[N + 2] + S[2 N] + 3 N^2 - 3 N - 8;

(* Print the cost table *)
Print["Naive N x N matvec ByteDMD cost:"];
Print[""];
Print["   N | ByteDMD"];
Print["-----|---------"];
Do[
  Print[
    "  " <> StringPadLeft[ToString[n], 2] <> " | " <>
    StringPadLeft[ToString[MatVecClosedForm[n]], 7]
  ],
  {n, {2, 4, 8, 16}}
];
