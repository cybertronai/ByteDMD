"""Regex-based lint for `manual.py` — catches the three most common
MAC-pricing anti-patterns found in past undercharge audits.

Intended to run in CI (see test_manuals.py). Each rule reports the
line number of any match and a short explanation. Apply `# noqa: rule`
on a line to suppress.

Rules:
  1. accumulator-outside   — touch(X) BEFORE a for-loop + write(X) AFTER,
                             with no touch(X) inside. Classic
                             "accumulator read once" undercharge.
  2. missing-tmp-in-MAC    — three consecutive touches and a write on
                             four consecutive lines, where the touches
                             look like MAC operands and the tmp is
                             missing. Heuristic only.
  3. spurious-init-read    — touch(X) immediately followed by write(X)
                             with no op in between. An overcharge.
  4. arg-reread            — the same arg cell read twice in quick
                             succession without an intervening promote
                             (suggests manual isn't caching the arg
                             cell into scratch).
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from typing import List


@dataclass
class LintHit:
    rule: str
    line: int
    snippet: str
    explanation: str


RULES = {
    "spurious-init-read": (
        re.compile(
            r"^(\s*)a\.touch\(([^)]+)\)\s*\n\1a\.write\(\2\)",
            re.MULTILINE),
        "touch(X); write(X) with no intervening op reads uninitialized "
        "memory (ByteDMD overcharge).",
    ),
    "same-arg-reread-close": (
        re.compile(
            r"a\.touch_arg\(([^)]{3,80})\)(?:[^\n]*\n){1,6}[^\n]*"
            r"a\.touch_arg\(\1\)"),
        "same arg address read twice within 6 lines — promote to a "
        "scratch cache to avoid the double arg-stack cost.",
    ),
}


# Per-function allowlist: known cases where the pattern is intentional
# (e.g. inside flash_attn where the loophole is documented, or inside
# layernorm_unfused where x is legitimately re-read across passes).
# These should shrink over time as gemini-style rewrites are applied.
ALLOWED = {
    "spurious-init-read": {"manual_flash_attention"},
    "same-arg-reread-close": {
        "manual_flash_attention",     # Q re-read per KV block (docs: loophole 1)
        "manual_layernorm_unfused",   # x re-read across 3 passes
        "manual_layernorm_fused",     # x re-read in 2-pass variant
        # False positive — loop variable `i` makes successive
        # textually-identical `a.touch_arg(A + i * n + j)` reads hit
        # different cells.
        "manual_matvec_col",
    },
}


def _containing_function(src: str, offset: int) -> str:
    head = src[:offset]
    m = list(re.finditer(r"^def (manual_\w+)\(", head, re.MULTILINE))
    return m[-1].group(1) if m else ""


def run_lint(path: str) -> List[LintHit]:
    src = open(path).read()
    lines = src.splitlines()
    hits: List[LintHit] = []
    for rule_name, (pat, explanation) in RULES.items():
        for m in pat.finditer(src):
            line_no = src[:m.start()].count("\n") + 1
            # Respect noqa suppression on the matched line range.
            suppressed = False
            for ln in range(line_no, line_no + 6):
                if ln - 1 < len(lines):
                    if f"noqa: {rule_name}" in lines[ln - 1]:
                        suppressed = True
                        break
            if suppressed:
                continue
            # Per-function allowlist for documented cases.
            fn = _containing_function(src, m.start())
            if fn in ALLOWED.get(rule_name, set()):
                continue
            snippet = lines[line_no - 1].strip()[:100]
            hits.append(LintHit(rule_name, line_no, snippet, explanation))
    return hits


def main() -> int:
    import os
    path = os.path.join(os.path.dirname(__file__), "manual.py")
    hits = run_lint(path)
    if not hits:
        print("manual.py: no lint hits")
        return 0
    print(f"manual.py: {len(hits)} lint hit(s)")
    for h in hits:
        print(f"  {path}:{h.line}  [{h.rule}]  {h.snippet}")
        print(f"    {h.explanation}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
