#!/usr/bin/env python3
r"""Fix Google-Docs-exported markdown so GitHub's MathJax can render it.

Google Docs markdown export double-escapes backslashes inside math
(`\\sqrt`, `\\sum`, `\\text{...}`), so MathJax sees escape pairs instead of
LaTeX commands and the formula renders as text. Even after unescaping,
GitHub's markdown parser strips a single `\_` to `_` inside `\text{}`,
which MathJax then rejects ("_ allowed only in math mode").

This script does, per math region only ($...$ and $$...$$):
  1. Unescape one level of backslash for ASCII punctuation.
  2. Replace any remaining `_` inside `\text{...}` with `-` to dodge
     GitHub's underscore-emphasis parser.

Usage:
  fix_gdoc_math.py FILE [FILE ...]      # rewrite in place
  fix_gdoc_math.py --stdin               # read stdin, write stdout
"""
import re
import sys

UNESC_RE = re.compile(r'\\([\\_=><.!()+\-*%&|{}])')
TEXT_UNDERSCORE_RE = re.compile(r'(\\text\{)([^}]*)(\})')


def _strip_underscore_in_text(body: str) -> str:
    def repl(m: re.Match) -> str:
        return m.group(1) + m.group(2).replace('_', '-') + m.group(3)
    return TEXT_UNDERSCORE_RE.sub(repl, body)


def _fix_math_body(body: str) -> str:
    body = UNESC_RE.sub(lambda m: m.group(1), body)
    body = _strip_underscore_in_text(body)
    return body


def fix(src: str) -> str:
    src = re.sub(
        r'\$\$\s*(.+?)\s*\$\$',
        lambda m: '$$' + _fix_math_body(m.group(1)) + '$$',
        src,
        flags=re.DOTALL,
    )
    src = re.sub(
        r'(?<!\$)\$([^\$\n]+?)\$(?!\$)',
        lambda m: '$' + _fix_math_body(m.group(1)) + '$',
        src,
    )
    src = re.sub(r'(\$\$)  +\n', r'\1\n\n', src)
    return src


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[1] == '--stdin':
        sys.stdout.write(fix(sys.stdin.read()))
        return 0
    if len(argv) < 2:
        sys.stderr.write(__doc__)
        return 1
    for path in argv[1:]:
        with open(path) as f:
            original = f.read()
        fixed = fix(original)
        if fixed != original:
            with open(path, 'w') as f:
                f.write(fixed)
            print(f'fixed: {path}')
        else:
            print(f'unchanged: {path}')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
