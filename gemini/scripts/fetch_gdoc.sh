#!/bin/bash
# Fetch a Google Doc as markdown and clean up math so GitHub renders it.
#
# Usage:
#   ./fetch_gdoc.sh <gdoc-url> <output-path>
#
# The URL can be any Google Docs link (edit, view, share). The doc must be
# accessible to the curl client (link-shared "Anyone with link can view"
# typically works without auth).

set -euo pipefail

if [ $# -ne 2 ]; then
  echo "usage: $0 <gdoc-url> <output-path>" >&2
  exit 1
fi

URL="$1"
OUT="$2"

DOC_ID="$(echo "$URL" | sed -nE 's|.*/document/d/([^/?#]+).*|\1|p')"
if [ -z "$DOC_ID" ]; then
  echo "could not extract doc id from URL: $URL" >&2
  exit 1
fi

EXPORT_URL="https://docs.google.com/document/d/${DOC_ID}/export?format=md"
HERE="$(cd "$(dirname "$0")" && pwd)"

curl -sL "$EXPORT_URL" | python3 "$HERE/fix_gdoc_math.py" --stdin > "$OUT"
LINES="$(wc -l < "$OUT")"
echo "wrote $OUT ($LINES lines)"
