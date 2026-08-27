#!/usr/bin/env python3
"""Merge all .md files in the gemini/ directory into a single chronological file."""

import os
import sys

def merge_md_files(directory, output_filename="gemini-merged.md"):
    """Read all .md files in directory, sort by mtime, write merged output."""
    md_files = []
    for f in os.listdir(directory):
        if not f.endswith(".md"):
            continue
        if f == output_filename:
            continue
        path = os.path.join(directory, f)
        if os.path.isfile(path):
            md_files.append((os.path.getmtime(path), f, path))

    md_files.sort()  # oldest first by mtime

    out_path = os.path.join(directory, output_filename)
    with open(out_path, "w") as out:
        out.write("# Gemini Notes — Merged Chronologically\n\n")
        for i, (mtime, name, path) in enumerate(md_files):
            out.write(f"---\n\n## {name}\n\n")
            with open(path) as f:
                out.write(f.read())
            out.write("\n\n")

    print(f"Merged {len(md_files)} files into {out_path}")
    for _, name, _ in md_files:
        print(f"  {name}")


if __name__ == "__main__":
    directory = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
    merge_md_files(directory)
