#!/usr/bin/env python3
"""Convert all .md files in the gemini directory to HTML files in gemini/HTML/."""

import glob
import os
import markdown

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "HTML")
os.makedirs(OUTPUT_DIR, exist_ok=True)

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; line-height: 1.6; color: #333; }}
  pre {{ background: #f6f8fa; padding: 16px; border-radius: 6px; overflow-x: auto; }}
  code {{ background: #f6f8fa; padding: 2px 6px; border-radius: 3px; font-size: 0.9em; }}
  pre code {{ background: none; padding: 0; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
  th {{ background: #f6f8fa; }}
  img {{ max-width: 100%; }}
  blockquote {{ border-left: 4px solid #ddd; margin: 0; padding-left: 16px; color: #666; }}
</style>
</head>
<body>
{body}
</body>
</html>"""

md_extensions = ["tables", "fenced_code", "codehilite", "toc", "nl2br"]

md_files = glob.glob(os.path.join(SCRIPT_DIR, "*.md"))
converted = 0

for md_path in sorted(md_files):
    filename = os.path.basename(md_path)
    html_filename = os.path.splitext(filename)[0] + ".html"
    html_path = os.path.join(OUTPUT_DIR, html_filename)

    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()

    html_body = markdown.markdown(md_content, extensions=md_extensions)
    title = os.path.splitext(filename)[0].replace("-", " ").title()
    html_full = HTML_TEMPLATE.format(title=title, body=html_body)

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_full)

    converted += 1
    print(f"  {filename} -> HTML/{html_filename}")

print(f"\nConverted {converted} files to HTML in {OUTPUT_DIR}")
