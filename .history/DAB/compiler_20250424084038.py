#!/usr/bin/env python
from pathlib import Path
import re, sys, subprocess
from collections import defaultdict
import os

ROOT         = Path(__file__).parent.resolve()
GROUPED_DIR  = ROOT / "grouped"
BOOK_DIR     = ROOT / "book"
BOOK_MD      = BOOK_DIR / "book.md"
SUMMARY_MD   = ROOT / "SUMMARY.md"

BOOK_DIR.mkdir(exist_ok=True)

H1_ANCHOR = re.compile(r'^# (.+?)\s*\{#([A-Za-z0-9_\-]+)\}', re.MULTILINE)
H2_TOC    = re.compile(r'^## Table of Contents.*?(?=^# )', re.S | re.M)

grouped_files = defaultdict(list)   # letter → list of (filename, content, anchor, subheadings)
grouped_text  = defaultdict(list)   # for book.md assembly: letter → list of full md content

# == Read and strip per-letter TOCs ============================================
for md in sorted(GROUPED_DIR.glob("[A-Z].md"), key=lambda p: p.name):
    letter = md.stem
    content = md.read_text(encoding='utf-8')
    cleaned = H2_TOC.sub('', content, count=1).lstrip()

    match = H1_ANCHOR.search(cleaned)
    anchor = match.group(2) if match else ''

    grouped_files[letter].append((md.name, cleaned, anchor, []))
    grouped_text[letter].append(cleaned)

# == Write book/book.md without TOCs ===========================================
parts = []
for letter in sorted(grouped_text.keys()):
    parts.append(f"# {letter} {{#{letter.lower()}}}")
    for content in grouped_text[letter]:
        parts.append(content)

BOOK_MD.write_text("\n\n".join(parts), encoding='utf-8')
print(f"✓ Merged content written to {BOOK_MD.relative_to(ROOT)}")

# == Write grouped-style SUMMARY.md ============================================
with SUMMARY_MD.open('w', encoding='utf-8') as summary:
    summary.write("# Summary\n\n")
    for letter in sorted(grouped_files.keys()):
        summary.write(f"* [{letter}](./grouped/{letter}.md)\n")
        for filename, _, anchor, _ in grouped_files[letter]:
            display_name = os.path.splitext(filename)[0].replace('-', ' ')
            if anchor:
                summary.write(f"  * [{display_name}](./grouped/{letter}.md#{anchor})\n")
print(f"✓ Grouped SUMMARY.md written to {SUMMARY_MD.relative_to(ROOT)}")

# == Optional: HonKit EPUB/PDF Build ===========================================
if len(sys.argv) > 1 and sys.argv[1].lower() in {"epub", "pdf"}:
    fmt = sys.argv[1].lower()
    out = f"data-archive-book.{fmt}"
    print(f"▶ HonKit {fmt.upper()} build …")
    subprocess.run(["npx", "honkit", fmt, "./", out], check=True)
    print(f"✓ {out} created")

print("🎉 compiler.py completed.")
