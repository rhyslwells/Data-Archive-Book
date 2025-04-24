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

# --- write flattened SUMMARY.md that targets book/book.md --------------------
summary_path = 'SUMMARY.md'
with open(summary_path, 'w', encoding='utf-8') as sm:
    sm.write("# Summary\n\n")
    for letter in sorted(grouped_files):
        for filename, _, anchor, _ in grouped_files[letter]:
            title = Path(filename).stem.replace('-', ' ')
            if anchor:
                sm.write(f"* [{title}](./book/book.md#{anchor})\n")
            else:
                sm.write(f"* [{title}](./book/book.md)\n")
print("✓ SUMMARY.md written (links into book/book.md)")



print("🎉 compiler.py completed.")
