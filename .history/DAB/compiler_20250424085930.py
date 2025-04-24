#!/usr/bin/env python
import os, re
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).parent.resolve()
CONTENT_DIR   = ROOT / "content"
BOOK_DIR      = ROOT / "book"
BOOK_MD       = BOOK_DIR / "book.md"
SUMMARY_MD    = ROOT / "SUMMARY.md"

BOOK_DIR.mkdir(exist_ok=True)

# ── regex helpers ─────────────────────────────────────────────────────────────
YAML_BLOCK = re.compile(r'^---.*?^---\s*', re.S)              # front‑matter
HIGHLIGHT  = re.compile(r'==(.+?)==')                         # ==mark==
H1_ANCHOR  = re.compile(r'^# (.+?)\s*\{#([A-Za-z0-9_\-]+)\}', re.M)

def strip_yaml(text:str) -> str:
    return YAML_BLOCK.sub('', text, 1).lstrip()

def convert_mark(text:str) -> str:
    return HIGHLIGHT.sub(r'<mark>\1</mark>', text)

# ── process files & build book ------------------------------------------------
merged_parts   = []
summary_lines  = ["# Summary\n"]

for md_path in sorted(CONTENT_DIR.glob("*.md"), key=lambda p: p.name.lower()):
    if md_path.name.lower() == "readme.md":
        continue

    raw   = md_path.read_text(encoding="utf-8")
    text  = convert_mark(strip_yaml(raw))

    # find first H1 with existing {#anchor}; if absent, make one
    m = H1_ANCHOR.search(text)
    if m:
        h1_title, anchor = m.group(1), m.group(2)
    else:
        # fabricate heading & anchor from file name
        h1_title = md_path.stem.replace('-', ' ')
        anchor   = re.sub(r'[^\w\- ]','', h1_title).lower().replace(' ', '-')
        text     = f"# {h1_title} {{#{anchor}}}\n\n{text}"

    merged_parts.append(text)
    summary_lines.append(f"* [{h1_title}](./book/book.md#{anchor})")

# write merged book
BOOK_MD.write_text("\n\n".join(merged_parts), encoding="utf-8")
print(f"✓ book created → {BOOK_MD.relative_to(ROOT)}")

# write flattened summary
SUMMARY_MD.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
print(f"✓ flattened SUMMARY.md written → {SUMMARY_MD.relative_to(ROOT)}")

print("🎉 Done: merged content + flattened summary ready for HonKit EPUB/PDF.")
