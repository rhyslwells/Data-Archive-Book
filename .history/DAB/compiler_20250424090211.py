#!/usr/bin/env python
import re, os
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.resolve()
CONTENT_DIR = ROOT / "content"
BOOK_DIR    = ROOT / "book"
BOOK_MD     = BOOK_DIR / "book.md"
SUMMARY_MD  = ROOT / "SUMMARY.md"
BOOK_DIR.mkdir(exist_ok=True)

# ── regex helpers ─────────────────────────────────────────────────────────────
YAML = re.compile(r'^---.*?^---\s*', re.S)                 # front‑matter
HILITE = re.compile(r'==(.+?)==')
H1   = re.compile(r'^# (.+?)(\s*\{#([A-Za-z0-9_\-]+)\})?$', re.M)  # capture title + anchor?

def slugify(txt: str) -> str:
    return re.sub(r'[^\w\- ]', '', txt).strip().lower().replace(' ', '-')

# ── process files -------------------------------------------------------------
merged = []
summary = ["# Summary\n"]

for md in sorted(CONTENT_DIR.glob("*.md"), key=lambda p: p.name.lower()):
    if md.name.lower() == "readme.md":
        continue

    raw = md.read_text(encoding="utf-8")
    txt = HILITE.sub(r'<mark>\1</mark>', YAML.sub('', raw, 1).lstrip())

    m = H1.search(txt)
    if m:
        title = m.group(1).strip()
        anchor = m.group(3) or slugify(title)
        # ensure anchor brace exists
        if not m.group(3):
            txt = H1.sub(f"# {title} {{#{anchor}}}", txt, count=1)
    else:
        title  = md.stem.replace('-', ' ')
        anchor = slugify(title)
        txt = f"# {title} {{#{anchor}}}\n\n{txt}"

    # inject invisible anchor line *before* first H1
    h1_pos = txt.find('#')
    txt = f'<a id="{anchor}"></a>\n' + txt[h1_pos:]

    merged.append(txt)
    summary.append(f"* [{title}](./book/book.md#{anchor})")

# ── output --------------------------------------------------------------------
BOOK_MD.write_text("\n\n".join(merged), encoding="utf-8")
print(f"✓ created {BOOK_MD.relative_to(ROOT)}")

SUMMARY_MD.write_text("\n".join(summary) + "\n", encoding="utf-8")
print(f"✓ created {SUMMARY_MD.relative_to(ROOT)}")

print("🎉 Done.  Use HonKit to build EPUB/PDF:")
print("   npx honkit epub ./ ./mybook.epub")
