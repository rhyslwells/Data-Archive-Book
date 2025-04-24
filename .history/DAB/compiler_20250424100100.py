#!/usr/bin/env python
import re
from pathlib import Path

ROOT         = Path(__file__).parent.resolve()
CONTENT_DIR  = ROOT / "content"
BOOK_DIR     = ROOT / "book"
BOOK_MD      = BOOK_DIR / "book.md"
SUMMARY_MD   = ROOT / "SUMMARY.md"
BOOK_DIR.mkdir(exist_ok=True)

# ---------- helpers ----------------------------------------------------------
def strip_yaml_front_matter(text: str) -> str:
    """Remove first YAML front‑matter block (--- ... ---) if present."""
    if text.startswith('---'):
        end = text.find('---', 3)
        if end != -1:
            return text[end + 3:].lstrip()
    return text

HILITE = re.compile(r'==(.+?)==')
H1     = re.compile(r'^# (.+?)(\s*\{#([A-Za-z0-9_\-]+)\})?$', re.MULTILINE)

def slugify(t: str) -> str:
    return re.sub(r'[^\w\- ]', '', t).strip().lower().replace(' ', '-')

# ---------- process ----------------------------------------------------------
merged_parts  = []
summary_lines = ["# Summary\n"]

for md in sorted(CONTENT_DIR.glob("*.md"), key=lambda p: p.name.lower()):
    if md.name.lower() == "readme.md":
        continue

    raw = md.read_text(encoding="utf-8")
    txt = strip_yaml_front_matter(raw)
    txt = HILITE.sub(r'<mark>\1</mark>', txt)

    # Process the main H1 header
    m = H1.search(txt)
    if m:
        title  = m.group(1).strip()
        anchor = m.group(3) or slugify(title)
        if not m.group(3):  # add {#anchor} if missing
            txt = H1.sub(f"# {title} {{#{anchor}}}", txt, count=1)
    else:  # fabricate heading if missing
        title  = md.stem.replace('-', ' ')
        anchor = slugify(title)
        txt    = f"# {title} {{#{anchor}}}\n\n{txt}"

    # Ensure that the header has the correct format (with anchor)
    txt = H1.sub(lambda m: f"# {m.group(1).strip()} {{#{slugify(m.group(1))}}}", txt)

    # Add to the merged content
    merged_parts.append(txt)
    summary_lines.append(f"* [{title}](./book/book.md#{anchor})")

# ---------- output -----------------------------------------------------------
BOOK_MD.write_text("\n\n".join(merged_parts), encoding="utf-8")
SUMMARY_MD.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

print("✓ book/book.md and SUMMARY.md generated – YAML headers stripped, anchors injected.")
