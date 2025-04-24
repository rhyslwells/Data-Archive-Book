#!/usr/bin/env python
import re
from pathlib import Path

ROOT        = Path(__file__).parent.resolve()
CONTENT_DIR = ROOT / "content"
BOOK_DIR    = ROOT / "book"
BOOK_MD     = BOOK_DIR / "book.md"
SUMMARY_MD  = ROOT / "SUMMARY.md"
BOOK_DIR.mkdir(exist_ok=True)

# ── helpers ───────────────────────────────────────────────────────────────────
def strip_yaml_front_matter(text: str) -> str:
    if text.startswith('---'):
        end = text.find('---', 3)
        if end != -1:
            return text[end + 3:].lstrip()
    return text

HILITE = re.compile(r'==(.+?)==')
H1     = re.compile(r'^# (.+?)(\s*\{#([A-Za-z0-9_\-]+)\})?$', re.MULTILINE)

def slugify(txt: str) -> str:
    return re.sub(r'[^\w\- ]', '', txt).strip().lower().replace(' ', '-')

# ── process files ─────────────────────────────────────────────────────────────
merged_parts  = []
summary_lines = ["# Summary\n"]

for md in sorted(CONTENT_DIR.glob("*.md"), key=lambda p: p.name.lower()):
    if md.name.lower() == "readme.md":
        continue

    text = strip_yaml_front_matter(md.read_text(encoding='utf-8'))
    text = HILITE.sub(r'<mark>\1</mark>', text)

    m = H1.search(text)
    if m:
        title  = m.group(1).strip()
        anchor = m.group(3) or slugify(title)
        if not m.group(3):                         # add {#anchor} if missing
            text = H1.sub(f"# {title} {{#{anchor}}}", text, count=1)
    else:                                          # fabricate heading
        title  = md.stem.replace('-', ' ')
        anchor = slugify(title)
        text   = f"# {title} {{#{anchor}}}\n\n{text}"

    # ── insert invisible anchor *after* the H1 line ──────────────────────────
    text = H1.sub(lambda m: f"{m.group(0)}\n<a id=\"{anchor}\"></a>", text, count=1)

    merged_parts.append(text)
    summary_lines.append(f"* [{title}](./book/book.md#{anchor})")

# ── output files ──────────────────────────────────────────────────────────────
BOOK_MD.write_text("\n\n".join(merged_parts), encoding="utf-8")
SUMMARY_MD.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

print("✓ book/book.md and SUMMARY.md generated – YAML stripped, anchors injected *after* each heading.")
