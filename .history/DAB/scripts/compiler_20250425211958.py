#!/usr/bin/env python
import re
from pathlib import Path

# Set ROOT to the parent directory of the current script (scripts)
ROOT        = Path(__file__).parent.resolve()  # This resolves to the 'scripts' directory
CONTENT_DIR = ROOT.parent / "content"  # Assuming the content folder is one level up
BOOK_DIR    = ROOT  # Using the current directory (scripts) for output
BOOK_MD     = BOOK_DIR / "book.md"  # book.md will be created in the 'scripts' directory
SUMMARY_MD  = ROOT.parent / "SUMMARY.md"  # SUMMARY.md will be created in the root (parent) directory
BOOK_DIR.mkdir(exist_ok=True)

# ── helpers ────────────────────────────────────────────────────────────────
def strip_yaml_front_matter(text: str) -> str:
    if text.startswith('---'):
        end = text.find('---', 3)
        if end != -1:
            return text[end + 3:].lstrip()
    return text

HILITE     = re.compile(r'==(.+?)==')
H1         = re.compile(r'^# (.+?)(\s*\{#([A-Za-z0-9_\-]+)\})?$', re.MULTILINE)

def slugify(t: str) -> str:
    return re.sub(r'[^\w\- ]', '', t).strip().lower().replace(' ', '-')

# ── Preface section ───────────────────────────────────────────────────────
preface_anchor = "preface"
preface_block = f"""# Preface {{#{preface_anchor}}}

Remember the Data Archive is non‑linear — use internal links for navigation.
If you need a specific page see the **Table of Contents** to jump to a topic.
"""

merged_parts  = [preface_block]
summary_lines = ["# Summary\n", f"* [Preface](./book.md#{preface_anchor})"]  # Updated link format

# ── process notes ─────────────────────────────────────────────────────────
for md in sorted(CONTENT_DIR.glob("*.md"), key=lambda p: p.name.lower()):
    if md.name.lower() == "readme.md":
        continue

    text = strip_yaml_front_matter(md.read_text(encoding="utf-8"))
    text = HILITE.sub(r'<mark>\1</mark>', text)

    # ─ header / anchor handling ─
    m = H1.search(text)
    if m:
        title  = m.group(1).strip()
        anchor = m.group(3) or slugify(title)
        if not m.group(3):
            text = H1.sub(f"# {title} {{#{anchor}}}", text, count=1)
    else:
        title  = md.stem.replace('-', ' ')
        anchor = slugify(title)
        text   = f"# {title} {{#{anchor}}}\n\n{text}"

    text = H1.sub(lambda m:
                  f"# {m.group(1).strip()} {{#{slugify(m.group(1))}}}",
                  text, count=1)

    merged_parts.append(text)
    summary_lines.append(f"* [{title}](./book.md#{anchor})")  # Updated link format

# ── write output ──────────────────────────────────────────────────────────
BOOK_MD.write_text("\n\n".join(merged_parts), encoding="utf-8")
SUMMARY_MD.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

print("Preface added, math fences left untouched, book.md & SUMMARY.md regenerated in scripts.")
