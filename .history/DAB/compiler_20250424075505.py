"""
compiler.py
-----------
• merges grouped/*.md → book/book.md   (drops local '## Table of Contents')
• writes grouped SUMMARY.md that links to ./book/book.md#anchors
• optional: `python compiler.py epub` or `python compiler.py pdf`
"""

from pathlib import Path
import re, sys, subprocess
from collections import defaultdict

ROOT        = Path(__file__).parent.resolve()
GROUPED_DIR = ROOT / "grouped"
BOOK_DIR    = ROOT / "book"
BOOK_MD     = BOOK_DIR / "book.md"
SUMMARY_MD  = ROOT / "SUMMARY.md"

BOOK_DIR.mkdir(exist_ok=True)

# --- regex helpers -----------------------------------------------------------
H1_ANCHOR   = re.compile(r'^# (.+?)\s*\{#([A-Za-z0-9_\-]+)\}', re.MULTILINE)
H2_TOC      = re.compile(r'^## Table of Contents.*?(?=^# )', re.S | re.M)

# --- collect grouped files ---------------------------------------------------
grouped: defaultdict[str, list] = defaultdict(list)   # letter -> [(title, anchor, text)]

for md in sorted(GROUPED_DIR.glob("[A-Z].md"), key=lambda p: p.name):
    letter = md.stem
    txt = md.read_text(encoding="utf-8")

    # strip entire local TOC block
    txt = H2_TOC.sub('', txt, count=1).lstrip()

    # first H1 anchor
    m = H1_ANCHOR.search(txt)
    if not m:
        continue
    anchor = m.group(2)
    title  = m.group(1)

    grouped[letter].append((title, anchor, txt))

# --- create book/book.md -----------------------------------------------------
parts = ["# Book Table of Contents\n"]
for letter in sorted(grouped):
    parts.append(f"* [{letter}](#{letter.lower()})")
parts.append("")

for letter in sorted(grouped):
    parts.append(f"# {letter} {{#{letter.lower()}}}")
    for _, _, txt in grouped[letter]:
        parts.append(txt)

BOOK_MD.write_text("\n\n".join(parts), encoding="utf-8")
print(f"✓ merged book written → {BOOK_MD.relative_to(ROOT)}")

# --- write grouped SUMMARY.md ------------------------------------------------
with SUMMARY_MD.open('w', encoding='utf-8') as sm:
    sm.write("# Summary\n\n")
    for letter in sorted(grouped):
        sm.write(f"* [{letter}](./book/book.md#{letter.lower()})\n")
        for title, anchor, _ in grouped[letter]:
            sm.write(f"  * [{title}](./book/book.md#{anchor})\n")
print("✓ SUMMARY.md written")

# --- optional HonKit ---------------------------------------------------------
if len(sys.argv) > 1 and sys.argv[1].lower() in {"epub", "pdf"}:
    fmt = sys.argv[1].lower()
    out = f"data-archive-book.{fmt}"
    cmd = ["npx", "honkit", fmt, "./", out]
    print("▶ HonKit", fmt.upper(), "build …")
    subprocess.run(cmd, check=True)
    print(f"✓ {out} created")

print("🎉 compiler.py finished.")
