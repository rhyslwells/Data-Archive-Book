import os
import re
from pathlib import Path

# ── folders you want to process ────────────────────────────────────────────────
GROUPED_DIR = Path("./grouped")          # <‑‑ adjust if your path differs
# -----------------------------------------------------------------------------

# pattern:  # Title {#anchor}
h1_with_anchor = re.compile(r'^(# )(.+?)\s*\{#([A-Za-z0-9_\-]+)\}', re.MULTILINE)

def inject_html_anchor(md_text: str) -> str:
    """
    Insert <a id="anchor"></a> on the line *just before* each H1 that already
    has a {#anchor}.  Skip if the anchor is already present.
    """
    def replacer(match: re.Match) -> str:
        full_line = match.group(0)
        anchor_id = match.group(3)
        # If the <a id="…"> is already directly above, leave unchanged
        prev_line_break = md_text.rfind('\n', 0, match.start())
        prev_line_start = prev_line_break + 1
        if md_text[prev_line_start:match.start()].strip() == f'<a id="{anchor_id}"></a>':
            return full_line          # anchor already in place
        return f'<a id="{anchor_id}"></a>\n{full_line}'
    # process the whole file
    return h1_with_anchor.sub(replacer, md_text)

def process_file(md_path: Path) -> None:
    text = md_path.read_text(encoding="utf‑8")
    new_text = inject_html_anchor(text)
    if new_text != text:
        md_path.write_text(new_text, encoding="utf‑8")
        print(f"✓ patched {md_path.relative_to(GROUPED_DIR)}")

# ── run over all markdown files ────────────────────────────────────────────────
for md_file in GROUPED_DIR.rglob("*.md"):
    process_file(md_file)

print("✅ All anchors injected.")
