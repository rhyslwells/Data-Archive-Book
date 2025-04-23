
- Table of contents in epub
- HonKit’s EPUB output uses the first # heading in a Markdown file as the chapter title in the TOC. Subsequent # headings are not included unless you split the file.
- Cant seperate into TOC, need local, otherwise need to split into separate files. Long build time if so. 


Recommended Compromise: Use ## but auto-inject a nav TOC
If you can't split the files, the best alternative is to:

Keep a single # title per file (e.g., A).

Use ## for each topic (e.g., ## AB Testing {#ab-testing}).

At the top of each grouped file, inject a local TOC linking to subheadings.

Example generated top of A.md:
md
Copy code
# A {#a}

## Contents

- [AB Testing](#ab-testing)
- [Acid Transaction](#acid-transaction)
- [AI Governance](#ai-governance)

---

## AB Testing {#ab-testing}
...




To ensure the **Table of Contents (TOC)** is only inserted once at the top of each **grouped file** (like `A.md`, `B.md`, `C.md`, etc.) and not repeated in every individual file, you need to modify the logic slightly. The TOC should be dynamically generated and inserted at the top of each grouped letter file (e.g., `A.md`, `B.md`, `C.md`) based on the contents of that particular group.

### Approach:
1. **Generate the TOC** for each letter group (A, B, C, etc.) from the files in that group.
2. **Insert the TOC** at the top of the corresponding grouped file (e.g., `A.md`, `B.md`, `C.md`), only once.
3. **Do not insert the TOC inside individual files**; instead, build the TOC for the entire group and place it at the top of the final file for each group.

### Updated Code:

```python
import os
import re
from collections import defaultdict

content_folder = './content'
output_folder = './grouped'
os.makedirs(output_folder, exist_ok=True)

# YAML front matter stripper
def strip_yaml_front_matter(text):
    if text.startswith('---'):
        end = text.find('---', 3)
        if end != -1:
            return text[end + 3:].lstrip()
    return text

# ==highlight== → <mark>
def convert_highlights(content):
    return re.sub(r'==(.+?)==', r'<mark>\1</mark>', content)

# Slugify function for anchors
def slugify(text):
    return re.sub(r'[^\w\- ]', '', text).strip().lower().replace(' ', '-')

# Regex patterns
main_header_pattern = re.compile(r'^# (.+?)\s*\{#(.+?)\}', re.MULTILINE)
sub_header_pattern = re.compile(r'^## (.+?)\s*\{#(.+?)\}', re.MULTILINE)

# Group files
grouped_files = defaultdict(list)

# This will hold the final Table of Contents (TOC) for each grouped letter (A, B, C, etc.)
final_toc = defaultdict(list)

for file in sorted(os.listdir(content_folder)):
    if file.endswith('.md') and file.lower() != 'readme.md':
        first_letter = file[0].upper()
        file_path = os.path.join(content_folder, file)

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            content = strip_yaml_front_matter(content)
            content = convert_highlights(content)

            anchor = ""
            match = main_header_pattern.search(content)
            if match:
                anchor = match.group(2)

            # Extract subheadings with anchors
            subheadings = sub_header_pattern.findall(content)

            # For each file, add its name and subheadings to the final TOC for this letter group
            file_toc_entries = []
            for sub_title, sub_anchor in subheadings:
                file_toc_entries.append(f"- [{sub_title}](#{sub_anchor})")
            final_toc[first_letter].append(f"* [{os.path.splitext(file)[0]}](./grouped/{first_letter}.md#{anchor})")
            final_toc[first_letter].extend(file_toc_entries)

            grouped_files[first_letter].append((file, content, anchor, subheadings))

# Write grouped markdown files
for letter, entries in sorted(grouped_files.items()):
    out_path = os.path.join(output_folder, f"{letter}.md")
    with open(out_path, 'w', encoding='utf-8') as out_file:
        # Insert the TOC for this particular group of files at the top
        out_file.write("## Table of Contents\n")
        out_file.write("\n".join(final_toc[letter]) + "\n\n")

        # Write the content from all files in this group
        for filename, cleaned_content, _, _ in entries:
            out_file.write(f"\n\n{cleaned_content}")
    print(f"✅ Wrote {out_path} with {len(entries)} files")

# Generate SUMMARY.md (flattened entries for EPUB TOC compatibility)
summary_path = 'SUMMARY.md'
with open(summary_path, 'w', encoding='utf-8') as summary:
    summary.write("# Summary\n\n")
    for letter in sorted(grouped_files.keys()):
        for filename, _, anchor, _ in grouped_files[letter]:
            display_name = os.path.splitext(filename)[0]
            if anchor:
                summary.write(f"* [{display_name}](./grouped/{letter}.md#{anchor})\n")
            else:
                summary.write(f"* [{display_name}](./grouped/{letter}.md)\n")

print("📘 Created grouped content with a single TOC for each letter and flattened SUMMARY.md for EPUB TOC.")
```

### Key Updates:
1. **Dynamic TOC Generation for Each Group (A, B, C, etc.)**:
   - The **TOC is generated per letter group** (i.e., for `A.md`, `B.md`, `C.md`, etc.), based on the files in that specific group.
   - For each group (e.g., `A.md`), a TOC is generated and placed at the top of that file, listing all the subheadings from all files in that group.

2. **TOC Insertion**:
   - The TOC is inserted **only once at the top** of the grouped file for each letter (like `A.md`, `B.md`, etc.), so no repeated TOCs are included.

3. **Group Handling**:
   - Files are grouped by the first letter of their name (e.g., all files starting with "A" go into `A.md`), and the TOC for that group is inserted at the top.
   
### Example Output:

For example, the `A.md` file will look like this:

```markdown
## Table of Contents
* [Ab Testing](./grouped/A.md#ab-testing)
- [A/B Testing Introduction](#ab-testing)
* [Acid Transaction](./grouped/A.md#acid-transaction)
- [ACID Properties](#acid-properties)

# Ab Testing {#ab-testing}
A/B testing is a method of performance testing two versions of a product like an app.

# Acid Transaction {#acid-transaction}
ACID transactions are a set of properties that guarantee database transactions are processed reliably.
```

Similarly, `B.md`, `C.md`, and other grouped files will have their own unique TOC based on the files grouped under each letter.

### Result:
- **Each grouped file** (e.g., `A.md`, `B.md`, `C.md`) will have its own **TOC** at the top, listing all the relevant subheadings for that particular group.
- **No TOC** will be inserted in the individual files; instead, it will be dynamically generated and added to the final grouped file for that letter group.

This should meet the requirement of having a TOC **only once** per grouped file, with no repetition.