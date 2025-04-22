import os
import re
from collections import defaultdict

content_folder = './content'
output_folder = './grouped'
os.makedirs(output_folder, exist_ok=True)

# Define a function to strip YAML front matter
def strip_yaml_front_matter(text):
    if text.startswith('---'):
        end = text.find('---', 3)
        if end != -1:
            return text[end + 3:].lstrip()  # Remove YAML front matter
    return text

# Group files by first letter and preserve their anchor links
grouped_files = defaultdict(list)

# Regex to find headers with anchors (e.g., # Title {#title-anchor})
header_pattern = re.compile(r'^# (.+?)\s*\{#(.+?)\}', re.MULTILINE)

# Process each file in the content folder
for file in sorted(os.listdir(content_folder)):
    if file.endswith('.md') and file.lower() != 'readme.md':
        first_letter = file[0].upper()
        file_path = os.path.join(content_folder, file)

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            cleaned = strip_yaml_front_matter(content)
            # Extract header and anchors from the first level heading
            match = header_pattern.search(cleaned)
            anchor = ""
            if match:
                anchor = match.group(2)  # Get the anchor from the header
            grouped_files[first_letter].append((file, cleaned, anchor))

# Write grouped markdown files (A.md, B.md, ...)
for letter, entries in sorted(grouped_files.items()):
    out_path = os.path.join(output_folder, f"{letter}.md")
    with open(out_path, 'w', encoding='utf-8') as out_file:
        for _, cleaned_content, _ in entries:
            out_file.write(f"\n\n{cleaned_content}")
    print(f"✅ Wrote {out_path} with {len(entries)} files")

# Generate SUMMARY.md with subpoints per letter group
summary_path = 'SUMMARY.md'
with open(summary_path, 'w', encoding='utf-8') as summary:
    summary.write("# Summary\n\n")
    for letter in sorted(grouped_files.keys()):
        summary.write(f"* [{letter}](./grouped/{letter}.md)\n")
        for filename, _, anchor in grouped_files[letter]:
            display_name = os.path.splitext(filename)[0]
            # Only add anchor links if the file has an anchor
            if anchor:
                summary.write(f"  * [{display_name}](./grouped/{letter}.md#{anchor})\n")
            else:
                summary.write(f"  * [{display_name}](./content/{filename})\n")

print("📘 Created grouped content and updated SUMMARY.md.")
