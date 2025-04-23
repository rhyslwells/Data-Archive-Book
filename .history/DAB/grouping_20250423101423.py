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

# Group files
grouped_files = defaultdict(list)

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

            grouped_files[first_letter].append((file, content, anchor))

# Write grouped markdown files
for letter, entries in sorted(grouped_files.items()):
    out_path = os.path.join(output_folder, f"{letter}.md")
    with open(out_path, 'w', encoding='utf-8') as out_file:
        for _, cleaned_content, _ in entries:
            out_file.write(f"\n\n{cleaned_content}")
    print(f"✅ Wrote {out_path} with {len(entries)} files")

# Generate SUMMARY.md (flattened entries for EPUB TOC compatibility)
summary_path = 'SUMMARY.md'
with open(summary_path, 'w', encoding='utf-8') as summary:
    summary.write("# Summary\n\n")
    for letter in sorted(grouped_files.keys()):
        for filename, _, anchor in grouped_files[letter]:
            display_name = os.path.splitext(filename)[0]
            if anchor:
                summary.write(f"* [{display_name}](./grouped/{letter}.md#{anchor})\n")
            else:
                summary.write(f"* [{display_name}](./grouped/{letter}.md)\n")

print("📘 Created grouped content and flattened SUMMARY.md for EPUB TOC.")
