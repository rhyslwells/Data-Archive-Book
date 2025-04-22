import os
import re
from collections import defaultdict

content_folder = './content'
output_folder = './grouped'
os.makedirs(output_folder, exist_ok=True)

def strip_yaml_front_matter(text):
    if text.startswith('---'):
        end = text.find('---', 3)
        if end != -1:
            return text[end+3:].lstrip()
    return text

# Group file contents by first letter
grouped_files = defaultdict(list)
for file in sorted(os.listdir(content_folder)):
    if file.endswith('.md') and file.lower() != 'readme.md':
        first_letter = file[0].upper()
        file_path = os.path.join(content_folder, file)
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            cleaned = strip_yaml_front_matter(content)
            grouped_files[first_letter].append((file, cleaned))

# Write grouped markdown files (A.md, B.md, ...)
for letter, entries in sorted(grouped_files.items()):
    out_path = os.path.join(output_folder, f"{letter}.md")
    with open(out_path, 'w', encoding='utf-8') as out_file:
        for _, cleaned_content in entries:
            out_file.write(f"\n\n{cleaned_content}")
    print(f"✅ Wrote {out_path} with {len(entries)} files")

# Generate SUMMARY.md with subpoints per letter group
summary_path = 'SUMMARY.md'
with open(summary_path, 'w', encoding='utf-8') as summary:
    summary.write("# Summary\n\n")
    for letter in sorted(grouped_files.keys()):
        summary.write(f"* [{letter}](./grouped/{letter}.md)\n")
        for filename, _ in grouped_files[letter]:
            display_name = os.path.splitext(filename)[0]
            summary.write(f"  * [{display_name}](./content/{filename})\n")

print("📘 Created grouped content and updated SUMMARY.md.")
