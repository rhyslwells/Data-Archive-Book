import os
import re
from collections import defaultdict

content_folder = './content'
output_folder = './grouped'
os.makedirs(output_folder, exist_ok=True)

# Define alphabetical ranges for chapters
alphabet_ranges = [('A', 'C'), ('D', 'F'), ('G', 'I'), ('J', 'L'),
                   ('M', 'O'), ('P', 'R'), ('S', 'U'), ('V', 'Z')]

def get_range(letter):
    for start, end in alphabet_ranges:
        if start <= letter <= end:
            return f"{start}-{end}"
    return "Other"

def strip_yaml_front_matter(text):
    if text.startswith('---'):
        end = text.find('---', 3)
        if end != -1:
            return text[end+3:].lstrip()
    return text

# Group file contents
grouped_files = defaultdict(list)
for file in sorted(os.listdir(content_folder)):
    if file.endswith('.md') and file.lower() != 'readme.md':
        initial = file[0].upper()
        range_key = get_range(initial)
        file_path = os.path.join(content_folder, file)
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            cleaned = strip_yaml_front_matter(content)
            grouped_files[range_key].append("\n\n" + cleaned)

# Write grouped chapter files
for group, contents in grouped_files.items():
    out_path = os.path.join(output_folder, f"{group}.md")
    with open(out_path, 'w', encoding='utf-8') as out_file:
        out_file.write('\n'.join(contents))
    print(f"✅ Wrote {out_path} with {len(contents)} files")

# Generate new SUMMARY.md
summary_path = 'SUMMARY.md'
with open(summary_path, 'w', encoding='utf-8') as summary:
    summary.write("# Summary\n\n")
    for group in sorted(grouped_files.keys()):
        summary.write(f"* [{group}](./grouped/{group}.md)\n")

print("📘 Grouped content and updated SUMMARY.md.")
