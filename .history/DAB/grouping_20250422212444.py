import os
import re
from collections import defaultdict

content_folder = './content'
group_folder = './grouped'
summary_file = 'SUMMARY.md'

os.makedirs(group_folder, exist_ok=True)

def slugify(text):
    return re.sub(r'[^\w\- ]', '', text).strip().lower().replace(' ', '-')

def get_group(char):
    char = char.upper()
    if char in 'ABC':
        return 'A-C'
    elif char in 'DEF':
        return 'D-F'
    elif char in 'GHI':
        return 'G-I'
    elif char in 'JKL':
        return 'J-L'
    elif char in 'MNO':
        return 'M-O'
    elif char in 'PQR':
        return 'P-R'
    elif char in 'STU':
        return 'S-U'
    elif char in 'VWX':
        return 'V-X'
    elif char in 'YZ':
        return 'Y-Z'
    elif char.isdigit():
        return '0-9'
    else:
        return 'Misc'

# Collect contents by group
grouped_content = defaultdict(list)

for file in sorted(os.listdir(content_folder)):
    if file.endswith('.md') and file.lower() != 'readme.md':
        path = os.path.join(content_folder, file)
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()

        base_name = os.path.splitext(file)[0]
        title = f"# {base_name.replace('-', ' ').title()} {{#{slugify(base_name)}}}\n\n"
        grouped_content[get_group(base_name[0])].append(title + content)

# Write grouped markdown files
for group, pages in grouped_content.items():
    out_path = os.path.join(group_folder, f"{group}.md")
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n\n---\n\n'.join(pages))
    print(f"✅ Created {out_path}")

# Generate SUMMARY.md for HonKit
with open(summary_file, 'w', encoding='utf-8') as f:
    f.write("# Summary\n")
    for group in sorted(grouped_content.keys()):
        f.write(f"* [{group}](./grouped/{group}.md)\n")

print(f"📘 SUMMARY.md generated with {len(grouped_content)} grouped files.")
