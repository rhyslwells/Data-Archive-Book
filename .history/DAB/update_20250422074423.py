import os
import re

# Paths
content_folder = './content'
images_folder = './content/images'  # Adjusted path for images folder
summary_file = 'SUMMARY.md'

# Function to update image links, wiki links, and inject titles
def update_links_and_titles(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Replace image links: ![[image.png]] → ![image.png](./images/image.png)
    def replace_image(match):
        image_name = match.group(1)
        encoded = image_name.replace(' ', '%20')
        return f'![{image_name}](./images/{encoded})'

    content = re.sub(r'!\[\[([^\]]+)\]\]', replace_image, content)

    # Replace wiki links: [[Page Name]] → [Page Name](./Page Name.md)
    def replace_wiki(match):
        page_name = match.group(1)
        encoded = page_name.replace(' ', '%20')
        return f'[{page_name}](./{encoded}.md)'
    content = re.sub(r'(?<!\!)\[\[([^\]]+)\]\]', replace_wiki, content)

    # Add title header after YAML front matter, or at top if not present
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    title = f"# {file_name.replace('-', ' ').title()}\n\n"

    if content.startswith('---'):
        yaml_end = content.find('---', 3)
        if yaml_end != -1:
            yaml_end += 3
            if title.strip() not in content[yaml_end:].strip():
                content = content[:yaml_end] + '\n' + title + content[yaml_end:]
    else:
        if not content.startswith('#'):
            content = title + content

    # Write back updated content
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)

# Function to generate the HonKit-compatible SUMMARY.md
def generate_summary(content_folder, summary_file):
    summary_lines = ['# Summary\n']
    for root, _, files in os.walk(content_folder):
        for file in sorted(files):
            if file.endswith('.md') and file.lower() != 'readme.md':
                name = os.path.splitext(file)[0]
                summary_lines.append(f"* [{name}](./content/{file})")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))
    print(f"📘 SUMMARY.md generated at: {summary_file}")

# Apply updates and build summary
for root, _, files in os.walk(content_folder):
    for file in files:
        if file.endswith('.md'):
            update_links_and_titles(os.path.join(root, file))

generate_summary(content_folder, summary_file)

print("✅ All markdown files updated and SUMMARY.md created.")
