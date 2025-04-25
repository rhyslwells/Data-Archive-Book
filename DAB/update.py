import os
import re

# Paths
content_folder = './content'

# Slugify helper: turns "Page Name" into "page-name"
def slugify(text):
    return re.sub(r'[^\w\- ]', '', text).strip().lower().replace(' ', '-')

# Function to update image links, wiki links, and inject titles
def update_links_and_titles(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Replace image links: ![[image.png]] → ![image.png](./images/image.png)
    def replace_image(match):
        image_name = match.group(1)
        encoded = image_name.replace(' ', '%20')
        return f'![{image_name}](../content/images/{encoded})' #FLAG!

    content = re.sub(r'!\[\[([^\]]+)\]\]', replace_image, content)

    # Replace wiki links: [[Page Name]] → [Page Name](#page-name)
    def replace_wiki(match):
        page_name = match.group(1)
        anchor = slugify(page_name)
        return f'[{page_name}](#{anchor})'

    content = re.sub(r'(?<!\!)\[\[([^\]]+)\]\]', replace_wiki, content)

    # Add title header after YAML front matter, or at top if not present
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    title_text = file_name.replace('-', ' ').title()
    slug = slugify(title_text)
    title = f"# {title_text} {{#{slug}}}\n\n"
    # title = f"# {title_text}\n\n"

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


# Apply updates
for root, _, files in os.walk(content_folder):
    for file in files:
        if file.endswith('.md'):
            update_links_and_titles(os.path.join(root, file))


print("✅ All markdown files updated.")
