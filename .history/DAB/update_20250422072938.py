import os
import re

# Paths
content_folder = './content'
images_folder = './content/images'  # Adjusted path for images folder

# Function to update image and wiki links
def update_links(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # --- Image links: ![[image.png]] -> ![image.png](./content/images/image.png)
    def replace_image(match):
        image_name = match.group(1)
        encoded = image_name.replace(' ', '%20')
        return f'![{image_name}]({images_folder}/{encoded})'

    content = re.sub(r'!\[\[([^\]]+)\]\]', replace_image, content)

    # Convert Obsidian-style wiki links [[Page Name]] to [Page Name](./Page Name.md)
    def replace_wiki(match):
        page_name = match.group(1)
        # Convert spaces to %20 for compatibility
        encoded = page_name.replace(' ', '%20')
        return f'[{page_name}](./{encoded}.md)'  # Relative link to other .md files

    content = re.sub(r'(?<!\!)\[\[([^\]]+)\]\]', replace_wiki, content)

    # --- Add title after YAML front matter
    if content.startswith('---'):
        yaml_end = content.find('---', 3)  # Find the end of the YAML block
        if yaml_end != -1:
            yaml_end += 3  # Move past the closing '---'
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            title = f"# {file_name.replace('-', ' ').title()}\n\n"
            if title.strip() not in content[yaml_end:].strip():  # Avoid duplicate titles
                content = content[:yaml_end] + '\n' + title + content[yaml_end:]

    # Write the updated content back to the file
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)

# Walk through all markdown files in the content folder and update the links
for root, dirs, files in os.walk(content_folder):
    for file in files:
        if file.endswith('.md'):
            update_links(os.path.join(root, file))

print("✅ Image and wiki links updated successfully!")