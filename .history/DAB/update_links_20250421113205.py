import os
import re

# Paths
content_folder = './content'
images_folder = '../images'

def update_links(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # --- Image links: ![[image.png]] -> ![image.png](../images/image.png)
    def replace_image(match):
        image_name = match.group(1)
        encoded = image_name.replace(' ', '%20')
        return f'![{image_name}]({images_folder}/{encoded})'

    content = re.sub(r'!\[\[([^\]]+)\]\]', replace_image, content)

    # --- Wiki links: [[Page Name]] -> [Page Name](./Page%20Name.md)
# --- Wiki links: [[Page Name]] -> [Page Name](./Page%20Name.md)
    def replace_wiki(match):
        page_name = match.group(1)
        encoded = page_name.replace(' ', '%20')
        return f'[{page_name}](./{encoded}.md)'  # link relative to local file


    content = re.sub(r'(?<!\!)\[\[([^\]]+)\]\]', replace_wiki, content)

    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)

# Walk through all markdown files in the content folder and update the links
for root, dirs, files in os.walk(content_folder):
    for file in files:
        if file.endswith('.md'):
            update_links(os.path.join(root, file))

print("Image and wiki links updated successfully!")
