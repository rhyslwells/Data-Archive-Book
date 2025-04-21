import os
import re

# Path to your content folder
content_folder = './content'
images_folder = '../images'

# Function to convert ![[Image.png]] to ![Image.png](../images/Image.png)
def update_image_links(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Regular expression to match ![[image.png]]
    pattern = r'!\[\[([^\]]+)\]\]'
    
    # Replace with the desired format
    def replace(match):
        image_name = match.group(1)
        # URL encode the image name (replace spaces with %20)
        encoded_image_name = image_name.replace(' ', '%20')
        return f'![{image_name}](../images/{encoded_image_name})'

    updated_content = re.sub(pattern, replace, content)

    # Write the updated content back to the file
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(updated_content)

# Walk through all markdown files in the content folder and update the links
for root, dirs, files in os.walk(content_folder):
    for file in files:
        if file.endswith('.md'):
            file_path = os.path.join(root, file)
            update_image_links(file_path)

print("Image links updated successfully!")
