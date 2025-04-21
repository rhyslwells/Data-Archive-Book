import os
import re

# Paths
content_folder = './content'
images_folder = '../images'

# Function to update image and wiki links
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
    def replace_wiki(match):
        page_name = match.group(1)
        encoded = page_name.replace(' ', '%20')
        return f'[{page_name}](./{encoded}.html)'  # link relative to local file

    content = re.sub(r'(?<!\!)\[\[([^\]]+)\]\]', replace_wiki, content)

    # Write the updated content back to the file
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)

# Walk through all markdown files in the content folder and update the links
for root, dirs, files in os.walk(content_folder):
    for file in files:
        if file.endswith('.md'):
            update_links(os.path.join(root, file))

print("✅ Image and wiki links updated successfully!")

# Path to the content folder
content_folder = './content'

# Output summary file
summary_file = 'SUMMARY.md'

# Function to generate the summary
def generate_summary(content_folder, summary_file):
    # Initialize the summary content with a title
    summary_content = "# Summary\n\n"

    # Walk through the content folder to find .md files
    for root, dirs, files in os.walk(content_folder):
        for file in files:
            if file.endswith('.md') and file != 'README.md':  # Avoid including README.md itself
                file_name_without_ext = os.path.splitext(file)[0]  # Remove .md extension
                # Create the markdown link with the relative .md path, keeping spaces
                summary_content += f"* [{file_name_without_ext}](./content/{file_name_without_ext}.md)\n"

    # Write the summary content to the summary.md file
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(summary_content)

    print(f"✅ Summary generated successfully in {summary_file}")

# Generate the summary
generate_summary(content_folder, summary_file)
