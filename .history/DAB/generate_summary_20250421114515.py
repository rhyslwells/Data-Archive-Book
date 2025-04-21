import os
import urllib.parse

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
                # URL-encode the file name to handle spaces or special characters
                file_name_encoded = urllib.parse.quote(file_name_without_ext)
                # Create the markdown link
                summary_content += f"* [{file_name_without_ext}]({os.path.join(root, file_name_encoded)}.html)\n"

    # Write the summary content to the summary.md file
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(summary_content)

    print(f"Summary generated successfully in {summary_file}")

# Generate the summary
generate_summary(content_folder, summary_file)
