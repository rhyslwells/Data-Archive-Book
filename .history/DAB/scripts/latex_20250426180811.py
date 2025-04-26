import re
from pathlib import Path
import os

ROOT = Path(__file__).parent.parent.resolve()
BOOK_MD = ROOT / "book" / "book.md"  # Path to your main book.md file

def replace_inline_math(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.md'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()

                # Step 1: Match inline math ($...$) and replace it with $$...$$
                updated_content = re.sub(r'(?<!\$)\$(.*?)\$(?!\$)', r'$$\1$$', content)

            # Step 2: Overwrite the file with the updated content
            with open(filepath, 'w', encoding='utf-8') as file:
                file.write(updated_content)
            print(f"Updated {filename}")

# Run the replacement in the book directory
replace_inline_math(ROOT / "book")  # The directory where book.md is located
