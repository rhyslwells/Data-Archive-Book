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

                # Only replace *single* $...$ (not $$...$$)
                updated_content = re.sub(
                    r'(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)', 
                    r'$$\1$$', 
                    content,
                    flags=re.DOTALL
                )

            with open(filepath, 'w', encoding='utf-8') as file:
                file.write(updated_content)
            print(f"Updated {filename}")

# Run the replacement
replace_inline_math(ROOT / "book")
