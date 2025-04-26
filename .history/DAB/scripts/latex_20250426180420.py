import os
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
BOOK_MD = ROOT / "book" / "book.md"  # Path to your main book.md file

def replace_inline_math(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.md'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()

            # Split content into lines (or paragraphs)
            lines = content.splitlines()
            updated_lines = []

            # Process each line
            for line in lines:
                # Check if inline math exists in the line
                while '$' in line:
                    start_idx = line.find('$')
                    end_idx = line.find('$', start_idx + 1)
                    if end_idx != -1:
                        # Inline math detected, replace it with $$...$$
                        inline_math = line[start_idx + 1:end_idx]
                        new_inline_math = f'${inline_math}$'  # Corrected this line
                        line = line[:start_idx] + f'$$' + inline_math + f'$$' + line[end_idx + 1:]
                    else:
                        break
                updated_lines.append(line)

            # Join the processed lines back together
            updated_content = "\n".join(updated_lines)

            # Overwrite the file with the updated content
            with open(filepath, 'w', encoding='utf-8') as file:
                file.write(updated_content)
            print(f"Updated {filename}")

# Run the replacement in the book directory
replace_inline_math(ROOT / "book")  # The directory where book.md is located
