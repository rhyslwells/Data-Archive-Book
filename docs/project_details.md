# Project Structure

## Features

- **Selective Book Building**  
  Compile customized books by selecting Markdown files through `selected_files_list.md` or dynamic queries in `search.md`.

- **Content Preprocessing**  
  Automatically standardizes Markdown files to:
  - Correct image links for EPUB compatibility
  - Adjust inline LaTeX for proper rendering

- **Build Workflow**  
  Comprehensive build instructions are provided in the [Build Documentation](docs/build.md).

- **Generated Outputs**  
  Finished PDF and EPUB files are stored in the `books/` directory.

---

## System Requirements

- [HonKit](https://honkit.netlify.app/) — for book generation (PDF and EPUB)
- [Calibre](https://calibre-ebook.com/) — for EPUB format handling
- Python 3.x — for preprocessing and content management scripts

---

## Additional Resources

- **Honkit Guide**  
  [How to Create Ebooks from Markdown Using Honkit](https://flaviocopes.com/how-to-create-ebooks-markdown/#:~:text=honkit%20works%20great.,and%20let%20CloudFlare%20distribute%20it.)

---

## Plugins

- **Include Code Blocks**  
  Adds the ability to embed and format code within books.
  ```bash
  npm install gitbook-plugin-include-codeblock --save-dev
  ```

- **KaTeX Math Rendering**  
  Ensures proper rendering of mathematical notation.
  ```bash
  npm install gitbook-plugin-katex --save-dev
  ```
  Further details available in the [Honkit KaTeX Setup Discussion](https://github.com/honkit/honkit/issues/217).

---

### Small refinements:
- "Compile customized books" instead of "choose which files" → sounds more deliberate
- "System Requirements" instead of just "Requirements" → standard for open-source READMEs
- More active phrasing ("ensures proper rendering")  
- Clear function description under each plugin install step
