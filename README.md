# Data-Archive-Book

This project transforms selected content from the [Data Archive](https://rhyslwells.github.io/Data-Archive/) into a **PDF** and **EPUB** format using [Honkit](https://honkit.netlify.app/).

Users can choose specific Markdown files from the Data Archive to include in the generated book using the `selected_files_list.md` file and `search.md`. This enables the creation of tailored versions of the archive for different purposes.

> **Note:** This is an active work in progress. The PDF and EPUB versions are periodically updated when significant changes are made to the Data Archive.

---

## Project Structure

- **Build Instructions:**  
  See [Build Documentation](docs/build.md) for detailed commands and workflow.

- **Known Issues and Planned Improvements:**  
  See [Issues](docs/issues.md) for ongoing work and feature planning.

- **Generated Examples:**  
  Completed PDF and EPUB files are stored in the `books/` directory.

---

## Additional Resources

- **Honkit Overview:**  
  [How to Create Ebooks from Markdown Using Honkit](https://flaviocopes.com/how-to-create-ebooks-markdown/#:~:text=honkit%20works%20great.,and%20let%20CloudFlare%20distribute%20it.)

- **Reference Inspiration:**  
  [Learn JavaScript — GitHub Repository](https://github.com/sumn2u/learn-javascript/tree/main)

---

## Plugins Used

- **Include Code Blocks:**
  ```bash
  npm install gitbook-plugin-include-codeblock --save-dev
  ```

- **Render Math with KaTeX:**
  ```bash
  npm install gitbook-plugin-katex --save-dev
  ```
  See also: [Honkit KaTeX Setup Discussion](https://github.com/honkit/honkit/issues/217)
