# Overview of `main.py`

## Purpose
This script automates the preparation of content for building the **Data-Archive-Book** project. It:

- Cleans and recreates the `DAB/content` and `DAB/content/images` directories.
- Copies selected standardized Markdown files and images into the content directory.
- Runs auxiliary scripts to update, process LaTeX/math formatting, and compile the content.
- (Optionally) Supports manual generation of PDF and EPUB formats via Honkit.

---

## Step-by-Step Workflow

### 1. Clean and Prepare the Content Directories
Delete any existing `DAB/content` folder and recreate the necessary structure:
- `DAB/content`
- `DAB/content/images`

Command:
```bash
python main.py --clean
```

---

### 2. Copy Standardized Content and Images
Copy:
- Markdown files listed in `DAB/selected_files_list.md` from  
  `Desktop/Data-Archive/content/standardised` ➔ `DAB/content`
- All images from  
  `Desktop/Data-Archive/content/storage/images` ➔ `DAB/content/images`

Command:
```bash
python main.py --copy
```

---

### 3. Run Update, LaTeX Correction, and Compiler Scripts
After copying, run the necessary Python scripts to update metadata, correct LaTeX inline syntax, and compile the book structure.

Commands:
```bash
python main.py --run update.py
python main.py --run compiler.py
python main.py --run latex.py   
```

---

### 4. Generate PDF and EPUB (Manual Step)
After preparing and compiling the content **(including LaTeX fixes)**, manually run either of the following Honkit commands:

For PDF:
```bash
npx honkit pdf ./ ./data-archive-book.pdf
```

For EPUB:
```bash
npx honkit epub ./ ./data-archive-book.epub
```

Once generated, move the resulting files (`data-archive-book.pdf`, `data-archive-book.epub`) into the `Data-Archive-Book/books` directory manually.

---

### 5. Optional Commands
You can also optionally:

- Build the static site:
  ```bash
  npx honkit build
  ```
- Launch a local development server:
  ```bash
  npx honkit serve
  ```

---
