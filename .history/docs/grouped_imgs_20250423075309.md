Note you need since we are compling with grouped files we need to reference the images correctly.

Either do this at the grouping.py or update.py file.

Lets just do at update.py for now. If we need to reuse the code and don't need grouping.py, we can change it then as it would be more approriate in grouping.py.

---

Thank you — since your **rendered markdown files are in the `grouped/` folder**, and the **images are in `content/images/`**, the image paths must be **relative from `grouped/` to `content/images/`**.

---

### ✅ Updated Structure

```
Data-Archive-Book/
├── content/
│   ├── images/
│   │   └── example.png
│   └── many-md-files.md
├── grouped/
│   ├── A.md  ← this is where HonKit will read from
├── SUMMARY.md
├── book.json
```

---

### ✅ Correct Image Path in `grouped/*.md`

To correctly reference images from the `grouped` files, use:

```markdown
![Description](../content/images/example