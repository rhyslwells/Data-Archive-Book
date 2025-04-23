To handle `==highlighted text==` syntax in your Markdown notes — which is **not natively supported by HonKit** — you can preprocess your Markdown files to convert this into HTML or a custom plugin-compatible format.

---

### ✅ Option 1: Convert `==...==` to `<mark>...</mark>`

You can preprocess your files and **replace**:

```markdown
==some text==
```

with:

```html
<mark>some text</mark>
```

This works in HonKit, as it supports inline HTML.

---

### ✨ Python Code Snippet to Convert in Preprocessing

Add this to your Markdown processing script (e.g., inside your `strip_yaml_front_matter` logic or after reading file content):

```python
def convert_highlights(content):
    return re.sub(r'==(.+?)==', r'<mark>\1</mark>', content)
```

Then apply it during your file processing:

```python
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()
    content = strip_yaml_front_matter(content)
    content = convert_highlights(content)
```

---

### ✅ Why `<mark>`?

HTML’s `<mark>` is the semantic way to indicate highlighted or marked text. HonKit will render it properly in output formats including HTML and EPUB.

---

Would you like help adding this into your existing file-grouping script?