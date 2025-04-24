3. Merges every grouped file into book/book.md
* Drops each per‑letter “## Table of Contents” block while merging
4. Creates a grouped SUMMARY.md whose links all target anchors inside ./book/book.md.
  Example entry → * [AB Testing](./book/book.md#ab-testing)
5. (Optional) can build EPUB/PDF with HonKit from the merged book.