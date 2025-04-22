Use update.py to generate:

1) Correct internal links [[]] - even though the do not currently work.
2) Correct image links ![]() - to the following format 
![Pasted image 20240124095916.png](images/Pasted%20image%2020240124095916.png)
3) Summary.md file

python update.py

Can we get internal links to work within a epub or pdf?

Images in epub:
- For epub to view images correctly, the images need to be in the same directory contents,not the same as pdf generation.
- The vscode epub viwer its bad use calibre to view the epub file.
- Honkit uses calibre to create the epub file.

Commands:
cd DAB
python update.py
npx honkit build
npx honkit serve
- Launch a development server to preview the content in the browser. Open http://localhost:4000 to view your book.

npx honkit pdf ./ ./data-archive-book.pdf
npx honkit epub ./ ./data-archive-book.epub