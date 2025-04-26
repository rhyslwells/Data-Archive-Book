
# main.py

Copy "Data-Archive\content\standardised" from Data-Archive-Book to DAB folder and name the folder content.
Copy "Data-Archive\content\storage\images" to contents folder in DAB.
 then manually run either

npx honkit pdf ./ ./data-archive-book.pdf
npx honkit epub ./ ./data-archive-book.epub

and once done move the resulting files to the Data-Archive-Book/books folder.

Optional:
- Launch a development server to preview the content in the browser. Open http://localhost:4000 to view your book.

npx honkit build
npx honkit serve


----
# old summary of main.py

Step 1

Copy the contents of this to the folder Notes/content
"\Desktop\Data-Archive\content\standardised"

Copy the contents of this to Notes/content/images
"C:\Users\RhysL\Desktop\Data-Archive\content\storage\images"

delete old DAB/content

Create DAB/content and DAB/content/images

Copy all images to from above to DAB/content/images

Then with selected_files.py copy selected files to DAB/content

run update.py

run compiler.py in DAB

update readme and title of book to be specified based on selected files.


#------------------