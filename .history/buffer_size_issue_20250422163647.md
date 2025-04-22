  at readableAddChunkPushByteMode (node:internal/streams/readable:510:3)
      at Readable.push (node:internal/streams/readable:390:5)
      at Pipe.onStreamRead (node:internal/stream_base_commons:191:23) {
    code: 'ERR_CHILD_PROCESS_STDIO_MAXBUFFER',
    cmd: ebook-convert C:\\Users\\RhysL\\AppData\\Local\\Temp\\honkit-sYYg02\\SUMMARY.html C:\\Users\\RhysL\\AppData\\Local\\Temp\\honkit-sYYg02\\index.epub --title="Data Archive Book" --comments="The Data Archive in book form" --authors="Rhys Wells" --book-producer="HonKit" --publisher="HonKit" --chapter="descendant-or-self::*[contains(concat(' ', normalize-space(@class), ' '), ' book-chapter ')]" --level1-toc="descendant-or-self::*[contains(concat(' ', normalize-space(@class), ' '), ' book-chapter-1 ')]" --level2-toc="descendant-or-self::*[contains(concat(' ', normalize-space(@class), ' '), ' book-chapter-2 ')]" --level3-toc="descendant-or-self::*[contains(concat(' ', normalize-space(@class), ' '), ' book-chapter-3 ')]" --max-levels="1" --no-chapters-in-toc --breadth-first --dont-split-on-page-breaks
  }
}

C:\Users\RhysL\Desktop\Data-Archive-Book\DAB>


ChatGPT said:
The warnings and error you’re seeing during npx honkit epub are fairly common when HonKit calls Calibre’s ebook-convert under the hood. Here's a breakdown of what’s happening and how to fix or mitigate it:

⚠️ Warnings: Invalid CSS Selector
pgsql
Copy
Edit
Ignoring CSS rule with invalid selector: 'code .hljs ::selection' (The pseudo-class :selection is not supported)
These come from your theme or syntax highlighting (likely Highlight.js). The ::selection pseudo-element is not supported in the EPUB spec via ebook-convert, and these warnings can safely be ignored. They will not stop your book from building or degrade usability, since text selection styles are purely cosmetic.

❌ Error: ERR_CHILD_PROCESS_STDIO_MAXBUFFER
This is the real issue:

css
Copy
Edit
code: 'ERR_CHILD_PROCESS_STDIO_MAXBUFFER'
This happens because the output from ebook-convert is too large and exceeds Node.js’s default buffer limit (stdout is flooded with logs). HonKit invokes Calibre's converter via a child process, but the output is too verbose, and the buffer overflows.

✅ Solutions
🔧 Option 1: Increase Node.js Buffer Limit
Run your HonKit build with a larger maxBuffer limit. Unfortunately, npx honkit does not expose this directly, so the most robust fix is to:

Install HonKit locally:

bash
Copy
Edit
npm install honkit --save-dev
Run a custom script in Node (e.g., build-epub.js):

js
Copy
Edit
const { exec } = require('child_process');

exec('npx honkit epub ./ ./data-archive-book.epub', { maxBuffer: 1024 * 5000 }, (error, stdout, stderr) => {
  if (error) {
    console.error(`❌ Error: ${error.message}`);
    return;
  }
  if (stderr) {
    console.warn(`⚠️ Warnings: ${stderr}`);
  }
  console.log(`✅ Output:\n${stdout}`);
});
Then run:

bash
Copy
Edit
node build-epub.js
This avoids ERR_CHILD_PROCESS_STDIO_MAXBUFFER.

📘 Optional: Clean Up Highlight.js CSS
To suppress CSS warnings altogether, you can remove or comment out the problematic selectors in your theme’s website/styles/website.css (or wherever you include custom styles):

css
Copy
Edit
/* Remove or comment out the ::selection rules */
code .hljs ::selection,
pre .hljs::selection {
    background: none;
}
But again, this step is cosmetic and optional.

Let me know if you'd like me to generate a fixed website.css or modify the book.json setup.
#epub #honkit #calibre #ebook #css







You said:
what does build-epub.js do


ChatGPT said:
The build-epub.js script is a custom Node.js wrapper that runs the HonKit EPUB build process with an increased buffer size. It’s necessary because the default buffer for child processes in Node.js is limited (200 KB), and when Calibre's ebook-convert command outputs a lot of logs, it can exceed that, causing the ERR_CHILD_PROCESS_STDIO_MAXBUFFER error.

🔧 Purpose
Problem it solves:
When you run:

bash
Copy
Edit
npx honkit epub ./ ./data-archive-book.epub
HonKit spawns a child process to run ebook-convert, but it doesn’t increase the buffer limit. If too much output is generated (e.g., from invalid CSS selectors), the buffer overflows and the process crashes.

Solution:
The script below increases the buffer to 5 MB to prevent that overflow.

🧠 How It Works
js
Copy
Edit
const { exec } = require('child_process');

// Run the HonKit EPUB build with an increased output buffer size
exec('npx honkit epub ./ ./data-archive-book.epub', { maxBuffer: 1024 * 5000 }, (error, stdout, stderr) => {
  if (error) {
    console.error(`❌ Error: ${error.message}`);
    return;
  }
  if (stderr) {
    console.warn(`⚠️ Warnings: ${stderr}`);
  }
  console.log(`✅ Output:\n${stdout}`);
});
Key Parts:
exec(...): Executes the shell command (npx honkit epub).

maxBuffer: 1024 * 5000: Sets the buffer size to 5 MB (instead of default ~200 KB).

error: If the process fails, logs the error.

stderr: Logs any non-fatal warnings (like unsupported CSS).

stdout: Shows the successful output of the build.

✅ Why Use This Script?
Avoids modifying global Node.js limits.

Keeps build command simple but safe.

Ensures HonKit can run successfully with verbose tools like Calibre.

