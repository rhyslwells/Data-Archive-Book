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
