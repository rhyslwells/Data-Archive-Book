const { exec } = require('child_process');

console.log('📚 Starting EPUB build... This may take a moment.\n');

// Spinner setup (optional visual feedback)
const spinnerChars = ['|', '/', '-', '\\'];
let spinnerIndex = 0;
const spinner = setInterval(() => {
  process.stdout.write(`\r⏳ Building... ${spinnerChars[spinnerIndex++]}`);
  spinnerIndex %= spinnerChars.length;
}, 250);

// Run the HonKit EPUB build with increased buffer
exec('npx honkit epub ./ ./data-archive-book.epub', { maxBuffer: 1024 * 5000 }, (error, stdout, stderr) => {
  clearInterval(spinner);
  process.stdout.write('\r'); // Clear spinner line

  if (error) {
    console.error(`❌ Error: ${error.message}`);
    return;
  }

  if (stderr) {
    console.warn(`⚠️ Warnings:\n${stderr}`);
  }

  console.log('✅ EPUB build complete!\n');
  console.log(`📤 Output:\n${stdout}`);
});
