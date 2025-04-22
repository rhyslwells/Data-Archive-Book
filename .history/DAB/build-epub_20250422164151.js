const { exec } = require('child_process');

console.log('📚 Starting EPUB build with HonKit...\n');

// Spinner setup (simple visual feedback)
const spinnerChars = ['|', '/', '-', '\\'];
let spinnerIndex = 0;
const spinner = setInterval(() => {
  process.stdout.write(`\r⏳ Building... ${spinnerChars[spinnerIndex++]}`);
  spinnerIndex %= spinnerChars.length;
}, 200);

// Run the HonKit EPUB build with increased buffer size
exec('npx honkit epub ./ ./data-archive-book.epub', { maxBuffer: 1024 * 5000 }, (error, stdout, stderr) => {
  clearInterval(spinner);
  process.stdout.write('\r'); // Clear spinner

  console.log('\n🔍 HonKit build process finished.\n');

  if (error) {
    console.error(`❌ Build error: ${error.message}`);
    return;
  }

  if (stderr) {
    console.warn(`⚠️ Warnings during build:\n${stderr}`);
  }

  console.log('✅ EPUB successfully created: ./data-archive-book.epub\n');
  console.log('📤 Build output:\n');
  console.log(stdout);
});
