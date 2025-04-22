const { exec } = require('child_process');

console.log('📚 Starting EPUB build with HonKit...\n');

// Interval to log progress messages every 5 seconds
const interval = setInterval(() => {
  const now = new Date().toLocaleTimeString();
  console.log(`🛠️ Still building... (${now})`);
}, 5000);

// Execute HonKit EPUB build
exec('npx honkit epub ./ ./data-archive-book.epub', { maxBuffer: 1024 * 5000 }, (error, stdout, stderr) => {
  clearInterval(interval);

  console.log('\n📘 Build process finished.\n');

  if (error) {
    console.error(`❌ Error: ${error.message}`);
    return;
  }

  if (stderr) {
    console.warn(`⚠️ Warnings:\n${stderr}`);
  }

  console.log('✅ EPUB successfully created: ./data-archive-book.epub\n');
  console.log('📤 Build output:\n');
  console.log(stdout);
});
