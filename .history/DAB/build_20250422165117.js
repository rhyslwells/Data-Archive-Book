const { spawn } = require('child_process');

console.log('📚 Starting HonKit HTML build...\n');

// Progress indicator every 5 seconds
const interval = setInterval(() => {
  const now = new Date().toLocaleTimeString();
  console.log(`🛠️ Still building... (${now})`);
}, 5000);

// Spawn HonKit HTML build command
const build = spawn('npx', ['honkit', 'build', './']);

// Forward stdout from the HonKit process
build.stdout.on('data', (data) => {
  process.stdout.write(data.toString());
});

// Forward stderr from the HonKit process
build.stderr.on('data', (data) => {
  process.stderr.write(data.toString());
});

// Final callback when the process exits
build.on('close', (code) => {
  clearInterval(interval);
  if (code === 0) {
    console.log('\n✅ HonKit HTML build completed successfully!');
  } else {
    console.error(`\n❌ HTML build failed with exit code ${code}`);
  }
});
