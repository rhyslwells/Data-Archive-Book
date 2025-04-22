const { spawn } = require('child_process');

console.log('📚 Starting HonKit HTML build...\n');

const interval = setInterval(() => {
  const now = new Date().toLocaleTimeString();
  console.log(`🛠️ Still building... (${now})`);
}, 5000);

// Use 'npx.cmd' on Windows
const build = spawn('npx.cmd', ['honkit', 'build', './']);

build.stdout.on('data', (data) => {
  process.stdout.write(data.toString());
});

build.stderr.on('data', (data) => {
  process.stderr.write(data.toString());
});

build.on('close', (code) => {
  clearInterval(interval);
  if (code === 0) {
    console.log('\n✅ HonKit HTML build completed successfully!');
  } else {
    console.error(`\n❌ HTML build failed with exit code ${code}`);
  }
});
