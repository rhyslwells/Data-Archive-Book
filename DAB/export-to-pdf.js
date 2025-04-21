// export-to-pdf.js
const puppeteer = require('puppeteer');
const path = require('path');

(async () => {
  const browser = await puppeteer.launch();
  const page = await browser.newPage();

  // The path to your _book folder's index.html file
  const indexPath = 'file://' + path.resolve(__dirname, '_book/index.html');
  await page.goto(indexPath, { waitUntil: 'networkidle0' });

  // Wait for the content to be fully loaded
  await page.waitForSelector('.book-body'); // Make sure that the book body is loaded

  // Enable full page screenshot and set some margins for better formatting
  await page.pdf({
    path: 'mybook.pdf',
    format: 'A4',
    printBackground: true,
    margin: {
      top: '20mm',
      bottom: '20mm',
      left: '15mm',
      right: '15mm'
    },
    landscape: false,
    preferCSSPageSize: true,
    scale: 1.0 // adjust the scale if necessary
  });

  await browser.close();
})();
