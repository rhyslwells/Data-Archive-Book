// export-to-pdf.js
const puppeteer = require('puppeteer');
const path = require('path');
const fs = require('fs');

(async () => {
  const browser = await puppeteer.launch();
  const page = await browser.newPage();

  const indexPath = 'file://' + path.resolve(__dirname, '_book/index.html');
  await page.goto(indexPath, { waitUntil: 'networkidle0' });

  await page.pdf({
    path: 'mybook.pdf',
    format: 'A4',
    printBackground: true,
    margin: {
      top: '20mm',
      bottom: '20mm',
      left: '15mm',
      right: '15mm'
    }
  });

  await browser.close();
})();
