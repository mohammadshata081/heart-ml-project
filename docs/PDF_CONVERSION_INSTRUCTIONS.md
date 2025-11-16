# How to Convert final_report.md to PDF

There are several ways to convert the markdown report to PDF. Choose the method that works best for you:

## Method 1: Using Python (WeasyPrint) - Recommended

1. Install required packages:
```bash
pip install weasyprint markdown
```

2. Run the conversion script:
```bash
python docs/generate_report_pdf.py
```

## Method 2: Using Node.js (md-to-pdf)

1. Install Node.js (if not already installed): https://nodejs.org/

2. Run the conversion:
```bash
npx --yes md-to-pdf docs/final_report.md
```

This will create `docs/final_report.pdf` automatically.

## Method 3: Using Pandoc

1. Install Pandoc: https://pandoc.org/installing.html

2. Run the conversion:
```bash
pandoc docs/final_report.md -o docs/final_report.pdf
```

## Method 4: Using VS Code Extension

1. Install VS Code extension "Markdown PDF" by yzane
2. Open `docs/final_report.md` in VS Code
3. Right-click → "Markdown PDF: Export (pdf)"

## Method 5: Online Converters

1. Go to one of these websites:
   - https://www.markdowntopdf.com/
   - https://dillinger.io/ (export as PDF)
   - https://md2pdf.netlify.app/

2. Upload or paste the content of `final_report.md`
3. Download the generated PDF

## Method 6: Using Google Docs / Microsoft Word

1. Copy the content from `final_report.md`
2. Paste into Google Docs or Microsoft Word
3. Format as needed
4. Export/Save as PDF

## Recommended: Method 2 (md-to-pdf)

The easiest and most reliable method is using Node.js with md-to-pdf:

```bash
npx --yes md-to-pdf docs/final_report.md --pdf-options '{"format": "A4", "margin": {"top": "20mm", "right": "20mm", "bottom": "20mm", "left": "20mm"}}'
```

This will create a nicely formatted PDF automatically.

