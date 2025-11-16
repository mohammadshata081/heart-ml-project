"""
Script to convert final_report.md to PDF
Requires: markdown, weasyprint or reportlab
"""

import os
import sys

def convert_md_to_pdf_markdown_pdf(md_file, pdf_file):
    """Convert markdown to PDF using markdown-pdf"""
    try:
        import subprocess
        result = subprocess.run(
            ['npx', '--yes', 'md-to-pdf', md_file, '--pdf-options', '{"format": "A4", "margin": {"top": "20mm", "right": "20mm", "bottom": "20mm", "left": "20mm"}}'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"SUCCESS: PDF generated successfully: {pdf_file}")
            return True
        else:
            print(f"ERROR: {result.stderr}")
            return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def convert_md_to_pdf_weasyprint(md_file, pdf_file):
    """Convert markdown to PDF using markdown + weasyprint"""
    try:
        import markdown
        from weasyprint import HTML, CSS
        from weasyprint.text.fonts import FontConfiguration
        
        # Read markdown file
        with open(md_file, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        # Convert markdown to HTML
        html_content = markdown.markdown(md_content, extensions=['extra', 'codehilite'])
        
        # Add CSS styling
        html_with_style = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                @page {{
                    size: A4;
                    margin: 2cm;
                }}
                body {{
                    font-family: 'Arial', sans-serif;
                    line-height: 1.6;
                    color: #333;
                }}
                h1 {{
                    color: #2c3e50;
                    border-bottom: 3px solid #3498db;
                    padding-bottom: 10px;
                }}
                h2 {{
                    color: #34495e;
                    margin-top: 30px;
                    border-bottom: 2px solid #ecf0f1;
                    padding-bottom: 5px;
                }}
                h3 {{
                    color: #7f8c8d;
                    margin-top: 20px;
                }}
                code {{
                    background-color: #f4f4f4;
                    padding: 2px 5px;
                    border-radius: 3px;
                    font-family: 'Courier New', monospace;
                }}
                pre {{
                    background-color: #f4f4f4;
                    padding: 10px;
                    border-radius: 5px;
                    overflow-x: auto;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin: 20px 0;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                th {{
                    background-color: #3498db;
                    color: white;
                }}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """
        
        # Convert HTML to PDF
        HTML(string=html_with_style).write_pdf(pdf_file)
        print(f"✅ PDF generated successfully: {pdf_file}")
        return True
    except ImportError:
        print("ERROR: weasyprint not installed. Install with: pip install weasyprint markdown")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def main():
    """Main function to convert markdown to PDF"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    md_file = os.path.join(base_dir, 'final_report.md')
    pdf_file = os.path.join(base_dir, 'final_report.pdf')
    
    if not os.path.exists(md_file):
        print(f"ERROR: {md_file} not found!")
        return
    
    print(f"Converting {md_file} to PDF...")
    print(f"Output: {pdf_file}")
    
    # Try different methods
    if convert_md_to_pdf_weasyprint(md_file, pdf_file):
        return
    
    print("\nAlternative methods to convert to PDF:")
    print("1. Install weasyprint: pip install weasyprint markdown")
    print("2. Use Node.js: npx --yes md-to-pdf docs/final_report.md")
    print("3. Use online converter: https://www.markdowntopdf.com/")
    print("4. Use VS Code extension: Markdown PDF")
    print("5. Use Pandoc: pandoc final_report.md -o final_report.pdf")

if __name__ == "__main__":
    main()

