"""report/report.md -> report/report.html -> report/report.pdf (headless Chrome).
  uv run --with markdown python report/build_pdf.py"""
import subprocess
from pathlib import Path
import markdown

D = Path(__file__).resolve().parent
CSS = """
@page { size: A4; margin: 13mm 13mm 12mm 13mm; }
body { font-family: -apple-system, 'Helvetica Neue', Arial, sans-serif; font-size: 8.6pt; line-height: 1.32; color: #111; }
h1 { font-size: 13.5pt; margin: 0 0 4px; } h2 { font-size: 10.5pt; margin: 9px 0 3px; border-bottom: 1px solid #bbb; }
p { margin: 3px 0 5px; } ul { margin: 2px 0 5px 16px; padding: 0; } li { margin: 1px 0; }
table { border-collapse: collapse; width: 100%; margin: 4px 0 6px; font-size: 7.9pt; }
th, td { border: 1px solid #c8c8c8; padding: 1.5px 4px; text-align: left; vertical-align: top; }
th { background: #f0f0f0; } code { font-size: 7.8pt; background: #f4f4f4; padding: 0 2px; }
"""
html = markdown.markdown((D / "report.md").read_text(), extensions=["tables"])
(D / "report.html").write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>Report</title>"
                                f"<style>{CSS}</style></head><body>{html}</body></html>")
chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                f"--print-to-pdf={D / 'report.pdf'}", (D / "report.html").as_uri()], check=True, capture_output=True)
print("pages:", (D / "report.pdf").read_bytes().count(b"/Type /Page") - (D / "report.pdf").read_bytes().count(b"/Type /Pages"))
