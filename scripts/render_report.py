# ============================================================================
# AI-assisted: Generated with Claude (Anthropic) as an ML pair-programmer for
# the ARCADE interview project, and reviewed by the candidate.
# See report/report.md, section "AI Tool Usage".
# ============================================================================
"""Render report/report.md -> report/report.pdf (<= 5 pages).

Primary path: markdown -> HTML -> PDF via WeasyPrint. Falls back to printing the
pandoc command if WeasyPrint is unavailable.
"""
from __future__ import annotations

from pathlib import Path

REPORT_DIR = Path(__file__).resolve().parents[1] / "report"
MD = REPORT_DIR / "report.md"
PDF = REPORT_DIR / "report.pdf"

CSS = """
@page { size: A4; margin: 1.6cm; }
body { font-family: 'Helvetica','Arial',sans-serif; font-size: 10.5px; line-height: 1.35; }
h1 { font-size: 17px; margin: 0 0 4px; }
h2 { font-size: 13px; margin: 10px 0 3px; border-bottom: 1px solid #ccc; }
h3 { font-size: 11.5px; margin: 7px 0 2px; }
table { border-collapse: collapse; width: 100%; font-size: 9.5px; }
th, td { border: 1px solid #999; padding: 2px 5px; text-align: center; }
th { background: #eee; }
img { max-width: 100%; }
code { background: #f2f2f2; padding: 0 2px; }
"""


def _xhtml2pdf(html: str) -> bool:
    """Pure-Python HTML→PDF via xhtml2pdf. No system libs needed."""
    try:
        from xhtml2pdf import pisa
        with PDF.open("wb") as fh:
            status = pisa.CreatePDF(html, dest=fh, encoding="utf-8")
        if not status.err:
            print(f"wrote {PDF}  (xhtml2pdf)")
            return True
        print(f"xhtml2pdf error: {status.err}")
        return False
    except Exception as e:
        print(f"xhtml2pdf failed: {e}")
        return False


def _weasyprint(html: str) -> bool:
    """Try WeasyPrint Python API."""
    try:
        from weasyprint import HTML
        doc = HTML(string=html, base_url=str(REPORT_DIR))
        try:
            doc.write_pdf(target=str(PDF))
        except TypeError:
            PDF.write_bytes(doc.write_pdf())
        print(f"wrote {PDF}  (WeasyPrint)")
        return True
    except Exception as e:
        print(f"WeasyPrint failed: {e}")
        return False


def main():
    if not MD.exists():
        raise SystemExit(f"missing {MD}")

    import markdown
    html_body = markdown.markdown(MD.read_text(), extensions=["tables", "fenced_code"])
    html = f"<html><head><style>{CSS}</style></head><body>{html_body}</body></html>"

    if not _weasyprint(html) and not _xhtml2pdf(html):
        raise SystemExit("Could not render PDF. Run: pip install xhtml2pdf")


if __name__ == "__main__":
    main()
