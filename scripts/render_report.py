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


def main():
    if not MD.exists():
        raise SystemExit(f"missing {MD}")
    try:
        import markdown
        from weasyprint import HTML

        html_body = markdown.markdown(
            MD.read_text(), extensions=["tables", "fenced_code"])
        html = f"<html><head><style>{CSS}</style></head><body>{html_body}</body></html>"
        HTML(string=html, base_url=str(REPORT_DIR)).write_pdf(str(PDF))
        print(f"wrote {PDF}")
    except Exception as e:  # pragma: no cover
        print(f"WeasyPrint path failed ({e}).")
        print("Fallback — render with pandoc:")
        print(f"  pandoc {MD} -o {PDF} --pdf-engine=weasyprint")


if __name__ == "__main__":
    main()
