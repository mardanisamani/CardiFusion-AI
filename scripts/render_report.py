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


def _weasyprint(html: str) -> bool:
    """Try WeasyPrint (handles API differences across versions). Returns True on success."""
    try:
        from weasyprint import HTML
        doc = HTML(string=html, base_url=str(REPORT_DIR))
        # WeasyPrint >= 61 changed write_pdf signature; try target= kwarg first
        try:
            doc.write_pdf(target=str(PDF))
        except TypeError:
            data = doc.write_pdf()
            PDF.write_bytes(data)
        print(f"wrote {PDF}  (WeasyPrint)")
        return True
    except Exception as e:
        print(f"WeasyPrint failed: {e}")
        return False


def _pandoc() -> bool:
    """Fallback: run pandoc if available. Returns True on success."""
    import shutil, subprocess
    if not shutil.which("pandoc"):
        print("pandoc not found — install it or fix WeasyPrint.")
        return False
    result = subprocess.run(
        ["pandoc", str(MD), "-o", str(PDF),
         "--pdf-engine=weasyprint", "--css=/dev/stdin"],
        input=CSS, text=True, capture_output=True
    )
    if result.returncode == 0:
        print(f"wrote {PDF}  (pandoc)")
        return True
    print(f"pandoc failed: {result.stderr}")
    return False


def main():
    if not MD.exists():
        raise SystemExit(f"missing {MD}")

    import markdown
    html_body = markdown.markdown(MD.read_text(), extensions=["tables", "fenced_code"])
    html = f"<html><head><style>{CSS}</style></head><body>{html_body}</body></html>"

    if not _weasyprint(html) and not _pandoc():
        raise SystemExit("Could not render PDF. Install WeasyPrint system libs:\n"
                         "  apt-get install -y libpango-1.0-0 libpangoft2-1.0-0 libcairo2")


if __name__ == "__main__":
    main()
