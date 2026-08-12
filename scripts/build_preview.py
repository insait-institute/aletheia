#!/usr/bin/env python3
"""Bundle docs/ into a single self-contained HTML file.

Inlines local CSS, the FontAwesome JS bundle, and all images (as data URIs)
so the page can be viewed without a web server or network access — e.g. as a
claude.ai artifact, whose sandbox blocks all external requests.

Usage: python3 scripts/build_preview.py [output.html]
"""
import base64
import re
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent / "docs"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("aletheia-preview.html")

html = (DOCS / "index.html").read_text()


def css_inline(m):
    path = m.group(1)
    # carousel/slider are unused; fontawesome ships broken webfont paths —
    # icons come from the JS bundle instead
    if any(x in path for x in ("carousel", "slider", "fontawesome")):
        return ""
    return "<style>\n" + (DOCS / path).read_text() + "\n</style>"


html = re.sub(r'<link rel="stylesheet" href="(static/css/[^"]+)">', css_inline, html)
html = re.sub(r'<link rel="stylesheet"\s*\n\s*href="https://cdn[^"]+">', "", html)

# external jquery is blocked in the artifact sandbox; index.js needs it,
# so replace it with just the vanilla copyBibtex helper
html = re.sub(r'<script src="https://ajax[^"]+"></script>', "", html)
html = html.replace(
    '<script defer src="static/js/fontawesome.all.min.js"></script>',
    "<script>\n" + (DOCS / "static/js/fontawesome.all.min.js").read_text() + "\n</script>",
)
html = re.sub(r'<script src="static/js/bulma-[^"]+"></script>', "", html)
html = html.replace(
    '<script src="static/js/index.js"></script>',
    '<script>function copyBibtex(){var t=document.getElementById("BibTeX");'
    "navigator.clipboard.writeText(t.innerHTML);}</script>",
)


def img_inline(m):
    path = m.group(1)
    mime = "image/svg+xml" if path.endswith(".svg") else "image/png"
    data = base64.b64encode((DOCS / path).read_bytes()).decode()
    return f'src="data:{mime};base64,{data}"'


html = re.sub(r'src="(static/images/[^"]+)"', img_inline, html)
html = re.sub(r'<link rel="icon"[^>]+>', "", html)
# Google Fonts is also blocked; drop the link to avoid console noise
html = re.sub(r'<link href="https://fonts[^"]+" rel="stylesheet">', "", html)

# the artifact runtime wraps content in its own doctype/head/body skeleton,
# so keep only the title, styles, scripts, and body content
head = re.search(r"<head>(.*?)</head>", html, re.S).group(1)
body = re.search(r"<body>(.*?)</body>", html, re.S).group(1)
keep = re.findall(r"<style>.*?</style>|<script>.*?</script>", head, re.S)
out = (
    "<title>Aletheia — TMLR 2026 (project page preview)</title>\n"
    + "\n".join(keep)
    + body
)
OUT.write_text(out)
print(f"wrote {OUT} ({len(out) // 1024} KB)")
