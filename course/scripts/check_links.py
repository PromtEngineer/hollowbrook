"""Check every relative link and image reference in the course resolves to a file.

    python3 scripts/check_links.py          # exit code 1 if anything is broken
"""
from __future__ import annotations

import glob
import os
import re
import sys
from urllib.parse import unquote

COURSE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD_LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
HTML_LINK = re.compile(r"(?:href|src)=[\"']([^\"']+)[\"']")
CODE_REF = re.compile(r"`((?:labs|interactive|figures|chapters|llm|tests|scripts)/[\w./-]+)`")


def targets(path: str) -> list[str]:
    text = open(path, encoding="utf-8", errors="replace").read()
    if path.endswith(".html"):
        return HTML_LINK.findall(text)
    return MD_LINK.findall(text) + CODE_REF.findall(text)


def main() -> int:
    files = (glob.glob(os.path.join(COURSE, "*.md")) + glob.glob(os.path.join(COURSE, "chapters", "*.md"))
             + glob.glob(os.path.join(COURSE, "interactive", "*.html")))
    broken: list[tuple[str, str]] = []
    n_checked = 0
    for f in files:
        base = os.path.dirname(f)
        for t in targets(f):
            if t.startswith(("http://", "https://", "mailto:", "#", "data:", "javascript:")) or "${" in t:
                continue
            t = unquote(t.split("#")[0])
            if not t:
                continue
            p = os.path.normpath(os.path.join(base if not t.startswith(("labs/", "interactive/", "figures/", "chapters/", "llm/", "tests/", "scripts/")) or "chapters" not in base else COURSE, t))
            # code refs like `labs/lab01_ngram.py` are course-root relative
            if not os.path.exists(p):
                alt = os.path.normpath(os.path.join(COURSE, t))
                if os.path.exists(alt):
                    n_checked += 1
                    continue
                broken.append((os.path.relpath(f, COURSE), t))
            n_checked += 1
    for f, t in broken:
        print(f"BROKEN  {f}  ->  {t}")
    print(f"{n_checked} links checked, {len(broken)} broken")
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
