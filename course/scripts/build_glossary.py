"""Merge chapters/_glossary_NN.md files into chapters/C-glossary.md (Appendix C).

Each source line looks like ``- **term** — definition (Ch. NN)``. Terms are merged
case-insensitively; when several chapters define the same term, the first
definition is kept and every chapter number is listed.
"""
from __future__ import annotations

import glob
import os
import re
from collections import OrderedDict

COURSE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CH = os.path.join(COURSE, "chapters")
LINE = re.compile(r"^\s*[-*]\s*\*\*(.+?)\*\*\s*[—–-]+\s*(.+?)\s*$")
CHREF = re.compile(r"\(?\s*Ch(?:apter)?\.?\s*(\d{1,2})\s*\)?\s*$")


def main() -> None:
    entries: "OrderedDict[str, dict]" = OrderedDict()
    for path in sorted(glob.glob(os.path.join(CH, "_glossary_*.md"))):
        chap = re.search(r"_glossary_(\d+)", path).group(1)
        for raw in open(path, encoding="utf-8"):
            m = LINE.match(raw)
            if not m:
                continue
            term, definition = m.group(1).strip(), m.group(2).strip()
            definition = CHREF.sub("", definition).strip().rstrip(".") + "."
            key = term.lower()
            if key in entries:
                if chap not in entries[key]["chapters"]:
                    entries[key]["chapters"].append(chap)
            else:
                entries[key] = {"term": term, "definition": definition, "chapters": [chap]}

    by_letter: dict[str, list[dict]] = {}
    for e in sorted(entries.values(), key=lambda e: e["term"].lower()):
        by_letter.setdefault(e["term"][0].upper(), []).append(e)

    out = ["# Appendix C: Glossary", "",
           "Every bold term defined in the course, with the chapter(s) that define it. "
           f"{len(entries)} entries.", ""]
    for letter in sorted(by_letter):
        out.append(f"## {letter}")
        out.append("")
        for e in by_letter[letter]:
            chs = ", ".join(f"Ch. {int(c)}" for c in e["chapters"])
            out.append(f"- **{e['term']}** — {e['definition']} *({chs})*")
        out.append("")
    with open(os.path.join(CH, "C-glossary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(f"wrote chapters/C-glossary.md with {len(entries)} entries from "
          f"{len(glob.glob(os.path.join(CH, '_glossary_*.md')))} chapter glossaries")


if __name__ == "__main__":
    main()
