"""Ingest .docx script bundles into individual per-script markdown files.

Each source .docx holds several scripts separated by numbered headings
("12. Some Title"). This splits them out to scripts/<category>/<NN>-<slug>.md,
with the title as an H1 and the narration body below.

Category is derived from the filename prefix (e.g. "stoicism-scripts-01-05.md.docx"
-> "stoicism"), which is more reliable than the (sometimes joke/misspelled)
folder names.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from .config import ROOT
from .util import slugify

SCRIPTS_DIR = ROOT / "scripts"

# A script heading: "1. Title" / "12. Title" — short, starts a new script.
_HEADING = re.compile(r"^\s*(\d{1,2})\.\s+(.+)$")
# Bundle header / subtitle lines to skip when they appear before any script.
_SKIP_PREFIXES = ("faceless", "full vo", "channel —", "channel -")


@dataclass
class ParsedScript:
    num: int
    title: str
    body: str


def _category_from(path: str) -> str:
    fn = os.path.basename(path).lower()
    return fn.split("-scripts")[0].split(".md")[0].split(".docx")[0].strip()


def _is_heading(text: str) -> bool:
    m = _HEADING.match(text)
    if not m or len(text) >= 120:
        return False
    return not text[:40].lower().startswith(_SKIP_PREFIXES)


def parse_docx(path: str | Path) -> list[ParsedScript]:
    """Split one .docx into its constituent scripts."""
    import docx  # local import; optional dependency

    paras = [p.text.strip() for p in docx.Document(str(path)).paragraphs]
    paras = [p for p in paras if p]

    scripts: list[ParsedScript] = []
    cur: dict | None = None
    for p in paras:
        m = _HEADING.match(p)
        if m and _is_heading(p):
            if cur:
                scripts.append(ParsedScript(cur["num"], cur["title"], "\n\n".join(cur["body"])))
            cur = {"num": int(m.group(1)), "title": m.group(2).strip(), "body": []}
        elif cur:
            cur["body"].append(p)
    if cur:
        scripts.append(ParsedScript(cur["num"], cur["title"], "\n\n".join(cur["body"])))
    return scripts


def ingest_folder(src: str | Path, out_dir: str | Path = SCRIPTS_DIR) -> dict[str, int]:
    """Walk src for .docx files, split them, and write per-script .md files.

    Returns a {category: count} summary.
    """
    src = Path(src)
    out_dir = Path(out_dir)
    files = sorted(src.rglob("*.docx"))
    summary: dict[str, int] = {}

    for f in files:
        category = _category_from(str(f))
        cat_dir = out_dir / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        for s in parse_docx(f):
            slug = slugify(s.title)
            name = f"{s.num:02d}-{slug}.md"
            content = f"# {s.title}\n\n{s.body}\n"
            (cat_dir / name).write_text(content, encoding="utf-8")
            summary[category] = summary.get(category, 0) + 1
    return summary
