#!/usr/bin/env python3
"""Check every number in a write-up against the data, then strip the check notes for publishing.

Drafts (README.md, the site page text, ...) carry hidden notes after each prose paragraph:

    <!-- CHECK P3
    - "(78% success)" <- results/summary.json:headline.smolvla.5.mean (78.0)
    - "about 47 per task" <- results/summary.json:robust_claim.0.act_all_demos (47)
    - "close" <- manual: plan, Phase 4 notes
    -->

For each data line the script checks that
  1. the value at <file>:<key path> (JSON or YAML; list items by index) equals the value in brackets,
  2. the quoted text is still in the paragraph above the note,
  3. if the quoted text contains numbers, one of them is that value rounded the way it is written
     (78% for 78.0, 5.3 for 5.344, 30,000 for 30000).
"manual:" lines are listed for checking by eye. Numbers in the prose that no quoted text covers are
reported as warnings (small experiment-design constants such as 3, 5, 10, 20, 50 are ignored).
Placeholders (TODO, TBD, XXX) outside the notes fail the check, so a draft cannot be published with one.

LaTeX files (writeup/report.tex) work the same way, with the note as a block of % comment lines:

    % CHECK P3
    % - "(78\\% success)" <- results/summary.json:headline.smolvla.5.mean (78.0)

Quoted text is matched against the LaTeX source as written (so 78\\% with the backslash). Only text after
\\begin{document} is checked; tabular environments, figure files, citations and labels are ignored.

Usage:
  python scripts/check_numbers.py README.md                 # check; exit code 1 if anything fails
  python scripts/check_numbers.py README.md --manual        # also list the manual checks
  python scripts/check_numbers.py README.md --strip > out.md   # the publishable version, without notes
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
BLOCK = re.compile(r"<!--\s*CHECK\s+(\S+)(.*?)-->", re.S)
COMMENT = re.compile(r"<!--.*?-->", re.S)
LINE = re.compile(r'^\s*-\s*"(.+?)"(?:\s*\([^)]*\))?\s*<-\s*(.+?)\s*$')
DATA = re.compile(r"^([\w./-]+\.(?:json|ya?ml)):(\S+)\s*\(([-\d.,]+)\)$")
NUM = re.compile(r"(?<![\w.-])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\w])")
IGNORE = {"0", "1", "2", "3", "5", "10", "20", "50", "100", "256", "1000", "2000", "3000"}


def numbers(text: str) -> list[str]:
    return NUM.findall(text)


def as_float(s: str) -> float:
    return float(s.replace(",", ""))


def decimals(s: str) -> int:
    return len(s.split(".")[1]) if "." in s else 0


def lookup(file: str, path: str):
    p = ROOT / file
    data = json.loads(p.read_text()) if p.suffix == ".json" else yaml.safe_load(p.read_text())
    for part in path.split("."):
        if isinstance(data, list):
            data = data[int(part)]
        elif part in data:
            data = data[part]
        elif part.isdigit() and int(part) in data:
            data = data[int(part)]
        else:
            raise KeyError(f"{part!r} not found in {file}:{path}")
    return data


def tex_to_notes(src: str) -> str:
    """LaTeX source -> the Markdown-style form used here: `% CHECK` comment blocks become <!-- CHECK ... -->
    notes, every other comment is dropped, and the preamble is left out."""
    if "\\begin{document}" in src:
        src = src[src.index("\\begin{document}"):]
    lines, out, i = src.split("\n"), [], 0
    while i < len(lines):
        if re.match(r"\s*%\s*CHECK\s+\S+", lines[i]):
            block = [lines[i].strip().lstrip("%").strip()]
            i += 1
            while i < len(lines) and lines[i].lstrip().startswith("%"):
                block.append(re.sub(r"^\s*%\s?", "", lines[i]))
                i += 1
            out.append("<!-- " + "\n".join(block) + "\n-->")
            continue
        out.append(re.sub(r"(?<!\\)%.*", "", lines[i]))
        i += 1
    return "\n".join(out)


def prose_tex(text: str) -> str:
    """Typed LaTeX prose only: no tabular content, figure files, references, lengths or command names."""
    text = re.sub(r"\\begin\{tabular\}.*?\\end\{tabular\}", "", text, flags=re.S)
    text = re.sub(r"\\(includegraphics|cite|citep|ref|label|url|vspace|hspace|setlength|small|footnotesize)"
                  r"(\[[^\]]*\])?(\{[^}]*\})?", "", text)
    text = re.sub(r"\\href\{[^}]*\}", "", text)
    text = re.sub(r"\\[A-Za-z]+\*?", "", text)
    return text


def prose(text: str) -> str:
    """Paragraph text without code blocks and tables (those are generated, not typed)."""
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    text = re.sub(r"\]\([^)]*\)", "]", text)          # link targets (URLs, arXiv ids)
    text = re.sub(r"`[^`]*`", "", text)                 # inline code (versions, commit ids)
    return "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("|"))


def check(path: Path, show_manual: bool) -> int:
    tex = path.suffix == ".tex"
    src = tex_to_notes(path.read_text()) if tex else path.read_text()
    fails, warns, manual, n_data = [], [], [], 0
    pos = 0
    for m in BLOCK.finditer(src):
        pid, body = m.group(1), m.group(2)
        region = COMMENT.sub("", src[pos:m.start()])
        flat = " ".join(region.split())
        pos = m.end()
        covered: set[str] = set()
        for raw in body.splitlines():
            lm = LINE.match(raw)
            if not lm:
                continue
            snippet, source = lm.group(1), lm.group(2)
            covered.update(as_float(n) for n in numbers(snippet) if n.strip("-"))
            if any(" ".join(piece.split()) not in flat for piece in snippet.split("...")):
                fails.append(f"{pid}: quoted text no longer in the paragraph: \"{snippet}\"")
            if source.startswith("manual:"):
                manual.append(f"{pid}: \"{snippet}\" -> {source[7:].strip()}")
                continue
            dm = DATA.match(source)
            if not dm:
                fails.append(f"{pid}: cannot read source \"{source}\"")
                continue
            n_data += 1
            file, key, stated = dm.groups()
            try:
                actual = lookup(file, key)
            except (KeyError, IndexError, FileNotFoundError) as e:
                fails.append(f"{pid}: {e}")
                continue
            if abs(float(actual) - as_float(stated)) > 0.0006 + 1e-9 * abs(float(actual)):
                fails.append(f"{pid}: {file}:{key} is {actual}, note says {stated}")
            nums = numbers(snippet)
            if nums and not any(abs(round(float(actual), decimals(n)) - as_float(n)) < 1e-9 for n in nums):
                fails.append(f"{pid}: \"{snippet}\" does not show {key} = {actual} (rounded as written)")
        for n in numbers(prose_tex(region) if tex else prose(region)):
            if n.lstrip("-") not in IGNORE and as_float(n) not in covered:
                warns.append(f"{pid}: number {n} in the prose is not covered by a CHECK line")
    for i, line in enumerate(COMMENT.sub("", src).splitlines(), 1):
        if re.search(r"\bTODO\b|\bTBD\b|\bXXX\b", line):
            fails.append(f"placeholder left in the published text: {line.strip()[:80]}")
    print(f"{path}: {n_data} data checks, {len(manual)} manual, {len(fails)} failed, {len(warns)} warnings")
    for f in fails:
        print("  FAIL", f)
    for w in warns:
        print("  warn", w)
    if show_manual:
        for x in manual:
            print("  manual", x)
    return 1 if fails else 0


def strip(path: Path) -> str:
    if path.suffix == ".tex":   # drop only the % CHECK blocks; LaTeX comments are invisible in the PDF anyway
        out, skip = [], False
        for line in path.read_text().split("\n"):
            if re.match(r"\s*%\s*CHECK\s+\S+", line):
                skip = True
                continue
            if skip and line.lstrip().startswith("%"):
                continue
            skip = False
            out.append(line)
        return re.sub(r"\n{3,}", "\n\n", "\n".join(out))
    out = COMMENT.sub("", path.read_text())
    return re.sub(r"\n{3,}", "\n\n", out).lstrip("\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", type=Path)
    ap.add_argument("--strip", action="store_true", help="print the file without any <!-- --> notes")
    ap.add_argument("--manual", action="store_true", help="also list the manual checks")
    args = ap.parse_args()
    if args.strip:
        sys.stdout.write(strip(args.file))
        return
    sys.exit(check(args.file, args.manual))


if __name__ == "__main__":
    main()
