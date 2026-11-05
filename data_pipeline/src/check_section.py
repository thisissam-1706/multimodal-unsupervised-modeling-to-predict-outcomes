"""Check that every number in report/section_data.md comes from the report it cites.

    python data_pipeline/src/check_section.py [path-to-section.md]

Rules
  * a citation (`name.md`, optionally `§N`; several in one bracket form one group) covers the text since the previous citation;
  * every number in that text must appear in a report of the group (thousands separators ignored); numbers after the last citation of a line fail;
  * a cited `§N` must exist as a `## N.` heading of that report.
Headings, table separators, code spans (`...`) and identifier-like tokens (v2, HbA1c, access_to_care_3) are not numbers.
Exit code 0 = clean, 1 = problems. No data is read: it compares text with text.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REPORTS = REPO / "data_pipeline" / "reports"
SECTION = REPO / "report" / "section_data.md"

CITE = re.compile(r"\b([a-z_0-9]+\.md)\b((?:\s*§\d+(?:\s*,\s*)?)*)")
NUM = re.compile(r"(?<![\w.§])(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?:th|%)?(?![\w])")


def _norm(text: str) -> str:
    return re.sub(r"(?<=\d),(?=\d{3})", "", text)


def _has(num: str, text: str) -> bool:
    return re.search(r"(?<![\d.])" + re.escape(num) + r"(?!\d|\.\d)", text) is not None


def _strip(line: str) -> str:
    line = re.sub(r"`[^`]*`", " ", line)                     # code spans
    line = CITE.sub(" ", line)                                # citations
    line = re.sub(r"§\d+", " ", line)
    line = re.sub(r"\b[A-Za-z_]+\d[\w]*\b", " ", line)        # v2, HbA1c, access_to_care_3 ...
    line = re.sub(r"\bv\d+(\.\d+)*\b", " ", line)
    return line


def _groups(raw: str):
    """Citation groups on a line: [(start, end, [(file, '§..')...])]. Citations separated only by ; , [ ] and spaces form ONE group."""
    ms = list(CITE.finditer(raw))
    groups = []
    for m in ms:
        if groups and re.fullmatch(r"[\s;,\[\]()]*", raw[groups[-1][1]:m.start()]):
            groups[-1][1] = m.end()
            groups[-1][2].append((m.group(1), m.group(2)))
        else:
            groups.append([m.start(), m.end(), [(m.group(1), m.group(2))]])
    return groups


def check(section: Path = SECTION, reports: Path = REPORTS) -> list[str]:
    """Claim-level: a citation covers the text between the previous citation (or the start of the line) and itself."""
    problems, texts, heads = [], {}, {}

    def load(f):
        if f not in texts:
            texts[f] = _norm((reports / f).read_text())
            heads[f] = set(re.findall(r"^## (\d+)\.", texts[f], flags=re.M))

    in_code = False
    for n, raw in enumerate(section.read_text().splitlines(), 1):
        if raw.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code or raw.startswith("#") or re.fullmatch(r"\|[\s\-|:]*\|?", raw.strip() or "x"):
            continue
        groups = _groups(raw)
        prev = 0
        for start, end, cites in groups:
            files = [f for f, _ in cites if (reports / f).is_file()]
            for f in files:
                load(f)
            for f, secs in cites:
                for sec in re.findall(r"§(\d+)", secs):
                    if f in heads and sec not in heads[f]:
                        problems.append(f"line {n}: {f} has no section {sec}")
            nums = [_norm(m.group(1)) for m in NUM.finditer(_strip(raw[prev:start]))]
            for num in nums:
                if not files:
                    problems.append(f"line {n}: number {num} is cited to no report file: {raw[:80]!r}")
                elif not any(_has(num, texts[f]) for f in files):
                    problems.append(f"line {n}: number {num} not found in cited {sorted(set(files))}: {raw[max(0, prev):start][-80:]!r}")
            prev = end
        tail = [_norm(m.group(1)) for m in NUM.finditer(_strip(raw[prev:]))]
        if tail:
            problems.append(f"line {n}: numbers {tail[:4]} come after the last citation (cite each claim): {raw[prev:][:80]!r}")
    return problems


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else SECTION
    probs = check(path)
    if probs:
        print(f"FAIL: {len(probs)} problems in {path}")
        for p in probs:
            print("  -", p)
        sys.exit(1)
    print(f"OK: every number in {path.name} appears in the report it cites")
