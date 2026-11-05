"""A8: report/section_data.md may only contain numbers that appear in the report it cites.
The checker itself is tested by corrupting the section and expecting it to complain."""
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import check_section as CS  # noqa: E402

SECTION = CS.SECTION
REPORTS = CS.REPORTS


def _clean(s: str) -> str:
    s = re.sub(r"[`*]", "", s).strip()
    return re.sub(r"(?<=\d),(?=\d{3})", "", s)


def _tables(text: str):
    rows = []
    for line in text.splitlines():
        t = line.strip()
        if t.startswith("|") and not re.fullmatch(r"\|[\s\-|:]*\|?", t):
            rows.append([_clean(c) for c in t.strip("|").split("|")])
    return rows


def test_real_section_only_uses_numbers_from_its_cited_reports():
    assert CS.check() == []


def _mutated(tmp_path, old, new, count=1):
    text = SECTION.read_text()
    assert old in text, f"fixture text not found: {old!r}"
    p = tmp_path / "section.md"
    p.write_text(text.replace(old, new, count))
    return p


def test_checker_catches_a_changed_number(tmp_path):
    p = _mutated(tmp_path, "1,067 rows with 1,067 unique", "1,068 rows with 1,067 unique")
    assert any("1068" in x for x in CS.check(p))


def test_checker_catches_a_changed_table_cell(tmp_path):
    p = _mutated(tmp_path, "| insulin | 58 | 33 | 39 | 130 | data_audit.md §2 |", "| insulin | 59 | 33 | 39 | 130 | data_audit.md §2 |")
    assert any("59" in x for x in CS.check(p))


def test_checker_catches_a_table_row_without_a_citation(tmp_path):
    p = _mutated(tmp_path, "| insulin | 58 | 33 | 39 | 130 | data_audit.md §2 |", "| insulin | 58 | 33 | 39 | 130 |  |")
    assert any("cited to no report" in x or "after the last citation" in x for x in CS.check(p))


def test_checker_catches_a_claim_that_lost_its_citation_inside_a_multi_citation_paragraph(tmp_path):
    # the last claim of the paragraph loses its citation: its numbers must NOT be validated against the paragraph's other citation
    p = _mutated(tmp_path, "Of the empty lab cells, 60.4% are `not_aligned`, 38.7% `not_collected` (no lab-draw visit, or a blank result) and 0.9% `qc_failed` [missingness_report.md §5].",
                 "Of the empty lab cells, 60.4% are `not_aligned`, 38.7% `not_collected` (no lab-draw visit, or a blank result) and 0.9% `qc_failed`.")
    assert any("after the last citation" in x for x in CS.check(p))


def test_a_citation_does_not_cover_numbers_that_belong_to_another_report(tmp_path):
    # 26.2% is in missingness_report.md; attributing it to data_audit.md must fail even though other sentences cite missingness_report.md
    p = _mutated(tmp_path, "the missing share is 26.2% for insulin, 17.5% for healthy, 13.2% for pre-diabetes and 13.3% for oral/non-insulin [missingness_report.md §5]",
                 "the missing share is 26.2% for insulin, 17.5% for healthy, 13.2% for pre-diabetes and 13.3% for oral/non-insulin [data_audit.md §5]")
    assert any("26.2" in x for x in CS.check(p))


def test_checker_catches_citing_the_wrong_report(tmp_path):
    p = _mutated(tmp_path, "| healthy | 292 | 40 | 40 | 372 | data_audit.md §2 |", "| healthy | 292 | 40 | 40 | 372 | index_rule_measurement.md §2 |")
    assert any("not found in cited" in x for x in CS.check(p))


def test_checker_catches_a_nonexistent_section(tmp_path):
    p = _mutated(tmp_path, "[missingness_report.md §6]", "[missingness_report.md §9]")
    assert any("no section 9" in x for x in CS.check(p))


def test_checker_ignores_identifiers_and_versions(tmp_path):
    p = tmp_path / "s.md"
    p.write_text("The `access_to_care_6` feature of v2 and v2.0.0 uses HbA1c and CO2 [data_audit.md §2].\n")
    assert CS.check(p) == []


def test_table_rows_match_the_same_row_of_the_cited_report():
    """Stronger than 'the number exists somewhere': each section table row must equal a row of its source report with the same first cell."""
    matched = 0
    sec_rows = []
    for line in SECTION.read_text().splitlines():
        t = line.strip()
        if t.startswith("|") and not re.fullmatch(r"\|[\s\-|:]*\|?", t):
            sec_rows.append([c.strip() for c in t.strip("|").split("|")])
    for cells in sec_rows:
        m = re.match(r"([a-z_0-9]+\.md)", cells[-1])
        if not m or not (REPORTS / m.group(1)).is_file() or cells[0] in ("study_group", "table", "group", "reason"):
            continue
        key, rest = _clean(cells[0]), [_clean(c) for c in cells[1:-1]]
        cands = [r for r in _tables((REPORTS / m.group(1)).read_text()) if r and r[0] == key]
        if not cands:
            continue                                                       # composite tables (e.g. rows by path) are covered by the number check
        assert any(all(c in r for c in rest) for r in cands), f"{cells} does not match any {key!r} row of {m.group(1)}"
        matched += 1
    assert matched >= 25, f"only {matched} table rows were matched against their report"


def test_every_cited_report_exists_and_the_section_has_the_requested_headings():
    text = SECTION.read_text()
    for f in set(re.findall(r"\b([a-z_0-9]+\.md)\b", text)):
        if f not in ("section_data.md",):
            assert (REPORTS / f).is_file() or f == "README.md", f
    heads = [h.lower() for h in re.findall(r"^## \d+\. (.+)$", text, flags=re.M)]
    for want in ("dataset and release", "cohort and splits", "feature registry", "index alignment and repeat handling", "missingness",
                 "leakage controls", "limitations"):
        assert want in heads, want
    for limitation in ("small n", "class imbalance", "cross-sectional target", "no raw-modality results"):
        assert limitation in text.lower(), limitation
    assert "count discrepancy" in text.lower() and "364" in text and "331" in text
