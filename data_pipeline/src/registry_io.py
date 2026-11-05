"""Read / write configs/feature_registry.yaml with a stable layout (so reruns are byte-identical)."""
from __future__ import annotations

from pathlib import Path

import yaml

HEADER = """\
# Phase 1 feature registry (plan section 4). ONE ENTRY PER FIELD.
#
#   status: verified | dropped, with `reason`. Set by `python data_pipeline/src/coverage.py --update-registry`
#           after checking the entry against the real v2.0.0 files (exists, units, range, coverage). Never edited to hide a failure.
#   source: exact selector string in the real data (measurement_source_value / observation_source_value). Observation strings are
#           truncated to 49 characters in the source files themselves. plan_source = string used by the plan/stub, kept for traceability.
#   range:  clinical plausibility bounds [lo, hi], fixed before modelling. Values outside become empty cells with reason qc_failed.
#   since:  first handoff version that includes the entry (1 = v1, 2 = v2). v1 is rebuilt from the entries with since <= 1.
#   hold:   explicit decision to keep an entry in the registry but not build it (status dropped, reason = the hold text).
#   codebook: what was found about the item's response labels (none are in the local payload) and how its codes are treated.
#   candidates: plan-listed fields NOT in the primary registry. They are never built until the user approves them.
# Never add: study_group, recommended_split, ids, dates, clinical_site, modality flags, mhterm_*, mh_dm_age, mh_a1c, cmtrt_*, paid*, dml*,
# diabetes diagnosis / medication rows.
"""

KEY_ORDER = ["name", "group", "kind", "dtype", "unit", "range", "table", "source", "code", "plan_source", "label", "omop",
             "temporal_rule", "missingness", "leakage", "codebook", "since", "hold", "status", "reason"]


def _ordered(entry: dict) -> dict:
    out = {k: entry[k] for k in KEY_ORDER if k in entry}
    out.update({k: v for k, v in entry.items() if k not in out})
    return out


class _D(yaml.SafeDumper):
    def ignore_aliases(self, data):  # no &id001 anchors for repeated text
        return True


_D.add_representer(list, lambda d, v: d.represent_sequence("tag:yaml.org,2002:seq", v,
                                                           flow_style=all(not isinstance(x, (dict, list)) for x in v)))


def dump_registry(path: Path, reg: dict) -> None:
    reg = dict(reg)
    reg["features"] = [_ordered(e) for e in reg["features"]]
    body = yaml.dump(reg, Dumper=_D, sort_keys=False, allow_unicode=True, width=140, default_flow_style=False)
    text = HEADER + body
    if path.exists() and path.read_text() == text:
        return
    tmp = path.with_suffix(".yaml.tmp")
    tmp.write_text(text)
    tmp.replace(path)


def load_registry(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text())
