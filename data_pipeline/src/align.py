"""Index-visit alignment: the single implementation of configs/index_rule.yaml.

attach() labels every event row with the path it took and whether it belongs to the index encounter. Pure function,
no I/O, no labels read: restartable and idempotent by construction.

Rule
  1. path "linked": visit_occurrence_id exists in visit_occurrence.csv AND belongs to the same person
        -> aligned iff that visit starts on study_visit_date
  2. any other path (id not in the visit table / id of another person / no id): same-calendar-day fallback
        -> aligned iff the row's own date == study_visit_date
  3. in both cases a row whose OWN date is after study_visit_date is rejected (post-index)
  anything else is not aligned: post_index, other_day (a different day, before the index date) or no_date (unreadable date)
"""
from __future__ import annotations

import pandas as pd

ALIGNED, POST, OTHER, NODATE = "aligned", "post_index", "other_day", "no_date"
LINKED = "linked"
PATHS = [LINKED, "id_not_in_visit_table", "id_belongs_to_other_person", "no_visit_id"]


def index_table(participants: pd.DataFrame) -> pd.DataFrame:
    idx = participants[["participant_id", "study_visit_date"]].rename(
        columns={"participant_id": "person_id", "study_visit_date": "index_date"})
    idx["index_date"] = pd.to_datetime(idx["index_date"], errors="raise")
    return idx


def attach(events: pd.DataFrame, date_col: str, visits: pd.DataFrame, index: pd.DataFrame) -> pd.DataFrame:
    """Return the events of cohort participants plus:

    path       linked | id_not_in_visit_table | id_belongs_to_other_person | no_visit_id
    link       'visit_id' for path linked, else 'same_day' (which rule path decided the row)
    off_visit  days from the index date to the linked visit's start (NaN when not linked)
    off_own    days from the index date to the row's own date
    off_ref    the offset that decides alignment (off_visit if linked else off_own)
    status     aligned | post_index | other_day | no_date
    """
    ev = events.merge(index, on="person_id", how="inner")
    ev["event_date"] = pd.to_datetime(ev[date_col], errors="coerce")

    v = visits[["visit_occurrence_id", "person_id", "visit_start_date"]].rename(
        columns={"person_id": "v_person", "visit_start_date": "v_date"})
    v["v_date"] = pd.to_datetime(v["v_date"], errors="coerce")
    ev = ev.merge(v.drop_duplicates("visit_occurrence_id"), on="visit_occurrence_id", how="left")

    path = pd.Series(LINKED, index=ev.index)
    path[ev["v_person"].notna() & (ev["v_person"] != ev["person_id"])] = "id_belongs_to_other_person"
    path[ev["v_person"].isna()] = "id_not_in_visit_table"
    path[ev["visit_occurrence_id"].isna()] = "no_visit_id"
    ev["path"] = path
    linked = path == LINKED
    ev["link"] = linked.map({True: "visit_id", False: "same_day"})

    ev["off_own"] = (ev["event_date"] - ev["index_date"]).dt.days
    ev["off_visit"] = (ev["v_date"] - ev["index_date"]).dt.days.where(linked)
    ev["off_ref"] = ev["off_visit"].where(linked, ev["off_own"])

    status = pd.Series(OTHER, index=ev.index)
    status[(ev["off_ref"] == 0) & (ev["off_own"] <= 0)] = ALIGNED
    status[(ev["off_own"] > 0) | (ev["off_ref"] > 0)] = POST
    status[ev["event_date"].isna() | ev["off_ref"].isna()] = NODATE
    ev["status"] = status
    return ev.drop(columns=["v_person", "v_date"])
