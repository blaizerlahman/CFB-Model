"""Weekly SP+ ratings from Bill Connelly's public season spreadsheets.

Connelly publishes a Google Sheet per season (linked from his weekly posts)
with a tab per week. From 2025 those weekly tabs carry the full FBS ratings
table alongside that week's picks, and the ratings shown are the ones the
picks were made from — i.e. the values that were current *before* the week's
games. That makes them a first-party, pre-kickoff source, and it reaches
week 1, which the Wayback route cannot (the ESPN article is first archived
after week 1 kicks off).

Earlier seasons' sheets exist but their weekly tabs hold only game picks;
ratings appear just for preseason and final, so they do not yield a weekly
history.
"""

from __future__ import annotations

import csv
import io
import re

import pandas as pd
import requests

from cfb_model.api.mapping import normalize_school

# Season -> sheet id, taken from Connelly's own links.
SEASON_SHEETS = {
    2025: "1a6hboWNnPeUzx5oUEjwJwf9vw4DuAV7lTaW4Q92Zpls",
    2024: "1CJImfkg0ouHIIIGOWRfbvwC0TNWh76n47xkz8nqrVBc",
    2022: "1llrN8luL0XWuP8Y-Pb1NXKU84JhXLeUPafy1RfITEDw",
}

TAB_RE = re.compile(r'items\.push\(\{name:\s*"([^"]+)".*?gid:\s*"(\d+)"')
WEEK_RE = re.compile(r"week\s*(\d+)", re.I)

RATING_COLUMNS = ["Team", "Conference", "Record", "SP+"]

# Connelly's sheets use their own shorthand for a handful of schools; map to
# the CFBD names the rest of the pipeline keys on before normalize_school.
SHEET_NAME_FIXES = {
    "Miami-FL": "Miami",
    "Miami-OH": "Miami (OH)",
    "Hawaii": "Hawai'i",
    "San Jose State": "San José State",
    "UL-Lafayette": "Louisiana",
    "UL-Monroe": "UL Monroe",
    "USF": "South Florida",
}


def _session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = "cfb-model-sp-sheets (personal research)"
    return s


def discover_tabs(sheet_id: str, session: requests.Session | None = None) -> dict[str, str]:
    """{tab name: gid} for a public sheet."""
    session = session or _session()
    resp = session.get(f"https://docs.google.com/spreadsheets/d/{sheet_id}/htmlview", timeout=90)
    resp.raise_for_status()
    return {name: gid for name, gid in TAB_RE.findall(resp.text)}


def fbs_week_tabs(tabs: dict[str, str]) -> dict[int, str]:
    """FBS weekly tabs keyed by week number, tolerating both naming styles
    ('FBS Week 7' and 'Week 7 FBS')."""
    out: dict[int, str] = {}
    for name, gid in tabs.items():
        if "fbs" not in name.lower():
            continue
        match = WEEK_RE.search(name)
        if match:
            out[int(match.group(1))] = gid
    return out


def fetch_tab(sheet_id: str, gid: str, session: requests.Session | None = None) -> list[list[str]]:
    session = session or _session()
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
    resp = session.get(url, timeout=90)
    resp.raise_for_status()
    return list(csv.reader(io.StringIO(resp.text)))


def parse_ratings(rows: list[list[str]]) -> pd.DataFrame:
    """Pull the Team/SP+ block out of a weekly tab.

    Weekly tabs place the picks table on the left and the ratings table to
    its right, so the ratings block is located by finding its 'Team' header
    rather than by a fixed column offset.
    """
    if not rows:
        return pd.DataFrame(columns=["Team", "Rating"])
    header = rows[0]
    starts = [i for i, c in enumerate(header) if c.strip() == "Team"]
    if not starts:
        return pd.DataFrame(columns=["Team", "Rating"])
    start = starts[0]
    try:
        sp_offset = next(i for i, c in enumerate(header[start:], start)
                         if c.strip() in ("SP+", "SP+ Rating"))
    except StopIteration:
        return pd.DataFrame(columns=["Team", "Rating"])

    records = []
    for row in rows[1:]:
        if len(row) <= sp_offset:
            continue
        team = row[start].strip()
        rating = row[sp_offset].strip()
        if not team or not rating:
            continue
        try:
            value = float(rating)
        except ValueError:
            continue
        team = SHEET_NAME_FIXES.get(team, team)
        records.append({"Team": normalize_school(team), "Rating": value})
    return pd.DataFrame(records, columns=["Team", "Rating"]).drop_duplicates(subset=["Team"])


def weekly_ratings(season: int, sheet_id: str | None = None, log=print) -> dict[int, pd.DataFrame]:
    """{week: ratings} for every FBS weekly tab that actually carries ratings."""
    sheet_id = sheet_id or SEASON_SHEETS.get(season)
    if not sheet_id:
        raise ValueError(f"No known SP+ sheet for {season}; pass sheet_id explicitly.")
    session = _session()
    tabs = discover_tabs(sheet_id, session)
    weeks = fbs_week_tabs(tabs)
    log(f"{season}: {len(tabs)} tabs, {len(weeks)} FBS weekly tabs")

    out: dict[int, pd.DataFrame] = {}
    for week in sorted(weeks):
        ratings = parse_ratings(fetch_tab(sheet_id, weeks[week], session))
        if ratings.empty:
            log(f"  week {week:>2}: picks only, no ratings block")
            continue
        out[week] = ratings
        log(f"  week {week:>2}: {len(ratings)} teams")
    return out
