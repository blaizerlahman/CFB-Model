"""Weekly SP+ snapshot backfill from Wayback Machine captures of ESPN's
living SP+ rankings article (user requirement: true week-by-week 2025 SP+,
never final-season values).

ESPN maintains ONE article per season with the full 136-team SP+ table,
updated every Tuesday. The Wayback Machine captured it near-daily through the
2025 season, and the full table is plain HTML. For each week W we take the
latest capture in the window (last game of week W-1, last game of week W] —
i.e., SP+ exactly as known while week-W predictions would have been made —
and store it as the (season, W) snapshot.

Zero CFBD API quota; requests go to web.archive.org only.
"""

from __future__ import annotations

import re
import time
from html import unescape as html_unescape
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

from cfb_model.api.mapping import normalize_school
from cfb_model.config import Settings, get_settings
from cfb_model.data.store import Store

# Season -> the ESPN living-article URL (find new seasons' ids by searching
# espn.com for "college football SP+ rankings for all 136 FBS teams").
SP_ARTICLE_URLS = {
    2025: "https://www.espn.com/college-football/story/_/id/46128861/2025-college-football-sp+-rankings-all-136-fbs-teams",
}

CDX_URL = "http://web.archive.org/cdx/search/cdx"

# ESPN display names / abbreviations -> CFBD names (applied after HTML
# unescaping and the trailing " St." -> " State" rule; normalize_school runs
# last). Built by diffing a parsed 2025 capture against the 129 FBS keys.
ESPN_NAME_FIXES = {
    "App. State": "App State",
    "BGSU": "Bowling Green",
    "Boston Coll.": "Boston College",
    "CMU": "Central Michigan",
    "Coastal Caro.": "Coastal Carolina",
    "ECU": "East Carolina",
    "EMU": "Eastern Michigan",
    "FAU": "Florida Atlantic",
    "FIU": "Florida International",
    "Ga. Southern": "Georgia Southern",
    "Ga. Tech": "Georgia Tech",
    "Hawaii": "Hawai'i",
    "J'ville State": "Jacksonville State",
    "Jax State": "Jacksonville State",
    "JMU": "James Madison",
    "JSU": "Jacksonville State",
    "Kennesaw": "Kennesaw State",
    "NIU": "Northern Illinois",
    "NMSU": "New Mexico State",
    "La. Tech": "Louisiana Tech",
    "Louisiana-Monroe": "UL Monroe",
    "Miami-OH": "Miami (OH)",
    "Miss. State": "Mississippi State",
    "MTSU": "Middle Tennessee",
    "N. Carolina": "North Carolina",
    "N. Illinois": "Northern Illinois",
    "N. Texas": "North Texas",
    "New Mex. State": "New Mexico State",
    "ODU": "Old Dominion",
    "S. Alabama": "South Alabama",
    "S. Carolina": "South Carolina",
    "San Jose State": "San José State",
    "So. Miss": "Southern Miss",
    "UL-Monroe": "UL Monroe",
    "ULM": "UL Monroe",
    "USF": "South Florida",
    "Va. Tech": "Virginia Tech",
    "W. Virginia": "West Virginia",
    "Wash. State": "Washington State",
    "WKU": "Western Kentucky",
    "WMU": "Western Michigan",
}

ROW_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S)
CELL_RE = re.compile(r"<td[^>]*>(.*?)</td>", re.S)
TEAM_RE = re.compile(r"^\s*(\d{1,3})\.\s*(.+?)(?:\s*\((\d+-\d+)\))?\s*$")
TAG_RE = re.compile(r"<[^>]+>")


def list_captures(year: int, session: requests.Session | None = None) -> list[str]:
    """All capture timestamps (yyyymmddHHMMSS) of the season's article from
    Aug 1 of the season through Jan 31 of the next year."""
    session = session or requests.Session()
    url = SP_ARTICLE_URLS[year]
    resp = session.get(
        CDX_URL,
        params={
            "url": url.replace("https://www.", "").replace("https://", ""),
            "from": f"{year}0801",
            "to": f"{year + 1}0131",
            "output": "json",
            "collapse": "timestamp:10",  # at most ~1 per hour
            "filter": "statuscode:200",
        },
        timeout=60,
    )
    resp.raise_for_status()
    rows = resp.json()
    return sorted(row[1] for row in rows[1:]) if len(rows) > 1 else []


def fetch_capture(timestamp: str, year: int, cache_dir: Path,
                  session: requests.Session | None = None) -> str:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"espn_sp_{year}_{timestamp}.html"
    if cached.exists():
        return cached.read_text(errors="replace")
    session = session or requests.Session()
    url = f"https://web.archive.org/web/{timestamp}/{SP_ARTICLE_URLS[year]}"
    resp = session.get(url, timeout=120)
    resp.raise_for_status()
    cached.write_text(resp.text, errors="replace")
    time.sleep(1)  # be polite to the archive
    return resp.text


def parse_sp_table(html: str) -> pd.DataFrame:
    """Extract the full rankings table -> DataFrame[Team, Rating] with CFBD
    canonical names. Returns the LARGEST parsed table in the page (the
    all-teams table; smaller tables are SOS/résumé sidebars)."""
    tables: list[list[tuple[str, float]]] = []
    current: list[tuple[str, float]] = []
    for row_html in ROW_RE.findall(html):
        cells = [TAG_RE.sub("", c).strip() for c in CELL_RE.findall(row_html)]
        if len(cells) < 2:
            continue
        m = TEAM_RE.match(cells[0])
        if not m:
            if current:
                tables.append(current)
                current = []
            continue
        name = html_unescape(m.group(2)).strip().rstrip(";").strip()
        name = re.sub(r"\s+St\.$", " State", name)
        name = ESPN_NAME_FIXES.get(name, name)
        name = normalize_school(name)
        try:
            rating = float(cells[1].replace("+", ""))
        except ValueError:
            continue
        current.append((name, rating))
    if current:
        tables.append(current)
    if not tables:
        return pd.DataFrame(columns=["Team", "Rating"])
    best = max(tables, key=len)
    df = pd.DataFrame(best, columns=["Team", "Rating"])
    return df.drop_duplicates(subset=["Team"], keep="first")


def week_windows(calendar_records: list[dict]) -> dict[int, tuple[datetime, datetime]]:
    """week -> (previous week's last game, this week's last game], regular
    season only. Week 1's window opens Aug 1."""
    regular = [e for e in calendar_records if e.get("seasonType") == "regular"]
    regular.sort(key=lambda e: e["week"])
    windows = {}
    prev_end = None
    for entry in regular:
        end = datetime.fromisoformat(entry["lastGameStart"].replace("Z", "+00:00"))
        start = prev_end or end.replace(month=8, day=1, hour=0, minute=0, second=0)
        # Defensive clamp: CFBD's calendar has typo'd years (2025 week 16
        # claims a 2026 end date); no regular-season week spans 30+ days.
        if prev_end is not None and (end - start).days > 30:
            end = start + timedelta(days=14)
        windows[entry["week"]] = (start, end)
        prev_end = end
    return windows


def backfill_sp(store: Store, year: int, calendar_records: list[dict],
                settings: Settings | None = None, log=print) -> dict[int, str]:
    """Fill sp_ratings(year, week) for every regular-season week from Wayback
    captures. Returns {week: capture_timestamp}. Raises if any week has no
    usable capture — per user requirement, final-season values are NEVER
    silently substituted."""
    settings = settings or get_settings()
    cache_dir = settings.db_path.parent / "sp_wayback_cache"
    session = requests.Session()
    session.headers["User-Agent"] = "cfb-model-sp-backfill (personal research)"

    captures = list_captures(year, session)
    if not captures:
        raise RuntimeError(f"No Wayback captures found for the {year} SP+ article")
    log(f"{len(captures)} Wayback captures of the {year} SP+ article")

    windows = week_windows(calendar_records)
    chosen: dict[int, str] = {}
    missing: list[int] = []

    for week, (start, end) in sorted(windows.items()):
        eligible = [
            ts for ts in captures
            if start < datetime.strptime(ts, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc) <= end
        ]
        if not eligible:
            missing.append(week)
            continue
        # Latest capture in the window = freshest ratings available while
        # that week's predictions would have been made.
        for ts in reversed(eligible):
            html = fetch_capture(ts, year, cache_dir, session)
            table = parse_sp_table(html)
            if len(table) >= 120:  # full FBS table (136 teams in 2025)
                store.upsert_sp(year, week, table)
                chosen[week] = ts
                log(f"  week {week:>2}: capture {ts} -> {len(table)} teams")
                break
        else:
            missing.append(week)

    if missing:
        raise RuntimeError(
            f"No usable SP+ capture for {year} weeks {missing} — STOP: do not "
            "substitute final-season ratings; present options to the user."
        )
    return chosen
