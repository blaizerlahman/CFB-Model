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


def snapshot_deadlines(game_records: list[dict]) -> dict[int, datetime]:
    """week -> the instant that week's genuinely first game kicks off.

    A rating may be used to predict week N only if it was public before any
    week-N game started; anything later can already encode results from the
    week being predicted.

    This is deliberately computed from actual game start times rather than the
    calendar's `firstGameStart`, which is a window boundary sitting ~1-2 days
    before the real opener (2024 week 2: boundary Sep 3 07:00Z, first kickoff
    Sep 4 23:00Z). Using the boundary rejects the Tuesday rating update that a
    real bettor would have had all week, leaving snapshots a full publication
    cycle stale.
    """
    earliest: dict[int, datetime] = {}
    for game in game_records:
        if game.get("seasonType") != "regular" or not game.get("startDate"):
            continue
        week = game.get("week")
        if week is None:
            continue
        start = datetime.fromisoformat(game["startDate"].replace("Z", "+00:00"))
        if week not in earliest or start < earliest[week]:
            earliest[week] = start
    return earliest


def backfill_sp(store: Store, year: int, game_records: list[dict],
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

    deadlines = snapshot_deadlines(game_records)
    chosen: dict[int, str] = {}
    missing: list[int] = []

    for week, deadline in sorted(deadlines.items()):
        # Freshest ratings that were public BEFORE this week's first kickoff.
        eligible = [
            ts for ts in captures
            if datetime.strptime(ts, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc) < deadline
        ]
        if not eligible:
            missing.append(week)
            continue
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
        log(
            f"WARNING: no pre-kickoff SP+ capture exists for {year} weeks {missing} "
            "— those weeks have NO snapshot and must be excluded from backtests. "
            "Final-season ratings are deliberately NOT substituted."
        )
    return chosen


# ---- manual import of saved ESPN pages -------------------------------------

WEEK_FILE_RE = re.compile(r"week[\s_-]*(\d+)", re.I)


def import_saved_html(store: Store, year: int, directory, log=print) -> dict[int, int]:
    """Import SP+ ratings from ESPN pages saved by hand.

    ESPN's 2019-2023 weekly rankings sit behind ESPN+ and the site refuses
    server-side fetches outright (HTTP 202, empty body), so those seasons can
    only be collected by a signed-in human saving each page. This reads that
    folder.

    Files are keyed by the week in their NAME, which is the week the article
    reports on ("after Week 5" -> week5.html). Those ratings were computed
    once week 5 finished, so they are the ones in hand for week 6 and are
    stored under week 6 — storing them under week 5 would leak week-5 results
    into week-5 predictions. Preseason (week0.html) therefore lands on week 1.

    Returns {stored_week: team_count}.
    """
    from pathlib import Path

    directory = Path(directory)
    stored: dict[int, int] = {}
    files = sorted(p for p in directory.glob("*.htm*"))
    if not files:
        log(f"No .html files in {directory}")
        return stored

    for path in files:
        match = WEEK_FILE_RE.search(path.stem)
        if not match:
            log(f"  {path.name}: no week number in the filename, skipped")
            continue
        article_week = int(match.group(1))
        target_week = article_week + 1

        table = parse_sp_table(path.read_text(errors="replace"))
        if len(table) < 100:
            log(f"  {path.name}: only {len(table)} teams parsed — is the page fully "
                "loaded and signed in? skipped")
            continue
        store.upsert_sp(year, target_week, table)
        stored[target_week] = len(table)
        log(f"  {path.name}: after week {article_week} -> stored as {year} week "
            f"{target_week} ({len(table)} teams)")
    return stored


def import_scraped_json(store: Store, path, year: int | None = None, log=print) -> dict[int, int]:
    """Import the JSON produced by scripts/espn_sp_scrape.js.

    Each entry records the week its article reports on; the ratings were
    computed once that week finished, so they are stored under the FOLLOWING
    week — the first week they could actually have been bet on.

    Team names arrive exactly as ESPN prints them and go through the same
    normalisation as the Wayback path, so both routes land on identical keys.
    """
    import json
    from pathlib import Path

    payload = json.loads(Path(path).read_text())
    year = year or payload.get("year")
    if not year:
        raise ValueError("No year in the file; pass --year explicitly.")

    stored: dict[int, int] = {}
    for entry in payload.get("articles", []):
        if entry.get("error"):
            log(f"  {entry['url']}: {entry['error']}")
            continue
        article_week = entry.get("week")
        rows = entry.get("rows") or []
        if article_week is None:
            log(f"  {entry['url']}: no week identified, skipped")
            continue
        if len(rows) < 100:
            log(f"  week {article_week}: only {len(rows)} teams — paywalled or partial, skipped")
            continue

        records = []
        for row in rows:
            name = str(row.get("team", "")).strip().rstrip(";").strip()
            name = re.sub(r"\s+St\.$", " State", name)
            name = ESPN_NAME_FIXES.get(name, name)
            records.append({"Team": normalize_school(name), "Rating": float(row["rating"])})
        table = pd.DataFrame(records).drop_duplicates(subset=["Team"])

        target_week = article_week + 1
        store.upsert_sp(year, target_week, table)
        stored[target_week] = len(table)
        log(f"  after week {article_week} -> {year} week {target_week} ({len(table)} teams)")
    return stored
