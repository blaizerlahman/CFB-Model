/* Collect weekly SP+ ratings from the ESPN articles your subscription covers.
 *
 * ESPN answers server-side requests with an empty 202, so these pages can only
 * be read by a signed-in browser. Run this in the DevTools console ON an
 * espn.com tab while logged in: the fetches are same-origin, so your session
 * rides along and the full article body comes back.
 *
 * Covers every season that still needs weekly ratings (2019-2023). 2024 and
 * 2025 are already stored from other sources and are deliberately skipped.
 *
 * It saves ONE json for all seasons. Chrome will offer a folder picker —
 * choose the project's  output/sp_manual  folder and the importer will find it
 * with no arguments:
 *
 *     python -m cfb_model import-sp-json
 *
 * If the picker is unavailable it falls back to your Downloads folder, which
 * the importer also checks.
 */
(async () => {
  const DELAY_MS = 4000;          // gentle with ESPN; a burst gets throttled
  const OUTFILE = "cfb_sp_plus_backfill.json";

  // Article ids per season. The week noted is the week the article REPORTS ON;
  // its ratings are the ones in hand for the following week, and the importer
  // applies that shift. Add any you find: only the id matters, the slug is
  // ignored by ESPN.
  const KNOWN = {
    2023: {
      0: "38196497", 1: "38332658", 2: "38368917", 3: "38422011", 4: "38478223",
      5: "38538922", 7: "38663643", 8: "38718073", 9: "38727877", 10: "38823946",
      11: "38881198", 12: "38934718", 13: "38983348",
      // week 6 not located; see the note printed at the end.
    },
    2022: { 2: "34569954" },
    2021: {},
    2020: {},
    2019: { 6: "27781603" },
  };

  const TEAM_CELL = /^\s*(\d{1,3})\.\s*(.+?)(?:\s*\((\d+-\d+)\))?\s*$/;
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const idUrl = (id) => `https://www.espn.com/college-football/insider/story/_/id/${id}`;

  // ESPN's search barely indexes older SP+ pieces, but it costs one request
  // per season and quietly fills gaps when it does return something.
  async function discover(year) {
    const found = {};
    const query = encodeURIComponent(`college football ${year} week sp+ rankings`);
    try {
      // Public endpoint: it answers with Access-Control-Allow-Origin: *, which
      // the browser refuses to combine with credentials, so omit them here.
      const resp = await fetch(
        `https://site.web.api.espn.com/apis/search/v2?region=us&lang=en&limit=50&query=${query}`,
        { credentials: "omit" });
      if (!resp.ok) return found;
      const data = await resp.json();
      for (const group of data.results || []) {
        for (const item of group.contents || []) {
          const link = (item.link || {}).web || "";
          const m = link.match(/\/id\/(\d+)\/([^/?#]*)/);
          if (!m || !/sp\+/i.test(m[2])) continue;
          const yr = m[2].match(/(20\d\d)/);
          if (!yr || Number(yr[1]) !== year) continue;
          const wk = m[2].match(/week-(\d+)/);
          found[wk ? Number(wk[1]) : 0] = m[1];
        }
      }
    } catch (_) { /* discovery is best-effort */ }
    return found;
  }

  // The ratings table is the biggest run of "N. Team" rows; shorter runs are
  // the strength-of-schedule and resume sidebars.
  function extractRatings(doc) {
    let best = [];
    for (const table of doc.querySelectorAll("table")) {
      const rows = [];
      for (const tr of table.querySelectorAll("tr")) {
        const cells = [...tr.querySelectorAll("td")].map((td) => td.textContent.trim());
        if (cells.length < 2) continue;
        const m = TEAM_CELL.exec(cells[0]);
        if (!m) continue;
        const rating = parseFloat(cells[1].replace("+", ""));
        if (Number.isNaN(rating)) continue;
        rows.push({ rank: parseInt(m[1], 10), team: m[2].trim(), rating });
      }
      if (rows.length > best.length) best = rows;
    }
    return best;
  }

  const payload = { captured: new Date().toISOString(), seasons: [] };
  let totalOk = 0, totalTried = 0;

  for (const year of Object.keys(KNOWN).map(Number).sort()) {
    const ids = { ...KNOWN[year] };
    const discovered = await discover(year);
    let added = 0;
    for (const [wk, id] of Object.entries(discovered)) {
      if (!(wk in ids)) { ids[wk] = id; added++; }
    }
    const weeks = Object.keys(ids).map(Number).sort((a, b) => a - b);
    console.log(`\n=== ${year}: ${weeks.length} articles` +
                (added ? ` (${added} found by search)` : "") + ` ===`);
    if (!weeks.length) { console.log("  none known — add ids to KNOWN above"); continue; }

    const season = { year, articles: [] };
    for (const week of weeks) {
      const url = idUrl(ids[week]);
      totalTried++;
      try {
        const resp = await fetch(url, { credentials: "include" });
        if (!resp.ok) {
          console.warn(`  week ${week}: HTTP ${resp.status}`);
          season.articles.push({ url, week, error: `HTTP ${resp.status}` });
        } else {
          const doc = new DOMParser().parseFromString(await resp.text(), "text/html");
          const rows = extractRatings(doc);
          const title = (doc.querySelector("title") || {}).textContent || "";
          season.articles.push({ url, week, title, count: rows.length, rows });
          if (rows.length >= 100) {
            totalOk++;
            console.log(`  week ${week}: ${rows.length} teams  ok`);
          } else {
            console.warn(`  week ${week}: only ${rows.length} teams — paywalled or not ` +
                         `rendered; it will be skipped on import`);
          }
        }
      } catch (err) {
        console.warn(`  week ${week}: ${err}`);
        season.articles.push({ url, week, error: String(err) });
      }
      await sleep(DELAY_MS);
    }
    payload.seasons.push(season);
  }

  console.log(`\ncollected ${totalOk}/${totalTried} articles with a full table`);
  if (!totalOk) {
    console.warn("Nothing usable. Are you signed in, and running this on an espn.com tab?");
    return;
  }
  const text = JSON.stringify(payload, null, 1);

  // Keep the payload reachable no matter what happens below.
  window.__SP_PLUS_JSON = text;
  console.log("payload also available as window.__SP_PLUS_JSON");

  function fallbackDownload() {
    // octet-stream so the browser saves rather than renders the JSON.
    const url = URL.createObjectURL(new Blob([text], { type: "application/octet-stream" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = OUTFILE;
    a.rel = "noopener";
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
  }

  // showSaveFilePicker needs a real user gesture, which a console run does not
  // have — so put a button on the page and let the click supply it.
  const btn = document.createElement("button");
  btn.textContent = `Save ${OUTFILE} (${totalOk} articles)`;
  Object.assign(btn.style, {
    position: "fixed", zIndex: 2147483647, top: "16px", right: "16px",
    padding: "14px 18px", fontSize: "15px", fontWeight: "600",
    background: "#0b7", color: "#fff", border: "none", borderRadius: "8px",
    cursor: "pointer", boxShadow: "0 2px 12px rgba(0,0,0,.35)",
  });
  btn.onclick = async () => {
    btn.disabled = true;
    try {
      if (window.showSaveFilePicker) {
        const handle = await window.showSaveFilePicker({
          suggestedName: OUTFILE,
          types: [{ description: "JSON", accept: { "application/json": [".json"] } }],
        });
        const w = await handle.createWritable();
        await w.write(text);
        await w.close();
        console.log(`saved ${OUTFILE} — now run:  python -m cfb_model import-sp-json`);
        btn.textContent = "Saved";
        return;
      }
    } catch (err) {
      console.warn("save picker unavailable or cancelled; downloading instead", err);
    }
    fallbackDownload();
    btn.textContent = "Downloaded";
    console.log(`downloaded ${OUTFILE} — now run:  python -m cfb_model import-sp-json`);
  };
  document.body.appendChild(btn);
  console.log("\nClick the green Save button at the top right of the page to write the file.");
  console.log("Choose the project's  output/sp_manual  folder and the importer needs no arguments.");
})();
