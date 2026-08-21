/* Collect weekly SP+ ratings from ESPN articles you have access to.
 *
 * ESPN refuses server-side requests, so these pages can only be read by a
 * signed-in browser. Run this in the DevTools console ON espn.com (any ESPN
 * page) while logged in: the fetches are same-origin, so your session cookie
 * rides along and the full article body comes back.
 *
 * It downloads one JSON file. Feed that to:
 *     python -m cfb_model import-sp-json --file sp_plus_<year>.json
 *
 * Edit YEAR and ARTICLES below for other seasons. Week numbers come from the
 * URL slug when present, otherwise from the article title.
 */
(async () => {
  const YEAR = 2023;
  const ARTICLES = [
    "https://www.espn.com/college-football/insider/story/_/id/38196497", // preseason
    "https://www.espn.com/college-football/insider/story/_/id/38332658", // after wk 1
    "https://www.espn.com/college-football/insider/story/_/id/38368917", // after wk 2
    "https://www.espn.com/college-football/insider/story/_/id/38422011", // after wk 3
    "https://www.espn.com/college-football/insider/story/_/id/38478223", // after wk 4
    "https://www.espn.com/college-football/insider/story/_/id/38538922", // after wk 5
    "https://www.espn.com/college-football/insider/story/_/id/38663643", // after wk 7
    "https://www.espn.com/college-football/insider/story/_/id/38718073", // after wk 8
    "https://www.espn.com/college-football/insider/story/_/id/38727877", // after wk 9
    "https://www.espn.com/college-football/insider/story/_/id/38823946", // after wk 10
    "https://www.espn.com/college-football/insider/story/_/id/38881198", // after wk 11
    "https://www.espn.com/college-football/insider/story/_/id/38934718", // after wk 12
    "https://www.espn.com/college-football/insider/story/_/id/38983348", // after wk 13
  ];

  const TEAM_CELL = /^\s*(\d{1,3})\.\s*(.+?)(?:\s*\((\d+-\d+)\))?\s*$/;
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  // The ratings table is the biggest run of "N. Team" rows on the page;
  // smaller runs are strength-of-schedule and resume sidebars.
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

  function weekOf(url, title) {
    const fromUrl = /week-(\d+)/i.exec(url);
    if (fromUrl) return parseInt(fromUrl[1], 10);
    const fromTitle = /week\s*(\d+)/i.exec(title || "");
    if (fromTitle) return parseInt(fromTitle[1], 10);
    if (/preseason/i.test(title || "") || /preseason/i.test(url)) return 0;
    return null;
  }

  const out = { year: YEAR, captured: new Date().toISOString(), articles: [] };

  for (const url of ARTICLES) {
    try {
      const resp = await fetch(url, { credentials: "include" });
      if (!resp.ok) {
        console.warn(`HTTP ${resp.status} for ${url}`);
        out.articles.push({ url, error: `HTTP ${resp.status}` });
        continue;
      }
      const doc = new DOMParser().parseFromString(await resp.text(), "text/html");
      const title = (doc.querySelector("title") || {}).textContent || "";
      // The canonical URL carries the slug even when we requested the bare id.
      const canonical = (doc.querySelector('link[rel="canonical"]') || {}).href || url;
      const rows = extractRatings(doc);
      const week = weekOf(canonical, title);

      out.articles.push({ url, canonical, title, week, count: rows.length, rows });
      const flag = rows.length >= 100 ? "ok" : "SUSPICIOUS - paywalled or not loaded?";
      console.log(`week ${week}: ${rows.length} teams  ${flag}  ${title.slice(0, 60)}`);
    } catch (err) {
      console.warn(`failed ${url}`, err);
      out.articles.push({ url, error: String(err) });
    }
    await sleep(1500); // be gentle; ESPN throttles bursts
  }

  const good = out.articles.filter((a) => (a.count || 0) >= 100).length;
  console.log(`\ncollected ${good}/${ARTICLES.length} articles with a full table`);
  if (!good) {
    console.warn("Nothing usable. Are you signed in, and running this on an espn.com tab?");
    return;
  }

  const blob = new Blob([JSON.stringify(out, null, 1)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `sp_plus_${YEAR}.json`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  console.log(`downloaded sp_plus_${YEAR}.json`);
})();
