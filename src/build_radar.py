"""
Radar page builder (Agent 9)

Turns the latest monthly scan into the shared Research Radar page: every matched paper as a dot by topic and
score, the shortlist with notes, this quarter's pilot and the KRA count. `prospector.py scan` runs this
automatically; run it by hand after `pick` or `result` so the page shows the new pilot status.

    python radar/build_radar.py                         # latest scans/Prospector_*.xlsx
    python radar/build_radar.py --scan scans/Prospector_2026-10.xlsx

Writes radar/radar.html (self-contained, data embedded) plus a dated copy next to the scan. Open it in any
browser, or share the file on SharePoint / Teams / email.
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter
from datetime import datetime

import pandas as pd

CODE = os.path.dirname(os.path.abspath(__file__))   # src/
HERE = os.environ.get("PROSPECTOR_DATA") or os.path.join(os.path.dirname(CODE), "data")
TEMPLATE = os.path.join(CODE, "radar_template.html")
OUT = os.path.join(HERE, "radar", "radar.html")
DEFAULT_TOPICS = {   # radar slices: keyword lists matched against each paper's "Matched" keywords
    "LLM & Agents": ["agent", "structured output", "evaluation"],
    "Retrieval & Documents": ["retrieval", "rag", "document", "extraction", "ocr", "pdf", "table"],
    "Data Quality & Trust": ["data quality", "reconciliation", "anomaly detection", "hallucination", "explainable",
                             "synthetic data"],
    "Quant Research": ["backtest", "overfitting", "look-ahead", "factor", "forecasting", "time series",
                       "transaction cost"],
    "Risk & Valuation": ["risk", "volatility", "valuation", "private equity", "credit", "portfolio", "financial"],
}


def latest_scan():
    scans = sorted(glob.glob(os.path.join(HERE, "scans", "Prospector_*.xlsx")), key=os.path.getmtime)
    return scans[-1] if scans else None


def pilot_status(today):
    """This quarter's pilot (if picked) and techniques adopted this year, from pilots.csv."""
    quarter = f"{today.year}Q{(today.month - 1) // 3 + 1}"
    path = os.path.join(HERE, "pilots.csv")
    if not os.path.exists(path):
        return quarter, None, 0
    p = pd.read_csv(path, dtype=str, keep_default_na=False)
    adopted = int(((p["Outcome"] == "adopted") & p["Quarter"].str.startswith(str(today.year))).sum())
    cur = p[p["Quarter"] == quarter]
    if cur.empty:
        return quarter, None, adopted
    r = cur.iloc[-1]
    return quarter, {"id": r["arXiv ID"], "title": r["Title"] or r["arXiv ID"], "owner": r["Owner"], "plan": r["Plan"],
                     "outcome": "" if r["Outcome"] == "in pilot" else r["Outcome"], "result": r["Business result"], "metric": r["Metric"]}, adopted


def build(scan=None, out=OUT):
    scan = scan or latest_scan()
    if not scan:
        sys.exit("No scan found in scans/. Run: python prospector.py scan")
    sheets = pd.read_excel(scan, sheet_name=None, dtype=str)  # dtype=str keeps IDs like 2610.11370 intact
    sl, am = sheets["Shortlist"].fillna(""), sheets["All Matches"].fillna("")
    if "Area" not in sl.columns:  # scans made before areas existed
        sl["Area"], am["Area"] = "Quant", "Quant"
    sys.path.insert(0, CODE)
    from prospector import load_config  # interests.xlsx when present, else interests.json
    cfg = load_config()
    topics = cfg.get("radar_topics") or DEFAULT_TOPICS
    weights = cfg.get("interests", {})
    names = list(topics)

    group_of_work = {w["name"]: w.get("group", "") for w in cfg.get("work", [])}

    def topic_of(matched, closest=""):
        kws = [k.strip() for k in str(matched).split(",") if k.strip()]
        score = {t: sum(weights.get(k, 1) for k in kws if k in topics[t]) for t in names}
        best = max(score, key=score.get)
        if score[best] > 0:
            return names.index(best)
        g = group_of_work.get(closest, "")  # found by meaning only: the slice of the closest work
        return names.index(g) if g in names else 0

    area_names = list(dict.fromkeys(list(sl["Area"]) + list(am["Area"])))
    short = []
    for _, r in sl.iterrows():
        p = {k: r.get(k, "") for k in ["Area", "arXiv ID", "Title", "Published", "Matched", "What it is",
                                        "Use for us", "Data needed", "Effort", "Link", "Closest work", "Meaning",
                                        "Found by"]}
        p["Score"] = float(r["Score"] or 0)
        short.append(p)
    pts = [[r["arXiv ID"], round(float(r["Score"] or 0), 2), area_names.index(r["Area"]),
            topic_of(r["Matched"], r.get("Closest work", "")), r["Title"][:140], r["Published"],
            r["Matched"] or f"meaning: closest to {r.get('Closest work', '')}"] for _, r in am.iterrows()]
    counts = Counter(p[3] for p in pts)

    summary_path = os.path.splitext(scan)[0] + ".json"   # written by prospector.py scan
    summary = {}
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
    today = datetime.now()
    quarter, pilot, adopted = pilot_status(today)
    data = {
        "families": names, "famcount": [counts[i] for i in range(len(names))], "short": short, "pts": pts,
        "areaNames": area_names,
        "areas": summary.get("areas") or {a: [sum(p[2] == i for p in pts), sum(p[2] == i for p in pts)]
                                          for i, a in enumerate(area_names)},
        "feed": summary.get("feed", len(pts)), "fresh": summary.get("fresh", len(pts)), "matched": len(pts),
        "to": summary.get("to") or max((p[5] for p in pts), default=today.strftime("%Y-%m-%d")),
        "quarter": quarter, "year": today.year, "pilot": pilot, "adopted": adopted,
        "voteForm": cfg.get("vote_form", ""), "run": summary.get("run"),
        "month": summary.get("month") or os.path.basename(scan)[len("Prospector_"):len("Prospector_") + 7],
    }
    with open(TEMPLATE) as f:
        html = f.read()
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    page = ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1,viewport-fit=cover\">"
            "<style>body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style></head><body>"
            + html.replace("__DATA__", blob) + "</body></html>")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        f.write(page)
    if out == OUT:  # keep a dated copy next to the scan it came from
        with open(os.path.splitext(scan)[0] + "_radar.html", "w") as f:
            f.write(page)
    return out, data


def main():
    ap = argparse.ArgumentParser(description="Build the Research Radar page from a monthly scan")
    ap.add_argument("--scan", help="scan .xlsx (default: latest in scans/)")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    out, d = build(a.scan, a.out)
    if a.out == OUT:  # the shared page: also refresh the copy in the share folder
        from prospector import share_page
        share_page(out)
    pilot = f"pilot: {d['pilot']['title'][:50]}" if d["pilot"] else "pilot: not picked yet"
    print(f"Radar page: {len(d['short'])} shortlisted, {d['matched']} matched, {d['quarter']} {pilot}, "
          f"{d['adopted']} adopted in {d['year']}\n-> {out}")


if __name__ == "__main__":
    main()
