"""
Weekly summary: what the Prospector did for each person, shown on the "Weekly summary" tab of the web page.

Every question asked on the page (or with `ask --user`) is saved automatically in data/users.json:

    {"users": {"ekta": {"name": "Ekta", "asks": [{when, question, check, outcome, seconds, ai,
                                                   papers: [{id, title, link, confidence, label, low}]}]}}}

weekly(user, week) builds one person's week (Monday to Sunday) from that file plus the team votes:
questions asked, answered and unanswered, papers found by confidence, low-confidence answers to double-check,
questions whose input was corrected, votes, and an estimate of time saved. The estimate assumes
MANUAL_MINUTES (env PROSPECTOR_MANUAL_MINUTES, default 30) for one manual search; replace it with your measured
baseline. Nothing here calls the AI.
"""
import json
import os
import re
import threading
from datetime import datetime, timedelta

import prospector as P

MANUAL_MINUTES = float(os.environ.get("PROSPECTOR_MANUAL_MINUTES", "30"))
KEEP = 500           # most questions kept per person
NO_NAME = "_anonymous"
_lock = threading.Lock()


def users_file():
    return os.path.join(P.HERE, "users.json")


def key(name):
    return re.sub(r"\s+", " ", str(name or "")).strip().lower()[:60]


def load():
    try:
        with open(users_file()) as f:
            data = json.load(f)
        return data if isinstance(data.get("users"), dict) else {"users": {}}
    except (OSError, ValueError, AttributeError):
        return {"users": {}}


def save(data):
    os.makedirs(os.path.dirname(users_file()), exist_ok=True)
    tmp = users_file() + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
    os.replace(tmp, users_file())   # replaced in one step: a crash never leaves half a file


def record(result, when=None):
    """Save one finished question for its user (no name: counted for the team only)."""
    name = re.sub(r"\s+", " ", str(result.get("user") or "")).strip()[:60]
    c = result.get("check") or {}
    row = {"when": (when or datetime.now()).strftime("%Y-%m-%d %H:%M"), "question": result["question"],
           "check": {"issue": c.get("issue", "none"), "note": c.get("note", ""),
                     "search_text": c.get("search_text", result["question"])},
           "outcome": result["outcome"], "seconds": result["seconds"], "ai": result["ai"],
           "steps": len(result["steps"]),
           "papers": [{"id": p["id"], "title": p["title"], "link": p["link"], "confidence": p.get("confidence", 0),
                       "label": p.get("confidence_label", ""), "low": bool(p.get("low_confidence"))}
                      for p in result["papers"]]}
    with _lock:
        data = load()
        u = data["users"].setdefault(key(name) or NO_NAME, {"name": name or "No name", "asks": []})
        u["asks"] = (u["asks"] + [row])[-KEEP:]
        save(data)


def week_of(day):
    """ISO week like 2026-W48 for a date."""
    y, w, _ = day.isocalendar()
    return f"{y}-W{w:02d}"


def week_range(week):
    monday = datetime.strptime(week + "-1", "%G-W%V-%u")
    return monday, monday + timedelta(days=6)


def users():
    """People with at least one saved question, as {key: name}."""
    return {k: u["name"] for k, u in load()["users"].items() if k != NO_NAME and u.get("asks")}


def weekly(user, week=None, votes=None, today=None):
    """One person's week. votes: {month: {voter: [ids]}} from votes.json."""
    today = today or datetime.now()
    week = week or week_of(today)
    if not re.fullmatch(r"\d{4}-W\d{2}", week):
        raise ValueError("Week looks like 2026-W48")
    start, end = week_range(week)
    data = load()["users"]
    u = data.get(key(user), {"name": user, "asks": []})
    weeks = sorted({week_of(datetime.strptime(a["when"][:10], "%Y-%m-%d")) for a in u["asks"]} | {week_of(today)},
                   reverse=True)
    inweek = lambda a: start.date() <= datetime.strptime(a["when"][:10], "%Y-%m-%d").date() <= end.date()
    asks = [a for a in u["asks"] if inweek(a)]
    answered = [a for a in asks if a["papers"]]
    papers = {}
    for a in answered:
        for p in a["papers"]:
            if p["id"] not in papers or p["confidence"] > papers[p["id"]]["confidence"]:
                papers[p["id"]] = {**p, "question": a["question"]}
    ranked = sorted(papers.values(), key=lambda p: -p["confidence"])
    by_label = {lbl: sum(p["label"] == lbl for p in ranked) for lbl in ("High", "Medium", "Low")}
    agent_min = sum(a["seconds"] for a in asks) / 60
    saved = max(0.0, len(answered) * MANUAL_MINUTES - agent_min)

    my_votes = []
    for month, voters in (votes or {}).items():
        if month in {start.strftime("%Y-%m"), end.strftime("%Y-%m")}:
            for voter, ids in voters.items():
                if key(voter) == key(user):
                    my_votes += [{"month": month, "id": i} for i in ids]

    team = [x for k, x in data.items() for a in x.get("asks", []) if inweek(a)]
    people = {k for k, x in data.items() if k != NO_NAME and any(inweek(a) for a in x.get("asks", []))}

    lines = []
    if asks:
        lines.append(f"You asked {len(asks)} question{'s' * (len(asks) != 1)}; {len(answered)} answered with "
                     f"{len(ranked)} paper{'s' * (len(ranked) != 1)}"
                     + (f", {len(asks) - len(answered)} honestly unanswered." if len(asks) > len(answered) else "."))
        if by_label["Low"]:
            lines.append(f"{by_label['Low']} paper{'s' * (by_label['Low'] != 1)} had low confidence: "
                         "read them before using the idea.")
        fixed = sum(a["check"]["issue"] != "none" for a in asks)
        if fixed:
            lines.append(f"The agent spotted a problem in {fixed} question{'s' * (fixed != 1)} "
                         "(typo, too vague or off topic) and said how it read them.")
        if answered:
            lines.append(f"Estimated time saved: about {saved:.0f} minutes (assumes {MANUAL_MINUTES:g} minutes "
                         f"per manual search; the agent took {agent_min:.1f} minutes in total).")
    else:
        lines.append("No questions this week yet. Ask one on the Radar tab and it shows up here.")

    return {"user": u["name"], "week": week, "from": f"{start:%Y-%m-%d}", "to": f"{end:%Y-%m-%d}", "weeks": weeks,
            "totals": {"questions": len(asks), "answered": len(answered), "unanswered": len(asks) - len(answered),
                       "papers": len(ranked), **{k.lower(): v for k, v in by_label.items()},
                       "agent_minutes": round(agent_min, 1), "minutes_saved": round(saved),
                       "manual_minutes_assumed": MANUAL_MINUTES, "votes": len(my_votes)},
            "highlights": lines,
            "papers": ranked[:10],
            "low": [p for p in ranked if p["low"]],
            "unanswered": [{"when": a["when"], "question": a["question"]} for a in asks if not a["papers"]],
            "corrected": [{"when": a["when"], "question": a["question"], "issue": a["check"]["issue"],
                           "note": a["check"]["note"], "searched_as": a["check"]["search_text"]}
                          for a in asks if a["check"]["issue"] != "none"],
            "questions": [{"when": a["when"], "question": a["question"], "outcome": a["outcome"],
                           "papers": len(a["papers"]), "seconds": a["seconds"]} for a in reversed(asks)],
            "votes": my_votes,
            "team": {"people": len(people), "questions": len(team)}}
