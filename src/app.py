"""
Research Radar web app (Agent 9): the radar page as a website, with team voting and older months.

    ./scripts/run_server.sh             # this Mac only:            http://127.0.0.1:8081
    ./scripts/run_server.sh --team      # colleagues on the network: http://<this computer's address>:8081

Pages are the ones `prospector.py scan` already builds (radar/radar.html and scans/Prospector_<month>_radar.html);
the app adds an API the page uses when served from here:

    GET  /                      latest month's radar page
    GET  /month/2026-10         that month's page
    GET  /api/months            months with a page
    GET  /api/votes?month=      {voter name: [arXiv IDs]}
    POST /api/votes             {"month", "voter", "papers": [...]}: replaces that voter's picks for the month
    POST /api/add/preview       {"text"}: what the local AI understood (topics, work, tasks, skip); nothing saved
    POST /api/add               {"rows"}: checks the rows again, appends them to interests.xlsx (backup first) and,
                                when something new was added, starts a paper search with the updated interests
    GET  /api/settings/download the current interests.xlsx (topics, work, tasks, skip words), to fill in
    POST /api/settings/upload   a filled copy (raw .xlsx body): checked like the real file; NEW rows are added,
                                rows deleted or changed in the copy are left as they are in the real file
    POST /api/ask               {"text", "user"}: the AGENT answers one question with papers (searches, the AI
                                checks the results and decides the next step); nothing is saved to the settings
    POST /api/ask/stream        the same run sent live, one JSON event per line (status, check, step, done)
    GET  /api/summary?user=&week=   that person's weekly summary (from data/users.json, saved on every question)
    GET  /api/summary/users     names with a saved question
    GET  /api/health            the latest run report (papers, AI notes, warnings)
    POST /api/run               start a scan (only with the X-Admin-Token header = PROSPECTOR_ADMIN_TOKEN)
    GET  /api/run               status of the last scan started from here

Votes are kept in data/votes.json and counted by `prospector.py quality`. Voters type their name; there is
no login, so keep it on the company network.
"""
import glob
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel, Field

CODE = os.path.dirname(os.path.abspath(__file__))   # src/
HOME = os.environ.get("PROSPECTOR_DATA") or os.path.join(os.path.dirname(CODE), "data")   # tests point it elsewhere
sys.path.insert(0, CODE)
import prospector  # noqa: E402  (settings: understand, check and append rows)
import ask_agent  # noqa: E402  (the agent: one question -> search, check, decide, repeat)
import summary  # noqa: E402  (weekly summary per person, data/users.json)
MONTH = re.compile(r"^\d{4}-\d{2}$")
ARXIV_ID = re.compile(r"^\d{4}\.\d{4,5}$")
_lock = threading.Lock()
_busy = {"since": None}   # when the running question started; one question at a time
_run = {"state": "idle"}

app = FastAPI(title="Research Radar", docs_url="/api/docs", redoc_url=None)


def votes_file():
    return os.path.join(HOME, "votes.json")


def scan_json(month):
    return os.path.join(HOME, "scans", f"Prospector_{month}.json")


def months():
    """Months that have a dated page, newest first."""
    found = [os.path.basename(f)[len("Prospector_"):len("Prospector_") + 7]
             for f in glob.glob(os.path.join(HOME, "scans", "Prospector_????-??_radar.html"))]
    return sorted(set(found), reverse=True)


def latest_month():
    files = sorted(glob.glob(os.path.join(HOME, "scans", "Prospector_????-??.json")), key=os.path.getmtime)
    if files:
        with open(files[-1]) as f:
            m = json.load(f).get("month")
        if m:
            return m
    found = months()
    return found[0] if found else None


def shortlist(month):
    path = os.path.join(HOME, "scans", f"Prospector_{month}.xlsx")
    if not os.path.exists(path):
        return set()
    return set(pd.read_excel(path, sheet_name="Shortlist", dtype={"arXiv ID": str})["arXiv ID"])


def serve(path, month):
    if not os.path.exists(path):
        raise HTTPException(404, f"No radar page for {month or 'any month'} yet. Run: python prospector.py scan")
    with open(path) as f:
        html = f.read()
    hook = f'<script>window.PROSPECTOR_API="/api";window.PROSPECTOR_MONTH={json.dumps(month or "")};</script>'
    html = html.replace("</head>", hook + "</head>", 1) if "</head>" in html else hook + html
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})   # always the latest page


def read_votes():
    if not os.path.exists(votes_file()):
        return {}
    with open(votes_file()) as f:
        return json.load(f)


@app.get("/", response_class=HTMLResponse)
def latest():
    for m in [latest_month()] + months():   # the latest scan's month, else the newest month with a page
        page = os.path.join(HOME, "scans", f"Prospector_{m}_radar.html") if m else ""
        if page and os.path.exists(page):
            return serve(page, m)
    return serve(os.path.join(HOME, "radar", "radar.html"), latest_month())


@app.get("/month/{month}", response_class=HTMLResponse)
def by_month(month: str):
    if not MONTH.match(month):
        raise HTTPException(400, "Month looks like 2026-10")
    return serve(os.path.join(HOME, "scans", f"Prospector_{month}_radar.html"), month)


@app.get("/month")
def month_redirect():
    return RedirectResponse("/")


@app.get("/api/months")
def api_months():
    return {"months": months(), "latest": latest_month()}


@app.get("/api/votes")
def api_votes(month: str):
    if not MONTH.match(month):
        raise HTTPException(400, "Month looks like 2026-10")
    return {"month": month, "votes": read_votes().get(month, {})}


class Vote(BaseModel):
    month: str
    voter: str = Field(min_length=1, max_length=60)
    papers: list[str] = Field(max_length=10)


@app.post("/api/votes")
def api_vote(v: Vote):
    voter = re.sub(r"\s+", " ", v.voter).strip()
    if not MONTH.match(v.month):
        raise HTTPException(400, "Month looks like 2026-10")
    if not voter:
        raise HTTPException(400, "Please give your name")
    allowed = shortlist(v.month)
    bad = [p for p in v.papers if not ARXIV_ID.match(p) or p not in allowed]
    if bad:
        raise HTTPException(400, f"Not on the {v.month} shortlist: {', '.join(bad)}")
    with _lock:   # one writer at a time; the file is replaced in one step
        data = read_votes()
        month = data.setdefault(v.month, {})
        if v.papers:
            month[voter] = sorted(set(v.papers))
        else:
            month.pop(voter, None)
        os.makedirs(os.path.dirname(votes_file()), exist_ok=True)
        tmp = votes_file() + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=1)
        os.replace(tmp, votes_file())
    return {"month": v.month, "votes": month}


class Message(BaseModel):
    text: str = Field(min_length=3, max_length=2000)


@app.post("/api/add/preview")
def api_add_preview(m: Message):
    """What the message would add. Nothing is written."""
    if not os.path.exists(prospector.SETTINGS):
        raise HTTPException(409, "interests.xlsx is missing on the server")
    rows = prospector.understand(m.text, prospector.load_config())
    ai = rows is not None
    if not ai:   # no local AI: the message is kept as today's task
        rows = {"topics": [], "work": [], "tasks": [m.text.strip()[:300]], "skip": []}
    return {"ai": ai, "rows": rows, "summary": prospector.describe(rows)}


class Rows(BaseModel):
    rows: dict


@app.post("/api/add")
def api_add(r: Rows):
    groups = list(prospector.load_config().get("radar_topics", {}))
    rows = prospector.clean_rows(r.rows, groups)   # never trust what the browser sends back
    rows = {k: v[:20] for k, v in rows.items()}
    if not any(rows.values()):
        raise HTTPException(400, "Nothing to add")
    with _lock:
        try:
            added, _ = prospector.apply_rows(rows)
        except ValueError as e:
            raise HTTPException(400, str(e))
    n = sum(len(v) for k, v in added.items() if k != "already there")
    searching = bool(n) and start_scan()   # something new: search papers with it; otherwise keep the page
    return {"added": n, "already_there": added["already there"], "searching": searching,
            "scan_busy": bool(n) and not searching}


@app.get("/api/settings/download")
def api_settings_download():
    if not os.path.exists(prospector.SETTINGS):
        raise HTTPException(404, "interests.xlsx is missing on the server")
    return FileResponse(prospector.SETTINGS, filename="Prospector settings.xlsx", headers={"Cache-Control": "no-store"},
                        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


@app.post("/api/settings/upload")
async def api_settings_upload(request: Request):
    body = await request.body()
    if not body:
        raise HTTPException(400, "No file received")
    if len(body) > 5_000_000:
        raise HTTPException(413, "The file is larger than 5 MB")
    if not body.startswith(b"PK"):   # .xlsx files are zip archives
        raise HTTPException(400, "Please upload the Excel file (.xlsx) you downloaded from this page")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "upload.xlsx")
        with open(path, "wb") as f:
            f.write(body)
        try:
            rows = prospector.rows_from_excel(path, prospector.load_config())
        except ValueError as e:
            raise HTTPException(400, str(e))
        except Exception:
            raise HTTPException(400, "This is not the settings Excel from this page (sheets or columns differ). "
                                     "Download it again and add your rows to it.")
    with _lock:
        try:
            added, _ = prospector.apply_rows(rows)   # appends only the new rows; backup first
        except ValueError as e:
            raise HTTPException(400, str(e))
    n = sum(len(v) for k, v in added.items() if k != "already there")
    searching = bool(n) and start_scan()
    new = {k: len(v) for k, v in added.items() if k != "already there"}
    return {"added": n, "new": new, "searching": searching, "scan_busy": bool(n) and not searching}


class Question(BaseModel):
    text: str = Field(min_length=3, max_length=500)
    user: str = Field(default="", max_length=60)


BUSY_MAX = 400   # seconds: a run that never finished (browser closed mid-stream) frees the agent after this


def take_agent():
    """One question at a time (the local AI answers one prompt at a time anyway)."""
    with _lock:
        since = _busy.get("since")
        if since and time.time() - since < BUSY_MAX:
            return False
        _busy["since"] = time.time()
        return True


def free_agent():
    _busy["since"] = None


@app.post("/api/ask")
def api_ask(q: Question):
    if not take_agent():
        raise HTTPException(409, "The agent is answering another question. Try again in a minute.")
    try:
        return ask_agent.ask(q.text, user=q.user)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    finally:
        free_agent()


@app.post("/api/ask/stream")
def api_ask_stream(q: Question):
    """The same agent run, sent step by step as it happens (one JSON object per line), so the page can show
    each step live: status, check (step 0), step, ..., done. A problem mid-run arrives as {"type": "error"}."""
    if not take_agent():
        raise HTTPException(409, "The agent is answering another question. Try again in a minute.")
    events = ask_agent.ask_steps(q.text, user=q.user)
    try:
        first = next(events)   # question and model problems become a normal HTTP error, before streaming starts
    except ValueError as e:
        free_agent()
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        free_agent()
        raise HTTPException(503, str(e))
    except BaseException:
        free_agent()
        raise

    # the run continues in its own thread, so a closed or refreshed browser never leaves the agent busy:
    # the run finishes, is saved for the weekly summary, and frees the agent for the next question
    box = queue.Queue()
    box.put(first)

    def work():
        try:
            for ev in events:
                box.put(ev)
        except Exception as e:  # noqa: BLE001  (shown on the page instead of a broken stream)
            box.put({"type": "error", "detail": f"The agent stopped: {e}"})
        finally:
            free_agent()
            box.put(None)

    threading.Thread(target=work, daemon=True).start()

    def lines():
        while (ev := box.get()) is not None:
            yield json.dumps(ev) + "\n"

    return StreamingResponse(lines(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


@app.get("/api/summary")
def api_summary(user: str, week: str | None = None):
    """One person's weekly summary (Monday to Sunday), built from data/users.json and the team votes."""
    if not summary.key(user):
        raise HTTPException(400, "Give your name to see your summary")
    try:
        return summary.weekly(user, week, votes=read_votes())
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/summary/users")
def api_summary_users():
    return {"users": sorted(summary.users().values(), key=str.lower)}


@app.get("/api/health")
def api_health():
    m = latest_month()
    if not m or not os.path.exists(scan_json(m)):
        return {"month": m, "run": None}
    with open(scan_json(m)) as f:
        return {"month": m, "run": json.load(f).get("run")}


def start_scan():
    """Run `prospector.py scan` in the background; False when one is already running."""
    with _lock:
        if _run.get("state") == "running":
            return False
        _run.update(state="running", started=f"{datetime.now():%Y-%m-%d %H:%M}", finished=None, ok=None)

    def work():
        r = subprocess.run([sys.executable, os.path.join(CODE, "prospector.py"), "scan"], cwd=CODE,
                           capture_output=True, text=True)
        _run.update(state="idle", finished=f"{datetime.now():%Y-%m-%d %H:%M}", ok=r.returncode == 0,
                    tail=(r.stdout + r.stderr)[-2000:])

    threading.Thread(target=work, daemon=True).start()
    return True


@app.post("/api/run")
def api_run(x_admin_token: str | None = Header(default=None)):
    token = os.environ.get("PROSPECTOR_ADMIN_TOKEN")
    if not token or x_admin_token != token:
        raise HTTPException(403, "Starting a scan needs the admin token")
    if not start_scan():
        raise HTTPException(409, "A scan is already running")
    return {"started": _run["started"]}


@app.get("/api/run")
def api_run_status():
    return _run
