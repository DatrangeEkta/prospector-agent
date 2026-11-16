# Prospector Agent

An AI research agent for Decimal Point Analytics. It reads new quant and LLM / AI research every month, ranks it
against what our teams work on, and answers any question about your work with real papers. It decides its own
search steps: search, check the results with the local AI, decide where to look next, repeat.

Everything runs locally (Ollama). No client data leaves the machine.

| | |
|---|---|
| **Owner** | Ekta Datrange (Quants Analyst) |
| **KRA** | Learning & New AI Tech Implementation: 1-2 new techniques a year with a documented business result |
| **Models** | `gemma4:e4b` (checks, decisions, notes; stand-in for M006 V4-Flash), `nomic-embed-text` (meaning search) |
| **Sources** | arXiv API (no key needed) |

## What it does

1. **Ask the agent** (on demand): type a question. The agent searches the stored papers, the AI keeps only papers
   that really answer it, and if they are not enough it decides the next step itself: older papers or a live arXiv
   search with search words it chooses. It stops at 3 useful papers, 4 steps or ~150 s, and every step is shown
   with who decided it (AI or rule). No paper is made up; if nothing fits it says so.
2. **Monthly scan** (scheduled): reads ~1,250 new papers, scores them by keywords + meaning against the team's work,
   and the AI writes "what it is / use for us / data needed / effort" for the top 10.
3. **Team page**: radar of every matched paper, the top 10 cards, voting, "Tell Prospector" (add interests or
   tasks by typing or by Excel upload), and the quarter's pilot.
4. **Pilot and KRA**: one technique piloted per quarter, its business result and metric recorded, yearly count.

## Quick start

```bash
# 1. once: Python packages and local models
pip install -r requirements.txt
open -a Ollama
ollama pull gemma4:e4b
ollama pull nomic-embed-text

# 2. the web app (Ask the agent + team page)
./scripts/run_server.sh            # http://127.0.0.1:8081
./scripts/run_server.sh --team     # colleagues on the office network

# 3. command line
python3 src/prospector.py ask "How do we stop hallucinations in client reports?"
python3 src/prospector.py scan                 # monthly scan (last 30 days)
python3 src/prospector.py settings             # check data/interests.xlsx
python3 src/prospector.py add "My interest is table extraction"   # add interests / tasks from a message
python3 src/prospector.py pick 2610.20417 --quarter 2026Q4 --owner Ekta --plan "try on 20 PDFs"
python3 src/prospector.py result 2610.20417 --outcome adopted --note "..." --metric "review time -40%"
python3 src/prospector.py report               # KRA summary (Excel)

# 4. tests (offline: sample feed, fake embeddings, no internet)
python3 -m unittest discover -s tests
```

## Project layout

```
prospector-agent/
├── src/
│   ├── prospector.py          monthly scan, ranking, settings, pilots, KRA, CLI entry point
│   ├── ask_agent.py           the agent loop: search -> AI check -> AI decides next step -> repeat
│   ├── models.py              connector for local / company models (Ollama or any OpenAI-compatible URL)
│   ├── build_radar.py         builds the team page from a scan
│   ├── radar_template.html    the page design
│   └── app.py                 FastAPI web app (page + API)
├── data/                      settings and everything the agent writes (kept out of git, see .gitignore)
│   ├── interests.xlsx         team settings: topics, work areas, daily tasks, skip words, page options
│   ├── interests.json         arXiv areas (technical)
│   ├── papers_seen.csv        papers already shown (so a month never repeats)
│   ├── papers_pool.jsonl      every downloaded paper with its abstract (the agent searches it)
│   ├── embeddings_cache.npz   paper vectors (each paper embedded once)
│   ├── votes.json             votes from the web page
│   ├── scans/                 monthly results (.xlsx, .md, .json, dated pages), run and ask logs
│   ├── radar/radar.html       the latest team page
│   └── settings_backups/      a copy of interests.xlsx before every change
├── scripts/
│   ├── run_server.sh          start the web app
│   └── run_monthly.sh         monthly run: start Ollama, check settings, scan, notify
├── tests/
│   ├── test_prospector.py     50 tests (scan, ranking, settings, web app, agent loop, page in headless Chrome)
│   └── fixtures/              saved arXiv feed and sample output
├── docs/ARCHITECTURE.md       how it works, models, safety rules
├── requirements.txt
└── .env.example               model and data settings
```

## Configuration

Settings for people are in `data/interests.xlsx` (one Excel file, with a "Read me" sheet). Technical settings
are environment variables (see `.env.example`):

| Variable | Default | Use |
|---|---|---|
| `WRITER_MODEL` / `WRITER_URL` / `WRITER_API_KEY` | `gemma4:e4b` on local Ollama | The AI that checks, decides and writes notes. Point it at the company M006 endpoint later |
| `EMBED_MODEL` / `EMBED_URL` | `nomic-embed-text` on local Ollama | Meaning search |
| `PROSPECTOR_DATA` | `./data` | Use another data folder |
| `PORT` | `8081` | Web app port |
| `PROSPECTOR_ADMIN_TOKEN` | not set | Needed to start a scan through `POST /api/run` |

## Web API

| Route | Use |
|---|---|
| `GET /` , `GET /month/2026-10` | latest / a given month's page |
| `POST /api/ask` `{"text"}` | the agent answers one question (steps + papers) |
| `GET/POST /api/votes` | team votes |
| `POST /api/add/preview`, `POST /api/add` | add interests / tasks from a message |
| `GET /api/settings/download`, `POST /api/settings/upload` | Excel round trip (uploads only add rows) |
| `GET /api/health`, `GET /api/months`, `POST/GET /api/run` | run report, months, start a scan (admin token) |

Interactive docs: http://127.0.0.1:8081/api/docs

## Safety rules

- Code ranks and decides; the AI checks, chooses search steps and writes notes inside fixed rules.
- Only real tools and real arXiv IDs are accepted; a made-up ID is dropped.
- The agent never repeats a search, has at most 4 steps, and may not give up before one live arXiv search.
- Paper text is outside data: it is summarised, never followed as instructions.
- "Ask the agent" never writes to the settings. Excel uploads only add rows, with a backup first.
- No client names or client data in settings, demos or the shared page.
- Without the local AI it still runs (meaning and keywords only) and says so.

## Origin

Built from `agents/09_technique_prospector` (Agent 9 on the KRA sheet) and its agent version
`agents/09b_prospector_agent`. Both folders are kept unchanged; this repo is the standalone developer copy.
