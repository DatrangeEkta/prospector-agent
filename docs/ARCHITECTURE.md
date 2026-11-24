# Architecture

## Two ways the agent works

### 1. Ask the agent (on demand) — `src/ask_agent.py`

```
question
   │
   ▼
STEP 0  input check: AI reads the question (typo / vague / off topic, corrected search text, same topic)
        + code spelling check against the words in our papers and the system word list
   │
   ▼
embed the corrected question (nomic-embed-text)
   │
   ▼
┌───────────────────────── loop: at most 4 steps, ~150 s ─────────────────────────┐
│ LOOK    run a tool, take the 6 papers closest by meaning (code)                  │
│ CHECK   AI reads each abstract, keeps only papers that answer the question (AI) │
│ DECIDE  AI picks the next tool and its search words (AI), inside the rules      │
│ RULES   real IDs and tools only, no repeated search, one live search before     │
│         giving up, stop at 3 useful papers (code)                                │
└──────────────────────────────────────────────────────────────────────────────────┘
   │
   ▼
AI writes "what it is / use for us / effort" for the best 3
   │
   ▼
CLAIM CHECK  judge role, for the 5 best papers: are "why / what / use for us" supported by the abstract?
             and does the paper's MAIN topic answer step 0's reading? (each yes/partly/no)
             + code: numbers in the claims that are not in the abstract
   │
   ▼
CONFIDENCE (code) = 30% closeness + 30% supported + 40% answers; "no" = always Low (light red), "partly" <= Medium;
             the 3 most confident are shown
   │
   ▼
papers sorted by confidence  →  saved to data/users.json (weekly summary)
```

Every stage is sent to the page as it happens (`ask_agent.ask_steps()` yields `status`, `check`, `step`, `done`
events; `POST /api/ask/stream` sends them one JSON object per line). The run itself happens in a background
thread, so the agent always finishes and frees itself even if the browser is closed. `ask()` collects the same
events, so the CLI and `POST /api/ask` are unchanged.

If arXiv fails once in a run, a rule stops the agent from trying it again and it searches all stored papers
instead.

Tools:

| Tool | Searches |
|---|---|
| `search_recent` | stored papers from the last 30 days (`data/papers_pool.jsonl`, vectors in `data/embeddings_cache.npz`) |
| `search_stored` | every stored paper |
| `search_arxiv` | arXiv live, last 12 months, every field, with the AI's search words (new papers are stored) |

Each step records: tool, words, papers looked at and checked, useful IDs, the AI's thought, any rule that
changed the AI's choice, and whether the AI or a rule decided the next step.

### 2. Monthly scan — `src/prospector.py scan`

```
arXiv API (Quant: q-fin ~180 / LLM-AI: cs.CL, cs.AI, cs.LG ~1,100)
   → skip papers shown in earlier months and skip words
   → keyword score (interests.xlsx weights; title counts double; recency)
   → meaning score (closeness to "Our work" and recent "My tasks", as a percentile)
   → final score = 50% keywords + 50% meaning, per area
   → top 5 per area (at most 2 per work area)
   → AI notes for the top 10 (made-up IDs dropped, effort must be S/M/L)
   → scans/Prospector_<month>.xlsx/.md/.json  → build_radar.py → data/radar/radar.html
```

## Models

| Role | Default (local) | Company target | Set with |
|---|---|---|---|
| Writer: check, decide, explain | `gemma4:e4b` | M006 V4-Flash | `WRITER_MODEL`, `WRITER_URL`, `WRITER_API_KEY` |
| Judge: claim check, answers-the-question check | `qwen3.5:9b` | — | `JUDGE_MODEL`, `JUDGE_URL`, `JUDGE_API_KEY` |
| Embeddings | `nomic-embed-text` | — | `EMBED_MODEL`, `EMBED_URL` |

Local Ollama is called through its native API with thinking off and a 16k context.

## Why no LangChain / LangGraph

Three tools and at most four steps: a plain Python loop keeps every rule visible and testable, adds no
dependencies, and works with small local models that do simple JSON replies more reliably than framework
tool-calling formats.

## Data and privacy

All data is in `data/` on this machine. The only outside call is the public arXiv API (search words and paper
IDs, never documents). The web app has no login: run it on the office network only.
