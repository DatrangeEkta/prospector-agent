"""
Model roles shared by the agents in this folder.

    CODER  (M012)  writes code: check specs (Agent 1), automation scripts (Agent 2)
    JUDGE  (M006)  classifies each mismatch and suggests the likely cause (Agent 1)
    NARRATOR (M016) explains results already computed by code in plain English (Agents 3, 8)
    WRITER (M006) drafts text from retrieved sources / cards (Agents 4, 6, 7, 9)
    embed()       turns text into vectors for search over past documents (Agent 4)
Two DIFFERENT models are used on purpose, so the one that writes a check
is never the one that judges it.

Any OpenAI-compatible chat endpoint works (local Ollama, vLLM, LM Studio, company gateway).
Configure with environment variables (defaults = local Ollama on a 24 GB Mac):
    CODER_URL / CODER_MODEL / CODER_API_KEY   default qwen3.5:9b
    JUDGE_URL / JUDGE_MODEL / JUDGE_API_KEY   default gemma4:e4b
    NARRATOR_URL / NARRATOR_MODEL / NARRATOR_API_KEY   default qwen3.5:9b
    WRITER_URL / WRITER_MODEL / WRITER_API_KEY         default gemma4:e4b (V4-Flash on the company server)
    EMBED_URL / EMBED_MODEL                            default nomic-embed-text
Switching to the company M012 / M006 later is only a change of these variables.

ask() returns None when the model is not reachable, so every step falls back to rules.
"""
import json
import os
import re
import urllib.request

OLLAMA = "http://localhost:11434/v1/chat/completions"
ROLES = {
    "coder": {"url": os.environ.get("CODER_URL", OLLAMA),
              "model": os.environ.get("CODER_MODEL", "qwen3.5:9b"),
              "key": os.environ.get("CODER_API_KEY", "")},
    "judge": {"url": os.environ.get("JUDGE_URL", OLLAMA),
              "model": os.environ.get("JUDGE_MODEL", "gemma4:e4b"),
              "key": os.environ.get("JUDGE_API_KEY", "")},
    "narrator": {"url": os.environ.get("NARRATOR_URL", OLLAMA),
                 "model": os.environ.get("NARRATOR_MODEL", "qwen3.5:9b"),
                 "key": os.environ.get("NARRATOR_API_KEY", "")},
    "writer": {"url": os.environ.get("WRITER_URL", OLLAMA),
               "model": os.environ.get("WRITER_MODEL", "gemma4:e4b"),
               "key": os.environ.get("WRITER_API_KEY", "")},
}
EMBED_URL = os.environ.get("EMBED_URL", "http://localhost:11434/api/embed")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")


def model_name(role):
    return ROLES[role]["model"]


def ask(role, prompt, system, temperature=0.0, timeout=600, json_mode=False, max_tokens=4096):
    cfg = ROLES[role]
    headers = {"Content-Type": "application/json"}
    if cfg["key"]:
        headers["Authorization"] = f"Bearer {cfg['key']}"
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    # local Ollama: use its native API so thinking can be switched off (otherwise a 9B
    # model "thinks" for minutes) and the context window raised from the 4096 default
    native = cfg["url"] == OLLAMA
    if native:
        url = OLLAMA.replace("/v1/chat/completions", "/api/chat")
        body = {"model": cfg["model"], "messages": messages, "stream": False, "think": False,
                # num_predict caps the reply: in JSON mode a small model can loop on whitespace forever
                "options": {"temperature": temperature, "num_ctx": 16384, "num_predict": max_tokens}}
        if json_mode:
            body["format"] = "json"
    else:
        url = cfg["url"]
        body = {"model": cfg["model"], "temperature": temperature, "messages": messages, "max_tokens": max_tokens}
    try:
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            res = json.loads(r.read())
        return (res["message"] if native else res["choices"][0]["message"])["content"].strip()
    except Exception:
        return None


def ask_json(role, prompt, system="Reply with valid JSON only. No prose, no markdown fences.", temperature=0.0):
    text = ask(role, prompt, system, temperature=temperature, json_mode=True)
    if not text:
        return None
    # small models sometimes add thinking or fences around the JSON
    text = text.split("</think>")[-1].strip()
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    start = min([i for i in (text.find("{"), text.find("[")) if i >= 0], default=-1)
    if start < 0:
        return None
    try:
        return json.loads(text[start:])
    except json.JSONDecodeError:
        try:
            return json.JSONDecoder().raw_decode(text[start:])[0]
        except json.JSONDecodeError:
            return None


def embed(texts, timeout=300):
    """Vectors for a list of texts (Ollama /api/embed). None when the model is not reachable."""
    try:
        req = urllib.request.Request(EMBED_URL, data=json.dumps({"model": EMBED_MODEL, "input": texts}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())["embeddings"]
    except Exception:
        return None


NUM = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?")


def numbers(text):
    out = []
    for m in NUM.findall(text):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def unverified_numbers(text, facts_text):
    """Numbers in a model's text that do not appear in the facts it was given (rounding allowed).
    Small whole numbers (one day, 2x, list numbering) are allowed."""
    allowed = numbers(facts_text)
    bad = []
    for x in numbers(text):
        if abs(x) <= 10 and x == int(x):
            continue
        if not any(abs(abs(x) - abs(a)) <= max(0.051, 0.005 * abs(a)) for a in allowed):
            bad.append(x)
    return bad
