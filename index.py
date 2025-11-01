# app.py
print(">>> FastAPI app loading successfully. lol sujay")

import os
import re
import json
import requests
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone

TOPIC_INSTRUCTIONS = {
    "ai": (
        "Return tweets strictly about AI developments, deployments, or AI adoption in industries or countries. "
        "Exclude unrelated items (no audit/merger/regulation tweets unless they specifically mention AI). "
        "Prioritise official accounts, vendors, research labs, regulators' AI announcements, CTO/CIO posts, or high-engagement posts."
    ),
    "cyber": (
        "Return tweets strictly about newly reported cyber incidents, breaches, ransomware, or large-scale security events. "
        "For each incident try to indicate sector (private / government), impact (data loss / downtime / money), type (ransomware / exploit), suspected actor if mentioned, and recovery efforts."
    ),
    "regulation": (
        "Return tweets strictly about regulatory developments: laws, policy announcements, guidance, enforcement actions or rule changes across fintech, banking, crypto, data privacy, taxation, auditing. "
        "Prefer tweets from regulators, law firms, major journalists, and authoritative accounts."
    ),
    "ma": (
        "Return tweets strictly about M&A deals announced or executed: acquirer, acquiree, deal value/size (if given), rationale/strategic reason, valuation metrics if included, and immediate market reaction."
    ),
    "market": (
        "Return tweets strictly about market updates (equities, FX, commodities, bonds, crypto, gold/silver). "
        "Prefer tweets that give prices, indices moves, indicators, or quick market commentary (and identify region — e.g., India/EU/US/EM)."
    ),
    "audit": (
        "Return tweets strictly about audit / consulting firms (EY, KPMG, PwC, Deloitte, Grant Thornton, BDO): fines, violations, AI adoption in audit, major client changes or regulatory interactions."
    )
}

# Strict JSON schema (string) for tweets (unchanged)
STRICT_SCHEMA_JSON = json.dumps({
    "tweets": [
        {
            "id": "<tweet id>",
            "author": "@handle",
            "created_at": "YYYY-MM-DDTHH:MM:SSZ",
            "text": "exact tweet text (verbatim, no added commentary)",
            "url": "https://x.com/handle/status/<id>",
            "retweets": 0,
            "replies": 0,
            "likes": 0,
            "why_selected": "short reason"
        }
    ],
    "summary": "3-6 line text summarizing the theme / developments",
    "cfo_insights": ["short bullet 1", "short bullet 2"]  # optional but requested
}, indent=2)

# Executive schema: includes machine-readable 'tables'
EXEC_SCHEMA_JSON = json.dumps({
    "document": "Full executive briefing document as a single string (with sections and dates).",
    "highlights": ["short bullet 1", "short bullet 2"],
    "tables": [
        {
            "title": "M&A table",
            "headers": ["Date", "Acquirer", "Acquiree", "Size/Valuation", "Rationale"],
            "rows": [["2025-10-29", "Acquirer", "Acquiree", "$X", "Reason"]]
        }
    ],
    "sources": [
        {"title": "source title", "url": "https://..."}
    ]
}, indent=2)

load_dotenv()

app = FastAPI(title="Twitter AI News — Grok Backend (Improved JSON extraction & Exec packs)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/ping")
def ping():
    return JSONResponse({"status": "ok"})

app.mount("/static", StaticFiles(directory="./static"), name="static")

GROK_API_URL = "https://api.x.ai/v1/chat/completions"
GROK_API_KEY = os.getenv("GROK_API_KEY")

# ----------------------
# Robust JSON extraction: find balanced JSON objects and try to parse them,
# prefer the largest valid JSON substring.
# ----------------------
def extract_json_from_text(text: str) -> Any:
    """
    Robust extractor:
     - scans text and finds all balanced {...} candidate substrings
     - tries to json.loads each candidate (and also simple quote fixes)
     - returns the first largest valid JSON object found (by length)
    Raises ValueError if none parse.
    """
    if not text or not isinstance(text, str):
        raise ValueError("No text provided for JSON extraction")

    candidates = []
    # stack-based scan for balanced braces
    starts = []
    for i, ch in enumerate(text):
        if ch == "{":
            starts.append(i)
        elif ch == "}":
            if starts:
                start = starts.pop()  # matched start
                end = i
                # Only consider candidates with some minimum reasonable length
                if end - start > 20:
                    candidates.append(text[start:end+1])

    # sort candidates by length descending so we try largest first
    candidates = sorted(set(candidates), key=lambda s: -len(s))

    def try_parse(s: str):
        try:
            return json.loads(s)
        except Exception:
            # quick heuristic fixes same as before
            fixed = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("\t", " ")
            try:
                fixed = re.sub(r"'([A-Za-z0-9_\- ]+)'\s*:", r'"\1":', fixed)
                fixed = re.sub(r':\s*\'([^\']*)\'', r': "\1"', fixed)
                return json.loads(fixed)
            except Exception:
                return None

    for cand in candidates:
        parsed = try_parse(cand)
        if parsed is not None:
            return parsed

    # Fallback to original approach: try first { ... last } as before
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last > first:
        candidate = text[first:last+1]
        parsed = try_parse(candidate)
        if parsed is not None:
            return parsed

    # If still no JSON found, raise informative error
    raise ValueError("No valid JSON object found in text (attempted multiple heuristics)")

# ----------------------
# Regex extraction fallback for tweets (unchanged)
# ----------------------
RE_TWEET_URL = re.compile(r"https?://(?:x\.com|twitter\.com)/(?P<author>[^/\s]+)/status/(?P<id>\d+)", re.IGNORECASE)
RE_ID = re.compile(r'"id"\s*:\s*"?(?P<id>\d{5,})"?')
RE_AUTHOR_FIELD = re.compile(r'"author"\s*:\s*"?(?P<author>@?[\w_]{1,30})"?')
RE_TEXT_FIELD = re.compile(r'"text"\s*:\s*"(?P<text>(?:\\.|[^"\\])*)"', re.S)
RE_CREATED_AT = re.compile(r'"created_at"\s*:\s*"(?P<created>[^"]+)"')
RE_WHY = re.compile(r'"why_selected"\s*:\s*"(?P<why>(?:\\.|[^"\\])*)"', re.S)
RE_SUMMARY_FIELD = re.compile(r'"summary"\s*:\s*"(?P<summary>(?:\\.|[^"\\])*)"', re.S)

def find_all_tweet_like_blocks(text: str) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    if not text:
        return results
    text_matches = list(RE_TEXT_FIELD.finditer(text))
    if text_matches:
        for m in text_matches:
            start, end = m.span()
            window_start = max(0, start - 500)
            window_end = min(len(text), end + 500)
            window = text[window_start:window_end]
            tweet = {"id": "", "author": "", "text": "", "url": "", "why_selected": "", "created_at": ""}
            tweet["text"] = m.group("text").encode('utf-8').decode('unicode_escape') if m.group("text") else ""
            id_m = RE_ID.search(window)
            if id_m:
                tweet["id"] = id_m.group("id")
            author_m = RE_AUTHOR_FIELD.search(window)
            if author_m:
                author = author_m.group("author")
                if not author.startswith("@"):
                    author = "@" + author
                tweet["author"] = author
            created_m = RE_CREATED_AT.search(window)
            if created_m:
                tweet["created_at"] = created_m.group("created")
            why_m = RE_WHY.search(window)
            if why_m:
                tweet["why_selected"] = why_m.group("why").encode('utf-8').decode('unicode_escape')
            url_m = RE_TWEET_URL.search(window)
            if url_m:
                tweet["url"] = url_m.group(0)
                if not tweet["id"]:
                    tweet["id"] = url_m.group("id")
                if not tweet["author"]:
                    a = url_m.group("author")
                    if not a.startswith("@"):
                        a = "@" + a
                    tweet["author"] = a
            results.append(tweet)
    if not results:
        for url_m in RE_TWEET_URL.finditer(text):
            tid = url_m.group("id")
            author_raw = url_m.group("author")
            author = "@" + author_raw if not author_raw.startswith("@") else author_raw
            start, end = url_m.span()
            window = text[max(0, start-200):min(len(text), end+400)]
            t_m = RE_TEXT_FIELD.search(window)
            txt = t_m.group("text") if t_m else ""
            why_m = RE_WHY.search(window)
            why = why_m.group("why") if why_m else ""
            created_m = RE_CREATED_AT.search(window)
            created = created_m.group("created") if created_m else ""
            results.append({
                "id": tid,
                "author": author,
                "text": txt,
                "url": url_m.group(0),
                "why_selected": why,
                "created_at": created
            })
    dedup: Dict[str, Dict[str, str]] = {}
    for r in results:
        key = r.get("id") or r.get("url") or r.get("text")[:40]
        if not key:
            continue
        if key in dedup:
            existing = dedup[key]
            for f in ("author", "text", "url", "why_selected", "created_at"):
                if not existing.get(f) and r.get(f):
                    existing[f] = r.get(f)
        else:
            dedup[key] = r
    return list(dedup.values())

def sanitize_tweet_obj(t: Dict[str, Any]) -> Dict[str, str]:
    return {
        "id": str(t.get("id") or "")[:32],
        "author": str(t.get("author") or "")[:48],
        "created_at": str(t.get("created_at") or "")[:64],
        "text": str(t.get("text") or "")[:1000],
        "url": str(t.get("url") or "")[:300],
        "why_selected": str(t.get("why_selected") or "")[:300]
    }

# ----------------------
# Exec prompt builder (explicitly requests machine-readable tables)
# ----------------------
def build_exec_prompt(countries: List[str], start_iso: str, end_iso: str) -> Dict[str, Any]:
    countries_text = ", ".join(countries)
    user_brief = (
        "Prepare Executive Briefing Pack covering regulatory developments, AI developments in tech "
        "and in deployment/adoption by different industries/countries, M&A update deals executed and its details, "
        "cyberattack. Discuss with date of events and cover your analysis in last 24 hours.\n\n"
        "Cyber attack : New attacks reported in different parts of world, details of incident, impact it caused, "
        "segregate that into private sector, govt sector, who caused it, what sort of attack (e.g., ransomware) and recovery efforts. "
        "If no attack, bring recovery efforts underway for earlier reported incidents.\n\n"
        "Rules and regulations development : in fintech, banking, different industries related regulatory new updates, crypto world developments, "
        "accounting, taxation, insurance, law, data privacy, auditing - only new updates.\n\n"
        "Audit /consulting firms news update : It can cover audit firms related news such as EY/KPMG/PwC related actions - violation/fines/use of AI/Deployment of AI in finance/audit world.\n\n"
        "Mergers & Acquisitions : New deals announced, acquirer, acquiree, size, valuation metrics, rationale, impact, valuation basis.\n\n"
        "Give citation references from where details sought so users can click and expand more. General CFO - Lessons from the above - just summary lines."
    )

    do_not_rules = (
        "Return only a single VALID JSON object and nothing else. Use double quotes only. "
        "Do NOT include salutations, greetings, 'Hello', 'Hi', 'Dear', or conversational openers. "
        "Start the `document` content directly with the briefing title or first section (no preamble). "
        "The `document` field should be readable and may include Markdown (including markdown tables). "
        "Additionally, include a machine-readable `tables` array: each table entry must be {title, headers: [...], rows: [[...],[...]]}. "
        "If you include a visual table inside `document`, also include the same table in `tables` for programmatic use."
    )

    prompt_text = (
        "You are an assistant that prepares high-quality Executive Briefing Packs for senior executives.\n\n"
        "Output Requirement:\n"
        "Return a STRICT, valid JSON object using the exact schema below (no extra keys, no commentary):\n\n"
        f"{EXEC_SCHEMA_JSON}\n\n"
        "Instructions:\n"
        f"- TIME WINDOW: Only consider news and social posts published between {start_iso} (inclusive) and {end_iso} (inclusive) — i.e., the last 24 hours.\n"
        f"- SCOPE: Limit your search and synthesis to events and reporting relating to these countries ONLY: {countries_text}.\n"
        f"- FORMAT: Provide the briefing in `document` (readable text). If there are tabular items (e.g., M&A or deals), embed a readable Markdown table in `document` and ALSO include a corresponding structured object in `tables`.\n\n"
        f"USER BRIEF:\n{user_brief}\n\n"
        f"Other rules:\n{do_not_rules}\n\n"
        "If you cannot find content for a subsection, explicitly state 'No material new items in the last 24 hours' for that subsection but provide a short analytical comment. Always return valid JSON even if some arrays are empty."
    )

    payload = {
        "model": "grok-3",
        "messages": [
            {"role": "system", "content": "You are a precise briefing writer. Output valid JSON only."},
            {"role": "user", "content": prompt_text}
        ],
        # increased to reduce truncation (model limits permitting)
        "temperature": 0.0,
        "max_tokens": 3500
    }

    return {"prompt": prompt_text, "payload": payload}


def build_grok_prompt(topic: str, n: int, prefer_verified: bool = True) -> Dict[str, Any]:
    """
    Build the payload (prompt + model args) to send to Grok for tweet extraction.
    Returns a dict with keys: 'prompt' (str) and 'payload' (dict for requests.post).
    """
    # Compute last-24-hours window in ISO8601 (UTC)
    now_utc = datetime.now(timezone.utc)
    start_utc = now_utc - timedelta(hours=24)
    start_iso = start_utc.replace(microsecond=0).isoformat()
    end_iso = now_utc.replace(microsecond=0).isoformat()

    # Ensure we have a topic-specific instruction
    topic_key = topic.lower()
    topic_instr = TOPIC_INSTRUCTIONS.get(topic_key, f"Return tweets strictly about {topic}.")

    # explicit "do not" rules and data expectations
    do_not_rules = (
        "Do NOT paraphrase or rewrite the tweet text field — return the tweet text EXACTLY as posted (verbatim). "
        "Do NOT add commentary inline. Do NOT invent URLs. If you cannot determine the exact tweet URL or it is not available, set url to \"\". "
        "All dates must be ISO 8601 (UTC). Use double quotes only. Output only a single top-level JSON object and nothing else."
    )

    # Ranking guidance
    ranking = (
        "Return up to {n} tweets, ranked primarily by engagement (likes+retweets+replies) and secondarily by recency. "
        "Prefer tweets from authoritative/verified accounts and official sources when available. "
    ).format(n=n)

    # Build the full instruction prompt
    prompt_text = (
        "You are an assistant that searches Twitter/X for very recent, high-quality posts.\n\n"
        "Output Requirement:\n"
        "Return a STRICT, valid JSON object using the exact schema below (no extra keys, no commentary):\n\n"
        f"{STRICT_SCHEMA_JSON}\n\n"
        "Search & filtering instructions:\n"
        f"- TIME WINDOW: Only consider tweets posted between {start_iso} (inclusive) and {end_iso} (inclusive) — i.e., the last 24 hours.\n"
        f"- TOPIC: {topic_instr}\n"
        f"- RANKING: {ranking}\n"
        f"- PREFER_VERIFIED: {'Yes' if prefer_verified else 'No'}\n\n"
        "Field rules and details:\n"
        "- tweet.text must be EXACT verbatim tweet text (do not summarize or paraphrase). Replace newline characters with a single space.\n"
        "- tweet.created_at must be ISO 8601 UTC (e.g. 2025-10-27T14:23:00Z). If you cannot get exact timestamp, set created_at to empty string.\n"
        "- tweet.url must be the canonical X/Twitter URL of the tweet if available (https://x.com/<handle>/status/<id>). If you cannot reliably provide the URL, set it to \"\".\n"
        "- Provide engagement numbers (retweets, replies, likes) if available; otherwise set to 0.\n"
        "- why_selected: one short sentence (<= 120 chars) explaining why this tweet was chosen.\n\n"
        f"Other rules:\n{do_not_rules}\n\n"
        "If you are unable to return the fully valid JSON (for example due to truncation), return the best-effort JSON object that remains valid JSON (do not return text or code blocks)."
    )

    # Build model payload
    payload = {
        "model": "grok-3",
        "messages": [
            {"role": "system", "content": "You are a precise data extractor. Output valid JSON only."},
            {"role": "user", "content": prompt_text}
        ],
        "temperature": 0.0,
        "max_tokens": 1400
    }

    return {"prompt": prompt_text, "payload": payload}

# ----------------------
# Standard tweet endpoint (unchanged logic)
# ----------------------
@app.get("/")
def serve_frontend():
    return FileResponse("./static/index.html")

@app.get("/get_summary")
def get_summary(
    topic: str = Query(..., description="Topic such as finance, cyber, regulation, etc"),
    n: int = Query(5, description="Number of top tweets to fetch (prefer <=10)"),
    raw: bool = Query(False, description="Return raw grok output for debugging")
):
    if not GROK_API_KEY:
        return {"error": "GROK_API_KEY not configured on server (set in .env)"}

    gp = build_grok_prompt(topic, n, prefer_verified=True)
    payload = gp["payload"]
    headers = {"Authorization": f"Bearer {GROK_API_KEY}", "Content-Type": "application/json"}

    try:
        resp = requests.post(GROK_API_URL, headers=headers, json=payload, timeout=90)
        resp.raise_for_status()
        res_json = resp.json()

        content = ""
        if isinstance(res_json, dict):
            if "choices" in res_json and res_json["choices"]:
                ch0 = res_json["choices"][0]
                content = ch0.get("message", {}).get("content") or ch0.get("text") or ""
            elif "output" in res_json:
                if isinstance(res_json["output"], list):
                    content = " ".join(map(str, res_json["output"]))
                else:
                    content = str(res_json["output"])
            else:
                content = json.dumps(res_json)
        else:
            content = str(res_json)

        cleaned_content = content.strip()
        cleaned_content = re.sub(r'```json|```', '', cleaned_content)
        cleaned_content = re.sub(r',\s*}', '}', cleaned_content)
        cleaned_content = re.sub(r',\s*\]', ']', cleaned_content)
        content = cleaned_content

        if raw:
            return {"raw_response": res_json, "content": content}

        try:
            parsed = extract_json_from_text(content)
            tweets = parsed.get("tweets", []) or []
            summary = parsed.get("summary", "") or ""
            cfo_insights = parsed.get("cfo_insights") or parsed.get("cfo_insights", []) or []
            sanitized = [sanitize_tweet_obj(t) for t in tweets][:n]
            return {
                "topic": topic,
                "tweets": sanitized,
                "summary": summary,
                "cfo_insights": cfo_insights,
                "source": "grok",
                "raw_content": None
            }
        except Exception as primary_err:
            fix_prompt = (
                "The content below was intended to be valid JSON following a strict schema, "
                "but the returned text appears malformed or truncated. "
                "Please OUTPUT ONLY a valid JSON object that follows the schema previously requested. "
                "If a tweet is incomplete, drop it. Keep fields compact and use double quotes.\n\n"
                "RAW CONTENT START:\n\n" + content + "\n\nRAW CONTENT END."
            )
            fix_payload = {
                "model": "grok-3",
                "messages": [
                    {"role": "system", "content": "You are a precise data extractor. Output VALID, STRICT JSON only. Never repeat keys or leave trailing commas."},
                    {"role": "user", "content": fix_prompt}
                ],
                "temperature": 0.0,
                "max_tokens": 1200
            }
            try:
                fix_resp = requests.post(GROK_API_URL, headers=headers, json=fix_payload, timeout=30)
                fix_resp.raise_for_status()
                fix_json = fix_resp.json()
                fix_content = ""
                if isinstance(fix_json, dict) and "choices" in fix_json and fix_json["choices"]:
                    ch0 = fix_json["choices"][0]
                    fix_content = ch0.get("message", {}).get("content") or ch0.get("text") or ""
                else:
                    fix_content = json.dumps(fix_json)
                try:
                    parsed2 = extract_json_from_text(fix_content)
                    tweets = parsed2.get("tweets", []) or []
                    summary = parsed2.get("summary", "") or ""
                    cfo_insights = parsed2.get("cfo_insights") or []
                    sanitized = [sanitize_tweet_obj(t) for t in tweets][:n]
                    return {
                        "topic": topic,
                        "tweets": sanitized,
                        "summary": summary,
                        "cfo_insights": cfo_insights,
                        "source": "grok_reformat",
                        "raw_content": content
                    }
                except Exception as reformat_err:
                    fallback = find_all_tweet_like_blocks(content)
                    sanitized = [sanitize_tweet_obj(t) for t in fallback][:n]
                    summary_m = RE_SUMMARY_FIELD.search(content)
                    summary_text = summary_m.group("summary") if summary_m else ""
                    return {
                        "topic": topic,
                        "tweets": sanitized,
                        "summary": summary_text,
                        "cfo_insights": [],
                        "source": "regex_fallback",
                        "raw_content": content,
                        "parse_error": str(primary_err),
                        "reformat_error": str(reformat_err)
                    }
            except Exception as fix_call_exc:
                fallback = find_all_tweet_like_blocks(content)
                sanitized = [sanitize_tweet_obj(t) for t in fallback][:n]
                summary_m = RE_SUMMARY_FIELD.search(content)
                summary_text = summary_m.group("summary") if summary_m else ""
                return {
                    "topic": topic,
                    "tweets": sanitized,
                    "summary": summary_text,
                    "cfo_insights": [],
                    "source": "regex_fallback_direct",
                    "raw_content": content,
                    "parse_error": str(primary_err),
                    "reformat_call_error": str(fix_call_exc)
                }

    except requests.Timeout:
        return {"error": "Grok API timed out. Try again later."}
    except requests.HTTPError as http_err:
        body = http_err.response.text if (http_err.response is not None) else ""
        return {"error": f"HTTPError calling Grok: {http_err}", "body": body}
    except Exception as e:
        return {"error": str(e)}

# ----------------------
# New/Improved endpoint: Executive Summary (robust + tables)
# ----------------------
@app.get("/get_exec_summary")
def get_exec_summary(
    countries: Optional[str] = Query(None, description="Comma-separated list of countries (default: 7 Gulf countries)"),
    raw: bool = Query(False, description="Return raw grok output for debugging")
):
    if not GROK_API_KEY:
        return {"error": "GROK_API_KEY not configured on server (set in .env)"}

    default_countries = ["Saudi Arabia", "United Arab Emirates", "Qatar", "Kuwait", "Bahrain", "Oman", "Iraq"]
    if countries:
        try:
            provided = [c.strip() for c in countries.split(",") if c.strip()]
            country_list = provided if provided else default_countries
        except Exception:
            country_list = default_countries
    else:
        country_list = default_countries

    now_utc = datetime.now(timezone.utc)
    start_utc = now_utc - timedelta(hours=24)
    start_iso = start_utc.replace(microsecond=0).isoformat()
    end_iso = now_utc.replace(microsecond=0).isoformat()

    gp = build_exec_prompt(country_list, start_iso, end_iso)
    payload = gp["payload"]
    headers = {"Authorization": f"Bearer {GROK_API_KEY}", "Content-Type": "application/json"}

    try:
        # longer timeout to reduce truncation risks
        resp = requests.post(GROK_API_URL, headers=headers, json=payload, timeout=180)
        resp.raise_for_status()
        res_json = resp.json()

        content = ""
        if isinstance(res_json, dict):
            if "choices" in res_json and res_json["choices"]:
                ch0 = res_json["choices"][0]
                content = ch0.get("message", {}).get("content") or ch0.get("text") or ""
            elif "output" in res_json:
                if isinstance(res_json["output"], list):
                    content = " ".join(map(str, res_json["output"]))
                else:
                    content = str(res_json["output"])
            else:
                content = json.dumps(res_json)
        else:
            content = str(res_json)

        cleaned_content = content.strip()
        # keep markdown tables if present; only remove explicit triple backticks around json blocks
        cleaned_content = re.sub(r'```json', '', cleaned_content)
        # we will not blindly strip all backticks here to preserve markdown tables in `document`
        cleaned_content = re.sub(r',\s*}', '}', cleaned_content)
        cleaned_content = re.sub(r',\s*\]', ']', cleaned_content)
        content = cleaned_content

        if raw:
            return {"raw_response": res_json, "content": content}

        # Try strong JSON extraction (robust)
        try:
            parsed = extract_json_from_text(content)
            document = parsed.get("document", "") or ""
            highlights = parsed.get("highlights", []) or []
            sources = parsed.get("sources", []) or []
            tables = parsed.get("tables", []) or []
            return {
                "document": document,
                "highlights": highlights,
                "tables": tables,
                "sources": sources,
                "source": "grok",
                "raw_content": None
            }
        except Exception as parse_err:
            # If parse fails, ask Grok to reformat the raw content into the exact schema,
            # and explicitly request converting any narrative table into arrays.
            fix_prompt = (
                "The content below was intended to be valid JSON following this schema:\n\n"
                f"{EXEC_SCHEMA_JSON}\n\n"
                "However it's malformed/truncated. Please OUTPUT ONLY a single VALID JSON object exactly following the schema. "
                "If you included any readable tables in the `document`, also add a corresponding entry in `tables` with 'title', 'headers' and 'rows'. "
                "If a subsection has no new items, put: 'No material new items in the last 24 hours' for that subsection.\n\n"
                "RAW CONTENT START:\n\n" + content + "\n\nRAW CONTENT END."
            )
            fix_payload = {
                "model": "grok-3",
                "messages": [
                    {"role": "system", "content": (
                        "You are a detailed executive intelligence assistant. "
                        "Write longer, analytical, and context-rich briefings (8,000–10,000 characters). "
                        "Cover implications, causal links, and impact where relevant. "
                        "Important! - Ensure all text fits within VALID JSON under the field names exactly as specified."
                    )},
                    {"role": "user", "content": fix_prompt}
                ],
                "temperature": 0.0,
                "max_tokens": 10000
            }
            try:
                fix_resp = requests.post(GROK_API_URL, headers=headers, json=fix_payload, timeout=80)
                fix_resp.raise_for_status()
                fix_json = fix_resp.json()
                fix_content = ""
                # print(len(fix_resp['choices'][0]['message']['content']))
                if isinstance(fix_json, dict) and "choices" in fix_json and fix_json["choices"]:
                    ch0 = fix_json["choices"][0]
                    fix_content = ch0.get("message", {}).get("content") or ch0.get("text") or ""
                else:
                    fix_content = json.dumps(fix_json)
                try:
                    parsed2 = extract_json_from_text(fix_content)
                    document = parsed2.get("document", "") or ""
                    highlights = parsed2.get("highlights", []) or []
                    sources = parsed2.get("sources", []) or []
                    tables = parsed2.get("tables", []) or []
                    return {
                        "document": document,
                        "highlights": highlights,
                        "tables": tables,
                        "sources": sources,
                        "source": "grok_reformat",
                        "raw_content": content
                    }
                except Exception as reformat_err:
                    # As last fallback, return raw content as document with a parse_error field
                    return {
                        "document": content,
                        "highlights": [],
                        "tables": [],
                        "sources": [],
                        "source": "grok_raw_fallback",
                        "raw_content": content,
                        "parse_error": str(parse_err),
                        "reformat_error": str(reformat_err)
                    }
            except Exception as fix_call_exc:
                return {
                    "document": content,
                    "highlights": [],
                    "tables": [],
                    "sources": [],
                    "source": "grok_reformat_failed",
                    "raw_content": content,
                    "reformat_call_error": str(fix_call_exc),
                    "parse_error": str(parse_err)
                }

    except requests.Timeout:
        return {"error": "Grok API timed out. Try again later."}
    except requests.HTTPError as http_err:
        body = http_err.response.text if (http_err.response is not None) else ""
        return {"error": f"HTTPError calling Grok: {http_err}", "body": body}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("index:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=False)

handler = app

