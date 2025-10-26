# app.py
import os
import re
import json
import requests
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Twitter AI News — Grok Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files at /static and index at /
app.mount("/static", StaticFiles(directory="./static"), name="static")

GROK_API_URL = "https://api.x.ai/v1/chat/completions"
GROK_API_KEY = os.getenv("GROK_API_KEY")

# ----------------------
# Helpers: JSON extraction
# ----------------------
def extract_json_from_text(text: str) -> Any:
    """
    Attempt to extract the first JSON object found in `text`.
    If it fails, raises ValueError with a message.
    """
    if not text or not isinstance(text, str):
        raise ValueError("No text provided for JSON extraction")

    # Locate first '{' and last '}' and try strict load
    first = text.find("{")
    last = text.rfind("}")
    if first == -1 or last == -1 or last < first:
        raise ValueError("No balanced JSON object braces found")

    candidate = text[first:last+1]

    # Try strict JSON parse
    try:
        return json.loads(candidate)
    except Exception as e:
        # Try simple fixes: normalize smart quotes -> normal quotes
        fixed = candidate.replace("“", '"').replace("”", '"').replace("’", "'").replace("\t", " ")
        # Convert single-quoted JSON-ish strings to double-quoted where safe:
        # - replace keys and simple values in a conservative manner
        try:
            # Replace `'key':` -> `"key":`
            fixed = re.sub(r"'([A-Za-z0-9_\- ]+)'\s*:", r'"\1":', fixed)
            # Replace : 'value' or : 'value with \'' -> : "value"
            fixed = re.sub(r':\s*\'([^\']*)\'', r': "\1"', fixed)
            return json.loads(fixed)
        except Exception as e2:
            raise ValueError(f"JSON parse failed (strict + heuristic attempts). Errors: {e}; {e2}")

# ----------------------
# Helpers: regex extraction fallback
# ----------------------
RE_TWEET_URL = re.compile(r"https?://(?:x\.com|twitter\.com)/(?P<author>[^/\s]+)/status/(?P<id>\d+)", re.IGNORECASE)
RE_ID = re.compile(r'"id"\s*:\s*"?(?P<id>\d{5,})"?')
RE_AUTHOR_FIELD = re.compile(r'"author"\s*:\s*"?(?P<author>@?[\w_]{1,30})"?')
RE_TEXT_FIELD = re.compile(r'"text"\s*:\s*"(?P<text>(?:\\.|[^"\\])*)"', re.S)
RE_CREATED_AT = re.compile(r'"created_at"\s*:\s*"(?P<created>[^"]+)"')
RE_WHY = re.compile(r'"why_selected"\s*:\s*"(?P<why>(?:\\.|[^"\\])*)"', re.S)
RE_SUMMARY_FIELD = re.compile(r'"summary"\s*:\s*"(?P<summary>(?:\\.|[^"\\])*)"', re.S)

def find_all_tweet_like_blocks(text: str) -> List[Dict[str, str]]:
    """
    Use several heuristics to extract tweet-like items from raw text.
    Returns a list of dicts containing possible fields (id, author, text, url, why_selected, created_at).
    The results are best-effort and prioritized by confidence.
    """
    results: List[Dict[str, str]] = []

    if not text:
        return results

    # 1) Find explicit JSON-ish "text": "..." blocks (likely present even in malformed JSON)
    # We'll find indexes of all occurrences of '"text":' and try to collect surrounding fields.
    text_matches = list(RE_TEXT_FIELD.finditer(text))
    if text_matches:
        for m in text_matches:
            start, end = m.span()
            # Look backwards a bit for id/author/created/why and forwards for url
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
                # ensure leading @
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

    # 2) If no text fields discovered, try to find URLs and build tweet objects from them
    if not results:
        for url_m in RE_TWEET_URL.finditer(text):
            tid = url_m.group("id")
            author_raw = url_m.group("author")
            author = "@" + author_raw if not author_raw.startswith("@") else author_raw
            # Try to find a text snippet nearby (short window)
            start, end = url_m.span()
            window = text[max(0, start-200):min(len(text), end+400)]
            # Try to find a quoted snippet after the url or before
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

    # 3) Deduplicate by id or url
    dedup: Dict[str, Dict[str, str]] = {}
    for r in results:
        key = r.get("id") or r.get("url") or r.get("text")[:40]
        if not key:
            continue
        if key in dedup:
            # merge missing fields
            existing = dedup[key]
            for f in ("author", "text", "url", "why_selected", "created_at"):
                if not existing.get(f) and r.get(f):
                    existing[f] = r.get(f)
        else:
            dedup[key] = r

    return list(dedup.values())

def sanitize_tweet_obj(t: Dict[str, Any]) -> Dict[str, str]:
    """Return sanitized tweet dict with required keys and safe strings."""
    return {
        "id": str(t.get("id") or "")[:32],
        "author": str(t.get("author") or "")[:48],
        "created_at": str(t.get("created_at") or "")[:64],
        "text": str(t.get("text") or "")[:1000],
        "url": str(t.get("url") or "")[:300],
        "why_selected": str(t.get("why_selected") or "")[:300]
    }

# ----------------------
# Main endpoint
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
    """
    Query Grok to fetch top tweets for a topic and return structured JSON.
    This function is defensive: it tries to parse JSON, asks Grok to reformat if needed,
    and finally falls back to regex-based extraction to guarantee some usable output.
    """

    if not GROK_API_KEY:
        return {"error": "GROK_API_KEY not configured on server (set in .env)"}

    # Keep prompt compact to reduce token usage and truncation
    schema_example = json.dumps({
        "tweets": [
            {
                "id": "<tweet id>",
                "author": "@handle",
                "created_at": "YYYY-MM-DDTHH:MM:SSZ",
                "text": "single-line tweet text (no newlines)",
                "url": "https://x.com/handle/status/<id>",
                "why_selected": "short reason"
            }
        ],
        "summary": "3-6 line summary text"
    }, indent=2)

    prompt = f"""
Return a strictly parseable JSON object ONLY (no commentary) with this exact structure and fields:

{schema_example}

Fetch the top {n} latest tweets from X (Twitter) related to the topic: "{topic}" (news, finance, cyber attacks, regulations, etc.).
Requirements:
- Use double quotes for JSON.
- Keep tweet.text to a single line (replace newlines with spaces).
- Keep why_selected short (under 120 chars).
- If a tweet URL is unavailable, set url to "".
- If you cannot fit {n} tweets in the token limit, return fewer tweets but do not break JSON.
- Do NOT add any explanation or markdown, only the JSON object.
"""

    headers = {"Authorization": f"Bearer {GROK_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "grok-3",
        "messages": [
            {"role": "system", "content": "You are an expert summarizer and JSON-output generator."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 1200
    }

    try:
        resp = requests.post(GROK_API_URL, headers=headers, json=payload, timeout=40)
        resp.raise_for_status()
        res_json = resp.json()

        # extract primary content text from chat-completion response
        content = ""
        if isinstance(res_json, dict):
            if "choices" in res_json and res_json["choices"]:
                # Newer chat format: choices[0].message.content
                ch0 = res_json["choices"][0]
                # support both "message" and older "text"
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

        # If raw requested, return quickly for debugging
        if raw:
            return {"raw_response": res_json, "content": content}

        # 1) Try strict JSON extraction
        try:
            parsed = extract_json_from_text(content)
            tweets = parsed.get("tweets", [])
            summary = parsed.get("summary", "")
            sanitized = [sanitize_tweet_obj(t) for t in tweets][:n]
            return {"topic": topic, "tweets": sanitized, "summary": summary, "source": "grok", "raw_content": None}
        except Exception as primary_err:
            # Attempt reformat: ask Grok to "fix" the content into valid JSON
            fix_prompt = (
                "The content below was meant to be valid JSON but appears malformed or truncated. "
                "Please output ONLY a valid JSON object following the previously given schema. "
                "If a tweet is incomplete, drop it. Keep fields compact. Here is the raw content:\n\n"
                + content
            )
            fix_payload = {
                "model": "grok-3",
                "messages": [
                    {"role": "system", "content": "You are a strict JSON reformatter. Output valid JSON only."},
                    {"role": "user", "content": fix_prompt}
                ],
                "temperature": 0.0,
                "max_tokens": 800
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

                # try parsing fixed content
                try:
                    parsed2 = extract_json_from_text(fix_content)
                    tweets = parsed2.get("tweets", [])
                    summary = parsed2.get("summary", "")
                    sanitized = [sanitize_tweet_obj(t) for t in tweets][:n]
                    return {
                        "topic": topic,
                        "tweets": sanitized,
                        "summary": summary,
                        "source": "grok_reformat",
                        "raw_content": content
                    }
                except Exception as reformat_err:
                    # Reformatting failed - fall back to regex extraction
                    fallback = find_all_tweet_like_blocks(content)
                    sanitized = [sanitize_tweet_obj(t) for t in fallback][:n]
                    # Extract summary heuristically
                    summary_m = RE_SUMMARY_FIELD.search(content)
                    summary_text = summary_m.group("summary") if summary_m else ""
                    return {
                        "topic": topic,
                        "tweets": sanitized,
                        "summary": summary_text,
                        "source": "regex_fallback",
                        "raw_content": content,
                        "parse_error": str(primary_err),
                        "reformat_error": str(reformat_err)
                    }
            except Exception as fix_call_exc:
                # Reformat call failed entirely - go straight to regex fallback
                fallback = find_all_tweet_like_blocks(content)
                sanitized = [sanitize_tweet_obj(t) for t in fallback][:n]
                summary_m = RE_SUMMARY_FIELD.search(content)
                summary_text = summary_m.group("summary") if summary_m else ""
                return {
                    "topic": topic,
                    "tweets": sanitized,
                    "summary": summary_text,
                    "source": "regex_fallback_direct",
                    "raw_content": content,
                    "parse_error": str(primary_err),
                    "reformat_call_error": str(fix_call_exc)
                }

    except requests.Timeout:
        return {"error": "Grok API timed out. Try again later."}
    except requests.HTTPError as http_err:
        body = http_err.response.text if http_err.response is not None else ""
        return {"error": f"HTTPError calling Grok: {http_err}", "body": body}
    except Exception as e:
        return {"error": str(e)}

# If run directly, allow local debug
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
