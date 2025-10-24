# app.py
import os
from typing import Dict, List
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import tweepy

load_dotenv()
BEARER = os.getenv("X_BEARER_TOKEN")
if not BEARER:
    raise RuntimeError("Set X_BEARER_TOKEN in environment (.env)")

app = FastAPI(title="X Topic-based Top Tweets Agent")
app.mount("/static", StaticFiles(directory="static"), name="static")

# initialize tweepy v4 client (read-only)
client = tweepy.Client(bearer_token=BEARER, wait_on_rate_limit=True)

# Topics mapping -> search query fragment (simple, tuned for finance/news)
TOPICS = {
    "news": "#news",
    "finance": "#finance OR finance",
    "stock": "#stock OR #stocks OR stock OR stocks",
    "markets": "#markets OR markets",
    "economy": "#economy OR economy",
    "regulation": "regulation OR regulatory OR 'regulatory' OR 'rule' OR 'SEC' OR 'regulator'"
}

def make_query(fragment: str) -> str:
    # restrict to English, exclude retweets & replies (clean feed)
    return f"({fragment}) lang:en -is:retweet -is:reply"

def build_tweet_item(tweet, users_map):
    uid = str(tweet.author_id) if tweet.author_id else None
    user = users_map.get(uid, {})
    author_name = user.get("username") or user.get("name") or ""
    metrics = tweet.public_metrics or {}
    return {
        "id": tweet.id,
        "text": tweet.text,
        "author_id": tweet.author_id,
        "author_name": author_name,
        "created_at": tweet.created_at.isoformat() if tweet.created_at else None,
        "like_count": metrics.get("like_count", 0),
        "retweet_count": metrics.get("retweet_count", 0),
        "reply_count": metrics.get("reply_count", 0),
        "quote_count": metrics.get("quote_count", 0),
        "url": f"https://x.com/{author_name}/status/{tweet.id}" if author_name else f"https://x.com/i/web/status/{tweet.id}"
    }

@app.get("/api/latest-tweets")
def latest_tweets(topics: str = Query(None, description="Comma-separated topic keys. Valid keys: news,finance,stock,markets,economy,regulation"),
                  top_n: int = Query(10, ge=1, le=30, description="Top N tweets per topic (1-30)")):
    """
    Returns an object mapping selected topic -> {count, tweets: [...]}
    Example: /api/latest-tweets?topics=news,finance&top_n=12
    """
    if not topics:
        raise HTTPException(status_code=400, detail="Please provide topics query parameter (e.g. topics=news,finance)")

    requested = [t.strip() for t in topics.split(",") if t.strip()]
    invalid = [t for t in requested if t not in TOPICS]
    if invalid:
        raise HTTPException(status_code=400, detail=f"Invalid topics: {invalid}. Valid: {list(TOPICS.keys())}")

    result: Dict[str, Dict] = {}
    for key in requested:
        fragment = TOPICS[key]
        q = make_query(fragment)
        # fetch up to 100 (max allowed by recent search) to allow picking top by likes
        try:
            resp = client.search_recent_tweets(
                query=q,
                max_results=100,
                tweet_fields=["created_at", "public_metrics", "author_id"],
                expansions=["author_id"],
                user_fields=["username", "name"]
            )
        except Exception as e:
            # handle rate limit / auth errors gracefully
            result[key] = {"count": 0, "error": str(e), "tweets": []}
            continue

        tweets = resp.data or []
        users_map = {}
        if resp.includes and "users" in resp.includes:
            for u in resp.includes["users"]:
                users_map[str(u.id)] = {"username": u.username, "name": u.name}

        items = [build_tweet_item(t, users_map) for t in tweets]
        # sort by like_count primarily, then retweet_count, then recency
        items.sort(key=lambda x: (x.get("like_count", 0), x.get("retweet_count", 0), x.get("created_at") or ""), reverse=True)
        top_items = items[:top_n]
        result[key] = {"count": len(top_items), "tweets": top_items}

    return {"topics": result}

@app.get("/")
def root():
    return FileResponse("static/index.html")
