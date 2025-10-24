# app.py
import os
from datetime import datetime, timedelta, timezone
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import tweepy
from transformers import pipeline

load_dotenv()


BEARER = os.getenv("X_BEARER_TOKEN")
if not BEARER:
    raise RuntimeError("Set X_BEARER_TOKEN in your .env file")

app = FastAPI(title="X/Twitter News Intelligence Agent")
app.mount("/static", StaticFiles(directory="static"), name="static")

client = tweepy.Client(bearer_token=BEARER, wait_on_rate_limit=True)

# ----------------------------
# TOPIC DEFINITIONS
# ----------------------------
TOPICS = {
    "cyber": "cyber attack OR ransomware OR data breach OR hack lang:en -is:retweet",
    "regulation": "regulation OR policy OR law OR compliance OR fintech regulation OR crypto regulation OR data privacy OR accounting OR taxation OR audit lang:en -is:retweet",
    "ai": "AI development OR AI adoption OR AI in finance OR AI in manufacturing OR AI in government OR generative AI OR LLM lang:en -is:retweet",
    "ma": "merger OR acquisition OR M&A OR buyout OR takeover OR deal announced lang:en -is:retweet",
    "market": "market update OR stock market OR equities OR bond market OR oil OR gold OR silver OR crypto OR bitcoin OR ethereum OR FX OR currency lang:en -is:retweet",
    "audit": "PWC OR EY OR KPMG OR Deloitte OR Grant Thornton OR BDO (audit OR fine OR sanction OR AI OR technology OR innovation) lang:en -is:retweet",
}

# ----------------------------
# HELPER FUNCTION
# ----------------------------

# Load summarizer model (small & efficient)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_tweets(tweets):
    if not tweets:
        return "No new events detected."
    
    combined_text = " ".join([t["text"] for t in tweets])
    combined_text = combined_text[:3000]  # limit input size
    summary = summarizer(combined_text, max_length=120, min_length=40, do_sample=False)
    return summary[0]["summary_text"]

def tweet_to_dict(tweet, users_map):
    author = users_map.get(str(tweet.author_id), {})
    author_name = author.get("username") or author.get("name") or ""
    return {
        "id": tweet.id,
        "text": tweet.text,
        "author": author_name,
        "created_at": tweet.created_at.isoformat() if tweet.created_at else "",
        "url": f"https://x.com/{author_name}/status/{tweet.id}" if author_name else f"https://x.com/i/web/status/{tweet.id}"
    }

def search_tweets_for_topic(topic_query: str, max_results: int = 15):
    """Fetch recent tweets for given query (last 24h)."""
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=1)
    try:
        resp = client.search_recent_tweets(
            query=topic_query,
            max_results=min(max_results, 100),
            start_time=start_time,
            end_time=end_time,
            tweet_fields=["created_at", "lang", "author_id"],
            expansions=["author_id"],
            user_fields=["username", "name"]
        )
    except Exception as e:
        print(f"[Error] Query failed for {topic_query[:40]}... -> {e}")
        return []

    if not resp or not resp.data:
        return []

    users = {}
    if resp.includes and "users" in resp.includes:
        for u in resp.includes["users"]:
            users[str(u.id)] = {"username": u.username, "name": u.name}

    tweets = [tweet_to_dict(t, users) for t in resp.data]
    tweets.sort(key=lambda x: x["created_at"], reverse=True)
    return tweets[:max_results]

# ----------------------------
# API ENDPOINTS
# ----------------------------

@app.get("/api/latest-tweets")
def latest_tweets(topic: str = Query(None, description="Topic keyword e.g. cyber, regulation, ai, ma, market, audit")):
    """
    Returns latest categorized tweets for the selected topic,
    or all topics if none specified.
    """
    if topic:
        topic = topic.lower()
        if topic not in TOPICS:
            return {"error": f"Invalid topic '{topic}'. Valid options: {list(TOPICS.keys())}"}
        q = TOPICS[topic]
        tweets = search_tweets_for_topic(q)
        return {
            "topic": topic,
            "count": len(tweets),
            "tweets": tweets
        }

    # If no topic given, fetch all categories
    all_data = {}
    for key, q in TOPICS.items():
        print(f"[Info] Fetching {key} tweets...")
        tweets = search_tweets_for_topic(q)
        all_data[key] = {
            "count": len(tweets),
            "tweets": tweets
        }

    # simple CFO summary placeholder
    cfo_summary = [
        "AI and regulatory updates continue to dominate financial news.",
        "Cyber attacks remain a global risk with increasing sophistication.",
        "Market sentiment stable amid M&A activities and policy shifts."
    ]

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "topics": all_data,
        "cfo_summary": cfo_summary
    }

@app.get("/news/{topic}")
def get_news(topic: str):
    if topic not in TOPICS:
        return {"error": "Invalid topic"}

    tweets = search_tweets_for_topic(TOPICS[topic], max_results=15)
    summary = summarize_tweets(tweets)
    return {"topic": topic, "tweets": tweets, "summary": summary}


@app.get("/")
def root():
    return FileResponse("static/index.html")
