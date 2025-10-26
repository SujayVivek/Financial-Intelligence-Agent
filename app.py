import os
import requests
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles
load_dotenv()

app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)



app.mount("/static", StaticFiles(directory="./static"), name="static")

# GROK API SETUP
GROK_API_URL = "https://api.x.ai/v1/chat/completions"  # Confirmed endpoint
GROK_API_KEY = os.getenv("GROK_API_KEY")

@app.get("/")
def home():
    return {"status": "Backend running ✅"}

@app.get("/get_summary")
def get_summary(topic: str = Query(..., description="Topic such as finance, cyber, regulation, etc"),
                n: int = Query(10, description="Number of top tweets to fetch")
                ):
    """
    Fetch and summarize latest tweets related to the given topic using Grok AI.
    """
    print(f"[DEBUG] Received request for topic: {topic}")

    prompt = f"""
    Fetch the top {n} latest tweets from X related to {topic} (news, finance, cyber attacks, regulations, etc.).
    For each tweet, summarize it in 1-2 lines.
    Then, create a 5-line summary covering the overall trend of the last 24 hours.
    Format clearly like this:
    ---
    🔹 **Top 10 Tweets Summary on {topic}**
    1. ...
    2. ...
    ...
    ---
    🧠 **Overall Summary:**
    ...
    """

    headers = {
        "Authorization": f"Bearer {GROK_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "grok-3",  # or grok-latest depending on access
        "messages": [
            {"role": "system", "content": "You are an expert summarizer for financial, regulatory, and tech topics."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }

    try:
        print("[DEBUG] Sending request to Grok API...")
        response = requests.post(GROK_API_URL, headers=headers, json=data, timeout=40)
        print(f"[DEBUG] Grok response status: {response.status_code}")
        response.raise_for_status()

        res_json = response.json()
        print(f"[DEBUG] Grok JSON keys: {list(res_json.keys())}")

        summary = res_json.get("choices", [{}])[0].get("message", {}).get("content", "No summary returned.")
        return {"topic": topic, "summary": summary}

    except requests.Timeout:
        print("[ERROR] Grok API timed out.")
        return {"error": "Grok API timed out. Try again later."}
    except Exception as e:
        print(f"[ERROR] Exception: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
