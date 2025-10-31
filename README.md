# NewsAgent
Just for fun : )

uvicorn app:app --reload --port 8000

## Improvements:
- Agent to check for authenticity (Get latest tweets, use a RAG model)
- Use Database to store news every 12 hrs(or whenever the API is called)
- Think of ways to improve speed of response retreival

## Identified Problems
- Even though Grok is connected to near-real-time data, its public API access is not the same as the “Grok on X” pipeline.
That internal version has post-processing, source ranking, and fact-checking layers — your direct API version doesn’t
-So when you ask for an “executive summary,” Grok is: Scraping headlines + pattern-matching recent events; Then generating filler context from prior patterns (which can be wrong)
That’s why one run looks authentic, another goes off-topic.

