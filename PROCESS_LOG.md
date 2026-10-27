# Process Log

## AI tools used

I used Claude (Opus 5.5, on claude.ai) as a coding assistant for drafting code, tests and docs, and for debugging. I set the direction: I mapped the rubric into a checklist, chose the corpus, providers and GCP project, ran and tested everything against real services, and decided what to change when things broke.

## Development narrative

The architecture came together fast. Getting it to run against real services took most of the time. My Mac's Python failed every HTTPS call until I installed its certificates. Gemini 3.8 and 3.7 Flash kept returning 503 on the free tier, so I tested a few models and moved to 3.5 Flash-Lite. arXiv rate-limited the corpus download, so the script now backs off and resumes. The strangest bug: Cloud Run quietly reserves `/healthz`, so my page said "backend unreachable" while the app was working fine.

## Verification narrative

Nineteen offline tests with fake LLMs cover routing, provider and Gemini-model fallback, and a forced worker failure. `check_models.py` confirmed both providers and Tavily with my keys. My first real smoke test passed 16 of 19 checks: NVIDIA answers had no citations, memory forgot early turns, and the failure test targeted the wrong agent. After fixing those it passed 19/19 locally, and I reran it against the live URL. RAGAS gave 0.983 faithfulness, 0.890 answer relevancy, 0.668 context precision and 0.822 context recall.

## What I learned

Most problems only showed up with real providers: quotas, overloaded models and platform quirks. No AI tool could know my free-tier limits or Cloud Run's reserved paths ahead of time. I had to run things, read the actual errors and decide what to change.
