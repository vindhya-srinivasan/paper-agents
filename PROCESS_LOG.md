# Process Log

## AI tools used
⟪FILL: e.g. Claude (Opus 5.5, claude.ai) for architecture planning, code generation, tests and docs; anything else you used, with versions⟫

## Development narrative
⟪FILL in your own words (2-4 sentences): what worked, what didn't, where you got stuck. Real events from this build you can mention: arxiv 4.0 removed `download_pdf`, so the fetch script downloads PDFs by URL; RAGAS 0.4.3 broke with langchain-community 0.4.2, so it was pinned to 0.4.1; the Gemini 3.8 Flash free tier (~20 requests/day) was too small for 3-4 calls per question; gcr.io is shut down for new projects, so deploys use Artifact Registry; plus anything that broke when you ran it for real⟫

## Verification narrative
⟪FILL (2-4 sentences): offline pytest suite with fake LLMs (17 tests, including forced worker failure and provider fallback); scripts/check_models.py against real keys; scripts/smoke_test.py against the live URL (N/19 passed); RAGAS run; cold vs warm latency; Firestore durability test across a forced new instance⟫

## What I learned
⟪FILL in your own words (2-4 sentences): what the AI tools couldn't do that you had to figure out yourself⟫
