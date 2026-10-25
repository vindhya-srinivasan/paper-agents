#!/usr/bin/env bash
# Build submission.zip from git-tracked files only, so ignored files (.env, venv, chroma_index,
# __pycache__, .DS_Store, raw PDFs) can never end up in it. Then audit the exact zip.
set -euo pipefail
cd "$(dirname "$0")/.."
git diff --quiet && git diff --cached --quiet || { echo "Commit your changes first (git add -A && git commit)"; exit 1; }
rm -f submission.zip
git archive --format=zip --prefix=csci599-a2/ -o submission.zip HEAD
echo "Created submission.zip ($(du -h submission.zip | cut -f1))"
python scripts/check_submission.py submission.zip
