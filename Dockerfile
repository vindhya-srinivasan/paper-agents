# Multi-stage build, python:3.13-slim, non-root user, uvicorn bound to Cloud Run's $PORT (assignment §4.2).

# ---------- build stage: dependencies + RAG index ----------
FROM python:3.13-slim AS build
WORKDIR /app
# Packages go in a venv outside /root so the non-root runtime user can read them.
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf
COPY requirements.txt .
# CPU-only torch wheel first (the default Linux wheel pulls ~3 GB of CUDA libraries Cloud Run can't use).
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0 && \
    pip install -r requirements.txt

# Build the Chroma index and SQLite metadata inside the image from the committed corpus text,
# so no generated artifact ever comes from the developer's machine.
COPY src ./src
COPY data/corpus ./data/corpus
COPY data/papers.json ./data/papers.json
RUN python -m src.rag.ingest

# ---------- runtime stage ----------
FROM python:3.13-slim
WORKDIR /app
ENV PATH=/opt/venv/bin:$PATH \
    HF_HOME=/opt/hf \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONUNBUFFERED=1 \
    STATE_BACKEND=firestore
COPY --from=build /opt/venv /opt/venv
COPY --from=build /opt/hf /opt/hf
COPY --from=build /app/chroma_index ./chroma_index
COPY --from=build /app/data ./data
COPY main.py ./
COPY src ./src
COPY static ./static
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser
EXPOSE 8080
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --timeout-keep-alive 75"]
