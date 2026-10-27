#!/usr/bin/env bash
# Build with Cloud Build and deploy to Cloud Run. Works on a fresh GCP project with billing enabled.
# Reads settings and keys from .env; no secret is stored in this script or in the image.
#   ./deploy.sh
set -euo pipefail
cd "$(dirname "$0")"

[ -f .env ] || { echo "Missing .env (copy .env.example and fill it in)"; exit 1; }
set -a; source .env; set +a

: "${GCP_PROJECT_ID:?Set GCP_PROJECT_ID in .env}"
: "${GCP_REGION:=us-west1}"
: "${CLOUD_RUN_SERVICE:=multi-agent-rag}"
: "${GOOGLE_API_KEY:?Set GOOGLE_API_KEY in .env}"
: "${NVIDIA_API_KEY:?Set NVIDIA_API_KEY in .env}"
: "${TAVILY_API_KEY:?Set TAVILY_API_KEY in .env}"
[ -f data/papers.json ] || { echo "Corpus missing: run python scripts/fetch_corpus.py first"; exit 1; }

# Artifact Registry (Container Registry / gcr.io is shut down for new projects).
AR_REPO="${AR_REPO:-cloud-run}"
IMAGE="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${AR_REPO}/${CLOUD_RUN_SERVICE}"

gcloud config set project "$GCP_PROJECT_ID"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com \
  firestore.googleapis.com compute.googleapis.com

if ! gcloud artifacts repositories describe "$AR_REPO" --location="$GCP_REGION" >/dev/null 2>&1; then
  gcloud artifacts repositories create "$AR_REPO" --repository-format=docker --location="$GCP_REGION"
fi

# Firestore (Native mode) holds durable conversation state. Create it once; ignore "already exists".
if ! gcloud firestore databases describe --database="(default)" >/dev/null 2>&1; then
  gcloud firestore databases create --database="(default)" --location="$GCP_REGION" --type=firestore-native
fi

# Grant the Cloud Run runtime service account access to Firestore.
PROJECT_NUMBER=$(gcloud projects describe "$GCP_PROJECT_ID" --format='value(projectNumber)')
gcloud projects add-iam-policy-binding "$GCP_PROJECT_ID" \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role="roles/datastore.user" --condition=None >/dev/null

gcloud builds submit --tag "$IMAGE" --timeout=1800s --machine-type=e2-highcpu-8

# 2Gi: torch + the embedding model + Chroma need ~1.2 GB resident; 1Gi risks OOM restarts.
# --cpu-boost speeds up cold starts; --timeout 300 covers long SSE streams.
gcloud run deploy "$CLOUD_RUN_SERVICE" \
  --image "$IMAGE" \
  --platform managed \
  --region "$GCP_REGION" \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 1 \
  --cpu-boost \
  --concurrency 10 \
  --timeout 300 \
  --min-instances 0 \
  --max-instances 3 \
  --set-env-vars "^@^GOOGLE_API_KEY=${GOOGLE_API_KEY}@GEMINI_MODEL=${GEMINI_MODEL}@NVIDIA_API_KEY=${NVIDIA_API_KEY}@NVIDIA_MODEL=${NVIDIA_MODEL}@NVIDIA_BASE_URL=${NVIDIA_BASE_URL:-https://integrate.api.nvidia.com/v1}@TAVILY_API_KEY=${TAVILY_API_KEY}@STATE_BACKEND=firestore@FIRESTORE_PROJECT_ID=${GCP_PROJECT_ID}@FIRESTORE_COLLECTION=${FIRESTORE_COLLECTION:-sessions}"

URL=$(gcloud run services describe "$CLOUD_RUN_SERVICE" --region "$GCP_REGION" --format='value(status.url)')
echo
echo "Deployed: $URL"
echo "Health:   curl -s $URL/health"
