#!/bin/bash
# Parameter Golf — Build and push training container to Artifact Registry
#
# Uses Cloud Build (no local Docker required). Falls back to local Docker if available.
#
# Usage:
#   bash infra/build_push.sh PROJECT_ID [REGION] [TAG]
#
# Example:
#   bash infra/build_push.sh my-gcp-project us-central1 latest
#   bash infra/build_push.sh my-gcp-project us-central1 v2
set -euo pipefail

PROJECT="${1:?Usage: bash infra/build_push.sh PROJECT_ID [REGION] [TAG]}"
REGION="${2:-us-central1}"
TAG="${3:-latest}"

REPO="${REGION}-docker.pkg.dev/${PROJECT}/parameter-golf"
IMAGE="${REPO}/training:${TAG}"

echo "=== Parameter Golf — Container Build & Push ==="
echo "  Project: ${PROJECT}"
echo "  Region:  ${REGION}"
echo "  Image:   ${IMAGE}"

# Create Artifact Registry repo if needed
if gcloud artifacts repositories describe parameter-golf \
    --location="${REGION}" --project="${PROJECT}" &>/dev/null; then
    echo "Artifact Registry repo already exists."
else
    echo "Creating Artifact Registry repo..."
    gcloud artifacts repositories create parameter-golf \
        --repository-format=docker \
        --location="${REGION}" \
        --project="${PROJECT}" \
        --description="Parameter Golf training containers"
fi

if command -v docker &>/dev/null; then
    # Local Docker build + push
    echo "Using local Docker..."
    gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet
    docker build -t "${IMAGE}" -f infra/Dockerfile .
    docker push "${IMAGE}"
else
    # Cloud Build (no local Docker needed)
    echo "Using Cloud Build (no local Docker found)..."
    gcloud builds submit \
        --project="${PROJECT}" \
        --region="${REGION}" \
        --tag="${IMAGE}" \
        --gcs-source-staging-dir="gs://${PROJECT}_cloudbuild/source" \
        --timeout=1800s
fi

echo ""
echo "=== Done ==="
echo "  Image: ${IMAGE}"
echo ""
echo "Update experiments.yaml container_uri to:"
echo "  ${IMAGE}"
