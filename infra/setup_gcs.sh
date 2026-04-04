#!/bin/bash
# Parameter Golf — One-time GCS data upload
#
# Uploads the pre-tokenized FineWeb dataset and tokenizer to a GCS bucket
# so Vertex AI training jobs can pull the data at startup.
#
# Usage:
#   bash infra/setup_gcs.sh BUCKET_NAME [REGION]
#
# Example:
#   bash infra/setup_gcs.sh parameter-golf-data us-central1
set -euo pipefail

BUCKET="${1:?Usage: bash infra/setup_gcs.sh BUCKET_NAME [REGION]}"
REGION="${2:-us-central1}"
PROJECT=$(gcloud config get-value project 2>/dev/null)

echo "=== Parameter Golf — GCS Data Setup ==="
echo "  Project: ${PROJECT}"
echo "  Bucket:  gs://${BUCKET}"
echo "  Region:  ${REGION}"

# Create bucket if it doesn't exist
if gsutil ls "gs://${BUCKET}" &>/dev/null; then
    echo "Bucket gs://${BUCKET} already exists."
else
    echo "Creating bucket gs://${BUCKET}..."
    gsutil mb -p "${PROJECT}" -l "${REGION}" -b on "gs://${BUCKET}"
fi

# Upload dataset shards (80 train + 1 val, ~8GB total)
echo ""
echo "--- Uploading dataset shards ---"
gsutil -m rsync -r \
    ./data/datasets/fineweb10B_sp1024/ \
    "gs://${BUCKET}/datasets/fineweb10B_sp1024/"

# Upload tokenizer files
echo ""
echo "--- Uploading tokenizer ---"
gsutil -m cp \
    ./data/tokenizers/fineweb_1024_bpe.model \
    ./data/tokenizers/fineweb_1024_bpe.vocab \
    "gs://${BUCKET}/tokenizers/"

echo ""
echo "=== Upload complete ==="
echo "  Datasets:  gs://${BUCKET}/datasets/fineweb10B_sp1024/"
echo "  Tokenizer: gs://${BUCKET}/tokenizers/"
echo ""
echo "Verify with:"
echo "  gsutil ls gs://${BUCKET}/datasets/fineweb10B_sp1024/ | head"
echo "  gsutil ls gs://${BUCKET}/tokenizers/"
