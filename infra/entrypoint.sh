#!/bin/bash
# Parameter Golf — Container Entrypoint
# 1. Downloads training data from GCS
# 2. Runs torchrun with 8 GPUs
# 3. Uploads results back to GCS
set -euo pipefail

# Required env vars
: "${GCS_DATA_BUCKET:?Set GCS_DATA_BUCKET (e.g. gs://parameter-golf-data)}"
: "${GCS_OUTPUT_BUCKET:?Set GCS_OUTPUT_BUCKET (e.g. gs://parameter-golf-experiments)}"
TRAINING_SCRIPT="${TRAINING_SCRIPT:-experiment1.py}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-unnamed}"
NUM_GPUS="${NUM_GPUS:-8}"

echo "=== Parameter Golf Training ==="
echo "  Script:     ${TRAINING_SCRIPT}"
echo "  Experiment: ${EXPERIMENT_NAME}"
echo "  GPUs:       ${NUM_GPUS}"
echo "  Data:       ${GCS_DATA_BUCKET}"
echo "  Output:     ${GCS_OUTPUT_BUCKET}/experiments/${EXPERIMENT_NAME}/"

# Download training data to the paths expected by the training scripts
mkdir -p ./data/datasets/fineweb10B_sp1024 ./data/tokenizers ./logs

echo "--- Downloading training data ---"
T_START=$(date +%s)
gsutil -m rsync -r "${GCS_DATA_BUCKET}/datasets/fineweb10B_sp1024/" ./data/datasets/fineweb10B_sp1024/
gsutil -m cp \
    "${GCS_DATA_BUCKET}/tokenizers/fineweb_1024_bpe.model" \
    "${GCS_DATA_BUCKET}/tokenizers/fineweb_1024_bpe.vocab" \
    ./data/tokenizers/
T_END=$(date +%s)
echo "Data download complete in $((T_END - T_START))s"

# Run training
echo "--- Starting training: torchrun --standalone --nproc_per_node=${NUM_GPUS} ${TRAINING_SCRIPT} ---"
set +e
torchrun --standalone --nproc_per_node="${NUM_GPUS}" "${TRAINING_SCRIPT}" 2>&1 | tee training_output.log
EXIT_CODE=${PIPESTATUS[0]}
set -e

echo "Training exited with code ${EXIT_CODE}"

# Upload results (best-effort, even if training failed)
OUTPUT_PREFIX="${GCS_OUTPUT_BUCKET}/experiments/${EXPERIMENT_NAME}"
echo "--- Uploading results to ${OUTPUT_PREFIX}/ ---"
gsutil -m cp final_model.* "${OUTPUT_PREFIX}/" 2>/dev/null || echo "No final_model.* files to upload"
gsutil -m cp -r logs/ "${OUTPUT_PREFIX}/logs/" 2>/dev/null || echo "No logs/ to upload"
gsutil cp training_output.log "${OUTPUT_PREFIX}/" 2>/dev/null || echo "No training_output.log to upload"

echo "=== Done (exit code: ${EXIT_CODE}) ==="
exit ${EXIT_CODE}
