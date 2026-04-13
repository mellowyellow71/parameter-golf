#!/usr/bin/env bash
# evo_benchmark.sh -- Bridge script for evo plugin integration
#
# Dispatches a single Parameter Golf experiment to GCE H100 infrastructure
# and returns the BPB score. Designed to be called by evo:optimize.
#
# Usage:
#   bash infra/evo_benchmark.sh --name "evo-trial-001" --env "QK_GAIN_INIT=3.5 MLP_MULT=3.5"
#
# Returns: Prints final BPB to stdout (float). Exit 0 on success, 1 on failure.
#
# Environment:
#   PGOLF_CONFIG  - Path to gce_config.yaml (default: infra/gce_config.yaml)
#   PGOLF_SCRIPT  - Training script (default: experiment1.py)
set -euo pipefail

CONFIG="${PGOLF_CONFIG:-infra/gce_config.yaml}"
SCRIPT="${PGOLF_SCRIPT:-experiment1.py}"

# Parse arguments
NAME=""
ENV_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --name) NAME="$2"; shift 2 ;;
        --env)  shift; while [[ $# -gt 0 && "$1" != --* ]]; do ENV_ARGS+=("$1"); shift; done ;;
        *)      echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "${NAME}" ]]; then
    NAME="evo-$(date +%Y%m%d-%H%M%S)"
fi

# Run the experiment
python3 infra/gce_run_experiment.py run \
    --config "${CONFIG}" \
    --name "${NAME}" \
    --script "${SCRIPT}" \
    --env "${ENV_ARGS[@]}"

# Parse result
RESULT_FILE="infra/result-${NAME}.json"
if [[ -f "${RESULT_FILE}" ]]; then
    BPB=$(python3 -c "import json; d=json.load(open('${RESULT_FILE}')); print(d.get('final_bpb', 0))")
    STATUS=$(python3 -c "import json; d=json.load(open('${RESULT_FILE}')); print(d.get('status', 'unknown'))")

    if [[ "${STATUS}" == "succeeded" && "${BPB}" != "0" ]]; then
        echo "${BPB}"
        exit 0
    else
        echo "Experiment ${STATUS}: BPB=${BPB}" >&2
        exit 1
    fi
else
    echo "No result file found: ${RESULT_FILE}" >&2
    exit 1
fi
