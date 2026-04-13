#!/bin/bash
# Parameter Golf — Golden Image Verification & Creation
#
# Verifies that an existing VM has all required dependencies for Parameter Golf
# training, optionally installs missing deps, and creates a reusable golden image.
#
# Usage:
#   bash infra/gce_golden_image.sh verify [VM_NAME]    # Check deps on a VM
#   bash infra/gce_golden_image.sh create  [VM_NAME]   # Create image from VM disk
#   bash infra/gce_golden_image.sh full    [VM_NAME]   # Verify + fix + create
#
# The VM must be RUNNING for verify/full. For create, it must be STOPPED.
#
# Default VM: real8xh100paramgolf (uses test1parametergolf image)
# Fallback VM: paramgolf8xh100s (500GB disk, last used 2026-04-12)
set -euo pipefail

PROJECT="bryan-usage-0"
ZONE="us-central1-a"
DEFAULT_VM="real8xh100paramgolf"
FALLBACK_VM="paramgolf8xh100s"
IMAGE_NAME="pgolf-golden-v2"
SSH_USER="ray"

VM="${2:-$DEFAULT_VM}"
ACTION="${1:-verify}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_ok()   { echo -e "${GREEN}[OK]${NC}   $1"; }
log_fail() { echo -e "${RED}[FAIL]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_info() { echo -e "       $1"; }

ssh_cmd() {
    gcloud compute ssh "${SSH_USER}@${VM}" \
        --zone="${ZONE}" \
        --project="${PROJECT}" \
        --strict-host-key-checking=no \
        --command="$1" 2>/dev/null
}

# --------------------------------------------------------------------------
# verify: Check all required dependencies on the running VM
# --------------------------------------------------------------------------
do_verify() {
    echo "=========================================="
    echo " Parameter Golf — Golden Image Verification"
    echo " VM: ${VM} (${ZONE})"
    echo "=========================================="
    echo ""

    local all_ok=true

    # Check VM is running
    local status
    status=$(gcloud compute instances describe "${VM}" \
        --zone="${ZONE}" --project="${PROJECT}" \
        --format='value(status)' 2>/dev/null || echo "NOT_FOUND")
    if [[ "${status}" != "RUNNING" ]]; then
        log_fail "VM ${VM} is ${status}. Start it first:"
        log_info "  gcloud compute instances start ${VM} --zone=${ZONE} --project=${PROJECT}"
        return 1
    fi
    log_ok "VM is RUNNING"

    # 1. NVIDIA drivers + GPUs
    echo ""
    echo "--- GPU / NVIDIA Drivers ---"
    if ssh_cmd "nvidia-smi --query-gpu=name,driver_version --format=csv,noheader" 2>/dev/null; then
        local gpu_count
        gpu_count=$(ssh_cmd "nvidia-smi --query-gpu=name --format=csv,noheader | wc -l")
        if [[ "${gpu_count}" -ge 8 ]]; then
            log_ok "nvidia-smi: ${gpu_count} GPUs detected"
        else
            log_warn "Only ${gpu_count} GPUs detected (expected 8 for a3-highgpu-8g)"
        fi
    else
        log_fail "nvidia-smi not found or not working"
        all_ok=false
    fi

    # 2. Python 3.13
    echo ""
    echo "--- Python ---"
    local py_ver
    py_ver=$(ssh_cmd "python3 --version 2>&1" || echo "NOT_FOUND")
    if echo "${py_ver}" | grep -q "3.1[3-9]"; then
        log_ok "Python: ${py_ver}"
    else
        # Check for python3.13 specifically
        py_ver=$(ssh_cmd "python3.13 --version 2>&1" || echo "NOT_FOUND")
        if echo "${py_ver}" | grep -q "3.13"; then
            log_ok "Python 3.13 found (as python3.13)"
            log_warn "python3 points elsewhere; may need symlink"
        else
            log_fail "Python 3.13 not found (got: ${py_ver})"
            all_ok=false
        fi
    fi

    # 3. PyTorch with CUDA
    echo ""
    echo "--- PyTorch ---"
    local torch_check
    torch_check=$(ssh_cmd "python3 -c \"import torch; print(f'torch={torch.__version__} cuda={torch.cuda.is_available()} devices={torch.cuda.device_count()}')\"" 2>&1 || echo "IMPORT_ERROR")
    if echo "${torch_check}" | grep -q "cuda=True"; then
        log_ok "PyTorch: ${torch_check}"
    else
        log_fail "PyTorch CUDA check failed: ${torch_check}"
        all_ok=false
    fi

    # 4. Triton
    echo ""
    echo "--- Triton ---"
    local triton_ver
    triton_ver=$(ssh_cmd "python3 -c \"import triton; print(triton.__version__)\"" 2>&1 || echo "NOT_FOUND")
    if [[ "${triton_ver}" != "NOT_FOUND" && "${triton_ver}" != *"Error"* ]]; then
        log_ok "Triton: ${triton_ver}"
    else
        log_fail "Triton not found: ${triton_ver}"
        all_ok=false
    fi

    # 5. Other Python deps
    echo ""
    echo "--- Python packages ---"
    for pkg in sentencepiece numpy tqdm psutil; do
        local pkg_check
        pkg_check=$(ssh_cmd "python3 -c \"import ${pkg}; print(getattr(${pkg}, '__version__', 'ok'))\"" 2>&1 || echo "NOT_FOUND")
        if [[ "${pkg_check}" != "NOT_FOUND" && "${pkg_check}" != *"Error"* ]]; then
            log_ok "${pkg}: ${pkg_check}"
        else
            log_fail "${pkg}: ${pkg_check}"
            all_ok=false
        fi
    done

    # 6. Training data
    echo ""
    echo "--- Training Data ---"
    local data_dir
    # Check common locations
    for candidate in "/home/${SSH_USER}/parameter-golf/data" "/workspace/parameter-golf/data" "/root/parameter-golf/data"; do
        local found
        found=$(ssh_cmd "ls -d ${candidate}/datasets/fineweb10B_sp1024 2>/dev/null" || echo "")
        if [[ -n "${found}" ]]; then
            data_dir="${candidate}"
            break
        fi
    done

    if [[ -n "${data_dir:-}" ]]; then
        log_ok "Data directory found: ${data_dir}"

        # Count training shards
        local train_count
        train_count=$(ssh_cmd "ls ${data_dir}/datasets/fineweb10B_sp1024/fineweb_train_*.bin 2>/dev/null | wc -l" || echo "0")
        if [[ "${train_count}" -ge 80 ]]; then
            log_ok "Training shards: ${train_count} (full dataset)"
        elif [[ "${train_count}" -ge 1 ]]; then
            log_warn "Training shards: ${train_count} (partial — full dataset is 195 shards)"
        else
            log_fail "No training shards found"
            all_ok=false
        fi

        # Validation data
        local val_count
        val_count=$(ssh_cmd "ls ${data_dir}/datasets/fineweb10B_sp1024/fineweb_val_*.bin 2>/dev/null | wc -l" || echo "0")
        if [[ "${val_count}" -ge 1 ]]; then
            log_ok "Validation shards: ${val_count}"
        else
            log_fail "No validation shards found"
            all_ok=false
        fi

        # Tokenizer
        local tok_found
        tok_found=$(ssh_cmd "ls ${data_dir}/tokenizers/fineweb_1024_bpe.model 2>/dev/null" || echo "")
        if [[ -n "${tok_found}" ]]; then
            log_ok "Tokenizer: fineweb_1024_bpe.model found"
        else
            log_fail "Tokenizer not found"
            all_ok=false
        fi
    else
        log_fail "Training data directory not found in common locations"
        all_ok=false
    fi

    # 7. parameter-golf repo
    echo ""
    echo "--- Repository ---"
    local repo_dir
    for candidate in "/home/${SSH_USER}/parameter-golf" "/workspace/parameter-golf" "/root/parameter-golf"; do
        local found
        found=$(ssh_cmd "ls ${candidate}/experiment1.py 2>/dev/null" || echo "")
        if [[ -n "${found}" ]]; then
            repo_dir="${candidate}"
            break
        fi
    done

    if [[ -n "${repo_dir:-}" ]]; then
        log_ok "Repo found: ${repo_dir}"
        local scripts_ok=true
        for script in experiment1.py train_gpt.py kernels.py; do
            if ssh_cmd "test -f ${repo_dir}/${script}" 2>/dev/null; then
                log_ok "  ${script} present"
            else
                log_fail "  ${script} missing"
                scripts_ok=false
                all_ok=false
            fi
        done
    else
        log_fail "parameter-golf repo not found"
        all_ok=false
    fi

    # 8. Disk space
    echo ""
    echo "--- Disk Space ---"
    local disk_info
    disk_info=$(ssh_cmd "df -h / | tail -1")
    log_info "Root disk: ${disk_info}"

    echo ""
    echo "=========================================="
    if ${all_ok}; then
        log_ok "ALL CHECKS PASSED — VM is ready for golden image creation"
    else
        log_fail "SOME CHECKS FAILED — fix issues before creating image"
    fi
    echo "=========================================="

    ${all_ok}
}

# --------------------------------------------------------------------------
# fix: Install missing dependencies on the running VM
# --------------------------------------------------------------------------
do_fix() {
    echo "--- Fixing missing dependencies on ${VM} ---"

    # Install Python 3.13 if missing
    ssh_cmd "python3.13 --version 2>/dev/null" || {
        log_info "Installing Python 3.13..."
        ssh_cmd "sudo add-apt-repository -y ppa:deadsnakes/ppa && \
                 sudo apt-get update && \
                 sudo apt-get install -y python3.13 python3.13-venv python3.13-dev && \
                 sudo ln -sf /usr/bin/python3.13 /usr/bin/python3"
    }

    # Install PyTorch if missing
    ssh_cmd "python3 -c 'import torch'" 2>/dev/null || {
        log_info "Installing PyTorch..."
        ssh_cmd "pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130"
    }

    # Install Triton if missing
    ssh_cmd "python3 -c 'import triton'" 2>/dev/null || {
        log_info "Installing Triton..."
        ssh_cmd "pip install triton==3.6.0"
    }

    # Install remaining deps
    ssh_cmd "pip install sentencepiece==0.2.1 numpy==2.4.4 tqdm==4.67.3 \
             typing_extensions==4.15.0 setuptools==81.0.0 psutil==7.2.2 2>/dev/null" || true

    # Clone repo if missing
    local repo_found=false
    for candidate in "/home/${SSH_USER}/parameter-golf" "/workspace/parameter-golf" "/root/parameter-golf"; do
        if ssh_cmd "test -f ${candidate}/experiment1.py" 2>/dev/null; then
            repo_found=true
            log_ok "Repo already at ${candidate}"
            break
        fi
    done
    if ! ${repo_found}; then
        log_info "Cloning parameter-golf repo..."
        ssh_cmd "cd /home/${SSH_USER} && git clone https://github.com/openai/parameter-golf.git"
    fi

    # Download training data if missing
    local data_dir="/home/${SSH_USER}/parameter-golf/data"
    local train_count
    train_count=$(ssh_cmd "ls ${data_dir}/datasets/fineweb10B_sp1024/fineweb_train_*.bin 2>/dev/null | wc -l" || echo "0")
    if [[ "${train_count}" -lt 10 ]]; then
        log_info "Downloading FineWeb training data (this takes a few minutes)..."
        ssh_cmd "cd /home/${SSH_USER}/parameter-golf && python3 data/cached_challenge_fineweb.py --variant sp1024"
    else
        log_ok "Training data already present (${train_count} shards)"
    fi

    log_ok "Fix phase complete"
}

# --------------------------------------------------------------------------
# create: Create a golden image from the VM's boot disk
# --------------------------------------------------------------------------
do_create() {
    echo "=========================================="
    echo " Parameter Golf — Golden Image Creation"
    echo " Source VM: ${VM} (${ZONE})"
    echo " Image name: ${IMAGE_NAME}"
    echo "=========================================="

    # VM must be stopped
    local status
    status=$(gcloud compute instances describe "${VM}" \
        --zone="${ZONE}" --project="${PROJECT}" \
        --format='value(status)' 2>/dev/null || echo "NOT_FOUND")
    if [[ "${status}" != "TERMINATED" ]]; then
        echo ""
        log_warn "VM is ${status}. Stopping it for image creation..."
        gcloud compute instances stop "${VM}" \
            --zone="${ZONE}" --project="${PROJECT}" --quiet
        echo "Waiting for VM to stop..."
        sleep 30
    fi

    # Get the boot disk name
    local boot_disk
    boot_disk=$(gcloud compute instances describe "${VM}" \
        --zone="${ZONE}" --project="${PROJECT}" \
        --format='value(disks[0].source)' | rev | cut -d'/' -f1 | rev)
    echo "Boot disk: ${boot_disk}"

    # Check if image already exists
    if gcloud compute images describe "${IMAGE_NAME}" --project="${PROJECT}" &>/dev/null; then
        log_warn "Image ${IMAGE_NAME} already exists. Deleting it first..."
        gcloud compute images delete "${IMAGE_NAME}" --project="${PROJECT}" --quiet
    fi

    # Create the image with multi-regional storage for cross-zone use
    echo ""
    echo "Creating image (this takes 5-15 minutes)..."
    gcloud compute images create "${IMAGE_NAME}" \
        --project="${PROJECT}" \
        --source-disk="${boot_disk}" \
        --source-disk-zone="${ZONE}" \
        --storage-location=us \
        --description="Parameter Golf golden image: Ubuntu 24.04, CUDA 12.8, PyTorch 2.11, Triton 3.6, FineWeb 10B dataset" \
        --labels=purpose=parameter-golf,created-by=automation

    echo ""
    log_ok "Image created: ${IMAGE_NAME}"
    echo ""
    echo "To use this image, update infra/gce_config.yaml:"
    echo "  golden_image: \"${IMAGE_NAME}\""
    echo ""
    echo "Verify with:"
    echo "  gcloud compute images describe ${IMAGE_NAME} --project=${PROJECT}"
}

# --------------------------------------------------------------------------
# full: Verify, fix, then create image
# --------------------------------------------------------------------------
do_full() {
    # Start VM if not running
    local status
    status=$(gcloud compute instances describe "${VM}" \
        --zone="${ZONE}" --project="${PROJECT}" \
        --format='value(status)' 2>/dev/null || echo "NOT_FOUND")
    if [[ "${status}" != "RUNNING" ]]; then
        echo "Starting VM ${VM}..."
        gcloud compute instances start "${VM}" \
            --zone="${ZONE}" --project="${PROJECT}"
        echo "Waiting for VM to boot (60s)..."
        sleep 60
    fi

    # Wait for SSH
    echo "Waiting for SSH..."
    local attempts=0
    while ! ssh_cmd "echo ready" &>/dev/null; do
        attempts=$((attempts + 1))
        if [[ ${attempts} -ge 30 ]]; then
            log_fail "SSH timed out after 5 minutes"
            exit 1
        fi
        sleep 10
    done
    log_ok "SSH connected"

    # Verify
    if ! do_verify; then
        echo ""
        echo "Verification failed. Running fix..."
        do_fix
        echo ""
        echo "Re-verifying..."
        do_verify || {
            log_fail "Verification still failing after fix. Manual intervention needed."
            exit 1
        }
    fi

    # Stop and create image
    echo ""
    echo "Stopping VM for image creation..."
    gcloud compute instances stop "${VM}" \
        --zone="${ZONE}" --project="${PROJECT}" --quiet
    echo "Waiting for VM to stop (30s)..."
    sleep 30

    do_create
}

# --------------------------------------------------------------------------
# Main dispatch
# --------------------------------------------------------------------------
case "${ACTION}" in
    verify)
        do_verify
        ;;
    fix)
        do_fix
        ;;
    create)
        do_create
        ;;
    full)
        do_full
        ;;
    *)
        echo "Usage: $0 {verify|fix|create|full} [VM_NAME]"
        echo ""
        echo "  verify  - Check if VM has all required dependencies"
        echo "  fix     - Install missing dependencies on running VM"
        echo "  create  - Create golden image from stopped VM"
        echo "  full    - Start VM, verify, fix, stop, create image"
        exit 1
        ;;
esac
