# Migration Guide — H100 (GCP) Setup for experiment1.py

This document is intended for the remote agent setting up the environment on the target machine.

---

## Source Environment

| Property | Value |
|---|---|
| Python | 3.13.12 |
| CUDA (torch build) | 13.0 |
| Target GPU (source) | RTX 3060 (SM86, Ampere) |
| Target GPU (destination) | H100 (SM90, Hopper) |

---

## Files to Transfer

The following files must be present in the working directory on the remote machine:

```
experiment1.py
experiment2.py
kernels.py          # local Triton kernel module — NOT the pip package
final_model.pt
final_model.int6.ptz
final_model.int8.ptz
```

Transfer via `gcloud compute scp`, `rsync`, or your preferred method.

---

## Step 1 — System Prerequisites

Ensure the remote machine has the following before proceeding:

- Python 3.13.x (`python3.13` or `python3`)
- `pip` for Python 3.13
- CUDA 13.0 drivers (`nvidia-smi` should show Driver >= 570.x for CUDA 13.0)
- `git` (for any source installs if needed)

Verify CUDA is visible:

```bash
nvidia-smi
```

---

## Step 2 — Create the Virtual Environment

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

---

## Step 3 — Install PyTorch (CUDA 13 build)

The source environment uses `torch==2.11.0` built against CUDA 13. Install from the PyTorch nightly/cu13 index:

```bash
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
```

If `cu130` is not available at that index (PyTorch may not yet publish a stable `cu130` wheel), use the nightly channel:

```bash
pip install --pre torch==2.11.0 --index-url https://download.pytorch.org/whl/nightly/cu130
```

Verify:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
```

Expected: `2.11.0`, `True`, `13.0`

---

## Step 4 — Install Remaining Dependencies

```bash
pip install \
  triton==3.6.0 \
  numpy==2.4.4 \
  sentencepiece==0.2.1 \
  tqdm==4.67.3 \
  datasets==4.8.4 \
  huggingface_hub==1.7.2 \
  tiktoken==0.12.0 \
  typing_extensions==4.15.0 \
  setuptools==81.0.0 \
  psutil==7.2.2
```

> **Note:** Do NOT install the `kernels` PyPI package. The `kernels` module used here is the local `kernels.py` file in the project directory. The pip package of the same name is unrelated.

---

## Step 5 — Data Setup

`experiment1.py` expects two data paths (overridable via environment variables):

| Env var | Default path |
|---|---|
| `DATA_PATH` | `./data/datasets/fineweb10B_sp1024` |
| `TOKENIZER_PATH` | `./data/tokenizers/fineweb_1024_bpe.model` |

Populate these directories with the FineWeb 10B dataset (pre-tokenized `.bin` shards) and the SentencePiece tokenizer model. If the data lives elsewhere, export the env vars before running:

```bash
export DATA_PATH=/path/to/fineweb10B_sp1024
export TOKENIZER_PATH=/path/to/fineweb_1024_bpe.model
```

---

## Step 6 — H100-Specific Notes

### Triton kernel compatibility
`kernels.py` contains Triton kernels with block configs labeled `_SM86_CONFIGS` (tuned for Ampere/RTX 3060). These configs will still run correctly on H100 (SM90, Hopper) — Triton's `@triton.autotune` will benchmark them and pick the fastest. They are not optimal for H100; if you want maximum throughput, the block configs in `kernels.py` can be extended with SM90 entries, but this is not required for correctness.

### Flash Attention
`experiment1.py` implements attention via `torch.nn.functional.scaled_dot_product_attention` with `enable_gqa=True`. On H100, PyTorch will automatically dispatch to FlashAttention-3 (FA3) if available in the installed torch build. No changes needed.

### bfloat16
The model trains in bfloat16. H100 has native bfloat16 tensor core support — no configuration needed.

### Triton GEMM flag
In `experiment1.py`, `_USE_TRITON_GEMM = True` enables the custom Triton GEMM kernel. On H100, cuBLAS is highly optimized and may outperform the Ampere-tuned Triton configs. Monitor step times and set `_USE_TRITON_GEMM = False` (or set via env) if the Triton path is slower.

---

## Step 7 — Verify the Setup

```bash
python -c "
import torch, triton, numpy, sentencepiece, datasets, tiktoken
print('torch:', torch.__version__)
print('triton:', triton.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0))
from kernels import fused_softcap_cross_entropy, tuned_linear
print('kernels.py loaded OK')
"
```

---

## Step 8 — Run

```bash
python experiment1.py
```

Key env vars for quick iteration:

```bash
ITERATIONS=1000 VAL_LOSS_EVERY=200 MAX_WALLCLOCK_SECONDS=3600 python experiment1.py
```

---

## Quick Reference — Pinned Versions

```
torch==2.11.0          (CUDA 13.0 build)
triton==3.6.0
numpy==2.4.4
sentencepiece==0.2.1
tqdm==4.67.3
datasets==4.8.4
huggingface_hub==1.7.2
tiktoken==0.12.0
typing_extensions==4.15.0
setuptools==81.0.0
psutil==7.2.2
Python 3.13.x
```
