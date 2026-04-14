# Parameter Golf — Evo Project Context

## What the target does
`experiment1.py` is a transformer language model training script for the OpenAI Parameter Golf competition. It trains a model on the FineWeb dataset and is evaluated by bits-per-byte (BPB) — lower is better.

## Architecture
- 11-layer transformer with 512-dim, 8 heads, 4 KV heads
- Muon optimizer with Newton-Schulz orthogonalization
- Features: XSA (cross-stream attention), EMA/SWA, QAT, GPTQ post-training quantization
- BigramHash embeddings, SmearGate, RoPE, softcap attention

## What can be changed
All hyperparameters are controlled via environment variables in the `Hyperparameters` class (lines 41-121). Key dimensions to optimize:

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| QK_GAIN_INIT | 1.5 | 0.5-10.0 | Attention sharpness |
| MLP_MULT | 3.0 | 2.0-5.0 | MLP width multiplier |
| NUM_LAYERS | 11 | 8-15 | Model depth |
| XSA_LAST_N | 4 | 0-11 | Cross-stream attention layers |
| MATRIX_LR | 0.025 | 0.01-0.05 | Matrix parameter learning rate |
| MUON_WD | 0.04 | 0.01-0.15 | Weight decay |
| WARMDOWN_ITERS | 3500 | 2000-6000 | Cooldown schedule length |

Architecture changes (adding new layer types, changing attention mechanisms, modifying quantization) are also valid.

## Benchmark
The benchmark dispatches to a remote GCE 8xH100 instance:
1. Provisions a3-highgpu-8g SPOT VM from golden image
2. SCPs the modified experiment1.py to the instance
3. Runs `torchrun --nproc_per_node=8 experiment1.py`
4. Parses final BPB from training output
5. Returns BPB as the score (lower = better)
6. Deletes the instance

Each run takes ~12-15 minutes (10 min training + provisioning overhead).

## Metric
- **Direction**: `min` (lower BPB = better compression = better model)
- **Current baseline**: ~1.12-1.13 BPB
- **Competition SOTA**: 1.1147 BPB (official), 0.9300 BPB (pending)
- **Target**: sub-1.10 BPB

## Constraints
- Artifact size must be < 16,000,000 bytes (code + compressed model)
- Training must complete in < 600 seconds on 8xH100
- No accessing validation data during training
- Test-time training must be backward-looking only (score-first)

## Gates
- `syntax_check`: Python syntax validation of target file
- `artifact_size`: Ensures model stays within 16MB limit

## Environment
- Remote execution on GCE a3-highgpu-8g (8x NVIDIA H100 80GB)
- SPOT instances across 19 zones (auto-failover)
- Ubuntu 24.04, CUDA 12.8, Python 3.13, PyTorch 2.11, Triton 3.6
