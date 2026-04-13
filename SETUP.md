# Parameter Golf — Full Setup & Operations Guide

## Project Overview
OpenAI Model Craft Challenge: Train the best language model in a 16MB artifact, 10 minutes on 8xH100, evaluated by bits-per-byte (BPB) on FineWeb validation set. Lower BPB = better.

- **Current best**: exp_0004, BPB=1.1527 (QK-Gain=4.0 + XSA all 11 layers)
- **Baseline**: exp_0003, BPB=1.1536
- **Competition SOTA**: 1.1147 (official), 0.9300 (pending with Scylla+SLOT)
- **Target**: sub-1.10 BPB

## GCP Infrastructure

### Golden Image
- **Name**: `pgolf-golden-v2`
- **Project**: `bryan-usage-0`
- **Contents**: Ubuntu 24.04, CUDA 12.8 driver (570), Python 3.13, PyTorch 2.11+cu126, Triton 3.6, 9 train shards + 1 val shard, experiment1.py + kernels
- **Storage**: Multi-regional US, 200GB, 50GB compressed

### Machine Type
- `a3-highgpu-8g` — 8x NVIDIA H100 80GB, 208 vCPUs, 1872GB RAM
- Bundles 16 local SSDs automatically — do NOT use `--accelerator` flag
- Available in 19 zones (see `infra/gce_config.yaml`)

### Instance Creation
```bash
# SPOT (cheaper, can be preempted)
gcloud compute instances create INSTANCE_NAME \
    --project=bryan-usage-0 --zone=ZONE \
    --machine-type=a3-highgpu-8g \
    --image=pgolf-golden-v2 --image-project=bryan-usage-0 \
    --boot-disk-size=200GB --boot-disk-type=pd-ssd \
    --network-interface=network=default,nic-type=GVNIC \
    --scopes=cloud-platform \
    --maintenance-policy=TERMINATE \
    --provisioning-model=SPOT \
    --instance-termination-action=STOP \
    --no-restart-on-failure

# On-demand (stable, more expensive)
# Same command without --provisioning-model and --instance-termination-action
```

### Known Issues
- SPOT H100s get preempted frequently (often within minutes)
- H100 capacity fluctuates — sometimes all 19 zones are exhausted
- HuggingFace data downloads hang on small VMs; use the golden image's pre-loaded data
- PyTorch must be cu126 (not cu130) to work with CUDA 12.8 driver

## File Structure

### Training Scripts (repo root)
- `experiment1.py` — Main training script (2491 lines), all hyperparameters as env vars
- `experiment2.py` — Enhanced variant with more features
- `train_gpt.py` — Baseline/reference script
- `kernels.py` — Triton kernels for fused softcap+CE and tuned GEMM

### Infrastructure (`infra/`)
| File | Purpose |
|------|---------|
| `gce_config.yaml` | 19 zones, SPOT prefs, early-kill thresholds |
| `gce_provision.py` | Zone scanning + instance creation/deletion |
| `gce_run_experiment.py` | Full experiment lifecycle: provision → SSH → train → monitor → upload → teardown |
| `gce_batch.py` | Batch orchestrator: 49 strategies, parallel execution, state persistence |
| `gce_golden_image.sh` | Verify/fix/create golden images |
| `autoresearch.py` | Autonomous research loop with hypothesis generation |
| `dashboard.py` | Gradio real-time monitoring dashboard |
| `evo_benchmark.py` | Bridge for evo plugin (dispatches to H100, returns BPB) |
| `evo_benchmark.sh` | Shell wrapper for evo benchmark |
| `launch.py` | Vertex AI Custom Job launcher (alternative path) |
| `experiments.yaml` | Vertex AI experiment configs |
| `batch_state.json` | Persistent state for batch/autoresearch experiments |
| `autoresearch_state.json` | Persistent state for autoresearch hypothesis generation |

### Data (`data/`)
- `datasets/fineweb10B_sp1024/` — Tokenized training/validation shards (~200MB each)
- `tokenizers/fineweb_1024_bpe.model` — 1024-token SentencePiece BPE tokenizer
- `cached_challenge_fineweb.py` — Downloads data from HuggingFace Hub

### Research Planning
- `pgolf_research_plan.md` — 87 strategies across 3 tiers with implementation plans
- `parametergolfanalyzer.md` — Deep analysis of competition strategies
- `prompt.md` — Research prompt context
- `CompInstructions.md` — Competition rules and leaderboard

## How To Run Everything

### 1. Start the Dashboard
```bash
cd ~/parameter-golf
python3 infra/dashboard.py &
# Opens http://localhost:7860
# Auto-refreshes every 30s from batch_state.json
```

### 2. Start Autoresearch (Hyperparameter Grid Search)
```bash
# Dry run first
python3 infra/autoresearch.py --dry-run --max-cycles 3

# Run for real (sequential)
python3 infra/autoresearch.py --parallel 1

# Run with 2 parallel H100 instances
python3 infra/autoresearch.py --parallel 2

# Check status
python3 infra/autoresearch.py status
```

The autoresearch loop:
1. Runs pre-defined Tier 1 strategies first (28 strategies, highest impact)
2. Analyzes results → identifies which hyperparameters matter most
3. Auto-generates new hypotheses (interpolation, combination, perturbation)
4. Avoids configs similar to early-killed experiments
5. Tracks GPU hours and cost
6. Continues until Ctrl+C (graceful shutdown)

### 3. Start Evo Optimization (Code-Level Architecture Changes)
```bash
# Check evo state
evo tree
evo scratchpad
evo status

# Create a new experiment branching from the best
evo new --parent exp_0004 -m "description of change"
# Edit the experiment1.py in the worktree path shown by evo new

# Run it (dispatches to H100, ~15 min)
evo run exp_XXXX --timeout 3600

# Record result
evo done exp_XXXX --score <BPB>
# or if it regressed:
evo discard exp_XXXX --reason "why"

# Annotate for other agents to learn from
evo annotate exp_XXXX "what changed, what happened"
```

For autonomous evo optimization, use `/evo:optimize` in Claude Code. It launches parallel subagents that modify experiment1.py, dispatch to H100s, evaluate, and iterate.

### 4. Run a Single Experiment Manually
```bash
# Via the experiment runner
python3 infra/gce_run_experiment.py run \
    --name "my-test" \
    --script experiment1.py \
    --env QK_GAIN_INIT=4.0 XSA_LAST_N=11 SEED=1337

# Or directly on a running instance
gcloud compute ssh ray@INSTANCE --zone=ZONE --project=bryan-usage-0 \
    --command="cd /home/ray/parameter-golf && \
    export RUN_ID=my-test && \
    torchrun --standalone --nproc_per_node=8 experiment1.py"
```

### 5. Batch Run Specific Strategies
```bash
# List all strategies
python3 infra/gce_batch.py list

# Run specific ones
python3 infra/gce_batch.py run --strategies T1-04,T1-07,T1-02

# Run all Tier 1
python3 infra/gce_batch.py run --tier 1 --parallel 2

# Resume after interruption
python3 infra/gce_batch.py run --tier 1 --resume

# Check status
python3 infra/gce_batch.py status
```

## Key Hyperparameters (experiment1.py)

All controlled via environment variables in the `Hyperparameters` class (lines 41-121):

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| QK_GAIN_INIT | 1.5 | 0.5-10.0 | Attention sharpness (4.0 proven good) |
| MLP_MULT | 3.0 | 2.0-5.0 | MLP width multiplier |
| NUM_LAYERS | 11 | 8-15 | Model depth |
| MODEL_DIM | 512 | 384-768 | Hidden dimension |
| XSA_LAST_N | 4 | 0-11 | Cross-stream attention layers (11=all) |
| MATRIX_LR | 0.025 | 0.01-0.05 | Matrix parameter learning rate |
| SCALAR_LR | 0.025 | 0.01-0.05 | Scalar parameter learning rate |
| MUON_WD | 0.04 | 0.01-0.15 | Muon weight decay |
| ADAM_WD | 0.04 | 0.01-0.15 | Adam weight decay |
| WARMDOWN_ITERS | 3500 | 2000-6000 | Cooldown schedule length |
| SWA_EVERY | 50 | 10-200 | Stochastic weight averaging frequency |
| MUON_MOMENTUM | 0.99 | 0.90-0.999 | Muon optimizer momentum |
| LOGIT_SOFTCAP | 30.0 | 15.0-60.0 | Logit softcap value |
| GPTQ_ENABLED | 1 | 0/1 | Post-training quantization |
| GPTQ_N_BATCHES | 32 | 16-128 | GPTQ calibration batches |
| TTT_ENABLED | 0 | 0/1 | Test-time training |
| QAT_ENABLED | 0 | 0/1 | Quantization-aware training |

## 87 Strategies (from pgolf_research_plan.md)

### Tier 1 — High Priority (28 strategies)
Run first. Highest expected competitive impact.

Key ones:
- **T1-01**: Scylla tokenizer (998-token TokenMonster, biggest single technique gain)
- **T1-02/03**: SLOT eval-time optimization (zero artifact cost, -0.015 to -0.030 BPB)
- **T1-04**: QK-Gain sweep 2.0-6.0 (proven in #1125, #1176)
- **T1-05**: MLP 3.5x + mixed int5/int6 quantization
- **T1-07**: XSA on all 11 layers (proven in #1019)
- **T1-10**: Legal score-first TTT
- **T1-13**: WD=0.085 + MLP 4x simplification approach
- **T1-14**: Mousse optimizer (curvature-aware Muon)
- **T1-21/22**: Self-generated / training-data GPTQ calibration

### Tier 2 — Medium Priority (34 strategies)
Solid ideas for parallel exploration:
- Optimizer variants (Muon-VS, NuMuon, cyclic momentum)
- Architecture (GLU attention, FlashSigmoid, cross-layer KV sharing, differential attention)
- Training (focal loss, MTP auxiliary heads, batch scheduling, WSM checkpoint merging)
- Quantization (CAGE QAT, temperature scaling, CPSVD)

### Tier 3 — Exploratory (25 strategies)
High risk, high reward:
- Custom tokenizers (FineWeb-aligned, aggressive vocab pruning)
- Exotic architectures (Mixture of Depths, Hymba SSM, text diffusion)
- Advanced quantization (VPTQ, 2-3 bit, ScaleBITS)
- State-space models, hypernetworks

## Evo Plugin Setup

```bash
# Already installed. Check with:
evo status
evo tree

# Evo dashboard at http://127.0.0.1:8080
# Benchmark: python3 infra/evo_benchmark.py --target {target}
# Metric: min (lower BPB = better)
# Gates: syntax_check, artifact_size
# Instrumentation: inline mode
```

## Current Experiment Results

| ID | Change | BPB | Status |
|----|--------|-----|--------|
| exp_0003 | Baseline (11L, 512d, XSA-4, EMA, GPTQ) | 1.1536 | Committed |
| exp_0004 | QK-Gain=4.0 + XSA all 11L | 1.1527 | Committed (best) |
| exp_0005 | WD=0.06 + warmdown=4000 | Running | Active |
| exp_0006 | WD=0.06 only | Failed (preempted) | Failed |

## Cost Tracking
- 8xH100 SPOT: ~$24.48/hour
- 8xH100 on-demand: ~$98/hour
- Each experiment: ~$5-8 (12-15 min SPOT)
- Budget: unlimited (user has GCP credits)

## Resuming After Session Break

```bash
cd ~/parameter-golf

# Check what's running
gcloud compute instances list --project=bryan-usage-0 --filter="name~pgolf"
evo tree
python3 infra/autoresearch.py status
python3 infra/gce_batch.py status

# Restart systems
python3 infra/dashboard.py &                    # Dashboard
python3 infra/autoresearch.py --parallel 1 &    # Grid search
# Then use /evo:optimize in Claude Code for architecture search
```

## Troubleshooting

### "ZONE_RESOURCE_POOL_EXHAUSTED"
All H100 zones are at capacity. Wait and retry — capacity fluctuates. The provisioner auto-scans all 19 zones.

### "RuntimeError: CUDA is required"
Wrong PyTorch version. The golden image needs PyTorch cu126 (not cu130). Verify with:
```bash
python3 -c "import torch; print(torch.__version__, torch.version.cuda)"
# Should show: 2.11.0+cu126 12.6
```

### SSH connection drops
Too many concurrent SSH sessions. Wait 30s and retry. Training continues in tmux.

### Training log stuck at step N
Logs only print every 500 steps (TRAIN_LOG_EVERY=500). Step 500 appears at ~55s.

### Experiment times out in evo
Use `--timeout 3600` with `evo run`. Default 1800s may not be enough with SPOT preemption retries.
