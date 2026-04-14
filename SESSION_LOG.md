# Parameter Golf Session Log — April 13, 2026

## What Was Asked
The user wants to win the OpenAI Parameter Golf competition: train the best language model that fits in 16MB, trains in 10 minutes on 8xH100, measured by bits-per-byte (BPB) on FineWeb. They have ~87 strategies identified and want an automated pipeline to test them all, iterate, and find the optimal architecture.

## What We Built (Chronological)

### Phase 1: GCE Infrastructure (completed)
**Problem**: Need to provision 8xH100 GPU clusters across GCP, but availability is sparse.
**Solution**: Built a zone-scanning provisioner that checks all 19 GCP zones for H100s.

Files created:
- `infra/gce_config.yaml` — 19 zones, SPOT preferences, thresholds
- `infra/gce_provision.py` — Zone scanner + instance creation/deletion with machine type override
- `infra/gce_run_experiment.py` — Full experiment lifecycle: provision → SSH → train → monitor → upload → teardown
- `infra/gce_batch.py` — Batch orchestrator for 49 pre-defined strategies
- `infra/gce_golden_image.sh` — Script to verify/create golden disk images

**Golden Image**: `pgolf-golden-v2` — Ubuntu 24.04, CUDA 12.8 driver, Python 3.13, PyTorch 2.11+cu126, Triton 3.6, pre-loaded training data.

**Key Issue Found**: PyTorch cu130 doesn't work with CUDA 12.8 driver. Fixed by installing cu126.

### Phase 2: Autoresearch + Dashboard (completed)
**Problem**: Need autonomous experimentation and monitoring.
**Solution**: Built an autoresearch loop (hypothesis generation from results) and Gradio dashboard.

Files created:
- `infra/autoresearch.py` — Autonomous research loop: ANALYZE → HYPOTHESIZE → SELECT → DISPATCH → MONITOR → RECORD. Generates interpolation, combination, and perturbation experiments.
- `infra/dashboard.py` — Gradio web dashboard at localhost:7860 with experiment table, BPB charts, timeline, tier analysis, autoresearch status.
- `infra/evo_benchmark.py` — Bridge between evo plugin and our GCE infrastructure.

### Phase 3: Evo Integration (completed)
**Problem**: Want to use evo-hq/evo plugin for tree-based parallel optimization.
**Solution**: Installed evo, configured it with our benchmark, established baseline.

- Evo installed as Claude Code plugin (`/evo:discover`, `/evo:optimize`)
- Benchmark dispatches to real H100 GPUs via our infrastructure
- Inline instrumentation mode (no SDK dependency)
- Gates: syntax_check, artifact_size
- Evo dashboard at http://127.0.0.1:8080

### Phase 4: First Experiments on Old Architecture (SP1024)
Ran experiments on the original SP1024 baseline (experiment1.py):

| Experiment | Change | BPB | Result |
|------------|--------|-----|--------|
| exp_0003 | Baseline | 1.1536 | Committed |
| exp_0004 | QK-Gain=4.0 + XSA-11 | 1.1527 | **Best** (-0.0009) |
| exp_0005 | WD=0.06 + warmdown=4000 | 1.1559 | Discarded (regression) |
| exp_0007 | QK-Gain=5.0 + XSA-11 | 1.1535 | Discarded (5.0 worse than 4.0) |
| exp_0008 | MLP 3.5x + QK4.0 + XSA-11 | ~1.1485 | Preempted (promising) |

**Key Learning**: QK-Gain=4.0 is the sweet spot on SP1024. Higher WD hurts with limited data.

### Phase 5: Major Pivot — Winning Architecture Discovery
**Problem**: Our SP1024 architecture is 0.07 BPB behind the competition winners.
**Discovery**: The winning submissions use a completely different stack:
- **SP8192 tokenizer** (8x larger vocab)
- **3-layer depth recurrence** (17 virtual layers from 11 physical)
- **Parallel residuals** (GPT-J style from layer 7+)
- **Skip gates** (sigmoid-gated U-Net connections)
- **Score-first TTT** (test-time training at eval)
- **MuonEq-R** (row-normalized Muon optimizer)
- **SDClip GPTQ** (principled quantization clipping)
- **Brotli-11** compression
- Result: **1.0810 BPB** vs our 1.1527

### Phase 6: Efficiency Redesign — Funnel Pipeline (completed)
**Problem**: Full 8xH100 runs cost $5-8 each with 43% SPOT preemption failure rate. Can't screen 80+ strategies this way.
**Solution**: 3-stage funnel with 1xH100 smoke tests.

Files created:
- `winning_base_decoded.py` — The decoded winning train_gpt.py as our new base
- `infra/funnel.py` — Multi-stage pipeline:
  - **Smoke**: 1xH100 (`a3-highgpu-1g`), 5 min, ~$0.25 — screens for crashes/convergence
  - **Qualify**: 8xH100, 3 min, ~$2 — step-1000 BPB estimate
  - **Full**: 8xH100, 10 min, ~$5-8 — final BPB with GPTQ

Key insight: 1xH100 instances are far more available and 8x cheaper. The training script supports world_size=1 natively.

### Phase 7: Mass Architecture Screening (IN PROGRESS)
**Problem**: Need to test 80+ architectures efficiently, not just tweak one.
**Solution**: Mass screening pipeline that tests many variants in parallel, ranks them, and promotes the best to evo tree nodes.

File created:
- `infra/mass_screen.py` — 29 strategies defined across 3 tiers, screened 3 at a time on 1xH100

**Current screening results (4/16 Tier 1 completed):**

| Rank | Train Loss | Strategy | vs Baseline |
|------|-----------|----------|-------------|
| 1 | 2.5386 | QK-Gain 5.25 | -0.0013 (better) |
| 2 | 2.5408 | QK-Gain 5.5 | +0.0009 (worse) |
| 3 | 2.5416 | Parallel start=5 | +0.0017 (worse) |
| - | error | QK-Gain 4.75 | Provisioning timeout |

Baseline = 2.5399 (winning_base_decoded.py unmodified on 1xH100)

## What Happens Next

1. **Mass screening completes** (~30 min remaining for 12 more strategies)
2. **Top 5 promoted to evo** — each becomes a branch in the evo tree
3. **`/evo:optimize`** iterates on ALL branches simultaneously
4. **Best overall winner** gets a full 8xH100 10-minute run
5. **If competitive** → custom kernel optimization → submission

## Architecture of the System

```
mass_screen.py (29 strategies)
    ↓ parallel smoke tests (1xH100, $0.25 each)
    ↓ rank by train_loss
    ↓
promote top 5 → evo tree nodes
    ↓
/evo:optimize (parallel subagents)
    ├─ Branch A: best architecture variant
    ├─ Branch B: second best
    ├─ Branch C: third best
    ├─ Branch D: ...
    └─ Branch E: ...
    ↓ each iterates via smoke tests
    ↓
Best overall → funnel qualify (8xH100 short) → funnel full (8xH100 10min)
    ↓
Submission
```

## Cost Tracking
- Golden image creation: ~$1 (cheap VM)
- Old experiments (8xH100): ~$30 (6 runs × ~$5)
- New smoke tests (1xH100): ~$4 so far (16 × $0.25)
- Estimated total to screen all 29: ~$7.25
- Estimated total for evo optimization: ~$20-50
- **Total budget used: ~$35**

## Key Files Reference
| File | Purpose |
|------|---------|
| `winning_base_decoded.py` | Current best architecture (1.0810 BPB base) |
| `experiment1.py` | Old SP1024 architecture (1.1527 BPB) |
| `infra/mass_screen.py` | Mass architecture screening |
| `infra/funnel.py` | Smoke → Qualify → Full pipeline |
| `infra/evo_benchmark.py` | Evo ↔ funnel bridge |
| `infra/gce_provision.py` | GCE zone scanner + provisioner |
| `infra/dashboard.py` | Gradio monitoring dashboard |
| `infra/autoresearch.py` | Autonomous hypothesis generation |
| `infra/gce_config.yaml` | All GCE configuration |
| `SETUP.md` | Full operations guide |
| `pgolf_research_plan.md` | 87 strategies across 3 tiers |

## Bugs Fixed Along the Way
1. PyTorch cu130 vs cu126 — CUDA driver compatibility
2. flash_attn_interface import — fallback to PyTorch SDPA
3. `extra_files` kwarg — funnel API mismatch
4. `SmokeResult` dict vs dataclass — subscript vs attribute access
5. HuggingFace data download hanging — accept partial data
6. SSH connection drops — too many concurrent sessions
7. SPOT preemption — zone failover + retry logic
