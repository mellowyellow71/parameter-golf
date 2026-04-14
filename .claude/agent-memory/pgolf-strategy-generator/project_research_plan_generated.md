---
name: Research Plan Generation Status
description: Tracks the comprehensive strategy research plan and v2 screening list for Parameter Golf competition
type: project
---

## Research Plan Files
- Original plan: `/home/ray/parameter-golf/pgolf_research_plan.md` (87 strategies, 2026-04-04)
- **Screening list v2**: `/home/ray/parameter-golf/infra/strategies_v2.py` (112 strategies, 2026-04-13)
  - Category A (HP Sweeps): 50 -- env-var only, no code changes
  - Category B (Architecture Mix): 32 -- needs code changes
  - Category C (Novel Ideas): 20 -- untried techniques
  - Category D (Different Base): 10 -- different architectures

## Current Winning Base (1.0810 BPB)
- QK_GAIN_INIT=5.25, MLP_MULT=4.0, NUM_LAYERS=11, MODEL_DIM=512
- NUM_LOOPS=2, LOOP_START=3, LOOP_END=5, ENABLE_LOOPING_AT=0.25
- PARALLEL_RESIDUAL_START=7, SKIP_GATES_ENABLED=1, XSA_LAST_N=11
- GPTQ: matrix_bits=6, embed_bits=8, matrix_clip_sigmas=12.85, brotli compression

## Key Env Vars Available (no code changes needed)
- WARMDOWN_SHAPE: sqrt|cosine (cooldown curve)
- GPTQ_ADAPTIVE_CLIP: 0|1 (per-layer clip sigmas)
- BATCH_WARMUP_FRAC: float (batch size ramp)
- EMBED_BITS: 7|8 (embedding quantization bits)
- TTT_ENABLED: 0|1, TTT_LR, TTT_EPOCHS
- ETLB_ENABLED: 0|1, ETLB_LR, ETLB_STEPS (SLOT -- needs eval_val_sliding_etlb impl)

## Immediate P0 Actions (15 strategies)
1. sqrt cooldown + adaptive GPTQ (env vars, run today)
2. Batch warmup 0.3 (env var)
3. Mega combos v1/v2/v3 (stacking env vars)
4. SLOT implementation (biggest single gain: -15 to -30 mBPB)
5. N-gram cache (GRAY_AREA legality, -50 to -150 mBPB)

## Smoke Test Results (from screen_state.json, 15 passed)
- Best smoke: win-combo-qk525-lr025 (loss=2.5208)
- Smoke tests only detect broken experiments -- qualify (step-1000 BPB) is the real decision point

## Confirmed Dead Ends (Do NOT Retry)
- MoE at this scale, INT4 quant, 2:4 sparsity training, Turbo-Muon on 8xH100
- Knowledge distillation, MC Dropout ensembling, Product quantization

**Why:** Provides the complete roadmap for 16 remaining days (deadline Apr 30).
**How to apply:** Use strategies_v2.py as the canonical strategy list for mass_screen.py.
