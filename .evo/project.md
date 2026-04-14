# Parameter Golf — Evo Project Context (Run 0001)

## URGENT: New Intelligence From Competition Analysis (Apr 13)

### Competition Landscape (1,443 PRs analyzed)
- **Official SOTA: 1.0810 BPB** (PR #1493, bigbag, Apr 9)
- **GDN-Hybrid at 1.0167 BPB is INVALID** — BPB metric bug (double-counts space bytes). Real score ~1.17.
- **Casefold tokenizer**: 1.0639 BPB (legit, PR #1585) — SP8192 retrained on NFKC+lowercased text, 10.4% compression gain
- **VarLen attention + doc-TTT**: 1.0734 BPB (PR #1530)
- **Per-layer adaptive GPTQ + int7 embeds**: 1.0749 BPB (PR #1586)
- **Deadline: April 30, 2026** (17 days remaining)

### Scaling Laws (from meta-analysis of 975 runs)
- **Step 1000 BPB predicts final BPB with r=0.86** — the critical signal
- Smoke-level train_loss at step 200 has VERY LIMITED discriminative power — all values cluster 2.53-2.54
- **Seed variance: only 0.5 mBPB** — architecture choices are 10-100x more impactful
- **25-30M param sweet spot** for int6 in 16MB
- **Second half of training has near-zero predictive value** for ranking
- **Which techniques matters far more than how many** (r=-0.09 for technique count)

### STOP doing tiny HP perturbations at smoke level
Smoke tests can only detect BROKEN experiments (loss > 2.60). They CANNOT distinguish between good and great configs — the signal is in the noise floor. Instead:

### HIGH-IMPACT changes to try (ordered by expected BPB gain)
1. **WARMDOWN_SHAPE=sqrt** — 1-sqrt cooldown schedule (env var, already implemented). Faster initial LR decay, slower tail. Zero risk.
2. **GPTQ_ADAPTIVE_CLIP=1** — Per-layer GPTQ clip sigmas (env var, already implemented). Tighter clipping for sensitive MLP_down layers 9-10 based on model autopsy data.
3. **EMBED_BITS=7** — Int7 embeddings (from int8). Frees ~62KB for better matrix quantization.
4. **BATCH_WARMUP_FRAC=0.3** — Ramp batch from 262K to 786K over first 30% of training. More gradient steps early.
5. **Casefold tokenizer** — Retrain SP8192 on NFKC+lowercased FineWeb. Biggest architectural opportunity.
6. **Stack multiple**: WARMDOWN_SHAPE=sqrt + GPTQ_ADAPTIVE_CLIP=1 + EMBED_BITS=7 (all orthogonal)

### Use the funnel properly
- Smoke ($0.25): crash/sanity gate ONLY
- **Qualify ($2): THE decision point** — step-1000 BPB predicts final with r=0.86
- Full ($5-8): only for configs that pass qualify
- `infra/scaling_laws.py` is now available — provides regression-based promote/kill decisions

## Target
`winning_base_decoded.py` — the SP8192 competition leader's training script (1.0810 BPB).

## Architecture (Current Best)
- 11L x 512d x 8H / 4KV, MLP 4x, LeakyReLU(0.5)^2
- **Depth recurrence**: Layers 3-5 loop 2x (17 virtual layers from 11 physical, activates at 35% training)
- **Parallel residuals**: From layer 7+, GPT-J style (attn+MLP read same input)
- **Skip gates**: Sigmoid-gated U-Net connections between encoder/decoder phases
- **Partial RoPE**: 16/64 dims, layerwise LN scale
- **MuonEq-R**: Row-normalized Muon optimizer, Newton-Schulz 5 steps
- **SDClip GPTQ**: int6 matrices (k=12.85), int8 embeddings (k=20.0), Brotli-11 compression

## New Features Available (env var controlled)
| Feature | Env Var | Description |
|---------|---------|-------------|
| Sqrt cooldown | WARMDOWN_SHAPE=sqrt | 1-sqrt(progress) LR decay curve |
| Cosine cooldown | WARMDOWN_SHAPE=cosine | Cosine LR decay curve |
| Adaptive GPTQ | GPTQ_ADAPTIVE_CLIP=1 | Per-layer clip sigmas (tighter for MLP_down) |
| Batch warmup | BATCH_WARMUP_FRAC=0.3 | Ramp batch size over first 30% |

## Benchmark (Funnel Pipeline)
- **Stage 1 — Smoke (1xH100, 5min, ~$0.25)**: crash/sanity gate only
- **Stage 2 — Qualify (8xH100, 3min, ~$2)**: Get step-1000 BPB (r=0.86 with final)
- Total per-iteration cost: ~$2.25 (vs $7 for full runs)

## Key Technique Effects (from meta-analysis)
- BigramHash: -46 mBPB (largest single effect)
- EMA: -30 mBPB
- XSA: -25 mBPB
- Int6 vs Int8: -46 mBPP

## Gates
- `syntax_check`: Python syntax validation
