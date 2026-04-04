---
name: Research Plan Generation Status
description: Tracks the comprehensive strategy research plan generated for Parameter Golf competition
type: project
---

## Research Plan File
- Location: `/home/bryandong24/parameter-golf/pgolf_research_plan.md`
- Generated: 2026-04-04
- Total strategies: 87 (28 Tier 1, 34 Tier 2, 25 Tier 3)

## Key Strategic Priorities Identified
1. **Scylla + SLOT combination** (T1-12): The "obvious untried combination" per analyzer -- estimated 0.92-0.94 BPB
2. **Systems-first** (T1-06, T1-08, T1-09, T1-11): Zero-risk throughput gains, significance test waived
3. **QK-Gain 4.0-5.0** (T1-04): Zero-cost architecture improvement, confirmed in #1176
4. **Mixed int5/int6 per autopsy** (T1-05): Knapsack allocation prioritizing MLP_down (70% of benefit)
5. **Per-Sample SLOT** (T1-03): Used in best pending submission #1229 (0.9300 BPB)

## Confirmed Dead Ends (Do NOT Retry)
- MoE at this scale (definitively unviable below 500M)
- INT4 quantization (catastrophic: +0.065 BPB)
- 2:4 structured sparsity during training (#1105: +0.672 BPB)
- Turbo-Muon on 8xH100 (worse AND over 16MB)
- Knowledge distillation (per-step overhead fatal at 600s)
- MC Dropout ensembling (sub-networks lack diversity)
- Product quantization (+292% BPP)

**Why:** Avoids wasting compute on approaches already proven ineffective.
**How to apply:** Check this list before implementing any strategy to avoid repeating known failures.
