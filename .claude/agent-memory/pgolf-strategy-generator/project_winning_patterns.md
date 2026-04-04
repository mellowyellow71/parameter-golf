---
name: Winning Model Patterns from Autopsy & Meta-Analysis
description: Key findings from pr-1105-model-autopsy and pgolf-meta about what makes winning models succeed
type: project
---

## From PR-1105 Autopsy
- MLP dominates quantization sensitivity (2.2x vs attention combined)
- MLP_down captures 70% of total int5->int6 upgrade benefit
- Layer 9 is most quantization-sensitive (+403e-6)
- Stable rank predicts quantization damage better than condition number
- MLP uses 95.2% of rank capacity; attention Q only 73.3%
- Mixed int5/int6 causes calibration degradation (ECE 0.24%->1.26%)
- Temperature scaling could recover calibration loss
- Layer 7 does most of the work (-3.82 bits/token logit lens)
- 28 previous-token heads, only 1 induction head — model relies on n-gram stats

## From pgolf-meta (975 runs)
- Training outcomes visible at step 1000 (r=0.86 correlation)
- Seed variance is 0.5 mBPB — architecture choice dominates
- Sweet spot: 25-30M parameters
- Median submission stacks 7 techniques; which matters more than how many
- BigramHash largest single-technique association: -0.046 BPB
- Technique stacking flatlines at 12-14 techniques

## Confirmed Winning Stack
- int6 + MLP3x + sliding window eval + FP16 embed + zstd-22 (core five)
- SmearGate + BigramHash + OrthoInit + WD 0.04
- XSA-all + EMA(0.997) + LeakyReLU(0.5)^2
- Full GPTQ + Parallel Muon + Coprime-stride loader

**Why:** Understanding what works guides strategy generation.
**How to apply:** New strategies should build on confirmed patterns, not contradict them without reason.
