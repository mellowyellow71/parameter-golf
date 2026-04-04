---
name: Parameter Golf Competition Rules & Legality
description: Core rules, parameter counting, legality boundaries, and enforcement history for the OpenAI Parameter Golf challenge
type: project
---

## Key Rules
- Artifact limit: 16,000,000 bytes (code + compressed model, decimal not MiB)
- Training: <=10 min on 8xH100 SXM (600s)
- Evaluation: <=10 min (separate budget)
- No network calls, no external downloads during eval
- Metric: BPB (bits per byte) on FineWeb validation set, tokenizer-agnostic
- New SOTA must beat current by >=0.005 nats at p<0.01 (3 seeds typical)
- Systems-only changes: significance test waived

## Legality Boundaries (Critical)
- Val data CANNOT be in artifact (paid prefix banned)
- TTT: ONLY backward-looking (score-first) allowed — adapt on tokens already graded
- Pre-eval adaptation on val data: BANNED
- GPTQ calibration: MUST be within 600s training budget (NOT eval time)
- Self-generated calibration data: "probably legal" per valerio-oai
- N-gram eval cache: "directionally legal" if properly normalized over full vocab
- Hashed n-gram without full-vocab normalization: BANNED
- Two-pass rescoring (score->TTT->rescore): BANNED
- Val-calibrated GPTQ: BANNED ("breaks autoregressivity")

## Official SOTA
- 1.1147 BPB (#1019, @abaybektursun, merged Mar 25)
- Best pending: #1229 (0.9300), #1184 (0.9485)
- Best standard-tokenizer: #1176 (1.0914)

**Why:** These rules determine which strategies are implementable. Violations get PRs closed.
**How to apply:** Every strategy must be cross-referenced against these rules before inclusion.
