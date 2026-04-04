# Autopsy of a SOTA Parameter-Golf GPT, Round 2

**Source:** [abay.tech/posts/pr-1105-model-autopsy](https://abay.tech/posts/pr-1105-model-autopsy)
**Author:** Abay Bektursun
**Context:** [PR #1105](https://github.com/openai/parameter-golf/pull/1105) · March 2026

---

The [PR #1019 autopsy](https://abay.tech/posts/pr-1019-model-autopsy) said MLP was 4× more sensitive to quantization than attention. So we expanded MLP from 3× to 3.5× and switched to mixed int5/int6. Result: 1.1086 BPB, −0.005 over the previous SOTA. The tradeoff: calibration ECE jumped from 0.24% to 1.26% — mixed quantization introduced systematic overconfidence across 99.6% of tokens.

| Metric | Value |
|---|---|
| Sliding BPB | 1.1086 |
| Parameters | 29.95 M |
| vs PR #1019 | −0.005 BPB |
| Artifact | 14.52 MB |
| Hardware | 8×H100 SXM |

---

## MLP Still Dominates — But the Gap Narrowed

Per-matrix int6 sensitivity, same methodology as PR #1019.

Same test: quantize each matrix individually to int6, keep everything else at full precision. MLP_down accounts for 1,302 × 10⁻⁶ total sensitivity. All four attention matrix types combined: 599 × 10⁻⁶. The ratio dropped from 4× to 2.2× — the wider MLP spreads quantization damage across more parameters, making each individual weight less critical.

Each cell: quantize only that one matrix to int6, measure BPB degradation. Baseline: 1.150966 BPB. Full int6: +2.3×10⁻³.

All 6 matrices of each layer quantized to int6. Decoder layers 9–10 are most sensitive.

**Key Insight:** Layer 9 is now the most sensitive (+403 × 10⁻⁶), overtaking Layer 10 from PR #1019. Full-model int6 penalty: +0.0023 BPB, down from +0.0083 in PR #1019. The wider MLP and mixed quantization strategy are working — less total damage per bit removed.

---

## Mixed Precision: Where the Extra Bit Matters

Per-matrix int5 vs int6 — the knapsack that makes 3.5× MLP fit in 16 MB.

Uniform int6 can't fit 29.95M parameters under 16 MB. The fix: default to int5 and selectively promote matrices where the extra bit pays for itself. Every single top-10 upgrade benefit is an MLP_down matrix. L9 MLP_down gains 1,167 × 10⁻⁶ BPB from one extra bit. L10 MLP_down: 1,066 × 10⁻⁶. The next non-MLP_down entry is L9 V at rank #11. The efficiency ranking (BPB gain per bit spent) feeds the knapsack that decides the final allocation.

**Key Insight:** Across all 66 matrices: MLP_down accounts for 6,058 × 10⁻⁶ of total upgrade benefit — 70% of the entire budget. MLP_up adds 1,541 × 10⁻⁶ (18%). All attention matrices combined: 1,065 × 10⁻⁶ (12%). 11 matrices show negative upgrade benefits (int5 actually outperforms int6) — concentrated in V and K matrices at layers 1–3 and 7–8. L3 V is the most negative at −118 × 10⁻⁶. At these magnitudes this is likely eval noise, but it means promoting those matrices would waste bits and hurt BPB.

---

## Condition Numbers: Still a Red Herring

Q up to 30,759, Out up to 25,556 — MLP still hurts more.

Q's condition number dropped from 54,000 to 30,759 with the wider MLP, but the story is the same: high condition number does not predict quantization sensitivity. MLP matrices have condition numbers of 4–13 and dominate the damage. Stable rank remains the real predictor — MLP utilizes 95% of its rank capacity vs 73% for Q.

**Key Insight:** MLP rank utilization: 95.2% (up from 94.4% in PR #1019). At 3.5× expansion the MLP is nearly fully packed. Further expansion would add parameters that actually get used. Q utilization at 73.3% means attention still has headroom — it's not parameter-starved.

---

## Stable Rank Explains the Gap

Why low condition numbers hurt more than high ones.

The singular value curves show the mechanism: Q concentrates energy in ~12% of its channels. The other 88% carry near-zero signal, so quantization errors there are harmless. MLP uses ~35% of its channels — 3× more active capacity, 3× more ways to accumulate rounding damage. Same pattern as PR #1019, now with wider matrices.

**Key Insight:** The 3.5× MLP increased MLP stable rank from ~160 to ~184 on average. More channels are active, but the utilization ratio (stable rank / full rank) held steady at ~95%. The expansion added capacity that the model actually uses — not dead parameters.

---

## Layer 7 Still Does Most of the Work

Projecting each layer's residual stream through the unembedding matrix.

Same logit lens methodology. Layer 7 contributes −3.82 bits/token (vs −4.35 in PR #1019). Layers 3–5 still show increased loss from skip connection reorganization. The pattern is structurally identical to PR #1019 despite the wider MLP and different quantization.

**Key Insight:** Layer 7: −3.82 bits/token. Layer 10: −0.52 bits/token, still the weakest contributor. The encoder layers 3–4 readability cost is larger here (+1.22, +0.99, +2.24 for L3–L5) vs PR #1019, suggesting the wider MLP is pushing more representational work through the skip connections.

---

## Calibration Degraded: The Cost of Mixed Quantization

ECE = 1.26% — up from 0.24% in PR #1019.

ECE jumped 5× from 0.24% to 1.26%. 99.6% of tokens land in overconfident bins — the model consistently believes it's more right than it is. This is a direct consequence of mixed int5/int6: the coarser int5 matrices introduce systematic confidence bias that uniform int6 did not.

**Key Insight:** The BPB gain from mixed quantization (-0.005) is worth the calibration cost — but this model would benefit from temperature scaling, unlike PR #1019. 70% of total loss still comes from tokens where P(correct) < 5%. The bottleneck is still accuracy, but confidence is now slightly misaligned.

---

## What Each Head Learns

Classifying 88 attention heads by function.

28 previous-token heads (vs 22 in PR #1019), 1 induction head (down from 2), and 2 positional heads (new). The wider MLP didn't change the fundamental attention patterns — this model still relies on n-gram statistics, not in-context copying.

- 1 induction head
- 28 previous-token heads
- 2 positional heads
- 57 other heads

**Key Insight:** The single induction head (L2H3, score 0.020) is marginal. 28 previous-token heads split evenly between encoder (15) and decoder (13). The emergence of 2 positional heads in the decoder (L6H2, L6H3) is new — these attend to absolute position, possibly compensating for the larger MLP's representational demands.

---

## Reading the Model's Mind

Token-level loss, top-k predictions at failure points, and generation vs. reality.

### Loss Heatmap

Light = predicted well, dark = surprised. The model's highest-loss tokens (most surprising) tend to be proper nouns, rare collocations, and structural transitions between documents.

### Top-k Predictions at High-Loss Tokens

186 tokens above the 90th-percentile loss threshold were analyzed. Key failure patterns:

- **Proper nouns and rare names:** e.g., "Newseum" (loss 11.765 nats), "Johannesen" — the model has no way to predict specific names
- **Unexpected continuations:** e.g., "included" after "Sam" (loss 11.236) — model predicted "and" (61.5%) but the actual text used an unusual parenthetical
- **Rare subword splits:** e.g., "roomates" being tokenized unexpectedly (loss 9.739)
- **Document boundaries:** Transitions between `<s>` delimited sequences produce high loss as the model has no context for the new topic

The model's failures are overwhelmingly on tokens that require memorization of specific facts rather than language modeling ability — proper nouns, specific numbers, exact word choices in quoted speech.

### Side-by-Side Generation

50-token prompt, 200-token continuation (temp=0.8, seed=42):

- **Position matches:** 6/200 (3.0%)
- **Unique token overlap:** 40 (33% of real)

The model captures the topic (insurance) but generates generic insurance content rather than reproducing the specific narrative. This is expected — the model has ~30M parameters and has never seen this text.

---

## What Changed from PR #1019

Expanding MLP was the right call — stable rank confirmed it was parameter-starved. Mixed quantization bought us the bit budget to fit the larger model under 16 MB. The calibration hit is real but recoverable with temperature scaling.

1. **MLP 3× → 3.5× worked.** SVD showed 94.4% rank utilization at 3× — nearly packed. At 3.5× it's 95.2% and still climbing. The extra 2.88M parameters are active.

2. **Mixed int5/int6 fits the budget.** The per-matrix analysis shows MLP_down captures 70% of total upgrade benefit. Promote MLP_down first, then MLP_up, then attention if bits remain.

3. **Calibration is now a concern.** ECE: 0.24% → 1.26%. Temperature scaling was unnecessary before. It's worth investigating now.

4. **Layer 9 overtook Layer 10** as the most quantization-sensitive layer (+403 vs +229 × 10⁻⁶). If narrowing a layer, L10 is still the candidate — low logit lens contribution (−0.52 bits/token) with high sensitivity.

---

**Model spec:** 11L GPT, 512d, 8H/4KV GQA, XSA-all, BigramHash 3072×112, MLP 3.5×, LeakyReLU(0.5)²

**Quantization:** Mixed int5/int6 GPTQ, Brotli-11, Parallel Muon NS5, 8×H100 SXM, seed 314
