# Mining 975 Expensive Training Runs

**Source:** [abay.tech/posts/pgolf-meta](https://abay.tech/posts/pgolf-meta)
**Author:** Abay Bektursun
**Context:** [Parameter Golf](https://github.com/openai/parameter-golf) · 8×H100 Track · March 2026

---

A single 10-minute run on 8×H100 SXM GPUs costs roughly $40. The Parameter Golf competition produced 975 of them—all public: training logs, source code, checkpoints. That's a dataset most of us could never afford to generate.

Can you predict how a model will perform on 8×H100s without actually running 8×H100s? From early checkpoints, from the source code, from a quick run on a cheaper GPU. This article opens up the dataset: what's in it, how much is real, and why the answer might be yes.

| Metric | Value |
|---|---|
| Submissions | 409 |
| Authors | 218 |
| Valid | 264 |
| Invalid | 132 |

---

## The Dataset

409 submissions to the 8×H100 track — training logs, source code, and metadata.

Each of the 409 submissions comes with training logs, source code, and metadata. The metric is validation bits-per-byte (BPB): the average number of bits to encode each byte of held-out FineWeb data. Lower is better. Not all have complete metadata: 63 lack dates, 45 lack author information, and 43 have no parseable training logs.

### Classifying 409 submissions

About 70% of submissions fork a shared template that ships dozens of gated features (TTT, N-gram caching, SWA, quantization), all toggled by environment variables. The code is there, but most features are disabled at runtime. You can't classify a submission by reading its code; you have to check what *actually ran.*

We took inspiration from natural science. First, a Claude agent examined PRs like specimens, iteratively building a taxonomy: 8 strategies, 49 techniques, 5 violation types. Categories emerged from observation, not a predefined schema.

With the taxonomy in hand, a simple script reads each submission's source code and training logs, calls an LLM once per PR with structured output, and produces tags: strategy, active techniques, validity. Final split: 264 valid, 132 invalid, 13 unauditable.

Valid submissions form a tight band between 1.02 and 1.25 BPB. Below that: invalid submissions plunging to near-zero BPB, almost entirely N-gram cache and illegal TTT.

### Arrival patterns

Submissions arrived in waves. The first few days were exploratory. Then a shared template appeared, and volume exploded. By day 7, N-gram caching arrived and the leaderboard went haywire.

Eight distinct strategies emerged. Nearly half of all submissions were "technique stackers"—teams that forked the shared template and piled on techniques.

---

## One Template, 975 Experiments

PR #64 became the foundation for the vast majority of submissions.

PR #64, nicknamed "Domination," became the foundation for the vast majority of submissions. It shipped with a well-tuned 11-layer transformer and a suite of gated techniques: sliding window evaluation, Muon optimizer, int6 quantization, MLP 3×, SmearGate, and more. Each could be toggled via environment variables.

Nearly everyone started from the same codebase. That accident turns 975 runs into a controlled experiment no individual team could have designed. Several non-template techniques became just as universal through independent adoption: LeakyReLU² (93%), logit softcap (88%), U-Net skip connections (74%).

The median submission added 7 non-template techniques on top of the base. A handful added zero (pure template tuners). A few outliers stacked 15+.

---

## What Moved the Needle

Filtering to 198 PRs with valid BPB and standard 8×H100 environments.

We compare median BPB for submissions *with* versus *without* each technique. BigramHash shows the largest single-technique association: −0.046 BPB. EMA and XSA follow. Int8 quantization is a negative signal; the frontier had moved to int6 with QAT, and teams still on int8 were behind the curve. These are associations, not causal effects.

Stacking more techniques helps, but only to a point. Median BPB drops from 1.179 (3–5 techniques) to 1.138 (12–14), then flatlines. The correlation is weak (r = −0.09) because variance at each level is huge. Which techniques you pick matters more than how many.

### The 16MB squeeze

In normal ML, bigger is better. Here, the 16MB artifact limit creates a compression frontier. The sweet spot turned out to be 25–30M parameters. 71% of the 731 environment-controlled runs landed there.

int6 dominates, appearing in 61% of valid submissions. int5 and GPTQ variants push compression further, trading precision for parameter count.

---

## Separating Signal from Noise

93% of the 97 N-gram submissions broke the rules.

Around PR #659, teams discovered they could precompute N-gram statistics from the training data and blend them with neural logits at evaluation time. Claimed BPB plummeted from ~1.1 to below 0.03. But 93% of the 97 N-gram submissions broke the rules.

The "cache lift" tells the story. For N-gram submissions, the gap between the neural model's honest BPB and the claimed submission BPB is enormous—the cache is doing all the work.

The neural models *underneath* the N-gram submissions are actually good. Their backbone BPB distribution (median 1.14) overlaps with non-N-gram submissions (median 1.16). These were talented engineers who chose to game the metric.

It got worse over time. In the first quartile of submissions, 4% were invalid. By the fourth quartile, 52%—driven by the N-gram explosion.

Five violation types, often co-occurring. Unnormalized distributions dominate (90 PRs).

For illegal TTT submissions, each point plots a submission's honest neural BPB against its claimed BPB. Points below the diagonal got an illegal boost.

The full BPB distribution makes the contamination obvious. Valid submissions cluster between 1.0 and 1.3. Invalid ones spread across the entire range, down to 0.0000 BPB.

---

## The Prediction Signal

Is training predictable?

Each line is one training curve from 279 environment-controlled runs, colored by outcome: green for the top 20% of final BPB, red for the bottom 20%.

Green separates from red in the first quarter. The first-quarter loss drop correlates with final BPB at r = 0.64. The second half of training? Near zero.

Techniques change the shape of the training curve itself. The chart shows the gap in median training loss between runs *with* vs *without* each technique, at every point during training. Below zero means the technique helps.

BigramHash helps uniformly. VRL and legal TTT show almost no gap early but widen steadily. Int8 starts in a hole and never climbs out.

### Early signal, final outcome

Of the 731 environment-controlled runs, 330 logged a validation checkpoint near step 1,000—roughly 90 seconds into a 10-minute run.

**r = 0.86.** After 15% of training, the outcome is largely determined. Pure neural models track the trend line. TTT-augmented models scatter below it.

The random seed barely matters. Across 114 multi-seed configurations, the median standard deviation is 0.5 mBPB. Architecture choice matters orders of magnitude more than random initialization.

Architecture is what matters, and you can see it 90 seconds in.

---

## What Comes Next

Training outcomes are visible at step 1,000.

After filtering the invalid submissions and building a labeled database from what's left, one thing is clear: **training outcomes are visible at step 1,000.**

That's 90 seconds into a 10-minute run. The correlation is 0.86. Seed variance is 0.5 mBPB. Architecture choice determines the result, and you can see it almost immediately.

In **Part 2**, we build on this: a model that predicts final BPB from early checkpoints—the 1,000-step oracle. Can you know the outcome before the run finishes?

In **Part 3**, we go further. If early H100 steps predict final H100 outcomes, can a short run on an A6000 or RTX 5090 predict what 8×H100s would produce? If so, you train cheap, predict expensive, and only burn H100 time when it counts.

---

*Data: 409 PRs from the 8×H100 SXM track, 975 training logs.*
*Analysis by [Abay Bektursun](https://github.com/abaybektursun)*
