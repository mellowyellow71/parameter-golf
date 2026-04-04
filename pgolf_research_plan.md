# Parameter Golf Comprehensive Research Plan
# Generated 2026-04-04 | Target: Sub-1.10 BPB | Budget: 8xH100 unlimited runs

---

## Executive Summary

**Total strategies: 87**
- Tier 1 (High Priority): 28 strategies -- run first, highest expected competitive impact
- Tier 2 (Medium Priority): 34 strategies -- solid ideas for parallel exploration
- Tier 3 (Exploratory): 25 strategies -- novel/speculative, high risk high reward

**Category breakdown:**
- Architecture: 22
- Quantization/Compression: 15
- Optimizer/Training: 16
- Eval-Time Methods: 14
- Systems/Throughput: 10
- Tokenizer: 5
- Hybrid: 5

**Current baselines:**
- Official SOTA: 1.1147 BPB (#1019, @abaybektursun)
- Best pending: 0.9300 BPB (#1229, @resouer, SLOT + GPTQ)
- Best standard tokenizer: 1.0914 BPB (#1176, QK-Gain + TTT + SLOT)
- Our experiment1.py: ~1.12-1.13 BPB estimated (11L, 512d, int6, Muon, SmearGate, BigramHash, XSA-4, EMA, GPTQ)

**Recommended execution order:**
1. Systems optimizations (significance waived, parallel with everything)
2. Scylla tokenizer + current best stack
3. SLOT variants on our base
4. Mixed quantization informed by autopsy
5. Architecture innovations (Hourglass FFN, DDL, Hyper-Connections)
6. N-gram cache with proper normalization
7. Novel optimizer variants
8. Speculative approaches

---

## Legality Reference

Before each strategy, legality is assessed against these confirmed rules:
- **LEGAL**: Explicitly allowed or used in merged leaderboard entries
- **LIKELY_LEGAL**: Consistent with rules, no organizer objection expected
- **GRAY_AREA**: Requires careful implementation to avoid violation; explanation provided

Key boundaries:
- GPTQ calibration MUST be within 600s training budget
- TTT: ONLY backward-looking (score-first, single-pass) allowed
- N-gram caches: MUST produce full-vocabulary normalized distributions
- Val data cannot be stored in artifact
- Self-generated calibration data is "probably legal"
- No two-pass rescoring (score -> adapt -> rescore same tokens)

---

## TIER 1: HIGH PRIORITY (28 strategies)

---

### Strategy T1-01: Scylla Tokenizer + Current Best Stack
- **Category**: tokenizer/hybrid
- **Legality**: LIKELY_LEGAL -- #1184 used Scylla successfully at 0.9485; tokenizer changes face extra scrutiny per README but are explicitly allowed
- **Parameter Budget Impact**: Reduces vocab from 1024 to 998, saving ~13K params in embedding; overall neutral as freed space enables wider MLP
- **Core Idea**: Replace the sp1024 BPE tokenizer with the Scylla 998-token TokenMonster-derived tokenizer from #1143. This was the single biggest technique-level BPB improvement in the competition (0.028 BPB over the next best). TokenMonster's ungreedy multi-branch search produces ~37.5% fewer tokens than BPE at equivalent vocab size.
- **Key Components**:
  - Component 1: Obtain Scylla tokenizer from #1143/#1184 (998-vocab TokenMonster derivative)
  - Component 2: Build explicit per-token byte metadata LUTs (not SentencePiece runtime) for BPB calculation
  - Component 3: Retrain full stack (11L, 512d, XSA-all, EMA, GPTQ, LeakyReLU^2, SmearGate, BigramHash) with Scylla
  - Component 4: Adjust BigramHash bucket count for 998-vocab (998^2 possible bigrams)
- **Implementation Plan**:
  1. Download Scylla tokenizer from PR #1143 records folder
  2. Replace `build_sentencepiece_luts()` with static LUT from tokenizer metadata
  3. Set `VOCAB_SIZE=998`, retrain on FineWeb with `data/cached_challenge_fineweb.py --variant scylla`
  4. If Scylla data variant not cached, tokenize FineWeb with TokenMonster library
  5. Train with identical hyperparameters: 11L, 512d, MLP_MULT=3.0, MUON_WD=0.04, EMA=0.997
  6. Apply Full GPTQ within training budget (self-gen calibration, 32 batches)
  7. Compress with zstd-22, validate BPB calculation carefully
- **Expected Outcome**: 1.05-1.08 BPB (based on #1184 achieving 0.9485 with Scylla + GPTQ + XSA-all)
- **Compute Estimate**: ~12 min per run (10 min train + 2 min eval), 3 seeds = 36 min
- **Priority**: HIGH
- **Combines With**: T1-02, T1-03, T1-04, T1-05, T1-06, T1-07, T1-10, T1-14
- **Source/Inspiration**: #1143 Scylla tokenizer (0.028 BPB single-technique gain), #1184 (0.9485)

---

### Strategy T1-02: SLOT (Selective Logit Offset Tuning) on Current Base
- **Category**: eval-time
- **Legality**: LEGAL -- used in 9 record submissions (#1172, #1176, #1217, #1229); Context-Only variant proven causal
- **Parameter Budget Impact**: Zero artifact cost -- SLOT operates entirely at eval time
- **Core Idea**: Add a learnable 512-dim delta vector at the last hidden layer, optimized per-batch during evaluation with AdamW. This is the most impactful eval-time method that has survived enforcement. #1172 showed SLOT alone outperforms TTT+SLOT.
- **Key Components**:
  - Component 1: Per-batch delta vector [1, 1, 512] initialized to zero
  - Component 2: AdamW optimizer (lr=0.005-0.008, betas=(0.9,0.999))
  - Component 3: 8-16 optimization steps per eval batch
  - Component 4: Cosine LR schedule within SLOT steps (0.008 -> 0.0008)
- **Implementation Plan**:
  1. After model training and GPTQ, during `eval_val_sliding()`:
  2. For each eval batch of size [B, seq_len]:
     a. Initialize delta = torch.zeros(1, 1, model_dim, requires_grad=True)
     b. Create AdamW optimizer for delta only: lr=0.008, weight_decay=0.0
     c. Create cosine LR scheduler over 16 steps
     d. For step in range(16):
        - Compute hidden states up to last layer
        - Add delta to hidden states: h = h + delta
        - Compute logits and loss on already-scored positions only
        - Backprop through delta only
        - optimizer.step()
     e. Apply final delta to compute scored logits
  3. Key: only use tokens at positions already scored (backward-looking)
  4. Sweep lr in {0.002, 0.005, 0.008, 0.01}, steps in {4, 8, 16, 32}
- **Expected Outcome**: -0.015 to -0.025 BPB improvement over sliding-window baseline
- **Compute Estimate**: Adds ~60-120s to eval time (well within 10 min eval budget)
- **Priority**: HIGH
- **Combines With**: T1-01, T1-03, T1-04, T1-05, T1-06, T1-10
- **Source/Inspiration**: #1084 (original), #1172 (best pure SLOT at 1.1015), #1229 (Scored-Position SLOT at 0.9300)

---

### Strategy T1-03: Per-Sample SLOT Delta
- **Category**: eval-time
- **Legality**: LEGAL -- used in #1229 (0.9300 BPB, record-eligible)
- **Parameter Budget Impact**: Zero artifact cost
- **Core Idea**: Instead of a shared delta [1,1,512] across the batch, optimize a separate delta per sample [bsz,1,512]. Each document in the batch gets its own adaptation vector. Combined with Scored-Position masking.
- **Key Components**:
  - Component 1: Per-sample delta tensor [batch_size, 1, model_dim]
  - Component 2: Scored-Position mask -- only positions already scored contribute to SLOT loss
  - Component 3: Optional logit bias [batch_size, 1, vocab_size] for per-sample output correction
  - Component 4: Cosine LR schedule 0.008 -> 0.0008 over 16 AdamW steps
- **Implementation Plan**:
  1. Modify SLOT to create delta of shape [B, 1, 512] instead of [1, 1, 512]
  2. Add position mask: `scored_mask[i, j] = 1 if position j has been scored in a previous window`
  3. SLOT loss computed only on masked (already-scored) positions
  4. Add learnable logit_bias [B, 1, V] initialized to zero, also optimized
  5. Total eval-time params per batch: B * (512 + V) ~= B * 1536
  6. 16 AdamW steps with cosine LR
- **Expected Outcome**: -0.020 to -0.030 BPB over baseline sliding eval
- **Compute Estimate**: ~90-150s eval overhead
- **Priority**: HIGH
- **Combines With**: T1-01, T1-02, T1-04, T1-06
- **Source/Inspiration**: #1229 (@resouer, 0.9300 BPB)

---

### Strategy T1-04: QK-Gain Scaling (gain=4.0-5.0)
- **Category**: architecture
- **Legality**: LEGAL -- used in #1176 (1.0914), #1217 (1.1027)
- **Parameter Budget Impact**: Zero -- 1 scalar per head, ~8 total params
- **Core Idea**: Add a learnable per-head scalar multiplier on queries after QK-norm. Higher gain = sharper attention = more decisive routing. Found via 45-experiment sweep in #1125.
- **Key Components**:
  - Component 1: Per-head learnable scalar `q_gain` initialized to target value
  - Component 2: Applied after Q projection and before attention score computation
  - Component 3: Included in scalar_lr parameter group for AdamW
- **Implementation Plan**:
  1. In Attention.__init__: `self.q_gain = nn.Parameter(torch.full((num_heads,), gain_init))`
  2. In Attention.forward, after computing Q: `q = q * self.q_gain.view(1, 1, num_heads, 1)`
  3. Set `QK_GAIN_INIT=4.0` as default (sweep 2.0, 3.0, 4.0, 5.0, 6.0)
  4. Add q_gain to scalar parameter group
  5. Train 3 seeds at each gain value
- **Expected Outcome**: -0.003 to -0.006 BPB
- **Compute Estimate**: 5 gain values * 3 seeds * 12 min = 3 hours
- **Priority**: HIGH
- **Combines With**: All strategies (orthogonal, zero-cost)
- **Source/Inspiration**: #1125 (45-experiment sweep), #1176 (record at 1.0914)

---

### Strategy T1-05: MLP 3.5x + Mixed Int5/Int6 Quantization (Autopsy-Informed)
- **Category**: architecture/quantization
- **Legality**: LEGAL -- #1105 used this exact approach (1.1086 BPB)
- **Parameter Budget Impact**: +2.88M params (from 3x to 3.5x MLP), offset by mixed int5/int6
- **Core Idea**: Widen MLP from 3x to 3.5x and use the autopsy-informed bit allocation: promote MLP_down matrices to int6 first (70% of upgrade benefit), then MLP_up (18%), then attention only if bits remain.
- **Key Components**:
  - Component 1: MLP expansion from 1536 to 1792 hidden dim
  - Component 2: Default int5 quantization for all matrices
  - Component 3: Selective int6 promotion using knapsack optimization on sensitivity data
  - Component 4: Priority: L9 MLP_down (1167e-6), L10 MLP_down (1066e-6), then remaining MLP_down by sensitivity
- **Implementation Plan**:
  1. Set `MLP_MULT=3.5` in Hyperparameters
  2. After training, during GPTQ quantization:
     a. Start with all matrices at int5 (5 bits)
     b. Compute artifact size at full int5
     c. Greedily promote matrices to int6 in order of BPB-gain-per-bit-spent:
        - MLP_down layers 9, 10, 8, 7, 6, 5 (sorted by sensitivity)
        - MLP_up layers 9, 10
        - Attention V/O only if budget remains
     d. Stop when artifact would exceed 16MB
  3. Use GPTQ with self-generated calibration data (32 batches)
  4. Compress with Brotli-11 + byte-shuffle (saves ~580KB over zstd-22)
  5. Validate total artifact <= 16,000,000 bytes
- **Expected Outcome**: -0.003 to -0.008 BPB over uniform int6 with 3x MLP
- **Compute Estimate**: ~15 min per run (extra GPTQ time for sensitivity analysis)
- **Priority**: HIGH
- **Combines With**: T1-01, T1-02, T1-04, T1-06, T1-07, T1-10
- **Source/Inspiration**: pr-1105-model-autopsy (MLP_down = 70% of upgrade benefit)

---

### Strategy T1-06: Brotli-11 + Byte-Shuffle Compression
- **Category**: compression
- **Legality**: LEGAL -- used in #1089 (1.1086), #1105
- **Parameter Budget Impact**: Saves ~580KB vs zstd-22, enabling ~93K extra int5 params
- **Core Idea**: Replace zstd-22 with Brotli quality=11 plus a byte-shuffle pre-filter that groups MSB/LSB bytes for better compression. This is a free lunch -- better compression of the same weights.
- **Key Components**:
  - Component 1: Byte-shuffle pre-filter: rearrange quantized weight bytes so all MSBs are contiguous, then LSBs
  - Component 2: Brotli compression at quality=11
  - Component 3: Corresponding decompression at eval time
- **Implementation Plan**:
  1. After quantization, before compression:
     ```python
     import brotli
     def byte_shuffle(data: bytes) -> bytes:
         arr = np.frombuffer(data, dtype=np.uint8)
         # Group by byte position within each N-byte element
         n_bytes = 2  # for int16 scale factors; 1 for int8 weights
         shuffled = arr.reshape(-1, n_bytes).T.ravel()
         return shuffled.tobytes()
     
     compressed = brotli.compress(byte_shuffle(weight_bytes), quality=11)
     ```
  2. At eval time: `weight_bytes = byte_unshuffle(brotli.decompress(compressed))`
  3. Add `brotli` to requirements.txt
  4. Measure: compare artifact sizes with zstd-22 vs brotli-11+shuffle
- **Expected Outcome**: 0 to -0.005 BPB (via larger model fitting in 16MB)
- **Compute Estimate**: Minutes (compression only, no retraining)
- **Priority**: HIGH
- **Combines With**: All strategies (orthogonal compression improvement)
- **Source/Inspiration**: #1089 (580KB savings), #1105

---

### Strategy T1-07: XSA-All (11 Layers) with Full GPTQ
- **Category**: architecture
- **Legality**: LEGAL -- used in official SOTA #1019 and #609 (1.1154)
- **Parameter Budget Impact**: Zero params; ~3ms/step overhead
- **Core Idea**: Apply Exclusive Self-Attention to all 11 layers instead of just the last 4. At the frontier, XSA-all with Full GPTQ overcomes the overhead penalty. The key insight: removing self-value bias from ALL layers forces the model to rely entirely on inter-token information.
- **Key Components**:
  - Component 1: Set `XSA_LAST_N=11` (or XSA_LAST_N=0 meaning all)
  - Component 2: GQA-aware XSA implementation for efficiency
  - Component 3: Full Hessian GPTQ calibration within training budget
- **Implementation Plan**:
  1. Set `XSA_LAST_N=11` in experiment env
  2. Verify XSA implementation handles GQA (num_kv_heads=4, num_heads=8)
  3. Train with standard stack + XSA-all
  4. Compare 3-seed results: XSA-4 vs XSA-8 vs XSA-11
  5. Monitor step time: expect ~85ms/step vs ~82ms for XSA-4
- **Expected Outcome**: -0.003 to -0.006 BPB vs XSA-4
- **Compute Estimate**: 3 configs * 3 seeds * 12 min = 108 min
- **Priority**: HIGH
- **Combines With**: All strategies
- **Source/Inspiration**: #609 (1.1154, best non-TTT), #1019 (official SOTA uses XSA-all)

---

### Strategy T1-08: Coprime-Stride Multi-Shard Data Loader
- **Category**: systems
- **Legality**: LEGAL -- used in #726, #1060 (1.1123), #1184 (0.9485); significance waived
- **Parameter Budget Impact**: Zero
- **Core Idea**: Replace sequential shard reading with coprime-stride block sampling across shards. Each GPU rank samples blocks at coprime intervals, ensuring full permutation cycle before any repetition. Maximizes batch diversity with zero step-time overhead.
- **Key Components**:
  - Component 1: Coprime stride selection: choose stride coprime to total shard count
  - Component 2: Per-rank offset to avoid overlap
  - Component 3: Wraparound logic for full permutation cycle
- **Implementation Plan**:
  1. In data loading, replace sequential iteration with:
     ```python
     n_blocks = total_tokens // (seq_len * world_size)
     stride = find_coprime(n_blocks)  # largest prime < n_blocks
     for step in range(max_steps):
         block_idx = (step * stride + rank_offset) % n_blocks
         tokens = load_block(block_idx)
     ```
  2. Ensure stride is coprime to n_blocks (use next prime below n_blocks)
  3. No other changes needed
- **Expected Outcome**: -0.001 to -0.003 BPB (more diverse batches)
- **Compute Estimate**: Zero overhead -- drop-in replacement
- **Priority**: HIGH
- **Combines With**: All strategies (orthogonal)
- **Source/Inspiration**: #726 (@DeepReinforce), adopted by #1060, #1184

---

### Strategy T1-09: IFNSO (Iteration-Free Newton-Schulz)
- **Category**: systems/optimizer
- **Legality**: LEGAL -- systems-only, significance waived
- **Parameter Budget Impact**: Zero
- **Core Idea**: Replace Muon's 5 Newton-Schulz iterations with a single polynomial evaluation that achieves the same orthogonalization. This is a pure systems speedup -- same mathematical operation, fewer FLOPS. More steps in 600s.
- **Key Components**:
  - Component 1: Polynomial approximation of the matrix sign function
  - Component 2: Single matrix multiply replacing iterative loop
  - Component 3: Same output quality as 5-step NS
- **Implementation Plan**:
  1. Replace `zeropower_via_newtonschulz5` with:
     ```python
     def zeropower_ifnso(G, eps=1e-7):
         # Iteration-free NS via polynomial evaluation
         # arxiv:2602.02500
         was_2d = G.ndim == 2
         if was_2d: G = G.unsqueeze(0)
         X = G.bfloat16()
         transposed = X.size(-2) > X.size(-1)
         if transposed: X = X.mT
         X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
         # Degree-5 polynomial coefficients for sign function
         A = X @ X.mT
         # P(A) = c0*I + c1*A + c2*A^2 + c3*A^3
         c0, c1, c2, c3 = (15/8, -10/8, 3/8, 0)  # approximate coefficients
         I = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype).unsqueeze(0)
         A2 = A @ A
         P = c0 * I + c1 * A + c2 * A2
         X = P @ X
         if transposed: X = X.mT
         if was_2d: X = X.squeeze(0)
         return X
     ```
  2. Benchmark step time reduction (expect 2-5ms/step saved)
  3. Verify BPB is identical within noise (3-seed)
  4. Note: exact polynomial coefficients need tuning from the paper
- **Expected Outcome**: +100-300 extra training steps, -0.001 to -0.003 BPB
- **Compute Estimate**: 3 seeds * 12 min = 36 min
- **Priority**: HIGH
- **Combines With**: All strategies
- **Source/Inspiration**: arXiv:2602.02500, parametergolfanalyzer.md Tier 2

---

### Strategy T1-10: Legal Score-First TTT with GPTQ in Training Budget
- **Category**: eval-time
- **Legality**: LEGAL -- explicit ruling: backward-looking TTT on already-scored tokens is allowed. GPTQ within 600s training budget is required.
- **Parameter Budget Impact**: Zero artifact cost
- **Core Idea**: Adapt model weights during evaluation using AdamW on already-scored tokens. The key recipe from #461: SGD+momentum(0.9), 3 epochs per 32K chunk, freeze first 2 blocks. Critical: GPTQ calibration must happen within the 600s training budget, not at eval time.
- **Key Components**:
  - Component 1: Score-first evaluation loop: score tokens, THEN adapt on scored tokens
  - Component 2: AdamW optimizer with cosine LR schedule
  - Component 3: Per-layer LR groups (3x for MLP output projections per #481)
  - Component 4: Freeze first 2 blocks to avoid catastrophic adaptation
  - Component 5: 3 epochs per 32K token chunk
- **Implementation Plan**:
  1. During training (within 600s): run GPTQ calibration with self-generated data
  2. During eval_val_sliding_ttt():
     a. Process eval in chunks of 32768 tokens
     b. For each chunk:
        - Forward pass with stride=64 sliding window -> get loss on all positions
        - Record scored positions
        - For epoch in range(3):
          * Create TTT optimizer: AdamW(params_to_adapt, lr=base_lr)
          * Set per-layer LR: MLP_down 3x, MLP_up 2x, attention 1x
          * Cosine schedule from base_lr to base_lr/10 over 3 epochs
          * Forward pass on scored positions only
          * Backprop and step
        - Continue to next chunk
     c. base_lr sweep: {0.0005, 0.001, 0.002}
  3. Freeze blocks 0-1 (set requires_grad=False on first 2 transformer blocks)
  4. Use torch.no_grad() on frozen params to save memory
- **Expected Outcome**: -0.005 to -0.015 BPB
- **Compute Estimate**: ~5-8 min eval time per run (within 10 min budget)
- **Priority**: HIGH
- **Combines With**: T1-01, T1-04, T1-05, but NOT with T1-02/T1-03 (SLOT replaces TTT per #1172)
- **Source/Inspiration**: #461 (SGD+momentum recipe), #481 (cosine TTT 3x multiplier), #549 (legal TTT on leaderboard)

---

### Strategy T1-11: Liger-Kernel Fused Operations
- **Category**: systems
- **Legality**: LEGAL -- pip-installable library, systems-only, significance waived
- **Parameter Budget Impact**: Zero
- **Core Idea**: Use LinkedIn's open-source Liger-Kernel for fused Triton operations: RMSNorm (6x speedup), linear+CE (3x), residual+norm fusion. Eliminates kernel launch overhead.
- **Key Components**:
  - Component 1: `pip install liger-kernel`
  - Component 2: Replace RMSNorm with LigerRMSNorm
  - Component 3: Replace cross-entropy with LigerCrossEntropyLoss (fused with linear)
  - Component 4: Fused residual+norm forward pass
- **Implementation Plan**:
  1. `pip install liger-kernel`
  2. Replace `nn.RMSNorm` with:
     ```python
     from liger_kernel.transformers import LigerRMSNorm
     self.norm1 = LigerRMSNorm(model_dim)
     ```
  3. Replace cross entropy loss:
     ```python
     from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
     self.ce_loss = LigerFusedLinearCrossEntropyLoss()
     ```
  4. Benchmark step time: expect 5-15% throughput improvement
  5. Verify BPB identical (3-seed)
- **Expected Outcome**: +200-500 extra steps, -0.001 to -0.005 BPB
- **Compute Estimate**: 3 seeds * 12 min = 36 min
- **Priority**: HIGH
- **Combines With**: All strategies
- **Source/Inspiration**: parametergolfanalyzer.md Tier 2, LinkedIn open-source

---

### Strategy T1-12: Scylla + SLOT Combination
- **Category**: hybrid
- **Legality**: LIKELY_LEGAL -- both components individually legal/record-eligible
- **Parameter Budget Impact**: Zero additional (both are existing techniques)
- **Core Idea**: The two best pending submissions (#1184 at 0.9485 with Scylla, #1229 at 0.9300 with SLOT) have never been combined. This is explicitly called out as "the obvious untried combination" in parametergolfanalyzer.md.
- **Key Components**:
  - Component 1: Scylla 998-token tokenizer (from T1-01)
  - Component 2: Per-Sample SLOT delta (from T1-03)
  - Component 3: Scored-Position masking
  - Component 4: Full GPTQ + XSA-all base
- **Implementation Plan**:
  1. Implement T1-01 (Scylla tokenizer) as base
  2. Implement T1-03 (Per-Sample SLOT) on top
  3. Train with: 11L, 512d, XSA-all, EMA, MLP 3.0-3.5x, GPTQ within budget
  4. Eval with: sliding window stride=64, SLOT 16 steps, per-sample delta
  5. Sweep SLOT lr: {0.005, 0.008, 0.01}
  6. 3 seeds for statistical significance
- **Expected Outcome**: 0.92-0.94 BPB (beating both #1184 and #1229 individually)
- **Compute Estimate**: ~15 min per run (train + SLOT eval), 3 seeds = 45 min
- **Priority**: HIGH
- **Combines With**: T1-04, T1-05, T1-06, T1-08
- **Source/Inspiration**: parametergolfanalyzer.md "Scylla + SLOT remains the obvious untried combination"

---

### Strategy T1-13: WD 0.085 + MLP 4x (Simplification Approach)
- **Category**: architecture/training
- **Legality**: LEGAL -- #1218 used this at 1.0979 BPB
- **Parameter Budget Impact**: ~35M params at MLP 4x, compressed to fit 16MB via high WD
- **Core Idea**: The RMS-compression insight from #1218: R^2 ~= 0.99 between weight matrix RMS and quantized+compressed size. Higher WD (0.085) keeps weights compressible, enabling MLP 4x (2048 hidden) within 16MB. Remove TTT, QAT, hash embeddings, SmearGate -- simplify.
- **Key Components**:
  - Component 1: MLP_MULT=4.0 (hidden dim = 2048)
  - Component 2: MUON_WD=0.085 (dramatically higher than standard 0.04)
  - Component 3: 4096-vocab tokenizer (sp4096)
  - Component 4: Remove: TTT, QAT, BigramHash, SmearGate
  - Component 5: Keep: XSA-all, EMA, Full GPTQ, Coprime-stride, Brotli
- **Implementation Plan**:
  1. Set hyperparameters:
     - MLP_MULT=4.0, MUON_WD=0.085, VOCAB_SIZE=4096
     - NUM_LAYERS=11, MODEL_DIM=512
     - BIGRAM_VOCAB_SIZE=0 (disabled)
  2. Remove SmearGate from model architecture
  3. Use sp4096 tokenizer: `data/cached_challenge_fineweb.py --variant sp4096`
  4. Train with standard optimizer settings
  5. Monitor RMS of weight matrices during training to verify WD effect
  6. Apply Full GPTQ + Brotli-11 compression
  7. Verify artifact <= 16MB
- **Expected Outcome**: 1.08-1.10 BPB (matching or beating complex stacks via simplicity)
- **Compute Estimate**: 12 min per run, 3 seeds = 36 min
- **Priority**: HIGH
- **Combines With**: T1-02, T1-04, T1-06, T1-08, T1-09
- **Source/Inspiration**: #1218 (@clarkkev, 1.0979 BPB)

---

### Strategy T1-14: Mousse Optimizer (Curvature-Aware Muon)
- **Category**: optimizer
- **Legality**: LEGAL -- drop-in optimizer replacement, no rules violated
- **Parameter Budget Impact**: Zero artifact impact; ~3% training overhead
- **Core Idea**: Mousse adds Shampoo-style preconditioning before Muon's orthogonalization step. This gives ~12% more effective training per step at only 3% compute overhead. Net: more useful steps in 600s.
- **Key Components**:
  - Component 1: Shampoo preconditioning matrices (accumulated second-order statistics)
  - Component 2: Preconditioning applied before Newton-Schulz orthogonalization
  - Component 3: Periodic preconditioning update (every 10-50 steps)
- **Implementation Plan**:
  1. Implement Mousse as Muon subclass:
     ```python
     class Mousse(Muon):
         def __init__(self, *args, precond_interval=10, **kwargs):
             super().__init__(*args, **kwargs)
             self.precond_interval = precond_interval
         
         def step(self, closure=None):
             # Every precond_interval steps, update Shampoo factors
             for group in self.param_groups:
                 for p in group['params']:
                     if p.grad is None: continue
                     state = self.state[p]
                     if 'step_count' not in state:
                         state['step_count'] = 0
                         state['L'] = torch.eye(p.shape[0], device=p.device, dtype=torch.bfloat16)
                         state['R'] = torch.eye(p.shape[1], device=p.device, dtype=torch.bfloat16) if p.ndim >= 2 else None
                     state['step_count'] += 1
                     if state['step_count'] % self.precond_interval == 0:
                         g = p.grad.bfloat16()
                         if p.ndim >= 2:
                             state['L'] = 0.9 * state['L'] + 0.1 * (g @ g.T) / g.shape[1]
                             state['R'] = 0.9 * state['R'] + 0.1 * (g.T @ g) / g.shape[0]
             # Then proceed with standard Muon step
             return super().step(closure)
     ```
  2. The above is simplified -- actual implementation should use arXiv:2603.09697
  3. Sweep precond_interval: {10, 25, 50}
  4. 3 seeds per configuration
- **Expected Outcome**: -0.003 to -0.008 BPB
- **Compute Estimate**: 12 min/run * 3 configs * 3 seeds = 108 min
- **Priority**: HIGH
- **Combines With**: All strategies except T1-09 (both modify optimizer)
- **Source/Inspiration**: arXiv:2603.09697, parametergolfanalyzer.md Tier 2

---

### Strategy T1-15: Deep Delta Learning (DDL) Residual Gates
- **Category**: architecture
- **Legality**: LEGAL -- pure architecture modification, ~5.6K params for 11L
- **Parameter Budget Impact**: +5.6K params (negligible)
- **Core Idea**: Add a rank-1 erasure gate on residuals: `x + beta * proj(x) + f(x)`. The learned gate erases stale features before writing new ones. Paper claims 3-5 ppl improvement at 124M params. Addresses residual-path interference in quantized models.
- **Key Components**:
  - Component 1: Per-layer rank-1 projection: `proj(x) = (u @ x) * v` where u, v are learned vectors of dim model_dim
  - Component 2: Learnable scalar beta per layer
  - Component 3: Total: 2*512 + 1 = 1025 params per layer, 11275 for 11L
- **Implementation Plan**:
  1. In Block.__init__:
     ```python
     self.ddl_u = nn.Parameter(torch.randn(model_dim) * 0.01)
     self.ddl_v = nn.Parameter(torch.randn(model_dim) * 0.01)
     self.ddl_beta = nn.Parameter(torch.zeros(1))
     ```
  2. In Block.forward, replace `x = x + attn_out` with:
     ```python
     erase = self.ddl_beta * (x @ self.ddl_u.unsqueeze(-1)) * self.ddl_v.unsqueeze(0).unsqueeze(0)
     x = x + erase + attn_out
     ```
  3. Same for MLP residual path
  4. Add ddl params to scalar parameter group
  5. 3 seeds
- **Expected Outcome**: -0.003 to -0.007 BPB
- **Compute Estimate**: 12 min * 3 seeds = 36 min
- **Priority**: HIGH
- **Combines With**: All strategies (orthogonal, tiny param cost)
- **Source/Inspiration**: arXiv:2601.00417, parametergolfanalyzer.md Tier 2

---

### Strategy T1-16: Hyper-Connections (Learned Multi-Depth Residual Mixing)
- **Category**: architecture
- **Legality**: LEGAL -- architecture modification, ~176 total params
- **Parameter Budget Impact**: +176 params (negligible) at n=2
- **Core Idea**: Replace simple residual `x + f(x)` with a learned connection matrix that mixes across multiple expansion dimensions. Richer than standard residuals, Catalytic Residuals, or DenseFormer. Drop-in replacement.
- **Key Components**:
  - Component 1: Connection matrix C of shape (n, n) per layer where n=2 (expansion factor)
  - Component 2: Input expansion: [x, x] -> C @ [x, f(x)]
  - Component 3: ~16 params per layer * 11 layers = 176 total
- **Implementation Plan**:
  1. In Block.__init__:
     ```python
     self.hc_alpha = nn.Parameter(torch.eye(2) * 0.5 + torch.randn(2, 2) * 0.01)
     ```
  2. In Block.forward:
     ```python
     # Expanded residual: stack [x, x] along new dim, apply connection matrix
     states = torch.stack([x, x], dim=-1)  # [..., dim, 2]
     attn_out = self.attn(x)
     new_states = torch.stack([x, attn_out], dim=-1)  # [..., dim, 2]
     mixed = (self.hc_alpha @ new_states.unsqueeze(-1)).squeeze(-1)
     x = mixed[..., 0]  # collapse back to residual stream
     ```
  3. Apply same pattern for MLP residual
  4. Add to scalar param group
- **Expected Outcome**: -0.003 to -0.008 BPB
- **Compute Estimate**: 12 min * 3 seeds = 36 min
- **Priority**: HIGH
- **Combines With**: T1-15 (but test independently first), all others
- **Source/Inspiration**: arXiv:2409.19606 (ICLR 2025), mHC: 2512.24880 (DeepSeek)

---

### Strategy T1-17: Hourglass FFN (Deeper MLP at Fewer Params)
- **Category**: architecture
- **Legality**: LEGAL -- pure architecture change
- **Parameter Budget Impact**: Saves ~10-15% MLP params at equal depth, or adds depth at equal params
- **Core Idea**: Replace the wide single-layer MLP (3x expansion) with stacked narrow-to-narrow sub-MLPs with residuals. Creates a deeper MLP at fewer parameters. Paper shows it outperforms conventional FFN up to 400M params. Freed params can go to extra layers or larger BigramHash.
- **Key Components**:
  - Component 1: Replace MLP(512->1536->512) with MLP(512->768->512) + MLP(512->768->512) with residual
  - Component 2: Each sub-MLP uses LeakyReLU(0.5)^2 activation
  - Component 3: Residual connection between sub-MLPs
  - Component 4: Total params: 2*(512*768 + 768*512) = 1.57M vs 512*1536 + 1536*512 = 1.57M (same params, more depth)
- **Implementation Plan**:
  1. Create HourglassMLP class:
     ```python
     class HourglassMLP(nn.Module):
         def __init__(self, dim, sub_mult=1.5, n_layers=2):
             super().__init__()
             sub_dim = int(dim * sub_mult)
             self.layers = nn.ModuleList([
                 nn.Sequential(
                     CastedLinear(dim, sub_dim),
                     # activation applied inline
                     CastedLinear(sub_dim, dim)
                 ) for _ in range(n_layers)
             ])
         
         def forward(self, x):
             for layer in self.layers:
                 h = layer[0](x)
                 h = F.leaky_relu(h, 0.5) ** 2
                 h = layer[1](h)
                 x = x + h  # residual
             return x
     ```
  2. Replace MLP in Block with HourglassMLP
  3. Sweep: sub_mult in {1.5, 2.0}, n_layers in {2, 3}
  4. Verify total param count matches or is smaller than MLP 3x
- **Expected Outcome**: -0.002 to -0.006 BPB
- **Compute Estimate**: 4 configs * 3 seeds * 12 min = 144 min
- **Priority**: HIGH
- **Combines With**: All strategies except T1-05 (alternative MLP design)
- **Source/Inspiration**: arXiv:2602.06471, parametergolfanalyzer.md Tier 2

---

### Strategy T1-18: AdamHD Huber Decay for Quantization-Friendly Weights
- **Category**: training/regularization
- **Legality**: LEGAL -- drop-in WD replacement
- **Parameter Budget Impact**: Zero
- **Core Idea**: Replace L2 weight decay with Huber regularizer: quadratic below threshold, linear above. Specifically suppresses large outlier weights that cause int6 clipping loss, without over-regularizing small weights. Drop-in for Muon's decoupled WD.
- **Key Components**:
  - Component 1: Huber decay function: `huber(w, delta) = w^2/(2*delta) if |w|<delta else |w|-delta/2`
  - Component 2: Threshold delta controls transition from quadratic to linear
  - Component 3: Applied as decoupled regularization (same as current WD)
- **Implementation Plan**:
  1. In Muon.step(), replace:
     ```python
     # Old: p.data.mul_(1.0 - lr * wd)
     # New: Huber decay
     delta = 0.1  # threshold
     mask_small = p.data.abs() < delta
     decay = torch.where(mask_small, wd * p.data, wd * delta * p.data.sign())
     p.data.sub_(lr * decay)
     ```
  2. Sweep delta: {0.05, 0.1, 0.2, 0.5}
  3. Keep WD at 0.04 (Huber replaces the decay function, not the coefficient)
  4. 3 seeds per configuration
- **Expected Outcome**: -0.002 to -0.005 BPB (fewer quantization outliers)
- **Compute Estimate**: 4 configs * 3 seeds * 12 min = 144 min
- **Priority**: HIGH
- **Combines With**: All strategies
- **Source/Inspiration**: arXiv:2511.14721, parametergolfanalyzer.md Tier 2

---

### Strategy T1-19: 1-sqrt Cooldown Shape
- **Category**: training
- **Legality**: LEGAL -- schedule change only
- **Parameter Budget Impact**: Zero
- **Core Idea**: Replace linear warmdown with `1-sqrt((t-T0)/(T+1-T0))`. Outperforms linear, cosine, and other cooldown shapes in WSD schedules per TMLR 2025 paper. Zero-cost swap.
- **Key Components**:
  - Component 1: Modified cooldown schedule function
  - Component 2: Same warmdown start point as current (step = total_steps - warmdown_iters)
- **Implementation Plan**:
  1. In the LR schedule computation, replace:
     ```python
     # Old linear warmdown:
     # progress = (step - warmdown_start) / warmdown_iters
     # lr_scale = 1.0 - progress
     
     # New 1-sqrt warmdown:
     progress = (step - warmdown_start) / warmdown_iters
     lr_scale = 1.0 - math.sqrt(progress)
     ```
  2. Train 3 seeds, compare to linear warmdown baseline
- **Expected Outcome**: -0.001 to -0.003 BPB
- **Compute Estimate**: 3 seeds * 12 min = 36 min
- **Priority**: HIGH
- **Combines With**: All strategies
- **Source/Inspiration**: arXiv:2508.01483 (TMLR 2025)

---

### Strategy T1-20: Batch Size Warmup (262K -> 786K)
- **Category**: training
- **Legality**: LEGAL -- training schedule change
- **Parameter Budget Impact**: Zero
- **Core Idea**: Start with small batch (262K tokens) and grow to 786K as the critical batch size increases during training. This gives 43% more gradient steps early on (when small batches are more efficient) while still achieving the throughput benefits of large batches later.
- **Key Components**:
  - Component 1: Initial batch size: 262144 tokens (1/3 of full)
  - Component 2: Linear ramp to 786432 over first 30% of training
  - Component 3: Constant at 786432 for remaining 70%
- **Implementation Plan**:
  1. Compute batch schedule:
     ```python
     ramp_steps = int(0.3 * total_steps)
     if step < ramp_steps:
         batch_tokens = 262144 + int((786432 - 262144) * step / ramp_steps)
     else:
         batch_tokens = 786432
     # Adjust grad_accum_steps accordingly
     grad_accum_steps = batch_tokens // (micro_batch_tokens * world_size)
     ```
  2. Keep LR schedule unchanged (warmup + warmdown still based on step count)
  3. Monitor: total tokens seen should be ~same as constant 786K (more steps but smaller batches)
  4. 3 seeds
- **Expected Outcome**: -0.002 to -0.005 BPB
- **Compute Estimate**: 3 seeds * 12 min = 36 min
- **Priority**: HIGH
- **Combines With**: All strategies
- **Source/Inspiration**: arXiv:2505.23971, parametergolfanalyzer.md Tier 2

---

### Strategy T1-21: Self-Generated GPTQ Calibration (AR Method)
- **Category**: quantization
- **Legality**: LEGAL -- used in official SOTA #1019 (merged); "probably legal" per valerio-oai
- **Parameter Budget Impact**: Zero
- **Core Idea**: Instead of using random data or training data for GPTQ calibration, have the model autoregressively generate its own calibration data. This closes 84% of the val-vs-random calibration gap. The generated text better represents what the model expects to see.
- **Key Components**:
  - Component 1: After training, generate 32 batches of text autoregressively (temperature=1.0)
  - Component 2: Use generated text as GPTQ calibration data
  - Component 3: Run GPTQ quantization with standard settings
  - Component 4: All within 600s training budget
- **Implementation Plan**:
  1. After training loop completes (check remaining time in 600s budget):
     ```python
     model.eval()
     calib_data = []
     for _ in range(32):
         tokens = torch.zeros(1, 2048, dtype=torch.long, device=device)
         tokens[0, 0] = random.randint(0, vocab_size-1)
         with torch.no_grad():
             for pos in range(1, 2048):
                 logits = model.forward_logits(tokens[:, :pos])
                 next_token = torch.multinomial(F.softmax(logits[:, -1], dim=-1), 1)
                 tokens[0, pos] = next_token
         calib_data.append(tokens)
     ```
  2. Use calib_data for GPTQ Hessian computation
  3. Time budget: typically 30-60s for generation + 30-60s for GPTQ
  4. Ensure total training + calibration + GPTQ <= 600s
- **Expected Outcome**: -0.002 to -0.004 BPB over random calibration data
- **Compute Estimate**: +60-90s within training budget
- **Priority**: HIGH
- **Combines With**: All strategies using GPTQ
- **Source/Inspiration**: #1019 (official SOTA, @abaybektursun)

---

### Strategy T1-22: Training-Data GPTQ Calibration (Within Budget)
- **Category**: quantization
- **Legality**: LEGAL -- #1229 used real training data (256 batches) within 600s
- **Parameter Budget Impact**: Zero
- **Core Idea**: Use actual training data (not val data, not self-generated) for GPTQ calibration within the 600s training budget. Simpler than AR self-generation and potentially more diverse.
- **Key Components**:
  - Component 1: Reserve 256 training batches for calibration
  - Component 2: Run GPTQ calibration as final step within 600s
  - Component 3: Standard GPTQ hyperparameters (blocksize=128, dampening=0.01)
- **Implementation Plan**:
  1. Before training loop, cache 256 random training batches
  2. After training completes (check wallclock), run GPTQ with cached batches
  3. Time budget: ~14s per #1219 (vs 220s for AR self-gen)
  4. This is much faster than T1-21, freeing more time for training
- **Expected Outcome**: -0.001 to -0.003 BPB over random calibration
- **Compute Estimate**: +15-20s within training budget
- **Priority**: HIGH
- **Combines With**: All strategies using GPTQ
- **Source/Inspiration**: #1229 (@resouer), #1219 (14s vs 220s)

---

### Strategy T1-23: Sigmoid-Gated Skip Connections
- **Category**: architecture
- **Legality**: LEGAL -- used in #1089 (1.1086), #1122
- **Parameter Budget Impact**: ~5 params for 5 skip paths
- **Core Idea**: Add learnable sigmoid gates on U-Net skip connections: `out = hidden + sigmoid(g) * skip`. Lets the model tune encoder-decoder blending per skip path.
- **Key Components**:
  - Component 1: One learnable scalar per skip connection
  - Component 2: Sigmoid activation to bound gate in [0, 1]
  - Component 3: Applied to existing U-Net skip paths in the architecture
- **Implementation Plan**:
  1. In GPT.__init__, replace `self.skip_weights` with:
     ```python
     self.skip_gates = nn.Parameter(torch.zeros(self.num_skip_weights))
     ```
  2. In forward, replace skip weight multiplication with:
     ```python
     skip_scale = torch.sigmoid(self.skip_gates[skip_idx])
     x = x + skip_scale * skip_tensor
     ```
  3. Initialize gates to 0 (sigmoid(0) = 0.5 = balanced mixing)
  4. Add to scalar param group
- **Expected Outcome**: -0.001 to -0.002 BPB
- **Compute Estimate**: 3 seeds * 12 min = 36 min
- **Priority**: HIGH
- **Combines With**: All strategies
- **Source/Inspiration**: #1089, #1122

---

### Strategy T1-24: Shallow Depth Recurrence (2-Pass, Layers 4-5)
- **Category**: architecture
- **Legality**: LEGAL -- #686 used this at 1.1182 (3-seed)
- **Parameter Budget Impact**: Saves 2 layers worth of params (reinvested in wider MLP or more BigramHash)
- **Core Idea**: Repeat layers 4 and 5 once each, creating 13 virtual layers from 11 physical layers. Use per-pass learnable block scalars (~2K params). Key: stay within the "2 loops survive" zone -- 3+ loops cause catastrophic GPTQ compounding.
- **Key Components**:
  - Component 1: Layers 4 and 5 are each executed twice in the forward pass
  - Component 2: Per-pass scalar multipliers (~2K params) to differentiate passes
  - Component 3: GPTQ applied to physical weights only (2 loops survive quantization)
- **Implementation Plan**:
  1. In GPT.forward, modify the block loop:
     ```python
     for i, block in enumerate(self.blocks):
         x = block(x, ...)
         if i in [4, 5]:  # repeat layers 4 and 5
             scale = self.recurrence_scales[i - 4]  # learned scalar
             x = x + scale * (block(x, ...) - x)  # residual recurrence
     ```
  2. Add `self.recurrence_scales = nn.Parameter(torch.ones(2) * 0.5)`
  3. Alternatively, use the freed parameter budget (2 layers saved) for MLP 3.5x
  4. 3 seeds
- **Expected Outcome**: -0.003 to -0.008 BPB (recovers ~70% of independent 12L gain)
- **Compute Estimate**: 12 min * 3 seeds = 36 min
- **Priority**: HIGH
- **Combines With**: T1-04, T1-05, T1-06, T1-07; NOT with T1-17 (both modify layer structure)
- **Source/Inspiration**: #686 (@msisovic, 1.1182, 3-seed)

---

### Strategy T1-25: EMA Decay Tuning + XSA Interaction
- **Category**: training
- **Legality**: LEGAL -- standard training technique
- **Parameter Budget Impact**: Zero
- **Core Idea**: EMA at 0.997 is confirmed optimal WITH XSA, but the interaction with other new techniques (SLOT, Scylla, MLP 3.5x) may shift the optimum. Systematic sweep of EMA decay on the new best stack.
- **Key Components**:
  - Component 1: EMA decay sweep: {0.995, 0.996, 0.997, 0.998, 0.999}
  - Component 2: Always combined with XSA (EMA fails without XSA per #201)
  - Component 3: Track both pre-quant and post-quant BPB at each setting
- **Implementation Plan**:
  1. Run 5 EMA values * 3 seeds each on the best current stack
  2. For each: measure pre-quant val_loss, post-GPTQ val_bpb, and artifact size
  3. Plot the tradeoff: higher EMA = smoother weights = better quantization
  4. Select optimal for the new stack
- **Expected Outcome**: -0.001 to -0.003 BPB (confirming or improving on 0.997)
- **Compute Estimate**: 5 * 3 * 12 min = 180 min
- **Priority**: HIGH
- **Combines With**: All strategies
- **Source/Inspiration**: #375 (EMA > SWA by 0.003), #287 (EMA 0.997 optimal)

---

### Strategy T1-26: Context-Only SLOT (Causal Variant)
- **Category**: eval-time
- **Legality**: LEGAL -- #1217 proved causal with only -0.0002 BPP cost
- **Parameter Budget Impact**: Zero artifact cost
- **Core Idea**: SLOT delta computed from context tokens only, not applied to new (unscored) tokens. This eliminates the causality concern raised by @abaybektursun in #1105. Only -0.0002 BPP worse than standard SLOT.
- **Key Components**:
  - Component 1: During SLOT optimization, mask out positions that haven't been scored yet
  - Component 2: Delta is optimized on past context, then applied only at scoring positions
  - Component 3: Avoids any future-token information leakage
- **Implementation Plan**:
  1. In SLOT eval loop:
     ```python
     # Compute delta from context positions only
     context_mask = positions < current_scoring_start
     context_hidden = hidden_states[:, context_mask, :]
     context_logits = model.head(context_hidden + delta)
     context_loss = F.cross_entropy(context_logits, context_targets)
     context_loss.backward()  # gradient flows through delta only
     ```
  2. Apply delta to scoring positions: `h_scored = hidden_scored + delta`
  3. Compare with standard SLOT on 3 seeds
- **Expected Outcome**: Equivalent to standard SLOT within 0.0002 BPP, but provably causal
- **Compute Estimate**: 3 seeds * 15 min = 45 min
- **Priority**: HIGH
- **Combines With**: T1-01, T1-04, T1-05
- **Source/Inspiration**: #1217 (Context-Only SLOT, 1.1027 BPB)

---

### Strategy T1-27: Prune-Then-Quantize Ordering
- **Category**: quantization
- **Legality**: LEGAL -- post-training processing change
- **Parameter Budget Impact**: May save 0.5-1% params via pruning
- **Core Idea**: The Progressive Intensity Hypothesis (ICLR 2026): weaker perturbations first, stronger later. Current approach does quantize-then-prune; reversing to prune-then-quantize is a zero-cost experiment that theory and experiments show gives 0.001-0.003 BPB free gain.
- **Key Components**:
  - Component 1: After training, apply magnitude pruning first (3-5% smallest weights zeroed)
  - Component 2: Then apply GPTQ quantization on pruned model
  - Component 3: Compare both orderings
- **Implementation Plan**:
  1. Train model as normal
  2. Approach A (current): GPTQ -> prune -> compress
  3. Approach B (new): prune (3% magnitude) -> GPTQ -> compress
  4. Compare BPB and artifact size for both orderings
  5. 3 seeds each
- **Expected Outcome**: -0.001 to -0.003 BPB (free gain)
- **Compute Estimate**: 2 orderings * 3 seeds * 12 min = 72 min
- **Priority**: HIGH
- **Combines With**: All strategies using GPTQ
- **Source/Inspiration**: arXiv:2603.18426 (ICLR 2026), parametergolfanalyzer.md Tier 2

---

### Strategy T1-28: N-gram Backoff Cache with Laplace Normalization
- **Category**: eval-time
- **Legality**: GRAY_AREA -- "directionally legal" per valerio-oai; MUST produce full-vocab normalized distributions. Laplace smoothing may satisfy this. #1185 (0.9641) and #1094 (0.4027) are record-eligible.
- **Parameter Budget Impact**: Zero artifact cost
- **Core Idea**: During sliding window eval, build a backward-looking n-gram cache from already-scored tokens. Use multi-order backoff (2-9) with Laplace smoothing for proper normalization. Mix with model predictions using entropy-adaptive alpha.
- **Key Components**:
  - Component 1: Hash table storing n-gram counts (orders 2-9, ~4M buckets)
  - Component 2: Laplace smoothing: `p_ngram(w|context) = (count(context,w) + alpha) / (count(context) + alpha*V)` for full-vocab normalization
  - Component 3: Multi-order backoff: try order 9 first, cascade down on low-count contexts
  - Component 4: Entropy-adaptive mixing: `mix_alpha = 0.05 + 0.55 * sigmoid(2 * (H_model - 4.0))`
- **Implementation Plan**:
  1. Implement NgramCache class:
     ```python
     class NgramCache:
         def __init__(self, max_order=9, n_buckets=4_000_000, vocab_size=1024):
             self.counts = {}  # hash -> count array [vocab_size]
             self.context_counts = {}  # hash -> total count
             self.max_order = max_order
             self.alpha = 1.0  # Laplace smoothing
         
         def update(self, tokens, position):
             """Update cache with token at position (already scored)"""
             for order in range(2, self.max_order + 1):
                 if position >= order - 1:
                     context = tuple(tokens[position-order+1:position].tolist())
                     h = hash(context) % self.n_buckets
                     if h not in self.counts:
                         self.counts[h] = torch.zeros(self.vocab_size)
                         self.context_counts[h] = 0
                     self.counts[h][tokens[position]] += 1
                     self.context_counts[h] += 1
         
         def predict(self, context_tokens):
             """Return full-vocab normalized distribution via backoff"""
             for order in range(self.max_order, 1, -1):
                 context = tuple(context_tokens[-order+1:].tolist())
                 h = hash(context) % self.n_buckets
                 if h in self.counts and self.context_counts[h] >= 2:
                     counts = self.counts[h]
                     total = self.context_counts[h]
                     probs = (counts + self.alpha) / (total + self.alpha * self.vocab_size)
                     return probs / probs.sum()  # ensure normalization
             return None  # fall back to model only
     ```
  2. In eval loop:
     ```python
     model_probs = F.softmax(logits, dim=-1)
     ngram_probs = cache.predict(context_tokens)
     if ngram_probs is not None:
         H = -(model_probs * model_probs.log()).sum()  # model entropy
         alpha = 0.05 + 0.55 * torch.sigmoid(2 * (H - 4.0))
         final_probs = (1 - alpha) * model_probs + alpha * ngram_probs
     else:
         final_probs = model_probs
     loss = -torch.log(final_probs[target_token])
     cache.update(context_tokens, target_token, position)
     ```
  3. Critical: Laplace smoothing ensures every token gets nonzero probability
  4. Validate: sum(final_probs) == 1.0 for every scored position
- **Expected Outcome**: -0.05 to -0.15 BPB (massive improvement, but legality must be confirmed)
- **Compute Estimate**: +2-4 min eval time
- **Priority**: HIGH (conditional on legality confirmation)
- **Combines With**: T1-01, T1-04, T1-05, T1-07; partially redundant with T1-02/T1-03 (SLOT)
- **Source/Inspiration**: #1185 (0.9641, Laplace normalized), #795 (0.8881, order-adaptive)

---

## TIER 2: MEDIUM PRIORITY (34 strategies)

---

### Strategy T2-01: Variance-Adaptive Muon (Muon-VS)
- **Category**: optimizer
- **Legality**: LEGAL -- drop-in optimizer modification
- **Parameter Budget Impact**: Zero
- **Core Idea**: Add variance normalization before NS orthogonalization. Reduces step-size sensitivity and hyperparameter sensitivity. Zero extra hyperparameters.
- **Implementation Plan**:
  1. Before `zeropower_via_newtonschulz5(update)`, add:
     ```python
     row_var = update.var(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
     update = update / row_var
     ```
  2. 3 seeds, compare to standard Muon
- **Expected Outcome**: -0.002 to -0.005 BPB
- **Compute Estimate**: 36 min
- **Priority**: MEDIUM
- **Combines With**: All except T1-09, T1-14
- **Source/Inspiration**: arXiv:2601.14603

---

### Strategy T2-02: GLU Attention on Values
- **Category**: architecture
- **Legality**: LEGAL -- zero-parameter modification
- **Parameter Budget Impact**: Zero additional params, zero overhead
- **Core Idea**: Apply GLU (Gated Linear Unit) nonlinearity on V projections. Composable with XSA.
- **Implementation Plan**:
  1. In Attention.forward, after V projection:
     ```python
     v1, v2 = v.chunk(2, dim=-1)  # requires doubling V projection dim
     v = v1 * torch.sigmoid(v2)
     ```
  2. Alternative (zero-param): apply element-wise sigmoid gating using Q as gate
  3. 3 seeds
- **Expected Outcome**: -0.002 to -0.005 BPB
- **Compute Estimate**: 36 min
- **Priority**: MEDIUM
- **Combines With**: All strategies
- **Source/Inspiration**: arXiv:2507.00022

---

### Strategy T2-03: Softpick / Rectified Softmax for Attention
- **Category**: architecture
- **Legality**: LEGAL -- attention mechanism change
- **Parameter Budget Impact**: Zero
- **Core Idea**: Replace softmax in attention with rectified variant. Eliminates attention sinks and massive activations. Produces sparse attention maps (47%). Key benefit: "Quantized Softpick outperforms quantized softmax at lower bit widths."
- **Implementation Plan**:
  1. Replace `F.scaled_dot_product_attention` with custom:
     ```python
     scores = q @ k.transpose(-2, -1) / math.sqrt(head_dim)
     if causal:
         scores = scores.masked_fill(causal_mask, -1e9)
     attn = F.relu(scores)  # rectified softmax (simplified)
     attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)
     out = attn @ v
     ```
  2. Note: may need FA3-compatible kernel for efficiency
  3. 3 seeds, compare to softmax
- **Expected Outcome**: -0.002 to -0.006 BPB
- **Compute Estimate**: 36 min
- **Priority**: MEDIUM
- **Combines With**: T1-04 (QK-Gain), T1-07 (XSA), T1-15 (DDL)
- **Source/Inspiration**: arXiv:2504.20966

---

### Strategy T2-04: FlashSigmoid Attention
- **Category**: architecture/systems
- **Legality**: LEGAL -- attention mechanism change, 17% kernel speedup
- **Parameter Budget Impact**: Zero
- **Core Idea**: Replace softmax with element-wise sigmoid in attention. Eliminates token competition entirely. 17% kernel speedup on H100 (systems benefit). Compatible with FA3.
- **Implementation Plan**:
  1. Replace attention computation:
     ```python
     scores = q @ k.transpose(-2, -1) / math.sqrt(head_dim)
     if causal:
         scores = scores.masked_fill(causal_mask, -1e9)
     attn = torch.sigmoid(scores)  # sigmoid attention
     # No normalization needed -- sigmoid is per-element
     out = attn @ v
     ```
  2. Use FlashSigmoid kernel if available: `pip install flash-sigmoid`
  3. May need per-head bias term for stable training
  4. 3 seeds
- **Expected Outcome**: -0.001 to -0.005 BPB + systems speedup
- **Compute Estimate**: 36 min
- **Priority**: MEDIUM
- **Combines With**: T1-04 (QK-Gain), T1-07 (XSA)
- **Source/Inspiration**: Apple ICLR 2025, parametergolfanalyzer.md

---

### Strategy T2-05: Progressive Window Warmup (Attention Span Curriculum)
- **Category**: systems/training
- **Legality**: LEGAL -- systems-only, significance waived
- **Parameter Budget Impact**: Zero
- **Core Idea**: Start with short local attention (128-384 tokens), grow to full 2048 during training. Faster early steps -> more total steps. Different from sequence length curriculum -- same input length, just restricted attention span.
- **Implementation Plan**:
  1. Implement attention windowing:
     ```python
     if step < warmup_window_steps:
         window = 128 + int((2048 - 128) * step / warmup_window_steps)
     else:
         window = 2048
     # Apply window mask in attention
     ```
  2. Set warmup_window_steps = 30% of total steps
  3. Monitor step time: expect 20-40% faster for early steps
  4. 3 seeds
- **Expected Outcome**: +200-500 extra steps, -0.002 to -0.005 BPB
- **Compute Estimate**: 36 min
- **Priority**: MEDIUM
- **Combines With**: All strategies
- **Source/Inspiration**: modded-nanogpt proven technique, parametergolfanalyzer.md Tier 2

---

### Strategy T2-06: CAGE QAT Gradient (Curvature-Aware STE)
- **Category**: quantization/training
- **Legality**: LEGAL -- QAT modification
- **Parameter Budget Impact**: Zero
- **Core Idea**: Replace hard STE with curvature-aware gradient using Adam's second-moment estimate. W3A3 CAGE matches W4A4 STE per paper. Composes with HESTIA/Soft-Round.
- **Implementation Plan**:
  1. In CastedLinear's STE, replace `grad = grad` with:
     ```python
     # CAGE: scale STE gradient by curvature
     if hasattr(optimizer_state, 'exp_avg_sq'):
         curvature = optimizer_state['exp_avg_sq'].sqrt() + eps
         cage_grad = grad / curvature
     else:
         cage_grad = grad
     ```
  2. Activate QAT in last 15% of training (late QAT)
  3. 3 seeds
- **Expected Outcome**: -0.002 to -0.005 BPB
- **Compute Estimate**: 36 min
- **Priority**: MEDIUM
- **Combines With**: T1-05 (mixed quant), T1-18 (Huber decay)
- **Source/Inspiration**: arXiv:2510.18784 (ICLR 2026)

---

### Strategy T2-07: EngramLite (Multi-Head Hash Embeddings)
- **Category**: architecture
- **Legality**: LEGAL -- used in #1089 (1.1086, record)
- **Parameter Budget Impact**: ~0.5-1MB for hash tables
- **Core Idea**: Replace single-head BigramHash with multi-head prime-based hash embeddings covering both bigrams and trigrams. Gating mechanism suppresses noisy lookups. From DeepSeek's Engram paper.
- **Implementation Plan**:
  1. Implement EngramLite:
     ```python
     class EngramLite(nn.Module):
         def __init__(self, vocab, n_buckets=8192, dim=128, out_dim=512, n_heads=2):
             super().__init__()
             self.bigram_tables = nn.ModuleList([
                 nn.Embedding(n_buckets, dim) for _ in range(n_heads)
             ])
             self.trigram_tables = nn.ModuleList([
                 nn.Embedding(n_buckets, dim) for _ in range(n_heads)
             ])
             self.gate = nn.Linear(dim * n_heads * 2, 1)
             self.proj = nn.Linear(dim * n_heads * 2, out_dim)
             # Prime-based hashing for reduced collisions
             self.primes = [31, 37, 41, 43]
         
         def hash_ngram(self, tokens, order, head_idx):
             h = 0
             for i, t in enumerate(tokens):
                 h = (h * self.primes[head_idx] + t) % self.n_buckets
             return h
     ```
  2. Replace BigramHashEmbedding with EngramLite
  3. Adjust n_buckets to fit artifact budget: try 4096, 8192
  4. 3 seeds
- **Expected Outcome**: -0.003 to -0.008 BPB
- **Compute Estimate**: 2 configs * 3 seeds * 12 min = 72 min
- **Priority**: MEDIUM
- **Combines With**: All except T1-13 (which removes hash embeddings)
- **Source/Inspiration**: #1089 (EngramLite), DeepSeek Engram (Jun 2025)

---

### Strategy T2-08: QK-Norm + Learned Temperature
- **Category**: architecture
- **Legality**: LEGAL -- used in Gemma 2, DeepSeek-V3
- **Parameter Budget Impact**: ~4 params per head = 32 total
- **Core Idea**: L2-normalize Q and K before dot product, plus learned per-head temperature. Prevents attention logit explosion -- the root cause that LN Scale patches. Could enable stable 12-13L training.
- **Implementation Plan**:
  1. In Attention.forward:
     ```python
     q = F.normalize(q, dim=-1)
     k = F.normalize(k, dim=-1)
     # learned temperature per head
     scores = (q @ k.T) * self.attn_temp  # self.attn_temp: nn.Parameter per head
     ```
  2. Initialize `self.attn_temp = nn.Parameter(torch.ones(num_heads) * math.sqrt(head_dim))`
  3. Try with and without Partial RoPE
  4. 3 seeds
- **Expected Outcome**: -0.001 to -0.004 BPB; may enable 12-13L
- **Compute Estimate**: 36 min
- **Priority**: MEDIUM
- **Combines With**: T1-04 (related but different), T1-07 (XSA)
- **Source/Inspiration**: arXiv:2010.04245, parametergolfanalyzer.md

---

### Strategy T2-09: WaveletGPT (Multi-Scale Haar Structure)
- **Category**: architecture
- **Legality**: LEGAL -- zero-parameter modification
- **Parameter Budget Impact**: Zero
- **Core Idea**: Apply multi-scale Haar wavelet structure on half of embedding dimensions. Claimed 40-60% faster convergence at zero parameter cost. The wavelet decomposition captures both fine-grained and coarse patterns simultaneously.
- **Implementation Plan**:
  1. Split embedding dim in half: first 256 dims normal, last 256 dims wavelet
  2. Apply Haar wavelet transform to second half:
     ```python
     def haar_transform(x):
         # x: [..., dim/2]
         even = x[..., 0::2]
         odd = x[..., 1::2]
         avg = (even + odd) / math.sqrt(2)
         diff = (even - odd) / math.sqrt(2)
         return torch.cat([avg, diff], dim=-1)
     ```
  3. Apply after embedding, before first transformer block
  4. Inverse transform before output projection
  5. 3 seeds
- **Expected Outcome**: -0.003 to -0.010 BPB
- **Compute Estimate**: 36 min
- **Priority**: MEDIUM
- **Combines With**: All strategies
- **Source/Inspiration**: arXiv:2409.12924

---

### Strategy T2-10: P2 / Focal Loss for Token Weighting
- **Category**: training
- **Legality**: LEGAL -- loss function modification
- **Parameter Budget Impact**: Zero
- **Core Idea**: Difficulty-aware loss: `(1-p)^gamma * (-log p)` down-weights easy tokens, focuses gradient on hard tokens. Used in #1180 (1.0577, unvalidated). For BPB optimization, hard tokens dominate the loss, so this may help.
- **Implementation Plan**:
  1. Replace cross-entropy with focal loss:
     ```python
     def focal_loss(logits, targets, gamma=2.0):
         ce = F.cross_entropy(logits, targets, reduction='none')
         p = torch.exp(-ce)  # probability of correct token
         focal = ((1 - p) ** gamma) * ce
         return focal.mean()
     ```
  2. Sweep gamma: {1.0, 2.0, 3.0}
  3. 3 seeds per gamma
- **Expected Outcome**: Unknown (no clean ablation exists); potentially -0.005 to -0.015 BPB
- **Compute Estimate**: 3 * 3 * 12 = 108 min
- **Priority**: MEDIUM
- **Combines With**: All strategies
- **Source/Inspiration**: #1180 (1.0577, unvalidated)

---

### Strategy T2-11: Cross-Layer KV Sharing (MLKV/CLA)
- **Category**: architecture
- **Legality**: LEGAL -- pure architecture change
- **Parameter Budget Impact**: Saves ~0.5MB (2 fewer KV projection matrices)
- **Core Idea**: Adjacent layer pairs share KV projections. Unlike depth recurrence, only KV is shared -- no quantization amplification. Freed 0.5MB enables wider MLP or more BigramHash.
- **Implementation Plan**:
  1. In GPT.__init__, create shared KV projections for layer pairs:
     ```python
     # Layers (0,1), (2,3), (4,5), (6,7), (8,9), 10 alone
     self.shared_kv = nn.ModuleList([
         CastedLinear(model_dim, 2 * kv_dim) for _ in range(6)  # 11 layers -> 6 unique KV
     ])
     ```
  2. In Block.forward, use shared KV from the pair leader:
     ```python
     kv_idx = layer_idx // 2
     k, v = self.shared_kv[kv_idx](x).chunk(2, dim=-1)
     ```
  3. Savings: 5 fewer KV matrices * 2 * kv_dim * model_dim = 5 * 2 * 256 * 512 = 1.3M params
  4. Reinvest in MLP 3.5x or larger BigramHash
  5. 3 seeds
- **Expected Outcome**: -0.002 to -0.006 BPB (from reinvested params)
- **Compute Estimate**: 36 min
- **Priority**: MEDIUM
- **Combines With**: All except T1-24 (both modify layer structure)
- **Source/Inspiration**: MLKV/CLA (NAACL 2025), parametergolfanalyzer.md Tier 2

---

### Strategy T2-12: Temperature Scaling Post Mixed-Quantization
- **Category**: quantization
- **Legality**: LEGAL -- post-processing at eval time
- **Parameter Budget Impact**: Zero (1 scalar)
- **Core Idea**: The autopsy showed mixed int5/int6 causes ECE to jump from 0.24% to 1.26% (systematic overconfidence). Temperature scaling at eval time can recover calibration. This is a zero-cost fix.
- **Implementation Plan**:
  1. After GPTQ quantization, find optimal temperature:
     ```python
     # On a small held-out validation subset (or first 10% of eval):
     temperatures = torch.linspace(0.9, 1.1, 21)
     for T in temperatures:
         logits_scaled = logits / T
         loss = F.cross_entropy(logits_scaled, targets)
         # Track best T
     ```
  2. Apply `T_opt` to all logits during evaluation
  3. Typical optimal T for overconfident models: 1.02-1.08
  4. Compatible with SLOT (apply T before SLOT delta)
- **Expected Outcome**: -0.001 to -0.003 BPB
- **Compute Estimate**: Minutes (sweep on existing checkpoint)
- **Priority**: MEDIUM
- **Combines With**: T1-05 (mixed quant), all others
- **Source/Inspiration**: pr-1105-model-autopsy (ECE degradation finding)

---

### Strategy T2-13: CPSVD (Column-Preserving SVD)
- **Category**: compression
- **Legality**: LEGAL -- post-training compression
- **Parameter Budget Impact**: Reduces param count via low-rank factorization of select columns
- **Core Idea**: Identify weight columns that compress cleanly via low-rank factorization, store the rest as int6. Orthogonal to quantization -- reduces param count, not precision. Entirely unexplored in competition.
- **Implementation Plan**:
  1. For each weight matrix W [m, n]:
     ```python
     U, S, Vt = torch.linalg.svd(W, full_matrices=False)
     # Find optimal rank k where low-rank is cheaper than int6
     for k in range(1, min(m, n)):
         low_rank_bytes = (m * k + k + k * n) * 2  # fp16 factors
         int6_bytes = m * n * 6 / 8
         if low_rank_bytes < int6_bytes * 0.8:  # 20% savings threshold
             # Use low-rank for this matrix
             W_approx = (U[:, :k] * S[:k]) @ Vt[:k, :]
             break
     ```
  2. Store low-rank factors for compressible matrices, int6 for others
  3. Mix within the 16MB budget via knapsack optimization
  4. 3 seeds (retrain with the factorized architecture for best results)
- **Expected Outcome**: -0.003 to -0.008 BPB (via freed bytes)
- **Compute Estimate**: +30 min analysis, then 3 * 12 min training = 66 min
- **Priority**: MEDIUM
- **Combines With**: T1-05 (mixed quant), T1-06 (Brotli)
- **Source/Inspiration**: arXiv:2510.19385, parametergolfanalyzer.md Tier 2

---

### Strategy T2-14: Compute-Optimal QAT Scheduling
- **Category**: training/quantization
- **Legality**: LEGAL -- training schedule change
- **Parameter Budget Impact**: Zero
- **Core Idea**: Activate QAT at warmdown onset, eliminating redundant FP updates. The Apple scaling law for optimal FP->QAT split suggests fusing cooldown and QAT phases. Principled replacement for empirical Late QAT thresholds.
- **Implementation Plan**:
  1. Set QAT activation to coincide with warmdown start:
     ```python
     qat_start_step = total_steps - warmdown_iters
     if step >= qat_start_step:
         enable_qat()
     ```
  2. Use STE for int6 quantization in forward pass
  3. Note: #315's Late QAT was dead code (torch.compile bug). Must verify activation.
  4. Test with tensor-scale STE (not class attribute):
     ```python
     # In CastedLinear.forward:
     if self.qat_enabled_tensor.item():  # tensor, not Python bool
         w = fake_quantize(self.weight)
     ```
  5. 3 seeds
- **Expected Outcome**: -0.001 to -0.004 BPB
- **Compute Estimate**: 36 min
- **Priority**: MEDIUM
- **Combines With**: T1-05, T1-18
- **Source/Inspiration**: arXiv:2509.22935 (Apple), #1032 (QAT dead-code confirmation)

---

### Strategy T2-15: Block AttnRes (Efficient Variant)
- **Category**: architecture
- **Legality**: LEGAL -- architecture modification
- **Parameter Budget Impact**: <2% overhead (vs 54% for full AttnRes)
- **Core Idea**: Original AttnRes failed at 54% throughput penalty (#362). Block partitioning (3 blocks at 11L) reduces overhead to <2% while retaining 1.25x convergence efficiency. Learned softmax over layer outputs within each block.
- **Implementation Plan**:
  1. Partition 11 layers into 3 blocks: [0-3], [4-7], [8-10]
  2. For each block, compute attention over block layer outputs:
     ```python
     block_outputs = [layers[i](x) for i in block_range]
     stacked = torch.stack(block_outputs, dim=0)  # [n_layers_in_block, B, T, D]
     # Learned routing weights
     weights = F.softmax(self.block_route_logits, dim=0)  # [n_layers_in_block]
     x = (weights.view(-1, 1, 1, 1) * stacked).sum(0)
     ```
  3. `self.block_route_logits = nn.Parameter(torch.zeros(block_size))`
  4. 3 seeds
- **Expected Outcome**: -0.003 to -0.008 BPB
- **Compute Estimate**: 36 min
- **Priority**: MEDIUM
- **Combines With**: All strategies
- **Source/Inspiration**: arXiv:2603.15031 (Kimi, Mar 2026)

---

### Strategy T2-16: FoX Forgetting Attention
- **Category**: architecture
- **Legality**: LEGAL -- attention mechanism modification
- **Parameter Budget Impact**: ~512 params per layer = 5.6K total
- **Core Idea**: Data-dependent forget gate on attention. Eliminates need for positional embeddings. FA3-compatible. Could replace or complement Partial RoPE.
- **Implementation Plan**:
  1. In Attention.forward:
     ```python
     forget_gate = torch.sigmoid(self.forget_proj(x))  # [B, T, H]
     # Cumulative forget: each position forgets accumulated past
     cum_forget = torch.cumsum(torch.log(forget_gate + 1e-8), dim=1)
     # Apply to attention scores
     scores = scores + cum_forget[:, :, None, :] - cum_forget[:, None, :, :]
     ```
  2. `self.forget_proj = nn.Linear(model_dim, num_heads, bias=False)`
  3. Try with and without RoPE
  4. 3 seeds
- **Expected Outcome**: -0.003 to -0.008 BPB
- **Compute Estimate**: 2 configs * 3 seeds * 12 min = 72 min
- **Priority**: MEDIUM
- **Combines With**: T1-04 (QK-Gain), T1-07 (XSA)
- **Source/Inspiration**: arXiv:2503.02130 (ICLR 2025)

---

### Strategy T2-17: DeepCrossAttention (Input-Dependent Depth Routing)
- **Category**: architecture
- **Legality**: LEGAL -- architecture modification
- **Parameter Budget Impact**: ~1K params for 11L
- **Core Idea**: Input-dependent depth routing over all previous layers, replacing simple residuals. Each layer attends to all previous layer outputs with a lightweight routing mechanism. Claims 3x convergence speed.
- **Implementation Plan**:
  1. Store layer outputs: `layer_outputs = []`
  2. After each layer:
     ```python
     layer_outputs.append(x.detach())  # or with grad
     if len(layer_outputs) > 1:
         stacked = torch.stack(layer_outputs[:-1], dim=0)  # [L, B, T, D]
         # Lightweight cross-attention: query from current, keys from previous
         q = x.mean(dim=1)  # [B, D]
         k = stacked.mean(dim=2)  # [L, B, D]
         scores = (q.unsqueeze(0) * k).sum(-1)  # [L, B]
         weights = F.softmax(scores, dim=0)
         context = (weights.unsqueeze(-1).unsqueeze(-1) * stacked).sum(0)
         x = x + 0.1 * context  # small residual from cross-attention
     ```
  3. 3 seeds
- **Expected Outcome**: -0.003 to -0.008 BPB
- **Compute Estimate**: 36 min
- **Priority**: MEDIUM
- **Combines With**: All except T2-15 (both modify inter-layer connections)
- **Source/Inspiration**: arXiv:2502.06785 (ICML 2025)

---

### Strategy T2-18: SSMax (Scalable-Softmax)
- **Category**: architecture
- **Legality**: LEGAL -- one scalar multiply
- **Parameter Budget Impact**: Zero
- **Core Idea**: Scale softmax by input sequence length to prevent attention flattening at seq2048. One scalar multiply. Compatible with FA3.
- **Implementation Plan**:
  1. In attention computation:
     ```python
     scale = math.log(seq_len) / math.log(512)  # base scale at 512
     scores = (q @ k.T) * (scale / math.sqrt(head_dim))
     ```
  2. 3 seeds
- **Expected Outcome**: -0.001 to -0.004 BPB
- **Compute Estimate**: 36 min
- **Priority**: MEDIUM
- **Combines With**: All strategies
- **Source/Inspiration**: arXiv:2501.19399

---

### Strategy T2-19: Conv Token Mixer (Causal Depthwise Conv1d)
- **Category**: architecture
- **Legality**: LEGAL -- architecture addition
- **Parameter Budget Impact**: ~3K params/layer for kernel=4
- **Core Idea**: Add causal depthwise conv1d for cheap local context mixing. Frees attention capacity for long-range. From ConvMixer/Conformer/Mamba lineage.
- **Implementation Plan**:
  1. In Block, add after normalization, before attention:
     ```python
     self.conv = nn.Conv1d(model_dim, model_dim, kernel_size=4, padding=3, groups=model_dim)
     # In forward:
     h = self.conv(x.transpose(1, 2))[:, :, :x.size(1)].transpose(1, 2)  # causal
     x = x + h
     ```
  2. 3 seeds
- **Expected Outcome**: Unknown (from #1180, no ablation)
- **Compute Estimate**: 36 min
- **Priority**: MEDIUM
- **Combines With**: All strategies
- **Source/Inspiration**: #1180 (1.0577, unvalidated), ConvMixer lineage

---

### Strategy T2-20: Seesaw LR + Batch Schedule
- **Category**: training
- **Legality**: LEGAL -- training schedule
- **Parameter Budget Impact**: Zero
- **Core Idea**: Multiply LR by 1/sqrt(2) and double batch size simultaneously at specific points. ~36% fewer serial steps at equal FLOPs. Principled foundation for the batch ramp.
- **Implementation Plan**:
  1. At 50% of training: `lr *= 1/sqrt(2); batch_size *= 2`
  2. At 75% of training: `lr *= 1/sqrt(2); batch_size *= 2`
  3. Starting: batch=262K, lr=0.025
  4. 50%: batch=524K, lr=0.0177
  5. 75%: batch=1048K, lr=0.0125
  6. 3 seeds
- **Expected Outcome**: -0.002 to -0.005 BPB
- **Compute Estimate**: 36 min
- **Priority**: MEDIUM
- **Combines With**: All strategies except T1-20 (alternative batch schedule)
- **Source/Inspiration**: arXiv:2510.14717

---

### Strategy T2-21: Layer-Wise FFN Scaling (Non-Uniform MLP Width)
- **Category**: architecture
- **Legality**: LEGAL -- just per-layer dims, zero cost
- **Parameter Budget Impact**: Same total params, better allocation
- **Core Idea**: Non-uniform FFN width per layer. Autopsy shows Layer 7 does most work (-3.82 bits/token). Give middle layers MLP-4x and edge layers MLP-2x. Same total params, better allocation.
- **Implementation Plan**:
  1. Define per-layer MLP multipliers:
     ```python
     mlp_mults = [2.0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.0, 3.5, 3.0, 2.5, 2.0]  # 11 layers
     # Verify: sum of mlp_mults * model_dim^2 ~= 11 * 3.0 * model_dim^2
     ```
  2. In GPT.__init__, create per-layer MLP dimensions:
     ```python
     for i in range(num_layers):
         mlp_dim = int(mlp_mults[i] * model_dim)
         # ... create MLP with this dim
     ```
  3. Try multiple allocation patterns: crown (wide middle), frame (wide edges), reverse
  4. 3 seeds per pattern
- **Expected Outcome**: -0.002 to -0.005 BPB
- **Compute Estimate**: 3 patterns * 3 seeds * 12 min = 108 min
- **Priority**: MEDIUM
- **Combines With**: All strategies
- **Source/Inspiration**: arXiv:2509.06518, pr-1105-model-autopsy (Layer 7 dominance)

---

### Strategy T2-22: NuMuon (Nuclear-Norm Constrained Muon)
- **Category**: optimizer
- **Legality**: LEGAL -- optimizer modification
- **Parameter Budget Impact**: Potentially smaller artifact (lower stable rank weights compress better)
- **Core Idea**: Nuclear-norm constraint on Muon updates produces lower stable rank weights that compress better with zstd/Brotli. Pushes compressibility into the optimizer itself.
- **Implementation Plan**:
  1. After Newton-Schulz orthogonalization, apply nuclear norm constraint:
     ```python
     U, S, Vt = torch.linalg.svd(update, full_matrices=False)
     S = torch.clamp(S, max=nuclear_budget)
     update = U @ torch.diag(S) @ Vt
     ```
  2. Sweep nuclear_budget: {0.5, 1.0, 2.0} relative to mean singular value
  3. Measure both BPB and artifact size
  4. 3 seeds
- **Expected Outcome**: -0.002 to -0.006 BPB (via smaller artifacts enabling larger model)
- **Compute Estimate**: 3 * 3 * 12 = 108 min
- **Priority**: MEDIUM
- **Combines With**: All except T1-14 (both modify Muon)
- **Source/Inspiration**: arXiv:2603.03597

---

### Strategy T2-23: Parallel Residuals (Dual-Lane Routing)
- **Category**: architecture
- **Legality**: LEGAL -- #1204 submitted, architecture modification
- **Parameter Budget Impact**: ~2K params for routing between lanes
- **Core Idea**: Separate attention and MLP residual lanes from layer 7. Learned cross-lane routing allows the attention and MLP pathways to specialize. Ported from modded-nanogpt PR #230.
- **Implementation Plan**:
  1. From layer 7 onward, maintain two residual streams:
     ```python
     if layer_idx >= 7:
         attn_stream = x_attn + self.attn(norm(x_attn))
         mlp_stream = x_mlp + self.mlp(norm(x_mlp))
         # Cross-lane routing
         x_attn = attn_stream + self.cross_route_a * mlp_stream
         x_mlp = mlp_stream + self.cross_route_m * attn_stream
     ```
  2. `self.cross_route_a = nn.Parameter(torch.tensor(0.1))`
  3. Merge at output: `x = (x_attn + x_mlp) / 2`
  4. 3 seeds
- **Expected Outcome**: -0.002 to -0.005 BPB
- **Compute Estimate**: 36 min
- **Priority**: MEDIUM
- **Combines With**: All strategies
- **Source/Inspiration**: #1204 (@msisovic), modded-nanogpt PR #230

---

### Strategy T2-24: MTP Auxiliary Loss (Training-Only, 2 Heads)
- **Category**: training
- **Legality**: LEGAL -- training-only signal, discarded at export
- **Parameter Budget Impact**: Zero artifact (MTP heads not exported)
- **Core Idea**: Multi-token prediction as auxiliary training signal with 2 heads and weight=0.1. Discarded at export. #1031 claimed -0.0037 BPP. Different from #212 which found MTP useless -- the difference is lower weight and auxiliary-only usage.
- **Implementation Plan**:
  1. Set `MTP_NUM_HEADS=2, MTP_LOSS_WEIGHT=0.1` (already in codebase)
  2. Verify MTP heads are NOT included in exported model
  3. 3 seeds, compare to MTP_NUM_HEADS=0
- **Expected Outcome**: -0.002 to -0.004 BPB (cautious -- #212 found no effect)
- **Compute Estimate**: 36 min
- **Priority**: MEDIUM
- **Combines With**: All strategies
- **Source/Inspiration**: #1031 (@michaelwinczuk, -0.0037 claimed)

---

### Strategy T2-25: Adaptive Entropy-Guided Stride (Two-Pass Eval)
- **Category**: eval-time
- **Legality**: LIKELY_LEGAL -- backward-looking, uses only model's own entropy
- **Parameter Budget Impact**: Zero
- **Core Idea**: First pass with stride=64 scores all tokens and records per-token entropy. Second pass re-evaluates high-entropy regions with smaller stride (16-32). Targets compute where it helps most.
- **Implementation Plan**:
  1. Pass 1: Standard sliding window, stride=64. Record entropy per position.
  2. Identify positions where entropy > threshold (e.g., > 4.0 nats)
  3. Pass 2: Re-evaluate those regions with stride=16
  4. Use second-pass scores for high-entropy positions, first-pass for low-entropy
  5. Key: both passes are forward-only, backward-looking
  6. Threshold sweep: {3.0, 4.0, 5.0}
- **Expected Outcome**: -0.005 to -0.015 BPB
- **Compute Estimate**: ~2x eval time, still within 10 min budget
- **Priority**: MEDIUM
- **Combines With**: T1-02, T1-03, T1-10, T1-28
- **Source/Inspiration**: parametergolfanalyzer.md Tier 1 untried

---

### Strategy T2-26: WSM Checkpoint Merging (Replace Warmdown)
- **Category**: training
- **Legality**: LEGAL -- training method
- **Parameter Budget Impact**: Zero
- **Core Idea**: Replace warmdown with constant-LR training plus offline checkpoint merge. More full-LR steps = more effective training. Theoretically optimal. Compatible with existing EMA.
- **Implementation Plan**:
  1. Remove warmdown: train at constant peak LR for full 600s
  2. Save checkpoints every 100 steps during last 30% of training
  3. Average all saved checkpoints: `merged = sum(checkpoints) / len(checkpoints)`
  4. Apply GPTQ to merged model
  5. Compare: merged model BPB vs EMA + warmdown BPB
  6. 3 seeds
- **Expected Outcome**: -0.002 to -0.006 BPB
- **Compute Estimate**: 36 min (overhead for saving checkpoints is minimal)
- **Priority**: MEDIUM
- **Combines With**: All strategies except EMA-based approaches
- **Source/Inspiration**: arXiv:2507.17634

---

### Strategy T2-27: Frequency-Ordered Tokenization Compression
- **Category**: compression
- **Legality**: LEGAL -- post-hoc encoding step
- **Parameter Budget Impact**: Saves 200-500KB of artifact, enabling larger model
- **Core Idea**: Reorder vocabulary by frequency and encode with variable-length integers before compression. Achieves 0.76-7.08% improvement on standard compressors. Applied to the serialized weight data.
- **Implementation Plan**:
  1. After quantization, analyze weight value distribution
  2. Reorder quantized values by frequency (most common first)
  3. Encode with variable-length integers (VLQ or Elias gamma coding)
  4. Apply zstd-22 or Brotli-11 on top
  5. At eval time: reverse the encoding
  6. Measure artifact size savings
- **Expected Outcome**: 200-500KB savings = 0 to -0.003 BPB (via larger model)
- **Compute Estimate**: Minutes (post-processing)
- **Priority**: MEDIUM
- **Combines With**: T1-06 (Brotli), all other compression strategies
- **Source/Inspiration**: arXiv:2602.22958

---

### Strategy T2-28: V:N:M Activation Sparsity (Structured Sparsity for relu^2)
- **Category**: systems
- **Legality**: LEGAL -- systems-only, significance waived
- **Parameter Budget Impact**: Zero
- **Core Idea**: LeakyReLU(0.5)^2 already produces >90% sparse activations. Enforce NVIDIA 2:4 structured sparsity pattern for 2x sparse matmul on H100 tensor cores. ~15-20% more training steps.
- **Implementation Plan**:
  1. After activation in MLP:
     ```python
     h = F.leaky_relu(h, 0.5) ** 2
     # Enforce 2:4 sparsity
     h_reshaped = h.reshape(*h.shape[:-1], -1, 4)
     _, indices = h_reshaped.abs().topk(2, dim=-1)
     mask = torch.zeros_like(h_reshaped).scatter_(-1, indices, 1.0)
     h = h * mask.reshape_as(h)
     ```
  2. Note: requires NVIDIA sparse tensor cores -- use `torch.sparse.to_sparse_semi_structured`
  3. Benchmark step time improvement
  4. WARNING: #1105 reported 2:4 sparsity at +0.672 BPB (definitively dead). This may need different enforcement (training with sparsity mask, not post-hoc)
  5. Alternative: use activation sparsity for inference-only speedup during eval
- **Expected Outcome**: UNCERTAIN -- reported dead in #1105. Try inference-only variant.
- **Compute Estimate**: 36 min
- **Priority**: MEDIUM (speculative given #1105 result)
- **Combines With**: All strategies
- **Source/Inspiration**: arXiv:2602.06183, arXiv:2503.16672

---

### Strategy T2-29: Over-Encoding Input Embeddings (Hierarchical N-gram)
- **Category**: architecture
- **Legality**: LEGAL -- architecture modification
- **Parameter Budget Impact**: <5% memory overhead for n-gram tables
- **Core Idea**: Keep output vocab at V but use hierarchical n-gram input embeddings: sum of 1-gram + 2-gram + 3-gram embedding tables. Creates exponentially larger effective input vocab with small overhead. Could be key to sub-0.9 with Scylla.
- **Implementation Plan**:
  1. Create multi-order input embedding:
     ```python
     class OverEncoding(nn.Module):
         def __init__(self, vocab_size, dim, max_order=3, n_buckets=4096):
             self.embed_1 = nn.Embedding(vocab_size, dim)
             self.embed_2 = nn.Embedding(n_buckets, dim)
             self.embed_3 = nn.Embedding(n_buckets, dim)
         
         def forward(self, token_ids):
             e1 = self.embed_1(token_ids)
             # 2-gram hash
             bigram_hash = (token_ids[:, :-1] * 31 + token_ids[:, 1:]) % self.n_buckets
             e2 = torch.zeros_like(e1)
             e2[:, 1:] = self.embed_2(bigram_hash)
             # 3-gram hash
             trigram_hash = (token_ids[:, :-2] * 31*31 + token_ids[:, 1:-1] * 31 + token_ids[:, 2:]) % self.n_buckets
             e3 = torch.zeros_like(e1)
             e3[:, 2:] = self.embed_3(trigram_hash)
             return e1 + e2 + e3
     ```
  2. Replace standard embedding with OverEncoding
  3. Works alongside BigramHash (different mechanism -- OE is input, BigramHash is additive)
  4. 3 seeds
- **Expected Outcome**: -0.005 to -0.015 BPB (especially with Scylla tokenizer)
- **Compute Estimate**: 36 min
- **Priority**: MEDIUM
- **Combines With**: T1-01 (Scylla), T1-12 (Scylla+SLOT), all others
- **Source/Inspiration**: Over-Encoding ICML 2025, parametergolfanalyzer.md path below 0.9

---

### Strategy T2-30: HESTIA Soft QAT (Temperature-Annealed)
- **Category**: quantization/training
- **Legality**: LEGAL -- QAT method change
- **Parameter Budget Impact**: Zero
- **Core Idea**: Replace hard STE with temperature-annealed softmax relaxation plus per-tensor Hessian guidance. Enables earlier QAT without premature discretization.
- **Implementation Plan**:
  1. In CastedLinear, replace hard STE with soft quantization:
     ```python
     def soft_quantize(w, temperature, n_levels=64):
         grid = torch.linspace(-32, 31, n_levels, device=w.device)
         distances = -(w.unsqueeze(-1) - grid) ** 2 / temperature
         soft_assignment = F.softmax(distances, dim=-1)
         w_soft = (soft_assignment * grid).sum(-1)
         return w_soft
     ```
  2. Anneal temperature: start at 1.0, decay to 0.01 during training
  3. Transition to hard quantization at temperature < 0.1
  4. 3 seeds
- **Expected Outcome**: -0.002 to -0.006 BPB
- **Compute Estimate**: 36 min
- **Priority**: MEDIUM
- **Combines With**: T1-05 (mixed quant), T2-14 (QAT scheduling)
- **Source/Inspiration**: arXiv:2601.20745

---

### Strategy T2-31: Relaxed Recursive Transformers + LoRA Deltas
- **Category**: architecture
- **Legality**: LEGAL -- architecture design
- **Parameter Budget Impact**: Effectively 20L+ model with ~11L parameter budget
- **Core Idea**: Share base weights across all layers, add tiny per-layer LoRA deltas (rank-32). Effectively creates a much deeper model within the 16MB budget. SVD-initialized.
- **Implementation Plan**:
  1. Create shared transformer block:
     ```python
     shared_block = Block(model_dim, num_heads, ...)
     ```
  2. Create per-layer LoRA adapters:
     ```python
     class LoRAAdapter(nn.Module):
         def __init__(self, dim, rank=32):
             self.A = nn.Parameter(torch.randn(dim, rank) * 0.01)
             self.B = nn.Parameter(torch.zeros(rank, dim))
         def forward(self, x):
             return x + x @ self.A @ self.B
     ```
  3. Forward pass: 20 virtual layers using shared block + per-layer LoRA
  4. GPTQ applies to shared block (1 copy) + all LoRAs
  5. Key risk: GPTQ compounding at 2+ loops (stay at 2 loops maximum)
  6. Try: 11 physical layers with 2 loops each = 22 virtual layers
  7. 3 seeds
- **Expected Outcome**: -0.010 to -0.030 BPB (highly uncertain)
- **Compute Estimate**: 36 min (but step time may increase 50-100%)
- **Priority**: MEDIUM
- **Combines With**: T1-06 (Brotli for compression), T2-12 (temp scaling)
- **Source/Inspiration**: arXiv:2410.20672 (ICLR 2025)

---

### Strategy T2-32: Predictive Batch Scheduling (Loss-Aware Data Ordering)
- **Category**: training
- **Legality**: LEGAL -- data ordering is not content curriculum
- **Parameter Budget Impact**: Zero
- **Core Idea**: Order training batches by predicted loss to avoid wasting gradient steps on too-easy or too-hard batches. NOT content-based curriculum (which failed per #212). This is loss-aware scheduling.
- **Implementation Plan**:
  1. Pre-compute approximate loss for each training batch (first 100 steps with random ordering)
  2. Sort batches: medium-difficulty first, then hard, then easy
  3. This front-loads the most informative gradients
  4. Alternative: interleave easy/hard batches for gradient diversity
  5. 3 seeds
- **Expected Outcome**: -0.002 to -0.005 BPB
- **Compute Estimate**: +30 min pre-computation, then 36 min training
- **Priority**: MEDIUM
- **Combines With**: T1-08 (coprime stride), all others
- **Source/Inspiration**: arXiv:2602.17066

---

### Strategy T2-33: ASQU (Per-Channel Learned Asymmetric Activation)
- **Category**: architecture
- **Legality**: LEGAL -- #1035 showed -0.0011 BPB
- **Parameter Budget Impact**: 512 params per MLP (negligible)
- **Core Idea**: Replace fixed LeakyReLU(0.5)^2 with per-channel learned asymmetric activation. Each channel learns its own negative slope and scaling.
- **Implementation Plan**:
  1. Replace activation:
     ```python
     class ASQU(nn.Module):
         def __init__(self, dim):
             self.neg_slope = nn.Parameter(torch.full((dim,), 0.5))
             self.scale = nn.Parameter(torch.ones(dim))
         def forward(self, x):
             return self.scale * F.leaky_relu(x, self.neg_slope.clamp(0.01, 0.99)) ** 2
     ```
  2. Initialize neg_slope=0.5 (matching LeakyReLU(0.5)^2 baseline)
  3. 3 seeds
- **Expected Outcome**: -0.001 to -0.003 BPB
- **Compute Estimate**: 36 min
- **Priority**: MEDIUM
- **Combines With**: All strategies
- **Source/Inspiration**: #1035 (@andrewmouldon, -0.0011 consistent)

---

### Strategy T2-34: PoPE (Polar Position Embedding)
- **Category**: architecture
- **Legality**: LEGAL -- positional embedding change
- **Parameter Budget Impact**: Negligible
- **Core Idea**: Decouple content (magnitude) from position (angle) in attention. Principled fix for what Partial RoPE approximates. Strong length extrapolation. OpenAI co-author.
- **Implementation Plan**:
  1. Replace Rotary embedding with PoPE:
     ```python
     class PoPE(nn.Module):
         def __init__(self, dim):
             self.angle_proj = nn.Linear(1, dim // 2, bias=False)
         
         def forward(self, x, positions):
             # Content = magnitude, position = angle
             mag = x.norm(dim=-1, keepdim=True)
             direction = x / (mag + 1e-8)
             angles = self.angle_proj(positions.float().unsqueeze(-1))
             cos_a, sin_a = angles.cos(), angles.sin()
             # Apply rotation (like RoPE but magnitude-preserving)
             x_rotated = direction * cos_a + direction.roll(1, -1) * sin_a
             return mag * x_rotated
     ```
  2. Apply to Q and K only
  3. 3 seeds, compare to Partial RoPE
- **Expected Outcome**: -0.002 to -0.005 BPB
- **Compute Estimate**: 36 min
- **Priority**: MEDIUM
- **Combines With**: All strategies except Partial RoPE (replacement)
- **Source/Inspiration**: arXiv:2509.10534

---

## TIER 3: EXPLORATORY (25 strategies)

---

### Strategy T3-01: FineWeb-Aligned Tokenizer Training
- **Category**: tokenizer
- **Legality**: LIKELY_LEGAL -- tokenizer changes face extra scrutiny but are allowed
- **Parameter Budget Impact**: Depends on vocab size chosen
- **Core Idea**: Train tokenizer on FineWeb training data itself rather than generic English. Reduces domain mismatch between tokenizer and eval data.
- **Implementation Plan**:
  1. Use SentencePiece to train BPE tokenizer on FineWeb train data
  2. Try vocab sizes: 512, 768, 1024, 1536, 2048
  3. Tokenize FineWeb with each tokenizer
  4. Train model with each, compare BPB
  5. Key: BPB calculation must be validated carefully per README
- **Expected Outcome**: -0.002 to -0.010 BPB
- **Compute Estimate**: 2 hours (tokenizer training + 5 model trainings)
- **Priority**: LOW
- **Combines With**: All strategies
- **Source/Inspiration**: parametergolfanalyzer.md tokenizer section

---

### Strategy T3-02: Aggressive Vocab Pruning (512-800 tokens)
- **Category**: tokenizer
- **Legality**: LIKELY_LEGAL -- tokenizer modification
- **Parameter Budget Impact**: Reduces embedding layer by 25-50%
- **Core Idea**: #1143 pruned 2.5% (1024->998). Literature suggests 60%+ tokens are removable. Try much more aggressive pruning: 800, 640, 512 tokens.
- **Implementation Plan**:
  1. Start from Scylla 998-token tokenizer
  2. For each target vocab {800, 640, 512}:
     a. Compute per-token frequency on FineWeb val
     b. Remove least frequent tokens, falling back to byte-level encoding
     c. Retokenize FineWeb
     d. Train model and measure BPB
  3. The tradeoff: fewer tokens = smaller embedding, but longer sequences = less context per position
- **Expected Outcome**: -0.005 to -0.020 BPB (highly uncertain)
- **Compute Estimate**: 4 hours
- **Priority**: LOW
- **Combines With**: T1-01, T1-12
- **Source/Inspiration**: parametergolfanalyzer.md tokenizer open directions

---

### Strategy T3-03: Differential Attention
- **Category**: architecture
- **Legality**: LEGAL -- attention mechanism
- **Parameter Budget Impact**: 2x attention heads (doubles Q/K projections)
- **Core Idea**: Compute difference of two softmax maps. Reduces outliers. Shown to improve at scale.
- **Implementation Plan**:
  1. Double the number of Q/K heads
  2. Compute two attention patterns and take their difference
  3. This requires significant parameter budget -- may need to trade MLP width
  4. 3 seeds
- **Expected Outcome**: -0.005 to -0.015 BPB (but high param cost)
- **Compute Estimate**: 36 min
- **Priority**: LOW
- **Combines With**: Limited (high param cost)
- **Source/Inspiration**: arXiv:2410.05258

---

### Strategy T3-04: Mixture of Depths (MoD)
- **Category**: architecture
- **Legality**: LEGAL -- architecture design
- **Parameter Budget Impact**: ~1K params per layer for router
- **Core Idea**: Per-layer router skips "easy" tokens through some layers. Budget parameter B controls skip fraction. Reduces eval compute for given depth, enabling longer windows or more SLOT steps.
- **Implementation Plan**:
  1. Add router to each block:
     ```python
     self.router = nn.Linear(model_dim, 1)
     # In forward:
     route_score = self.router(x).sigmoid()  # [B, T, 1]
     if route_score.mean() > budget:
         # Process all tokens normally
         x = x + block(x)
     else:
         # Skip easy tokens
         mask = route_score > threshold
         x_hard = x[mask]
         x_hard = x_hard + block(x_hard)
         x[mask] = x_hard
     ```
  2. Budget B: 0.5 (skip 50% of tokens per layer)
  3. 3 seeds
- **Expected Outcome**: -0.002 to -0.008 BPB (faster eval -> longer windows)
- **Compute Estimate**: 36 min
- **Priority**: LOW
- **Combines With**: T1-02 (SLOT), T1-10 (TTT)
- **Source/Inspiration**: arXiv:2404.02258

---

### Strategy T3-05: DCMHA (Dynamically Composable Multi-Head Attention)
- **Category**: architecture
- **Legality**: LEGAL -- attention mechanism
- **Parameter Budget Impact**: Few KB for 11L
- **Core Idea**: Input-dependent transforms on attention score/weight matrices. Matches 1.7-2x compute models at 405M. Few KB params for 11L.
- **Implementation Plan**: Implement per arXiv:2405.08553
- **Expected Outcome**: -0.005 to -0.015 BPB
- **Compute Estimate**: 36 min
- **Priority**: LOW (moderate-high complexity)
- **Combines With**: T1-07 (XSA), T1-04 (QK-Gain)
- **Source/Inspiration**: arXiv:2405.08553 (ICML 2024 Oral)

---

### Strategy T3-06: VPTQ (Vector Post-Training Quantization)
- **Category**: quantization
- **Legality**: LEGAL -- post-training quantization
- **Parameter Budget Impact**: Potentially 2-3 bits per weight (vs 5-6 current)
- **Core Idea**: Vector PTQ guided by second-order Hessian. Beats GPTQ by 0.01-0.34 ppl at 2-3 bits. Practical within 600s budget.
- **Implementation Plan**: Implement VPTQ from arXiv:2409.17066
- **Expected Outcome**: -0.002 to -0.006 BPB
- **Compute Estimate**: 2 hours (implementation + testing)
- **Priority**: LOW
- **Combines With**: T1-05 (alternative to GPTQ)
- **Source/Inspiration**: arXiv:2409.17066 (EMNLP 2024)

---

### Strategy T3-07: Larger Model at 2-3 Bit Quantization
- **Category**: architecture/quantization
- **Legality**: LEGAL -- model design choice
- **Parameter Budget Impact**: 40-50M params at 3-bit fit in 16MB
- **Core Idea**: "Train larger, quantize harder." A 14L/640d model at int3 could outperform 11L/512d at int6. Requires int3 GPTQ infrastructure.
- **Implementation Plan**:
  1. Scale model: 14L, 640d, MLP 3x (= ~45M params)
  2. Implement int3 quantization (3 bits = 8 levels per weight)
  3. Apply aggressive QAT from 50% of training
  4. GPTQ at int3 with self-gen calibration
  5. Check artifact fits in 16MB
- **Expected Outcome**: UNCERTAIN (high risk, high reward: -0.010 to -0.030 BPB)
- **Compute Estimate**: 2 hours (implementation + 3 seeds)
- **Priority**: LOW
- **Combines With**: T1-06 (Brotli), T2-30 (HESTIA)
- **Source/Inspiration**: #641 (binary/ternary experiments), parametergolfanalyzer.md

---

### Strategy T3-08: Hypernetwork Weight Generation
- **Category**: architecture
- **Legality**: LEGAL -- #336 submitted prototype
- **Parameter Budget Impact**: 9.34x compression (2.8M generates 26.5M)
- **Core Idea**: Shared-trunk MLP generates full GPT weights from compact conditioning vectors. Highest compression-ratio approach seen (9.34x). No BPB result yet from #336.
- **Implementation Plan**: Implement per #336 (hypernet prototype)
- **Expected Outcome**: UNKNOWN (proof-of-concept stage)
- **Compute Estimate**: 4+ hours
- **Priority**: LOW
- **Combines With**: Limited (replaces standard model)
- **Source/Inspiration**: #336 (@jackopenn)

---

### Strategy T3-09: Hymba Hybrid Attention + Mamba SSM
- **Category**: architecture
- **Legality**: LEGAL -- #599 submitted (1.1828 BPB, non-record)
- **Parameter Budget Impact**: Similar to transformer
- **Core Idea**: Hybrid architecture with parallel attention and SSM branches. SSM makes each layer more powerful, so 7L beats deeper pure transformers at same step budget.
- **Implementation Plan**: Fork #599 implementation, improve with current stack
- **Expected Outcome**: -0.005 to -0.015 BPB (speculative)
- **Compute Estimate**: 4+ hours
- **Priority**: LOW
- **Combines With**: Limited (different architecture)
- **Source/Inspiration**: #599 (@mkenney2, 1.1828)

---

### Strategy T3-10: Window Attention Training (Mixed Seq_Len)
- **Category**: training/systems
- **Legality**: LEGAL -- #1212 submitted
- **Parameter Budget Impact**: Zero
- **Core Idea**: Sliding window of 512 on even layers during training (21% faster). Different GPUs train at different sequence lengths (5x2048 + 3x6144).
- **Implementation Plan**: Implement per #1212 (@Gusanidas)
- **Expected Outcome**: +15-21% more steps, -0.002 to -0.008 BPB
- **Compute Estimate**: 36 min
- **Priority**: LOW
- **Combines With**: T2-05 (progressive window)
- **Source/Inspiration**: #1212

---

### Strategy T3-11: Soft Quantization via Weight Coupling
- **Category**: quantization/training
- **Legality**: LEGAL -- training regularizer
- **Parameter Budget Impact**: Zero
- **Core Idea**: Physics-inspired coupling regularizer pulls weights toward discrete clusters during training. No STE needed -- weights self-discretize.
- **Implementation Plan**:
  1. Add coupling loss term:
     ```python
     coupling_loss = 0
     for p in model.parameters():
         if p.ndim >= 2:
             grid = torch.linspace(-32, 31, 64, device=p.device)
             dists = (p.unsqueeze(-1) - grid).abs().min(-1).values
             coupling_loss += dists.mean()
     total_loss = ce_loss + 0.01 * coupling_loss
     ```
  2. Anneal coupling strength from 0 to 0.1 over training
  3. 3 seeds
- **Expected Outcome**: -0.001 to -0.003 BPB
- **Compute Estimate**: 36 min
- **Priority**: LOW
- **Combines With**: All strategies
- **Source/Inspiration**: arXiv:2601.21219

---

### Strategy T3-12: Anti-Layer Removal Analysis
- **Category**: architecture
- **Legality**: LEGAL -- zero-cost ablation on existing checkpoint
- **Parameter Budget Impact**: Saves 1-2 layers worth of params if anti-layers found
- **Core Idea**: Some layers are "anti-layers" whose removal IMPROVES performance. Run ablation pass on trained checkpoint: remove each layer one at a time, measure BPB.
- **Implementation Plan**:
  1. After training, for each layer i in range(11):
     ```python
     model_ablated = remove_layer(model, i)
     bpb = eval_val_sliding(model_ablated)
     print(f"Layer {i} removed: BPB = {bpb}")
     ```
  2. If any removal improves BPB: remove that layer, reinvest params
  3. Zero training cost (uses existing checkpoint)
- **Expected Outcome**: -0.002 to -0.006 BPB (if anti-layers exist)
- **Compute Estimate**: 11 evals * 5 min = 55 min (no retraining)
- **Priority**: LOW
- **Combines With**: All strategies (post-hoc analysis)
- **Source/Inspiration**: arXiv:2603.19348

---

### Strategy T3-13: Late-Stage SAM (Sharpness-Aware Minimization)
- **Category**: training
- **Legality**: LEGAL -- optimizer modification
- **Parameter Budget Impact**: Zero
- **Core Idea**: Apply SAM in last 5-10% of training for flatter minima. Flatter minima complement EMA and improve quantization robustness.
- **Implementation Plan**:
  1. In last 10% of steps, add SAM perturbation:
     ```python
     if step > 0.9 * total_steps:
         # SAM: perturb weights, compute gradient, restore, update
         with torch.no_grad():
             for p in model.parameters():
                 if p.grad is not None:
                     eps = 0.05 * p.grad.sign()
                     p.add_(eps)
         loss = model(x, y)
         loss.backward()
         with torch.no_grad():
             for p in model.parameters():
                 if p.grad is not None:
                     p.sub_(eps)
         optimizer.step()
     ```
  2. SAM rho sweep: {0.02, 0.05, 0.10}
  3. 3 seeds
- **Expected Outcome**: -0.002 to -0.005 BPB
- **Compute Estimate**: 3 * 3 * 12 = 108 min
- **Priority**: LOW
- **Combines With**: All strategies
- **Source/Inspiration**: arXiv:2410.10373

---

### Strategy T3-14: TEON Cross-Layer Muon
- **Category**: optimizer
- **Legality**: LEGAL -- optimizer modification
- **Parameter Budget Impact**: Zero
- **Core Idea**: Joint tensor orthogonalization across ALL layers (vs Muon's per-layer NS). Captures inter-layer gradient relationships.
- **Implementation Plan**: Implement per arXiv:2601.23261
- **Expected Outcome**: -0.003 to -0.007 BPB
- **Compute Estimate**: 108 min
- **Priority**: LOW
- **Combines With**: All except other Muon variants
- **Source/Inspiration**: arXiv:2601.23261

---

### Strategy T3-15: YAQA Adaptive Rounding (Drop-in GPTQ Replacement)
- **Category**: quantization
- **Legality**: LEGAL -- post-training quantization
- **Parameter Budget Impact**: Zero
- **Core Idea**: Optimizes rounding toward full model's KL divergence (not just per-layer error) via Kronecker-factored Hessian. ~30% less quantization error than GPTQ.
- **Implementation Plan**: Implement YAQA from arXiv:2505.22988
- **Expected Outcome**: -0.001 to -0.003 BPB
- **Compute Estimate**: 2+ hours (implementation)
- **Priority**: LOW
- **Combines With**: All strategies (replaces GPTQ step)
- **Source/Inspiration**: arXiv:2505.22988

---

### Strategy T3-16: HybridNorm (Mixed Pre/Post-Norm)
- **Category**: architecture
- **Legality**: LEGAL -- normalization change
- **Parameter Budget Impact**: Zero
- **Core Idea**: Mixed Pre/Post-Norm for better depth utilization. First layers use Post-Norm (better gradient flow), later layers use Pre-Norm (more stable).
- **Implementation Plan**:
  1. Layers 0-5: Post-Norm (norm after residual addition)
  2. Layers 6-10: Pre-Norm (norm before attention/MLP)
  3. 3 seeds
- **Expected Outcome**: -0.002 to -0.006 BPB
- **Compute Estimate**: 36 min
- **Priority**: LOW
- **Combines With**: All strategies
- **Source/Inspiration**: arXiv:2503.04598

---

### Strategy T3-17: CUTLASS EVT Backward MLP Fusion
- **Category**: systems
- **Legality**: LEGAL -- kernel optimization, significance waived
- **Parameter Budget Impact**: Zero
- **Core Idea**: Fuse `(grad @ W_down) * act_grad` into GEMM epilogue via CUTLASS Epilogue Visitor Tree. Intermediate never touches HBM. Hopper-only. Used in #1105 for -3.7% step time.
- **Implementation Plan**: Port CUTLASS EVT fusion from #1105
- **Expected Outcome**: +500 extra steps, -0.001 to -0.003 BPB
- **Compute Estimate**: 4+ hours (kernel development)
- **Priority**: LOW
- **Combines With**: All strategies
- **Source/Inspiration**: #1105 (@abaybektursun)

---

### Strategy T3-18: Gravity Tokenizer (Ablation-Leverage Scoring)
- **Category**: tokenizer
- **Legality**: LIKELY_LEGAL -- #755 reached 1.0321 BPB
- **Parameter Budget Impact**: Depends on final vocab size
- **Core Idea**: Replace BPE merge tokens using ablation-leverage scoring -- measure each token's contribution to BPB and remove unhelpful ones. 12L 384d vanilla model, no standard stack.
- **Implementation Plan**: Study and adapt #755's approach
- **Expected Outcome**: -0.005 to -0.020 BPB (especially at small model scale)
- **Compute Estimate**: 4+ hours (tokenizer development)
- **Priority**: LOW
- **Combines With**: T1-12, T2-29
- **Source/Inspiration**: #755 (@dcrow85, 1.0321)

---

### Strategy T3-19: qTTT (Query-Only Test-Time Training)
- **Category**: eval-time
- **Legality**: GRAY_AREA -- TTT variant; must be backward-looking and GPTQ in training budget
- **Parameter Budget Impact**: Zero artifact
- **Core Idea**: Cache K/V once, adapt only Q projection weights during eval. 2-3x more TTT epochs within eval budget since K/V computation is skipped.
- **Implementation Plan**:
  1. During eval, freeze all params except Q projections
  2. Cache K, V for the current window
  3. Run 5-10 AdamW epochs adapting Q only on scored tokens
  4. Use cosine LR schedule
  5. Requires GPTQ within training budget
- **Expected Outcome**: -0.003 to -0.010 BPB
- **Compute Estimate**: 36 min
- **Priority**: LOW
- **Combines With**: T1-01, T1-04, NOT with T1-02 (SLOT replacement)
- **Source/Inspiration**: arXiv:2512.13898

---

### Strategy T3-20: LaCT (Large Chunk TTT)
- **Category**: eval-time
- **Legality**: GRAY_AREA -- same TTT concerns apply
- **Parameter Budget Impact**: Zero artifact
- **Core Idea**: Document-sized chunks for TTT -> 70% GPU utilization (vs <5% for per-token). Uses Muon as fast-weight optimizer.
- **Implementation Plan**: Implement per arXiv:2505.23884
- **Expected Outcome**: -0.002 to -0.008 BPB over standard TTT
- **Compute Estimate**: 2 hours (implementation)
- **Priority**: LOW
- **Combines With**: T1-10 (variant)
- **Source/Inspiration**: arXiv:2505.23884 (ICLR 2026 Oral)

---

### Strategy T3-21: ScaleBITS (Automated Per-Layer Bit-Width Search)
- **Category**: quantization
- **Legality**: LEGAL -- post-training analysis
- **Parameter Budget Impact**: Optimized bit allocation
- **Core Idea**: Automated sensitivity analysis + greedy optimization to determine which layers get int5 vs int6 under the 16MB constraint. Replaces manual knapsack from autopsy.
- **Implementation Plan**: Implement per arXiv:2602.17698
- **Expected Outcome**: -0.002 to -0.006 BPB over manual allocation
- **Compute Estimate**: +30 min analysis per checkpoint
- **Priority**: LOW
- **Combines With**: T1-05 (enhanced version)
- **Source/Inspiration**: arXiv:2602.17698

---

### Strategy T3-22: Text Diffusion (MDLM)
- **Category**: architecture
- **Legality**: LEGAL -- README wishlist item
- **Parameter Budget Impact**: Different architecture entirely
- **Core Idea**: First discrete diffusion to beat AR baseline. Bidirectional attention enables better global context. #1100 reached 1.1465 BPP.
- **Implementation Plan**: Extend #1100's implementation with current stack
- **Expected Outcome**: UNCERTAIN (1.1465 current, unknown ceiling)
- **Compute Estimate**: 4+ hours
- **Priority**: LOW (exploratory, README wishlist)
- **Combines With**: Limited (non-AR architecture)
- **Source/Inspiration**: #1100 (@agalimova, 1.1465), #1053

---

### Strategy T3-23: Cyclic Muon Momentum (Triangle Wave)
- **Category**: optimizer
- **Legality**: LEGAL -- training schedule
- **Parameter Budget Impact**: Zero
- **Core Idea**: Replace constant momentum with triangle wave cycling between 0.85-0.95 with period=50. May help escape local minima.
- **Implementation Plan**:
  1. In optimizer step:
     ```python
     cycle_pos = (step % 50) / 50  # 0 to 1
     momentum = 0.85 + 0.10 * (1 - abs(2 * cycle_pos - 1))  # triangle wave
     ```
  2. 3 seeds
- **Expected Outcome**: -0.001 to -0.003 BPB
- **Compute Estimate**: 36 min
- **Priority**: LOW
- **Combines With**: All strategies
- **Source/Inspiration**: #623 (@SPThole)

---

### Strategy T3-24: MiLe Loss (Entropy-Weighted Token Loss)
- **Category**: training
- **Legality**: LEGAL -- loss function change
- **Parameter Budget Impact**: Zero
- **Core Idea**: Entropy-weighted token loss with gamma=1.1 decaying to 0 during warmdown. Upweights uncertain tokens early, converges to standard CE.
- **Implementation Plan**:
  1. Compute token-level entropy weight:
     ```python
     probs = F.softmax(logits, dim=-1)
     entropy = -(probs * probs.log()).sum(-1)
     gamma = 1.1 * lr_scale  # decay with warmdown
     weights = 1.0 + gamma * entropy
     weighted_loss = (weights * token_losses).mean()
     ```
  2. 3 seeds
- **Expected Outcome**: -0.001 to -0.005 BPB
- **Compute Estimate**: 36 min
- **Priority**: LOW
- **Combines With**: T2-10 (alternative loss weighting)
- **Source/Inspiration**: #703 (@Gusanidas)

---

### Strategy T3-25: Coprime-Stride + Mixed Seq_Len Training
- **Category**: systems/training
- **Legality**: LEGAL -- training optimization
- **Parameter Budget Impact**: Zero
- **Core Idea**: Combine coprime-stride data loading with mixed sequence lengths across GPUs: 5 ranks train at seq2048, 3 ranks at seq6144. Diversity + long-context capability.
- **Implementation Plan**: Implement per #1212 with coprime-stride from T1-08
- **Expected Outcome**: -0.003 to -0.008 BPB
- **Compute Estimate**: 36 min
- **Priority**: LOW
- **Combines With**: T1-08 (coprime stride), T2-05 (progressive window)
- **Source/Inspiration**: #1212 (@Gusanidas)

---

## Combination Matrix

Strategies are grouped by compatibility. "+" means combinable, "-" means conflicting or redundant.

| Strategy | T1-01 Scylla | T1-02 SLOT | T1-04 QK-Gain | T1-05 MLP3.5 | T1-07 XSA-all | T1-10 TTT | T1-14 Mousse | T1-28 N-gram |
|----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| T1-01 Scylla | - | + | + | + | + | + | + | + |
| T1-02 SLOT | + | - | + | + | + | - | + | + |
| T1-04 QK-Gain | + | + | - | + | + | + | + | + |
| T1-05 MLP3.5 | + | + | + | - | + | + | + | + |
| T1-07 XSA-all | + | + | + | + | - | + | + | + |
| T1-10 TTT | + | - | + | + | + | - | + | + |
| T1-14 Mousse | + | + | + | + | + | + | - | + |
| T1-28 N-gram | + | + | + | + | + | + | + | - |

Key conflicts:
- T1-02 (SLOT) vs T1-10 (TTT): SLOT replaces TTT per #1172 finding. Pick one.
- T1-09 (IFNSO) vs T1-14 (Mousse): Both modify Muon. Test independently.
- T1-13 (4x MLP simplify) vs T1-05 (3.5x mixed quant): Different philosophies. Test independently.
- T1-17 (Hourglass) vs T1-05 (MLP 3.5x): Alternative MLP designs. Test independently.
- T1-24 (Depth recurrence) vs T2-11 (KV sharing): Both modify layer structure. Test independently.

---

## Execution Schedule

**Phase 1: Zero-Cost / Systems (Day 1, parallel)**
Run simultaneously on separate GPU nodes:
- T1-06: Brotli + byte-shuffle (minutes, apply to existing checkpoints)
- T1-08: Coprime-stride loader (drop-in, 3 seeds)
- T1-09: IFNSO (drop-in, 3 seeds)
- T1-11: Liger-Kernel (pip install, 3 seeds)
- T1-19: 1-sqrt cooldown (1-line change, 3 seeds)
- T3-12: Anti-layer removal (ablation on existing checkpoint)
Estimated time: 1-2 hours total

**Phase 2: High-Impact Architecture (Day 1-2, parallel)**
- T1-04: QK-Gain sweep (5 values, 3 seeds)
- T1-05: MLP 3.5x + mixed quant (3 seeds)
- T1-07: XSA-all (3 configs, 3 seeds)
- T1-15: DDL residual gates (3 seeds)
- T1-16: Hyper-Connections (3 seeds)
- T1-23: Sigmoid-gated skips (3 seeds)
- T1-24: Shallow depth recurrence (3 seeds)
Estimated time: 3-4 hours total

**Phase 3: Eval-Time Methods (Day 2, parallel)**
- T1-02: SLOT basic (sweep 4 lr values, 3 seeds each)
- T1-03: Per-Sample SLOT (3 seeds)
- T1-26: Context-Only SLOT (3 seeds)
- T1-10: Legal TTT (sweep 3 lr values, 3 seeds)
Estimated time: 2-3 hours total

**Phase 4: Tokenizer (Day 2-3, sequential)**
- T1-01: Scylla tokenizer + full stack (3 seeds)
- T1-12: Scylla + SLOT combination (3 seeds)
- T1-13: 4096-vocab simplification (3 seeds)
Estimated time: 3-4 hours

**Phase 5: Optimizer & Training (Day 3, parallel)**
- T1-14: Mousse optimizer (3 configs, 3 seeds)
- T1-18: AdamHD Huber decay (4 delta values, 3 seeds)
- T1-20: Batch size warmup (3 seeds)
- T1-25: EMA sweep (5 values, 3 seeds)
Estimated time: 3-4 hours

**Phase 6: Quantization (Day 3-4, sequential)**
- T1-21: Self-gen GPTQ calibration (3 seeds)
- T1-22: Training-data GPTQ calibration (3 seeds)
- T1-27: Prune-then-quantize ordering (2 orderings, 3 seeds)
- T2-12: Temperature scaling (sweep, minutes)
Estimated time: 2-3 hours

**Phase 7: Medium Priority (Day 4-6, parallel)**
Run all Tier 2 strategies in batches of 8 parallel experiments

**Phase 8: Best Combination Assembly (Day 7)**
Take the top 5-8 independent improvements from Phases 1-6, combine into a single submission. 3 seeds for statistical significance.

**Phase 9: Exploratory (Days 8-10)**
Run Tier 3 strategies on remaining compute budget

---

## Monitoring Criteria

**Kill an experiment early if:**
- Step 1000 val_loss is >5% worse than baseline (r=0.86 correlation to final BPB)
- Step time is >15% slower without corresponding per-step improvement
- Artifact size exceeds 17MB (no path to 16MB)
- NaN/Inf in training loss

**Double down on an experiment if:**
- Step 1000 val_loss is >2% better than baseline
- Step time is faster with no quality degradation
- Artifact is 1MB+ smaller than baseline (room for bigger model)

**Key metrics to track:**
- val_bpb (primary metric, lower is better)
- step_time_ms (throughput proxy)
- artifact_size_bytes (must be < 16,000,000)
- pre_quant_val_loss vs post_quant_val_bpb gap (quantization damage)
- 3-seed standard deviation (should be < 0.001 BPB)
- ECE (calibration, especially for mixed-quant strategies)

**Early signal checkpoints:**
- Step 500: check for catastrophic failures
- Step 1000: primary go/no-go decision (r=0.86 with final)
- Step 3000: mid-training check
- Step 5000: final-third check
- Step 7000: pre-warmdown baseline

---

## Glossary of Key Abbreviations

- BPB: Bits Per Byte
- BPP: Bits Per Byte (used interchangeably with BPB in competition)
- GPTQ: Gradient Post-Training Quantization
- XSA: Exclusive Self-Attention
- EMA: Exponential Moving Average
- SWA: Stochastic Weight Averaging
- TTT: Test-Time Training
- SLOT: Selective Logit Offset Tuning
- QAT: Quantization-Aware Training
- STE: Straight-Through Estimator
- NS: Newton-Schulz (iteration for Muon)
- WD: Weight Decay
- FA3: Flash Attention 3
- GQA: Grouped Query Attention
- MLP: Multi-Layer Perceptron (feedforward network)
- RoPE: Rotary Position Embedding
- DDL: Deep Delta Learning
- IFNSO: Iteration-Free Newton-Schulz Orthogonalization

---

## Dispatch Commands: Tier 1 Strategies

Each command below launches a 3-seed experiment on 8xH100. The base script is `experiment1.py`.
Replace `SEED=X` with 1337, 42, 7 for 3-seed runs. All env vars override `Hyperparameters` defaults.

### T1-01: Scylla Tokenizer
```bash
# Requires: Scylla tokenizer + retokenized FineWeb dataset
for SEED in 1337 42 7; do
RUN_ID=t1_01_scylla_s${SEED} \
SEED=$SEED \
DATA_PATH=./data/datasets/fineweb10B_scylla/ \
TOKENIZER_PATH=./data/tokenizers/scylla_998.model \
VOCAB_SIZE=998 \
NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 \
MLP_MULT=3.0 MUON_WD=0.04 XSA_LAST_N=11 \
SWA_ENABLED=0 GPTQ_ENABLED=1 GPTQ_N_BATCHES=32 \
BIGRAM_VOCAB_SIZE=2048 BIGRAM_DIM=128 \
torchrun --standalone --nproc_per_node=8 experiment1.py
done
```

### T1-02: SLOT (Basic)
```bash
# Requires: SLOT eval loop modification in experiment1.py
# Code change: add SLOT delta optimization in eval_val_sliding
for SEED in 1337 42 7; do
RUN_ID=t1_02_slot_s${SEED} \
SEED=$SEED \
SLOT_ENABLED=1 SLOT_LR=0.008 SLOT_STEPS=16 SLOT_PER_SAMPLE=0 \
torchrun --standalone --nproc_per_node=8 experiment1.py
done
```

### T1-03: Per-Sample SLOT
```bash
for SEED in 1337 42 7; do
RUN_ID=t1_03_slot_persample_s${SEED} \
SEED=$SEED \
SLOT_ENABLED=1 SLOT_LR=0.008 SLOT_STEPS=16 SLOT_PER_SAMPLE=1 \
torchrun --standalone --nproc_per_node=8 experiment1.py
done
```

### T1-04: QK-Gain Sweep
```bash
for GAIN in 2.0 3.0 4.0 5.0 6.0; do
for SEED in 1337 42 7; do
RUN_ID=t1_04_qkgain${GAIN}_s${SEED} \
SEED=$SEED \
QK_GAIN_INIT=$GAIN \
torchrun --standalone --nproc_per_node=8 experiment1.py
done
done
```

### T1-05: MLP 3.5x + Mixed Int5/Int6
```bash
for SEED in 1337 42 7; do
RUN_ID=t1_05_mlp35x_mixedquant_s${SEED} \
SEED=$SEED \
MLP_MULT=3.5 \
GPTQ_ENABLED=1 GPTQ_N_BATCHES=32 \
MIXED_QUANT_ENABLED=1 \
torchrun --standalone --nproc_per_node=8 experiment1.py
done
```

### T1-06: Brotli + Byte-Shuffle
```bash
# No retraining needed - apply to existing checkpoints
# Code change: replace zstd compression with brotli-11 + byte-shuffle in save_model
for SEED in 1337 42 7; do
RUN_ID=t1_06_brotli_s${SEED} \
SEED=$SEED \
COMPRESSION=brotli BROTLI_QUALITY=11 BYTE_SHUFFLE=1 \
torchrun --standalone --nproc_per_node=8 experiment1.py
done
```

### T1-07: XSA-All (11 Layers)
```bash
for N in 4 8 11; do
for SEED in 1337 42 7; do
RUN_ID=t1_07_xsa${N}_s${SEED} \
SEED=$SEED \
XSA_LAST_N=$N \
torchrun --standalone --nproc_per_node=8 experiment1.py
done
done
```

### T1-08: Coprime-Stride Loader
```bash
# Code change: modify TokenStream/DistributedTokenLoader with coprime stride
for SEED in 1337 42 7; do
RUN_ID=t1_08_coprime_s${SEED} \
SEED=$SEED \
COPRIME_STRIDE=1 \
torchrun --standalone --nproc_per_node=8 experiment1.py
done
```

### T1-09: IFNSO
```bash
# Code change: replace zeropower_via_newtonschulz5 with polynomial eval
for SEED in 1337 42 7; do
RUN_ID=t1_09_ifnso_s${SEED} \
SEED=$SEED \
MUON_BACKEND=ifnso \
torchrun --standalone --nproc_per_node=8 experiment1.py
done
```

### T1-10: Legal Score-First TTT
```bash
for LR in 0.0005 0.001 0.002; do
for SEED in 1337 42 7; do
RUN_ID=t1_10_ttt_lr${LR}_s${SEED} \
SEED=$SEED \
TTT_ENABLED=1 TTT_LR=$LR TTT_EPOCHS=3 \
TTT_CHUNK_TOKENS=32768 TTT_FREEZE_BLOCKS=2 \
TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 \
GPTQ_ENABLED=1 \
torchrun --standalone --nproc_per_node=8 experiment1.py
done
done
```

### T1-11: Liger-Kernel
```bash
# Requires: pip install liger-kernel
# Code change: replace RMSNorm and CE loss with Liger variants
for SEED in 1337 42 7; do
RUN_ID=t1_11_liger_s${SEED} \
SEED=$SEED \
LIGER_ENABLED=1 \
torchrun --standalone --nproc_per_node=8 experiment1.py
done
```

### T1-12: Scylla + SLOT
```bash
for SEED in 1337 42 7; do
RUN_ID=t1_12_scylla_slot_s${SEED} \
SEED=$SEED \
DATA_PATH=./data/datasets/fineweb10B_scylla/ \
TOKENIZER_PATH=./data/tokenizers/scylla_998.model \
VOCAB_SIZE=998 \
XSA_LAST_N=11 GPTQ_ENABLED=1 \
SLOT_ENABLED=1 SLOT_LR=0.008 SLOT_STEPS=16 SLOT_PER_SAMPLE=1 \
torchrun --standalone --nproc_per_node=8 experiment1.py
done
```

### T1-13: WD 0.085 + MLP 4x Simplification
```bash
for SEED in 1337 42 7; do
RUN_ID=t1_13_simplify_s${SEED} \
SEED=$SEED \
MLP_MULT=4.0 MUON_WD=0.085 ADAM_WD=0.085 \
BIGRAM_VOCAB_SIZE=0 \
XSA_LAST_N=11 GPTQ_ENABLED=1 \
torchrun --standalone --nproc_per_node=8 experiment1.py
done
```

### T1-14: Mousse Optimizer
```bash
for PRECOND in 10 25 50; do
for SEED in 1337 42 7; do
RUN_ID=t1_14_mousse_pi${PRECOND}_s${SEED} \
SEED=$SEED \
OPTIMIZER=mousse MOUSSE_PRECOND_INTERVAL=$PRECOND \
torchrun --standalone --nproc_per_node=8 experiment1.py
done
done
```

### T1-15: DDL Residual Gates
```bash
# Code change: add DDL u, v, beta parameters to Block class
for SEED in 1337 42 7; do
RUN_ID=t1_15_ddl_s${SEED} \
SEED=$SEED \
DDL_ENABLED=1 \
torchrun --standalone --nproc_per_node=8 experiment1.py
done
```

### T1-16: Hyper-Connections
```bash
# Code change: replace residual connection with learned mixing matrix in Block
for SEED in 1337 42 7; do
RUN_ID=t1_16_hyperconn_s${SEED} \
SEED=$SEED \
HYPERCONN_ENABLED=1 HYPERCONN_N=2 \
torchrun --standalone --nproc_per_node=8 experiment1.py
done
```

### T1-17: Hourglass FFN
```bash
# Code change: replace MLP class with HourglassMLP
for SUB_MULT in 1.5 2.0; do
for N_SUB in 2 3; do
for SEED in 1337 42 7; do
RUN_ID=t1_17_hourglass_m${SUB_MULT}_n${N_SUB}_s${SEED} \
SEED=$SEED \
MLP_TYPE=hourglass HOURGLASS_SUB_MULT=$SUB_MULT HOURGLASS_N_LAYERS=$N_SUB \
torchrun --standalone --nproc_per_node=8 experiment1.py
done
done
done
```

### T1-18: AdamHD Huber Decay
```bash
# Code change: modify Muon.step() weight decay to use Huber function
for DELTA in 0.05 0.1 0.2 0.5; do
for SEED in 1337 42 7; do
RUN_ID=t1_18_huber_d${DELTA}_s${SEED} \
SEED=$SEED \
WD_TYPE=huber HUBER_DELTA=$DELTA \
torchrun --standalone --nproc_per_node=8 experiment1.py
done
done
```

### T1-19: 1-sqrt Cooldown
```bash
# Code change: modify LR schedule warmdown shape
for SEED in 1337 42 7; do
RUN_ID=t1_19_sqrt_cooldown_s${SEED} \
SEED=$SEED \
WARMDOWN_SHAPE=sqrt \
torchrun --standalone --nproc_per_node=8 experiment1.py
done
```

### T1-20: Batch Size Warmup
```bash
# Code change: implement batch size ramp in training loop
for SEED in 1337 42 7; do
RUN_ID=t1_20_bswarmup_s${SEED} \
SEED=$SEED \
BATCH_WARMUP=1 BATCH_WARMUP_START=262144 BATCH_WARMUP_FRAC=0.3 \
torchrun --standalone --nproc_per_node=8 experiment1.py
done
```

### T1-21: Self-Gen GPTQ Calibration
```bash
for SEED in 1337 42 7; do
RUN_ID=t1_21_selfgen_gptq_s${SEED} \
SEED=$SEED \
GPTQ_ENABLED=1 GPTQ_CALIB=selfgen GPTQ_N_BATCHES=32 \
torchrun --standalone --nproc_per_node=8 experiment1.py
done
```

### T1-22: Training-Data GPTQ Calibration
```bash
for SEED in 1337 42 7; do
RUN_ID=t1_22_traindata_gptq_s${SEED} \
SEED=$SEED \
GPTQ_ENABLED=1 GPTQ_CALIB=training GPTQ_N_BATCHES=256 \
torchrun --standalone --nproc_per_node=8 experiment1.py
done
```

### T1-23: Sigmoid-Gated Skip Connections
```bash
# Code change: replace skip_weights with sigmoid-gated version in GPT.forward
for SEED in 1337 42 7; do
RUN_ID=t1_23_sigskip_s${SEED} \
SEED=$SEED \
SIGMOID_SKIP=1 \
torchrun --standalone --nproc_per_node=8 experiment1.py
done
```

### T1-24: Shallow Depth Recurrence
```bash
# Code change: repeat layers 4-5 in GPT.forward with learned scalars
for SEED in 1337 42 7; do
RUN_ID=t1_24_recur_s${SEED} \
SEED=$SEED \
DEPTH_RECURRENCE=1 RECUR_LAYERS=4,5 \
torchrun --standalone --nproc_per_node=8 experiment1.py
done
```

### T1-25: EMA Decay Sweep
```bash
for DECAY in 0.995 0.996 0.997 0.998 0.999; do
for SEED in 1337 42 7; do
RUN_ID=t1_25_ema${DECAY}_s${SEED} \
SEED=$SEED \
SWA_ENABLED=0 EMA_ENABLED=1 EMA_DECAY=$DECAY \
torchrun --standalone --nproc_per_node=8 experiment1.py
done
done
```

### T1-26: Context-Only SLOT
```bash
for SEED in 1337 42 7; do
RUN_ID=t1_26_slot_contextonly_s${SEED} \
SEED=$SEED \
SLOT_ENABLED=1 SLOT_CONTEXT_ONLY=1 SLOT_LR=0.008 SLOT_STEPS=16 \
torchrun --standalone --nproc_per_node=8 experiment1.py
done
```

### T1-27: Prune-Then-Quantize
```bash
for ORDER in quant_prune prune_quant; do
for SEED in 1337 42 7; do
RUN_ID=t1_27_${ORDER}_s${SEED} \
SEED=$SEED \
GPTQ_ENABLED=1 POST_PRUNE_ORDER=$ORDER PRUNE_FRAC=0.03 \
torchrun --standalone --nproc_per_node=8 experiment1.py
done
done
```

### T1-28: N-gram Backoff Cache
```bash
# Code change: add NgramCache class and integrate into eval_val_sliding
for SEED in 1337 42 7; do
RUN_ID=t1_28_ngram_s${SEED} \
SEED=$SEED \
NGRAM_CACHE=1 NGRAM_MAX_ORDER=9 NGRAM_BUCKETS=4000000 \
torchrun --standalone --nproc_per_node=8 experiment1.py
done
```

---

## Best Combined Submission Assembly Guide

After running Phases 1-6, assemble the final submission by stacking the winning strategies.
Use this decision tree:

### Step 1: Choose Tokenizer Track
- **If Scylla (T1-01) beats sp1024 by >0.02 BPB** -> Use Scylla as base
- **Else** -> Stay on sp1024

### Step 2: Pick Eval-Time Method (mutually exclusive)
- **If SLOT (T1-02/T1-03) delta > TTT (T1-10) delta** -> Use SLOT (SLOT replaces TTT per #1172)
- **Else** -> Use TTT
- **If both help** -> Try SLOT only (it's simpler and more robust)

### Step 3: Stack Architecture Wins (additive, verify no conflicts)
Combine all architecture changes that showed >0.001 BPB improvement in isolation:
- QK-Gain (T1-04) — zero-cost, always include
- XSA coverage (T1-07) — use the best N from sweep
- DDL gates (T1-15) — if positive
- Hyper-Connections (T1-16) — if positive, but test with DDL jointly
- Sigmoid skips (T1-23) — near-zero cost, include if positive
- Shallow recurrence (T1-24) — if positive, but watch step time

### Step 4: Stack Training/Optimizer Wins
- Best cooldown shape (T1-19)
- Batch warmup (T1-20) — if positive
- Best EMA decay (T1-25)
- Mousse OR IFNSO (T1-14 or T1-09) — whichever was better, not both
- Huber WD (T1-18) — if positive

### Step 5: Apply Best Quantization Pipeline
Order: prune-then-quantize (T1-27) if it won -> then best GPTQ calibration method (T1-21 vs T1-22) -> Brotli+shuffle (T1-06) -> mixed int5/int6 (T1-05) if artifact room

### Step 6: Add Eval-Time Enhancements
- Temperature scaling (T2-12) if mixed quant used
- N-gram cache (T1-28) if legality confirmed and delta is large

### Step 7: Validate Combined Submission
```bash
# Run 5 seeds for final statistical significance
for SEED in 1337 42 7 314 2718; do
RUN_ID=final_combined_s${SEED} \
SEED=$SEED \
[ALL WINNING ENV VARS FROM STEPS 1-6] \
torchrun --standalone --nproc_per_node=8 experiment1.py
done
```
Check: `p < 0.01` that score beats current SOTA by >0.005 nats.

### Decision Matrix for Common Scenarios

| Scenario | Best Path |
|----------|-----------|
| Scylla works + SLOT works | T1-01 + T1-03 + T1-04 + T1-07 + T1-06 = target 0.92-0.94 |
| Scylla works, SLOT neutral | T1-01 + T1-10 (TTT) + T1-04 + T1-07 + T1-06 = target 0.95-0.98 |
| sp1024 only, SLOT works | T1-03 + T1-04 + T1-05 + T1-07 + T1-06 + architecture wins = target 1.08-1.10 |
| sp1024, simplification wins | T1-13 (WD 0.085 + MLP 4x) + T1-02 + T1-04 + T1-06 = target 1.07-1.09 |
| N-gram cache confirmed legal | Add T1-28 to ANY of above for additional -0.05 to -0.15 BPB |

---

## Machine-Readable Experiment Manifest

```json
{
  "manifest_version": "1.0",
  "generated": "2026-04-04",
  "base_script": "experiment1.py",
  "base_command": "torchrun --standalone --nproc_per_node=8 experiment1.py",
  "seeds": [1337, 42, 7],
  "default_env": {
    "DATA_PATH": "./data/datasets/fineweb10B_sp1024",
    "TOKENIZER_PATH": "./data/tokenizers/fineweb_1024_bpe.model",
    "VOCAB_SIZE": "1024",
    "NUM_LAYERS": "11",
    "MODEL_DIM": "512",
    "NUM_HEADS": "8",
    "NUM_KV_HEADS": "4",
    "MLP_MULT": "3.0",
    "MUON_WD": "0.04",
    "XSA_LAST_N": "4",
    "GPTQ_ENABLED": "1",
    "SWA_ENABLED": "1",
    "BIGRAM_VOCAB_SIZE": "2048",
    "BIGRAM_DIM": "128",
    "ROPE_DIMS": "16",
    "LN_SCALE": "1",
    "VE_ENABLED": "1",
    "TRAIN_SEQ_LEN": "2048",
    "EVAL_SEQ_LEN": "2048",
    "EVAL_STRIDE": "64"
  },
  "experiments": [
    {
      "id": "T1-01",
      "name": "Scylla Tokenizer",
      "tier": 1,
      "category": "tokenizer",
      "legality": "LIKELY_LEGAL",
      "expected_bpb_delta": [-0.02, -0.06],
      "requires_code_change": true,
      "env_overrides": {
        "DATA_PATH": "./data/datasets/fineweb10B_scylla/",
        "TOKENIZER_PATH": "./data/tokenizers/scylla_998.model",
        "VOCAB_SIZE": "998",
        "XSA_LAST_N": "11"
      },
      "phase": 4
    },
    {
      "id": "T1-02",
      "name": "SLOT Basic",
      "tier": 1,
      "category": "eval-time",
      "legality": "LEGAL",
      "expected_bpb_delta": [-0.015, -0.025],
      "requires_code_change": true,
      "env_overrides": {
        "SLOT_ENABLED": "1",
        "SLOT_LR": "0.008",
        "SLOT_STEPS": "16",
        "SLOT_PER_SAMPLE": "0"
      },
      "phase": 3
    },
    {
      "id": "T1-03",
      "name": "Per-Sample SLOT",
      "tier": 1,
      "category": "eval-time",
      "legality": "LEGAL",
      "expected_bpb_delta": [-0.020, -0.030],
      "requires_code_change": true,
      "env_overrides": {
        "SLOT_ENABLED": "1",
        "SLOT_LR": "0.008",
        "SLOT_STEPS": "16",
        "SLOT_PER_SAMPLE": "1"
      },
      "phase": 3
    },
    {
      "id": "T1-04",
      "name": "QK-Gain Sweep",
      "tier": 1,
      "category": "architecture",
      "legality": "LEGAL",
      "expected_bpb_delta": [-0.003, -0.006],
      "requires_code_change": false,
      "sweep": {"QK_GAIN_INIT": ["2.0", "3.0", "4.0", "5.0", "6.0"]},
      "env_overrides": {},
      "phase": 2
    },
    {
      "id": "T1-05",
      "name": "MLP 3.5x + Mixed Int5/Int6",
      "tier": 1,
      "category": "quantization",
      "legality": "LEGAL",
      "expected_bpb_delta": [-0.003, -0.008],
      "requires_code_change": true,
      "env_overrides": {
        "MLP_MULT": "3.5",
        "MIXED_QUANT_ENABLED": "1"
      },
      "phase": 2
    },
    {
      "id": "T1-06",
      "name": "Brotli + Byte-Shuffle",
      "tier": 1,
      "category": "compression",
      "legality": "LEGAL",
      "expected_bpb_delta": [0, -0.005],
      "requires_code_change": true,
      "env_overrides": {
        "COMPRESSION": "brotli",
        "BROTLI_QUALITY": "11",
        "BYTE_SHUFFLE": "1"
      },
      "phase": 1
    },
    {
      "id": "T1-07",
      "name": "XSA-All Sweep",
      "tier": 1,
      "category": "architecture",
      "legality": "LEGAL",
      "expected_bpb_delta": [-0.003, -0.006],
      "requires_code_change": false,
      "sweep": {"XSA_LAST_N": ["4", "8", "11"]},
      "env_overrides": {},
      "phase": 2
    },
    {
      "id": "T1-08",
      "name": "Coprime-Stride Loader",
      "tier": 1,
      "category": "systems",
      "legality": "LEGAL",
      "expected_bpb_delta": [-0.001, -0.003],
      "requires_code_change": true,
      "env_overrides": {"COPRIME_STRIDE": "1"},
      "phase": 1
    },
    {
      "id": "T1-09",
      "name": "IFNSO",
      "tier": 1,
      "category": "systems",
      "legality": "LEGAL",
      "expected_bpb_delta": [-0.001, -0.003],
      "requires_code_change": true,
      "env_overrides": {"MUON_BACKEND": "ifnso"},
      "phase": 1
    },
    {
      "id": "T1-10",
      "name": "Legal TTT Sweep",
      "tier": 1,
      "category": "eval-time",
      "legality": "LEGAL",
      "expected_bpb_delta": [-0.005, -0.015],
      "requires_code_change": false,
      "sweep": {"TTT_LR": ["0.0005", "0.001", "0.002"]},
      "env_overrides": {
        "TTT_ENABLED": "1",
        "TTT_EPOCHS": "3",
        "TTT_CHUNK_TOKENS": "32768",
        "TTT_FREEZE_BLOCKS": "2"
      },
      "phase": 3
    },
    {
      "id": "T1-11",
      "name": "Liger-Kernel",
      "tier": 1,
      "category": "systems",
      "legality": "LEGAL",
      "expected_bpb_delta": [-0.001, -0.005],
      "requires_code_change": true,
      "env_overrides": {"LIGER_ENABLED": "1"},
      "phase": 1
    },
    {
      "id": "T1-12",
      "name": "Scylla + SLOT",
      "tier": 1,
      "category": "hybrid",
      "legality": "LIKELY_LEGAL",
      "expected_bpb_delta": [-0.05, -0.10],
      "requires_code_change": true,
      "env_overrides": {
        "DATA_PATH": "./data/datasets/fineweb10B_scylla/",
        "TOKENIZER_PATH": "./data/tokenizers/scylla_998.model",
        "VOCAB_SIZE": "998",
        "XSA_LAST_N": "11",
        "SLOT_ENABLED": "1",
        "SLOT_LR": "0.008",
        "SLOT_STEPS": "16",
        "SLOT_PER_SAMPLE": "1"
      },
      "phase": 4
    },
    {
      "id": "T1-13",
      "name": "WD 0.085 + MLP 4x Simplification",
      "tier": 1,
      "category": "architecture",
      "legality": "LEGAL",
      "expected_bpb_delta": [-0.01, -0.03],
      "requires_code_change": false,
      "env_overrides": {
        "MLP_MULT": "4.0",
        "MUON_WD": "0.085",
        "ADAM_WD": "0.085",
        "BIGRAM_VOCAB_SIZE": "0",
        "XSA_LAST_N": "11"
      },
      "phase": 4
    },
    {
      "id": "T1-14",
      "name": "Mousse Optimizer",
      "tier": 1,
      "category": "optimizer",
      "legality": "LEGAL",
      "expected_bpb_delta": [-0.003, -0.008],
      "requires_code_change": true,
      "sweep": {"MOUSSE_PRECOND_INTERVAL": ["10", "25", "50"]},
      "env_overrides": {"OPTIMIZER": "mousse"},
      "phase": 5
    },
    {
      "id": "T1-15",
      "name": "DDL Residual Gates",
      "tier": 1,
      "category": "architecture",
      "legality": "LEGAL",
      "expected_bpb_delta": [-0.003, -0.007],
      "requires_code_change": true,
      "env_overrides": {"DDL_ENABLED": "1"},
      "phase": 2
    },
    {
      "id": "T1-16",
      "name": "Hyper-Connections",
      "tier": 1,
      "category": "architecture",
      "legality": "LEGAL",
      "expected_bpb_delta": [-0.003, -0.008],
      "requires_code_change": true,
      "env_overrides": {"HYPERCONN_ENABLED": "1", "HYPERCONN_N": "2"},
      "phase": 2
    },
    {
      "id": "T1-17",
      "name": "Hourglass FFN",
      "tier": 1,
      "category": "architecture",
      "legality": "LEGAL",
      "expected_bpb_delta": [-0.002, -0.006],
      "requires_code_change": true,
      "sweep": {
        "HOURGLASS_SUB_MULT": ["1.5", "2.0"],
        "HOURGLASS_N_LAYERS": ["2", "3"]
      },
      "env_overrides": {"MLP_TYPE": "hourglass"},
      "phase": 2
    },
    {
      "id": "T1-18",
      "name": "AdamHD Huber Decay",
      "tier": 1,
      "category": "training",
      "legality": "LEGAL",
      "expected_bpb_delta": [-0.002, -0.005],
      "requires_code_change": true,
      "sweep": {"HUBER_DELTA": ["0.05", "0.1", "0.2", "0.5"]},
      "env_overrides": {"WD_TYPE": "huber"},
      "phase": 5
    },
    {
      "id": "T1-19",
      "name": "1-sqrt Cooldown",
      "tier": 1,
      "category": "training",
      "legality": "LEGAL",
      "expected_bpb_delta": [-0.001, -0.003],
      "requires_code_change": true,
      "env_overrides": {"WARMDOWN_SHAPE": "sqrt"},
      "phase": 1
    },
    {
      "id": "T1-20",
      "name": "Batch Size Warmup",
      "tier": 1,
      "category": "training",
      "legality": "LEGAL",
      "expected_bpb_delta": [-0.002, -0.005],
      "requires_code_change": true,
      "env_overrides": {
        "BATCH_WARMUP": "1",
        "BATCH_WARMUP_START": "262144",
        "BATCH_WARMUP_FRAC": "0.3"
      },
      "phase": 5
    },
    {
      "id": "T1-21",
      "name": "Self-Gen GPTQ Calibration",
      "tier": 1,
      "category": "quantization",
      "legality": "LEGAL",
      "expected_bpb_delta": [-0.002, -0.004],
      "requires_code_change": true,
      "env_overrides": {"GPTQ_CALIB": "selfgen", "GPTQ_N_BATCHES": "32"},
      "phase": 6
    },
    {
      "id": "T1-22",
      "name": "Training-Data GPTQ Calibration",
      "tier": 1,
      "category": "quantization",
      "legality": "LEGAL",
      "expected_bpb_delta": [-0.001, -0.003],
      "requires_code_change": true,
      "env_overrides": {"GPTQ_CALIB": "training", "GPTQ_N_BATCHES": "256"},
      "phase": 6
    },
    {
      "id": "T1-23",
      "name": "Sigmoid-Gated Skips",
      "tier": 1,
      "category": "architecture",
      "legality": "LEGAL",
      "expected_bpb_delta": [-0.001, -0.002],
      "requires_code_change": true,
      "env_overrides": {"SIGMOID_SKIP": "1"},
      "phase": 2
    },
    {
      "id": "T1-24",
      "name": "Shallow Depth Recurrence",
      "tier": 1,
      "category": "architecture",
      "legality": "LEGAL",
      "expected_bpb_delta": [-0.003, -0.008],
      "requires_code_change": true,
      "env_overrides": {"DEPTH_RECURRENCE": "1", "RECUR_LAYERS": "4,5"},
      "phase": 2
    },
    {
      "id": "T1-25",
      "name": "EMA Decay Sweep",
      "tier": 1,
      "category": "training",
      "legality": "LEGAL",
      "expected_bpb_delta": [-0.001, -0.003],
      "requires_code_change": false,
      "sweep": {"EMA_DECAY": ["0.995", "0.996", "0.997", "0.998", "0.999"]},
      "env_overrides": {"SWA_ENABLED": "0", "EMA_ENABLED": "1"},
      "phase": 5
    },
    {
      "id": "T1-26",
      "name": "Context-Only SLOT",
      "tier": 1,
      "category": "eval-time",
      "legality": "LEGAL",
      "expected_bpb_delta": [-0.015, -0.025],
      "requires_code_change": true,
      "env_overrides": {
        "SLOT_ENABLED": "1",
        "SLOT_CONTEXT_ONLY": "1",
        "SLOT_LR": "0.008",
        "SLOT_STEPS": "16"
      },
      "phase": 3
    },
    {
      "id": "T1-27",
      "name": "Prune-Then-Quantize",
      "tier": 1,
      "category": "quantization",
      "legality": "LEGAL",
      "expected_bpb_delta": [-0.001, -0.003],
      "requires_code_change": true,
      "sweep": {"POST_PRUNE_ORDER": ["quant_prune", "prune_quant"]},
      "env_overrides": {"PRUNE_FRAC": "0.03"},
      "phase": 6
    },
    {
      "id": "T1-28",
      "name": "N-gram Backoff Cache",
      "tier": 1,
      "category": "eval-time",
      "legality": "GRAY_AREA",
      "expected_bpb_delta": [-0.05, -0.15],
      "requires_code_change": true,
      "env_overrides": {
        "NGRAM_CACHE": "1",
        "NGRAM_MAX_ORDER": "9",
        "NGRAM_BUCKETS": "4000000"
      },
      "phase": 3
    }
  ],
  "phases": {
    "1": {"name": "Zero-Cost / Systems", "parallel": true, "experiments": ["T1-06", "T1-08", "T1-09", "T1-11", "T1-19"]},
    "2": {"name": "High-Impact Architecture", "parallel": true, "experiments": ["T1-04", "T1-05", "T1-07", "T1-15", "T1-16", "T1-17", "T1-23", "T1-24"]},
    "3": {"name": "Eval-Time Methods", "parallel": true, "experiments": ["T1-02", "T1-03", "T1-10", "T1-26", "T1-28"]},
    "4": {"name": "Tokenizer", "parallel": false, "experiments": ["T1-01", "T1-12", "T1-13"]},
    "5": {"name": "Optimizer & Training", "parallel": true, "experiments": ["T1-14", "T1-18", "T1-20", "T1-25"]},
    "6": {"name": "Quantization", "parallel": false, "experiments": ["T1-21", "T1-22", "T1-27"]},
    "7": {"name": "Tier 2 Batch", "parallel": true, "experiments": ["T2-*"]},
    "8": {"name": "Best Combination Assembly", "parallel": false, "experiments": ["COMBINED"]},
    "9": {"name": "Tier 3 Exploratory", "parallel": true, "experiments": ["T3-*"]}
  },
  "kill_criteria": {
    "step_1000_val_loss_worse_pct": 5,
    "step_time_slower_pct": 15,
    "artifact_size_max_bytes": 17000000
  },
  "doubledown_criteria": {
    "step_1000_val_loss_better_pct": 2,
    "artifact_savings_min_bytes": 1000000
  }
}
```
