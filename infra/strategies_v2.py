"""
Comprehensive strategy list for Parameter Golf mass screening.
Generated 2026-04-13 | Target: Sub-1.08 BPB | Deadline: April 30

Usage:
    # Import into mass_screen.py:
    from strategies_v2 import ALL_STRATEGIES

    # Or run standalone to see strategy count/tiers:
    python infra/strategies_v2.py

Winning base env vars (defaults in winning_base_decoded.py):
    QK_GAIN_INIT=5.25, MLP_MULT=4.0, NUM_LAYERS=11, MODEL_DIM=512
    NUM_LOOPS=2, LOOP_START=3, LOOP_END=5, ENABLE_LOOPING_AT=0.25
    PARALLEL_RESIDUAL_START=7, SKIP_GATES_ENABLED=1
    MATRIX_LR=0.022, SCALAR_LR=0.02, MUON_WD=0.095, EMBED_WD=0.085
    EMA_DECAY=0.9965, WARMDOWN_FRAC=0.72
    GPTQ: matrix_bits=6, embed_bits=8, matrix_clip_sigmas=12.85
    COMPRESSOR=brotli, XSA_LAST_N=11, TIE_EMBEDDINGS=1

Key insights:
- Step 1000 BPB predicts final with r=0.86 -- qualify runs are THE decision point
- Seed variance 0.5 mBPB -- architecture choices 10-100x more impactful
- Smoke tests only detect BROKEN experiments (loss > 2.60)
- BigramHash: -46 mBPB, EMA: -30 mBPB, XSA: -25 mBPB, Int6 vs Int8: -46 mBPB
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Strategy:
    name: str
    script: str
    env: dict[str, str]
    description: str = ""
    tier: int = 1          # 0=proven/must-run, 1=high, 2=medium, 3=exploratory
    priority: str = "P1"   # P0=must test, P1=should test, P2=nice to have
    needs_code: bool = False
    code_changes: str = ""
    expected_mbpb: str = ""  # expected improvement in mBPB (negative = better)
    source: str = ""
    category: str = ""     # A=HP sweep, B=architecture mix, C=novel, D=different base


# ============================================================================
# CATEGORY A: HP SWEEPS ON WINNING BASE (env var only, no code changes)
# ============================================================================

CATEGORY_A: list[Strategy] = [
    # --- A1: Proven Quick Wins (tier 0, must-run) ---
    Strategy(
        "a1-sqrt-cooldown", "winning_base_decoded.py",
        {"WARMDOWN_SHAPE": "sqrt"},
        "1-sqrt cooldown (faster initial decay, slower tail) -- TMLR 2025",
        tier=0, priority="P0", expected_mbpb="-1 to -3", source="arXiv:2508.01483",
        category="A",
    ),
    Strategy(
        "a1-adaptive-gptq", "winning_base_decoded.py",
        {"GPTQ_ADAPTIVE_CLIP": "1"},
        "Per-layer GPTQ clip sigmas (tighter for MLP_down layers 9-10)",
        tier=0, priority="P0", expected_mbpb="-1 to -3", source="model-autopsy",
        category="A",
    ),
    Strategy(
        "a1-stack-quickwins", "winning_base_decoded.py",
        {"WARMDOWN_SHAPE": "sqrt", "GPTQ_ADAPTIVE_CLIP": "1"},
        "Stack sqrt cooldown + adaptive GPTQ (both orthogonal)",
        tier=0, priority="P0", expected_mbpb="-2 to -5", source="combination",
        category="A",
    ),
    Strategy(
        "a1-cosine-cooldown", "winning_base_decoded.py",
        {"WARMDOWN_SHAPE": "cosine"},
        "Cosine cooldown curve (compare against sqrt)",
        tier=0, priority="P0", expected_mbpb="-1 to -2", source="pgolf-meta",
        category="A",
    ),
    Strategy(
        "a1-batch-warmup-030", "winning_base_decoded.py",
        {"BATCH_WARMUP_FRAC": "0.3"},
        "Ramp batch 262K->786K over first 30% (43% more grad steps early)",
        tier=0, priority="P0", expected_mbpb="-2 to -5", source="arXiv:2505.23971",
        category="A",
    ),

    # --- A2: QK-Gain Sweep ---
    Strategy(
        "a2-qkgain-475", "winning_base_decoded.py",
        {"QK_GAIN_INIT": "4.75"},
        "QK-Gain 4.75 (between 4.0 and 5.25)", tier=1, priority="P1",
        expected_mbpb="-1 to -3", source="interpolation", category="A",
    ),
    Strategy(
        "a2-qkgain-500", "winning_base_decoded.py",
        {"QK_GAIN_INIT": "5.0"},
        "QK-Gain 5.0 (original winning value)", tier=1, priority="P1",
        expected_mbpb="baseline", source="PR#1493", category="A",
    ),
    Strategy(
        "a2-qkgain-550", "winning_base_decoded.py",
        {"QK_GAIN_INIT": "5.5"},
        "QK-Gain 5.5 (push further past 5.25)", tier=1, priority="P1",
        expected_mbpb="-1 to -3", source="interpolation", category="A",
    ),
    Strategy(
        "a2-qkgain-600", "winning_base_decoded.py",
        {"QK_GAIN_INIT": "6.0"},
        "QK-Gain 6.0 (aggressive sharpening)", tier=2, priority="P2",
        expected_mbpb="uncertain", source="exploration", category="A",
    ),

    # --- A3: LR / WD Sweep ---
    Strategy(
        "a3-lr-025-wd-010", "winning_base_decoded.py",
        {"MATRIX_LR": "0.025", "SCALAR_LR": "0.023", "MUON_WD": "0.10", "EMBED_WD": "0.09"},
        "Higher LR + higher WD (compensate for each other)", tier=1, priority="P1",
        expected_mbpb="-1 to -3", source="combination", category="A",
    ),
    Strategy(
        "a3-lr-018", "winning_base_decoded.py",
        {"MATRIX_LR": "0.018", "SCALAR_LR": "0.016"},
        "Lower LR (more conservative)", tier=2, priority="P2",
        expected_mbpb="uncertain", source="sweep", category="A",
    ),
    Strategy(
        "a3-wd-012", "winning_base_decoded.py",
        {"MUON_WD": "0.12", "EMBED_WD": "0.10"},
        "Much higher WD (pushes weights toward quantization-friendly dist)",
        tier=1, priority="P1", expected_mbpb="-1 to -3", source="#1218", category="A",
    ),
    Strategy(
        "a3-wd-007", "winning_base_decoded.py",
        {"MUON_WD": "0.07", "EMBED_WD": "0.06"},
        "Lower WD (less regularization)", tier=2, priority="P2",
        expected_mbpb="uncertain", source="sweep", category="A",
    ),

    # --- A4: Warmdown Shape / Frac Sweep ---
    Strategy(
        "a4-warmdown-080", "winning_base_decoded.py",
        {"WARMDOWN_FRAC": "0.80"},
        "Longer warmdown (80% vs 72%)", tier=1, priority="P1",
        expected_mbpb="-1 to -2", source="sweep", category="A",
    ),
    Strategy(
        "a4-warmdown-065", "winning_base_decoded.py",
        {"WARMDOWN_FRAC": "0.65"},
        "Shorter warmdown (65%)", tier=2, priority="P2",
        expected_mbpb="uncertain", source="sweep", category="A",
    ),
    Strategy(
        "a4-warmdown-085-sqrt", "winning_base_decoded.py",
        {"WARMDOWN_FRAC": "0.85", "WARMDOWN_SHAPE": "sqrt"},
        "Very long sqrt warmdown (85%)", tier=1, priority="P1",
        expected_mbpb="-2 to -4", source="combination", category="A",
    ),

    # --- A5: EMA Sweep ---
    Strategy(
        "a5-ema-0993", "winning_base_decoded.py",
        {"EMA_DECAY": "0.993"},
        "Lower EMA decay (faster adaptation)", tier=2, priority="P2",
        expected_mbpb="uncertain", source="sweep", category="A",
    ),
    Strategy(
        "a5-ema-0997", "winning_base_decoded.py",
        {"EMA_DECAY": "0.997"},
        "Higher EMA decay (optimal in prior sweeps at 0.997)", tier=1, priority="P1",
        expected_mbpb="-1 to -2", source="#287", category="A",
    ),
    Strategy(
        "a5-ema-0998", "winning_base_decoded.py",
        {"EMA_DECAY": "0.998"},
        "Even higher EMA decay", tier=2, priority="P2",
        expected_mbpb="uncertain", source="sweep", category="A",
    ),

    # --- A6: Depth Recurrence Variations ---
    Strategy(
        "a6-loop-2-6", "winning_base_decoded.py",
        {"LOOP_START": "2", "LOOP_END": "6"},
        "Wider recurrence (layers 2-6 instead of 3-5)", tier=1, priority="P1",
        expected_mbpb="-2 to -5", source="architecture", category="A",
    ),
    Strategy(
        "a6-loop-2-6-early", "winning_base_decoded.py",
        {"LOOP_START": "2", "LOOP_END": "6", "ENABLE_LOOPING_AT": "0.20"},
        "Wider recurrence + very early activation", tier=1, priority="P1",
        expected_mbpb="-3 to -6", source="combination", category="A",
    ),
    Strategy(
        "a6-3loops", "winning_base_decoded.py",
        {"NUM_LOOPS": "3"},
        "3x depth recurrence (more virtual layers, GPTQ risk)", tier=1, priority="P1",
        expected_mbpb="-3 to -8 or +5 (GPTQ compounding)", source="architecture", category="A",
    ),
    Strategy(
        "a6-3loops-narrow-4-5", "winning_base_decoded.py",
        {"NUM_LOOPS": "3", "LOOP_START": "4", "LOOP_END": "5"},
        "3 loops on narrow range (less GPTQ risk)", tier=1, priority="P1",
        expected_mbpb="-3 to -6", source="architecture", category="A",
    ),
    Strategy(
        "a6-loop-early-020", "winning_base_decoded.py",
        {"ENABLE_LOOPING_AT": "0.20"},
        "Very early recurrence activation (20%)", tier=1, priority="P1",
        expected_mbpb="-1 to -3", source="architecture", category="A",
    ),
    Strategy(
        "a6-no-loop", "winning_base_decoded.py",
        {"NUM_LOOPS": "1"},
        "ABLATION: disable depth recurrence (standard 11L)", tier=2, priority="P2",
        expected_mbpb="+5 to +15 (regression)", source="ablation", category="A",
    ),

    # --- A7: Parallel Residual Variations ---
    Strategy(
        "a7-parallel-5", "winning_base_decoded.py",
        {"PARALLEL_RESIDUAL_START": "5"},
        "Earlier parallel residuals (layer 5 vs 7)", tier=1, priority="P1",
        expected_mbpb="-1 to -3", source="architecture", category="A",
    ),
    Strategy(
        "a7-parallel-9", "winning_base_decoded.py",
        {"PARALLEL_RESIDUAL_START": "9"},
        "Later parallel residuals (layer 9)", tier=2, priority="P2",
        expected_mbpb="uncertain", source="architecture", category="A",
    ),
    Strategy(
        "a7-no-parallel", "winning_base_decoded.py",
        {"PARALLEL_RESIDUAL_START": "99"},
        "ABLATION: disable parallel residuals (all sequential)", tier=2, priority="P2",
        expected_mbpb="+3 to +10 (regression)", source="ablation", category="A",
    ),

    # --- A8: MLP Width ---
    Strategy(
        "a8-mlp-35", "winning_base_decoded.py",
        {"MLP_MULT": "3.5"},
        "Narrower MLP (3.5x vs 4x, more training steps)", tier=1, priority="P1",
        expected_mbpb="-1 to -3", source="param-budget", category="A",
    ),
    Strategy(
        "a8-mlp-45", "winning_base_decoded.py",
        {"MLP_MULT": "4.5"},
        "Wider MLP (4.5x, more capacity, fewer steps)", tier=2, priority="P2",
        expected_mbpb="uncertain", source="param-budget", category="A",
    ),

    # --- A9: Quantization Tweaks ---
    Strategy(
        "a9-embed-bits-7", "winning_base_decoded.py",
        {"EMBED_BITS": "7"},
        "Int7 embeddings (frees ~62KB for matrix quantization)", tier=1, priority="P1",
        expected_mbpb="-1 to -3", source="pgolf-meta", category="A",
    ),
    Strategy(
        "a9-matrix-clip-10", "winning_base_decoded.py",
        {"MATRIX_CLIP_SIGMAS": "10.0"},
        "Tighter matrix clipping (10.0 vs 12.85)", tier=2, priority="P2",
        expected_mbpb="-1 to +2", source="sweep", category="A",
    ),
    Strategy(
        "a9-matrix-clip-15", "winning_base_decoded.py",
        {"MATRIX_CLIP_SIGMAS": "15.0"},
        "Looser matrix clipping (15.0 vs 12.85)", tier=2, priority="P2",
        expected_mbpb="-1 to +2", source="sweep", category="A",
    ),
    Strategy(
        "a9-calib-128", "winning_base_decoded.py",
        {"GPTQ_CALIBRATION_BATCHES": "128"},
        "More GPTQ calibration batches (128 vs 64)", tier=1, priority="P1",
        expected_mbpb="-1 to -2", source="quantization", category="A",
    ),

    # --- A10: TTT / ETLB Eval-Time ---
    Strategy(
        "a10-ttt-enabled", "winning_base_decoded.py",
        {"TTT_ENABLED": "1", "TTT_LR": "0.005", "TTT_EPOCHS": "3"},
        "Enable test-time training (backward-looking, legal variant)", tier=1, priority="P1",
        expected_mbpb="-5 to -15", source="#461", category="A",
    ),
    Strategy(
        "a10-ttt-lr-001", "winning_base_decoded.py",
        {"TTT_ENABLED": "1", "TTT_LR": "0.001", "TTT_EPOCHS": "3"},
        "TTT with lower LR", tier=1, priority="P1",
        expected_mbpb="-5 to -12", source="sweep", category="A",
    ),
    Strategy(
        "a10-ttt-lr-010", "winning_base_decoded.py",
        {"TTT_ENABLED": "1", "TTT_LR": "0.010", "TTT_EPOCHS": "5"},
        "TTT aggressive (higher LR, more epochs)", tier=2, priority="P2",
        expected_mbpb="-5 to -20 or diverge", source="sweep", category="A",
    ),

    # --- A11: Skip Gate Ablation ---
    Strategy(
        "a11-no-skipgates", "winning_base_decoded.py",
        {"SKIP_GATES_ENABLED": "0"},
        "ABLATION: disable skip gates (simpler architecture)", tier=2, priority="P2",
        expected_mbpb="+1 to +5 (regression)", source="ablation", category="A",
    ),

    # --- A12: Mega Combos (stacking best single-dim winners) ---
    Strategy(
        "a12-combo-best-v1", "winning_base_decoded.py",
        {
            "WARMDOWN_SHAPE": "sqrt",
            "GPTQ_ADAPTIVE_CLIP": "1",
            "BATCH_WARMUP_FRAC": "0.3",
            "EMA_DECAY": "0.997",
            "WARMDOWN_FRAC": "0.80",
        },
        "Mega combo v1: sqrt + adaptive GPTQ + batch warmup + EMA 0.997 + warmdown 0.80",
        tier=0, priority="P0", expected_mbpb="-5 to -10", source="combination",
        category="A",
    ),
    Strategy(
        "a12-combo-best-v2", "winning_base_decoded.py",
        {
            "WARMDOWN_SHAPE": "sqrt",
            "GPTQ_ADAPTIVE_CLIP": "1",
            "BATCH_WARMUP_FRAC": "0.3",
            "EMBED_BITS": "7",
            "MATRIX_LR": "0.025",
            "SCALAR_LR": "0.023",
            "MUON_WD": "0.10",
            "EMBED_WD": "0.09",
        },
        "Mega combo v2: sqrt + adaptive GPTQ + batch warmup + int7 embeds + higher LR/WD",
        tier=0, priority="P0", expected_mbpb="-5 to -12", source="combination",
        category="A",
    ),
    Strategy(
        "a12-combo-best-v3", "winning_base_decoded.py",
        {
            "WARMDOWN_SHAPE": "sqrt",
            "GPTQ_ADAPTIVE_CLIP": "1",
            "BATCH_WARMUP_FRAC": "0.3",
            "LOOP_START": "2",
            "LOOP_END": "6",
            "ENABLE_LOOPING_AT": "0.20",
            "PARALLEL_RESIDUAL_START": "5",
        },
        "Mega combo v3: sqrt + GPTQ + batch + wider loop + earlier parallel",
        tier=0, priority="P0", expected_mbpb="-5 to -15", source="combination",
        category="A",
    ),
    Strategy(
        "a12-combo-3loop-narrow-sqrt", "winning_base_decoded.py",
        {
            "NUM_LOOPS": "3",
            "LOOP_START": "4",
            "LOOP_END": "5",
            "WARMDOWN_SHAPE": "sqrt",
            "GPTQ_ADAPTIVE_CLIP": "1",
            "PARALLEL_RESIDUAL_START": "5",
        },
        "3 loops narrow range + sqrt + adaptive GPTQ + early parallel",
        tier=1, priority="P0", expected_mbpb="-5 to -12", source="combination",
        category="A",
    ),

    # --- A13: Grad clip / momentum tweaks ---
    Strategy(
        "a13-gradclip-05", "winning_base_decoded.py",
        {"GRAD_CLIP_NORM": "0.5"},
        "Higher grad clip (0.5 vs 0.3)", tier=2, priority="P2",
        expected_mbpb="uncertain", source="sweep", category="A",
    ),
    Strategy(
        "a13-gradclip-02", "winning_base_decoded.py",
        {"GRAD_CLIP_NORM": "0.2"},
        "Lower grad clip (0.2 vs 0.3)", tier=2, priority="P2",
        expected_mbpb="uncertain", source="sweep", category="A",
    ),
    Strategy(
        "a13-momentum-warmup-2000", "winning_base_decoded.py",
        {"MUON_MOMENTUM_WARMUP_STEPS": "2000"},
        "Longer Muon momentum warmup (2000 vs 1500)", tier=2, priority="P2",
        expected_mbpb="-0.5 to -1", source="sweep", category="A",
    ),

    # --- A14: Eval stride ---
    Strategy(
        "a14-stride-32", "winning_base_decoded.py",
        {"EVAL_STRIDE": "32"},
        "Half eval stride (more overlap, 2x eval time)", tier=1, priority="P1",
        expected_mbpb="-2 to -5", source="eval-opt", category="A",
    ),
    Strategy(
        "a14-stride-16", "winning_base_decoded.py",
        {"EVAL_STRIDE": "16"},
        "Quarter eval stride (4x eval time, max overlap)", tier=2, priority="P2",
        expected_mbpb="-3 to -8", source="eval-opt", category="A",
    ),

    # --- A15: Logit softcap ---
    Strategy(
        "a15-softcap-25", "winning_base_decoded.py",
        {"LOGIT_SOFTCAP": "25.0"},
        "Lower logit softcap (25 vs 30, more aggressive capping)", tier=2, priority="P2",
        expected_mbpb="-0.5 to -2", source="sweep", category="A",
    ),
    Strategy(
        "a15-softcap-40", "winning_base_decoded.py",
        {"LOGIT_SOFTCAP": "40.0"},
        "Higher logit softcap (40 vs 30, less capping)", tier=2, priority="P2",
        expected_mbpb="uncertain", source="sweep", category="A",
    ),

    # --- A16: GPTQ reserve timing ---
    Strategy(
        "a16-gptq-reserve-8", "winning_base_decoded.py",
        {"GPTQ_RESERVE_SECONDS": "8.0"},
        "Less GPTQ reserve (8s vs 12s = more training steps)", tier=1, priority="P1",
        expected_mbpb="-0.5 to -2", source="timing", category="A",
    ),
]


# ============================================================================
# CATEGORY B: ARCHITECTURE MIXTURES (combining techniques, needs code changes)
# ============================================================================

CATEGORY_B: list[Strategy] = [
    # --- B1: SLOT (Selective Logit Offset Tuning) ---
    Strategy(
        "b1-slot-basic", "winning_base_decoded.py",
        {"ETLB_ENABLED": "1", "ETLB_LR": "0.008", "ETLB_STEPS": "16"},
        "SLOT: per-batch delta vector at last hidden layer, 16 AdamW steps",
        tier=1, priority="P0", needs_code=True,
        code_changes="Implement eval_val_sliding_etlb() with delta[1,1,512] optimized on scored positions",
        expected_mbpb="-15 to -25", source="#1084,#1172", category="B",
    ),
    Strategy(
        "b1-slot-persample", "winning_base_decoded.py",
        {"ETLB_ENABLED": "1", "ETLB_LR": "0.008", "ETLB_STEPS": "16"},
        "Per-Sample SLOT: delta[B,1,512] + logit_bias[B,1,V] per sample",
        tier=1, priority="P0", needs_code=True,
        code_changes="Modify SLOT to per-sample delta + optional logit bias",
        expected_mbpb="-20 to -30", source="#1229", category="B",
    ),
    Strategy(
        "b1-slot-context-only", "winning_base_decoded.py",
        {"ETLB_ENABLED": "1", "ETLB_LR": "0.008", "ETLB_STEPS": "16"},
        "Context-Only SLOT: delta from context positions only (provably causal)",
        tier=1, priority="P0", needs_code=True,
        code_changes="Mask SLOT optimization to context-only positions",
        expected_mbpb="-14 to -24", source="#1217", category="B",
    ),

    # --- B2: Architecture Enhancements ---
    Strategy(
        "b2-ddl-residual-gates", "winning_base_decoded.py",
        {},
        "Deep Delta Learning: rank-1 erasure gate on residuals (11K params)",
        tier=1, priority="P1", needs_code=True,
        code_changes="Add ddl_u, ddl_v, ddl_beta per Block; erase = beta*(x@u)*v before residual add",
        expected_mbpb="-3 to -7", source="arXiv:2601.00417", category="B",
    ),
    Strategy(
        "b2-hyper-connections", "winning_base_decoded.py",
        {},
        "Hyper-Connections: 2x2 connection matrix per layer replacing residuals",
        tier=1, priority="P1", needs_code=True,
        code_changes="Add hc_alpha[2,2] per Block; mix [x, f(x)] via learned matrix",
        expected_mbpb="-3 to -8", source="arXiv:2409.19606", category="B",
    ),
    Strategy(
        "b2-hourglass-ffn", "winning_base_decoded.py",
        {},
        "Hourglass FFN: 2x stacked narrow MLPs (512->768->512) with residual",
        tier=1, priority="P1", needs_code=True,
        code_changes="Replace MLP with HourglassMLP(sub_mult=1.5, n_layers=2)",
        expected_mbpb="-2 to -6", source="arXiv:2602.06471", category="B",
    ),
    Strategy(
        "b2-fox-forgetting-attn", "winning_base_decoded.py",
        {},
        "FoX: data-dependent forget gate on attention (could replace RoPE)",
        tier=2, priority="P1", needs_code=True,
        code_changes="Add forget_proj Linear(dim,num_heads); cumulative forget gate on attn scores",
        expected_mbpb="-3 to -8", source="arXiv:2503.02130", category="B",
    ),

    # --- B3: Optimizer Enhancements ---
    Strategy(
        "b3-huber-decay", "winning_base_decoded.py",
        {},
        "Huber WD: quadratic below delta=0.1, linear above (suppresses quant outliers)",
        tier=1, priority="P1", needs_code=True,
        code_changes="In Muon.step() replace p.data.mul_(1-lr*wd) with Huber decay",
        expected_mbpb="-2 to -5", source="arXiv:2511.14721", category="B",
    ),
    Strategy(
        "b3-ifnso", "winning_base_decoded.py",
        {},
        "IFNSO: single polynomial replacing 5-step Newton-Schulz (more steps in 600s)",
        tier=1, priority="P1", needs_code=True,
        code_changes="Replace zeropower_via_newtonschulz5 with polynomial approximation",
        expected_mbpb="-1 to -3 (via +100-300 extra steps)", source="arXiv:2602.02500", category="B",
    ),
    Strategy(
        "b3-mousse", "winning_base_decoded.py",
        {},
        "Mousse: Shampoo preconditioning before Muon NS (12% more effective per step)",
        tier=1, priority="P1", needs_code=True,
        code_changes="Add Shampoo L,R factors to Muon; precondition every 10 steps",
        expected_mbpb="-3 to -8", source="arXiv:2603.09697", category="B",
    ),

    # --- B4: Quantization Enhancements ---
    Strategy(
        "b4-mixed-int5-int6", "winning_base_decoded.py",
        {},
        "Autopsy-informed mixed quant: int6 for MLP_down(9,10), int5 for rest",
        tier=1, priority="P1", needs_code=True,
        code_changes="Per-layer bit allocation in gptq_mixed_quantize based on sensitivity",
        expected_mbpb="-3 to -8", source="pr-1105-model-autopsy", category="B",
    ),
    Strategy(
        "b4-prune-then-quantize", "winning_base_decoded.py",
        {},
        "Reverse ordering: magnitude prune 3% -> GPTQ (vs current GPTQ first)",
        tier=1, priority="P1", needs_code=True,
        code_changes="Add magnitude pruning step before GPTQ in serialize()",
        expected_mbpb="-1 to -3", source="arXiv:2603.18426", category="B",
    ),
    Strategy(
        "b4-temp-scaling", "winning_base_decoded.py",
        {},
        "Post-GPTQ temperature scaling (fix ECE from 1.26% to ~0.24%)",
        tier=1, priority="P1", needs_code=True,
        code_changes="Grid search T in [0.9, 1.1] on first 10% of eval; apply to all logits",
        expected_mbpb="-1 to -3", source="model-autopsy", category="B",
    ),

    # --- B5: Eval-Time Enhancements ---
    Strategy(
        "b5-ngram-cache", "winning_base_decoded.py",
        {},
        "N-gram backoff cache (orders 2-9, Laplace smoothed, entropy-adaptive alpha)",
        tier=1, priority="P0", needs_code=True,
        code_changes="Implement NgramCache class; mix with model probs during sliding eval",
        expected_mbpb="-50 to -150 (massive, GRAY_AREA legality)", source="#1185,#795",
        category="B",
    ),
    Strategy(
        "b5-ttt-freeze-early", "winning_base_decoded.py",
        {"TTT_ENABLED": "1", "TTT_LR": "0.005", "TTT_EPOCHS": "3"},
        "TTT with frozen first 2 blocks (reduce catastrophic adaptation)",
        tier=1, priority="P1", needs_code=True,
        code_changes="In eval_val_ttt, set blocks[0:2].requires_grad_(False)",
        expected_mbpb="-5 to -15", source="#461", category="B",
    ),
    Strategy(
        "b5-entropy-stride", "winning_base_decoded.py",
        {},
        "Two-pass eval: stride=64 first, then re-eval high-entropy with stride=16",
        tier=2, priority="P1", needs_code=True,
        code_changes="Add entropy tracking in first pass; second pass on positions with H>4.0",
        expected_mbpb="-5 to -15", source="analyzer untried", category="B",
    ),

    # --- B6: Systems Throughput ---
    Strategy(
        "b6-coprime-stride-loader", "winning_base_decoded.py",
        {},
        "Coprime-stride data loading (max batch diversity, zero overhead)",
        tier=1, priority="P1", needs_code=True,
        code_changes="Replace sequential shard reading with coprime-stride block sampling",
        expected_mbpb="-1 to -3", source="#726", category="B",
    ),
    Strategy(
        "b6-liger-kernels", "winning_base_decoded.py",
        {},
        "Liger-Kernel fused ops (RMSNorm 6x, linear+CE 3x)",
        tier=1, priority="P1", needs_code=True,
        code_changes="pip install liger-kernel; replace RMSNorm and CE loss with Liger variants",
        expected_mbpb="-1 to -5 (via +200-500 extra steps)", source="LinkedIn", category="B",
    ),

    # --- B7: Combo Architectures ---
    Strategy(
        "b7-slot-plus-sqrt-gptq", "winning_base_decoded.py",
        {"WARMDOWN_SHAPE": "sqrt", "GPTQ_ADAPTIVE_CLIP": "1", "ETLB_ENABLED": "1", "ETLB_LR": "0.008", "ETLB_STEPS": "16"},
        "Full stack: SLOT + sqrt cooldown + adaptive GPTQ",
        tier=1, priority="P0", needs_code=True,
        code_changes="Implement SLOT + use sqrt/adaptive GPTQ env vars",
        expected_mbpb="-18 to -30", source="combination", category="B",
    ),
    Strategy(
        "b7-ttt-plus-sqrt-batch", "winning_base_decoded.py",
        {"TTT_ENABLED": "1", "TTT_LR": "0.005", "TTT_EPOCHS": "3",
         "WARMDOWN_SHAPE": "sqrt", "BATCH_WARMUP_FRAC": "0.3"},
        "TTT + sqrt cooldown + batch warmup (non-SLOT path)",
        tier=1, priority="P1", needs_code=False,
        expected_mbpb="-8 to -18", source="combination", category="B",
    ),
    Strategy(
        "b7-full-stack-best", "winning_base_decoded.py",
        {
            "WARMDOWN_SHAPE": "sqrt",
            "GPTQ_ADAPTIVE_CLIP": "1",
            "BATCH_WARMUP_FRAC": "0.3",
            "EMBED_BITS": "7",
            "LOOP_START": "2",
            "LOOP_END": "6",
            "ENABLE_LOOPING_AT": "0.20",
            "PARALLEL_RESIDUAL_START": "5",
            "EMA_DECAY": "0.997",
            "ETLB_ENABLED": "1",
            "ETLB_LR": "0.008",
            "ETLB_STEPS": "16",
        },
        "EVERYTHING: SLOT + sqrt + GPTQ + batch + int7 + wide loop + early parallel + EMA",
        tier=0, priority="P0", needs_code=True,
        code_changes="Implement SLOT (eval_val_sliding_etlb); all other features are env vars",
        expected_mbpb="-20 to -40", source="combination", category="B",
    ),

    # --- B8: Cross-Layer KV Sharing ---
    Strategy(
        "b8-kv-sharing", "winning_base_decoded.py",
        {},
        "Adjacent layer pairs share KV projections (saves ~0.5MB, reinvest in MLP)",
        tier=2, priority="P1", needs_code=True,
        code_changes="Create 6 shared KV proj for 11 layers; layer_idx//2 indexes KV",
        expected_mbpb="-2 to -6", source="MLKV/CLA", category="B",
    ),

    # --- B9: Architecture Replacements ---
    Strategy(
        "b9-asqu-activation", "winning_base_decoded.py",
        {},
        "ASQU: per-channel learned neg_slope and scale in activation",
        tier=2, priority="P1", needs_code=True,
        code_changes="Replace fixed LeakyReLU(0.5)^2 with per-channel ASQU(hidden_dim)",
        expected_mbpb="-1 to -3", source="#1035", category="B",
    ),
    Strategy(
        "b9-wavelet-embedding", "winning_base_decoded.py",
        {},
        "WaveletGPT: Haar wavelet on half of embedding dims (40-60% faster convergence)",
        tier=2, priority="P1", needs_code=True,
        code_changes="Split embedding; apply Haar wavelet to second half; inverse before output",
        expected_mbpb="-3 to -10", source="arXiv:2409.12924", category="B",
    ),
    Strategy(
        "b9-conv-token-mixer", "winning_base_decoded.py",
        {},
        "Causal depthwise Conv1d (kernel=4) before attention in each block",
        tier=2, priority="P2", needs_code=True,
        code_changes="Add nn.Conv1d(dim, dim, 4, padding=3, groups=dim) to Block",
        expected_mbpb="-1 to -3", source="#1180", category="B",
    ),

    # --- B10: Non-Uniform FFN ---
    Strategy(
        "b10-crown-ffn", "winning_base_decoded.py",
        {},
        "Crown-shaped FFN: wide middle (4x), narrow edges (2x), same total params",
        tier=2, priority="P1", needs_code=True,
        code_changes="Per-layer mlp_mults=[2,2,2.5,3,3.5,4,4,3.5,3,2.5,2]",
        expected_mbpb="-2 to -5", source="arXiv:2509.06518, autopsy", category="B",
    ),

    # --- B11: MTP Training Signal ---
    Strategy(
        "b11-mtp-aux", "winning_base_decoded.py",
        {},
        "MTP auxiliary loss (2 heads, weight=0.1, discarded at export)",
        tier=2, priority="P2", needs_code=True,
        code_changes="Add 2 MTP heads during training; weight=0.1; remove before GPTQ",
        expected_mbpb="-2 to -4", source="#1031", category="B",
    ),

    # --- B12: Focal/Weighted Loss ---
    Strategy(
        "b12-focal-loss", "winning_base_decoded.py",
        {},
        "Focal loss: (1-p)^2 * (-log p), focuses gradient on hard tokens",
        tier=2, priority="P1", needs_code=True,
        code_changes="Replace F.cross_entropy with focal_loss(gamma=2.0)",
        expected_mbpb="-5 to -15 (uncertain)", source="#1180", category="B",
    ),

    # --- B13: QAT Variants ---
    Strategy(
        "b13-late-qat", "winning_base_decoded.py",
        {},
        "Activate QAT at warmdown onset (int6 STE in last 28% of training)",
        tier=2, priority="P1", needs_code=True,
        code_changes="fake_quantize in CastedLinear.forward when step >= warmdown_start",
        expected_mbpb="-1 to -4", source="arXiv:2509.22935", category="B",
    ),

    # --- B14: Block AttnRes ---
    Strategy(
        "b14-block-attnres", "winning_base_decoded.py",
        {},
        "Block AttnRes: 3 blocks, learned softmax over block layer outputs (<2% overhead)",
        tier=2, priority="P1", needs_code=True,
        code_changes="Partition 11L into 3 blocks; softmax routing over block outputs",
        expected_mbpb="-3 to -8", source="arXiv:2603.15031", category="B",
    ),

    # --- B15: EngramLite ---
    Strategy(
        "b15-engramlite", "winning_base_decoded.py",
        {},
        "Multi-head prime-based hash embeddings (bigrams + trigrams)",
        tier=2, priority="P1", needs_code=True,
        code_changes="Replace BigramHash with EngramLite(n_buckets=8192, n_heads=2)",
        expected_mbpb="-3 to -8", source="#1089", category="B",
    ),

    # --- B16: Seesaw LR + Batch ---
    Strategy(
        "b16-seesaw-schedule", "winning_base_decoded.py",
        {},
        "Seesaw: LR*=1/sqrt(2) + batch*=2 at 50% and 75% training",
        tier=2, priority="P1", needs_code=True,
        code_changes="Modify lr_mul and batch schedule in train_model()",
        expected_mbpb="-2 to -5", source="arXiv:2510.14717", category="B",
    ),
]


# ============================================================================
# CATEGORY C: NOVEL IDEAS (things nobody has tried yet)
# ============================================================================

CATEGORY_C: list[Strategy] = [
    Strategy(
        "c1-logistic-domain-mixing", "winning_base_decoded.py",
        {},
        "N-gram mixing in log-odds space instead of linear (PAQ-style)",
        tier=2, priority="P1", needs_code=True,
        code_changes="Mix p_model and p_ngram via: logit_model + alpha*logit_ngram",
        expected_mbpb="-2 to -5 (on top of n-gram)", source="analyzer untried", category="C",
    ),
    Strategy(
        "c2-ppmii-backoff", "winning_base_decoded.py",
        {},
        "PPMII escape estimation with information inheritance (Shkarin 2001)",
        tier=2, priority="P1", needs_code=True,
        code_changes="Replace heuristic backoff with principled escape probs + full exclusions",
        expected_mbpb="-10 to -30 (on top of basic n-gram)", source="compression theory",
        category="C",
    ),
    Strategy(
        "c3-match-model", "winning_base_decoded.py",
        {},
        "Longest-match prediction from scored history (LPAQ/PAQ style)",
        tier=2, priority="P1", needs_code=True,
        code_changes="Find longest match in scored tokens; predict next from what followed match",
        expected_mbpb="-5 to -10", source="PAQ lineage", category="C",
    ),
    Strategy(
        "c4-sparse-skipgram", "winning_base_decoded.py",
        {},
        "Sparse/skip-gram context models (non-contiguous positions as context)",
        tier=2, priority="P1", needs_code=True,
        code_changes="Hash positions at -1,-3,-5 as context; multiple gap patterns",
        expected_mbpb="-5 to -15", source="analyzer untried", category="C",
    ),
    Strategy(
        "c5-cascaded-sse", "winning_base_decoded.py",
        {},
        "3-5 chained Adaptive Probability Map stages (PAQ/PAQAR style)",
        tier=2, priority="P1", needs_code=True,
        code_changes="Chain multiple APM stages with progressively higher-order contexts",
        expected_mbpb="-5 to -15", source="PAQ lineage", category="C",
    ),
    Strategy(
        "c6-complementary-distill", "winning_base_decoded.py",
        {},
        "Complementary distillation: -lambda*KL(P_ngram||P_model) in training loss",
        tier=2, priority="P1", needs_code=True,
        code_changes="Add KL divergence term pushing model to diverge from n-gram distribution",
        expected_mbpb="-10 to -30 (with n-gram at eval)", source="EMNLP 2022", category="C",
    ),
    Strategy(
        "c7-soft-quantize-coupling", "winning_base_decoded.py",
        {},
        "Physics-inspired coupling regularizer: weights self-discretize to int6 grid",
        tier=3, priority="P2", needs_code=True,
        code_changes="Add coupling_loss pulling weights toward quantization levels",
        expected_mbpb="-1 to -3", source="arXiv:2601.21219", category="C",
    ),
    Strategy(
        "c8-anti-layer-removal", "winning_base_decoded.py",
        {},
        "Post-training: try removing each layer one at a time (find anti-layers)",
        tier=2, priority="P1", needs_code=True,
        code_changes="After training, evaluate 11 ablated models; remove anti-layers",
        expected_mbpb="-2 to -6 (if anti-layers found)", source="arXiv:2603.19348", category="C",
    ),
    Strategy(
        "c9-differential-attention", "winning_base_decoded.py",
        {},
        "Differential attention: difference of two softmax maps (reduces outliers)",
        tier=3, priority="P2", needs_code=True,
        code_changes="Double Q/K heads; compute attn1-attn2; trade MLP width for attention",
        expected_mbpb="-5 to -15 (high param cost)", source="arXiv:2410.05258", category="C",
    ),
    Strategy(
        "c10-mixture-of-depths", "winning_base_decoded.py",
        {},
        "MoD: per-layer router skips easy tokens (50% skip fraction)",
        tier=3, priority="P2", needs_code=True,
        code_changes="Add router Linear(dim,1) per block; skip low-score tokens",
        expected_mbpb="-2 to -8 (via faster eval)", source="arXiv:2404.02258", category="C",
    ),
    Strategy(
        "c11-late-sam", "winning_base_decoded.py",
        {},
        "SAM in last 10% of training (flatter minima, better quantization)",
        tier=3, priority="P2", needs_code=True,
        code_changes="Add SAM perturbation loop in last 10% of training",
        expected_mbpb="-2 to -5", source="arXiv:2410.10373", category="C",
    ),
    Strategy(
        "c12-wsm-replace-warmdown", "winning_base_decoded.py",
        {},
        "WSM: constant LR + offline checkpoint merge (replace warmdown entirely)",
        tier=2, priority="P1", needs_code=True,
        code_changes="Remove warmdown; save checkpoints every 100 steps in last 30%; average",
        expected_mbpb="-2 to -6", source="arXiv:2507.17634", category="C",
    ),
    Strategy(
        "c13-numuon", "winning_base_decoded.py",
        {},
        "NuMuon: nuclear-norm constraint on Muon updates (lower stable rank, better compression)",
        tier=2, priority="P2", needs_code=True,
        code_changes="After NS, apply SVD nuclear norm constraint on updates",
        expected_mbpb="-2 to -6 (via smaller artifact)", source="arXiv:2603.03597", category="C",
    ),
    Strategy(
        "c14-lora-depth-recurrence", "winning_base_decoded.py",
        {},
        "Shared blocks + per-pass LoRA rank-32 deltas (22 virtual layers from 11 physical)",
        tier=2, priority="P2", needs_code=True,
        code_changes="Share base block weights; add LoRA A,B per virtual layer",
        expected_mbpb="-10 to -30 (uncertain, GPTQ risk)", source="arXiv:2410.20672", category="C",
    ),
    Strategy(
        "c15-guided-quant", "winning_base_decoded.py",
        {},
        "GuidedQuant: gradient-aware PTQ replacing standard GPTQ",
        tier=2, priority="P2", needs_code=True,
        code_changes="Integrate end-loss gradients into layer-wise quantization",
        expected_mbpb="-2 to -5", source="OpenReview 2025", category="C",
    ),
    Strategy(
        "c16-fixedshare-hedge", "winning_base_decoded.py",
        {},
        "Fixed-Share Hedge for non-stationary expert tracking (n-gram mixing)",
        tier=2, priority="P1", needs_code=True,
        code_changes="Replace static alpha with Herbster-Warmuth switching tracker",
        expected_mbpb="-3 to -8 (on top of n-gram)", source="analyzer untried", category="C",
    ),
    Strategy(
        "c17-higher-order-compl-train", "winning_base_decoded.py",
        {},
        "Use 7-gram (not bigram) for complementary training loss weighting",
        tier=2, priority="P1", needs_code=True,
        code_changes="Replace bigram-based token weights with 7-gram statistics matching eval cache",
        expected_mbpb="-5 to -10 (with n-gram at eval)", source="analyzer untried", category="C",
    ),
    Strategy(
        "c18-dcmha", "winning_base_decoded.py",
        {},
        "DCMHA: input-dependent transforms on attention (matches 1.7x compute models)",
        tier=3, priority="P2", needs_code=True,
        code_changes="Implement per arXiv:2405.08553",
        expected_mbpb="-5 to -15", source="arXiv:2405.08553", category="C",
    ),
    Strategy(
        "c19-hybridnorm", "winning_base_decoded.py",
        {},
        "Mixed Pre/Post-Norm: layers 0-5 Post-Norm, layers 6-10 Pre-Norm",
        tier=3, priority="P2", needs_code=True,
        code_changes="Modify Block.forward to use post-norm for early layers",
        expected_mbpb="-2 to -6", source="arXiv:2503.04598", category="C",
    ),
    Strategy(
        "c20-ssmax", "winning_base_decoded.py",
        {},
        "SSMax: scale softmax by log(seq_len)/log(512) to prevent attention flattening",
        tier=2, priority="P2", needs_code=True,
        code_changes="Multiply attention scale by math.log(seq_len)/math.log(512)",
        expected_mbpb="-1 to -4", source="arXiv:2501.19399", category="C",
    ),
]


# ============================================================================
# CATEGORY D: DIFFERENT BASE ARCHITECTURES
# ============================================================================

CATEGORY_D: list[Strategy] = [
    Strategy(
        "d1-sp4096-base", "winning_base_decoded.py",
        {"VOCAB_SIZE": "4096"},
        "SP4096 tokenizer (4x vocab, different compression tradeoff)",
        tier=2, priority="P1", needs_code=False,
        expected_mbpb="uncertain (different regime)", source="#1218", category="D",
    ),
    Strategy(
        "d2-12layer-narrower", "winning_base_decoded.py",
        {"NUM_LAYERS": "12", "MLP_MULT": "3.5"},
        "12 layers x 3.5x MLP (deeper but narrower, more virtual with loops)",
        tier=2, priority="P1", needs_code=False,
        expected_mbpb="-3 to -8 (if fits in 16MB)", source="scaling", category="D",
    ),
    Strategy(
        "d3-13layer-3x-mlp", "winning_base_decoded.py",
        {"NUM_LAYERS": "13", "MLP_MULT": "3.0"},
        "13 layers x 3.0x MLP (deepest feasible at int6)",
        tier=2, priority="P2", needs_code=False,
        expected_mbpb="uncertain", source="scaling", category="D",
    ),
    Strategy(
        "d4-9layer-5x-mlp", "winning_base_decoded.py",
        {"NUM_LAYERS": "9", "MLP_MULT": "5.0"},
        "9 layers x 5x MLP (fewer layers, much wider MLP, with 2x loop = 11 virtual)",
        tier=2, priority="P2", needs_code=False,
        expected_mbpb="uncertain", source="width-vs-depth", category="D",
    ),
    Strategy(
        "d5-dim-384-14layer", "winning_base_decoded.py",
        {"MODEL_DIM": "384", "EMBEDDING_DIM": "384", "NUM_LAYERS": "14", "NUM_HEADS": "6", "NUM_KV_HEADS": "3", "MLP_MULT": "4.0"},
        "Narrower 384d x 14L (trades width for depth)",
        tier=3, priority="P2", needs_code=False,
        expected_mbpb="uncertain", source="depth-scaling", category="D",
    ),
    Strategy(
        "d6-dim-640-8layer", "winning_base_decoded.py",
        {"MODEL_DIM": "640", "EMBEDDING_DIM": "640", "NUM_LAYERS": "8", "NUM_HEADS": "10", "NUM_KV_HEADS": "5", "MLP_MULT": "3.0"},
        "Wider 640d x 8L (trades depth for width, with loop for depth recovery)",
        tier=3, priority="P2", needs_code=False,
        expected_mbpb="uncertain", source="width-scaling", category="D",
    ),
    Strategy(
        "d7-sp1024-experiment1", "experiment1.py",
        {},
        "Baseline SP1024 experiment1.py for comparison", tier=3, priority="P2",
        expected_mbpb="baseline (1.12-1.13)", source="baseline", category="D",
    ),
    Strategy(
        "d8-simplify-stack", "winning_base_decoded.py",
        {
            "MLP_MULT": "4.0",
            "MUON_WD": "0.12",
            "EMBED_WD": "0.10",
            "NUM_LOOPS": "1",
            "SKIP_GATES_ENABLED": "0",
            "PARALLEL_RESIDUAL_START": "99",
        },
        "Simplify: MLP 4x + high WD + no loops + no skip gates + no parallel (clarkkev approach)",
        tier=2, priority="P1", needs_code=False,
        expected_mbpb="-2 to +5 (test if simpler is better)", source="#1218", category="D",
    ),
    Strategy(
        "d9-int5-wider", "winning_base_decoded.py",
        {"MATRIX_BITS": "5", "MLP_MULT": "5.0", "NUM_LAYERS": "12"},
        "Int5 quantization + wider/deeper model (40M params at int5 fits 16MB)",
        tier=2, priority="P1", needs_code=False,
        expected_mbpb="-5 to +10 (int5 quality vs more params)", source="scaling", category="D",
    ),
    Strategy(
        "d10-sp2048", "winning_base_decoded.py",
        {"VOCAB_SIZE": "2048"},
        "SP2048 tokenizer (middle ground between 1024 and 8192)",
        tier=2, priority="P2", needs_code=False,
        expected_mbpb="uncertain", source="tokenizer-sweep", category="D",
    ),
]


# ============================================================================
# ALL STRATEGIES COMBINED
# ============================================================================

ALL_STRATEGIES: list[Strategy] = CATEGORY_A + CATEGORY_B + CATEGORY_C + CATEGORY_D


# ============================================================================
# Utility functions
# ============================================================================

def get_by_priority(priority: str) -> list[Strategy]:
    """Get strategies by priority level."""
    return [s for s in ALL_STRATEGIES if s.priority == priority]


def get_by_tier(tier: int) -> list[Strategy]:
    """Get strategies by tier."""
    return [s for s in ALL_STRATEGIES if s.tier == tier]


def get_by_category(category: str) -> list[Strategy]:
    """Get strategies by category letter."""
    return [s for s in ALL_STRATEGIES if s.category == category]


def get_env_only() -> list[Strategy]:
    """Get strategies that need NO code changes (env var only)."""
    return [s for s in ALL_STRATEGIES if not s.needs_code]


def get_needs_code() -> list[Strategy]:
    """Get strategies that need code changes."""
    return [s for s in ALL_STRATEGIES if s.needs_code]


def print_summary():
    """Print strategy summary statistics."""
    total = len(ALL_STRATEGIES)
    by_cat = {}
    by_tier = {}
    by_priority = {}
    env_only = 0
    needs_code = 0

    for s in ALL_STRATEGIES:
        by_cat[s.category] = by_cat.get(s.category, 0) + 1
        by_tier[s.tier] = by_tier.get(s.tier, 0) + 1
        by_priority[s.priority] = by_priority.get(s.priority, 0) + 1
        if s.needs_code:
            needs_code += 1
        else:
            env_only += 1

    print(f"\n{'='*60}")
    print(f" Parameter Golf Strategy Summary")
    print(f" Total strategies: {total}")
    print(f"{'='*60}")
    print(f"\nBy Category:")
    for cat in sorted(by_cat):
        label = {"A": "HP Sweeps", "B": "Architecture Mix", "C": "Novel Ideas", "D": "Different Base"}.get(cat, cat)
        print(f"  Category {cat} ({label}): {by_cat[cat]}")
    print(f"\nBy Tier:")
    for tier in sorted(by_tier):
        label = {0: "Proven/Must-Run", 1: "High Priority", 2: "Medium Priority", 3: "Exploratory"}.get(tier, str(tier))
        print(f"  Tier {tier} ({label}): {by_tier[tier]}")
    print(f"\nBy Priority:")
    for p in sorted(by_priority):
        print(f"  {p}: {by_priority[p]}")
    print(f"\nCode Changes:")
    print(f"  Env-var only (no code): {env_only}")
    print(f"  Needs code changes: {needs_code}")

    print(f"\n{'='*60}")
    print(f" RECOMMENDED EXECUTION ORDER")
    print(f"{'='*60}")
    print("""
Phase 1 (Day 1-2): Env-var only, P0 strategies -- $5 total
  - Run all tier-0/P0 strategies through qualify (8xH100, 3min each)
  - a1-stack-quickwins, a1-sqrt-cooldown, a1-adaptive-gptq
  - a1-batch-warmup-030, a12-combo-best-v1/v2/v3
  - These can run in parallel (7 strategies x $2 = $14)

Phase 2 (Day 2-3): Code changes for SLOT -- $10 total
  - Implement eval_val_sliding_etlb (SLOT)
  - Run b1-slot-basic, b1-slot-persample, b1-slot-context-only
  - Run b7-full-stack-best (everything combined)

Phase 3 (Day 3-5): P1 HP sweeps + arch combos -- $30 total
  - All P1 env-var strategies through qualify
  - Top 3 promoted to full runs ($5 each)

Phase 4 (Day 5-8): Novel techniques -- $40 total
  - Implement n-gram cache (b5-ngram-cache) if legality confirmed
  - Implement Huber decay, IFNSO, Mousse
  - Run all P1 code-change strategies

Phase 5 (Day 8-14): Polish & seed variance -- $20 total
  - 3-seed runs on top 3 configs
  - Full 10min eval runs
  - Final submission preparation

Total estimated cost: ~$120
""")

    # Print P0 strategies for immediate execution
    p0 = get_by_priority("P0")
    print(f"\n{'='*60}")
    print(f" P0 STRATEGIES ({len(p0)} total) -- RUN FIRST")
    print(f"{'='*60}")
    for s in p0:
        code_tag = " [CODE]" if s.needs_code else ""
        print(f"  {s.name:45s} {s.expected_mbpb:>20s} mBPB{code_tag}")


if __name__ == "__main__":
    print_summary()
