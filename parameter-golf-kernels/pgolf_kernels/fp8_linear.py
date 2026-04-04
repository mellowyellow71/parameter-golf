"""
P0 — FP8 Forward Pass Investigation

STATUS: FP8 CANNOT provide lossless speedup.

Measured on H100 NVL (torch 2.6+cu126):

  Pure FP8 GEMM speedup: 1.28-1.57x over BF16 cuBLAS
  Per-step savings (forward only): 11.3ms (if casting is free)
  Net with cast overhead: ~5-6ms

  BUT: 3.7% mean relative error per GEMM (FP8 E4M3, 3 mantissa bits).
  This is INTRINSIC to the format — no scaling strategy fixes it:
    - Per-tensor scaling: 3.7% error
    - Per-row scaling: 3.7% error (rowwise not supported in torch 2.6 anyway)
    - Block-wise (128-element): 3.7% error, 9x slower

  For 99.99% match requirement: FP8 gives 370x too much error per GEMM.

  FP8 is only viable if we accept ~0.002 BPB training quality degradation
  (confirmed by research: arXiv:2502.05967). But user requires lossless.

KEY HARDWARE FINDINGS:
  - torch._scaled_mm supports only scalar scales in torch 2.6 (no rowwise)
  - E4M3 beats E5M2 in practice (better scaling properties)
  - use_fast_accum=False improves precision marginally (0.014ms cost)
  - Fused RMSNorm->FP8 Triton kernel: 7x faster than separate ops (0.103ms vs 0.725ms)
  - But the downstream GEMM error dominates, not the cast precision

POTENTIAL PATH (requires user approval of quality tradeoff):
  If ~0.002 BPB degradation is acceptable:
  - Use static unit scaling (arXiv:2502.05967)
  - Pre-cast weights to FP8 (amortized over forward pass)
  - Fuse input cast into RMSNorm kernel (saves 0.6ms per call)
  - FP8 forward + BF16 backward
  - Net savings: ~11ms/step
"""
