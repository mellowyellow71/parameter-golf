"""
parameter-golf-kernels: Custom H100-tuned training pipeline for Parameter Golf.

Target: 8xH100 SXM, 11L/512d transformer, 600s training budget.
Current: ~85ms/step (~7,050 steps). Target: ~32ms/step (~15,000+ steps).

Components (priority order):
  P0: fp8_linear        — FP8 forward via torch._scaled_mm + static unit scaling
  P0: parameter_bank    — 3D contiguous weight banks (eliminates stack/unstack)
  P0: polar_express     — Minimax-optimal batched orthogonalization (replaces Newton-Schulz)
  P1: fused_cross_entropy — Softcap + CE fwd+bwd via triton_op (NOT custom_op)
  P1: parallel_muon     — Async scatter-compute-gather distributed optimizer
  P2: tuned_gemm        — Profile-guided GEMM tuning (only if cuBLAS < 650 TFLOPS)

Critical: All Triton kernels must use torch.library.triton_op for torch.compile compatibility.
"""

__version__ = "0.2.0"
