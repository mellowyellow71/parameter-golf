"""
P0 — Polar Express: Minimax-Optimal Batched Orthogonalization

Replaces the standard quintic Newton-Schulz iteration in Muon optimizer.
Standard NS has poor initial convergence when the starting matrix is far
from orthogonal. Polar Express (arXiv:2505.16932) solves a minimax
optimization at each iteration, guaranteeing worst-case optimal convergence.

Key properties:
  - Same structure as NS (matrix-matrix multiplies only, no SVD)
  - Minimax-optimized polynomial coefficients (not the fixed a,b,c = 3.4445, -4.7750, 2.0315)
  - Dynamically adapted polynomial for each iteration step
  - Numerically stable in BF16 (critical for memory bandwidth constraints)
  - Converges in fewer effective iterations than standard NS

Implementation: Fused Triton kernel over batched Parameter Banks.
Instead of 120 sequential tiny GEMMs, runs 3 batched calls (one per shape group):

    for bank in [q_bank, kv_bank, mlp_bank]:
        update = batched_polar_express(bank.grad, steps=5)

The batched GEMMs (e.g., (11, 512, 512) via torch.bmm) actually utilize
the GPU — individual (512, 512) matmuls use <1% of H100 capacity.

Paper: arXiv:2505.16932 (The Polar Express: Optimal Matrix Sign Methods)

Owner: TBD
Est. savings: Part of the ~8ms optimizer savings (combined with Parameter Banking)
"""
