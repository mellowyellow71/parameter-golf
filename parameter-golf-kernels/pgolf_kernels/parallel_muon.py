"""
P1 — Parallel Muon: Async Scatter-Compute-Gather Optimizer

Standard Muon forces a hard synchronization wall: all-reduce gradients,
then ALL ranks run Newton-Schulz on ALL matrices. This wastes both
compute (redundant work) and bandwidth (synchronous blocking).

Parallel Muon (arXiv:2511.07464) restructures as:

  1. Reduce-Scatter: gradients reduced to rank-specific shards
     (each rank owns 1/8 of the parameter banks)
  2. Local Polar Express: each rank orthogonalizes ONLY its own shard
     (no redundant computation across ranks)
  3. Async All-Gather: orthogonalized updates gathered back to all ranks
     using dist.all_gather(async_op=True)

Key: communication of layer N overlaps with computation of layer N+1.
handle.wait() only blocks when the next layer's compute needs the result.

Benefits:
  - Compute: each rank does 1/8 the NS work (8 matrices, not 66)
  - Bandwidth: BF16 all-gather is half the bytes of FP32 Adam states
  - Overlap: NS compute hides behind async all-gather

Requires Parameter Banking (parameter_bank.py) for contiguous shards.

Paper: arXiv:2511.07464 (Parallel Muon)

Owner: TBD
Est. savings: 4-5ms/step
"""
