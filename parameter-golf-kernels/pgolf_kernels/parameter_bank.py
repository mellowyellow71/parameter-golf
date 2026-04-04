"""
P0 — Parameter Banking: Contiguous 3D Weight Banks

Instead of N separate nn.Linear layers with independent weight tensors,
store all same-shaped weights as a single contiguous 3D parameter:

    OLD: [layer.c_q.weight for layer in blocks]  # 11 separate (512, 512) tensors
    NEW: q_bank = nn.Parameter(zeros(11, 512, 512))  # one contiguous tensor

Why this matters:
  - Muon optimizer receives pre-batched tensors — zero stack/unstack overhead
  - torch.bmm on the batched bank is compute-bound (vs latency-bound per-matrix)
  - Eliminates ~120 kernel launches per rank in the optimizer step
  - Memory layout is contiguous — optimal for GPU cache lines

CRITICAL: torch.stack() in the optimizer loop negates ALL batching gains.
The model architecture must natively use 3D banks.

Shape groups (11 layers):
  q_bank:    (11, 512, 512)   — Q projections
  kv_bank:   (22, 256, 512)   — K + V projections (interleaved or separate)
  out_bank:  (11, 512, 512)   — output projections
  mlp_up:    (11, 1536, 512)  — MLP fc weights
  mlp_down:  (11, 512, 1536)  — MLP proj weights

Forward: F.linear(x, bank[layer_idx])
Optimizer: batched Polar Express on entire bank at once

Owner: TBD
Est. savings: Part of the ~8ms optimizer savings (combined with Polar Express)
"""
