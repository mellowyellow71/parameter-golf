"""
P1 — Fused Cross-Entropy with Logit Softcap (Forward + Backward)

Fuse softcap + softmax + cross-entropy + backward into one Triton kernel
per row. V=1024 fits in SRAM registers (4KB per row). Eliminates ~200MB
intermediate tensor materialization.

CRITICAL: Must use torch.library.triton_op (NOT custom_op).
  - custom_op: kernel is opaque to Inductor, prevents fusion, causes graph breaks
  - triton_op: Inductor traces the Triton AST, enables macro-kernel fusion,
    satisfies fullgraph=True constraint

Forward per row:
  1. Load 1024 logits (in registers)
  2. Softcap: 30 * tanh(logits / 30)
  3. Online softmax (numerically stable, single pass)
  4. loss = log_sum_exp - target_logit

Backward per row:
  1. Recompute softcap + softmax (from saved row_max)
  2. grad = softmax - one_hot(target)
  3. Chain rule through tanh: d(softcap)/d(raw) = 1 - tanh^2

Owner: TBD
Est. savings: 2-3ms/step (200MB HBM elimination + Inductor fusion with adjacent ops)
"""
