# parameter-golf-kernels

Custom H100-tuned kernel library for the [Parameter Golf](https://github.com/openai/parameter-golf) challenge.

**Goal:** Maximize training steps in 600 seconds for the SOTA transformer (PR #315) on 8×H100 SXM. Current: ~7,050 steps at ~85ms/step. **Revised target: ~15,000+ steps at ~32ms/step** — research shows the initial 65ms target was conservative. NanoGPT speedrun trains 124M params at 21-25ms/step on identical hardware; our 26.8M model should target 30-35ms.

## Critical Architectural Constraints

1. **`torch.compile(model, fullgraph=True)`** — The entire forward+backward is compiled. Custom kernels MUST use **`torch.library.triton_op`** (NOT `custom_op` — custom_op makes kernels opaque to Inductor, preventing fusion). `triton_op` lets Inductor trace the Triton AST for macro-kernel fusion.

2. **Compilation warmup is a SILENT KILLER** — `torch.compile` with `max-autotune` can take 3-5 minutes to JIT. In a 600s budget, that's 30% lost. Must use **PT2 archives** for pre-compilation.

3. **`mode='max-autotune'`** forces Inductor to capture the entire step as a **CUDA Graph**, driving CPU launch overhead to exactly zero. Requires no Python control flow in the hot path.

## Target Architecture (PR #315, 1.1248 BPB)

Verified from PR #315's actual `train_gpt.py`. Run config: `NUM_LAYERS=11`.

- **11 layers**, 512 dim, 8 heads, 4 KV heads (GQA), head_dim=64
- 3× MLP (hidden=1536), relu² activation
- **U-Net skip**: 5 encoder, 6 decoder layers, 5 skip connections
- **SmearGate**: applied ONCE after embedding (not per-layer)
- **resid_mix**: applied at START of each Block (before attention)
- **LN Scale**: `1/sqrt(layer_idx + 1)` scalar on RMSNorm output
- **XSA** on last 4 layers: projects out self-value from attention output
- **Partial RoPE**: rotary on first 16 of 64 head dims
- **EMA** (β=0.997) every step in FP32
- **Late QAT**: Int6 STE only when lr_scale < 0.1
- Muon optimizer (5-step Newton-Schulz) + AdamW for 1D params
- Flash Attention 3, Bigram hash (2048 × 128 → 512 projection)
- Logit softcap 30.0, tied FP16 embeddings, zstd-22
- `torch.compile(fullgraph=True)` on model, eval, AND Newton-Schulz

### Per-GPU Shapes (B×T = 98,304)

786K batch / 8 GPUs = 98,304 tokens → 48 sequences × 2048 = (48, 2048, 512)

---

## Baseline Time Breakdown (85ms, unoptimized)

| Phase | Time | Bottleneck |
|-------|------|-----------|
| Forward (BF16 GEMMs + FA3 + elementwise) | ~28ms | cuBLAS dispatch, memory bandwidth |
| Backward (2× forward GEMMs + FA3 bwd) | ~42ms | Synchronous DDP, inefficient comm overlap |
| Optimizer (unbatched NS, 120 tiny launches) | ~10ms | Kernel launch overhead, <1% GPU utilization |
| Comm + EMA | ~5ms | Synchronous NCCL barriers |

## Projected Optimized Breakdown (~32ms)

| Phase | Time | Optimizations Applied |
|-------|------|-----------------------|
| Forward | ~10.5ms | **FP8 GEMMs** (2× TFLOPS), fused CE, FA3 autotuning |
| Backward | ~18ms | TMA/WGMMA for asymmetric shapes, triton_op fusion |
| Optimizer | ~2.5ms | **Parameter Banking** + Batched **Polar Express** |
| Comm + EMA | ~1ms | **Parallel Muon** async overlap, fused EMA |

---

## Components (Final Priority Order)

### P0 — K4: FP8 Forward Pass (~12-15ms savings)

**THE SINGLE BIGGEST WIN.** H100 does 1,979 TFLOPS in FP8 vs 990 in BF16 — forward GEMMs nearly halve.

**Key insight from research:** FP8 training is viable for sub-100M models with <0.002 BPB degradation. The relu² dynamic range concern is solved by **static μnit Scaling** (arXiv:2502.05967) — no dynamic per-tensor scaling needed.

**Implementation pattern (from modded-nanogpt CastedLinearT):**
```python
# Static pre-scaling before FP8 cast — avoids ±448 overflow from relu²
x_scaled = x / x_s                          # pre-divide by static scale
x_fp8 = x_scaled.to(torch.float8_e4m3fn)   # cast to FP8
out = torch._scaled_mm(x_fp8, w_fp8, scale_a=x_s, scale_b=w_s)  # Tensor Core FP8 GEMM
```

- Forward: FP8 via `torch._scaled_mm` with static scales
- Backward: stays BF16 for gradient precision
- Fuse relu² into a Triton kernel that processes in higher precision within SRAM
- Fully orthogonal to Late QAT (QAT replaces FP8 scaling only in final 4% of steps)

**Paper: arXiv:2502.05967 (μnit Scaling)**

### P0 — K1: Parameter Banking + Batched Polar Express (~8ms savings)

**Two changes, inseparable:**

**A. Parameter Banking (architecture refactor):**
Instead of 22 separate `nn.Linear` layers with independent weight tensors, store weights as contiguous 3D parameter banks:

```python
# OLD: 22 separate (512, 512) weight tensors
self.c_q = nn.Linear(512, 512)  # per layer, separate allocation

# NEW: single contiguous bank
self.q_bank = nn.Parameter(torch.zeros(11, 512, 512))  # all layers, contiguous
# Forward: F.linear(x, self.q_bank[layer_idx])
```

This gives the optimizer a pre-batched tensor — **zero stack/unstack overhead**. Critical: `torch.stack` in the optimizer loop would negate all batching gains.

**B. Polar Express (replace Newton-Schulz):**
The standard quintic NS has poor initial convergence. Polar Express (arXiv:2505.16932) uses minimax-optimized polynomials that guarantee worst-case optimal convergence at every iteration. Stays in BF16. Implemented as fused Triton kernel over the batched parameter banks.

```python
# OLD: 120 sequential tiny GEMMs per rank
for W in matrices:
    X = newton_schulz_5step(W.grad)  # 15 launches each

# NEW: 3 batched Polar Express calls (one per shape group)
for bank in [q_bank, kv_bank, mlp_bank]:
    X = batched_polar_express(bank.grad)  # 15 batched launches total
```

**Papers: arXiv:2505.16932 (Polar Express), arXiv:2511.07464 (Parallel Muon)**

### P0 — NEW: Pre-Compilation + FA3 Autotuning

**A. PT2 Archive pre-compilation:**
`torch.compile` with `max-autotune` takes 3-5 minutes to JIT. In 600s, this kills ~30% of the training budget. Solution: pre-compile to a PT2 archive, load instantly at runtime.

**B. FA3 tile autotuning:**
Default FA3 tile sizes are suboptimal for seq_len=2048. Using `ct_experimental.autotune_launch` with 64×64 tiles gives ~45% latency reduction in the attention mechanism. Verified from the modded-nanogpt community.

### P1 — K5: Parallel Muon (Async Communication) (~4-5ms savings)

**Scatter-Compute-Gather paradigm** (arXiv:2511.07464):

1. **Reduce-Scatter** gradients to rank-specific shards
2. **Local Polar Express** on each rank's shard only
3. **Async All-Gather** orthogonalized updates back to all ranks

Overlaps NS computation with all-gather using `dist.all_gather(async_op=True)`. Each rank only orthogonalizes its own shard, then communication of layer N hides behind computation of layer N+1.

### P1 — K3: Fused Cross-Entropy via `triton_op` (~2-3ms savings)

Fuse logit softcap + cross-entropy forward + backward into single Triton kernel. V=1024 fits in SRAM registers (4KB per row).

**MUST use `torch.library.triton_op`** (NOT `custom_op`):
- `custom_op` → kernel is opaque to Inductor → no fusion → graph breaks
- `triton_op` → Inductor traces Triton AST → macro-kernel fusion → no graph breaks

Eliminates ~200MB intermediate tensor write to HBM.

### P2 — K2: TMA GEMM Tuning (only if profiling shows need)

Only pursue for the (98304, 512, 1536) MLP shapes IF profiling shows cuBLAS < 650 TFLOPS (65% of H100 peak). Options in order:
1. `torch.cuda.tunable_op` (cuBLAS algorithm selector, zero code)
2. CUTLASS 3.x with Hopper TMA + WGMMA Ping-Pong persistent kernels
3. Triton TMA kernels (1.4-2.2× over cuBLAS for tall-skinny shapes, per research)

---

## What to CUT (from previous plan)

| Cut | Why |
|-----|-----|
| `torch.library.custom_op` | Makes kernels opaque to Inductor. Use `triton_op` instead. |
| Standard quintic Newton-Schulz | Poor initial convergence. Replace with Polar Express minimax polynomials. |
| `torch.stack`/`unstack` in optimizer | Memory allocation kills batching gains. Refactor model to use Parameter Banks. |
| Separate EMA pass | Fuse EMA multiply-add into the optimizer parameter update kernel. Eliminates 214MB HBM round-trip. |
| C++ training loop (old K5) | CUDA Graphs via `mode='max-autotune'` eliminates ALL CPU overhead. |
| Explicit elementwise fusion (old K3) | Inductor already fuses these. Only intervene if profiling shows specific failures. |

---

## Correct Operation Sequence (Block.forward, verified from PR #315)

```
1. resid_mix: x = mix[0] * x + mix[1] * x0
2. attn_norm: RMSNorm(x) * (1/sqrt(layer_idx+1))
3. Q,K,V projections: 3 GEMMs (→ FP8 in K4)
4. Q,K RMSNorm
5. Partial RoPE (first 16 of 64 dims)
6. Q gain scaling
7. Flash Attention 3 (→ autotuned tiles)
8. XSA (last 4 layers): project out self-value
9. Output projection: 1 GEMM (→ FP8 in K4)
10. Residual: x = x + attn_scale * attn_out
11. mlp_norm: RMSNorm(x) * (1/sqrt(layer_idx+1))
12. MLP up: 1 GEMM (→ FP8 in K4)
13. relu²: relu(x).square() (→ fused Triton kernel for FP8 safety)
14. MLP down: 1 GEMM (→ FP8 in K4)
15. Residual: x = x + mlp_scale * mlp_out
```

SmearGate: ONCE at embedding. U-Net skip: weighted adds between encoder/decoder.

---

## Key Papers

| Paper | Relevance |
|-------|-----------|
| arXiv:2505.16932 (Polar Express) | Minimax-optimal NS replacement for Muon |
| arXiv:2502.05967 (μnit Scaling) | FP8 training without dynamic scaling |
| arXiv:2511.07464 (Parallel Muon) | Async scatter-compute-gather for distributed Muon |
| arXiv:2603.09078 (XSA) | Exclusive Self-Attention in SOTA architecture |

## Dependencies

- PyTorch >= 2.4 (for `triton_op`, `torch._scaled_mm`, PT2 archives)
- Triton >= 3.0
- CUDA >= 12.4
- Flash Attention 3 with autotuning support
- CUTLASS 3.x (optional, for K2)
