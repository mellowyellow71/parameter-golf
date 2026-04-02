"""
kernels.py  —  SM86 (Ampere / RTX 3060) Triton kernels for parameter-golf

Exports
-------
fused_softcap_cross_entropy(logits, targets, softcap) -> scalar loss
    Fuses softcap(tanh) + numerically-stable softmax + NLL into one kernel.
    Eliminates the large (B*T, vocab) intermediate float32 tensor.

tuned_linear(x, weight) -> Tensor
    Drop-in for F.linear(x, weight) (no bias).
    Forward uses Triton GEMM with Ampere-tuned block configs.
    Backward uses standard cuBLAS (via PyTorch) for correctness.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor


# ─────────────────────────────────────────────────────────────────────────────
# 1. Fused Softcap + Cross-Entropy
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def _fused_ce_fwd(
    logits_ptr, targets_ptr, losses_ptr,
    n_cols: tl.constexpr,
    softcap: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(
        logits_ptr + row * n_cols + cols, mask=mask, other=-float("inf")
    ).to(tl.float32)

    # softcap
    x = softcap * tl.extra.cuda.libdevice.tanh(x * (1.0 / softcap))

    # numerically-stable softmax denominator
    m = tl.max(tl.where(mask, x, -float("inf")), axis=0)
    e = tl.where(mask, tl.exp(x - m), 0.0)
    z = tl.sum(e, axis=0)

    target = tl.load(targets_ptr + row)
    t_logit = tl.sum(tl.where(mask & (cols == target), x - m, 0.0), axis=0)

    tl.store(losses_ptr + row, tl.log(z) - t_logit)


@triton.jit
def _fused_ce_bwd(
    logits_ptr, targets_ptr, scale_ptr, dlogits_ptr,
    n_cols: tl.constexpr,
    softcap: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(
        logits_ptr + row * n_cols + cols, mask=mask, other=0.0
    ).to(tl.float32)

    th = tl.extra.cuda.libdevice.tanh(x * (1.0 / softcap))
    sc_x = softcap * th

    m = tl.max(tl.where(mask, sc_x, -float("inf")), axis=0)
    e = tl.where(mask, tl.exp(sc_x - m), 0.0)
    z = tl.sum(e, axis=0)
    sm = tl.where(mask, e / z, 0.0)

    target = tl.load(targets_ptr + row)
    is_target = mask & (cols == target)

    # d(CE)/d(sc_x) = softmax - one_hot
    dce = sm - tl.where(is_target, 1.0, 0.0)
    # d(softcap*tanh(x/softcap))/dx = 1 - tanh²
    dsc = 1.0 - th * th

    scale = tl.load(scale_ptr)  # grad_output / n_rows
    dx = (dce * dsc * scale).to(tl.bfloat16)
    tl.store(dlogits_ptr + row * n_cols + cols, dx, mask=mask)


class _FusedSoftcapCEFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits: Tensor, targets: Tensor, softcap: float) -> Tensor:
        n_rows, n_cols = logits.shape
        BLOCK = triton.next_power_of_2(n_cols)
        losses = torch.empty(n_rows, dtype=torch.float32, device=logits.device)
        _fused_ce_fwd[(n_rows,)](
            logits, targets, losses,
            n_cols=n_cols, softcap=softcap, BLOCK=BLOCK,
        )
        ctx.save_for_backward(logits, targets)
        ctx.softcap = softcap
        ctx.BLOCK = BLOCK
        ctx.n_rows = n_rows
        return losses.mean()

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        logits, targets = ctx.saved_tensors
        n_rows, n_cols = logits.shape
        dlogits = torch.empty_like(logits)
        # scale = upstream_grad / n_rows  (mean-reduction distributes equally)
        scale = (grad_output / n_rows).contiguous()
        _fused_ce_bwd[(n_rows,)](
            logits, targets, scale, dlogits,
            n_cols=n_cols, softcap=ctx.softcap, BLOCK=ctx.BLOCK,
        )
        return dlogits, None, None


def fused_softcap_cross_entropy(
    logits: Tensor, targets: Tensor, softcap: float
) -> Tensor:
    """
    Fused softcap + softmax + NLL.
    logits:  (N, vocab)  bfloat16
    targets: (N,)        int64
    returns: scalar float32 loss
    """
    return _FusedSoftcapCEFn.apply(logits, targets, softcap)


# ─────────────────────────────────────────────────────────────────────────────
# 2. SM86-tuned Triton GEMM  (forward fast, backward delegates to cuBLAS)
# ─────────────────────────────────────────────────────────────────────────────

# Block configs tuned for Ampere (SM86): favour 4-stage pipelines,
# L2-friendly group ordering, and the specific tile sizes that hit the
# tensor-core shape requirements on GA106.
_SM86_CONFIGS = [
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 32, "GROUP": 8}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 32, "GROUP": 8}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP": 8}, num_stages=4, num_warps=8),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 32, "GROUP": 8}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 64, "GROUP": 8}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 64, "GROUP": 8}, num_stages=4, num_warps=8),
    triton.Config({"BLOCK_M": 32,  "BLOCK_N": 64,  "BLOCK_K": 32, "GROUP": 8}, num_stages=5, num_warps=2),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 32,  "BLOCK_K": 32, "GROUP": 8}, num_stages=5, num_warps=2),
]


@triton.autotune(configs=_SM86_CONFIGS, key=["M", "N", "K"])
@triton.jit
def _matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    sa_m, sa_k,   # A strides
    sb_k, sb_n,   # B strides (B is W stored as (N,K); we access as (K,N))
    sc_m, sc_n,   # C strides
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP: tl.constexpr,
):
    # L2-friendly swizzled tile ordering
    pid = tl.program_id(0)
    n_tiles_m = tl.cdiv(M, BLOCK_M)
    n_tiles_n = tl.cdiv(N, BLOCK_N)
    tiles_per_group = GROUP * n_tiles_n
    group_id = pid // tiles_per_group
    first_m = group_id * GROUP
    group_sz = tl.minimum(n_tiles_m - first_m, GROUP)
    pid_m = first_m + (pid % group_sz)
    pid_n = (pid % tiles_per_group) // group_sz

    rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    rn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    rk = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + rm[:, None] * sa_m + rk[None, :] * sa_k
    # B is weight (N,K); read as (K,N) by swapping strides
    b_ptrs = B_ptr + rk[:, None] * sb_n + rn[None, :] * sb_k

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_off in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = (k_off * BLOCK_K + rk) < K
        a = tl.load(a_ptrs, mask=k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * sa_k
        b_ptrs += BLOCK_K * sb_n

    c = acc.to(tl.bfloat16)
    c_ptrs = C_ptr + rm[:, None] * sc_m + rn[None, :] * sc_n
    m_mask = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) < M
    n_mask = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) < N
    tl.store(c_ptrs, c, mask=m_mask[:, None] & n_mask[None, :])


def _triton_mm(x: Tensor, weight: Tensor) -> Tensor:
    """x (M,K) @ weight.T (K,N)  →  (M,N)  in bfloat16."""
    x = x.contiguous()
    w = weight.to(torch.bfloat16).contiguous()  # (N, K)
    M, K = x.shape
    N = w.shape[0]
    out = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
    _matmul_kernel[grid](
        x, w, out,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),   # sb_k=w.stride(0)=K, sb_n=w.stride(1)=1
        out.stride(0), out.stride(1),
    )
    return out


class _TunedLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor) -> Tensor:
        orig = x.shape
        x2 = x.reshape(-1, orig[-1]).to(torch.bfloat16)
        out = _triton_mm(x2, weight)
        ctx.save_for_backward(x2, weight)
        ctx.orig_shape = orig  # preserve for backward reshape
        return out.reshape(*orig[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, grad: Tensor):
        x2, weight = ctx.saved_tensors
        g = grad.reshape(-1, grad.shape[-1]).to(torch.float32)
        dx = torch.mm(g, weight.to(torch.float32)).to(x2.dtype).reshape(ctx.orig_shape)
        dw = torch.mm(g.t(), x2.to(torch.float32)).to(weight.dtype)
        return dx, dw


def tuned_linear(x: Tensor, weight: Tensor) -> Tensor:
    """
    Drop-in for F.linear(x, weight) (no bias).
    Forward: Triton GEMM with SM86-autotuned tile configs.
    Backward: standard cuBLAS.

    weight may be float32 (bank weights) — cast to bf16 happens inside.
    """
    return _TunedLinearFn.apply(x, weight)
