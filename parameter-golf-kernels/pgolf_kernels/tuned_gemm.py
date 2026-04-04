"""
P2 — Triton GEMM tuned for exact Parameter Golf shapes.

cuBLAS hits 45-56% of H100 NVL peak (835 TFLOPS BF16).
Can Triton do better for these specific shapes?
"""

import torch
import triton
import triton.language as tl
import time


@triton.autotune(
    configs=[
        triton.Config({"BM": 128, "BN": 256, "BK": 64, "G": 8}, num_stages=3, num_warps=8),
        triton.Config({"BM": 256, "BN": 128, "BK": 64, "G": 8}, num_stages=3, num_warps=8),
        triton.Config({"BM": 128, "BN": 128, "BK": 64, "G": 8}, num_stages=4, num_warps=4),
        triton.Config({"BM": 128, "BN": 128, "BK": 32, "G": 8}, num_stages=4, num_warps=4),
        triton.Config({"BM": 64, "BN": 256, "BK": 64, "G": 8}, num_stages=4, num_warps=4),
        triton.Config({"BM": 256, "BN": 64, "BK": 64, "G": 8}, num_stages=4, num_warps=4),
        triton.Config({"BM": 64, "BN": 128, "BK": 64, "G": 8}, num_stages=4, num_warps=4),
        triton.Config({"BM": 128, "BN": 64, "BK": 64, "G": 8}, num_stages=4, num_warps=4),
        triton.Config({"BM": 256, "BN": 256, "BK": 32, "G": 8}, num_stages=3, num_warps=8),
        triton.Config({"BM": 256, "BN": 256, "BK": 64, "G": 4}, num_stages=3, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    sAM, sAK, sBK, sBN, sCM, sCN,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, G: tl.constexpr,
):
    pid = tl.program_id(0)
    num_bm = tl.cdiv(M, BM)
    num_bn = tl.cdiv(N, BN)
    num_in_group = G * num_bn
    gid = pid // num_in_group
    first = gid * G
    gsz = min(num_bm - first, G)
    pid_m = first + ((pid % num_in_group) % gsz)
    pid_n = (pid % num_in_group) // gsz

    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)

    A = A + (rm[:, None] * sAM + rk[None, :] * sAK)
    B = B + (rk[:, None] * sBK + rn[None, :] * sBN)

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BK)):
        a = tl.load(A, mask=rk[None, :] < K, other=0.0)
        b = tl.load(B, mask=rk[:, None] < K, other=0.0)
        acc = tl.dot(a, b, acc)
        A += BK * sAK
        B += BK * sBK
        rk += BK

    c = acc.to(tl.bfloat16)
    rm2 = pid_m * BM + tl.arange(0, BM)
    rn2 = pid_n * BN + tl.arange(0, BN)
    C = C + (rm2[:, None] * sCM + rn2[None, :] * sCN)
    mask = (rm2[:, None] < M) & (rn2[None, :] < N)
    tl.store(C, c, mask=mask)


def triton_mm(a, b):
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda META: (triton.cdiv(M, META["BM"]) * triton.cdiv(N, META["BN"]),)
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
    )
    return c


def benchmark():
    shapes = [
        ("Q/Out proj", 98304, 512, 512),
        ("K/V proj", 98304, 512, 256),
        ("MLP up", 98304, 512, 1536),
        ("MLP down", 98304, 1536, 512),
        ("LM head", 98304, 512, 1024),
    ]

    H100_NVL_BF16 = 835

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"\n{'Name':<15} {'cuBLAS ms':>10} {'Triton ms':>10} {'Speedup':>8} {'cuBLAS TF':>10} {'Triton TF':>10} {'cuBLAS %':>8} {'Triton %':>8}")
    print("-" * 90)

    for name, M, K, N in shapes:
        a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
        flops = 2 * M * K * N

        # cuBLAS
        for _ in range(5):
            torch.mm(a, b)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(50):
            torch.mm(a, b)
        torch.cuda.synchronize()
        cublas_ms = (time.perf_counter() - t0) / 50 * 1000

        # Triton (autotune warmup)
        for _ in range(5):
            triton_mm(a, b)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(50):
            triton_mm(a, b)
        torch.cuda.synchronize()
        triton_ms = (time.perf_counter() - t0) / 50 * 1000

        cublas_tf = flops / cublas_ms / 1e9
        triton_tf = flops / triton_ms / 1e9
        speedup = cublas_ms / triton_ms

        print(f"{name:<15} {cublas_ms:>9.3f} {triton_ms:>9.3f} {speedup:>7.2f}x {cublas_tf:>9.0f} {triton_tf:>9.0f} {cublas_tf/H100_NVL_BF16*100:>7.1f}% {triton_tf/H100_NVL_BF16*100:>7.1f}%")


if __name__ == "__main__":
    benchmark()
