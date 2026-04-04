"""
Benchmark GEMMs for all Parameter Golf shapes.

Shapes verified from PR #315 train_gpt.py with NUM_LAYERS=11, B*T=98304.

Usage:
    python bench/bench_gemm.py              # all shapes
    python bench/bench_gemm.py --shape G3   # specific shape

Requires H100 GPU.
"""

import argparse
import time

import torch

# Forward pass GEMM shapes (verified from code)
FORWARD_SHAPES = {
    "Q_proj":    (98304, 512, 512),    # c_q: (B*T, dim) x (dim, dim)
    "K_proj":    (98304, 512, 256),    # c_k: (B*T, dim) x (dim, kv_dim)
    "V_proj":    (98304, 512, 256),    # c_v: same as K
    "Out_proj":  (98304, 512, 512),    # proj: (B*T, dim) x (dim, dim)
    "MLP_up":    (98304, 512, 1536),   # mlp.fc: (B*T, dim) x (dim, hidden)
    "MLP_down":  (98304, 1536, 512),   # mlp.proj: (B*T, hidden) x (hidden, dim)
    "LM_head":   (98304, 512, 1024),   # F.linear(x, tok_emb.weight)
    "Bigram":    (98304, 128, 512),    # bigram.proj: (B*T, bigram_dim) x (bigram_dim, dim)
}

# Muon Newton-Schulz GEMM shapes (the NS iteration GEMMs, not the weight shapes)
MUON_NS_SHAPES = {
    "NS_512x512_AAt":   (512, 512, 512),    # A = X @ X.T for (512,512) matrices
    "NS_512x1536_AAt":  (512, 512, 512),    # A = X @ X.T for (512,1536) — X is transposed to (512,1536)
    "NS_512x1536_BX":   (512, 512, 1536),   # B @ X for (512,1536) matrices
    "NS_256x512_AAt":   (256, 256, 256),    # A = X @ X.T for K/V (256,512)
    "NS_256x512_BX":    (256, 256, 512),    # B @ X for K/V matrices
}

CALLS_PER_STEP = {
    "Q_proj": 11, "K_proj": 11, "V_proj": 11, "Out_proj": 11,
    "MLP_up": 11, "MLP_down": 11, "LM_head": 1, "Bigram": 1,
}


def bench_torch_mm(M, K, N, warmup=10, iters=100):
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    for _ in range(warmup):
        torch.mm(a, b)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        torch.mm(a, b)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def bench_torch_bmm(B, M, K, N, warmup=10, iters=100):
    """Benchmark batched GEMM for Muon NS batching."""
    a = torch.randn(B, M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(B, K, N, device="cuda", dtype=torch.bfloat16)
    for _ in range(warmup):
        torch.bmm(a, b)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        torch.bmm(a, b)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", type=str, default=None)
    parser.add_argument("--muon", action="store_true", help="Benchmark Muon NS shapes")
    parser.add_argument("--batched", action="store_true", help="Benchmark batched Muon NS")
    args = parser.parse_args()

    if args.batched:
        print("Batched Muon NS GEMMs (K1 optimization target)")
        print(f"{'Group':>15} {'Batch':>5} {'Shape':>20} {'BMM (ms)':>10} {'Loop (ms)':>10} {'Speedup':>8}")
        print("-" * 75)
        for name, B, M, K, N in [
            ("512x512 grads", 22, 512, 512, 512),
            ("256x512 grads", 22, 256, 256, 256),
            ("512x1536 grads", 22, 512, 512, 1536),
        ]:
            bmm_ms = bench_torch_bmm(B, M, K, N)
            loop_ms = sum(bench_torch_mm(M, K, N, warmup=3, iters=10) for _ in range(B))
            print(f"{name:>15} {B:>5} {str((M,K,N)):>20} {bmm_ms:>9.4f} {loop_ms:>9.4f} {loop_ms/bmm_ms:>7.2f}x")
        return

    shapes = MUON_NS_SHAPES if args.muon else FORWARD_SHAPES

    print(f"{'Name':>20} {'(M, K, N)':>25} {'cuBLAS (ms)':>12} {'TFLOPS':>8} {'% H100 peak':>12}")
    print("-" * 80)

    H100_PEAK_BF16 = 990  # TFLOPS

    for name, (M, K, N) in shapes.items():
        ms = bench_torch_mm(M, K, N)
        flops = 2 * M * K * N
        tflops = flops / ms / 1e9
        pct = tflops / H100_PEAK_BF16 * 100
        calls = CALLS_PER_STEP.get(name, "")
        print(f"{name:>20} {str((M,K,N)):>25} {ms:>11.4f} {tflops:>7.1f} {pct:>10.1f}%")


if __name__ == "__main__":
    main()
