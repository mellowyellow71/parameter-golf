# GEMM Optimization Plan: Closing the cuBLAS Utilization Gap

## Problem Statement

cuBLAS BF16 achieves only **45-56% of H100 NVL peak** (835 TFLOPS) on the exact GEMM shapes used in Parameter Golf training. A custom kernel hitting 75-80% peak would yield **1.4-1.6× lossless speedup** on the forward and backward pass, saving ~14ms per training step.

This is a pure kernel optimization problem: same BF16 precision, same math, just faster execution through better hardware utilization.

## Measured Baseline (H100 NVL, BF16, cuBLAS via torch.mm)

| ID | Shape (M, K) × (K, N) | ms/call | TFLOPS | % of 835 peak | Calls/step | ms/step |
|----|----------------------|---------|--------|---------------|------------|---------|
| G1 | (98304, 512) × (512, 512) | 0.131 | 395 | 47.2% | 66 | 8.6 |
| G2 | (98304, 512) × (512, 256) | 0.069 | 373 | 44.6% | 66 | 4.6 |
| G3 | (98304, 512) × (512, 1536) | 0.348 | 445 | 53.2% | 33 | 11.5 |
| G4 | (98304, 1536) × (1536, 512) | 0.331 | 467 | 55.9% | 33 | 10.9 |
| G5 | (98304, 512) × (512, 1024) | 0.254 | 406 | 48.7% | 3 | 0.8 |
| | | | | | **Total** | **36.4ms** |

### What We Already Tried (and failed)

- **Triton autotune GEMM**: 0.81-0.91× cuBLAS. Loses on every shape. Standard Triton without Hopper TMA cannot compete.
- **torch.compile max-autotune**: 0.45-0.82× cuBLAS. Inductor's generated Triton GEMM is worse.
- **torch.cuda.tunable_op**: Marginal improvement. cuBLAS heuristic selection is already near-optimal within cuBLAS's algorithm space.
- **FP8 _scaled_mm**: 1.28-1.57× but introduces 3.7% mean relative error per GEMM. Not lossless.

## Root Cause Analysis

### Why cuBLAS Underperforms on These Shapes

**1. Tall-skinny aspect ratio (M >> K, N)**

M=98304 is very large. K and N range from 256 to 1536. This creates an extreme aspect ratio. cuBLAS is optimized for roughly square matrices where M ≈ K ≈ N. For tall-skinny shapes, the tile decomposition along N produces very few tiles, limiting parallelism along that dimension.

For G1: (98304, 512) × (512, 512)
- With tile_M=128, tile_N=128: 768 × 4 = 3072 tiles
- H100 NVL has 132 SMs
- 3072 / 132 = 23.3 waves
- Last wave: only 40 of 132 SMs active (30% waste)

For G2: (98304, 512) × (512, 256)
- With tile_M=128, tile_N=128: 768 × 2 = 1536 tiles
- 1536 / 132 = 11.6 waves
- Last wave: 79 of 132 SMs active (40% waste)

**2. K-dimension is small**

K=512 means only 8 iterations of the inner loop at BK=64, or 4 at BK=128. This leaves very little room to hide global memory latency behind compute. The pipeline is frequently stalled waiting for data.

**3. Memory bandwidth pressure**

For G1 (M=98304, K=512, N=512):
- Read A: 98304 × 512 × 2 bytes = 100.7 MB
- Read B: 512 × 512 × 2 bytes = 0.5 MB
- Write C: 98304 × 512 × 2 bytes = 100.7 MB
- Total: 201.9 MB
- Arithmetic intensity: 51.5 GFLOPS / 201.9 MB = 255 FLOPS/byte
- H100 NVL balance point: 835 TFLOPS / 3.35 TB/s = 249 FLOPS/byte
- This is RIGHT at the boundary — partially memory-bandwidth limited

For G3 (M=98304, K=512, N=1536):
- Read A: 100.7 MB, Read B: 1.5 MB, Write C: 301.9 MB → 404 MB
- Arithmetic intensity: 154.6 GFLOPS / 404 MB = 383 FLOPS/byte
- Solidly compute-bound. Yet only 53% peak. Tile scheduling is the bottleneck.

**4. cuBLAS does not use Hopper TMA optimally**

The H100 Tensor Memory Accelerator (TMA) enables asynchronous bulk data movement from global HBM to shared SRAM without going through registers. cuBLAS heuristics may not select the TMA-optimized code path for these specific shapes. CUTLASS 3.x exposes TMA directly.

## Target Performance

| Shape | Current | Target (80% peak) | Speedup | Saved ms/step |
|-------|---------|-------------------|---------|---------------|
| G1 (98304,512,512) | 395 TF (47%) | 668 TF (80%) | 1.69× | 3.5 |
| G2 (98304,512,256) | 373 TF (45%) | 668 TF (80%) | 1.79× | 2.0 |
| G3 (98304,512,1536) | 445 TF (53%) | 668 TF (80%) | 1.50× | 3.8 |
| G4 (98304,1536,512) | 467 TF (56%) | 668 TF (80%) | 1.43× | 3.3 |
| G5 (98304,512,1024) | 406 TF (49%) | 668 TF (80%) | 1.65× | 0.3 |
| **Total** | | | | **12.9ms** |

## Implementation Approach: CUTLASS 3.x Persistent Kernel with TMA

### Why CUTLASS 3.x

CUTLASS (CUDA Templates for Linear Algebra Subroutines) is NVIDIA's open-source library for high-performance GEMM. Version 3.x has first-class Hopper support:

- **TMA (Tensor Memory Accelerator)**: Hardware-accelerated async global→shared memory copies. Eliminates register file pressure for data movement.
- **WGMMA (Warp Group MMA)**: Hopper's Tensor Core instruction operating on warp groups (128 threads). Wider than Ampere's mma instructions.
- **Persistent kernel paradigm**: One kernel launch, thread blocks fetch work from a global queue. Eliminates per-tile launch overhead.
- **Ping-pong scheduling**: Double-buffered shared memory with software pipelining. Computation on buffer A while loading buffer B.

### Kernel Design for Our Shapes

**Tile configuration (to be tuned per shape):**

For G3 (98304, 512, 1536) — our hottest shape:
- Tile: 128 × 128 × 64 (M × N × K)
- M tiles: 98304 / 128 = 768
- N tiles: 1536 / 128 = 12
- K tiles: 512 / 64 = 8
- Total tiles: 768 × 12 = 9216
- Waves: 9216 / 132 = 69.8 (good utilization, <1% last-wave waste)

For G1 (98304, 512, 512):
- Tile: 128 × 64 × 64 (smaller N tile since N=512 is small)
- M tiles: 768, N tiles: 8, Total: 6144
- Waves: 6144 / 132 = 46.5 (ok)
- OR: Tile 256 × 64 × 64
- M tiles: 384, N tiles: 8, Total: 3072
- Waves: 3072 / 132 = 23.3 (same wave quantization as cuBLAS — need different approach)

For G2 (98304, 512, 256):
- N=256 is very small. Tile 128 × 128 gives only 2 N-tiles.
- Better: 128 × 256 × 64 (cover full N in one tile)
- M tiles: 768, N tiles: 1, Total: 768
- Waves: 768 / 132 = 5.8 (good)
- But each tile does a 128×256 output — need to verify WGMMA supports this

**Split-K for memory-bound shapes:**

For shapes where K is small (512), split-K can increase parallelism:
- Split K into 2-4 segments, compute partial results in parallel, reduce
- Increases tile count, improves SM occupancy
- Adds a reduction step but this is cheap for large M

### Implementation Steps

1. **Set up CUTLASS 3.x build environment**
   - Clone CUTLASS repo, build against our CUDA 12.6/12.8
   - Verify Hopper SM90 target compiles

2. **Write kernel for G3 (hottest shape, 11.5ms/step)**
   - Start from CUTLASS 3.x Hopper GEMM example
   - Configure: BF16 input, BF16 output, FP32 accumulator
   - Tile: 128×128×64 with TMA + WGMMA
   - Benchmark vs cuBLAS

3. **Write kernel for G1 (second hottest, 8.6ms/step)**
   - Same approach, different tile configuration
   - Test split-K for the 512×512 case

4. **Write kernel for G4 (third hottest, 10.9ms/step)**
   - Transposed shape of G3, may need different tile layout

5. **Write kernel for G2 (fourth, 4.6ms/step)**
   - Smallest N=256, test 128×256 tile covering full N

6. **Package as PyTorch extension**
   - C++ wrapper exposing `tuned_mm(a, b)` function
   - Shape-based dispatch to the correct kernel
   - Fallback to cuBLAS for untuned shapes

### Alternative: cuBLASLt Direct API

Before writing CUTLASS kernels, try cuBLASLt with explicit algorithm selection:

```cpp
cublasLtMatmulAlgoGetHeuristic(...)  // Get all algorithms
// For each algorithm:
cublasLtMatmulAlgoConfigSetAttribute(algo, CUBLASLT_ALGO_CONFIG_TILE_ID, ...)
cublasLtMatmulAlgoConfigSetAttribute(algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, ...)
// Benchmark all combinations
```

This exhaustively searches cuBLAS's internal algorithm space. If an algorithm exists that hits 80% peak for our shapes, this will find it without writing a custom kernel.

## File Structure

```
pgolf_kernels/
├── csrc/
│   ├── gemm_kernels.cu          # CUTLASS 3.x kernel implementations
│   ├── gemm_dispatch.cpp        # Shape-based dispatch logic
│   └── cublaslt_tuner.cpp       # cuBLASLt exhaustive algorithm search
├── tuned_gemm.py                # Python interface
├── setup.py                     # Build configuration (sm_90 target)
```

## Build Requirements

- CUDA Toolkit >= 12.4 (for Hopper sm_90)
- CUTLASS 3.x (clone from github.com/NVIDIA/cutlass)
- PyTorch C++ extensions (`torch.utils.cpp_extension`)
- Compiler: nvcc with `-arch=sm_90a` for Hopper features

## Integration

```python
# In train_gpt.py, replace F.linear:
from pgolf_kernels import tuned_mm

class TunedLinear(nn.Module):
    def forward(self, x):
        return tuned_mm(x.reshape(-1, x.size(-1)), self.weight.t()).reshape(*x.shape[:-1], -1)
```

The kernel is a drop-in replacement for `torch.mm`. Same inputs, same outputs, same precision. Just faster.

## Verification

Every kernel must pass:
```python
ref = torch.mm(a, b)                    # cuBLAS reference
out = tuned_mm(a, b)                    # custom kernel
assert torch.allclose(ref, out, atol=1e-5, rtol=1e-3)  # BF16 precision
```

The tolerance accounts for BF16 non-associativity (different reduction order). The math is the same; only the schedule differs.

## Risk Assessment

- **High confidence**: cuBLASLt algorithm search (may find 70%+ utilization with zero custom code)
- **Medium confidence**: CUTLASS 3.x persistent kernel (proven in literature, but complex to implement)
- **Measured fallback**: If we only reach 65% peak (from 50%), that's still 1.3× = ~10ms saved. Worth it.

## Key References

- NVIDIA CUTLASS 3.x: github.com/NVIDIA/cutlass
- Hopper GEMM examples: cutlass/examples/48_hopper_warp_specialized_gemm/
- TMA documentation: CUDA Programming Guide §9.7.14
- cuBLASLt API: docs.nvidia.com/cuda/cublas/index.html#cublasLtMatmul
