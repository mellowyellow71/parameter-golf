"""
Profile the actual compiled training step to find where time goes.

DO THIS BEFORE WRITING ANY CUSTOM KERNELS.

Usage:
    python bench/profile_step.py --script path/to/train_gpt.py --num_steps 20

Outputs:
    - Per-kernel timing breakdown (GEMM, FA3, elementwise, optimizer)
    - Memory allocation trace
    - GPU utilization per-phase
    - Identification of bottleneck kernels
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Profile Parameter Golf training step")
    parser.add_argument("--script", type=str, required=True, help="Path to train_gpt.py")
    parser.add_argument("--num_steps", type=int, default=20, help="Steps to profile (skip first 5 for warmup)")
    parser.add_argument("--output", type=str, default="profile_output", help="Output directory")
    args = parser.parse_args()

    print("Profile step harness — TODO: implement")
    print(f"  Script: {args.script}")
    print(f"  Steps: {args.num_steps}")
    print()
    print("Implementation plan:")
    print("  1. Import and initialize the model + optimizer from train_gpt.py")
    print("  2. Run 5 warmup steps (torch.compile JIT + CUDA warmup)")
    print("  3. Profile num_steps with torch.profiler.profile()")
    print("  4. Export Chrome trace + summary table")
    print("  5. Group kernels by: forward GEMM, backward GEMM, FA3, elementwise,")
    print("     Muon NS, Muon comm, DDP comm, EMA, grad clip, data loading")
    print()
    print("Key questions to answer:")
    print("  - What % of step time is GEMMs vs FA3 vs elementwise vs optimizer?")
    print("  - Which specific GEMM shapes are below 80% peak TFLOPS?")
    print("  - How much time does Muon NS take per rank?")
    print("  - Is Muon's all_reduce overlapped or blocking?")
    print("  - What does torch.compile's Inductor actually generate for elementwise chains?")


if __name__ == "__main__":
    main()
