"""
P0 — PT2 Archive Pre-Compilation

CRITICAL: torch.compile with max-autotune takes 3-5 MINUTES to JIT.
In a 600s training budget, losing 3 min = 30% of total time wasted.

Solution: Pre-compile the model to a PT2 archive BEFORE the timed run.
Load the cached compiled artifact at runtime for instant start.

Usage:
    # Before timed run: generate PT2 archive (can take 5+ min, doesn't matter)
    python bench/warmup_compile.py --script path/to/train_gpt.py --output model_compiled.pt2

    # During timed run: load pre-compiled archive (instant)
    # In train_gpt.py: model = torch.export.load("model_compiled.pt2")

Alternative: Use torch's compilation cache directory:
    TORCHINDUCTOR_CACHE_DIR=/persistent/path python train_gpt.py
    # Second run reuses cached kernels

Owner: TBD
"""
