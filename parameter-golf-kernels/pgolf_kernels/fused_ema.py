"""
Fused EMA: Merge Exponential Moving Average into Optimizer Step

Current: EMA runs as a separate pass over all 26.8M params after the
optimizer step. This reads + writes ~214MB to HBM unnecessarily.

Solution: Fuse the EMA multiply-add into the optimizer's parameter
update kernel. When the optimizer writes updated params to HBM, it
simultaneously computes:

    ema[name] = decay * ema[name] + (1 - decay) * param

This happens in the same SRAM register lifecycle — zero extra HBM
round-trip. The EMA buffer is FP32 (critical — BF16 EMA causes
catastrophic accumulation errors, discovered in PR #199).

Implementation: Add EMA pointer + decay constant to the Polar Express
or parameter update Triton kernel. After writing param, write EMA in
the same kernel launch.

Owner: TBD
Est. savings: ~0.5-1ms/step (small but free when fused)
"""
