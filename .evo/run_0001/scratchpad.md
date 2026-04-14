# Scratchpad

## Status
- Metric: `min`
- Current eval epoch: `1`
- Best score: `1.1236`
- Total experiments: `22`
- Committed: `6`
- Discarded: `8`
- Active workers: `3`

## Tree
```
root root gates=1
└── exp_0000 committed score=2.5399 epoch=1 winning SP8192 base: depth recurrence + parallel residuals + QK5.25 (smoke verified, train_loss=2.54@step200)
    ├── exp_0001 discarded score=99.0 epoch=1 PARALLEL_RESIDUAL_START=5: earlier parallel residuals for more attn/MLP specialization
    ├── exp_0002 discarded score=99.0 epoch=1 MATRIX_LR=0.025, SCALAR_LR=0.023: higher LR to compensate for SDPA vs flash-attn-3
    ├── exp_0003 discarded score=99.0 epoch=1 QK_GAIN_INIT=5.25: match latest competition record attention sharpness
    ├── exp_0004 discarded score=99.0 epoch=1 QK_GAIN_INIT=5.25: competition record uses 5.25 vs our 5.0, sharper attention initialization
    ├── exp_0005 failed score=99.0 epoch=1 QK_GAIN_INIT=5.25
    ├── exp_0006 failed score=99.0 epoch=1 PARALLEL_RESIDUAL_START=5
    ├── exp_0007 failed score=99.0 epoch=1 MATRIX_LR=0.025 SCALAR_LR=0.023
    ├── exp_0008 failed score=99.0 epoch=1 Fix evo_benchmark.py infrastructure: remove unsupported extra_files kwarg from run_smoke/run_qualify calls
    ├── exp_0009 active epoch=1 QK_GAIN_INIT=5.25 + fixed benchmark (correct dataclass attrs, copy target to repo root)
    ├── exp_0010 failed score=2.5374 epoch=1 QK_GAIN_INIT=5.25
    ├── exp_0011 active epoch=1 PARALLEL_RESIDUAL_START=5
    ├── exp_0012 active epoch=1 MATRIX_LR=0.025 SCALAR_LR=0.023
    ├── exp_0013 discarded score=99.0 epoch=1 QK_GAIN_INIT=5.25: competition record default (fixed benchmark: stdout redirect, GCS tokenizer fallback)
    ├── exp_0014 committed score=2.538 epoch=1 QK_GAIN_INIT=5.25: retry after GCP capacity recovery
    │   ├── exp_0015 discarded score=99.0 epoch=1 ENABLE_LOOPING_AT=0.25: earlier depth recurrence activation (from 0.35), activates looping at 25% vs 35% of training
    │   └── exp_0016 discarded score=2.5444 epoch=1 ENABLE_LOOPING_AT=0.25: earlier depth recurrence (retry, SPOT preemption on first attempt)
    ├── exp_0017 committed score=2.1318 epoch=1 GPTQ_ADAPTIVE_CLIP=1: qualify step-1000 train_loss=2.1318 on 8xH100
    └── exp_0018 committed score=2.129 epoch=1 WARMDOWN_SHAPE=sqrt: qualify step-1000 train_loss=2.1290 (beats adaptive GPTQ 2.1318)
        ├── exp_0019 discarded score=2.1291 epoch=1 STACKED sqrt+GPTQ+int7: qualify step-1000=2.1291 (same as sqrt alone, GPTQ/int7 only affect quantization)
        ├── exp_0020 committed score=1.9433 epoch=1 FULL RUN stacked (sqrt+GPTQ+int7+QK5.25): train_loss=1.9433@step4000 (vs baseline 1.9957). Preempted before GPTQ eval. Config: WARMDOWN_SHAPE=sqrt GPTQ_ADAPTIVE_CLIP=1 EMBED_BITS=7 QK_GAIN_INIT=5.25
        └── exp_0021 committed score=1.1236 epoch=1 FULL RUN v2 (sqrt+GPTQ+int7+QK5.25): pre-quant val_bpb=1.1236, post-EMA=1.1236. Preempted during GPTQ. Estimated final ~1.127-1.129. SP1024 vocab.
```

## Best Path
root -> exp_0000 (2.5399) -> exp_0018 (2.129) -> exp_0021 (1.1236)

## Frontier
- `exp_0014` score=`2.538` epoch=`1` QK_GAIN_INIT=5.25: retry after GCP capacity recovery
- `exp_0017` score=`2.1318` epoch=`1` GPTQ_ADAPTIVE_CLIP=1: qualify step-1000 train_loss=2.1318 on 8xH100
- `exp_0020` score=`1.9433` epoch=`1` FULL RUN stacked (sqrt+GPTQ+int7+QK5.25): train_loss=1.9433@step4000 (vs baseline 1.9957). Preempted before GPTQ eval. Config: WARMDOWN_SHAPE=sqrt GPTQ_ADAPTIVE_CLIP=1 EMBED_BITS=7 QK_GAIN_INIT=5.25
- `exp_0021` score=`1.1236` epoch=`1` FULL RUN v2 (sqrt+GPTQ+int7+QK5.25): pre-quant val_bpb=1.1236, post-EMA=1.1236. Preempted during GPTQ. Estimated final ~1.127-1.129. SP1024 vocab.

## Gates
- `syntax_check` (root): `python3 -c "import ast; ast.parse(open('{target}').read()); print('syntax OK', file=__import__('sys').stderr)"`

## Recent Experiments
- `exp_0021` `committed` score=`1.1236` FULL RUN v2 (sqrt+GPTQ+int7+QK5.25): pre-quant val_bpb=1.1236, post-EMA=1.1236. Preempted during GPTQ. Estimated final ~1.127-1.129. SP1024 vocab.
- `exp_0020` `committed` score=`1.9433` FULL RUN stacked (sqrt+GPTQ+int7+QK5.25): train_loss=1.9433@step4000 (vs baseline 1.9957). Preempted before GPTQ eval. Config: WARMDOWN_SHAPE=sqrt GPTQ_ADAPTIVE_CLIP=1 EMBED_BITS=7 QK_GAIN_INIT=5.25
- `exp_0019` `discarded` score=`2.1291` STACKED sqrt+GPTQ+int7: qualify step-1000=2.1291 (same as sqrt alone, GPTQ/int7 only affect quantization)
- `exp_0018` `committed` score=`2.129` WARMDOWN_SHAPE=sqrt: qualify step-1000 train_loss=2.1290 (beats adaptive GPTQ 2.1318)
- `exp_0016` `discarded` score=`2.5444` ENABLE_LOOPING_AT=0.25: earlier depth recurrence (retry, SPOT preemption on first attempt)
- `exp_0017` `committed` score=`2.1318` GPTQ_ADAPTIVE_CLIP=1: qualify step-1000 train_loss=2.1318 on 8xH100
- `exp_0015` `discarded` score=`99.0` ENABLE_LOOPING_AT=0.25: earlier depth recurrence activation (from 0.35), activates looping at 25% vs 35% of training
- `exp_0014` `committed` score=`2.538` QK_GAIN_INIT=5.25: retry after GCP capacity recovery

## Recent Diffs
- `exp_0014`: winning_base_decoded.py (+1/-1)

## Annotations
- task `global` / `exp_0016`: ENABLE_LOOPING_AT=0.25 regresses: 2.5444 vs parent exp_0014's 2.538. Earlier looping (25% vs 35%) hurts at smoke-test scale. The depth recurrence benefits from starting later in training when representations are more mature. ENABLE_LOOPI...

## What Not To Try
- PARALLEL_RESIDUAL_START=5: earlier parallel residuals for more attn/MLP specialization
- MATRIX_LR=0.025, SCALAR_LR=0.023: higher LR to compensate for SDPA vs flash-attn-3
- QK_GAIN_INIT=5.25: match latest competition record attention sharpness
- QK_GAIN_INIT=5.25: competition record uses 5.25 vs our 5.0, sharper attention initialization
- QK_GAIN_INIT=5.25: competition record default (fixed benchmark: stdout redirect, GCS tokenizer fallback)
- ENABLE_LOOPING_AT=0.25: earlier depth recurrence activation (from 0.35), activates looping at 25% vs 35% of training
- ENABLE_LOOPING_AT=0.25: earlier depth recurrence (retry, SPOT preemption on first attempt)
- STACKED sqrt+GPTQ+int7: qualify step-1000=2.1291 (same as sqrt alone, GPTQ/int7 only affect quantization)

## Infrastructure Log
- No infrastructure events yet.

## Notes
# Notes
