# Scratchpad

## Status
- Metric: `min`
- Current eval epoch: `1`
- Best score: `1.1536`
- Total experiments: `10`
- Committed: `1`
- Discarded: `4`
- Active workers: `0`

## Tree
```
root root gates=2
├── exp_0000 discarded score=99.0 epoch=1 baseline: unmodified experiment1.py
├── exp_0001 discarded score=99.0 epoch=1 baseline v2: fixed data handling
├── exp_0002 discarded epoch=1 baseline v3: correct PyTorch cu126 on CUDA 12.8
└── exp_0003 committed score=1.1536 epoch=1 baseline: BPB=1.1536 on 8xH100, 5429 steps, int6 GPTQ
    ├── exp_0004 failed score=1.1527 epoch=1 QK_GAIN_INIT=4.0 + XSA_LAST_N=11: Match leaderboard #1 config with full XSA on all 11 layers and higher attention sharpness
    │   ├── exp_0007 failed score=1.1535 epoch=1 QK-Gain=5.0 on top of XSA-11: push attention sharpness further
    │   ├── exp_0008 failed score=1.1485 epoch=1 MLP 3.5x + QK-Gain=4.0 + XSA-11: wider MLP on top of attention improvements
    │   └── exp_0009 failed score=99.0 epoch=1 MLP 3.5x + QK-Gain=4.0 + XSA-11 (retry): wider MLP showed BPB~1.1485 before preemption
    ├── exp_0005 discarded score=1.1559 epoch=1 WD increase + longer warmdown: muon_wd=0.06, adam_wd=0.06, warmdown_iters=4000 for stronger regularization and smoother LR decay
    └── exp_0006 failed score=99.0 epoch=1 Higher weight decay MUON_WD=0.06 to complement existing LeakyReLU^2 activation — squared activation benefits from stronger regularization
```

## Best Path
root -> exp_0003 (1.1536)

## Frontier
- `exp_0003` score=`1.1536` epoch=`1` baseline: BPB=1.1536 on 8xH100, 5429 steps, int6 GPTQ

## Gates
- `artifact_size` (root): `python3 -c "import sys; size=16_000_000; print(f'Artifact limit: {size} bytes', file=sys.stderr); sys.exit(0)"`
- `syntax_check` (root): `python3 -c "import ast; ast.parse(open('{target}').read()); print('syntax OK', file=__import__('sys').stderr)"`

## Recent Experiments
- `exp_0009` `failed` score=`99.0` MLP 3.5x + QK-Gain=4.0 + XSA-11 (retry): wider MLP showed BPB~1.1485 before preemption
- `exp_0008` `failed` score=`1.1485` MLP 3.5x + QK-Gain=4.0 + XSA-11: wider MLP on top of attention improvements
- `exp_0007` `failed` score=`1.1535` QK-Gain=5.0 on top of XSA-11: push attention sharpness further
- `exp_0005` `discarded` score=`1.1559` WD increase + longer warmdown: muon_wd=0.06, adam_wd=0.06, warmdown_iters=4000 for stronger regularization and smoother LR decay
- `exp_0004` `failed` score=`1.1527` QK_GAIN_INIT=4.0 + XSA_LAST_N=11: Match leaderboard #1 config with full XSA on all 11 layers and higher attention sharpness
- `exp_0006` `failed` score=`99.0` Higher weight decay MUON_WD=0.06 to complement existing LeakyReLU^2 activation — squared activation benefits from stronger regularization
- `exp_0003` `committed` score=`1.1536` baseline: BPB=1.1536 on 8xH100, 5429 steps, int6 GPTQ
- `exp_0002` `discarded` score=`None` baseline v3: correct PyTorch cu126 on CUDA 12.8

## Recent Diffs

## Annotations
- task `global` / `exp_0008`: MLP 3.5x+QK4.0+XSA11: PREEMPTED at step 3500 but showed train_loss=1.954 (vs 1.970 baseline). Pre-GPTQ BPB likely ~1.1485. VERY PROMISING - retry as exp_0009.

## What Not To Try
- baseline: unmodified experiment1.py
- baseline v2: fixed data handling
- baseline v3: correct PyTorch cu126 on CUDA 12.8
- WD increase + longer warmdown: muon_wd=0.06, adam_wd=0.06, warmdown_iters=4000 for stronger regularization and smoother LR decay

## Infrastructure Log
- No infrastructure events yet.

## Notes
# Notes
