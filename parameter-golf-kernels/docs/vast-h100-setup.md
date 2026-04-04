# Cheap H100 on Vast.ai

## 1. Install CLI & Login

```bash
pip install vastai
vastai set api-key YOUR_API_KEY
```

Get your API key at https://cloud.vast.ai/account/

## 2. Find Cheap Interruptible H100

```bash
vastai search offers 'gpu_name=H100_NVL num_gpus=1 disk_space>=50' -t 'interruptible' -o 'dph' --limit 10
```

`-t interruptible` = spot pricing (~$0.20/hr vs $2/hr on-demand).
`-o dph` = sort by price.
`disk_space>=50` = avoid the 16GB default traps.

## 3. Create Instance

```bash
vastai create instance <OFFER_ID> --image pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel --disk 100 --ssh
```

**Always set `--disk 100`** (GB). Default is ~10GB which fills up instantly with PyTorch + FA3.

## 4. Get SSH Command

```bash
vastai show instances
```

Connect: `ssh -p <PORT> root@<HOST>`

## 5. Setup Environment

```bash
# On the H100:
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="/root/.local/bin:$PATH"
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install triton numpy sentencepiece zstandard einops ninja
uv pip install pip
pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch2100
```

## 6. Verify

```bash
python3 -c "
import torch
print(f'torch {torch.__version__}, GPU: {torch.cuda.get_device_name(0)}')
from flash_attn_interface import flash_attn_func
print('FA3: available')
"
```

## 7. Manage Instances

```bash
vastai show instances          # list running
vastai destroy instance <ID>   # kill (stops billing)
```

## Notes

- **Interruptible instances lose disk on preemption.** Push to git before walking away.
- **Container disk is fixed at creation.** Can't resize later.
- Typical cost: $0.15-0.25/hr for 1×H100 NVL interruptible.
