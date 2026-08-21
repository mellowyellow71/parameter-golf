#!/bin/bash
# Fast parallel qualify loop for Parameter Golf.
# Provisions N instances ONCE, then runs experiments back-to-back on each.
# Only re-provisions if preempted.
#
# Usage:
#   nohup bash infra/overnight_loop.sh > infra/overnight.log 2>&1 &
#   tail -f infra/overnight.log
#   kill $(cat infra/overnight.pid)

set -uo pipefail
cd /home/ray/parameter-golf

echo $$ > infra/overnight.pid

NUM_WORKERS=3
MAX_EXPERIMENTS=50

echo "[$(date)] Fast parallel loop started (PID $$, $NUM_WORKERS workers, $MAX_EXPERIMENTS max)"

# Worker function: provision once, run experiments until preempted or done
worker() {
    local wid=$1
    local prefix="[W$wid]"
    local instance_name=""
    local instance_zone=""
    local instance_ip=""
    local experiments_run=0

    echo "$prefix Worker starting"

    while true; do
        # Check if there's anything left to test
        local next=$(python3 infra/auto_qualify.py --next 2>&1 | head -1)
        if echo "$next" | grep -q "All strategies tested"; then
            echo "$prefix All strategies tested. Worker done."
            break
        fi

        # Provision if we don't have an instance
        if [ -z "$instance_ip" ]; then
            echo "$prefix Provisioning SPOT 8xH100..."
            local provision_out=$(python3 -c "
import sys; sys.path.insert(0, 'infra')
from gce_provision import load_config, find_and_create, wait_for_ssh
from gce_run_experiment import sync_code, ensure_data
config = load_config('infra/gce_config.yaml')
instance = find_and_create('worker-$wid', config)
if instance is None:
    print('FAIL:no_capacity')
else:
    if not wait_for_ssh(instance, config):
        print(f'FAIL:ssh_timeout:{instance.name}:{instance.zone}')
    else:
        sync_code(instance, config)
        ensure_data(instance, config, vocab_size=1024)
        print(f'OK:{instance.name}:{instance.zone}:{instance.external_ip}')
" 2>&1 | tail -1)

            if echo "$provision_out" | grep -q "^OK:"; then
                instance_name=$(echo "$provision_out" | cut -d: -f2)
                instance_zone=$(echo "$provision_out" | cut -d: -f3)
                instance_ip=$(echo "$provision_out" | cut -d: -f4)
                echo "$prefix Instance ready: $instance_name ($instance_zone) @ $instance_ip"
            else
                echo "$prefix Provisioning failed: $provision_out"
                echo "$prefix Sleeping 120s before retry..."
                sleep 120
                continue
            fi
        fi

        # Claim next strategy
        local strategy_info=$(python3 -c "
import sys, json, fcntl; sys.path.insert(0, 'infra')
from auto_qualify import claim_next, get_queue, load_results
s = claim_next()
if s is None:
    print('DONE')
else:
    env = dict(s.env)
    env.setdefault('VOCAB_SIZE', '1024')
    env.setdefault('COMPRESSOR', 'lzma')
    env_str = ' '.join(f'{k}={v}' for k,v in env.items())
    print(f'{s.name}|{s.script}|{env_str}|{s.description}')
" 2>&1 | tail -1)

        if [ "$strategy_info" = "DONE" ]; then
            echo "$prefix All strategies claimed. Worker done."
            break
        fi

        local strat_name=$(echo "$strategy_info" | cut -d'|' -f1)
        local strat_script=$(echo "$strategy_info" | cut -d'|' -f2)
        local strat_env=$(echo "$strategy_info" | cut -d'|' -f3)
        local strat_desc=$(echo "$strategy_info" | cut -d'|' -f4)

        echo "$prefix Running: $strat_name ($strat_desc)"

        # Run experiment on existing instance via SSH
        local train_cmd="cd ~/parameter-golf && export MAX_WALLCLOCK_SECONDS=240 VAL_LOSS_EVERY=1000 TRAIN_LOG_EVERY=100 $strat_env && torchrun --standalone --nproc_per_node=8 $strat_script 2>&1 | tee /tmp/train_${strat_name}.log; echo EXIT_CODE=\$?"

        # Kill any existing training, start new one
        gcloud compute ssh ray@$instance_name --zone=$instance_zone --project=bryan-usage-0 \
            --ssh-flag="-o ConnectTimeout=15" \
            --command="tmux kill-session -t training 2>/dev/null; tmux new-session -d -s training '$train_cmd'" 2>&1

        if [ $? -ne 0 ]; then
            echo "$prefix SSH failed — instance likely preempted. Re-provisioning..."
            instance_ip=""
            instance_name=""
            instance_zone=""
            # Mark strategy as untested so it gets picked up again
            python3 -c "
import json, fcntl
from pathlib import Path
with open('infra/qualify_results.lock', 'w') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    data = json.loads(Path('infra/qualify_results.json').read_text())
    if '$strat_name' in data['results'] and data['results']['$strat_name'].get('status') == 'running':
        del data['results']['$strat_name']
    Path('infra/qualify_results.json').write_text(json.dumps(data, indent=2))
" 2>&1
            continue
        fi

        # Poll for completion (check every 30s for up to 6 min)
        local elapsed=0
        local max_wait=360
        local log_text=""
        while [ $elapsed -lt $max_wait ]; do
            sleep 30
            elapsed=$((elapsed + 30))

            log_text=$(gcloud compute ssh ray@$instance_name --zone=$instance_zone --project=bryan-usage-0 \
                --ssh-flag="-o ConnectTimeout=10" \
                --command="cat /tmp/train_${strat_name}.log 2>/dev/null" 2>&1)

            if [ $? -ne 0 ]; then
                echo "$prefix SSH lost at ${elapsed}s — preempted?"
                instance_ip=""
                instance_name=""
                instance_zone=""
                break
            fi

            # Check if training finished
            if echo "$log_text" | grep -q "EXIT_CODE=\|stopping_early\|ema:applying"; then
                echo "$prefix Training complete at ${elapsed}s"
                break
            fi

            # Show progress
            local last_step=$(echo "$log_text" | grep -oP '\d+/20000 train_loss' | tail -1 | cut -d/ -f1)
            [ -n "$last_step" ] && echo "$prefix   step $last_step at ${elapsed}s"
        done

        # Parse results
        if [ -n "$instance_ip" ] && [ -n "$log_text" ]; then
            python3 -c "
import re, json, fcntl, subprocess
from pathlib import Path
from datetime import datetime

log = '''$log_text'''

# Parse val metrics (step, val_loss, val_bpb)
val_pattern = r'(\d+)/\d+ val_loss: ([\d.]+) val_bpb: ([\d.]+)'
vals = [(int(m[0]), float(m[1]), float(m[2])) for m in re.findall(val_pattern, log)]
useful = [v for v in vals if v[0] >= 500]

# Parse last train step
train_pattern = r'(\d+)/\d+ train_loss: ([\d.]+)'
trains = re.findall(train_pattern, log)
last_step = int(trains[-1][0]) if trains else 0

result = {
    'name': '$strat_name',
    'status': 'pass' if useful else 'fail',
    'step_1000_bpb': useful[0][2] if useful else None,
    'val_loss_1000': useful[0][1] if useful else None,
    'last_step': last_step,
    'error': None if useful else 'no val metrics past step 500',
    'env': dict(x.split('=',1) for x in '$strat_env'.split() if '=' in x),
    'description': '$strat_desc',
    'timestamp': datetime.now().isoformat(),
}

with open('infra/qualify_results.lock', 'w') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    data = json.loads(Path('infra/qualify_results.json').read_text())
    data['results']['$strat_name'] = result
    data['runs'] = data.get('runs', 0) + 1
    if result.get('step_1000_bpb') and result['step_1000_bpb'] < data.get('best_bpb', 99.0):
        data['best'] = '$strat_name'
        data['best_bpb'] = result['step_1000_bpb']
    Path('infra/qualify_results.json').write_text(json.dumps(data, indent=2))

bpb = result.get('step_1000_bpb')
if bpb:
    print(f'RESULT: $strat_name = {bpb:.4f} BPB')
    # Log to evo
    r = subprocess.run(['evo', 'new', '--parent', 'exp_0000', '-m',
        f'$strat_name: qualify val_bpb={bpb:.4f}. $strat_desc'],
        capture_output=True, text=True, timeout=30)
    exp_id = None
    if r.returncode == 0:
        m = re.search(r'exp_\d+', r.stdout)
        exp_id = m.group(0) if m else None
    if exp_id:
        subprocess.run(['evo', 'done', exp_id, '--score', str(bpb), '--no-compare'],
            capture_output=True, text=True, timeout=30)
        # Fix status to committed
        gp = Path('.evo/run_0003/graph.json')
        if gp.exists():
            g = json.loads(gp.read_text())
            if exp_id in g.get('nodes', {}):
                g['nodes'][exp_id]['status'] = 'committed'
                gp.write_text(json.dumps(g, indent=2))
        print(f'EVO: {exp_id} score={bpb:.4f}')
else:
    print(f'FAIL: $strat_name — no val BPB')
" 2>&1 | sed "s/^/$prefix /"
        fi

        experiments_run=$((experiments_run + 1))
        echo "$prefix Completed $experiments_run experiments on this instance"
    done

    # Cleanup instance
    if [ -n "$instance_name" ]; then
        echo "$prefix Cleaning up $instance_name"
        gcloud compute instances delete "$instance_name" --zone="$instance_zone" --project=bryan-usage-0 --quiet 2>&1 &
    fi
    echo "$prefix Worker done ($experiments_run experiments)"
}

# Launch workers
for i in $(seq 1 $NUM_WORKERS); do
    worker $i &
    sleep 5  # stagger provisioning
done

# Wait for all workers
wait
echo "[$(date)] All workers done"
python3 infra/auto_qualify.py --status
