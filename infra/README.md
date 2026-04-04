# Parameter Golf — Vertex AI Experiment Infrastructure

Serverless 8x H100 experiment runner on Google Cloud Vertex AI. Launch parallel training runs, pay only for compute used (~$2-3/run with Spot VMs).

## Prerequisites

- GCP project with billing enabled
- `a3-highgpu-8g` quota in Vertex AI Custom Training ([request here](https://console.cloud.google.com/iam-admin/quotas))
- `gcloud` CLI installed
- Docker installed locally

## Setup (one-time)

### 1. Authenticate

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud auth application-default login   # for the Python SDK
```

### 2. Install local dependencies

```bash
pip install -r infra/requirements.txt
```

### 3. Upload training data to GCS

```bash
bash infra/setup_gcs.sh parameter-golf-data us-central1
```

### 4. Build and push the training container

```bash
bash infra/build_push.sh YOUR_PROJECT_ID us-central1 latest
```

### 5. Configure experiments.yaml

Edit `infra/experiments.yaml` and set the `global` section:

```yaml
global:
  project: "YOUR_PROJECT_ID"
  location: "us-central1"
  data_bucket: "parameter-golf-data"
  output_bucket: "parameter-golf-experiments"
  container_uri: "us-central1-docker.pkg.dev/YOUR_PROJECT_ID/parameter-golf/training:latest"
  use_spot: true
```

## CLI Usage

```bash
# Launch a single experiment
python infra/launch.py launch --name "lr-sweep-03" --script experiment1.py \
  --env MATRIX_LR=0.03 NUM_LAYERS=11

# Launch all experiments from YAML
python infra/launch.py batch --config infra/experiments.yaml

# Launch only matching experiments
python infra/launch.py batch --config infra/experiments.yaml --filter "lr-sweep"

# Check job status
python infra/launch.py status
python infra/launch.py status --name "lr-sweep"

# Download results
python infra/launch.py download --name "lr-sweep-03"
python infra/launch.py download --name "lr-sweep-03" --output-dir ./my-results

# Wait for a job to finish
python infra/launch.py wait --job "pgolf-lr-sweep-03-20260404-153022"

# Cancel a running job
python infra/launch.py cancel --job "pgolf-lr-sweep-03-20260404-153022"

# Disable Spot VMs for a single run
python infra/launch.py launch --name "important-run" --no-spot --env ITERATIONS=20000
```

## Python API

```python
from infra.launch import VertexLauncher

# Initialize from YAML config
launcher = VertexLauncher.from_config("infra/experiments.yaml")

# Or with explicit params
launcher = VertexLauncher(
    project="my-gcp-project",
    location="us-central1",
    data_bucket="parameter-golf-data",
    output_bucket="parameter-golf-experiments",
    container_uri="us-central1-docker.pkg.dev/my-project/parameter-golf/training:latest",
)
```

### `launcher.launch(name, script, env, spot) -> dict`

Submit a single experiment.

```python
job = launcher.launch(
    name="lr-sweep-03",                              # required
    script="experiment1.py",                          # default: "experiment1.py"
    env={"MATRIX_LR": "0.03", "NUM_LAYERS": "11"},   # hyperparameter overrides
    spot=True,                                        # default: True
)
# Returns:
# {
#     "job_name": "pgolf-lr-sweep-03-20260404-153022",
#     "resource_name": "projects/123/locations/us-central1/customJobs/456",
#     "output_uri": "gs://parameter-golf-experiments/experiments/lr-sweep-03/20260404-153022/",
#     "state": "JOB_STATE_PENDING",
# }
```

### `launcher.batch(config_path, filter) -> list[dict]`

Launch all (or filtered) experiments from a YAML config file.

```python
results = launcher.batch(
    config_path="infra/experiments.yaml",   # required
    filter="lr-sweep",                      # optional: only matching names
)
# Returns: list of job info dicts (same shape as launch())
```

### `launcher.status(name, limit) -> list[dict]`

List recent jobs and their states.

```python
jobs = launcher.status(
    name="lr-sweep",   # optional: filter by name prefix
    limit=20,          # default: 20
)
# Returns:
# [
#     {"job_name": "pgolf-lr-sweep-03-...", "state": "JOB_STATE_SUCCEEDED",
#      "create_time": "2026-04-04T15:30:22Z", "resource_name": "...", "output_uri": "gs://..."},
# ]
```

### `launcher.download(name, output_dir) -> Path`

Download experiment outputs from GCS to a local directory.

```python
local_path = launcher.download(
    name="lr-sweep-03",        # required
    output_dir="./results",    # default: "./results"
)
# Returns: Path("./results/lr-sweep-03")
# Contains: final_model.pt, final_model.int6.ptz, logs/, training_output.log
```

### `launcher.wait(job_name, poll_interval, timeout) -> str`

Block until a job reaches a terminal state.

```python
state = launcher.wait(
    job_name="pgolf-lr-sweep-03-20260404-153022",   # required
    poll_interval=30,                                # default: 30 seconds
    timeout=1200,                                    # default: 1200 seconds
)
# Returns: "JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", or "JOB_STATE_CANCELLED"
# Raises: TimeoutError if timeout exceeded
```

### `launcher.cancel(job_name) -> None`

Cancel a running job.

```python
launcher.cancel(job_name="pgolf-lr-sweep-03-20260404-153022")
```

## Experiment Config Format

`experiments.yaml` has two sections:

```yaml
global:
  project: "your-gcp-project-id"       # GCP project ID
  location: "us-central1"              # Region (must have a3-highgpu-8g quota)
  data_bucket: "parameter-golf-data"   # GCS bucket with training data
  output_bucket: "parameter-golf-experiments"  # GCS bucket for outputs
  container_uri: "..."                 # Full Artifact Registry image URI
  use_spot: true                       # Default Spot VM setting

experiments:
  - name: "my-experiment"              # Unique name (used in job name + GCS path)
    script: "experiment1.py"           # Training script (in container at /app/)
    env:                               # Hyperparameter overrides (all optional)
      MATRIX_LR: "0.03"
      NUM_LAYERS: "11"
      XSA_LAST_N: "4"
```

Any env var from `experiment1.py`'s `Hyperparameters` class can be set in `env`. See the full list in `experiment1.py:41-121` and `train_gpt.py:39-88`.

## Architecture

```
Local machine                         Google Cloud
--------------                        -------------------------
launch.py ----Vertex AI SDK----->     Vertex AI Custom Job
                                        |
                                        v
                                      a3-highgpu-8g (8x H100)
                                        |
                                      entrypoint.sh
                                        1. gsutil rsync (GCS -> local data)
                                        2. torchrun --nproc_per_node=8
                                        3. gsutil cp (results -> GCS)
                                        |
experiments.yaml                      GCS buckets
  global config  --------->            parameter-golf-data/
  experiment list                        datasets/fineweb10B_sp1024/
                                         tokenizers/
                                       parameter-golf-experiments/
                                         experiments/{name}/{timestamp}/
                                           final_model.pt
                                           final_model.int6.ptz
                                           logs/
                                           training_output.log
```

## Rebuilding the Container

When training scripts (`train_gpt.py`, `experiment1.py`, etc.) change:

```bash
bash infra/build_push.sh YOUR_PROJECT_ID us-central1 v2
```

Then update `container_uri` in `experiments.yaml` (or keep using `latest`).

Docker layer caching means only the `COPY` layer rebuilds — takes seconds, not minutes.

## Cost

| Instance | On-demand (10 min) | Spot (10 min) |
|---|---|---|
| a3-highgpu-8g (8x H100) | ~$6-8 | ~$2-3 |

10 parallel experiments with Spot: ~$20-30 total.
