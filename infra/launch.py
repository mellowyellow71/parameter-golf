#!/usr/bin/env python3
"""
Vertex AI Custom Job launcher for Parameter Golf experiments.

Python API:
    from infra.launch import VertexLauncher
    launcher = VertexLauncher.from_config("infra/experiments.yaml")
    job = launcher.launch(name="my-exp", script="experiment1.py", env={"MATRIX_LR": "0.03"})

CLI:
    python infra/launch.py launch  --name EXP --script SCRIPT [--env K=V ...]
    python infra/launch.py batch   --config experiments.yaml [--filter PATTERN]
    python infra/launch.py status  [--name PREFIX]
    python infra/launch.py download --name EXP [--output-dir ./results]
    python infra/launch.py wait    --job JOB_NAME [--timeout 1200]
    python infra/launch.py cancel  --job JOB_NAME
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from google.cloud import aiplatform


class VertexLauncher:
    """Manages Vertex AI Custom Jobs for Parameter Golf experiments."""

    def __init__(
        self,
        project: str,
        location: str = "us-central1",
        data_bucket: str = "parameter-golf-data",
        output_bucket: str = "parameter-golf-experiments",
        container_uri: str = "",
        use_spot: bool = True,
    ):
        self.project = project
        self.location = location
        self.data_bucket = data_bucket
        self.output_bucket = output_bucket
        self.container_uri = container_uri
        self.use_spot = use_spot
        aiplatform.init(project=project, location=location)

    @classmethod
    def from_config(cls, config_path: str) -> "VertexLauncher":
        """Create a VertexLauncher from a YAML config file's global section."""
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        g = cfg.get("global", {})
        return cls(
            project=g["project"],
            location=g.get("location", "us-central1"),
            data_bucket=g.get("data_bucket", "parameter-golf-data"),
            output_bucket=g.get("output_bucket", "parameter-golf-experiments"),
            container_uri=g.get("container_uri", ""),
            use_spot=g.get("use_spot", True),
        )

    def launch(
        self,
        name: str,
        script: str = "experiment1.py",
        env: dict[str, str] | None = None,
        spot: bool | None = None,
    ) -> dict[str, Any]:
        """Submit a single Vertex AI Custom Job.

        Args:
            name: Experiment name (used in job display name and GCS output path).
            script: Training script filename (must exist in the container at /app/).
            env: Hyperparameter overrides as env var key-value pairs.
            spot: Use Spot VMs. Defaults to self.use_spot.

        Returns:
            Dict with job_name, resource_name, output_uri, and state.
        """
        if env is None:
            env = {}
        if spot is None:
            spot = self.use_spot

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        job_name = f"pgolf-{name}-{timestamp}"
        output_uri = f"gs://{self.output_bucket}/experiments/{name}/{timestamp}/"

        # Build env var list: system vars + user hyperparameters
        all_env = {
            "GCS_DATA_BUCKET": f"gs://{self.data_bucket}",
            "GCS_OUTPUT_BUCKET": f"gs://{self.output_bucket}",
            "EXPERIMENT_NAME": f"{name}/{timestamp}",
            "TRAINING_SCRIPT": script,
            "RUN_ID": job_name,
        }
        all_env.update(env)
        env_list = [{"name": k, "value": str(v)} for k, v in all_env.items()]

        worker_pool_spec = {
            "machine_spec": {
                "machine_type": "a3-highgpu-8g",
                "accelerator_type": "NVIDIA_H100_80GB",
                "accelerator_count": 8,
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": self.container_uri,
                "env": env_list,
            },
        }

        custom_job = aiplatform.CustomJob(
            display_name=job_name,
            worker_pool_specs=[worker_pool_spec],
            staging_bucket=f"gs://{self.output_bucket}/staging",
        )

        # Submit the job (non-blocking)
        custom_job.run(sync=False)

        # Wait for the job to be created on the server (typically 2-5s)
        resource_name = "pending"
        for _ in range(60):
            try:
                resource_name = custom_job.resource_name
                break
            except RuntimeError:
                time.sleep(1)

        state = "PENDING"
        try:
            state = custom_job.state.name if custom_job.state else "PENDING"
        except RuntimeError:
            pass

        return {
            "job_name": job_name,
            "resource_name": resource_name,
            "output_uri": output_uri,
            "state": state,
        }

    def batch(
        self,
        config_path: str,
        filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Launch all experiments from a YAML config file.

        Args:
            config_path: Path to experiments.yaml.
            filter: Optional substring to filter experiment names.

        Returns:
            List of job info dicts (same shape as launch() return value).
        """
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        experiments = cfg.get("experiments", [])
        if filter:
            experiments = [e for e in experiments if filter in e["name"]]

        if not experiments:
            print("No experiments matched.")
            return []

        print(f"Launching {len(experiments)} experiment(s)...")
        results = []

        with ThreadPoolExecutor(max_workers=min(10, len(experiments))) as pool:
            futures = {}
            for exp in experiments:
                future = pool.submit(
                    self.launch,
                    name=exp["name"],
                    script=exp.get("script", "experiment1.py"),
                    env=exp.get("env", {}),
                )
                futures[future] = exp["name"]

            for future in as_completed(futures):
                exp_name = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    print(f"  OK  {exp_name:40s} -> {result['job_name']}")
                except Exception as e:
                    print(f"  ERR {exp_name:40s} -> {e}")

        return results

    def status(
        self,
        name: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """List recent Parameter Golf Custom Jobs and their status.

        Args:
            name: Optional prefix filter (matches against display_name).
            limit: Max number of jobs to return.

        Returns:
            List of dicts with job_name, state, create_time, output_uri.
        """
        filter_str = 'display_name:"pgolf-"'
        if name:
            filter_str = f'display_name:"pgolf-{name}"'

        jobs = aiplatform.CustomJob.list(
            filter=filter_str,
            order_by="create_time desc",
        )

        results = []
        for job in jobs[:limit]:
            # Extract experiment name/timestamp from display_name
            # Format: pgolf-{name}-{YYYYMMDD-HHMMSS}
            parts = job.display_name.split("-", 1)
            exp_suffix = parts[1] if len(parts) > 1 else job.display_name

            results.append({
                "job_name": job.display_name,
                "state": job.state.name if job.state else "UNKNOWN",
                "create_time": str(job.create_time),
                "resource_name": job.resource_name,
                "output_uri": f"gs://{self.output_bucket}/experiments/{exp_suffix}/",
            })

        return results

    def download(
        self,
        name: str,
        output_dir: str = "./results",
    ) -> Path:
        """Download experiment outputs from GCS.

        Args:
            name: Experiment name (matches the GCS prefix under experiments/).
            output_dir: Local directory to download into.

        Returns:
            Path to the local directory containing the downloaded results.
        """
        target = Path(output_dir) / name
        target.mkdir(parents=True, exist_ok=True)
        gcs_prefix = f"gs://{self.output_bucket}/experiments/{name}/"

        print(f"Downloading {gcs_prefix} -> {target}/")
        subprocess.run(
            ["gsutil", "-m", "rsync", "-r", gcs_prefix, str(target)],
            check=True,
        )
        print(f"Done. Results in {target}/")
        return target

    def wait(
        self,
        job_name: str,
        poll_interval: int = 30,
        timeout: int = 1200,
    ) -> str:
        """Block until a job completes or times out.

        Args:
            job_name: The display_name or resource_name of the job.
            poll_interval: Seconds between status checks.
            timeout: Max seconds to wait before raising TimeoutError.

        Returns:
            Final job state string (e.g. "JOB_STATE_SUCCEEDED").
        """
        # Find the job by display_name
        jobs = aiplatform.CustomJob.list(
            filter=f'display_name="{job_name}"',
            order_by="create_time desc",
        )
        if not jobs:
            raise ValueError(f"No job found with display_name={job_name}")
        job = jobs[0]

        terminal_states = {
            "JOB_STATE_SUCCEEDED",
            "JOB_STATE_FAILED",
            "JOB_STATE_CANCELLED",
        }

        start = time.time()
        while True:
            state = job.state.name if job.state else "UNKNOWN"
            elapsed = int(time.time() - start)
            print(f"  [{elapsed:4d}s] {job_name}: {state}")

            if state in terminal_states:
                return state

            if time.time() - start > timeout:
                raise TimeoutError(
                    f"Job {job_name} still in state {state} after {timeout}s"
                )

            time.sleep(poll_interval)
            job = aiplatform.CustomJob.list(
                filter=f'display_name="{job_name}"',
                order_by="create_time desc",
            )[0]

    def cancel(self, job_name: str) -> None:
        """Cancel a running job.

        Args:
            job_name: The display_name of the job to cancel.
        """
        jobs = aiplatform.CustomJob.list(
            filter=f'display_name="{job_name}"',
            order_by="create_time desc",
        )
        if not jobs:
            raise ValueError(f"No job found with display_name={job_name}")

        jobs[0].cancel()
        print(f"Cancelled: {job_name}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_env_args(env_args: list[str]) -> dict[str, str]:
    """Parse KEY=VALUE pairs from CLI --env arguments."""
    result = {}
    for item in env_args:
        if "=" not in item:
            raise ValueError(f"Invalid env format: {item!r} (expected KEY=VALUE)")
        key, value = item.split("=", 1)
        result[key] = value
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Parameter Golf — Vertex AI Experiment Launcher",
    )
    parser.add_argument(
        "--config", default="infra/experiments.yaml",
        help="Path to experiments.yaml (used for global settings)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # launch
    p_launch = sub.add_parser("launch", help="Submit a single experiment")
    p_launch.add_argument("--name", required=True, help="Experiment name")
    p_launch.add_argument("--script", default="experiment1.py", help="Training script")
    p_launch.add_argument("--env", nargs="*", default=[], help="KEY=VALUE env var overrides")
    p_launch.add_argument("--no-spot", action="store_true", help="Disable Spot VMs")

    # batch
    p_batch = sub.add_parser("batch", help="Launch all experiments from YAML")
    p_batch.add_argument("--filter", default=None, help="Substring filter on experiment names")

    # status
    p_status = sub.add_parser("status", help="Check job status")
    p_status.add_argument("--name", default=None, help="Filter by experiment name prefix")
    p_status.add_argument("--limit", type=int, default=20, help="Max jobs to show")

    # download
    p_download = sub.add_parser("download", help="Download experiment results from GCS")
    p_download.add_argument("--name", required=True, help="Experiment name")
    p_download.add_argument("--output-dir", default="./results", help="Local output directory")

    # wait
    p_wait = sub.add_parser("wait", help="Wait for a job to complete")
    p_wait.add_argument("--job", required=True, help="Job display_name")
    p_wait.add_argument("--timeout", type=int, default=1200, help="Max wait seconds")
    p_wait.add_argument("--poll", type=int, default=30, help="Poll interval seconds")

    # cancel
    p_cancel = sub.add_parser("cancel", help="Cancel a running job")
    p_cancel.add_argument("--job", required=True, help="Job display_name")

    args = parser.parse_args()

    launcher = VertexLauncher.from_config(args.config)

    if args.command == "launch":
        env = _parse_env_args(args.env)
        result = launcher.launch(
            name=args.name,
            script=args.script,
            env=env,
            spot=not args.no_spot,
        )
        print(f"\nJob submitted:")
        print(f"  Name:     {result['job_name']}")
        print(f"  Resource: {result['resource_name']}")
        print(f"  Output:   {result['output_uri']}")
        print(f"  State:    {result['state']}")

    elif args.command == "batch":
        results = launcher.batch(config_path=args.config, filter=args.filter)
        print(f"\n{len(results)} job(s) launched.")

    elif args.command == "status":
        jobs = launcher.status(name=args.name, limit=args.limit)
        if not jobs:
            print("No jobs found.")
        else:
            print(f"{'JOB NAME':50s} {'STATE':25s} {'CREATED'}")
            print("-" * 100)
            for j in jobs:
                print(f"{j['job_name']:50s} {j['state']:25s} {j['create_time']}")

    elif args.command == "download":
        launcher.download(name=args.name, output_dir=args.output_dir)

    elif args.command == "wait":
        state = launcher.wait(
            job_name=args.job,
            poll_interval=args.poll,
            timeout=args.timeout,
        )
        print(f"\nFinal state: {state}")

    elif args.command == "cancel":
        launcher.cancel(job_name=args.job)


if __name__ == "__main__":
    main()
