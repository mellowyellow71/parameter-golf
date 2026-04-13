#!/usr/bin/env python3
"""
GCE Zone Scanner & Instance Provisioner for Parameter Golf.

Probes 19 a3-highgpu-8g zones for SPOT availability, falls back to on-demand,
creates instances from the golden image, returns connection details.

Usage:
    python infra/gce_provision.py probe [--spot-only]
    python infra/gce_provision.py create --name my-exp [--zone us-central1-a] [--no-spot]
    python infra/gce_provision.py delete --name INSTANCE --zone ZONE
    python infra/gce_provision.py ssh --name INSTANCE --zone ZONE [--command CMD]

Requires: gcloud CLI configured with project access.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class InstanceInfo:
    name: str
    zone: str
    external_ip: str
    internal_ip: str
    provisioning_model: str  # "SPOT" or "STANDARD"
    creation_time: str


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: str = "infra/gce_config.yaml") -> dict:
    """Load GCE configuration from YAML."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg["gce"]


# ---------------------------------------------------------------------------
# gcloud helpers
# ---------------------------------------------------------------------------

def _run_gcloud(
    args: list[str],
    timeout: int = 300,
    capture: bool = True,
) -> subprocess.CompletedProcess:
    """Execute a gcloud command. Raises CalledProcessError on failure."""
    cmd = ["gcloud"] + args
    return subprocess.run(
        cmd,
        capture_output=capture,
        text=True,
        timeout=timeout,
    )


def _run_gcloud_json(args: list[str], timeout: int = 300) -> dict | list | None:
    """Execute a gcloud command with --format=json, return parsed JSON."""
    result = _run_gcloud(args + ["--format=json"], timeout=timeout)
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Zone probing
# ---------------------------------------------------------------------------

def probe_zone(
    zone: str,
    project: str,
    machine_type: str = "a3-highgpu-8g",
) -> dict:
    """Check if a3-highgpu-8g is available in a zone.

    Returns {"zone": str, "available": bool, "error": str | None}.
    """
    result = _run_gcloud([
        "compute", "machine-types", "describe", machine_type,
        "--zone", zone,
        "--project", project,
    ])
    if result.returncode == 0:
        return {"zone": zone, "available": True, "error": None}
    error = result.stderr.strip() if result.stderr else "unknown error"
    return {"zone": zone, "available": False, "error": error}


def probe_all_zones(config: dict, spot_only: bool = False) -> list[dict]:
    """Probe all configured zones in parallel.

    Returns list of {"zone", "available", "error"} ordered by config priority.
    """
    zones = config["zones"]
    project = config["project"]
    machine_type = config["machine_type"]

    results = {}
    with ThreadPoolExecutor(max_workers=min(19, len(zones))) as pool:
        futures = {
            pool.submit(probe_zone, z, project, machine_type): z
            for z in zones
        }
        for future in as_completed(futures):
            zone = futures[future]
            try:
                results[zone] = future.result()
            except Exception as e:
                results[zone] = {"zone": zone, "available": False, "error": str(e)}

    # Return in config priority order
    return [results[z] for z in zones if z in results]


# ---------------------------------------------------------------------------
# Instance creation
# ---------------------------------------------------------------------------

def _build_create_cmd(
    instance_name: str,
    zone: str,
    config: dict,
    spot: bool = True,
) -> list[str]:
    """Build the gcloud compute instances create command.

    IMPORTANT: a3-highgpu-8g bundles 8x H100 GPUs and 16 local SSDs automatically.
    The --accelerator flag must NOT be used — it conflicts with bundled accelerators.
    """
    cmd = [
        "compute", "instances", "create", instance_name,
        "--project", config["project"],
        "--zone", zone,
        "--machine-type", config["machine_type"],
        "--image", config["golden_image"],
        "--image-project", config["project"],
        "--boot-disk-size", f"{config['boot_disk_size_gb']}GB",
        "--boot-disk-type", config["boot_disk_type"],
        "--network-interface", f"network={config['network']},nic-type=GVNIC",
        "--scopes", ",".join(config["scopes"]),
        "--maintenance-policy", "TERMINATE",
        "--no-restart-on-failure",
        "--metadata", "enable-osconfig=TRUE",
    ]

    if spot:
        cmd.extend([
            "--provisioning-model", "SPOT",
            "--instance-termination-action", "STOP",
        ])

    return cmd


def create_instance(
    name_suffix: str,
    zone: str,
    config: dict,
    spot: bool = True,
) -> Optional[InstanceInfo]:
    """Create an a3-highgpu-8g instance from the golden image.

    Returns InstanceInfo on success, None on failure.
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # Sanitize name: GCE names must be lowercase, alphanumeric, hyphens only
    safe_suffix = name_suffix.lower().replace("_", "-").replace(".", "-")
    instance_name = f"{config['instance_prefix']}-{safe_suffix}-{timestamp}"
    # GCE name max 63 chars
    instance_name = instance_name[:63]

    cmd = _build_create_cmd(instance_name, zone, config, spot=spot)
    prov_model = "SPOT" if spot else "STANDARD"

    print(f"  Creating {prov_model} instance {instance_name} in {zone}...", flush=True)
    result = _run_gcloud(cmd, timeout=300)

    if result.returncode != 0:
        error = result.stderr.strip() if result.stderr else "unknown"
        # Common capacity errors
        if any(phrase in error for phrase in [
            "ZONE_RESOURCE_POOL_EXHAUSTED",
            "RESOURCE_NOT_FOUND",
            "does not have enough resources",
            "resource pool exhausted",
            "Quota",
            "quota",
            "UNSUPPORTED_OPERATION",
        ]):
            print(f"    Capacity unavailable in {zone}: {error[:120]}")
        else:
            print(f"    Create failed in {zone}: {error[:200]}")
        return None

    # Fetch instance details to get IPs
    info = _run_gcloud_json([
        "compute", "instances", "describe", instance_name,
        "--zone", zone,
        "--project", config["project"],
    ])
    if not info:
        print(f"    Created but can't describe instance {instance_name}")
        return InstanceInfo(
            name=instance_name,
            zone=zone,
            external_ip="unknown",
            internal_ip="unknown",
            provisioning_model=prov_model,
            creation_time=timestamp,
        )

    # Extract IPs
    external_ip = "none"
    internal_ip = "none"
    for iface in info.get("networkInterfaces", []):
        internal_ip = iface.get("networkIP", "none")
        for ac in iface.get("accessConfigs", []):
            external_ip = ac.get("natIP", "none")

    return InstanceInfo(
        name=instance_name,
        zone=zone,
        external_ip=external_ip,
        internal_ip=internal_ip,
        provisioning_model=prov_model,
        creation_time=timestamp,
    )


def find_and_create(
    name_suffix: str,
    config: dict,
    exclude_zones: list[str] | None = None,
) -> Optional[InstanceInfo]:
    """Try zones in priority order until an instance is created.

    Phase 1: SPOT in all zones (if prefer_spot=true).
    Phase 2: STANDARD in all zones (if fallback_to_ondemand=true).

    Returns InstanceInfo or None if all zones exhausted.
    """
    if exclude_zones is None:
        exclude_zones = []

    zones = [z for z in config["zones"] if z not in exclude_zones]
    prefer_spot = config.get("prefer_spot", True)
    fallback = config.get("fallback_to_ondemand", True)

    phases = []
    if prefer_spot:
        phases.append(("SPOT", True))
    if fallback:
        phases.append(("ON-DEMAND", False))
    if not phases:
        phases.append(("SPOT", True))

    for phase_name, spot in phases:
        print(f"\n--- Trying {phase_name} across {len(zones)} zones ---")
        for zone in zones:
            instance = create_instance(name_suffix, zone, config, spot=spot)
            if instance is not None:
                print(f"\n  Instance created: {instance.name}")
                print(f"  Zone: {instance.zone}")
                print(f"  External IP: {instance.external_ip}")
                print(f"  Model: {instance.provisioning_model}")
                return instance

    print("\n  All zones exhausted. No instance could be created.")
    return None


# ---------------------------------------------------------------------------
# SSH helpers
# ---------------------------------------------------------------------------

def wait_for_ssh(
    instance: InstanceInfo,
    config: dict,
    timeout: int | None = None,
) -> bool:
    """Block until SSH is available on the instance. Returns True on success."""
    if timeout is None:
        timeout = config.get("ssh_timeout_seconds", 300)
    ssh_user = config.get("ssh_user", "ray")
    project = config["project"]

    print(f"  Waiting for SSH on {instance.name} ({instance.zone})...", flush=True)
    start = time.time()
    attempt = 0

    while time.time() - start < timeout:
        attempt += 1
        result = _run_gcloud([
            "compute", "ssh", f"{ssh_user}@{instance.name}",
            "--zone", instance.zone,
            "--project", project,
            "--strict-host-key-checking=no",
            "--command", "echo ready",
            "--ssh-flag=-o ConnectTimeout=10",
        ], timeout=30)

        if result.returncode == 0 and "ready" in result.stdout:
            elapsed = int(time.time() - start)
            print(f"  SSH ready after {elapsed}s ({attempt} attempts)")
            return True

        if attempt % 3 == 0:
            elapsed = int(time.time() - start)
            print(f"    Still waiting... ({elapsed}s elapsed)")
        time.sleep(10)

    print(f"  SSH timed out after {timeout}s")
    return False


def ssh_exec(
    instance: InstanceInfo,
    config: dict,
    command: str,
    timeout: int = 600,
) -> subprocess.CompletedProcess:
    """Execute a command on the instance via SSH."""
    ssh_user = config.get("ssh_user", "ray")
    return _run_gcloud([
        "compute", "ssh", f"{ssh_user}@{instance.name}",
        "--zone", instance.zone,
        "--project", config["project"],
        "--strict-host-key-checking=no",
        "--command", command,
    ], timeout=timeout)


def scp_to_instance(
    instance: InstanceInfo,
    config: dict,
    local_paths: list[str],
    remote_dir: str,
) -> bool:
    """SCP files to the instance. Returns True on success."""
    ssh_user = config.get("ssh_user", "ray")
    remote = f"{ssh_user}@{instance.name}:{remote_dir}"

    cmd = [
        "compute", "scp",
        *local_paths,
        remote,
        "--zone", instance.zone,
        "--project", config["project"],
        "--strict-host-key-checking=no",
    ]
    result = _run_gcloud(cmd, timeout=120)
    return result.returncode == 0


# ---------------------------------------------------------------------------
# Instance status & cleanup
# ---------------------------------------------------------------------------

def get_instance_status(
    name: str,
    zone: str,
    project: str,
) -> str:
    """Get instance status. Returns status string or 'NOT_FOUND'."""
    result = _run_gcloud([
        "compute", "instances", "describe", name,
        "--zone", zone,
        "--project", project,
        "--format", "value(status)",
    ])
    if result.returncode == 0:
        return result.stdout.strip()
    return "NOT_FOUND"


def delete_instance(name: str, zone: str, project: str) -> bool:
    """Delete a GCE instance. Returns True on success."""
    print(f"  Deleting instance {name} in {zone}...")
    result = _run_gcloud([
        "compute", "instances", "delete", name,
        "--zone", zone,
        "--project", project,
        "--quiet",
    ], timeout=120)
    if result.returncode == 0:
        print(f"  Deleted: {name}")
        return True
    print(f"  Delete failed: {result.stderr.strip() if result.stderr else 'unknown'}")
    return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_probe(args, config):
    """Probe all zones for availability."""
    results = probe_all_zones(config, spot_only=args.spot_only)
    print(f"\n{'ZONE':30s} {'AVAILABLE':12s} {'ERROR'}")
    print("-" * 80)
    available_count = 0
    for r in results:
        avail = "YES" if r["available"] else "NO"
        error = r.get("error") or ""
        if r["available"]:
            available_count += 1
        print(f"{r['zone']:30s} {avail:12s} {error[:40]}")
    print(f"\n{available_count}/{len(results)} zones have a3-highgpu-8g")


def cmd_create(args, config):
    """Create an instance."""
    if args.zone:
        instance = create_instance(
            args.name, args.zone, config, spot=not args.no_spot,
        )
    else:
        instance = find_and_create(args.name, config)

    if instance is None:
        print("\nFailed to create instance.")
        sys.exit(1)

    # Wait for SSH
    if not args.no_wait:
        ssh_ok = wait_for_ssh(instance, config)
        if not ssh_ok:
            print("SSH not available. Instance is up but not reachable.")

    print(f"\nInstance ready:")
    print(f"  Name:     {instance.name}")
    print(f"  Zone:     {instance.zone}")
    print(f"  IP:       {instance.external_ip}")
    print(f"  Model:    {instance.provisioning_model}")
    print(f"\nConnect with:")
    ssh_user = config.get("ssh_user", "ray")
    print(f"  gcloud compute ssh {ssh_user}@{instance.name} --zone={instance.zone} --project={config['project']}")
    print(f"\nDelete with:")
    print(f"  python infra/gce_provision.py delete --name {instance.name} --zone {instance.zone}")

    # Write instance info for other scripts
    info_path = Path("infra") / f".instance-{args.name}.json"
    info_path.write_text(json.dumps(asdict(instance), indent=2))
    print(f"\nInstance info saved to: {info_path}")


def cmd_delete(args, config):
    """Delete an instance."""
    ok = delete_instance(args.name, args.zone, config["project"])
    if not ok:
        sys.exit(1)
    # Clean up instance info file
    for p in Path("infra").glob(f".instance-*.json"):
        try:
            data = json.loads(p.read_text())
            if data.get("name") == args.name:
                p.unlink()
        except Exception:
            pass


def cmd_ssh(args, config):
    """SSH into an instance or run a command."""
    info = InstanceInfo(
        name=args.name, zone=args.zone,
        external_ip="", internal_ip="",
        provisioning_model="", creation_time="",
    )
    if args.command:
        result = ssh_exec(info, config, args.command)
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        sys.exit(result.returncode)
    else:
        # Interactive SSH
        ssh_user = config.get("ssh_user", "ray")
        subprocess.run([
            "gcloud", "compute", "ssh", f"{ssh_user}@{args.name}",
            "--zone", args.zone,
            "--project", config["project"],
            "--strict-host-key-checking=no",
        ])


def main():
    parser = argparse.ArgumentParser(
        description="Parameter Golf — GCE Zone Scanner & Provisioner",
    )
    parser.add_argument(
        "--config", default="infra/gce_config.yaml",
        help="Path to GCE config YAML",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # probe
    p_probe = sub.add_parser("probe", help="Probe zones for a3-highgpu-8g availability")
    p_probe.add_argument("--spot-only", action="store_true", help="Only check SPOT availability")

    # create
    p_create = sub.add_parser("create", help="Create an a3-highgpu-8g instance")
    p_create.add_argument("--name", required=True, help="Experiment name suffix")
    p_create.add_argument("--zone", default=None, help="Specific zone (skip zone scanning)")
    p_create.add_argument("--no-spot", action="store_true", help="Use on-demand instead of SPOT")
    p_create.add_argument("--no-wait", action="store_true", help="Don't wait for SSH")

    # delete
    p_delete = sub.add_parser("delete", help="Delete an instance")
    p_delete.add_argument("--name", required=True, help="Instance name")
    p_delete.add_argument("--zone", required=True, help="Instance zone")

    # ssh
    p_ssh = sub.add_parser("ssh", help="SSH into an instance")
    p_ssh.add_argument("--name", required=True, help="Instance name")
    p_ssh.add_argument("--zone", required=True, help="Instance zone")
    p_ssh.add_argument("--command", default=None, help="Command to run (interactive if omitted)")

    args = parser.parse_args()
    config = load_config(args.config)

    if args.command == "probe":
        cmd_probe(args, config)
    elif args.command == "create":
        cmd_create(args, config)
    elif args.command == "delete":
        cmd_delete(args, config)
    elif args.command == "ssh":
        cmd_ssh(args, config)


if __name__ == "__main__":
    main()
