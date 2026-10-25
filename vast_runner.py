#!/usr/bin/env python3
"""
Vast.ai GPU Automation & Provisioning Runner for QwenSFT_YarnBall.
==================================================================

This script simplifies searching for, renting, training on, and terminating
cloud GPUs on Vast.ai for fine-tuning the Qwen2.5-7B YarnBall financial model.

Quickstart Workflow (4 Steps):
------------------------------
1. Search for available GPUs:
   python vast_runner.py search --gpu A100 --min-vram 80 --min-reliability 99.0

2. Launch and auto-train on your chosen Offer ID:
   python vast_runner.py launch --offer-id 32303769 --auto-train

3. Check instance status & SSH connection details:
   python vast_runner.py status

4. Destroy instance when training is complete (STOPS ALL BILLING):
   python vast_runner.py destroy <INSTANCE_ID> -y


Subcommands & Command-Line Options:
-----------------------------------

1. `search` - Find and filter available cloud GPU offers
   -----------------------------------------------------
   Usage: python vast_runner.py search [OPTIONS]

   Filtering Options:
     --gpu <NAME>            Filter GPU model name (case-insensitive substring).
                             Examples: 'A100', 'H100', '4090', '6000', or 'any'.
                             Default: 'any'.

     --min-vram <GB>         Minimum GPU memory in Gigabytes.
                             Examples: 24 (RTX 4090 / A5000), 40 (A100-40GB), 80 (A100/H100-80GB).
                             Default: 24.

     --max-price <$/hr>      Maximum hourly price ceiling in USD.
                             Examples: 1.50, 2.50, 3.50.
                             Default: 3.50.

     --country <CODE>        Filter by 2-letter country code or geographic region.
                             Examples: 'US' (United States), 'GB' (United Kingdom), 'SE' (Sweden), 'CZ'.
                             Default: None (searches worldwide).

     --min-reliability <PCT> Minimum host uptime reliability percentage.
                             Examples: 98 (for 98%), 99.5.
                             Default: None (displays all reliability levels).

     --max-compute-cap <CAP> Maximum CUDA compute capability to avoid unvetted architectures.
                             Tip: Use '900' (or '9.0') to strictly filter out Blackwell ('1200')
                             while keeping Ampere ('800') and Hopper ('900') cards.
                             Default: None.

     --min-compute-cap <CAP> Minimum CUDA compute capability.
                             Tip: Use '800' (or '8.0') to ensure modern Ampere/Hopper Tensor Cores
                             and exclude older Volta/Turing architectures.
                             Default: None.

     --min-mem-bw <GB/s>     Minimum physical GPU memory bandwidth in GB/s.
                             Tip: Setting to 1000 ensures high-speed HBM2e memory (A100/H100).
                             Default: None.

     --nvlink                Require high-speed NVLink interconnect (SXM4/SXM5 nodes).
                             Filters out PCIe-only bus cards.

     --datacenter            Restrict search strictly to institutional datacenter hosts.

     --verified              Restrict search strictly to hosts verified by Vast.ai staff.

     --limit <N>             Maximum number of offers to display in the table.
                             Default: 25.


2. `launch` - Provision an instance and start training
   ----------------------------------------------------
   Usage: python vast_runner.py launch [OPTIONS]

   Launch Options:
     --offer-id <ID>         The exact Offer ID chosen from the search table.
                             (If omitted, the script automatically rents the cheapest offer
                             matching your search criteria).

     --auto-train            1-Click Autonomous Mode: Upon container boot, automatically
                             clones the repository, installs dependencies, pulls Hugging Face
                             credentials from .env, and executes the fine-tuning job in the
                             background (logging to /workspace/QwenSFT_YarnBall/training.log).

     --disk <GB>             Storage disk partition size in Gigabytes.
                             Default: 50.

     --image <DOCKER_IMAGE>  Docker base container image to deploy.
                             Default: 'pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel'.

     --dry-run               Simulates provisioning without actually renting or spending credits.

     -y, --yes               Bypasses the interactive confirmation prompt.


3. `status` - Check active instances
   ---------------------------------
   Usage: python vast_runner.py status
   Displays Instance ID, run status (loading, running), GPU model, hourly cost,
   and the ready-to-use SSH connection string.


4. `ssh` - Connect to your running instance
   ----------------------------------------
   Usage: python vast_runner.py ssh [INSTANCE_ID]
   Prints the exact SSH command to connect to your container terminal.


5. `destroy` - Terminate an instance and stop billing
   --------------------------------------------------
   Usage: python vast_runner.py destroy <INSTANCE_ID> [-y]
   Permanently deletes the container and immediately stops all billing charges.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# Load workspace environment variables
SCRIPT_DIR = Path(__file__).resolve().parent
load_dotenv(SCRIPT_DIR / ".env")


def get_api_key() -> str:
    """Resolve Vast.ai API key from environment."""
    key = os.environ.get("VASTAI_FULL_ACCESS_TOKEN_01") or os.environ.get("VAST_API_KEY")
    if not key:
        print("Error: Vast.ai API key not found in .env (expected VASTAI_FULL_ACCESS_TOKEN_01).", file=sys.stderr)
        sys.exit(1)
    return key.strip()


def get_hf_token() -> Optional[str]:
    """Resolve Hugging Face write token from environment."""
    return os.environ.get("HUGGINGFACE_FULL_ACCESS_TOKEN_01") or os.environ.get("HF_TOKEN")


def find_vastai_bin() -> str:
    """Find vastai executable in virtualenv or PATH."""
    venv_vast = SCRIPT_DIR / ".venv" / "bin" / "vastai"
    if venv_vast.is_file() and os.access(venv_vast, os.X_OK):
        return str(venv_vast)
    which_vast = shutil.which("vastai")
    if which_vast:
        return which_vast
    print("Error: 'vastai' CLI binary not found in .venv or PATH.", file=sys.stderr)
    sys.exit(1)


def run_vast_cmd(cmd_args: List[str], raw_json: bool = True) -> Any:
    """Execute vastai CLI command with authentication."""
    vast_bin = find_vastai_bin()
    api_key = get_api_key()

    full_cmd = [vast_bin, "--api-key", api_key]
    if raw_json and "--raw" not in cmd_args:
        full_cmd.append("--raw")
    full_cmd.extend(cmd_args)

    res = subprocess.run(full_cmd, capture_output=True, text=True)
    if res.returncode != 0:
        err = res.stderr.strip() or res.stdout.strip()
        print(f"Vast.ai API Error: {err}", file=sys.stderr)
        return None

    out = res.stdout.strip()
    if raw_json and out:
        try:
            return json.loads(out)
        except json.JSONDecodeError:
            return out
    return out


def extract_country(geolocation: str) -> str:
    """Extract 2-letter country code or clean location from geolocation string."""
    if not geolocation or not isinstance(geolocation, str):
        return "-"
    parts = [p.strip() for p in geolocation.split(",") if p.strip()]
    return parts[-1] if parts else "-"


def normalize_compute_cap(val: Optional[float]) -> Optional[int]:
    """Normalize compute capability (e.g. 8.0 -> 800, 80 -> 800, 800 -> 800)."""
    if val is None:
        return None
    val_float = float(val)
    if val_float < 15.0:
        return int(round(val_float * 100))
    elif val_float < 150.0:
        return int(round(val_float * 10))
    return int(round(val_float))


def search_offers(
    gpu_name: str = "any",
    min_vram: int = 24,
    min_disk: int = 40,
    max_dph: float = 3.50,
    min_down: float = 50.0,
    country: Optional[str] = None,
    min_reliability: Optional[float] = None,
    datacenter_only: bool = False,
    verified_only: bool = False,
    min_compute_cap: Optional[float] = None,
    max_compute_cap: Optional[float] = None,
    min_mem_bw: Optional[float] = None,
    nvlink_only: bool = False,
    limit: int = 25,
) -> List[Dict[str, Any]]:
    """Query offers with flexible, relaxed defaults."""
    filters = [
        "num_gpus=1",
        "rented=false",
        f"gpu_ram >= {min_vram}",
        f"disk_space >= {min_disk}",
        f"dph_total <= {max_dph}",
        f"inet_down >= {min_down}",
    ]
    if datacenter_only:
        filters.append("datacenter=true")
    if verified_only:
        filters.append("verified=true")

    query_str = " ".join(filters)
    cmd = ["search", "offers", query_str, "--order", "dph+"]
    raw = run_vast_cmd(cmd, raw_json=True)
    if isinstance(raw, list):
        filtered = raw
        # Flexible GPU substring matching (e.g. 'A100' matches 'A100 PCIE', 'A100_SXM4', 'A100-80GB')
        if gpu_name and gpu_name.lower() not in ("any", "all"):
            target_gpu = gpu_name.strip().upper().replace("_", " ").replace("-", " ")
            filtered = [
                o for o in filtered
                if target_gpu in str(o.get("gpu_name", "")).upper().replace("_", " ").replace("-", " ")
            ]
        if country:
            country_target = country.strip().upper()
            filtered = [
                o for o in filtered
                if extract_country(str(o.get("geolocation", ""))).upper() == country_target
                or country_target in str(o.get("geolocation", "")).upper()
            ]
        if min_reliability is not None:
            thresh = min_reliability / 100.0 if min_reliability > 1.0 else min_reliability
            filtered = [
                o for o in filtered
                if float(o.get("reliability2", 0)) >= thresh
            ]
        if min_compute_cap is not None:
            min_cap = normalize_compute_cap(min_compute_cap)
            filtered = [
                o for o in filtered
                if int(o.get("compute_cap") or 0) >= min_cap
            ]
        if max_compute_cap is not None:
            max_cap = normalize_compute_cap(max_compute_cap)
            filtered = [
                o for o in filtered
                if int(o.get("compute_cap") or 0) <= max_cap
            ]
        if min_mem_bw is not None:
            filtered = [
                o for o in filtered
                if float(o.get("gpu_mem_bw") or 0) >= float(min_mem_bw)
            ]
        if nvlink_only:
            filtered = [
                o for o in filtered
                if float(o.get("bw_nvlink") or 0) > 0.0
            ]
        return filtered[:limit]
    return []


def format_offers_table(offers: List[Dict[str, Any]]) -> str:
    """Format offers into a clean scannable text table."""
    if not offers:
        return "No matching verified datacenter instances found within constraints."

    header = f"{'Offer ID':<10} {'GPU Model':<18} {'VRAM':<8} {'Disk Space':<12} {'$/hr':<8} {'DL (Mbps)':<10} {'Country':<8} {'Reliability':<12}"
    lines = [header, "-" * len(header)]

    for o in offers:
        oid = str(o.get("id", "-"))
        gpu = str(o.get("gpu_name", "-"))[:17]
        raw_vram = float(o.get("gpu_ram", 0))
        vram_gb = raw_vram / 1024 if raw_vram > 500 else raw_vram
        vram = f"{vram_gb:.0f} GB"
        raw_disk = float(o.get("disk_space", 0))
        disk = f"{raw_disk:.0f} GB"
        dph = f"${float(o.get('dph_total', 0)):.2f}"
        down = f"{float(o.get('inet_down', 0)):.0f}"
        geo = str(o.get("geolocation", "-"))
        loc = extract_country(geo)
        rel = f"{float(o.get('reliability2', 0)) * 100:.1f}%"
        lines.append(f"{oid:<10} {gpu:<18} {vram:<8} {disk:<12} {dph:<8} {down:<10} {loc:<8} {rel:<12}")

    return "\n".join(lines)



def cmd_search(args: argparse.Namespace) -> None:
    """Handle 'search' subcommand."""
    country_msg = f", Country: {args.country}" if getattr(args, "country", None) else ""
    rel_val = getattr(args, "min_reliability", None)
    rel_msg = f", Min Rel: {rel_val:.1f}%" if rel_val is not None else ""
    print(f"\nSearching available offers (GPU: {args.gpu}, max ${args.max_price:.2f}/hr{country_msg}{rel_msg})...")
    offers = search_offers(
        gpu_name=args.gpu,
        min_vram=args.min_vram,
        min_disk=getattr(args, "min_disk", 40),
        max_dph=args.max_price,
        country=args.country,
        min_reliability=args.min_reliability,
        datacenter_only=args.datacenter,
        verified_only=args.verified,
        min_compute_cap=getattr(args, "min_compute_cap", None),
        max_compute_cap=getattr(args, "max_compute_cap", None),
        min_mem_bw=getattr(args, "min_mem_bw", None),
        nvlink_only=getattr(args, "nvlink", False),
        limit=args.limit,
    )
    print("\n" + format_offers_table(offers) + "\n")


def cmd_launch(args: argparse.Namespace) -> None:
    """Handle 'launch' subcommand."""
    offer_id = args.offer_id

    if not offer_id:
        print(f"Searching best available verified datacenter offer (GPU: {args.gpu}, max ${args.max_price:.2f}/hr)...")
        offers = search_offers(
            gpu_name=args.gpu,
            min_vram=args.min_vram,
            max_dph=args.max_price,
            country=args.country,
            min_reliability=args.min_reliability,
            datacenter_only=args.datacenter,
            verified_only=args.verified,
            min_compute_cap=getattr(args, "min_compute_cap", None),
            max_compute_cap=getattr(args, "max_compute_cap", None),
            min_mem_bw=getattr(args, "min_mem_bw", None),
            nvlink_only=getattr(args, "nvlink", False),
            limit=1,
        )
        if not offers:
            print("No matching offers found. Try relaxing price ceiling with --max-price or changing --gpu.", file=sys.stderr)
            sys.exit(1)
        best = offers[0]
        offer_id = best["id"]
        print(f"\nSelected Top Verified Offer: ID {offer_id} ({best.get('gpu_name')}, ${best.get('dph_total'):.2f}/hr, {best.get('country')})")

    if args.dry_run:
        print(f"[DRY-RUN] Would provision offer ID {offer_id} with image {args.image} and {args.disk}GB disk.")
        return

    if not args.yes:
        confirm = input(f"Proceed with provisioning offer {offer_id}? [y/N]: ").strip().lower()
        if confirm not in ("y", "yes"):
            print("Aborted.")
            return

    create_args = [
        "create", "instance", str(offer_id),
        "--image", args.image,
        "--disk", str(args.disk),
        "--ssh",
        "--direct",
    ]

    hf_token = get_hf_token()
    if args.auto_train:
        # Script automatically runs upon container boot
        token_env = f"export HF_TOKEN='{hf_token}';" if hf_token else ""
        onstart_cmd = (
            f"bash -c 'apt-get update && apt-get install -y git && "
            f"git clone https://github.com/CamiloEmiliano/QwenSFT_YarnBall.git && "
            f"cd QwenSFT_YarnBall && {token_env} bash launch_vast.sh > training.log 2>&1 &'"
        )
        create_args.extend(["--onstart-cmd", onstart_cmd])

    print(f"\nProvisioning instance from offer {offer_id}...")
    res = run_vast_cmd(create_args, raw_json=True)
    print("Response:", res)
    print("\nInstance creation requested! Check status with: python vast_runner.py status")


def cmd_status(args: argparse.Namespace) -> None:
    """Handle 'status' subcommand."""
    instances = run_vast_cmd(["show", "instances"], raw_json=True)
    if not isinstance(instances, list) or len(instances) == 0:
        print("\nNo active Vast.ai instances running.\n")
        return

    header = f"{'Instance ID':<12} {'Status':<12} {'GPU Model':<16} {'$/hr':<8} {'SSH Command'}"
    lines = ["\n" + header, "-" * 75]

    for inst in instances:
        iid = str(inst.get("id", "-"))
        status = str(inst.get("actual_status", inst.get("status_msg", "-")))
        gpu = str(inst.get("gpu_name", "-"))[:15]
        dph = f"${float(inst.get('dph_total', 0)):.2f}"
        ssh_host = inst.get("ssh_host", "")
        ssh_port = inst.get("ssh_port", "")
        ssh_cmd = f"ssh -p {ssh_port} root@{ssh_host}" if ssh_host and ssh_port else "pending..."
        lines.append(f"{iid:<12} {status:<12} {gpu:<16} {dph:<8} {ssh_cmd}")

    print("\n".join(lines) + "\n")


def cmd_ssh(args: argparse.Namespace) -> None:
    """Print or connect to active instance."""
    instances = run_vast_cmd(["show", "instances"], raw_json=True)
    if not isinstance(instances, list) or len(instances) == 0:
        print("No active instances found.", file=sys.stderr)
        return

    target = None
    if args.instance_id:
        for inst in instances:
            if str(inst.get("id")) == str(args.instance_id):
                target = inst
                break
    else:
        target = instances[0]

    if not target:
        print(f"Instance {args.instance_id} not found.", file=sys.stderr)
        return

    ssh_host = target.get("ssh_host")
    ssh_port = target.get("ssh_port")
    if not (ssh_host and ssh_port):
        print(f"Instance {target.get('id')} is not ready yet (Status: {target.get('actual_status')}).")
        return

    ssh_cmd = f"ssh -p {ssh_port} root@{ssh_host}"
    print(f"\nConnect with:\n  {ssh_cmd}\n")


def cmd_destroy(args: argparse.Namespace) -> None:
    """Handle 'destroy' subcommand to terminate instance and avoid idle charges."""
    iid = str(args.instance_id)
    if not args.yes:
        confirm = input(f"Are you sure you want to DESTROY instance {iid}? Billing will stop immediately. [y/N]: ").strip().lower()
        if confirm not in ("y", "yes"):
            print("Aborted.")
            return

    res = run_vast_cmd(["destroy", "instance", iid, "--yes"], raw_json=True)
    print(f"Instance {iid} terminated:", res)


def main():
    parser = argparse.ArgumentParser(description="Vast.ai Datacenter GPU Automation Runner")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # Subcommand: search
    p_search = subparsers.add_parser("search", help="Search available GPU instances")
    p_search.add_argument("--gpu", default="any", help="GPU filter: A100, 4090, 6000, A5000, any")
    p_search.add_argument("--min-vram", type=int, default=24, help="Minimum GPU VRAM in GB (default: 24)")
    p_search.add_argument("--min-disk", type=int, default=40, help="Minimum available disk space in GB (default: 40)")
    p_search.add_argument("--max-price", type=float, default=3.50, help="Maximum price per hour ceiling ($/hr)")
    p_search.add_argument("--country", default=None, help="Filter by country code (e.g. US, GB, IE, SE)")
    p_search.add_argument("--min-reliability", type=float, default=None, help="Minimum host reliability percentage (e.g. 98 for 98%%)")
    p_search.add_argument("--datacenter", action="store_true", help="Restrict strictly to datacenter hosts")
    p_search.add_argument("--verified", action="store_true", help="Restrict strictly to verified badge hosts")
    p_search.add_argument("--min-compute-cap", type=float, default=None, help="Minimum CUDA compute capability (e.g. 8.0, 80, or 800 for Ampere)")
    p_search.add_argument("--max-compute-cap", type=float, default=None, help="Maximum CUDA compute capability (e.g. 9.0, 90, or 900 to exclude Blackwell)")
    p_search.add_argument("--min-mem-bw", type=float, default=None, help="Minimum GPU memory bandwidth in GB/s (e.g. 1000 for HBM2e)")
    p_search.add_argument("--nvlink", action="store_true", help="Require NVLink interconnect (SXM4/SXM5)")
    p_search.add_argument("--limit", type=int, default=25, help="Max results to display (default: 25)")
    p_search.set_defaults(func=cmd_search)

    # Subcommand: launch
    p_launch = subparsers.add_parser("launch", help="Provision a verified datacenter instance")
    p_launch.add_argument("--offer-id", type=int, default=None, help="Specific offer ID to rent (optional)")
    p_launch.add_argument("--gpu", default="any", help="GPU filter if offer-id is not specified")
    p_launch.add_argument("--min-vram", type=int, default=24, help="Minimum GPU VRAM in GB (default: 24)")
    p_launch.add_argument("--max-price", type=float, default=3.50, help="Maximum price per hour ceiling ($/hr)")
    p_launch.add_argument("--country", default=None, help="Filter by country code (e.g. US, GB, IE, SE)")
    p_launch.add_argument("--min-reliability", type=float, default=95.0, help="Minimum host reliability percentage (default: 95.0%%)")
    p_launch.add_argument("--datacenter", action="store_true", help="Restrict strictly to datacenter hosts")
    p_launch.add_argument("--verified", action="store_true", help="Restrict strictly to verified badge hosts")
    p_launch.add_argument("--min-compute-cap", type=float, default=None, help="Minimum CUDA compute capability")
    p_launch.add_argument("--max-compute-cap", type=float, default=None, help="Maximum CUDA compute capability (e.g. 900 to exclude Blackwell)")
    p_launch.add_argument("--min-mem-bw", type=float, default=None, help="Minimum GPU memory bandwidth in GB/s")
    p_launch.add_argument("--nvlink", action="store_true", help="Require NVLink interconnect")
    p_launch.add_argument("--disk", type=int, default=50, help="Local disk partition in GB")
    p_launch.add_argument("--image", default="pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel", help="Docker image")
    p_launch.add_argument("--auto-train", action="store_true", help="Automatically run training upon container boot")
    p_launch.add_argument("--dry-run", action="store_true", help="Simulate provisioning without renting")
    p_launch.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompt")
    p_launch.set_defaults(func=cmd_launch)

    # Subcommand: status
    p_status = subparsers.add_parser("status", help="List active instances and SSH info")
    p_status.set_defaults(func=cmd_status)

    # Subcommand: ssh
    p_ssh = subparsers.add_parser("ssh", help="Print SSH connection string for an instance")
    p_ssh.add_argument("instance_id", nargs="?", default=None, help="Instance ID (optional, defaults to first active)")
    p_ssh.set_defaults(func=cmd_ssh)

    # Subcommand: destroy
    p_destroy = subparsers.add_parser("destroy", help="Terminate instance to stop billing")
    p_destroy.add_argument("instance_id", help="Instance ID to terminate")
    p_destroy.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompt")
    p_destroy.set_defaults(func=cmd_destroy)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
