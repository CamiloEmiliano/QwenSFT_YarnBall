#!/usr/bin/env python3
"""
Automated Vast.ai Datacenter GPU Runner for QwenSFT_YarnBall.

Enforces strict institutional security:
- datacenter=true: strictly enterprise colocation facilities (zero residential miners).
- verified=true: physically and network verified by Vast.ai.
- Automated provisioning, SSH credential handling, training launch, and teardown.
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


def search_offers(
    gpu_name: str = "A100_SXM4",
    min_vram: int = 40,
    max_dph: float = 2.50,
    min_down: float = 100.0,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """Query verified enterprise datacenter offers."""
    filters = [
        "datacenter=true",
        "verified=true",
        "num_gpus=1",
        "rented=false",
        f"gpu_ram >= {min_vram}",
        f"dph_total <= {max_dph}",
        f"inet_down >= {min_down}",
    ]
    if gpu_name and gpu_name.lower() not in ("any", "all"):
        filters.append(f"gpu_name={gpu_name}")

    query_str = " ".join(filters)
    cmd = ["search", "offers", query_str, "--order", "dph+"]
    raw = run_vast_cmd(cmd, raw_json=True)
    if isinstance(raw, list):
        return raw[:limit]
    return []


def format_offers_table(offers: List[Dict[str, Any]]) -> str:
    """Format offers into a clean scannable text table."""
    if not offers:
        return "No matching verified datacenter instances found within constraints."

    header = f"{'Offer ID':<10} {'GPU Model':<18} {'VRAM':<8} {'$/hr':<8} {'DL (Mbps)':<10} {'Country':<8} {'Reliability':<12}"
    lines = [header, "-" * len(header)]

    for o in offers:
        oid = str(o.get("id", "-"))
        gpu = str(o.get("gpu_name", "-"))[:17]
        raw_vram = float(o.get("gpu_ram", 0))
        vram_gb = raw_vram / 1024 if raw_vram > 500 else raw_vram
        vram = f"{vram_gb:.0f} GB"
        dph = f"${float(o.get('dph_total', 0)):.2f}"
        down = f"{float(o.get('inet_down', 0)):.0f}"
        country = str(o.get("country", "-"))[:7]
        rel = f"{float(o.get('reliability2', 0)) * 100:.1f}%"
        lines.append(f"{oid:<10} {gpu:<18} {vram:<8} {dph:<8} {down:<10} {country:<8} {rel:<12}")

    return "\n".join(lines)



def cmd_search(args: argparse.Namespace) -> None:
    """Handle 'search' subcommand."""
    print(f"\nSearching verified datacenter offers (GPU: {args.gpu}, max ${args.max_price:.2f}/hr)...")
    offers = search_offers(
        gpu_name=args.gpu,
        min_vram=args.min_vram,
        max_dph=args.max_price,
        limit=args.limit,
    )
    print("\n" + format_offers_table(offers) + "\n")


def cmd_launch(args: argparse.Namespace) -> None:
    """Handle 'launch' subcommand."""
    offer_id = args.offer_id

    if not offer_id:
        print(f"Searching best available verified datacenter offer (GPU: {args.gpu}, max ${args.max_price:.2f}/hr)...")
        offers = search_offers(gpu_name=args.gpu, min_vram=args.min_vram, max_dph=args.max_price, limit=1)
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
    p_search = subparsers.add_parser("search", help="Search verified datacenter GPU instances")
    p_search.add_argument("--gpu", default="A100_SXM4", help="GPU filter: A100_SXM4, A100_PCIE, A100, L40S, any")
    p_search.add_argument("--min-vram", type=int, default=40, help="Minimum GPU VRAM in GB")
    p_search.add_argument("--max-price", type=float, default=2.50, help="Maximum price per hour ceiling ($/hr)")
    p_search.add_argument("--limit", type=int, default=10, help="Max results to display")
    p_search.set_defaults(func=cmd_search)

    # Subcommand: launch
    p_launch = subparsers.add_parser("launch", help="Provision a verified datacenter instance")
    p_launch.add_argument("--offer-id", type=int, default=None, help="Specific offer ID to rent (optional)")
    p_launch.add_argument("--gpu", default="A100_SXM4", help="GPU filter if offer-id is not specified")
    p_launch.add_argument("--min-vram", type=int, default=40, help="Minimum GPU VRAM in GB")
    p_launch.add_argument("--max-price", type=float, default=2.00, help="Maximum price per hour ceiling ($/hr)")
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
