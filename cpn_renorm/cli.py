from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

from .chain_tuning import run_chain_tuning
from .config import load_config
from .pipeline import run_pipeline
from .pilot import resolve_device, run_pilot
from .workspace import (WorkspaceChange, format_workspace_change,
                        inspect_workspace, write_workspace_state)


def _override(cfg: dict, args: argparse.Namespace) -> None:
    if getattr(args, "device", None):
        cfg["runtime"]["device"] = args.device
    if getattr(args, "seed", None) is not None:
        cfg["runtime"]["seed"] = args.seed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m cpn_renorm")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "pilot", "run"):
        p = sub.add_parser(name)
        p.add_argument("config", type=Path)
        p.add_argument("--device")
        p.add_argument("--seed", type=int)
        p.add_argument("--output-dir", type=Path, default=Path("runs"))
        if name != "validate":
            p.add_argument("--accept-changes", action="store_true")
        if name == "run":
            p.add_argument("--skip-topo", action="store_true")
            p.add_argument("--force", action="store_true",
                           help="regenerate cached ensembles and completed steps")
    batch = sub.add_parser("batch")
    batch.add_argument("pattern")
    batch.add_argument("--device")
    batch.add_argument("--seed", type=int)
    batch.add_argument("--output-dir", type=Path, default=Path("runs"))
    batch.add_argument("--skip-topo", action="store_true")
    batch.add_argument("--force", action="store_true")
    batch.add_argument("--accept-changes", action="store_true")
    tune = sub.add_parser("tune-chains")
    tune.add_argument("config", type=Path)
    tune.add_argument("--device")
    tune.add_argument("--seed", type=int)
    tune.add_argument("--output-dir", type=Path, default=Path("runs"))
    tune.add_argument("--max-chains", type=int, default=65536)
    tune.add_argument("--warmup-sweeps", type=int, default=5)
    tune.add_argument("--timed-sweeps", type=int, default=10)
    tune.add_argument("--gpu-memory-fraction", type=float, default=0.8)
    tune.add_argument("--max-host-memory-gb", type=float, default=8.0)
    tune.add_argument("--accept-changes", action="store_true")
    return parser


def _approve_changes(changes: list[WorkspaceChange], *, accepted: bool) -> None:
    pending = [change for change in changes if change.needs_confirmation]
    for change in pending:
        print(format_workspace_change(change), flush=True)
    if not pending or accepted:
        return
    if not sys.stdin.isatty():
        raise SystemExit(
            "configuration changed; rerun with --accept-changes to continue")
    answer = input("Continue? [y/N] ").strip().lower()
    if answer not in ("y", "yes"):
        raise SystemExit("configuration change was not accepted")


def _start_workspaces(changes: list[WorkspaceChange], command: str) -> None:
    for change in changes:
        write_workspace_state(change, command=command, status="running")


def _complete_workspace(change: WorkspaceChange, command: str) -> None:
    write_workspace_state(change, command=command, status="complete")


def _print_tuning(report: dict, path: Path) -> None:
    print("stage        chains  chain-sweeps/s  gpu GiB  host GiB  status", flush=True)
    for name, result in report["stages"].items():
        for index, row in enumerate(result["candidates"]):
            stage = name if index == 0 else ""
            speed = row.get("chain_sweeps_per_s")
            gpu = row.get("peak_gpu_gib")
            host = row.get("estimated_host_measurement_gib")
            status = "safe" if row.get("safe") else row.get("reason", "unsafe")
            print(f"{stage:<12} {row['chains']:>6}  "
                  f"{speed if speed is not None else float('nan'):>14.3f}  "
                  f"{gpu if gpu is not None else float('nan'):>7.3f}  "
                  f"{host if host is not None else float('nan'):>8.3f}  {status}",
                  flush=True)
        print(f"  -> recommended chains: {result['recommended_chains']}", flush=True)
        if result.get("search", {}).get("stop_reason") == "max_chains_reached":
            print("  -> warning: search was still improving at --max-chains",
                  flush=True)
    print("\nSuggested TOML:\n" + report["toml"], flush=True)
    print(f"\nFull report: {path}", flush=True)


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "batch":
        paths = sorted(glob.glob(args.pattern, recursive=True))
        if not paths:
            raise SystemExit(f"no TOML files matched {args.pattern!r}")
        prepared = []
        latest_by_workspace: dict[Path, WorkspaceChange] = {}
        for path in paths:
            cfg = load_config(path)
            _override(cfg, args)
            cfg["_force"] = args.force
            run_dir = (Path(args.output_dir).resolve() /
                       Path(cfg["_config_path"]).stem)
            change = inspect_workspace(
                cfg, args.output_dir, baseline=latest_by_workspace.get(run_dir))
            latest_by_workspace[run_dir] = change
            prepared.append((path, cfg, change))
        changes = [item[2] for item in prepared]
        _approve_changes(changes, accepted=args.accept_changes)
        for path, cfg, change in prepared:
            _start_workspaces([change], args.command)
            print(f"[cpn-renorm] running {path}", flush=True)
            run_pipeline(cfg, args.output_dir, skip_topo=args.skip_topo)
            _complete_workspace(change, args.command)
        return
    cfg = load_config(args.config)
    _override(cfg, args)
    cfg["_force"] = getattr(args, "force", False)
    if args.command == "validate":
        print(json.dumps({"valid": True, "device": str(resolve_device(cfg["runtime"]["device"])),
                          "config": str(args.config.resolve())}, indent=2))
        return

    change = inspect_workspace(cfg, args.output_dir)
    _approve_changes([change], accepted=args.accept_changes)
    _start_workspaces([change], args.command)
    if args.command == "pilot":
        result = run_pilot(cfg, change.run_dir)
        print(json.dumps(result, indent=2))
    elif args.command == "tune-chains":
        result = run_chain_tuning(
            cfg, args.output_dir, max_chains=args.max_chains,
            warmup_sweeps=args.warmup_sweeps, timed_sweeps=args.timed_sweeps,
            gpu_memory_fraction=args.gpu_memory_fraction,
            max_host_memory_gib=args.max_host_memory_gb)
        report_path = args.output_dir / args.config.stem / "chain_tuning.json"
        _print_tuning(result, report_path.resolve())
    else:
        summary = run_pipeline(cfg, args.output_dir, skip_topo=args.skip_topo)
        print(json.dumps(summary, indent=2))
    _complete_workspace(change, args.command)


if __name__ == "__main__":
    main(sys.argv[1:])
