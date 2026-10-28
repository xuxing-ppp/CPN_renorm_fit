"""Collect run warnings using active stages and their measurement references."""
from __future__ import annotations

import math
from pathlib import Path

from .cache import read_json
from .storage import atomic_json


def measurement_warnings(directory: Path, measured: dict,
                         previous: list[dict] | None = None) -> list[dict]:
    """Keep warnings even for exploratory ensembles removed after fitting."""
    warnings = list(previous or [])
    for result in measured.values():
        if result.get("_folder"):
            source = str((Path(result["_folder"]) / "result.json").relative_to(directory))
            for message in result.get("diagnostics", {}).get("warnings", []):
                entry = {"source": source, "location": "diagnostics", "message": message}
                if entry not in warnings:
                    warnings.append(entry)
    return warnings


def collect_diagnostics(root: Path, summary: dict, cfg: dict) -> dict:
    """Aggregate warnings without loading arrays or including unused caches."""
    root = Path(root)
    warnings = []
    seen = set()

    def add(path: Path, location: str, message: str) -> None:
        key = (str(path.relative_to(root)), location, message)
        if key not in seen:
            seen.add(key)
            warnings.append(dict(zip(("source", "location", "message"), key)))

    def walk(value, path: Path, location: str, directory: Path) -> None:
        if isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, path, f"{location}[{index}]", directory)
        elif isinstance(value, dict):
            for message in value.get("diagnostics", {}).get("warnings", []):
                add(path, f"{location}.diagnostics".lstrip("."), message)
            for entry in value.get("measurement_warnings", []):
                add(directory / entry["source"], entry["location"], entry["message"])
            if value.get("success") is False:
                add(path, location, "fit did not converge: " + str(value.get("message", "success=false")))
            if value.get("sampling", {}).get("stop_reason") == "max_meas_total" and "diagnostics" not in value:
                add(path, f"{location}.sampling".lstrip("."),
                    "maximum measurements reached before every primary observable met target ESS")
            if value.get("warmup", {}).get("stop_reason") == "max_warmup" and "diagnostics" not in value:
                add(path, f"{location}.warmup".lstrip("."),
                    "maximum warmup reached before autocorrelation times stabilized")
            if value.get("valid") is False and value.get("reason"):
                add(path, location, value["reason"])
            if "accept_rate" in value and "diagnostics" not in value:
                rate = value["accept_rate"]
                if isinstance(rate, (float, int)) and math.isfinite(rate) and not (
                        cfg["diagnostics"]["min_accept"] <= rate <= cfg["diagnostics"]["max_accept"]):
                    add(path, location, f"HMC acceptance {rate:.3f} outside configured range")
            if value.get("fine_path"):
                visit(directory / value["fine_path"], directory)
            for entry in value.get("measurements", {}).values():
                visit(directory / entry["path"], directory)
            for key, item in value.items():
                if key not in {"diagnostics", "measurement_warnings", "measurements", "preliminary", "trace"}:
                    walk(item, path, f"{location}.{key}".lstrip("."), directory)

    visited = set()

    def visit(path: Path, directory: Path, *, stage: bool = False) -> None:
        if path in visited or not path.is_file():
            return
        visited.add(path)
        value = read_json(path)
        if stage:
            value = dict(value)
            if "matches" in value:
                value["matches"] = {k: v for k, v in value["matches"].items()
                                    if k in summary["renormalization"]["beta1_methods"]}
                for method, match in value["matches"].items():
                    if match.get("bracketed") is False or match.get("correction_success") is False:
                        add(path, f"matches.{method}", match.get("reason", "scan did not bracket target"))
            if "fits" in value:
                kinds = {name.split("_mod")[0] for name in summary["branches"]}
                value["fits"] = {k: v for k, v in value["fits"].items() if k in kinds}
        walk(value, path, "", directory)

    common_dirs = ([root / "common" / f"mod{mod}" for mod in summary["model"]["mod"]]
                   if float(summary["model"]["beta1"]) != 0 else [root])
    for directory in common_dirs:
        for stage in ("pilot", "step1_1plaq", "step2_2plaq"):
            visit(directory / stage / "result.json", directory, stage=True)
    for branch in summary["branches"]:
        directory = root / "branches" / branch
        if summary["renormalization"]["beta1_methods"]:
            visit(directory / "step3_observable" / "result.json", directory, stage=True)
            unavailable = branch.startswith("U_mod") and float(summary["model"].get("alpha", 0)) == 0
            if summary["renormalization"]["alpha_topo_fit"] and not unavailable:
                visit(directory / "step4_topo" / "result.json", directory, stage=True)
    warnings.sort(key=lambda item: (item["source"], item["location"], item["message"]))
    return {"valid": not warnings, "warning_count": len(warnings), "warnings": warnings}


def refresh_summary(root: Path) -> dict:
    """Backfill an existing summary without running sampling or fitting."""
    summary = read_json(root / "summary.json")
    cfg = read_json(root / "resolved.json")["config"]
    summary["diagnostics"] = collect_diagnostics(root, summary, cfg)
    atomic_json(root / "summary.json", summary)
    return summary["diagnostics"]
