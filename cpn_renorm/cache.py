"""JSON-only measurement caches and shared simulation contexts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import fingerprint
from .storage import atomic_json, atomic_npz, split_measurement

POINT_FIELDS = ("L", "beta", "beta1", "alpha", "alpha1", "kind")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_root(directory: Path) -> Path:
    for parent in (directory, *directory.parents):
        if parent.name == "branches":
            return parent.parent
    return directory


def point_identity(payload: dict) -> tuple[str, dict, dict]:
    parameters = {key: payload[key] for key in POINT_FIELDS}
    context = {key: value for key, value in payload.items()
               if key not in POINT_FIELDS}
    # Matching methods may use identical physical ensembles.
    context["tag"] = "topo_coarse" if context["tag"].startswith("topo_coarse_") else context["tag"]
    return fingerprint({"parameters": parameters, "context": context}), parameters, context


def store_point(cfg: dict, directory: Path, payload: dict, result: dict) -> Path:
    key, parameters, context = point_identity(payload)
    root = run_root(directory)
    resolved_path = root / "resolved.json"
    resolved = read_json(resolved_path) if resolved_path.exists() else {}
    context_key = fingerprint(context)
    resolved.setdefault("contexts", {})[context_key] = context
    atomic_json(resolved_path, resolved)
    folder = directory / "ensembles" / f"{context['tag']}_{key}"
    metadata, arrays = split_measurement(result)
    for name in ("accept_rate", "s_accept_rate"):
        if name in arrays and arrays[name].size:
            metadata[name] = float(np.mean(arrays[name]))
    metadata = {k: v for k, v in metadata.items() if not k.startswith("_")
                and k not in ("parameters", "fingerprint", "cache", "context", "phases")}
    atomic_json(folder / "result.json", {"parameters": parameters,
                                         "context": context_key, **metadata})
    if cfg.get("storage", {}).get("save_npz", False) and arrays:
        atomic_npz(folder / "data.npz", **arrays)
    return folder


def load_point(folder: Path) -> dict:
    result = read_json(folder / "result.json")
    result["_folder"] = str(folder)
    result["cache"] = "reused"
    return result


def scan_measurements(directory: Path, stage: dict) -> tuple[dict[float, dict], dict[float, set[str]]]:
    measured, phases = {}, {}
    for value, entry in stage.get("measurements", {}).items():
        path = directory / entry["path"]
        if path.exists():
            measured[float(value)] = load_point(path.parent)
            phases[float(value)] = set(entry.get("phases", []))
    return measured, phases


def write_scan_points(directory: Path, measured: dict[float, dict],
                      phases: dict[float, set[str]], *, tag: str) -> dict[str, Any]:
    entries = {}
    for value, result in measured.items():
        folder = Path(result["_folder"]) if result.get("_folder") else directory / "ensembles" / f"{tag}_{fingerprint({"value": value})}"
        if not (folder / "result.json").exists():
            atomic_json(folder / "result.json", {k: v for k, v in result.items()
                                                if not k.startswith("_")})
        entries[str(value)] = {"path": str((folder / "result.json").relative_to(directory)),
                               "phases": sorted(phases.get(value, {"historical"}))}
    return entries
