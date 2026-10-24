"""Generate a fine-lattice configuration from a completed renormalization."""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any

import tomli_w

from .config import DEFAULTS, _merge, load_config, validate_config
from .methods import BETA1_METHODS


def resolve_run_dir(source: str | Path, runs_root: Path) -> Path:
    """Resolve explicit paths, or a unique bare directory name under runs_root."""
    source = Path(source)
    runs_root = runs_root.resolve()
    if not source.is_absolute() and len(source.parts) == 1:
        matches = sorted({path.resolve() for path in runs_root.rglob(source.name)
                          if path.is_dir() and path.name == source.name
                          and (path / "summary.json").is_file()})
        if len(matches) > 1:
            raise ValueError("ambiguous source run; specify a path:\n" +
                             "\n".join(f"  {path}" for path in matches))
        if matches:
            return matches[0]
        raise FileNotFoundError(f"source run {str(source)!r} not found under {runs_root}")
    path = source if source.is_absolute() else runs_root.parent / source
    path = path.resolve()
    if not (path / "summary.json").is_file():
        raise FileNotFoundError(f"source summary not found: {path / 'summary.json'}")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _apply_overrides(cfg: dict[str, Any], overrides: dict[str, Any]) -> None:
    protected = {"renormalization.type", "renormalization.factor",
                 "renormalization.beta1_methods", "renormalization.alpha_topo_fit"}
    for name, value in overrides.items():
        parts = name.split(".")
        if parts[0] == "model" or name in protected:
            raise ValueError(f"override {name!r} is controlled by the dedicated parameters")
        target = cfg
        for part in parts[:-1]:
            if part.startswith("_") or not isinstance(target.get(part), dict):
                raise ValueError(f"unknown configuration field {name!r}")
            target = target[part]
        if (parts[-1].startswith("_") or parts[-1] not in target or
                isinstance(target[parts[-1]], dict)):
            raise ValueError(f"override must name an existing configuration value: {name!r}")
        target[parts[-1]] = deepcopy(value)


def generate_next_config(
    source: str | Path, *, mod: int, source_type: str,
    beta_method: str, alpha_method: str, factor: int, output_dir: str | Path,
    project_root: str | Path, filename: str | None = None,
    next_type: str = "U", overwrite: bool = False,
    overrides: dict[str, Any] | None = None,
) -> Path:
    """Write a TOML in output_dir (relative to configs/), optionally naming it."""
    project_root = Path(project_root).resolve()
    if type(mod) is not int or mod not in (0, 1):
        raise ValueError("mod must be 0 or 1")
    if source_type not in ("U", "z") or next_type not in ("U", "z"):
        raise ValueError("source_type and next_type must be U or z")
    if beta_method not in BETA1_METHODS or alpha_method not in ("1plaq", "topo"):
        raise ValueError("beta_method must be magsus or corrG; alpha_method must be 1plaq or topo")
    if type(factor) is not int or factor < 1:
        raise ValueError("factor must be a positive integer")
    run_dir = resolve_run_dir(source, project_root / "runs")
    directory = Path(output_dir)
    if not directory.is_absolute():
        directory = project_root / "configs" / directory
    if filename is None or filename == "":
        filename = f"{run_dir.name}-L{factor}_mod{mod}_{beta_method}_{alpha_method}"
    if (not isinstance(filename, str) or filename in (".", "..") or
            not filename.strip() or "/" in filename or "\\" in filename):
        raise ValueError("filename must be a file name without a directory path")
    if not filename.endswith(".toml"):
        filename += ".toml"
    output = (directory / filename).resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output}; set OVERWRITE = True to replace it")
    if output in (run_dir / "input.toml",):
        raise ValueError("output must not replace the source run's input.toml")

    summary = _read_json(run_dir / "summary.json")
    branch_name = f"{source_type}_mod{mod}"
    branches = summary.get("branches", {})
    if branch_name not in branches:
        raise ValueError(f"source branch {branch_name!r} not found; available: {', '.join(branches)}")
    result_name = f"{beta_method}_{'obs_fit_topo' if alpha_method == 'topo' else '1plaq'}"
    results = branches[branch_name].get("results", {})
    if result_name not in results:
        raise ValueError(f"source result {result_name!r} not found in {branch_name}; "
                         f"available: {', '.join(results)}")
    couplings = results[result_name].get("couplings", {})
    model = {"N": summary.get("model", {}).get("N")}
    if type(model["N"]) is not int or model["N"] <= 1:
        raise ValueError("source summary must contain an integer model.N > 1")
    for key in ("beta", "beta1", "alpha", "alpha1"):
        value = couplings.get(key)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"source {branch_name}/{result_name} has invalid coupling {key}: {value!r}")
        model[key] = value
    model["mod"] = [mod]

    resolved_path = run_dir / "resolved.json"
    inherited = _read_json(resolved_path).get("config") if resolved_path.exists() else None
    if inherited is None:
        cfg = load_config(run_dir / "input.toml")
    elif isinstance(inherited, dict):
        cfg = _merge(DEFAULTS, inherited)
    else:
        raise ValueError(f"invalid config in {resolved_path}")
    cfg = {key: value for key, value in cfg.items() if not key.startswith("_")}
    cfg["model"] = model
    cfg["renormalization"].update(type=[next_type], factor=factor,
                                beta1_methods=[beta_method], alpha_topo_fit=alpha_method == "topo")
    _apply_overrides(cfg, overrides or {})
    validate_config(cfg)
    header = (f"# Generated from: {str(run_dir)!r}\n"
              f"# Source branch: {branch_name}; result: {result_name}\n\n")
    text = header + tomli_w.dumps(cfg)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w" if overwrite else "x", encoding="utf-8") as stream:
        stream.write(text)
    print(f"Generated: {output}\nSource: {run_dir} / {branch_name} / {result_name}")
    print("Model: " + json.dumps(model, ensure_ascii=False))
    return output
