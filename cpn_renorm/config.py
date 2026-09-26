from __future__ import annotations

import hashlib
import json
import math
import tomllib
from copy import deepcopy
from pathlib import Path
from typing import Any


DEFAULTS: dict[str, Any] = {
    "runtime": {"device": "auto", "dtype": "float64", "seed": 1729},
    "pilot": {"chains": 16, "initial_L": 16, "max_L": 256,
              "min_L_over_xi": 8.0,
              "growth_factor": 2.0, "min_warmup": 300, "max_warmup": 4000,
              "min_meas_total": 4800, "target_ess": 200,
              "max_meas_total": 19200},
    "geometry": {"boundary_bc": "OBC", "padding": -1, "coarse_L": -1,
                 "padding_xi_mul": 3.0, "obs_L_xi_mul": 10.0},
    "hmc": {"epsilon": 0.05, "trajectory_length": 1.0, "mass_a": 1.0,
            "mass_z": 1.0, "target_accept": 0.75, "adapt_steps": 200,
            "s_step": 0.5, "s_updates": 1, "warmup_tau_multiplier": 10.0,
            "tau_stability_rtol": 0.25},
    "diagnostics": {"min_accept": 0.50, "max_accept": 0.95,
                    "strict": False},
    "scan": {"beta1": {"min": -8.0, "max": 8.0,
                         "growth_factor": 2.0, "max_rounds": 7},
             "alpha": {"min": 0.001, "max": 8.0,
                       "growth_factor": 2.0, "max_rounds": 7}},
    "steps": {
        "two_plaq": {"chains": 32, "fit_times": 5000,
                     "min_warmup": 400, "max_warmup": 4000,
                     "min_meas_per_boundary": 3000,
                     "target_ess_per_boundary": 500,
                     "max_meas_per_boundary": 12000, "bins": 60},
        "observable": {"chains": 16, "min_warmup": 500, "max_warmup": 6000,
                       "min_meas_total": 8000, "target_ess": 200,
                       "max_meas_total": 32000},
        "one_plaq": {"chains": 32, "fit_times": 2000,
                     "min_warmup": 400, "max_warmup": 4000,
                     "min_meas_per_boundary": 1000,
                     "target_ess_per_boundary": 500,
                     "max_meas_per_boundary": 4000, "zero_pad": 1},
        "topo": {"chains": 16, "min_warmup": 1000, "max_warmup": 12000,
                 "min_meas_total": 16000, "target_ess": 100,
                 "max_meas_total": 64000},
    },
}

OBSOLETE_FIELDS = {
    "bootstrap_samples": "remove it; xi uncertainty now uses the Gamma method",
    "n_meas": "use min_meas_total, target_ess, and max_meas_total",
    "n_meas_total": "rename it to min_meas_total",
    "meas_interval": "remove it; every HMC sweep is measured",
    "n_warmup": "use min_warmup and max_warmup",
    "n_leapfrog": "use hmc.trajectory_length",
}


def _reject_obsolete(raw: dict[str, Any], prefix: str = "") -> None:
    for key, value in raw.items():
        path = f"{prefix}.{key}" if prefix else key
        if key in OBSOLETE_FIELDS:
            raise ValueError(f"obsolete configuration field {path}: {OBSOLETE_FIELDS[key]}")
        if isinstance(value, dict):
            _reject_obsolete(value, path)


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        out[key] = (_merge(out[key], value)
                    if isinstance(value, dict) and isinstance(out.get(key), dict)
                    else value)
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open("rb") as stream:
        raw = tomllib.load(stream)
    _reject_obsolete(raw)
    cfg = _merge(DEFAULTS, raw)
    cfg["_config_path"] = str(path)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    _reject_obsolete(cfg)
    for section in ("model", "renormalization"):
        if section not in cfg:
            raise ValueError(f"missing TOML section [{section}]")
    model = cfg["model"]
    for key in ("N", "beta", "beta1", "alpha", "alpha1", "mod"):
        if key not in model:
            raise ValueError(f"model.{key} is required")
    if int(model["N"]) <= 1 or int(model["mod"]) not in (0, 1):
        raise ValueError("model.N must be > 1 and model.mod must be 0 or 1")
    if float(model["alpha"]) < 0:
        raise ValueError("model.alpha must be non-negative for the full Villain action")
    ren = cfg["renormalization"]
    if ren.get("type") not in ("U", "z") or int(ren.get("factor", 0)) < 1:
        raise ValueError("renormalization.type must be U or z and factor positive")
    if cfg["runtime"]["dtype"] not in ("float32", "float64"):
        raise ValueError("runtime.dtype must be float32 or float64")
    if "chains" in cfg["runtime"]:
        raise ValueError("runtime.chains is not used; set pilot.chains and steps.<name>.chains")
    for name, settings in cfg["steps"].items():
        duplicated_hmc = sorted(set(settings) & set(cfg["hmc"]))
        if duplicated_hmc:
            key = duplicated_hmc[0]
            raise ValueError(
                f"steps.{name}.{key} is not allowed; configure it once as hmc.{key}")
    if cfg["geometry"]["boundary_bc"] not in ("OBC", "PBC"):
        raise ValueError("geometry.boundary_bc must be OBC or PBC")
    for name in ("beta1", "alpha"):
        scan = cfg["scan"][name]
        if "min" not in scan or "max" not in scan or scan["min"] >= scan["max"]:
            raise ValueError(f"scan.{name}.min/max are required and must be ordered")
        points = scan.get("points", [])
        if points and (len(set(points)) < 2 or any(x < scan["min"] or x > scan["max"]
                                                   for x in points)):
            raise ValueError(f"scan.{name}.points need at least two unique values inside min/max")
    if cfg["scan"]["alpha"]["min"] < 0:
        raise ValueError("scan.alpha.min must be non-negative")
    pilot = cfg["pilot"]
    if pilot["initial_L"] > pilot["max_L"] or pilot["min_L_over_xi"] <= 0:
        raise ValueError("invalid pilot lattice limits")
    if int(pilot["chains"]) < 1:
        raise ValueError("pilot.chains must be positive")
    aggregate_sections = [("pilot", cfg["pilot"])] + [
        (name, cfg["steps"][name]) for name in ("observable", "topo")]
    for name, settings in aggregate_sections:
        prefix = name if name == "pilot" else f"steps.{name}"
        if int(settings["chains"]) < 1:
            raise ValueError(f"{prefix}.chains must be positive")
        if any(int(settings[key]) < 1 for key in
               ("min_warmup", "max_warmup", "min_meas_total", "max_meas_total")):
            raise ValueError(f"{prefix} sampling counts must be positive")
        if settings["min_warmup"] > settings["max_warmup"]:
            raise ValueError(f"{prefix}.min_warmup must not exceed max_warmup")
        if settings["min_meas_total"] > settings["max_meas_total"]:
            raise ValueError(f"{prefix}.min_meas_total must not exceed max_meas_total")
        if float(settings["target_ess"]) <= 0:
            raise ValueError(f"{prefix}.target_ess must be positive")
    for name in ("two_plaq", "one_plaq"):
        settings = cfg["steps"][name]
        prefix = f"steps.{name}"
        legacy_totals = [key for key in ("min_meas_total", "target_ess", "max_meas_total")
                         if key in settings]
        if legacy_totals:
            raise ValueError(
                f"{prefix}.{legacy_totals[0]} is not valid for frozen-boundary fits; "
                "use the corresponding *_per_boundary field")
        if int(settings["chains"]) < 1 or int(settings["fit_times"]) < 1:
            raise ValueError(f"{prefix}.chains and fit_times must be positive")
        if any(int(settings[key]) < 1 for key in
               ("min_warmup", "max_warmup", "min_meas_per_boundary",
                "max_meas_per_boundary")):
            raise ValueError(f"{prefix} sampling counts must be positive")
        if settings["min_warmup"] > settings["max_warmup"]:
            raise ValueError(f"{prefix}.min_warmup must not exceed max_warmup")
        if settings["min_meas_per_boundary"] > settings["max_meas_per_boundary"]:
            raise ValueError(
                f"{prefix}.min_meas_per_boundary must not exceed max_meas_per_boundary")
        if float(settings["target_ess_per_boundary"]) <= 0:
            raise ValueError(f"{prefix}.target_ess_per_boundary must be positive")
        if float(settings["target_ess_per_boundary"]) > int(
                settings["max_meas_per_boundary"]):
            raise ValueError(f"{prefix}.target_ess_per_boundary cannot exceed the sample cap")
    geometry = cfg["geometry"]
    if int(geometry["padding"]) < -1 or (int(geometry["coarse_L"]) != -1 and
                                          int(geometry["coarse_L"]) < 2):
        raise ValueError("geometry.padding/coarse_L must be -1 or a valid override")
    if geometry["boundary_bc"] == "PBC" and int(geometry["padding"]) == 0:
        raise ValueError("geometry.padding must be -1 or at least 1 for PBC patches")
    if geometry["padding_xi_mul"] <= 0 or geometry["obs_L_xi_mul"] <= 0:
        raise ValueError("geometry xi multipliers must be positive")
    hmc = cfg["hmc"]
    if any(hmc[key] <= 0 for key in ("epsilon", "trajectory_length", "mass_a", "mass_z",
                                     "warmup_tau_multiplier")):
        raise ValueError("HMC epsilon, trajectory length, masses, and warmup multiplier must be positive")
    diagnostic = cfg["diagnostics"]
    if not 0 <= diagnostic["min_accept"] < diagnostic["max_accept"] <= 1:
        raise ValueError("diagnostics acceptance range must lie inside [0,1]")


def resolved_geometry(cfg: dict[str, Any], xi: float) -> dict[str, int | float]:
    factor = int(cfg["renormalization"]["factor"])
    requested_padding = int(cfg["geometry"]["padding"])
    requested_coarse = int(cfg["geometry"]["coarse_L"])
    padding = (requested_padding if requested_padding >= 0 else
               int(math.ceil(cfg["geometry"]["padding_xi_mul"] * xi)))
    coarse = (requested_coarse if requested_coarse > 0 else
              max(2, int(math.ceil(cfg["geometry"]["obs_L_xi_mul"] * xi / factor))))
    fine = factor * coarse
    return {"xi": float(xi), "padding": padding, "L_fine": fine,
            "L_coarse": fine // factor, "factor": factor}


def fingerprint(payload: dict[str, Any]) -> str:
    clean = {k: v for k, v in payload.items() if not k.startswith("_")}
    blob = json.dumps(clean, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:20]


def section(cfg: dict[str, Any], step: str) -> dict[str, Any]:
    """Return shared runtime/HMC settings plus one step's sampling budget."""
    return _merge(_merge(cfg["runtime"], cfg["hmc"]), cfg["steps"][step])
