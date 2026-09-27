from __future__ import annotations

import json
import math
import shutil
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import fingerprint, section
from .fit_client import extend_scan, isolated_fit, isolated_match, isolated_plot
from .observables import measure_ensemble
from .patches import (PATCH_CACHE_SCHEMA, generate_one_plaq, generate_two_plaq,
                      patch_result_key)
from .pilot import resolve_device, run_pilot, torch_dtype
from .progress import SimulationProgress, ensemble_summary
from .sampler import BatchedHMCSampler, Couplings
from .storage import atomic_copy, atomic_json, atomic_npz, split_measurement
from .workspace import config_fingerprint, workspace_dir


def _log(message: str) -> None:
    print(f"[cpn-renorm] {message}", flush=True)


def _point_sampler(cfg: dict, settings: dict, *, L: int, beta: float, beta1: float,
                   alpha: float, alpha1: float, kind: str, seed: int) -> BatchedHMCSampler:
    rt, m = cfg["runtime"], cfg["model"]
    return BatchedHMCSampler(
        chains=int(settings["chains"]), Lx=L, Ly=L, N=int(m["N"]),
        couplings=Couplings(beta, beta1, alpha, alpha1, int(m["mod"])),
        kind=kind, periodic=True, device=resolve_device(rt["device"]),
        dtype=torch_dtype(rt["dtype"]), seed=seed,
        epsilon=float(settings["epsilon"]),
        trajectory_length=float(settings["trajectory_length"]),
        mass_a=float(settings["mass_a"]), mass_z=float(settings["mass_z"]))


def _apply_point_diagnostics(cfg: dict, tag: str, result: dict) -> None:
    accept_mean = float(np.mean(result["accept_rate"]))
    diag_cfg = cfg["diagnostics"]
    warnings = []
    if not diag_cfg["min_accept"] <= accept_mean <= diag_cfg["max_accept"]:
        cause = ("integration error is likely too large"
                 if accept_mean < diag_cfg["min_accept"] else
                 "epsilon is likely too small and leapfrog cost excessive")
        warnings.append(
            f"HMC acceptance {accept_mean:.3f} outside configured range; {cause}")
    if result["sampling"]["stop_reason"] == "max_meas_total":
        warnings.append(
            "maximum measurements reached before every primary observable met target ESS")
    if result["warmup"]["stop_reason"] == "max_warmup":
        warnings.append("maximum warmup reached before autocorrelation times stabilized")
    result["diagnostics"] = {"valid": not warnings, "warnings": warnings}
    if warnings and diag_cfg.get("strict"):
        raise RuntimeError(f"{tag} diagnostics failed: {'; '.join(warnings)}")


def _measure_point(cfg: dict, run_dir: Path, settings: dict, *, tag: str, L: int,
                   beta: float, beta1: float, alpha: float, alpha1: float,
                   kind: str) -> dict:
    payload = {"schema": 3, "tag": tag, "L": L, "beta": beta, "beta1": beta1,
               "alpha": alpha, "alpha1": alpha1, "kind": kind,
               "N": cfg["model"]["N"], "mod": cfg["model"]["mod"],
               "settings": settings, "runtime": cfg["runtime"]}
    key = fingerprint(payload)
    folder = run_dir / "ensembles" / f"{tag}_{key}"
    meta_file, data_file = folder / "result.json", folder / "data.npz"
    if meta_file.exists() and data_file.exists() and not cfg.get("_force"):
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        with np.load(data_file) as data:
            meta.update({name: data[name] for name in data.files})
        meta["cache"] = "reused"
        meta["_folder"] = str(folder)
        _apply_point_diagnostics(cfg, tag, meta)
        _log(f"reuse {tag}: L={L} beta1={beta1:.6g} alpha={alpha:.6g}")
        return meta
    _log(f"simulate {tag}: L={L} beta1={beta1:.6g} alpha={alpha:.6g} kind={kind}")
    seed = int(cfg["runtime"]["seed"]) + int(key[:8], 16) % 1_000_000
    sampler = _point_sampler(cfg, settings, L=L, beta=beta, beta1=beta1,
                             alpha=alpha, alpha1=alpha1, kind=kind, seed=seed)
    label = f"{tag} L={L} beta1={beta1:.6g} alpha={alpha:.6g}"
    with SimulationProgress(label) as progress:
        result = measure_ensemble(
            sampler, min_warmup=int(settings["min_warmup"]),
            max_warmup=int(settings["max_warmup"]),
            min_meas_total=int(settings["min_meas_total"]),
            target_ess=float(settings["target_ess"]),
            max_meas_total=int(settings["max_meas_total"]),
            adapt_steps=int(settings["adapt_steps"]),
            target_accept=float(settings["target_accept"]),
            warmup_tau_multiplier=float(settings["warmup_tau_multiplier"]),
            tau_stability_rtol=float(settings["tau_stability_rtol"]),
            probe_kind=("topology" if tag.startswith("topo") else "observable"),
            s_step=float(settings["s_step"]), s_updates=int(settings["s_updates"]),
            progress=progress)
    _apply_point_diagnostics(cfg, tag, result)
    meta, arrays = split_measurement(result)
    meta.update({"fingerprint": key, "parameters": payload, "cache": "generated"})
    atomic_npz(data_file, **arrays)
    atomic_json(meta_file, meta)
    _log(ensemble_summary(label, result))
    result.update({"fingerprint": key, "cache": "generated", "_folder": str(folder)})
    return result


def _scan_points(scan: dict) -> list[float]:
    if scan.get("points"):
        return sorted(set(float(x) for x in scan["points"]))
    lo, hi = float(scan["min"]), float(scan["max"])
    points = [x for x in (-0.1, 0.0, 0.1) if lo <= x <= hi]
    return points if len(points) >= 2 else [lo, 0.5 * (lo + hi), hi]


def _scan_fingerprint(kind: str, cfg: dict, geometry: dict, **upstream: float) -> str:
    step = "observable" if kind == "beta1" else "topo"
    return fingerprint({"schema": 2, "kind": kind, "geometry": geometry,
                        "scan": cfg["scan"][kind], "settings": section(cfg, step),
                        "model": cfg["model"], "renormalization": cfg["renormalization"],
                        "upstream": upstream})


def _same_point(value: float, candidates: set[float]) -> bool:
    return any(math.isclose(value, item, rel_tol=0.0, abs_tol=1e-12)
               for item in candidates)


def _cleanup_candidates(measured: dict[float, dict], used: set[float]) -> list[str]:
    return [str(result["_folder"]) for value, result in measured.items()
            if not _same_point(value, used) and result.get("_folder")]


def _remove_scan_ensembles(run_dir: Path, folders: list[str]) -> list[str]:
    ensembles = (run_dir / "ensembles").resolve()
    removed = []
    for raw in sorted(set(folders)):
        folder = Path(raw).resolve()
        try:
            folder.relative_to(ensembles)
        except ValueError as exc:
            raise RuntimeError(f"refusing to clean ensemble outside {ensembles}: {folder}") from exc
        if folder.exists():
            shutil.rmtree(folder)
            removed.append(folder.name)
    return removed


def _censored_xi_bracket(xs: np.ndarray, ys: np.ndarray,
                         target: float) -> list[float] | None:
    """Bracket xi onset when an undefined lower point precedes xi > target."""
    for high_idx, (x_high, y_high) in enumerate(zip(xs, ys)):
        if not np.isfinite(y_high) or y_high <= target:
            continue
        lower_null = [float(xs[idx]) for idx in range(high_idx)
                      if not np.isfinite(ys[idx])]
        finite_below = [idx for idx in range(high_idx)
                        if np.isfinite(ys[idx]) and ys[idx] <= target]
        if finite_below:
            return None
        if lower_null:
            return [max(lower_null), float(x_high)]
    return None


def _raw_scan_match(xs: np.ndarray, ys: np.ndarray, target: float) -> dict:
    """Cheap finite-point bracket state used while simulations are still growing."""
    finite = np.isfinite(xs) & np.isfinite(ys)
    x, y = np.asarray(xs)[finite], np.asarray(ys)[finite]
    if len(x) < 2:
        return {"value": None, "bracketed": False, "direction": "low",
                "raw_bracket": None}
    crossings = np.flatnonzero((y[:-1] - target) * (y[1:] - target) <= 0)
    if len(crossings):
        anchors = []
        for idx in crossings:
            dy = y[idx + 1] - y[idx]
            root = (0.5 * (x[idx] + x[idx + 1]) if abs(dy) < 1e-15 else
                    x[idx] + (target - y[idx]) * (x[idx + 1] - x[idx]) / dy)
            anchors.append((abs(float(root) - float(x[np.argmin(abs(y - target))])), idx,
                            float(root)))
        _, idx, root = min(anchors)
        return {"value": root, "bracketed": True,
                "raw_bracket": [float(x[idx]), float(x[idx + 1])]}
    direction = "low" if abs(y[0] - target) < abs(y[-1] - target) else "high"
    return {"value": None, "bracketed": False, "direction": direction,
            "raw_bracket": None, "range": [float(np.min(y)), float(np.max(y))]}


def _scan_decision(match: dict, *, tolerance: float, max_refine: int,
                   refine_count: int, censored: list[float] | None = None) -> tuple[float | None, str]:
    bracket = match.get("raw_bracket") if match.get("bracketed") else censored
    if bracket is None:
        return None, "unbracketed"
    width = float(bracket[1] - bracket[0])
    # A censored xi bracket still lacks a finite point below the target. Keep
    # bisecting it even below the ordinary localization tolerance.
    if match.get("bracketed") and width <= tolerance:
        return None, "tolerance"
    if refine_count >= max_refine:
        return None, "max_refine_rounds"
    return 0.5 * float(bracket[0] + bracket[1]), "refine"


def _dynamic_fit_spacing(error: float | None, scan: dict) -> float:
    spacing = float(scan["fit_spacing_min"])
    if error is not None and math.isfinite(float(error)) and float(error) > 0:
        spacing = float(scan["fit_sigma_multiplier"]) * float(error)
    spacing = max(float(scan["fit_spacing_min"]), spacing)
    return min(float(scan["fit_spacing_max"]), spacing)


def _uniform_fit_grid(center: float, spacing: float, scan: dict) -> list[float]:
    count = int(scan["fit_points"])
    lo, hi = float(scan["min"]), float(scan["max"])
    spacing = min(float(spacing), (hi - lo) / (count - 1))
    half = count // 2
    start = float(center) - half * spacing
    if start < lo:
        start = lo
    if start + (count - 1) * spacing > hi:
        start = hi - (count - 1) * spacing
    return [start + index * spacing for index in range(count)]


def _grid_significantly_brackets(values: np.ndarray, errors: np.ndarray,
                                  target: float, target_err: float,
                                  increasing: bool, confidence: float) -> tuple[bool, str]:
    values, errors = np.asarray(values, float), np.asarray(errors, float)
    if not np.isfinite(values).all() or not np.isfinite(errors).all():
        return False, "nonfinite_grid"
    middle = len(values) // 2
    left, right = middle - 1, middle + 1
    left_sigma = math.hypot(float(errors[left]), float(target_err))
    right_sigma = math.hypot(float(errors[right]), float(target_err))
    if increasing:
        separated = (values[left] + confidence * left_sigma < target and
                     values[right] - confidence * right_sigma > target)
    else:
        separated = (values[left] - confidence * left_sigma > target and
                     values[right] + confidence * right_sigma < target)
    return (True, "significant_bracket") if separated else (False, "insufficient_separation")


def _uniform_grid_match(*, measured: dict[float, dict], measure: Callable[[float], dict],
                        observable: Callable[[dict], tuple[float, float]],
                        target: float, target_err: float, increasing: bool,
                        scan: dict, bootstrap_seed: int) -> dict:
    """Build and fit a statistically spaced uniform grid around a preliminary root."""
    def lookup(value: float) -> dict:
        for existing, result in measured.items():
            if math.isclose(float(existing), float(value), rel_tol=0.0, abs_tol=1e-12):
                return result
        raise KeyError(value)

    def arrays(points: list[float] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        xs = np.array(sorted(measured) if points is None else points, dtype=float)
        pairs = [observable(lookup(float(value))) for value in xs]
        return xs, np.array([item[0] for item in pairs]), np.array([item[1] for item in pairs])

    all_x, all_y, all_err = arrays()
    preliminary = isolated_match(
        all_x, all_y, target, all_err, target_err, increasing=increasing,
        max_points=int(scan["fit_points"]), bootstrap_seed=bootstrap_seed)
    if not preliminary.get("bracketed") or preliminary.get("value") is None:
        return {**preliminary, "preliminary": preliminary, "fit_grid": [],
                "fit_spacing": None, "fit_grid_rounds": 0,
                "reason": preliminary.get("reason", "preliminary fit was not bracketed")}
    center = float(preliminary["value"])
    spacing = _dynamic_fit_spacing(preliminary.get("error"), scan)
    confidence = float(scan["fit_confidence_sigma"])
    last_grid: list[float] = []
    last_reason = "fit grid was not attempted"
    for grid_round in range(int(scan["max_fit_grid_rounds"]) + 1):
        grid = _uniform_fit_grid(center, spacing, scan)
        last_grid = grid
        for value in grid:
            if not _same_point(value, set(measured)):
                measured[value] = measure(value)
        grid_x, grid_y, grid_err = arrays(grid)
        accepted, last_reason = _grid_significantly_brackets(
            grid_y, grid_err, target, target_err, increasing, confidence)
        if accepted:
            final = isolated_match(
                grid_x, grid_y, target, grid_err, target_err,
                increasing=increasing, max_points=len(grid),
                bootstrap_seed=bootstrap_seed + 1000 + grid_round)
            if final.get("bracketed"):
                return {**final, "preliminary": preliminary,
                        "fit_grid": grid, "fit_spacing": spacing,
                        "fit_grid_rounds": grid_round,
                        "grid_validation": last_reason}

        all_x, all_y, all_err = arrays()
        updated = isolated_match(
            all_x, all_y, target, all_err, target_err, increasing=increasing,
            max_points=int(scan["fit_points"]),
            bootstrap_seed=bootstrap_seed + grid_round + 1)
        if updated.get("bracketed") and updated.get("value") is not None:
            center = float(updated["value"])
            proposed = _dynamic_fit_spacing(updated.get("error"), scan)
            if last_reason == "insufficient_separation":
                proposed = max(proposed, min(float(scan["fit_spacing_max"]),
                                             spacing * 1.5))
            else:
                proposed = min(proposed, spacing)
            spacing = max(float(scan["fit_spacing_min"]), proposed)
            if last_reason == "nonfinite_grid":
                finite_x = all_x[np.isfinite(all_y) & np.isfinite(all_err)]
                half = int(scan["fit_points"]) // 2
                if len(finite_x) and not np.isfinite(grid_y[0]):
                    center = max(center, float(np.min(finite_x)) + half * spacing)
                if len(finite_x) and not np.isfinite(grid_y[-1]):
                    center = min(center, float(np.max(finite_x)) - half * spacing)
        else:
            raw = _raw_scan_match(all_x, all_y, target)
            direction = raw.get("direction")
            center += spacing if direction == "high" else -spacing
    return {"value": None, "error": None, "bracketed": False,
            "reason": f"uniform fit grid failed: {last_reason}",
            "preliminary": preliminary, "fit_grid": last_grid,
            "fit_points": [], "fit_spacing": spacing,
            "fit_grid_rounds": int(scan["max_fit_grid_rounds"]),
            "grid_validation": last_reason}


def _run_two_plaq(cfg: dict, run_dir: Path, geometry: dict) -> dict:
    result_path = run_dir / "step1_2plaq" / "result.json"
    result_key = patch_result_key(cfg, geometry, "two_plaq")
    if result_path.exists() and not cfg.get("_force"):
        cached = json.loads(result_path.read_text(encoding="utf-8"))
        if (cached.get("cache_schema") == PATCH_CACHE_SCHEMA and
                cached.get("fingerprint") == result_key):
            _log("reuse completed step 1 (2plaq)")
            return cached
    _log("step 1/4: 2plaq fit")
    batch_dir = run_dir / "step1_2plaq" / "batches"
    raw = generate_two_plaq(cfg, geometry, batch_dir)
    tag = cfg["renormalization"]["type"]
    fit = isolated_fit("two_plaq", {"N": int(cfg["model"]["N"]),
                       "p0": cfg["steps"]["two_plaq"].get("p0", [0.5, 0.5])},
                       X=raw[f"X_{tag}"], freq=raw[f"freq_{tag}"])
    atomic_npz(run_dir / "step1_2plaq" / "measurements.npz", **raw)
    stop_reasons = np.asarray(raw["sampling_stop_reason"]).astype(str)
    warmup_reasons = np.asarray(raw["warmup_stop_reason"]).astype(str)
    outer_warmup_reasons = np.asarray(raw["outer_warmup_stop_reason"]).astype(str)
    warnings = ([] if not np.any(stop_reasons == "max_meas_per_boundary") else
                ["one or more batches reached maximum per-boundary measurements before target ESS"])
    if np.any(warmup_reasons == "max_warmup"):
        warnings.append("one or more inner patches reached maximum warmup")
    if np.any(outer_warmup_reasons == "max_warmup"):
        warnings.append("one or more outer boundary lattices reached maximum warmup")
    patch_accept = float(np.mean(raw["accept_rate"]))
    if not cfg["diagnostics"]["min_accept"] <= patch_accept <= cfg["diagnostics"]["max_accept"]:
        warnings.append(f"HMC acceptance {patch_accept:.3f} outside configured range")
    if warnings and cfg["diagnostics"].get("strict"):
        raise RuntimeError("two_plaq diagnostics failed: " + "; ".join(warnings))
    result = {"step": "2plaq", "cache_schema": PATCH_CACHE_SCHEMA,
              "fingerprint": result_key, **fit,
              "accept_rate": {"mean": float(np.mean(raw["accept_rate"])),
                              "std": float(np.std(raw["accept_rate"]))},
              "fit_times": {"requested": int(raw["requested_fit_times"]),
                            "effective": int(raw["effective_fit_times"]),
                            "chains": int(raw["chains"]),
                            "batches": int(raw["batch_count"])},
              "epsilon": {"mean": float(np.mean(raw["epsilon"])),
                          "per_batch": np.asarray(raw["epsilon"])},
              "warmup": {"outer_sweeps": np.asarray(raw["outer_warmup_sweeps"]),
                         "outer_tau": np.asarray(raw["outer_warmup_tau"]),
                         "outer_stop_reason": outer_warmup_reasons,
                         "inner_sweeps": np.asarray(raw["warmup_sweeps"]),
                         "inner_tau": np.asarray(raw["warmup_tau"]),
                         "inner_stop_reason": warmup_reasons,
                         "inner_initialization": np.asarray(raw["inner_initialization"]),
                         "inner_adapt_steps": np.asarray(raw["inner_adapt_steps"])},
              "sampling": {"minimum_per_boundary": int(raw["sampling_minimum"]),
                           "actual_per_boundary": np.asarray(raw["sampling_actual"]),
                           "total_actual": int(raw["sampling_total_actual"]),
                           "maximum_per_boundary": int(raw["sampling_maximum"]),
                           "target_ess_per_boundary": float(raw["sampling_target_ess"]),
                           "tau": np.asarray(raw["sampling_tau"]),
                           "ess": np.asarray(raw["sampling_ess"]),
                           "stop_reason": stop_reasons},
              "diagnostics": {"valid": not warnings, "warnings": warnings}}
    atomic_json(result_path, result)
    return result


def _plot_beta1_scan(run_dir: Path, scale_factor: int, fine: dict,
                     measured: dict[float, dict], matches: dict[str, dict]) -> None:
    """Plot magnetic susceptibility and correlation length on separate axes."""
    xs = np.array(sorted(measured))
    isolated_plot(
        run_dir / "step2_observable" / "beta1_scan_magsus.png", x=xs,
        series={"magsus_scaled": np.array([
            measured[x]["magsus"] * scale_factor ** 2 for x in xs])},
        errors={"magsus_scaled": np.array([
            measured[x]["magsus_err"] * scale_factor ** 2 for x in xs])},
        matches={"magsus_scaled": matches["magsus"]},
        targets={"fine_magsus": fine["magsus"]},
        colors={"magsus_scaled": "tab:blue", "fine_magsus": "tab:blue"},
        xlabel="beta1", ylabel="scaled magnetic susceptibility")
    isolated_plot(
        run_dir / "step2_observable" / "beta1_scan_corr.png", x=xs,
        series={"xi_scaled": np.array([
            measured[x]["xi"] * scale_factor
            if measured[x].get("xi") is not None else np.nan for x in xs])},
        errors={"xi_scaled": np.array([
            measured[x]["xi_err"] * scale_factor
            if measured[x].get("xi_err") is not None else np.nan for x in xs])},
        matches={"xi_scaled": matches["corr"]},
        targets={"fine_xi": fine["xi"]},
        colors={"xi_scaled": "tab:orange", "fine_xi": "tab:orange"},
        xlabel="beta1", ylabel="scaled correlation length")
    old_plot = run_dir / "step2_observable" / "beta1_scan.png"
    if old_plot.exists():
        old_plot.unlink()


def _run_beta1_scan(cfg: dict, run_dir: Path, geometry: dict,
                    beta_c: float, alpha_eff_c: float) -> dict:
    _log("step 2/4: observable beta1 scan")
    settings, scan = section(cfg, "observable"), cfg["scan"]["beta1"]
    m, R, Lf, Lc = cfg["model"], geometry["factor"], geometry["L_fine"], geometry["L_coarse"]
    result_path = run_dir / "step2_observable" / "result.json"
    result_key = _scan_fingerprint("beta1", cfg, geometry, beta_c=beta_c,
                                   alpha_eff_c=alpha_eff_c)
    if result_path.exists() and not cfg.get("_force"):
        cached = json.loads(result_path.read_text(encoding="utf-8"))
        if cached.get("fingerprint") == result_key:
            _log("reuse completed step 2 (observable beta1 scan)")
            cached_measured = {
                float(value): observation
                for value, observation in cached["coarse"].items()
            }
            _plot_beta1_scan(run_dir, R, cached["fine"], cached_measured,
                             cached["matches"])
            return cached
    alpha_eff_f = float(m["alpha"] + m["alpha1"])
    fine = _measure_point(cfg, run_dir, settings, tag="obs_fine", L=Lf,
                          beta=float(m["beta"]), beta1=float(m["beta1"]),
                          alpha=alpha_eff_f, alpha1=0.0, kind="half")
    points, measured = _scan_points(scan), {}
    matches: dict[str, dict] = {}
    expansion = {"magsus": 0, "corr": 0}
    refinement = {"magsus": 0, "corr": 0}
    stop_reason = {"magsus": "unbracketed", "corr": "unbracketed"}
    max_iterations = 2 * int(scan["max_rounds"]) + 2 * int(scan["max_refine_rounds"]) + 8
    for _ in range(max_iterations):
        for value in points:
            if value not in measured:
                measured[value] = _measure_point(
                    cfg, run_dir, settings, tag="obs_coarse", L=Lc, beta=beta_c,
                    beta1=value, alpha=alpha_eff_c, alpha1=0.0, kind="half")
        xs = np.array(sorted(measured))
        ordered = [measured[x] for x in xs]
        magsus = np.array([r["magsus"] * R ** 2 for r in ordered])
        xis = np.array([(float(r["xi"]) * R if r.get("xi") is not None else np.nan)
                        for r in ordered])
        magsus_err = np.array([r["magsus_err"] * R ** 2 for r in ordered])
        xi_errs = np.array([(float(r["xi_err"]) * R
                             if r.get("xi_err") is not None else np.nan)
                            for r in ordered])
        matches = {"magsus": _raw_scan_match(xs, magsus, float(fine["magsus"])),
                   "corr": _raw_scan_match(xs, xis, float(fine["xi"]))}
        censored = _censored_xi_bracket(xs, xis, float(fine["xi"]))
        if censored is not None and not matches["corr"].get("bracketed"):
            matches["corr"]["censored_bracket"] = censored
        additions: set[float] = set()
        for method in ("magsus", "corr"):
            candidate, reason = _scan_decision(
                matches[method], tolerance=float(scan["refine_tolerance"]),
                max_refine=int(scan["max_refine_rounds"]),
                refine_count=refinement[method],
                censored=(censored if method == "corr" else None))
            if candidate is not None and candidate not in measured:
                additions.add(candidate)
                refinement[method] += 1
                stop_reason[method] = "refining"
                continue
            if reason != "unbracketed":
                stop_reason[method] = reason
                continue
            if expansion[method] >= int(scan["max_rounds"]):
                stop_reason[method] = "max_rounds"
                continue
            expanded = extend_scan(sorted(measured), matches[method],
                                   (float(scan["min"]), float(scan["max"])),
                                   float(scan["growth_factor"]))
            new_values = [value for value in expanded if value not in measured]
            if new_values:
                additions.update(new_values)
                expansion[method] += 1
                stop_reason[method] = "expanding"
            else:
                stop_reason[method] = "scan_limit"
        if not additions:
            break
        points = sorted(set(points) | additions)
    def measure_beta1(value: float) -> dict:
        return _measure_point(
            cfg, run_dir, settings, tag="obs_coarse", L=Lc, beta=beta_c,
            beta1=value, alpha=alpha_eff_c, alpha1=0.0, kind="half")

    final_matches = {
        "magsus": _uniform_grid_match(
            measured=measured, measure=measure_beta1,
            observable=lambda result: (float(result["magsus"]) * R ** 2,
                                       float(result["magsus_err"]) * R ** 2),
            target=float(fine["magsus"]), target_err=float(fine["magsus_err"]),
            increasing=True, scan=scan,
            bootstrap_seed=int(cfg["runtime"]["seed"]) + 201),
        "corr": _uniform_grid_match(
            measured=measured, measure=measure_beta1,
            observable=lambda result: (
                float(result["xi"]) * R if result.get("xi") is not None else np.nan,
                float(result["xi_err"]) * R
                if result.get("xi_err") is not None else np.nan),
            target=float(fine["xi"]), target_err=float(fine["xi_err"]),
            increasing=True, scan=scan,
            bootstrap_seed=int(cfg["runtime"]["seed"]) + 202),
    }
    if censored is not None and not final_matches["corr"].get("bracketed"):
        final_matches["corr"]["censored_bracket"] = censored
    matches = final_matches
    if not matches.get("corr", {}).get("bracketed") and matches.get("corr", {}).get(
            "censored_bracket"):
        matches["corr"]["reason"] = "xi onset reached before a finite lower bracket point"
    for method, match in matches.items():
        match["stop_reason"] = stop_reason[method]
        match["search_stop_reason"] = stop_reason[method]
        match["expansion_rounds"] = expansion[method]
        match["refinement_rounds"] = refinement[method]
    used = {float(value) for match in matches.values()
            for value in match.get("fit_points", [])}
    result = {"step": "observable_beta1", "fingerprint": result_key,
              "fine": {"xi": fine["xi"],
              "xi_err": fine["xi_err"], "magsus": fine["magsus"],
              "magsus_err": fine["magsus_err"], "mode_tau": fine["mode_tau"],
              "mode_ess": fine["mode_ess"], "sampling": fine["sampling"]},
              "points": sorted(measured),
              "coarse": {str(x): {"xi": measured[x]["xi"],
                                   "xi_err": measured[x]["xi_err"],
                                   "magsus": measured[x]["magsus"],
                                   "magsus_err": measured[x]["magsus_err"],
                                   "mode_tau": measured[x]["mode_tau"],
                                   "mode_ess": measured[x]["mode_ess"],
                                   "sampling": measured[x]["sampling"]}
                         for x in sorted(measured)},
              "matches": matches}
    _plot_beta1_scan(run_dir, R, fine, measured, matches)
    atomic_json(result_path, result)
    return {**result, "_cleanup_folders": _cleanup_candidates(measured, used)}


def _run_one_plaq(cfg: dict, run_dir: Path, geometry: dict) -> dict:
    result_path = run_dir / "step3a_1plaq" / "result.json"
    result_key = patch_result_key(cfg, geometry, "one_plaq")
    if result_path.exists() and not cfg.get("_force"):
        cached = json.loads(result_path.read_text(encoding="utf-8"))
        if (cached.get("cache_schema") == PATCH_CACHE_SCHEMA and
                cached.get("fingerprint") == result_key):
            _log("reuse completed step 3a (1plaq)")
            return cached
    _log("step 3a/4: 1plaq alpha fit")
    batch_dir = run_dir / "step3a_1plaq" / "batches"
    raw = generate_one_plaq(cfg, geometry, batch_dir)
    selected = "s" if cfg["renormalization"]["type"] == "U" else "z"
    p0 = float(cfg["steps"]["one_plaq"].get("p0", 0.5))
    fit = isolated_fit("one_plaq", {"p0": p0}, X=raw[f"X_{selected}"],
                       freq=raw[f"freq_{selected}"])
    atomic_npz(run_dir / "step3a_1plaq" / "measurements.npz", **raw)
    stop_reasons = np.asarray(raw["sampling_stop_reason"]).astype(str)
    warmup_reasons = np.asarray(raw["warmup_stop_reason"]).astype(str)
    outer_warmup_reasons = np.asarray(raw["outer_warmup_stop_reason"]).astype(str)
    warnings = ([] if not np.any(stop_reasons == "max_meas_per_boundary") else
                ["one or more batches reached maximum per-boundary measurements before vortex target ESS"])
    if np.any(warmup_reasons == "max_warmup"):
        warnings.append("one or more inner patches reached maximum warmup")
    if np.any(outer_warmup_reasons == "max_warmup"):
        warnings.append("one or more outer boundary lattices reached maximum warmup")
    patch_accept = float(np.mean(raw["accept_rate"]))
    if not cfg["diagnostics"]["min_accept"] <= patch_accept <= cfg["diagnostics"]["max_accept"]:
        warnings.append(f"HMC acceptance {patch_accept:.3f} outside configured range")
    if warnings and cfg["diagnostics"].get("strict"):
        raise RuntimeError("one_plaq diagnostics failed: " + "; ".join(warnings))
    result = {"step": "one_plaq", "cache_schema": PATCH_CACHE_SCHEMA,
              "fingerprint": result_key,
              "selected": selected, **fit,
              "accept_rate": float(np.mean(raw["accept_rate"])),
              "s_accept_rate": (float(np.mean(raw["s_accept_rate"]))
                                if np.asarray(raw["s_accept_rate"]).size else None),
              "fit_times": {"requested": int(raw["requested_fit_times"]),
                            "effective": int(raw["effective_fit_times"]),
                            "chains": int(raw["chains"]),
                            "batches": int(raw["batch_count"])},
              "epsilon": {"mean": float(np.mean(raw["epsilon"])),
                          "per_batch": np.asarray(raw["epsilon"])},
              "warmup": {"outer_sweeps": np.asarray(raw["outer_warmup_sweeps"]),
                         "outer_tau": np.asarray(raw["outer_warmup_tau"]),
                         "outer_stop_reason": outer_warmup_reasons,
                         "inner_sweeps": np.asarray(raw["warmup_sweeps"]),
                         "inner_tau": np.asarray(raw["warmup_tau"]),
                         "inner_stop_reason": warmup_reasons,
                         "inner_initialization": np.asarray(raw["inner_initialization"]),
                         "inner_adapt_steps": np.asarray(raw["inner_adapt_steps"])},
              "sampling": {"minimum_per_boundary": int(raw["sampling_minimum"]),
                           "actual_per_boundary": np.asarray(raw["sampling_actual"]),
                           "total_actual": int(raw["sampling_total_actual"]),
                           "maximum_per_boundary": int(raw["sampling_maximum"]),
                           "target_ess_per_boundary": float(raw["sampling_target_ess"]),
                           "tau": np.asarray(raw["sampling_tau"]),
                           "ess": np.asarray(raw["sampling_ess"]),
                           "stop_reason": stop_reasons},
              "diagnostics": {"valid": not warnings, "warnings": warnings}}
    atomic_json(result_path, result)
    return result


def _default_alpha_points(center: float, scan: dict) -> list[float]:
    if scan.get("points"):
        return _scan_points(scan)
    values = (center * np.linspace(0.8, 1.2, 5) if center > 0 else
              np.linspace(0.0, min(0.4, float(scan["max"])), 5))
    lo, hi = float(scan["min"]), float(scan["max"])
    points = sorted(set(max(lo, min(hi, float(x))) for x in values))
    if len(points) < 2:
        upper = min(hi, max(lo * 1.5, lo + 0.001))
        points = sorted(set(float(x) for x in np.linspace(lo, upper, 5)))
    return points


def _run_topo(cfg: dict, run_dir: Path, geometry: dict, beta_c: float,
              alpha_eff_c: float, beta1_matches: dict, alpha_1plaq: float) -> dict:
    _log("step 3b/4: topological alpha scans")
    settings, scan = section(cfg, "topo"), cfg["scan"]["alpha"]
    m, R, Lf, Lc = cfg["model"], geometry["factor"], geometry["L_fine"], geometry["L_coarse"]
    result_path = run_dir / "step3b_topo" / "result.json"
    result_key = _scan_fingerprint(
        "alpha", cfg, geometry, beta_c=beta_c, alpha_eff_c=alpha_eff_c,
        alpha_1plaq=alpha_1plaq,
        beta1_magsus=beta1_matches["magsus"].get("value"),
        beta1_corr=beta1_matches["corr"].get("value"))
    if result_path.exists() and not cfg.get("_force"):
        cached = json.loads(result_path.read_text(encoding="utf-8"))
        if cached.get("fingerprint") == result_key:
            _log("reuse completed step 3b (topological alpha scans)")
            return cached
    fine_kind = "full" if (cfg["renormalization"]["type"] == "U"
                           or abs(float(m["alpha"])) > 1e-12) else "half"
    fine_alpha = float(m["alpha"]) if fine_kind == "full" else float(m["alpha1"])
    fine_alpha1 = float(m["alpha1"]) if fine_kind == "full" else 0.0
    fine = _measure_point(cfg, run_dir, settings, tag="topo_fine", L=Lf,
                          beta=float(m["beta"]), beta1=float(m["beta1"]),
                          alpha=fine_alpha, alpha1=fine_alpha1, kind=fine_kind)
    fine_comp = "Q_s" if cfg["renormalization"]["type"] == "U" else "Q_z"
    target = fine["topology"][fine_comp]["topo_sus"] * R ** 2
    target_err = fine["topology"][fine_comp]["err"] * R ** 2
    all_results = {}
    cleanup_folders: list[str] = []
    for method in ("magsus", "corr"):
        bmatch = beta1_matches[method]
        if not bmatch.get("bracketed"):
            all_results[method] = {"value": None, "reason": "beta1 was not bracketed"}
            continue
        beta1_c, points, measured = float(bmatch["value"]), _default_alpha_points(alpha_1plaq, scan), {}
        match = {}
        expansion = 0
        refinement = 0
        stop_reason = "unbracketed"
        max_iterations = 2 * int(scan["max_rounds"]) + 2 * int(scan["max_refine_rounds"]) + 8
        for _ in range(max_iterations):
            for alpha in points:
                if alpha not in measured:
                    measured[alpha] = _measure_point(
                        cfg, run_dir, settings, tag=f"topo_coarse_{method}", L=Lc,
                        beta=beta_c, beta1=beta1_c, alpha=alpha,
                        alpha1=alpha_eff_c - alpha, kind="full")
            xs = np.array(sorted(measured))
            ys = np.array([measured[x]["topology"]["Q_s"]["topo_sus"] for x in xs])
            errs = np.array([measured[x]["topology"]["Q_s"]["err"] for x in xs])
            match = _raw_scan_match(xs, ys, target)
            candidate, reason = _scan_decision(
                match, tolerance=float(scan["refine_tolerance"]),
                max_refine=int(scan["max_refine_rounds"]),
                refine_count=refinement)
            if candidate is not None and candidate not in measured:
                points = sorted(set(points) | {candidate})
                refinement += 1
                stop_reason = "refining"
                continue
            if reason != "unbracketed":
                stop_reason = reason
                break
            if expansion >= int(scan["max_rounds"]):
                stop_reason = "max_rounds"
                break
            new_points = extend_scan(points, match, (float(scan["min"]),
                                     float(scan["max"])), float(scan["growth_factor"]))
            if new_points == points:
                stop_reason = "scan_limit"
                break
            points = new_points
            expansion += 1
            stop_reason = "expanding"
        def measure_alpha(value: float) -> dict:
            return _measure_point(
                cfg, run_dir, settings, tag=f"topo_coarse_{method}", L=Lc,
                beta=beta_c, beta1=beta1_c, alpha=value,
                alpha1=alpha_eff_c - value, kind="full")

        match = _uniform_grid_match(
            measured=measured, measure=measure_alpha,
            observable=lambda result: (
                float(result["topology"]["Q_s"]["topo_sus"]),
                float(result["topology"]["Q_s"]["err"])),
            target=float(target), target_err=float(target_err), increasing=False,
            scan=scan, bootstrap_seed=int(cfg["runtime"]["seed"]) +
                                      (301 if method == "magsus" else 302))
        match["stop_reason"] = stop_reason
        match["search_stop_reason"] = stop_reason
        match["expansion_rounds"] = expansion
        match["refinement_rounds"] = refinement
        used = {float(value) for value in match.get("fit_points", [])}
        cleanup_folders.extend(_cleanup_candidates(measured, used))
        all_results[method] = {**match, "beta1_c": beta1_c,
                               "points": sorted(measured), "target": target,
                               "target_err": target_err,
                               "measurements": {str(x): {
                                   "value": measured[x]["topology"]["Q_s"]["topo_sus"],
                                   "error": measured[x]["topology"]["Q_s"]["err"],
                                   "tau": measured[x]["topology"]["Q_s"]["tau"],
                                   "ess": measured[x]["topology"]["Q_s"]["ess"],
                                   "sampling": measured[x]["sampling"]}
                                   for x in sorted(measured)}}
        xs = np.array(sorted(measured))
        isolated_plot(run_dir / "step3b_topo" / f"alpha_scan_{method}.png", x=xs,
                      series={"topo_sus_Qs": np.array([
                          measured[x]["topology"]["Q_s"]["topo_sus"] for x in xs])},
                      errors={"topo_sus_Qs": np.array([
                          measured[x]["topology"]["Q_s"]["err"] for x in xs])},
                      matches={"topo_sus_Qs": match},
                      targets={f"fine_{fine_comp}_scaled": target},
                      colors={"topo_sus_Qs": "tab:purple",
                              f"fine_{fine_comp}_scaled": "tab:purple"},
                      xlabel="alpha", ylabel="topological susceptibility")
    result = {"step": "topo_alpha", "fingerprint": result_key,
              "fine_component": fine_comp,
              "fine_target": target, "fine_target_err": target_err,
              "fine_tau": fine["topology"][fine_comp]["tau"],
              "fine_ess": fine["topology"][fine_comp]["ess"],
              "fine_sampling": fine["sampling"], "matches": all_results}
    atomic_json(result_path, result)
    return {**result, "_cleanup_folders": cleanup_folders}


def run_pipeline(cfg: dict, output_root: str | Path, *, skip_topo: bool = False) -> dict:
    run_name = Path(cfg["_config_path"]).stem
    current_config_fingerprint = config_fingerprint(cfg)
    run_id = run_name
    run_dir = workspace_dir(cfg, output_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_copy(cfg["_config_path"], run_dir / "input.toml")
    _log(f"run {run_id} on {resolve_device(cfg['runtime']['device'])}")
    pilot = run_pilot(cfg, run_dir)
    geometry = pilot["geometry"]
    _log(f"pilot xi={pilot['xi']:.6g}±{pilot['xi_err']:.2g}; "
         f"padding={geometry['padding']} Lf={geometry['L_fine']} Lc={geometry['L_coarse']}")
    atomic_json(run_dir / "resolved.json", {"config": cfg, "pilot": pilot})
    step1 = _run_two_plaq(cfg, run_dir, geometry)
    beta_c, alpha_eff_c = round(step1["beta"], 3), round(step1["alpha_eff"], 3)
    step2 = _run_beta1_scan(cfg, run_dir, geometry, beta_c, alpha_eff_c)
    _log(f"step 2/4 beta1 matches: {step2['matches']}")
    skip_alpha = (cfg["renormalization"]["type"] == "U"
                  and abs(float(cfg["model"]["alpha"])) < 1e-12)
    if skip_alpha:
        alpha_1 = 0.0
        topo = {"matches": {"magsus": {"value": 0.0}, "corr": {"value": 0.0}},
                "skipped": "U-renorm with unconstrained fine s"}
    else:
        step3a = _run_one_plaq(cfg, run_dir, geometry)
        alpha_1 = float(step3a["alpha"])
        topo = ({"matches": {"magsus": {"value": None}, "corr": {"value": None}},
                 "skipped": "--skip-topo"} if skip_topo else
                _run_topo(cfg, run_dir, geometry, beta_c, alpha_eff_c,
                          step2["matches"], alpha_1))
    beta1 = {k: (float(v["value"]) if v.get("value") is not None else None)
             for k, v in step2["matches"].items()}
    beta1_err = {k: (float(v["error"]) if v.get("error") is not None else None)
                 for k, v in step2["matches"].items()}
    atop = {k: (float(v["value"]) if v.get("value") is not None else None)
            for k, v in topo["matches"].items()}
    atop_err = {k: (float(v["error"]) if v.get("error") is not None else None)
                for k, v in topo["matches"].items()}

    def combo(bkey: str, alpha: float | None) -> dict:
        return {"beta_c": beta_c, "beta1_c": beta1[bkey], "alpha_c": alpha,
                "alpha1_c": (float(alpha_eff_c - alpha) if alpha is not None else None),
                "alpha_eff_c": alpha_eff_c}

    combos = {"magsus_obs_fit_topo": combo("magsus", atop["magsus"]),
              "corr_obs_fit_topo": combo("corr", atop["corr"]),
              "magsus_1plaq": combo("magsus", alpha_1),
              "corr_1plaq": combo("corr", alpha_1)}
    def toml_fragment(values: dict) -> str | None:
        fields = (values["beta_c"], values["beta1_c"],
                  values["alpha_c"], values["alpha1_c"])
        if any(value is None for value in fields):
            return None
        return (f"beta = {float(fields[0]):.12g}\n"
                f"beta1 = {float(fields[1]):.12g}\n"
                f"alpha = {float(fields[2]):.12g}\n"
                f"alpha1 = {float(fields[3]):.12g}")

    def fit_design(match: dict) -> dict:
        preliminary = match.get("preliminary", {})
        return {"preliminary_value": preliminary.get("value"),
                "preliminary_error": preliminary.get("error"),
                "fit_spacing": match.get("fit_spacing"),
                "fit_grid": match.get("fit_grid", []),
                "fit_grid_rounds": match.get("fit_grid_rounds", 0),
                "grid_validation": match.get("grid_validation"),
                "reason": match.get("reason")}

    summary = {"run_name": run_name, "run_id": run_id,
               "config_fingerprint": current_config_fingerprint,
               "created_at": datetime.now().isoformat(timespec="seconds"),
               "device": str(resolve_device(cfg["runtime"]["device"])),
               "geometry": geometry,
               "renormalized_couplings": {"beta_c": beta_c, "alpha_eff_c": alpha_eff_c,
                                            "beta1_c": beta1, "beta1_c_err": beta1_err,
                                            "alpha_c_1plaq": alpha_1,
                                            "alpha_c_topo": atop,
                                            "alpha_c_topo_err": atop_err},
               "fit_points": {
                   "beta1": {name: value.get("fit_points", [])
                             for name, value in step2["matches"].items()},
                   "alpha": {name: value.get("fit_points", [])
                            for name, value in topo["matches"].items()}},
               "fit_design": {
                   "beta1": {name: fit_design(value)
                             for name, value in step2["matches"].items()},
                   "alpha": {name: fit_design(value)
                            for name, value in topo["matches"].items()}},
               "four_combos": combos,
               "renorm_as_fine": {name: fragment for name, value in combos.items()
                                  if (fragment := toml_fragment(value)) is not None}}
    cleanup = list(step2.get("_cleanup_folders", [])) + list(
        topo.get("_cleanup_folders", []))
    removed = _remove_scan_ensembles(run_dir, cleanup)
    summary["cleanup"] = {"removed_unused_coarse_ensembles": removed,
                          "count": len(removed)}
    atomic_json(run_dir / "summary.json", summary)
    _log(f"done: {run_dir / 'summary.json'}")
    return summary
