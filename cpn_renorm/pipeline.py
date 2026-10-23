from __future__ import annotations

import json
import math
import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .cache import (load_point, point_identity, read_json,
                    scan_measurements, store_point, write_scan_points)
from .config import fingerprint, mod_config, section
from .fit_client import extend_scan, isolated_fit, isolated_match, isolated_plot
from .methods import BETA1_METHODS
from .observables import measure_ensemble
from .patches import (generate_one_plaq, generate_two_plaq,
                      patch_result_key)
from .pilot import resolve_device, run_pilot, torch_dtype
from .progress import SimulationProgress, ensemble_summary
from .sampler import BatchedHMCSampler, Couplings
from .storage import atomic_copy, atomic_json, atomic_npz
from .workspace import workspace_dir


def _log(message: str) -> None:
    print(f"[cpn-renorm] {message}", flush=True)


def _renorm_types(cfg: dict) -> list[str]:
    return list(cfg["renormalization"]["type"])


def _point_sampler(cfg: dict, settings: dict, *, L: int, beta: float, beta1: float,
                   alpha: float, alpha1: float, kind: str, seed: int) -> BatchedHMCSampler:
    rt, m = cfg["runtime"], cfg["model"]
    if kind == "full" and alpha == 0:
        kind, alpha, alpha1 = "half", alpha1, 0.0
    return BatchedHMCSampler(
        chains=int(settings["chains"]), Lx=L, Ly=L, N=int(m["N"]),
        couplings=Couplings(beta, beta1, alpha, alpha1, int(m["mod"])),
        kind=kind, periodic=True, device=resolve_device(rt["device"]),
        dtype=torch_dtype(rt["dtype"]), seed=seed,
        epsilon=float(settings["epsilon"]),
        trajectory_length=float(settings["trajectory_length"]),
        mass_a=float(settings["mass_a"]), mass_z=float(settings["mass_z"]))


def _apply_point_diagnostics(cfg: dict, tag: str, result: dict) -> None:
    accept_mean = float(np.mean(result.get("accept_rate", np.nan)))
    diag_cfg = cfg["diagnostics"]
    warnings = []
    if math.isfinite(accept_mean) and not diag_cfg["min_accept"] <= accept_mean <= diag_cfg["max_accept"]:
        cause = ("integration error is likely too large"
                 if accept_mean < diag_cfg["min_accept"] else
                 "epsilon is likely too small and leapfrog cost excessive")
        warnings.append(
            f"HMC acceptance {accept_mean:.3f} outside configured range; {cause}")
    if result["sampling"]["stop_reason"] == "max_meas_total":
        warnings.append(
            "maximum measurements reached before every primary observable met target ESS")
    if result.get("warmup", {}).get("stop_reason") == "max_warmup":
        warnings.append("maximum warmup reached before autocorrelation times stabilized")
    result["diagnostics"] = {"valid": not warnings, "warnings": warnings}
    if warnings and diag_cfg.get("strict"):
        raise RuntimeError(f"{tag} diagnostics failed: {'; '.join(warnings)}")


def _measure_point(cfg: dict, run_dir: Path, settings: dict, *, tag: str, L: int,
                   beta: float, beta1: float, alpha: float, alpha1: float,
                   kind: str) -> dict:
    stage = "topo" if tag.startswith("topo") else "observable"
    lattice = "fine" if "_fine" in tag else "coarse"
    settings = {**settings, "chains": cfg["chains"][f"{stage}_{lattice}"]}
    key_name = f"{stage}_{lattice}"
    requested = {"sampling_revision": 3, "tag": tag, "L": L, "beta": beta, "beta1": beta1,
                 "alpha": alpha, "alpha1": alpha1, "kind": kind,
                 "N": cfg["model"]["N"], "mod": cfg["model"]["mod"],
                 "settings": settings, "runtime": cfg["runtime"]}
    requested_key, _, requested_context = point_identity(requested)
    existing = run_dir / "ensembles" / f"{requested_context['tag']}_{requested_key}"
    if (existing / "result.json").exists() and not cfg.get("_force"):
        result = load_point(existing)
        _apply_point_diagnostics(cfg, tag, result)
        return result
    if cfg["runtime"]["auto_chains"] and key_name not in cfg.setdefault("_tuned_measurements", set()):
        from .chain_tuning import Workload, _sampling_sweeps, tune_stage
        workload = Workload(tag, L, L, True, kind,
                            Couplings(beta, beta1, alpha, alpha1, int(cfg["model"]["mod"])),
                            _sampling_sweeps(settings),
                            n_meas=int(math.ceil(settings["max_meas_total"] / settings["chains"])),
                            stores_correlations=True,
                            measurement_total=int(settings["max_meas_total"]),
                            settings_name=stage)
        settings["chains"] = tune_stage(cfg, key_name, [workload])
        cfg["_tuned_measurements"].add(key_name)
    payload = {"sampling_revision": 3, "tag": tag, "L": L, "beta": beta, "beta1": beta1,
               "alpha": alpha, "alpha1": alpha1, "kind": kind,
               "N": cfg["model"]["N"], "mod": cfg["model"]["mod"],
               "settings": settings, "runtime": cfg["runtime"]}
    key, _, context = point_identity(payload)
    folder = run_dir / "ensembles" / f"{context['tag']}_{key}"
    meta_file = folder / "result.json"
    if meta_file.exists() and not cfg.get("_force"):
        meta = load_point(folder)
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
    folder = store_point(cfg, run_dir, payload, result)
    _log(ensemble_summary(label, result))
    result.update({"cache": "generated", "_folder": str(folder)})
    return result


def _scan_points(scan: dict) -> list[float]:
    if scan.get("points"):
        return sorted(set(float(x) for x in scan["points"]))
    lo, hi = float(scan["min"]), float(scan["max"])
    points = [x for x in (-0.1, 0.0, 0.1) if lo <= x <= hi]
    return points if len(points) >= 2 else [lo, 0.5 * (lo + hi), hi]


def _scan_fingerprint(kind: str, cfg: dict, geometry: dict, *, renorm_type: str,
                      **upstream: float) -> str:
    step = "observable" if kind == "beta1" else "topo"
    scan = dict(cfg["scan"][kind])
    # Preserve pre-option cache keys at the new default; existing fits remain valid.
    if float(scan.get("fit_spacing_growth_factor", 2.0)) == 2.0:
        scan.pop("fit_spacing_growth_factor", None)
    return fingerprint({"kind": kind, "geometry": geometry,
                        "scan": scan, "settings": section(cfg, step),
                        "fine_chains": cfg["chains"][f"{step}_fine"],
                        "model": cfg["model"],
                        "renormalization": {"factor": cfg["renormalization"]["factor"],
                                            "type": renorm_type},
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


def _censored_xi_G_bracket(xs: np.ndarray, ys: np.ndarray,
                         target: float) -> list[float] | None:
    """Bracket xi_G onset when an undefined lower point precedes xi_G > target."""
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
    # A censored xi_G bracket still lacks a finite point below the target. Keep
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
                                             spacing * float(scan["fit_spacing_growth_factor"])))
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


def _one_fit_key(cfg: dict, geometry: dict) -> str:
    return fingerprint({"raw": patch_result_key(cfg, geometry, "one_plaq"), "algorithm": 3,
                              "types": _renorm_types(cfg), "device": str(resolve_device(cfg["runtime"]["device"]))})


def _two_fit_key(cfg: dict, geometry: dict, one_plaq: dict) -> str:
    return fingerprint({"raw": patch_result_key(cfg, geometry, "two_plaq"),
                              "fixed_alpha": one_plaq["fits"], "algorithm": 3,
                              "device": str(resolve_device(cfg["runtime"]["device"]))})


def _two_raw_key(cfg: dict, geometry: dict) -> str:
    # patch_result_key canonicalizes mod when the fine beta1 is zero.
    return patch_result_key(cfg, geometry, "two_plaq")


def _run_two_plaq(cfg: dict, run_dir: Path, geometry: dict, one_plaq: dict) -> dict:
    result_path = run_dir / "step2_2plaq" / "result.json"
    result_key = _two_fit_key(cfg, geometry, one_plaq)
    if result_path.exists() and not cfg.get("_force"):
        cached = json.loads(result_path.read_text(encoding="utf-8"))
        if cached.get("fingerprint") == result_key:
            _log("reuse completed step 2 (2plaq)")
            return cached
    _log("step 2/4: 2plaq old/new fits")
    batch_dir = run_dir / "step2_2plaq" / "batches"
    raw = generate_two_plaq(cfg, geometry, batch_dir)
    p0 = cfg["steps"]["two_plaq"]["p0"]
    fits = {}
    for tag in _renorm_types(cfg):
        fixed = one_plaq["fits"][tag]
        fits[tag] = isolated_fit("two_fixed", {
            "N": int(cfg["model"]["N"]), "p0": p0[tag], "alpha": fixed["alpha"],
            "device": str(resolve_device(cfg["runtime"]["device"]))},
            X=raw[f"X_{tag}"], freq=raw[f"freq_{tag}"])
        fits[tag]["new"].update(alpha_source=fixed.get("source", "1plaq"),
                                  fixed_alpha_error=fixed.get("alpha_err"))
    atomic_json(run_dir / "step2_2plaq" / "fit.log", fits)
    raw["two_plaq_raw_key"] = np.asarray(_two_raw_key(cfg, geometry))
    cfg["_two_raw"] = raw
    measurement_path = run_dir / "step2_2plaq" / "measurements.npz"
    if cfg["storage"]["save_npz"]:
        atomic_npz(measurement_path, **raw)
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
    result = {"step": "2plaq",
              "fingerprint": result_key, "fits": fits,
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
    xs = np.array(sorted(measured))
    for method, field, scale in (("magsus", "magsus", scale_factor ** 2),
                                  ("corrG", "xi_G", scale_factor)):
        if method not in matches:
            continue
        label = "magsus_scaled" if method == "magsus" else "xi_G_scaled"
        isolated_plot(run_dir / "step3_observable" / f"beta1_scan_{method}.png", x=xs,
            series={label: np.array([float(measured[x][field]) * scale
                if measured[x].get(field) is not None else np.nan for x in xs])},
            errors={label: np.array([float(measured[x][field + "_err"]) * scale
                if measured[x].get(field + "_err") is not None else np.nan for x in xs])},
            matches={label: matches[method]}, targets={f"fine_{field}": fine[field]},
            xlabel="beta1", ylabel=field)
    old_plot = run_dir / "step3_observable" / "beta1_scan.png"
    if old_plot.exists():
        old_plot.unlink()


def _method_key(cfg: dict, geometry: dict, *, method: str, kind: str,
                renorm_type: str, **upstream: float) -> str:
    return fingerprint({"scan": _scan_fingerprint(kind, cfg, geometry,
                       renorm_type=renorm_type, **upstream), "method": method})


def _search_fit(cfg: dict, scan: dict, measured: dict, phases: dict,
                measure: Callable, observable: Callable, target: float,
                target_err: float, *, initial: list[float], increasing: bool,
                seed: int, censored_xi_G: bool = False) -> dict:
    def add(value: float, phase: str) -> None:
        phases.setdefault(value, set()).add(phase)
        if not _same_point(value, set(measured)):
            measured[value] = measure(value)

    for value in initial:
        add(value, "coarse")
    expansion = refinement = 0
    censored = None
    for _ in range(int(scan["max_rounds"]) + int(scan["max_refine_rounds"]) + 2):
        xs = np.array(sorted(measured))
        ys = np.array([observable(measured[x])[0] for x in xs])
        raw = _raw_scan_match(xs, ys, target)
        censored = _censored_xi_G_bracket(xs, ys, target) if censored_xi_G else None
        candidate, reason = _scan_decision(
            raw, tolerance=float(scan["refine_tolerance"]),
            max_refine=int(scan["max_refine_rounds"]), refine_count=refinement,
            censored=censored)
        if candidate is not None and not _same_point(candidate, set(measured)):
            add(candidate, "refine")
            refinement += 1
            continue
        if reason != "unbracketed":
            break
        if expansion >= int(scan["max_rounds"]):
            reason = "max_rounds"
            break
        expanded = extend_scan(sorted(measured), raw,
                               (float(scan["min"]), float(scan["max"])),
                               float(scan["growth_factor"]))
        additions = [x for x in expanded if not _same_point(x, set(measured))]
        if not additions:
            reason = "scan_limit"
            break
        for value in additions:
            add(value, "coarse")
        expansion += 1

    def final_measure(value: float) -> dict:
        phases.setdefault(value, set()).add("fit_candidate")
        return measure(value)

    match = _uniform_grid_match(
        measured=measured, measure=final_measure, observable=observable,
        target=target, target_err=target_err, increasing=increasing,
        scan=scan, bootstrap_seed=int(cfg["runtime"]["seed"]) + seed)
    for value in match.get("fit_points", []):
        phases.setdefault(value, set()).add("final")
    if censored and not match.get("bracketed"):
        match["censored_bracket"] = censored
        match["reason"] = "xi_G onset reached before a finite lower bracket point"
    match.update(search_stop_reason=reason, expansion_rounds=expansion,
                 refinement_rounds=refinement)
    return match


def _retain_scan(directory: Path, measured: dict, phases: dict, *, tag: str) -> tuple[dict, list[str]]:
    keep = {value for value, labels in phases.items()
            if labels & {"coarse", "final"}}
    cleanup = _cleanup_candidates(measured, keep)
    retained = {value: result for value, result in measured.items() if _same_point(value, keep)}
    return write_scan_points(directory, retained, phases, tag=tag), cleanup


def _run_beta1_scan(cfg: dict, run_dir: Path, geometry: dict,
                    beta_c: float, alpha1_c: float,
                    renorm_type: str | None = None, *, alpha_c: float = 0.0) -> dict:
    methods = cfg["renormalization"]["beta1_methods"]
    if not methods:
        return {"matches": {}}
    settings, scan = section(cfg, "observable"), cfg["scan"]["beta1"]
    m, R = cfg["model"], geometry["factor"]
    renorm_type = renorm_type or _renorm_types(cfg)[0]
    path = run_dir / "step3_observable" / "result.json"
    cached = read_json(path) if path.exists() and not cfg.get("_force") else {}
    keys = {name: _method_key(cfg, geometry, method=name, kind="beta1",
                              renorm_type=renorm_type, beta_c=beta_c,
                              alpha1_c=alpha1_c, alpha_c=alpha_c) for name in methods}
    measured, phases = scan_measurements(run_dir, cached)
    # Measurement parameters can change independently of method selection.
    compatible = cached.get("measurement_key") == _scan_fingerprint(
        "beta1", cfg, geometry, renorm_type=renorm_type,
        beta_c=beta_c, alpha1_c=alpha1_c, alpha_c=alpha_c)
    if not compatible:
        measured, phases = {}, {}
    matches = dict(cached.get("matches", {})) if compatible else {}
    fingerprints = dict(cached.get("method_fingerprints", {})) if compatible else {}
    fine_path = run_dir / cached.get("fine_path", "missing")
    fine = load_point(fine_path.parent) if compatible and fine_path.is_file() else _measure_point(
        cfg, run_dir, settings, tag="obs_fine", L=geometry["L_fine"],
        beta=float(m["beta"]), beta1=float(m["beta1"]),
        alpha=float(m["alpha"]), alpha1=float(m["alpha1"]), kind="full")

    def measure(value: float) -> dict:
        return _measure_point(cfg, run_dir, settings, tag="obs_coarse",
                              L=geometry["L_coarse"], beta=beta_c, beta1=value,
                              alpha=alpha_c, alpha1=alpha1_c, kind="full")

    for name in methods:
        if fingerprints.get(name) == keys[name] and name in matches:
            continue
        descriptor = BETA1_METHODS[name]
        field, scale = descriptor["field"], R ** descriptor["power"]

        def observable(result: dict) -> tuple[float, float]:
            return (float(result[field]) * scale if result.get(field) is not None else np.nan,
                    float(result[field + "_err"]) * scale if result.get(field + "_err") is not None else np.nan)

        matches[name] = _search_fit(
            cfg, scan, measured, phases, measure, observable,
            float(fine[field]), float(fine[field + "_err"]),
            initial=_scan_points(scan), increasing=descriptor["increasing"],
            seed=descriptor["seed"], censored_xi_G=field == "xi_G")
        fingerprints[name] = _method_key(cfg, geometry, method=name, kind="beta1",
            renorm_type=renorm_type, beta_c=beta_c, alpha1_c=alpha1_c, alpha_c=alpha_c)
    entries, cleanup = _retain_scan(run_dir, measured, phases, tag="obs_coarse")
    fine_entries = write_scan_points(run_dir, {0.0: fine}, {0.0: {"reference"}}, tag="obs_fine")
    result = {"step": "observable_beta1", "measurement_key": _scan_fingerprint(
        "beta1", cfg, geometry, renorm_type=renorm_type, beta_c=beta_c,
        alpha1_c=alpha1_c, alpha_c=alpha_c), "fine_path": fine_entries["0.0"]["path"],
        "measurements": entries, "matches": matches, "method_fingerprints": fingerprints}
    atomic_json(path, result)
    selected = {name: matches[name] for name in methods}
    _plot_beta1_scan(run_dir, R, fine, measured, selected)
    return {**result, "matches": selected, "_cleanup_folders": cleanup}


def _run_one_plaq(cfg: dict, run_dir: Path, geometry: dict) -> dict:
    result_path = run_dir / "step1_1plaq" / "result.json"
    result_key = _one_fit_key(cfg, geometry)
    if result_path.exists() and not cfg.get("_force"):
        cached = json.loads(result_path.read_text(encoding="utf-8"))
        if cached.get("fingerprint") == result_key:
            _log("reuse completed step 1 (1plaq)")
            return cached
    _log("step 1/4: 1plaq alpha fit")
    batch_dir = run_dir / "step1_1plaq" / "batches"
    if _renorm_types(cfg) == ["U"] and float(cfg["model"]["alpha"]) == 0.0:
        result = {"step": "one_plaq", "fingerprint": result_key, "fits": {
            "U": {"alpha": 0.0, "alpha_err": None, "source": "alpha_zero_subspace_convention"}}}
        atomic_json(result_path, result)
        return result
    raw = generate_one_plaq(cfg, geometry, batch_dir)
    p0 = cfg["steps"]["one_plaq"]["p0"]
    fits = {}
    for tag in _renorm_types(cfg):
        if tag == "U" and float(cfg["model"]["alpha"]) == 0.0:
            fits[tag] = {"alpha": 0.0, "alpha_err": None, "source": "alpha_zero_subspace_convention"}
        else:
            fits[tag] = isolated_fit("one_plaq", {"p0": float(p0[tag]),
                "device": str(resolve_device(cfg["runtime"]["device"]))},
                X=raw["X_s" if tag == "U" else "X_z"], freq=raw["freq_s" if tag == "U" else "freq_z"])
            fits[tag]["source"] = "1plaq"
    atomic_json(run_dir / "step1_1plaq" / "fit.log", fits)
    if cfg["storage"]["save_npz"]:
        atomic_npz(run_dir / "step1_1plaq" / "measurements.npz", **raw)
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
    result = {"step": "one_plaq",
              "fingerprint": result_key, "fits": fits,
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
    lo, hi = float(scan["min"]), float(scan["max"])
    if not lo <= center <= hi:
        raise ValueError(f"1plaq alpha center {center} outside scan range [{lo}, {hi}]")
    if scan.get("points"):
        return _scan_points(scan)
    spacing = max(float(scan["refine_tolerance"]), float(scan["fit_spacing_min"]))
    return sorted(set([max(lo, center-spacing), center, min(hi, center+spacing)]))


def _run_topo(cfg: dict, run_dir: Path, geometry: dict, beta_c: float,
              alpha1_c: float, beta1_matches: dict, alpha_1plaq: float | None = None,
              renorm_type: str | None = None) -> dict:
    methods = cfg["renormalization"]["beta1_methods"]
    settings, scan = section(cfg, "topo"), cfg["scan"]["alpha"]
    if alpha_1plaq is None:
        raise ValueError("topo requires an independently estimated 1plaq alpha center")
    initial_points = _default_alpha_points(float(alpha_1plaq), scan)
    m, R = cfg["model"], geometry["factor"]
    renorm_type = renorm_type or _renorm_types(cfg)[0]
    path = run_dir / "step4_topo" / "result.json"
    cached = read_json(path) if path.exists() and not cfg.get("_force") else {}
    fine_kind = "full"
    component = "Q_s" if renorm_type == "U" else "Q_z"
    fine_parameters = dict(tag="topo_fine", L=geometry["L_fine"],
        beta=float(m["beta"]), beta1=float(m["beta1"]), kind=fine_kind,
        alpha=float(m["alpha"]) if fine_kind == "full" else float(m["alpha1"]),
        alpha1=float(m["alpha1"]) if fine_kind == "full" else 0.0)
    fine_key = fingerprint({"parameters": fine_parameters, "settings": settings,
                            "runtime": cfg["runtime"], "N": m["N"], "mod": m["mod"]})
    fine_path = run_dir / cached.get("fine_path", "missing")
    fine = load_point(fine_path.parent) if cached.get("fine_key") == fine_key and fine_path.is_file() else _measure_point(
        cfg, run_dir, settings, **fine_parameters)
    target = float(fine["topology"][component]["topo_sus"]) * R ** 2
    target_err = float(fine["topology"][component]["err"]) * R ** 2
    matches = dict(cached.get("matches", {}))
    fingerprints = dict(cached.get("method_fingerprints", {}))
    cleanup = []
    for name in methods:
        beta_match = beta1_matches[name]
        if not beta_match.get("bracketed"):
            matches[name] = {"value": None, "error": None, "reason": "beta1 was not bracketed",
                             "alpha_before": alpha_1plaq, "alpha_after": alpha_1plaq,
                             "alpha_correction": 0.0, "correction_success": False}
            fingerprints.pop(name, None)
            continue
        beta1_c = float(beta_match["value"])
        key = _method_key(cfg, geometry, method=name, kind="alpha",
                          renorm_type=renorm_type, beta_c=beta_c,
                          alpha1_c=alpha1_c, alpha_c=alpha_1plaq, beta1_c=beta1_c)
        if fingerprints.get(name) == key and name in matches:
            continue
        previous = matches.get(name, {}) if fingerprints.get(name) == key else {}
        measured, phases = scan_measurements(run_dir, previous)

        def measure(value: float) -> dict:
            return _measure_point(cfg, run_dir, settings, tag=f"topo_coarse_{name}",
                L=geometry["L_coarse"], beta=beta_c, beta1=beta1_c, alpha=value,
                alpha1=alpha1_c, kind="full")

        def observable(result: dict) -> tuple[float, float]:
            t = result["topology"]["Q_s"]
            return float(t["topo_sus"]), float(t["err"])

        match = _search_fit(cfg, scan, measured, phases, measure, observable,
            target, target_err, initial=initial_points,
            increasing=False, seed=int(BETA1_METHODS[name]["seed"]) + 100)
        entries, unused = _retain_scan(run_dir, measured, phases, tag=f"topo_coarse_{name}")
        cleanup.extend(unused)
        match.update(beta1_c=beta1_c, measurements=entries,
                     alpha_before=alpha_1plaq,
                     alpha_after=match.get("value") if match.get("bracketed") else alpha_1plaq,
                     alpha_correction=(float(match["value"])-alpha_1plaq) if match.get("bracketed") else 0.0,
                     correction_success=bool(match.get("bracketed")))
        matches[name] = match
        fingerprints[name] = _method_key(cfg, geometry, method=name, kind="alpha",
            renorm_type=renorm_type, beta_c=beta_c, alpha1_c=alpha1_c, alpha_c=alpha_1plaq,
            beta1_c=beta1_c)
        xs = np.array(sorted(measured))
        isolated_plot(run_dir / "step4_topo" / f"alpha_scan_{name}.png", x=xs,
            series={"topo_sus_Qs": np.array([observable(measured[x])[0] for x in xs])},
            errors={"topo_sus_Qs": np.array([observable(measured[x])[1] for x in xs])},
            matches={"topo_sus_Qs": match}, targets={f"fine_{component}_scaled": target},
            xlabel="alpha", ylabel="topological susceptibility")
    fine_entries = write_scan_points(run_dir, {0.0: fine}, {0.0: {"reference"}}, tag="topo_fine")
    fine_key = fingerprint({"parameters": fine_parameters, "settings": section(cfg, "topo"),
                            "runtime": cfg["runtime"], "N": m["N"], "mod": m["mod"]})
    result = {"step": "topo_alpha", "fine_key": fine_key,
              "fine_path": fine_entries["0.0"]["path"], "fine_component": component,
              "matches": matches, "method_fingerprints": fingerprints}
    atomic_json(path, result)
    # Another method may share the same physical ensemble.
    retained = {str((run_dir / entry["path"]).parent.resolve())
                for match in matches.values() for entry in match.get("measurements", {}).values()}
    cleanup = [folder for folder in cleanup if str(Path(folder).resolve()) not in retained]
    return {**result, "matches": {name: matches[name] for name in methods},
            "_cleanup_folders": cleanup}


def _save_topo(run_dir: Path, topo: dict) -> None:
    path = run_dir / "step4_topo" / "result.json"
    stored = read_json(path) if path.exists() else {}
    matches = {**stored.get("matches", {}), **topo["matches"]}
    atomic_json(path, {**{key: value for key, value in topo.items()
                         if not key.startswith("_")}, "matches": matches})


def _finish_topo(cfg: dict, run_dir: Path, common_dir: Path, geometry: dict,
                 common: dict, topo: dict, renorm_type: str, alpha_1: float) -> None:
    """Refit beta/alpha1 once at each matched alpha, without new scan ensembles."""
    two_plaq = common["two_plaq"]
    fit2 = two_plaq["fits"][renorm_type]["new"]
    device = str(resolve_device(cfg["runtime"]["device"]))
    raw_key = _two_raw_key(cfg, geometry)
    p0 = cfg["steps"]["two_plaq"]["p0"][renorm_type]
    for method, match in topo.get("matches", {}).items():
        if not match.get("bracketed"):
            continue
        key = fingerprint({"algorithm": 1, "raw": raw_key,
                           "initial_fit": two_plaq.get("fingerprint", fit2),
                           "type": renorm_type, "alpha": match["value"],
                           "p0": p0, "device": device})
        cached = match.get("two_plaq_refit", {})
        if not cfg.get("_force") and cached.get("fingerprint") == key:
            if cached.get("fit", {}).get("new", {}).get("success") is True:
                cached["fit"]["new"]["fixed_alpha_error"] = match.get("error")
                _log(f"reuse topo 2plaq refit: {renorm_type} {method}")
                continue
        raw = common.get("raw")
        if raw is None:
            path = common_dir / "step2_2plaq" / "measurements.npz"
            if path.exists() and not cfg.get("_force"):
                with np.load(path, allow_pickle=False) as stored:
                    # Older untagged files cannot establish compatibility.
                    if str(stored.get("two_plaq_raw_key", "")) == raw_key:
                        raw = {name: stored[name] for name in stored.files}
                if raw is not None:
                    common["raw_source"] = "persisted_measurements"
            if raw is None:
                source_cfg = mod_config(cfg, 0) if float(cfg["model"]["beta1"]) == 0 else cfg
                raw = generate_two_plaq(source_cfg, geometry,
                                        common_dir / "step2_2plaq" / "batches")
                raw["two_plaq_raw_key"] = np.asarray(raw_key)
                common["raw_source"] = "regenerated_measurements"
                if cfg["storage"]["save_npz"]:
                    atomic_npz(path, **raw)
            common["raw"] = raw
        source = common["raw_source"]
        arrays = {"X": raw[f"X_{renorm_type}"], "freq": raw[f"freq_{renorm_type}"]}
        request = {"N": int(cfg["model"]["N"]), "device": device}
        baseline_mse = fit2["mse"]
        if source == "regenerated_measurements":
            baseline_mse = isolated_fit("two_mse", {**request,
                "parameters": [fit2["beta"], alpha_1, fit2["alpha1"]]}, **arrays)["mse"]
        diagnostic = isolated_fit("two_mse", {**request,
            "parameters": [fit2["beta"], match["value"], fit2["alpha1"]]}, **arrays)
        _log(f"topo 2plaq refit: {renorm_type} {method} alpha={match['value']:.6g}")
        fitted = isolated_fit("two_fixed", {**request, "p0": p0,
                              "alpha": match["value"]}, **arrays)
        final = fitted["new"]
        if final.get("success") is not True or not all(
                np.isfinite(final[name]) for name in ("beta", "alpha", "alpha1", "mse")):
            raise RuntimeError(f"topo 2plaq refit failed: {renorm_type} {method}")
        final.update(alpha_source="topo_single_correction", fixed_alpha_error=match.get("error"))
        match.pop("two_plaq_final_diagnostic", None)  # Replaced by the explicitly alpha-only check.
        match.update(two_plaq_mse_before=baseline_mse, two_plaq_mse_after=diagnostic["mse"],
                     two_plaq_fit_mse=fit2["mse"], two_plaq_diagnostic_data_source=source,
                     two_plaq_alpha_only_diagnostic=diagnostic,
                     two_plaq_refit={"fingerprint": key, "raw_key": raw_key,
                         "data_source": source, "fit": fitted,
                         "parameters_before": {"beta": fit2["beta"], "alpha": alpha_1,
                                               "alpha1": fit2["alpha1"]},
                         "correction": {"alpha": final["alpha"] - alpha_1,
                                        "beta": final["beta"] - fit2["beta"],
                                        "alpha1": final["alpha1"] - fit2["alpha1"]}})
        # Persist each successful method before starting the next fit.
        _save_topo(run_dir, topo)


def _branch_summary(cfg: dict, run_name: str, renorm_type: str, geometry: dict,
                    beta_c: float, alpha1_c: float, alpha_1: float,
                    step2: dict, topo: dict, *, fit2: dict | None = None,
                    fit1: dict | None = None) -> dict:
    fit2, fit1 = fit2 or {}, fit1 or {}
    results, fragments = {}, {}
    for method in cfg["renormalization"]["beta1_methods"] or [None]:
        beta_match = step2.get("matches", {}).get(method, {})
        baseline = {"beta": beta_c, "beta1": beta_match.get("value"),
                    "alpha": alpha_1, "alpha1": alpha1_c}
        match = topo.get("matches", {}).get(method, {})
        # Always retain the independent 1plaq parameter combination for diagnostics.
        variants = [("1plaq", alpha_1, fit1.get("alpha_err"), fit2)]
        if cfg["renormalization"]["alpha_topo_fit"] and match.get("bracketed"):
            final = match["two_plaq_refit"]["fit"]["new"]
            variants.append(("obs_fit_topo", match["value"], match.get("error"), final))
        for suffix, alpha, error, fitted in variants:
            name = "_".join(x for x in (method, suffix) if x)
            couplings = {**baseline, "alpha": alpha, "beta": fitted["beta"],
                         "alpha1": fitted["alpha1"]}
            results[name] = {"couplings": couplings, "errors": {
                "beta": fitted.get("beta_err"), "beta1": beta_match.get("error"),
                "alpha": error, "alpha1": fitted.get("alpha1_err")},
                "error_scope": ("beta_and_alpha1_conditional_on_1plaq_alpha" if suffix == "1plaq"
                                else "beta_and_alpha1_conditional_on_topo_alpha"),
                "alpha_source": fit1.get("source", "1plaq") if suffix == "1plaq" else "topo_single_correction"}
            if suffix == "obs_fit_topo":
                results[name]["topo_correction"] = {
                    key: couplings[key] - baseline[key] for key in ("alpha", "beta", "alpha1")}
            if all(value is not None for value in couplings.values()):
                fragments[name] = "\n".join(f"{key} = {value!r}" for key,value in couplings.items())
    return {"results": results, "renorm_as_fine": fragments}


def _run_mod_pipeline(cfg: dict, output_root: str | Path) -> dict:
    run_name = Path(cfg["_config_path"]).stem
    run_dir = workspace_dir(cfg, output_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_copy(cfg["_config_path"], run_dir / "input.toml")
    common_dir = Path(cfg.get("_common_dir", run_dir))
    shared = cfg["_shared_common"]
    resolved_path = run_dir / "resolved.json"
    previous = read_json(resolved_path) if resolved_path.exists() else {}
    previous_chains = previous.get("chains", {})
    if cfg["runtime"]["auto_chains"] and not cfg.get("_force"):
        for kind in _renorm_types(cfg):
            widths = previous_chains.get(f"{kind}_mod{cfg['model']['mod']}", {})
            if widths:
                cfg["chains"].update(widths)
                break
    if shared:
        common = shared[0]
        pilot, two_plaq, one_plaq = common["pilot"], common["two_plaq"], common["one_plaq"]
        cfg["chains"].update(common["chains"])
    else:
        common_cfg = mod_config(cfg, 0) if float(cfg["model"]["beta1"]) == 0 else cfg
        common_cfg["_tuning_records"] = cfg["_tuning_records"]
        pilot = run_pilot(common_cfg, common_dir)
        geometry = pilot["geometry"]
        if cfg["runtime"]["auto_chains"]:
            from .chain_tuning import _workloads, tune_stage
            workloads = _workloads(common_cfg, pilot)
            one_path = common_dir / "step1_1plaq" / "result.json"
            identify_alpha = ("z" in _renorm_types(common_cfg) or float(common_cfg["model"]["alpha"]) != 0.0)
            if identify_alpha and (cfg.get("_force") or not one_path.exists() or
                    read_json(one_path).get("fingerprint") != _one_fit_key(common_cfg, geometry)):
                tune_stage(common_cfg, "one_plaq", workloads["one_plaq"])
        one_plaq = _run_one_plaq(common_cfg, common_dir, geometry)
        if cfg["runtime"]["auto_chains"]:
            two_path = common_dir / "step2_2plaq" / "result.json"
            if (cfg.get("_force") or not two_path.exists() or
                    read_json(two_path).get("fingerprint") != _two_fit_key(common_cfg, geometry, one_plaq)):
                tune_stage(common_cfg, "two_plaq", workloads["two_plaq"])
        two_plaq = _run_two_plaq(common_cfg, common_dir, geometry, one_plaq)
        common_chains = {key: common_cfg["chains"][key]
                         for key in ("pilot", "two_plaq", "one_plaq")}
        cfg["chains"].update(common_chains)
        common = {"pilot": pilot, "two_plaq": two_plaq, "one_plaq": one_plaq,
                  "chains": common_chains, "raw": common_cfg.pop("_two_raw", None),
                  "raw_source": "current_fit"}
        shared.append(common)
    geometry = pilot["geometry"]
    branches, chains = {}, {}
    for renorm_type in _renorm_types(cfg):
        branch_name = f"{renorm_type}_mod{cfg['model']['mod']}"
        branch_dir = run_dir / "branches" / branch_name
        cfg["_tuning_prefix"] = f"{branch_name}/"
        if cfg["runtime"]["auto_chains"] and not cfg.get("_force"):
            cfg["chains"].update({key: value for key, value in
                previous_chains.get(branch_name, {}).items()
                if key.startswith(("observable_", "topo_"))})
        cfg["_tuned_measurements"] = set()
        fit2 = two_plaq["fits"][renorm_type]["new"]
        beta_c, alpha1_c = float(fit2["beta"]), float(fit2["alpha1"])
        fit1 = one_plaq["fits"][renorm_type]
        alpha_1 = float(fit1["alpha"])
        step2 = (_run_beta1_scan(cfg, branch_dir, geometry, beta_c, alpha1_c,
                                renorm_type=renorm_type, alpha_c=alpha_1)
                 if cfg["renormalization"]["beta1_methods"] else {"matches": {}})
        unavailable = renorm_type == "U" and float(cfg["model"]["alpha"]) == 0.0
        if (not cfg["renormalization"]["alpha_topo_fit"] or
                not cfg["renormalization"]["beta1_methods"] or unavailable):
            topo = {"matches": {}}
        else:
            topo = _run_topo(cfg, branch_dir, geometry, beta_c, alpha1_c,
                             step2["matches"], alpha_1, renorm_type=renorm_type)
        _finish_topo(cfg, branch_dir, common_dir, geometry, common, topo, renorm_type, alpha_1)
        if topo.get("matches"):
            _save_topo(branch_dir, topo)
        branches[branch_name] = _branch_summary(
            cfg, run_name, renorm_type, geometry, beta_c, alpha1_c, alpha_1,
            step2, topo, fit2=fit2, fit1=fit1)
        old_alpha_eff = two_plaq["fits"][renorm_type].get("old", {}).get("alpha_eff")
        alpha_sum = alpha_1 + alpha1_c
        branches[branch_name]["two_plaq_comparison"] = {
            "old_alpha_eff": old_alpha_eff, "new_alpha_plus_alpha1": alpha_sum,
            "difference": None if old_alpha_eff is None else alpha_sum - old_alpha_eff}
        cleanup = list(step2.get("_cleanup_folders", [])) + list(topo.get("_cleanup_folders", []))
        _remove_scan_ensembles(branch_dir, cleanup)
        chains[branch_name] = dict(cfg["chains"])
    return {"geometry": geometry, "pilot": pilot, "branches": branches, "chains": chains}


def run_pipeline(cfg: dict, output_root: str | Path) -> dict:
    """Run only requested methods, sharing compatible type/mod measurements."""
    run_dir = workspace_dir(cfg, output_root)
    shared, records, results, branches, chains = [], {}, {}, {}, {}
    try:
        for mod in cfg["model"]["mod"]:
            branch_cfg = mod_config(cfg, mod)
            independent = float(cfg["model"]["beta1"]) != 0
            branch_cfg["_common_dir"] = str(run_dir / "common" / f"mod{mod}") if independent else str(run_dir)
            branch_cfg["_shared_common"] = [] if independent else shared
            branch_cfg["_tuning_records"] = records
            branch_cfg["_tuning_prefix"] = f"common_mod{mod}/" if independent else "common/"
            result = _run_mod_pipeline(branch_cfg, output_root)
            results[f"mod{mod}"] = result
            branches.update(result["branches"])
            chains.update(result["chains"])
        geometries = {name: item["geometry"] for name, item in results.items()}
        summary = {"model": cfg["model"], "renormalization": cfg["renormalization"],
                   "geometries": geometries, "branches": branches}
        resolved_path = run_dir / "resolved.json"
        resolved = read_json(resolved_path) if resolved_path.exists() else {}
        resolved.update(config={k: v for k, v in cfg.items() if not k.startswith("_")},
                        geometries=geometries, chains=chains)
        # Pilot details have a single owner: pilot/result.json.
        resolved.pop("pilots", None)
        resolved.pop("chains_source", None)
        atomic_json(resolved_path, resolved)
        atomic_json(run_dir / "summary.json", summary)
        if not cfg["storage"]["save_npz"]:
            common_dirs = ([run_dir / "common" / f"mod{mod}" for mod in cfg["model"]["mod"]]
                           if float(cfg["model"]["beta1"]) != 0 else [run_dir])
            for common_dir in common_dirs:
                (common_dir / "step2_2plaq" / "measurements.npz").unlink(missing_ok=True)
        shared.clear()
        _log(f"done: {run_dir / 'summary.json'}")
        return summary
    finally:
        if cfg["runtime"]["auto_chains"]:
            report_path = run_dir / "chain_tuning.json"
            previous_report = read_json(report_path) if report_path.exists() else {}
            atomic_json(report_path, {
                "device": str(resolve_device(cfg["runtime"]["device"])),
                "dtype": cfg["runtime"]["dtype"],
                "stages": {**previous_report.get("stages", {}), **records}})
