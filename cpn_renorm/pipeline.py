from __future__ import annotations

import json
import math
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import fingerprint, section
from .fit_client import extend_scan, isolated_fit, isolated_match, isolated_plot
from .observables import measure_ensemble
from .patches import PATCH_CACHE_SCHEMA, generate_one_plaq, generate_two_plaq
from .pilot import resolve_device, run_pilot, torch_dtype
from .progress import SimulationProgress, ensemble_summary
from .sampler import BatchedHMCSampler, Couplings
from .storage import atomic_json, atomic_npz, split_measurement


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
    accept_mean = float(np.mean(result["accept_rate"]))
    diag_cfg = cfg["diagnostics"]
    warnings = []
    if not diag_cfg["min_accept"] <= accept_mean <= diag_cfg["max_accept"]:
        cause = ("integration error is likely too large" if accept_mean < diag_cfg["min_accept"]
                 else "epsilon is likely too small and leapfrog cost excessive")
        warnings.append(f"HMC acceptance {accept_mean:.3f} outside configured range; {cause}")
    if result["sampling"]["stop_reason"] == "max_meas_total":
        warnings.append("maximum measurements reached before every primary observable met target ESS")
    if result["warmup"]["stop_reason"] == "max_warmup":
        warnings.append("maximum warmup reached before autocorrelation times stabilized")
    result["diagnostics"] = {"valid": not warnings, "warnings": warnings}
    if warnings and diag_cfg.get("strict"):
        raise RuntimeError(f"{tag} diagnostics failed: {'; '.join(warnings)}")
    meta, arrays = split_measurement(result)
    meta.update({"fingerprint": key, "parameters": payload, "cache": "generated"})
    atomic_npz(data_file, **arrays)
    atomic_json(meta_file, meta)
    _log(ensemble_summary(label, result))
    result.update({"fingerprint": key, "cache": "generated"})
    return result


def _scan_points(scan: dict) -> list[float]:
    if scan.get("points"):
        return sorted(set(float(x) for x in scan["points"]))
    lo, hi = float(scan["min"]), float(scan["max"])
    points = [x for x in (-0.1, 0.0, 0.1) if lo <= x <= hi]
    return points if len(points) >= 2 else [lo, 0.5 * (lo + hi), hi]


def _run_two_plaq(cfg: dict, run_dir: Path, geometry: dict) -> dict:
    result_path = run_dir / "step1_2plaq" / "result.json"
    if result_path.exists() and not cfg.get("_force"):
        cached = json.loads(result_path.read_text(encoding="utf-8"))
        if cached.get("cache_schema") == PATCH_CACHE_SCHEMA:
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
    result = {"step": "2plaq", "cache_schema": PATCH_CACHE_SCHEMA, **fit,
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


def _run_beta1_scan(cfg: dict, run_dir: Path, geometry: dict,
                    beta_c: float, alpha_eff_c: float) -> dict:
    _log("step 2/4: observable beta1 scan")
    settings, scan = section(cfg, "observable"), cfg["scan"]["beta1"]
    m, R, Lf, Lc = cfg["model"], geometry["factor"], geometry["L_fine"], geometry["L_coarse"]
    alpha_eff_f = float(m["alpha"] + m["alpha1"])
    fine = _measure_point(cfg, run_dir, settings, tag="obs_fine", L=Lf,
                          beta=float(m["beta"]), beta1=float(m["beta1"]),
                          alpha=alpha_eff_f, alpha1=0.0, kind="half")
    points, measured = _scan_points(scan), {}
    matches = {}
    refined = False
    for round_id in range(int(scan["max_rounds"]) + 2):
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
        xi_errs = np.array([float(r["xi_err"]) * R for r in ordered])
        matches = {"magsus": isolated_match(xs, magsus, fine["magsus"], magsus_err,
                                              fine["magsus_err"]),
                   "corr": isolated_match(xs, xis, fine["xi"], xi_errs,
                                           fine["xi_err"])}
        if all(r.get("bracketed") for r in matches.values()):
            mids = {0.5 * sum(r["raw_bracket"]) for r in matches.values()}
            missing = [x for x in mids if x not in measured]
            if missing and not refined:
                points = sorted(set(points) | set(missing))
                refined = True
                continue
            break
        if round_id == int(scan["max_rounds"]):
            break
        expanded = set(points)
        for result in matches.values():
            if not result.get("bracketed"):
                expanded.update(extend_scan(points, result,
                                (float(scan["min"]), float(scan["max"])),
                                float(scan["growth_factor"])))
        new_points = sorted(expanded)
        if new_points == points:
            break
        points = new_points
    result = {"step": "observable_beta1", "fine": {"xi": fine["xi"],
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
    xs = np.array(sorted(measured))
    isolated_plot(run_dir / "step2_observable" / "beta1_scan.png", x=xs,
                  series={"magsus_scaled": np.array([measured[x]["magsus"] * R ** 2 for x in xs]),
                          "xi_scaled": np.array([(measured[x]["xi"] * R
                                                  if measured[x].get("xi") is not None
                                                  else np.nan) for x in xs])},
                  targets={"fine_magsus": fine["magsus"], "fine_xi": fine["xi"]},
                  xlabel="beta1", ylabel="matched observables")
    atomic_json(run_dir / "step2_observable" / "result.json", result)
    return result


def _run_one_plaq(cfg: dict, run_dir: Path, geometry: dict) -> dict:
    result_path = run_dir / "step3a_1plaq" / "result.json"
    if result_path.exists() and not cfg.get("_force"):
        cached = json.loads(result_path.read_text(encoding="utf-8"))
        if cached.get("cache_schema") == PATCH_CACHE_SCHEMA:
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
    for method in ("magsus", "corr"):
        bmatch = beta1_matches[method]
        if not bmatch.get("bracketed"):
            all_results[method] = {"value": None, "reason": "beta1 was not bracketed"}
            continue
        beta1_c, points, measured = float(bmatch["value"]), _default_alpha_points(alpha_1plaq, scan), {}
        match = {}
        refined = False
        for round_id in range(int(scan["max_rounds"]) + 2):
            for alpha in points:
                if alpha not in measured:
                    measured[alpha] = _measure_point(
                        cfg, run_dir, settings, tag=f"topo_coarse_{method}", L=Lc,
                        beta=beta_c, beta1=beta1_c, alpha=alpha,
                        alpha1=alpha_eff_c - alpha, kind="full")
            xs = np.array(sorted(measured))
            ys = np.array([measured[x]["topology"]["Q_s"]["topo_sus"] for x in xs])
            errs = np.array([measured[x]["topology"]["Q_s"]["err"] for x in xs])
            match = isolated_match(xs, ys, target, errs, target_err)
            if match.get("bracketed"):
                midpoint = 0.5 * sum(match["raw_bracket"])
                if midpoint not in measured and not refined:
                    points = sorted(set(points) | {midpoint})
                    refined = True
                    continue
                break
            if round_id >= int(scan["max_rounds"]):
                break
            new_points = extend_scan(points, match, (float(scan["min"]),
                                     float(scan["max"])), float(scan["growth_factor"]))
            if new_points == points:
                break
            points = new_points
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
                      targets={f"fine_{fine_comp}_scaled": target},
                      xlabel="alpha", ylabel="topological susceptibility")
    result = {"step": "topo_alpha", "fine_component": fine_comp,
              "fine_target": target, "fine_target_err": target_err,
              "fine_tau": fine["topology"][fine_comp]["tau"],
              "fine_ess": fine["topology"][fine_comp]["ess"],
              "fine_sampling": fine["sampling"], "matches": all_results}
    atomic_json(run_dir / "step3b_topo" / "result.json", result)
    return result


def run_pipeline(cfg: dict, output_root: str | Path, *, skip_topo: bool = False) -> dict:
    run_name = Path(cfg["_config_path"]).stem
    clean_cfg = {key: value for key, value in cfg.items() if not key.startswith("_")}
    run_id = f"{run_name}-{fingerprint({'schema': 4, 'config': clean_cfg})}"
    run_dir = Path(output_root).resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _log(f"run {run_id} on {resolve_device(cfg['runtime']['device'])}")
    shutil.copy2(cfg["_config_path"], run_dir / "input.toml")
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
        alpha_1 = round(step3a["alpha"], 3)
        topo = ({"matches": {"magsus": {"value": None}, "corr": {"value": None}},
                 "skipped": "--skip-topo"} if skip_topo else
                _run_topo(cfg, run_dir, geometry, beta_c, alpha_eff_c,
                          step2["matches"], alpha_1))
    beta1 = {k: (round(v["value"], 3) if v.get("value") is not None else None)
             for k, v in step2["matches"].items()}
    atop = {k: (round(v.get("value"), 3) if v.get("value") is not None else None)
            for k, v in topo["matches"].items()}

    def combo(bkey: str, alpha: float | None) -> dict:
        return {"beta_c": beta_c, "beta1_c": beta1[bkey], "alpha_c": alpha,
                "alpha1_c": (round(alpha_eff_c - alpha, 3) if alpha is not None else None),
                "alpha_eff_c": alpha_eff_c}

    combos = {"magsus_obs_fit_topo": combo("magsus", atop["magsus"]),
              "corr_obs_fit_topo": combo("corr", atop["corr"]),
              "magsus_1plaq": combo("magsus", alpha_1),
              "corr_1plaq": combo("corr", alpha_1)}
    summary = {"run_name": run_name, "run_id": run_id,
               "created_at": datetime.now().isoformat(timespec="seconds"),
               "device": str(resolve_device(cfg["runtime"]["device"])),
               "geometry": geometry,
               "renormalized_couplings": {"beta_c": beta_c, "alpha_eff_c": alpha_eff_c,
                                            "beta1_c": beta1, "alpha_c_1plaq": alpha_1,
                                            "alpha_c_topo": atop},
               "four_combos": combos,
               "renorm_as_fine": {name: {"beta": x["beta_c"], "beta1": x["beta1_c"],
                                          "alpha": x["alpha_c"], "alpha1": x["alpha1_c"]}
                                  for name, x in combos.items()}}
    atomic_json(run_dir / "summary.json", summary)
    _log(f"done: {run_dir / 'summary.json'}")
    return summary
