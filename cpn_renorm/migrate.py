"""Migrate active run JSON caches without rerunning Monte Carlo sampling.

Usage: python -m cpn_renorm.migrate runs [--apply]
Historical runs under runs/old are intentionally excluded.
"""
from __future__ import annotations

import argparse
import re
from copy import deepcopy
from pathlib import Path
import shutil
import tempfile

import numpy as np

from .cache import (load_point, point_identity, read_json, store_point,
                    write_scan_points)
from .config import fingerprint, mod_config, section
from .patches import patch_result_key
from .pilot import _pilot_fingerprint
from .pipeline import (_method_key, _scan_fingerprint, _scan_points,
                       _raw_scan_match, _scan_decision, _censored_xi_G_bracket, _same_point)
from .fit_client import extend_scan
from .methods import BETA1_METHODS
from .storage import atomic_copy, atomic_json
from .workspace import config_fingerprint


def _branch_summary(cfg: dict, run_name: str, renorm_type: str, geometry: dict,
                    beta_c: float, alpha_eff_c: float, alpha_1: float | None,
                    step2: dict, topo: dict, *, fit2: dict | None = None,
                    fit1: dict | None = None) -> dict:
    """Serialize historical results only; never used by the active pipeline."""
    fit2, fit1 = fit2 or {}, fit1 or {}
    beta_methods = cfg["renormalization"]["beta1_methods"]
    alpha_methods = cfg["renormalization"].get("alpha_methods", ("1plaq", "topo"))
    results, fragments = {}, {}
    for beta_method in beta_methods or [None]:
        beta_match = step2.get("matches", {}).get(beta_method, {})
        for alpha_method in alpha_methods or [None]:
            if alpha_method == "1plaq":
                alpha, alpha_err = alpha_1, fit1.get("alpha_err")
            elif alpha_method == "topo":
                match = topo.get("matches", {}).get(beta_method, {})
                alpha, alpha_err = match.get("value"), match.get("error")
            else:
                alpha = alpha_err = None
            name = (f"{beta_method}_obs_fit_topo" if alpha_method == "topo" else
                    "_".join(x for x in (beta_method, alpha_method) if x) or "unfitted")
            couplings = {"beta": beta_c, "beta1": beta_match.get("value"),
                         "alpha": alpha,
                         "alpha1": alpha_eff_c - alpha if alpha is not None else None,
                         "alpha_eff": alpha_eff_c}
            errors = {"beta": fit2.get("beta_err"), "beta1": beta_match.get("error"),
                      "alpha": alpha_err, "alpha_eff": fit2.get("alpha_eff_err")}
            results[name] = {"couplings": couplings, "errors": errors}
            if all(couplings[key] is not None for key in ("beta", "beta1", "alpha", "alpha1")):
                fragments[name] = "\n".join(f"{key} = {couplings[key]!r}"
                                            for key in ("beta", "beta1", "alpha", "alpha1"))
    return {"results": results, "renorm_as_fine": fragments}


def _require(condition: bool, detail: object) -> None:
    if not condition:
        raise ValueError(f"migration verification failed: {detail}")


def strip_versions(value):
    if isinstance(value, dict):
        return {key: strip_versions(item) for key, item in value.items()
                if key not in ("schema", "cache_schema")}
    if isinstance(value, list):
        return [strip_versions(item) for item in value]
    return value


def _rename_legacy_length(value):
    """Rename old second-moment length identifiers without touching other words."""
    def label(text: str) -> str:
        return re.sub(r"(?<![A-Za-z0-9])xi(?![A-Za-z0-9]|_G(?:_|\b))", "xi_G", text)
    if isinstance(value, dict):
        return {label(key): _rename_legacy_length(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_rename_legacy_length(item) for item in value]
    return label(value) if isinstance(value, str) else value


def update_toml(path: Path, beta1: list[str]) -> None:
    text = _rename_legacy_length(path.read_text(encoding="utf-8"))
    text = re.sub(r"(?m)^\s*alpha_methods\s*=.*\n?", "", text)
    if "beta1_methods" not in text:
        text = text.replace("[renormalization]\n", "[renormalization]\n"
                            f"beta1_methods = {beta1!r}\n".replace("'", '"'))
    if "alpha_topo_fit" not in text:
        text = text.replace("[renormalization]\n", "[renormalization]\nalpha_topo_fit = true\n")
    if "[storage]" not in text:
        text += "\n[storage]\nsave_npz = false\n"
    path.write_text(text, encoding="utf-8")


def _geometry(cfg: dict, root: Path, mod: int) -> tuple[Path, dict]:
    common = root / "common" / f"mod{mod}" if float(cfg["model"]["beta1"]) != 0 else root
    return common, read_json(common / "pilot" / "result.json")["geometry"]


def _replay_search(scan: dict, initial: list[float], measured: dict,
                   targets: dict, matches: dict) -> set[float]:
    """Replay the old shared search using stored values, without fitting/sampling."""
    active, coarse = {}, set()
    counts = {name: [0, 0] for name in targets}

    def add(value: float, phase: str) -> None:
        key = next((x for x in measured if abs(x - value) <= 1e-12), None)
        _require(key is not None, f"missing historical scan point {value}")
        active[key] = measured[key]
        if phase == "coarse":
            coarse.add(key)

    for value in initial:
        add(value, "coarse")
    for _ in range(2 * int(scan["max_rounds"]) + 2 * int(scan["max_refine_rounds"]) + 8):
        xs = np.array(sorted(active))
        additions = {}
        for name, (observable, target, use_censoring) in targets.items():
            ys = np.array([observable(active[x]) for x in xs])
            raw = _raw_scan_match(xs, ys, target)
            censored = _censored_xi_G_bracket(xs, ys, target) if use_censoring else None
            candidate, reason = _scan_decision(raw,
                tolerance=float(scan["refine_tolerance"]),
                max_refine=int(scan["max_refine_rounds"]),
                refine_count=counts[name][1], censored=censored)
            if candidate is not None and not _same_point(candidate, set(active)):
                additions.setdefault(candidate, "refine")
                counts[name][1] += 1
                continue
            if reason != "unbracketed" or counts[name][0] >= int(scan["max_rounds"]):
                continue
            expanded = extend_scan(sorted(active), raw,
                (float(scan["min"]), float(scan["max"])), float(scan["growth_factor"]))
            new = [x for x in expanded if not _same_point(x, set(active))]
            if new:
                counts[name][0] += 1
            for value in new:
                additions[value] = "coarse"
        if not additions:
            break
        for value, phase in additions.items():
            add(value, phase)
    for name, (expansion, refinement) in counts.items():
        for key, value in (("expansion_rounds", expansion), ("refinement_rounds", refinement)):
            if key in matches[name]:
                _require(matches[name][key] == value, (name, key, value, matches[name][key]))
    return coarse


def _legacy_alpha_points(center: float, scan: dict) -> list[float]:
    if scan.get("points"):
        return _scan_points(scan)
    lo, hi = float(scan["min"]), float(scan["max"])
    values = center * np.linspace(.8, 1.2, 5) if center > 0 else np.linspace(0., min(.4, hi), 5)
    points = sorted(set(max(lo, min(hi, float(x))) for x in values))
    if len(points) < 2:
        points = sorted(set(float(x) for x in np.linspace(lo, min(hi, max(lo * 1.5, lo + .001)), 5)))
    return points


def _refresh_copy(root: Path) -> dict:
    """Correct over-retained historical points and rebuild the final summary."""
    from .cache import scan_measurements
    resolved = read_json(root / "resolved.json")
    cfg = resolved["config"]
    report_path = root / "migration_report.json"
    report = read_json(report_path) if report_path.exists() else {"run": root.name}
    stats, summaries, candidates, retained_paths = {}, {}, set(), set()
    aliases = set()
    rekeyed = 0

    def classify(branch: Path, stage: dict, coarse: set[float], matches: dict,
                 parameters: dict, variable: str) -> dict:
        nonlocal rekeyed
        entries = stage.get("measurements", {})
        final = {float(x) for match in matches.values() for x in match.get("fit_points", [])}
        counts = {"coarse_points": 0, "final_points": 0, "retained_points": 0, "discarded_points": 0}
        kept = {}
        for value, entry in entries.items():
            x = float(value)
            phases = []
            if _same_point(x, coarse):
                phases.append("coarse")
                counts["coarse_points"] += 1
            if _same_point(x, final):
                phases.append("final")
                counts["final_points"] += 1
            folder = (branch / entry["path"]).parent
            if phases:
                data = read_json(folder / "result.json")
                expected = {**data["parameters"], **parameters, variable: x}
                if variable == "alpha":
                    expected["alpha1"] = fit2["alpha_eff"] - x
                if data["parameters"] != expected:
                    context = read_json(root / "resolved.json")["contexts"][data["context"]]
                    payload = {**context, **expected}
                    key, _, canonical = point_identity(payload)
                    destination = branch / "ensembles" / f"{canonical['tag']}_{key}"
                    if (destination / "result.json").exists():
                        actual = read_json(destination / "result.json")
                        for field in ("xi_G", "xi_G_err", "magsus", "magsus_err"):
                            if field in data:
                                _require(actual.get(field) == data[field], (folder, field))
                        if "topology" in data:
                            _require(all(actual["topology"]["Q_s"].get(k) == v
                                for k, v in data["topology"]["Q_s"].items()), folder)
                    else:
                        _require("warmup" not in data, f"cannot relabel original measurement {folder}")
                        store_point(scalar, branch, payload, data)
                    aliases.add(folder)
                    folder = destination
                    entry = {**entry, "path": str((folder / "result.json").relative_to(branch))}
                    rekeyed += 1
                kept[value] = {**entry, "phases": phases}
                retained_paths.add(folder)
                counts["retained_points"] += 1
            else:
                candidates.add(folder)
                counts["discarded_points"] += 1
        stage["measurements"] = kept
        return counts

    for branch in sorted(root.glob("branches/*")):
        if not branch.is_dir():
            continue
        kind, mod = branch.name.split("_mod")
        scalar = mod_config(cfg, int(mod))
        common, geometry = _geometry(cfg, root, int(mod))
        fit2 = read_json(common / "step1_2plaq/result.json")["fits"][kind]
        one_path = common / "step3a_1plaq/result.json"
        fit1 = read_json(one_path).get("fits", {}).get(kind, {}) if one_path.exists() else {}
        obs_path = branch / "step2_observable/result.json"
        obs = read_json(obs_path) if obs_path.exists() else {"matches": {}}
        branch_stats = {}
        if obs.get("measurements"):
            measured, phases = scan_measurements(branch, obs)
            if any("historical" in labels for labels in phases.values()):
                fine = read_json(branch / obs["fine_path"])
                R = geometry["factor"]
                targets = {}
                for method in obs["matches"]:
                    definition = BETA1_METHODS[method]
                    field, scale = definition["field"], R ** definition["power"]
                    targets[method] = (lambda r, f=field, s=scale:
                        float(r[f]) * s if r.get(f) is not None else np.nan,
                        fine[field], field == "xi_G")
                coarse = _replay_search(cfg["scan"]["beta1"], _scan_points(cfg["scan"]["beta1"]),
                                        measured, targets, obs["matches"])
            else:
                coarse = {x for x, labels in phases.items() if "coarse" in labels}
            branch_stats["obs"] = classify(branch, obs, coarse, obs["matches"],
                {"beta": fit2["beta"], "alpha": fit2["alpha_eff"], "alpha1": 0.0}, "beta1")
            atomic_json(obs_path, obs)
        topo_path = branch / "step3b_topo/result.json"
        topo = read_json(topo_path) if topo_path.exists() else {"matches": {}}
        if topo.get("fine_path"):
            fine = read_json(branch / topo["fine_path"])
            target = fine["topology"][topo["fine_component"]]["topo_sus"] * geometry["factor"] ** 2
            for name, match in topo["matches"].items():
                if not match.get("measurements"):
                    continue
                measured, phases = scan_measurements(branch, match)
                if any("historical" in labels for labels in phases.values()):
                    initial = _legacy_alpha_points(float(fit1["alpha"]), cfg["scan"]["alpha"])
                    coarse = _replay_search(cfg["scan"]["alpha"], initial, measured,
                        {name: (lambda r: r["topology"]["Q_s"]["topo_sus"], target, False)}, {name: match})
                else:
                    coarse = {x for x, labels in phases.items() if "coarse" in labels}
                branch_stats[f"topo_{name}"] = classify(branch, match, coarse, {name: match},
                    {"beta": fit2["beta"], "beta1": match["beta1_c"]}, "alpha")
            atomic_json(topo_path, topo)
        summaries[branch.name] = _branch_summary(scalar, root.name, kind, geometry,
            fit2["beta"], fit2["alpha_eff"], fit1.get("alpha"), obs, topo, fit2=fit2, fit1=fit1)
        stats[branch.name] = branch_stats
    summary = read_json(root / "summary.json")
    summary["branches"] = summaries
    atomic_json(root / "summary.json", summary)
    # All changed references must be valid before deleting discarded points.
    verify_run(root)
    removed = candidates - retained_paths
    for folder in (candidates | aliases) - retained_paths:
        shutil.rmtree(folder)
    report.pop("historical_point_phases", None)
    if "previous_recovered_scalar_points" not in report:
        report["previous_recovered_scalar_points"] = report.get("recovered_scalar_points", 0)
    for branch, stages in stats.items():
        for name, counts in stages.items():
            counts["discarded_points"] += report.get("scans", {}).get(branch, {}).get(name, {}).get("discarded_points", 0)
    report["scans"] = stats
    report["rekeyed_point_caches"] = report.get("rekeyed_point_caches", 0) + rekeyed
    # Only coarse points outside the final fits needed recovery from the old JSON.
    recovered = set()
    for branch in root.glob("branches/*"):
        for path in (branch / "step2_observable/result.json", branch / "step3b_topo/result.json"):
            if not path.exists():
                continue
            data = read_json(path)
            stages = [data] if "measurements" in data else list(data.get("matches", {}).values())
            for stage in stages:
                for entry in stage.get("measurements", {}).values():
                    if entry["phases"] == ["coarse"]:
                        recovered.add(str(branch / entry["path"]))
    report["recovered_scalar_points"] = len(recovered)
    report["retention"] = "coarse search and points actually used in final fits only"
    report["removed_excess_points"] = report.get("removed_excess_points", 0) + len(removed)
    atomic_json(report_path, report)
    return report


def _rename_legacy_corr(value):
    """Translate the old second-moment method label only in legacy snapshots."""
    def label(text: str) -> str:
        if text == "corr":
            return "corrG"
        text = re.sub(r"corr(?=_(?:1plaq|obs_fit_topo)(?:$|[^a-zA-Z0-9]))", "corrG", text)
        return re.sub(r"(topo_coarse_|topo_|beta1_scan_|alpha_scan_)corr(?=$|[_.])",
                      lambda match: match[1] + "corrG", text)
    if isinstance(value, dict):
        return {label(key): _rename_legacy_corr(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_rename_legacy_corr(item) for item in value]
    return label(value) if isinstance(value, str) else value


def _migrate_copy(root: Path) -> dict:
    """Convert one active workspace; verify all references before removing arrays."""
    report_path = root / "migration_report.json"
    old = read_json(root / "summary.json")
    if "renormalization" not in old:
        for path in root.rglob("*.json"):
            data = read_json(path)
            renamed = _rename_legacy_length(_rename_legacy_corr(data))
            if renamed != data:
                atomic_json(path, renamed)
        old = read_json(root / "summary.json")
    if "renormalization" in old and all("beta1" in b for b in old["branches"].values()):
        # Resume the verified cleanup phase if an earlier migration was interrupted.
        report = read_json(report_path) if report_path.exists() else {"run": root.name}
        verify_run(root)
        removed = list(root.rglob("*.npz"))
        for path in removed:
            path.unlink()
        return report
    state = read_json(root / "run_state.json")
    resolved = read_json(root / "resolved.json")
    cfg = strip_versions(deepcopy(state.get("config", resolved["config"])))
    beta1 = [name for name in ("magsus", "corrG")
             if any(name in branch["renormalized_couplings"].get("beta1_c", {})
                    for branch in old["branches"].values())]
    cfg["renormalization"].pop("alpha_methods", None)
    cfg["renormalization"].setdefault("alpha_topo_fit", True)
    cfg["renormalization"].update(beta1_methods=beta1)
    cfg["storage"] = {"save_npz": False}
    cfg["_config_path"] = state.get("source", str(root / "input.toml"))
    chains = resolved.get("chains", old.get("chains", {}))
    resolved = {"config": {k: v for k, v in cfg.items() if not k.startswith("_")},
                "chains": chains, "geometries": old["geometries"],
                "contexts": resolved.get("contexts", {})}
    atomic_json(root / "resolved.json", resolved)
    source_points = list(root.glob("branches/*/ensembles/*/result.json"))
    old_folders = []
    payloads = {}
    for path in source_points:
        data = strip_versions(read_json(path))
        payload = data.get("parameters", {})
        if "settings" not in payload:
            continue
        payloads.setdefault((path.parents[2].name, payload["tag"]), []).append(payload)
        arrays_path = path.parent / "data.npz"
        if arrays_path.exists():
            with np.load(arrays_path, allow_pickle=False) as arrays:
                for name in ("accept_rate", "s_accept_rate"):
                    if name in arrays and arrays[name].size:
                        data[name] = float(np.mean(arrays[name]))
        destination = store_point(cfg, path.parents[2], payload, data)
        if destination != path.parent:
            old_folders.append(path.parent)
    def select_payload(branch_name: str, tag: str, expected: dict) -> dict:
        candidates = payloads.get((branch_name, tag), [])
        for payload in candidates:
            if all(payload.get(key) == value for key, value in expected.items()):
                return payload
        raise ValueError(f"no matching measurement context: {branch_name}/{tag}: {expected}")

    summaries = {}
    checked_common = set()
    recovered = 0
    for branch_name, previous_summary in old["branches"].items():
        branch = root / "branches" / branch_name
        kind, mod_string = branch_name.split("_mod")
        scalar = mod_config(cfg, int(mod_string))
        scalar["chains"].update(chains.get(branch_name, {}))
        common, geometry = _geometry(cfg, root, int(mod_string))
        common_cfg = mod_config(scalar, 0) if float(cfg["model"]["beta1"]) == 0 else scalar
        if common not in checked_common:
            pilot_path = common / "pilot" / "result.json"
            pilot = strip_versions(read_json(pilot_path))
            pilot["fingerprint"] = _pilot_fingerprint(common_cfg)
            atomic_json(pilot_path, pilot)
            for stage, folder in (("two_plaq", "step1_2plaq"), ("one_plaq", "step3a_1plaq")):
                path = common / folder / "result.json"
                if path.exists():
                    data = strip_versions(read_json(path))
                    data["fingerprint"] = patch_result_key(common_cfg, geometry, stage)
                    atomic_json(path, data)
            checked_common.add(common)
        fit2 = read_json(common / "step1_2plaq" / "result.json")["fits"][kind]
        one_path = common / "step3a_1plaq" / "result.json"
        fit1 = read_json(one_path)["fits"].get(kind, {}) if one_path.exists() else {}
        beta_c, alpha_eff = fit2["beta"], fit2["alpha_eff"]
        obs_path = branch / "step2_observable" / "result.json"
        obs = strip_versions(read_json(obs_path))
        measured, phases = {}, {}
        obs_payload = select_payload(branch_name, "obs_coarse",
            {"beta": beta_c, "alpha": alpha_eff, "alpha1": 0.0})
        if obs_payload is None:
            raise ValueError(f"cannot infer observation context in {branch}")
        for value, observation in obs.get("coarse", {}).items():
            payload = {**obs_payload, "beta1": float(value)}
            key, _, context = point_identity(payload)
            folder = branch / "ensembles" / f"{context['tag']}_{key}"
            if not (folder / "result.json").exists():
                store_point(scalar, branch, payload, observation)
                recovered += 1
            measured[float(value)] = load_point(folder)
            phases[float(value)] = {"historical"}
        for match in obs["matches"].values():
            for value in match.get("fit_grid", match.get("fit_points", [])):
                phases.setdefault(float(value), set()).add("final")
        obs["measurements"] = write_scan_points(branch, measured, phases, tag="obs_coarse")
        fine_payload = select_payload(branch_name, "obs_fine",
            {"beta": cfg["model"]["beta"], "beta1": cfg["model"]["beta1"],
             "alpha": cfg["model"]["alpha"] + cfg["model"]["alpha1"]})
        key, _, context = point_identity(fine_payload)
        obs["fine_path"] = f"ensembles/{context['tag']}_{key}/result.json"
        obs["measurement_key"] = _scan_fingerprint("beta1", scalar, geometry,
            renorm_type=kind, beta_c=beta_c, alpha_eff_c=alpha_eff)
        obs["method_fingerprints"] = {name: _method_key(scalar, geometry,
            method=name, kind="beta1", renorm_type=kind, beta_c=beta_c,
            alpha_eff_c=alpha_eff) for name in obs["matches"]}
        for name in ("coarse", "fine", "points", "fingerprint"):
            obs.pop(name, None)
        atomic_json(obs_path, obs)
        topo_path = branch / "step3b_topo" / "result.json"
        topo = strip_versions(read_json(topo_path)) if topo_path.exists() else {"matches": {}}
        if topo_path.exists():
            fine_payload = select_payload(branch_name, "topo_fine",
                {"beta": cfg["model"]["beta"], "beta1": cfg["model"]["beta1"]})
            key, _, context = point_identity(fine_payload)
            topo["fine_path"] = f"ensembles/{context['tag']}_{key}/result.json"
            fine_parameters = {key: fine_payload[key] for key in
                               ("tag", "L", "beta", "beta1", "kind", "alpha", "alpha1")}
            topo["fine_key"] = fingerprint({"parameters": fine_parameters,
                "settings": section(scalar, "topo"), "runtime": scalar["runtime"],
                "N": scalar["model"]["N"], "mod": scalar["model"]["mod"]})
            topo["method_fingerprints"] = {}
            for method, match in topo["matches"].items():
                if match.get("beta1_c") is None:
                    continue
                payload = select_payload(branch_name, f"topo_coarse_{method}",
                    {"beta": beta_c, "beta1": match["beta1_c"]})
                if payload is None:
                    raise ValueError(f"cannot infer topology context in {branch}/{method}")
                measured, phases = {}, {}
                for value, observation in match.get("measurements", {}).items():
                    point_payload = {**payload, "alpha": float(value), "alpha1": alpha_eff - float(value)}
                    key, _, context = point_identity(point_payload)
                    folder = branch / "ensembles" / f"{context['tag']}_{key}"
                    if not (folder / "result.json").exists():
                        store_point(scalar, branch, point_payload, {
                            "topology": {"Q_s": {"topo_sus": observation["value"],
                                "err": observation["error"], "tau": observation["tau"],
                                "ess": observation["ess"]}}, "sampling": observation["sampling"]})
                        recovered += 1
                    measured[float(value)] = load_point(folder)
                    phases[float(value)] = {"historical"}
                for value in match.get("fit_grid", match.get("fit_points", [])):
                    phases.setdefault(float(value), set()).add("final")
                match["measurements"] = write_scan_points(branch, measured, phases, tag=f"topo_coarse_{method}")
                for name in ("points", "target", "target_err"):
                    match.pop(name, None)
                topo["method_fingerprints"][method] = _method_key(scalar, geometry,
                    method=method, kind="alpha", renorm_type=kind, beta_c=beta_c,
                    alpha_eff_c=alpha_eff, beta1_c=match["beta1_c"])
            for name in ("fingerprint", "fine_target", "fine_target_err", "fine_tau", "fine_ess", "fine_sampling"):
                topo.pop(name, None)
            atomic_json(topo_path, topo)
        summaries[branch_name] = _branch_summary(scalar, root.name, kind, geometry,
            beta_c, alpha_eff, fit1.get("alpha"), obs, topo, fit2=fit2, fit1=fit1)
        original = previous_summary["renormalized_couplings"]
        _require(fit2["beta"] == original["beta_c"], branch_name)
        _require(fit2.get("beta_err") == original.get("beta_c_err"), branch_name)
        _require(fit2.get("alpha_eff_err") == original.get("alpha_eff_c_err"), branch_name)
        _require(fit2["alpha_eff"] == original["alpha_eff_c"], branch_name)
        for method, match in obs["matches"].items():
            _require(match.get("value") == original["beta1_c"][method], (branch_name, method))
            _require(match.get("error") == original["beta1_c_err"][method], (branch_name, method))
        _require(fit1.get("alpha") == original["alpha_c_1plaq"], branch_name)
        _require(fit1.get("alpha_err") == original.get("alpha_c_1plaq_err"), branch_name)
        for method, match in topo["matches"].items():
            _require(match.get("value") == original["alpha_c_topo"][method], (branch_name, method))
            _require(match.get("error") == original["alpha_c_topo_err"][method], (branch_name, method))
    summary = {"model": cfg["model"], "renormalization": cfg["renormalization"],
               "geometries": old["geometries"], "branches": summaries}
    atomic_json(root / "summary.json", summary)
    state = {key: value for key, value in state.items() if key not in ("config", "schema")}
    state["config_fingerprint"] = config_fingerprint(cfg)
    atomic_json(root / "run_state.json", state)
    for path in root.rglob("*.toml"):
        update_toml(path, beta1)
    report = {"run": root.name, "recovered_scalar_points": recovered,
              "historical_point_phases": "unknown; retained as historical",
              "removed_npz": len(list(root.rglob("*.npz"))),
              "removed_branch_summaries": len(list(root.glob("branches/*/summary.json")))}
    atomic_json(report_path, report)
    verify_run(root)
    # Destructive cleanup is last, after all scalar results and references pass.
    for path in root.rglob("*.npz"):
        path.unlink()
    for path in old_folders:
        if path.exists():
            shutil.rmtree(path)
    for path in root.glob("branches/*/summary.json"):
        path.unlink()
    tuning_path = root / "chain_tuning.json"
    if tuning_path.exists():
        tuning = strip_versions(read_json(tuning_path))
        tuning.pop("chains", None)
        atomic_json(tuning_path, tuning)
    return _refresh_copy(root)


def verify_run(root: Path) -> None:
    contexts = read_json(root / "resolved.json").get("contexts", {})
    for path in root.rglob("*.json"):
        data = read_json(path)
        def check(value):
            if isinstance(value, dict):
                if "context" in value:
                    _require(value["context"] in contexts, path)
                if "path" in value and "phases" in value:
                    _require((path.parents[1] / value["path"]).is_file(), (path, value))
                if "fine_path" in value:
                    _require((path.parents[1] / value["fine_path"]).is_file(), path)
                for child in value.values():
                    check(child)
            elif isinstance(value, list):
                for child in value:
                    check(child)
        check(data)


def migrate_run(root: Path) -> dict:
    """Stage and validate a full conversion before publishing or deleting data."""
    root = root.resolve()
    pending = root / ".migration_pending.json"
    if pending.exists():
        staging = Path(read_json(pending)["staging"])
    else:
        if "renormalization" in read_json(root / "summary.json"):
            verify_run(root)
            report_path = root / "migration_report.json"
            report = read_json(report_path) if report_path.exists() else {}
            if ("rekeyed_point_caches" in report and
                    report.get("retention") == "coarse search and points actually used in final fits only") and all(
                    "results" in b for b in read_json(root / "summary.json")["branches"].values()):
                return report
        staging = Path(tempfile.mkdtemp(prefix="cpn-migrate-")) / root.name
        try:
            shutil.copytree(root, staging)
            if "renormalization" in read_json(staging / "summary.json"):
                _refresh_copy(staging)
            else:
                _migrate_copy(staging)
            verify_run(staging)
        except Exception:
            shutil.rmtree(staging.parent)
            raise
        atomic_json(pending, {"staging": str(staging)})
    files = [path for path in staging.rglob("*") if path.is_file()]
    desired = {path.relative_to(staging) for path in files}
    # Keep the old summary until all dependent JSON and TOML are published.
    files.sort(key=lambda path: path == staging / "summary.json")
    for path in files:
        atomic_copy(path, root / path.relative_to(staging))
    verify_run(root)
    for path in list(root.rglob("*")):
        if (path.is_file() and path != pending and
                path.relative_to(root) not in desired and path.suffix in (".json", ".npz")):
            path.unlink()
    for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()
    pending.unlink()
    report = read_json(root / "migration_report.json")
    shutil.rmtree(staging.parent)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    for root in sorted(args.root.iterdir()):
        if root.name == "old" or not (root / "summary.json").exists():
            continue
        print(migrate_run(root.resolve()) if args.apply else {
            "run": root.name, "npz_to_remove": len(list(root.rglob("*.npz")))})


if __name__ == "__main__":
    main()
