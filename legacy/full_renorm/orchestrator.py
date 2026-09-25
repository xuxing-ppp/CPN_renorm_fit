#!/usr/bin/env python3
"""Full renormalization orchestrator (new pipeline).

Chains four fits for a given fine model (beta_f, beta1_f, alpha_f, alpha1_f):

    step 1   CPN_2plaq_fit             -> beta_c , alpha_eff_c (= alpha_c + alpha1_c)
    step 2   CPN_observable_fit        -> beta1_c  {magsus, corr}   (halfRefVil)
    step 3a  CPN_1plaq_fit             -> alpha_c  (1plaq vortex fit)
    step 3b  CPN_observable_fit        -> alpha_c  {magsus, corr}   (topo-sus match,
                                           coarse alpha-scan per beta1)

beta1 comes from step 2 (two estimates: magsus, corr). alpha comes from EITHER
step 3a (1plaq) OR step 3b (topo, run per beta1). The final deliverable is the
cross product -- FOUR renormalization combos:

    magsus & obs_fit_topo      corr & obs_fit_topo
    magsus & 1plaq             corr & 1plaq

ONE `renorm_type` ("U"|"z", default "z") is applied consistently across all steps
(2plaq conn_type, 1plaq vortex tag, fine topo component). The action family is
fixed by the pipeline (halfRefVil for the beta1 obs fit, RefVil for the coarse
topo alpha-scan), so unlike full_renorm_original there is no results_{type}/ split.

Alpha-less mode: with renorm_type="U" and alpha_f~=0 the fine integer s field
(1plaq s-vortex, topo Q_s) is unconstrained, so steps 3a/3b are skipped and
alpha_c is FIXED to 0.0 (alpha1_c = alpha_eff_c) in every combo; the summary
then carries beta_c, beta1_c {magsus, corr} and alpha_eff_c.

Each step runs as an isolated subprocess with cwd = its subproject, because the
subprojects ship colliding top-level `func` packages whose data-generators spawn
ProcessPoolExecutor workers (macOS `spawn` re-imports __main__). The orchestrator
itself never imports any subproject -- it only shells out.

Usage:
    python orchestrator.py <config.json>
"""

import json
import math
import os
import subprocess
import sys
from datetime import datetime

import step_common

_HERE = os.path.dirname(os.path.abspath(__file__))   # .../full_renorm
_BASE = os.path.dirname(_HERE)                        # .../CPN_renorm_fit_mod

_STEP_LAUNCH = [
    # (step_key, step script, subproject dir)
    ("2plaq",     "step1_2plaq.py",   "CPN_2plaq_fit"),
    ("obs_beta1", "step2_obs_beta1.py", "CPN_observable_fit"),
    ("1plaq",     "step3a_1plaq.py",  "CPN_1plaq_fit"),
    ("topo",      "step3b_topo.py",   "CPN_observable_fit"),
]
_SCRIPT_FOR = {k: (os.path.join(_HERE, s), os.path.join(_BASE, d))
               for k, s, d in _STEP_LAUNCH}

_CONFIGS_ROOT = os.path.join(_HERE, "configs")


def _resolve_config_path(config_name):
    """Resolve a config name to an absolute path.

    Accepts either an existing path (returned as-is) or a bare filename, which is
    searched for recursively under configs/ (top level + every subfolder), so
    configs can live in N{N}_L{L}_{renorm_type}_mod{mod}/ subfolders. Raises
    FileNotFoundError if not found, ValueError if ambiguous across subfolders.
    """
    if os.path.isfile(config_name):
        return os.path.abspath(config_name)
    name = os.path.basename(config_name)
    matches = []
    for dirpath, _dirs, files in os.walk(_CONFIGS_ROOT):
        if name in files:
            matches.append(os.path.join(dirpath, name))
    if not matches:
        raise FileNotFoundError(
            f"Config {name!r} not found under {_CONFIGS_ROOT} "
            f"(searched top level and all subfolders)")
    if len(matches) > 1:
        raise ValueError(
            f"Config name {name!r} is ambiguous across subfolders; matches: {matches}")
    return matches[0]


def _collect_all_configs():
    """Every .json under configs/, recursively across all subfolders, sorted."""
    all_configs = []
    for dirpath, _dirs, files in os.walk(_CONFIGS_ROOT):
        for fn in files:
            if fn.endswith(".json"):
                all_configs.append(os.path.join(dirpath, fn))
    return sorted(all_configs)


def _run_step(step_key, step_config_path):
    script, subproject_dir = _SCRIPT_FOR[step_key]
    cmd = [sys.executable, script, step_config_path]
    print(f"\n[orchestrator] {' '.join(cmd)}\n[orchestrator]   cwd={subproject_dir}")
    # Inherit stdio so the subprocess's tqdm/print stream live. check=True raises
    # CalledProcessError on non-zero exit -- a hard failure (no partial summary).
    subprocess.run(cmd, cwd=subproject_dir, check=True)


def _validate(config):
    renorm_type = config.get("renorm_type", "z")
    if renorm_type not in ("U", "z"):
        raise ValueError(f"renorm_type must be 'U' or 'z', got {renorm_type!r}")
    if config.get("mod") not in (0, 1):
        raise ValueError(f"mod must be 0 or 1, got {config.get('mod')!r}")
    return renorm_type


def _round_opt(x, nd=3):
    return None if x is None else round(x, nd)


def run_full_renorm(config_path, n_workers, auto_regenerate, do_topo=True):
    config = step_common.load_json(config_path)

    # Stamp n_workers onto every block (overrides per-block values).
    for block in ("2plaq", "1plaq", "observable", "topo"):
        config.setdefault(block, {})["n_workers"] = n_workers
    # Record the data-reuse policy (flows into each step config via `shared`,
    # so step1/step3a can pass it to decide_data_action). True: overwrite stale
    # data files with fresh MC; False: raise on a mismatched/legacy file.
    config["auto_regenerate"] = auto_regenerate

    renorm_type = _validate(config)
    N, mod = config["N"], config["mod"]
    L, L_coarse = config["L"], config["L_coarse"]

    # results/ classified by (N, L, renorm_type, mod), config stem as leaf folder.
    run_name = os.path.splitext(os.path.basename(config_path))[0]
    class_dir = f"N{N}_L{L}_{renorm_type}_mod{mod}"
    results_dir = os.path.join(_HERE, "results", class_dir, run_name)
    os.makedirs(results_dir, exist_ok=True)
    step_common.dump_json(config, os.path.join(results_dir, "config.json"))

    fine = config["fine"]
    # 3-decimal rounding throughout (matches the .3f in all path templates).
    beta_f, beta1_f = round(fine["beta_f"], 3), round(fine["beta1_f"], 3)
    alpha_f, alpha1_f = round(fine["alpha_f"], 3), round(fine["alpha1_f"], 3)
    alpha_eff_f = round(alpha_f + alpha1_f, 3)

    # U-renorm reads alpha from the fine integer s field (1plaq s-vortex, topo Q_s).
    # s is constrained ONLY by alpha_f: the alpha1*cos(da + 2*pi*s) term is
    # s-invariant (cos(da+2*pi*s) = cos(da) for integer s), so with alpha_f ~= 0
    # the fine s field is unconstrained -- vs ~= 0 in run_Original, and a random
    # walk in the RefVil sampler -- making BOTH alpha methods degenerate,
    # regardless of beta1_f. Steps 1-2 (beta, beta1, alpha_eff renormalization)
    # remain valid, so instead of erroring we skip 3a/3b and FIX alpha_c = 0
    # (alpha1_c = alpha_eff_c) in the summary. (Also covers the case that would
    # otherwise crash inside step3b_topo's own alpha_f guard.)
    skip_alpha_fits = renorm_type == "U" and abs(alpha_f) < 1e-10
    if skip_alpha_fits:
        print("[orchestrator] U-renorm with alpha_f~=0: fine s-field unconstrained "
              "-> running steps 1-2 only (beta, beta1, alpha_eff); 1plaq and topo "
              "alpha fits skipped, alpha_c fixed to 0.0 (alpha1_c=alpha_eff_c)")
    BC, pad = config["boundary"]["BC"], config["boundary"]["pad"]

    shared = {"base_dir": _BASE, "N": N, "mod": mod, "renorm_type": renorm_type,
              "L": L, "BC": BC, "pad": pad, "auto_regenerate": auto_regenerate}

    def dump(cfg, name):
        path = os.path.join(results_dir, name)
        step_common.dump_json(cfg, path)
        return path

    # ---- Step 1: 2plaq -> beta_c, alpha_eff_c --------------------------------
    s1 = {**shared, "beta_f": beta_f, "beta1_f": beta1_f, "alpha_eff_f": alpha_eff_f,
          "2plaq": config["2plaq"],
          "_result_path": os.path.join(results_dir, "step1_result.json")}
    _run_step("2plaq", dump(s1, "step1_config.json"))
    r1 = step_common.load_json(s1["_result_path"])
    beta_c, alpha_eff_c = round(r1["beta_c"], 3), round(r1["alpha_eff_c"], 3)
    beta_c_err, alpha_eff_c_err = r1["beta_c_err"], r1["alpha_eff_c_err"]

    # ---- Step 2: obs (halfRefVil) -> beta1_c {magsus, corr} ------------------
    s2 = {**shared, "L_coarse": L_coarse,
          "beta_f": beta_f, "beta1_f": beta1_f, "alpha_eff_f": alpha_eff_f,
          "beta_c": beta_c, "alpha_eff_c": alpha_eff_c,
          "pp_xy_list_coarse": config["pp_xy_list_coarse"],
          "argzz_xy_list_coarse": config["argzz_xy_list_coarse"],
          "wilson_xy_list_coarse": config["wilson_xy_list_coarse"],
          "beta1_list": config["beta1_list"],
          "observable": config["observable"],
          "_result_path": os.path.join(results_dir, "step2_result.json")}
    _run_step("obs_beta1", dump(s2, "step2_config.json"))
    r2 = step_common.load_json(s2["_result_path"])
    _b1c = r2["beta1_c"]
    beta1_c = {
        "magsus": _round_opt(_b1c.get("magsus")),
        "corr":   _round_opt(_b1c.get("corr")),
    }

    # ---- Step 3a: 1plaq -> alpha_c -------------------------------------------
    if skip_alpha_fits:
        # Not fitted: alpha_c fixed to 0.0 by construction (see skip_alpha_fits).
        alpha_c_1plaq, alpha_c_1plaq_err = 0.0, None
        print("[orchestrator] skipping step 3a (1plaq s-vortex fit needs "
              "alpha_f > 0 under U-renorm); alpha_c fixed to 0.0")
    else:
        s3a = {**shared, "beta_f": beta_f, "beta1_f": beta1_f,
               "alpha_f": alpha_f, "alpha1_f": alpha1_f,
               "1plaq": config["1plaq"],
               "_result_path": os.path.join(results_dir, "step3a_result.json")}
        _run_step("1plaq", dump(s3a, "step3a_config.json"))
        r3a = step_common.load_json(s3a["_result_path"])
        alpha_c_1plaq = _round_opt(r3a["alpha_c"])
        alpha_c_1plaq_err = r3a.get("alpha_c_err")

    # ---- Step 3b: topo -> alpha_c(topo) {magsus, corr} -----------------------
    # do_topo=False skips this whole step (alpha_c_topo stays None for both
    # methods, same as a topo fit that did not match). Use it during early
    # trials: beta1_c (step 2) is only as good as the supplied beta1_list, so
    # until that scan brackets beta1_c the topo alpha -- which runs AT that
    # beta1_c -- is unreliable, and running it just burns the expensive topo MC.
    r3b = None
    alpha_c_topo = {"magsus": None, "corr": None}
    if skip_alpha_fits:
        # Not fitted: alpha_c fixed to 0.0 by construction (see skip_alpha_fits).
        alpha_c_topo = {"magsus": 0.0, "corr": 0.0}
        print("[orchestrator] skipping step 3b (topo alpha fit needs fine Q_s, "
              "ill-defined at alpha_f~=0 under U-renorm); alpha_c fixed to 0.0")
    elif do_topo:
        s3b = {**shared, "L_coarse": L_coarse,
               "beta_f": beta_f, "beta1_f": beta1_f,
               "alpha_f": alpha_f, "alpha1_f": alpha1_f, "alpha_eff_f": alpha_eff_f,
               "beta_c": beta_c, "alpha_eff_c": alpha_eff_c,
               "beta1_c": beta1_c,
               "alpha_list": config.get("alpha_list", []),
               # Lets step3b auto-build alpha_list (bracket around the 1plaq alpha_c)
               # when the config leaves it empty.
               "alpha_1plaq": alpha_c_1plaq,
               "topo": config["topo"],
               "_result_path": os.path.join(results_dir, "step3b_result.json")}
        _run_step("topo", dump(s3b, "step3b_config.json"))
        r3b = step_common.load_json(s3b["_result_path"])
        _act = r3b["alpha_c"]
        alpha_c_topo = {
            "magsus": _round_opt(_act.get("magsus")),
            "corr":   _round_opt(_act.get("corr")),
        }
    else:
        print("[orchestrator] do_topo=False -> skipping step 3b (topo alpha fit)")

    # ---- Assemble the four combos --------------------------------------------
    def combo(beta1_key, alpha_val):
        """One renormalized coarse model: beta1 from {magsus,corr}, alpha given.
        alpha1 = alpha_eff_c - alpha."""
        a1 = None if alpha_val is None else round(alpha_eff_c - alpha_val, 3)
        return {"beta_c": beta_c, "beta1_c": beta1_c[beta1_key],
                "alpha_c": alpha_val, "alpha1_c": a1, "alpha_eff_c": alpha_eff_c}

    four_combos = {
        "magsus_obs_fit_topo": combo("magsus", alpha_c_topo["magsus"]),
        "corr_obs_fit_topo":   combo("corr",   alpha_c_topo["corr"]),
        "magsus_1plaq":        combo("magsus", alpha_c_1plaq),
        "corr_1plaq":          combo("corr",   alpha_c_1plaq),
    }

    # HMC acceptance rates, one block per step ({"mean","std"} or per-point maps;
    # null where the step was skipped or its ensemble used the heatbath sampler).
    # Step3b's per-method acc blocks are read from its fit_details entries (the
    # process_case_topo outputs) -- the step result does not duplicate them at
    # the top level.
    def _topo_acc(key):
        if r3b is None:
            return None
        fd = r3b.get("fit_details") or {}
        return {m: (fd.get(m) or {}).get(key) for m in ("magsus", "corr")}

    acc_rate = {
        "step1_2plaq": r1.get("acc_rate"),
        "step2_obs_beta1": r2.get("acc_rate"),
        "step3a_1plaq": r3a.get("acc_rate") if not skip_alpha_fits else None,
        "step3b_topo": _topo_acc("acc_rate"),
    }
    # s-Metropolis acceptance rates (full-Villain samplers only). The 2plaq
    # halfRefVil action has no integer s field, so its entry is structurally null.
    acc_rate_metro = {
        "step1_2plaq": None,
        "step2_obs_beta1": r2.get("acc_rate_metro"),
        "step3a_1plaq": r3a.get("acc_rate_metro") if not skip_alpha_fits else None,
        "step3b_topo": _topo_acc("acc_rate_metro"),
    }

    warnings = []
    for name, c in four_combos.items():
        if c["alpha_c"] is not None:
            drift = abs((c["alpha_c"] + (c["alpha1_c"] or 0.0)) - alpha_eff_c)
            if drift > 1e-6:
                warnings.append(
                    f"{name}: alpha_c+alpha1_c ({c['alpha_c'] + (c['alpha1_c'] or 0.0):.6g}) "
                    f"!= alpha_eff_c ({alpha_eff_c:.6g}); drift={drift:.2e}")
        else:
            reason = ("skipped (do_topo=False)"
                      if not do_topo and "topo" in name
                      else "fit did not match")
            warnings.append(f"{name}: alpha_c is None ({reason})")

    if skip_alpha_fits:
        warnings.append(
            "alpha_f~=0 fine model under U-renorm: 1plaq/topo alpha fits skipped; "
            "alpha_c fixed to 0.0 and alpha1_c=alpha_eff_c in all combos (not fitted)")

    if skip_alpha_fits:
        skip_note = ("SKIPPED (U-renorm alpha_f~=0: alpha_c fixed to 0.0, "
                     "alpha1_c=alpha_eff_c)")
        prov_3a = prov_3b = skip_note
    else:
        prov_3a = "step3a_result.json"
        prov_3b = "step3b_result.json" if do_topo else "SKIPPED (do_topo=False)"

    summary = {
        "run_name": run_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            "N": N, "mod": mod, "renorm_type": renorm_type, "L": L, "L_coarse": L_coarse,
            "n_workers": n_workers, "auto_regenerate": auto_regenerate,
            "do_topo": do_topo,
            "skip_alpha_fits": skip_alpha_fits,
            "beta_f": beta_f, "beta1_f": beta1_f,
            "alpha_f": alpha_f, "alpha1_f": alpha1_f, "alpha_eff_f": alpha_eff_f,
            "pp_xy_list_coarse": config["pp_xy_list_coarse"],
            "argzz_xy_list_coarse": config["argzz_xy_list_coarse"],
            "wilson_xy_list_coarse": config["wilson_xy_list_coarse"],
            "beta1_list": config["beta1_list"],
            # The actual scan used (auto-built default if the config left it empty).
            # When topo was skipped there was no scan -> fall back to the configured list.
            "alpha_list": (r3b or {}).get("alpha_list", config.get("alpha_list", [])),
        },
        "renormalized_couplings": {
            "beta_c": beta_c, "beta_c_err": beta_c_err,
            "alpha_eff_c": alpha_eff_c, "alpha_eff_c_err": alpha_eff_c_err,
            "beta1_c": beta1_c,
            "alpha_c": {"1plaq": alpha_c_1plaq,
                        "obs_fit_topo": alpha_c_topo},
        },
        "four_combos": four_combos,
        "acc_rate": acc_rate,
        "acc_rate_metro": acc_rate_metro,
        # Each combo reshaped as a config `fine` block for a follow-up run.
        "renorm_as_fine": {
            name: {"beta_f": c["beta_c"], "beta1_f": c["beta1_c"],
                   "alpha_f": c["alpha_c"], "alpha1_f": c["alpha1_c"]}
            for name, c in four_combos.items()
        },
        "provenance": {
            "step1_2plaq":      "step1_result.json",
            "step2_obs_beta1":  "step2_result.json",
            "step3a_1plaq":     prov_3a,
            "step3b_topo":      prov_3b,
        },
        "warnings": warnings,
    }
    summary_path = os.path.join(results_dir, "full_renorm_summary.json")
    step_common.dump_json(summary, summary_path)

    print("\n" + "=" * 70)
    print(f"[orchestrator] DONE -> {summary_path}")
    print("Renormalized couplings:")
    print(json.dumps(summary["renormalized_couplings"], indent=2))
    print("Four combos:")
    print(json.dumps(summary["four_combos"], indent=2))
    if warnings:
        print("Warnings:")
        for w in warnings:
            print(f"  - {w}")
    return summary


if __name__ == "__main__":
    # Config filenames to run. Bare names are searched recursively under configs/
    # (top level + every subfolder). Set to None to run EVERY .json under configs/.
    json_config_list = None
    json_config_list = ["N4_4_z_b1.0_mod0.json", "N4_4_z_b1.0_mod1.json"]

    n_workers = 10  # shared across all four steps (overrides any per-block value)
    # Data-reuse policy for step1 (2plaq) / step3a (1plaq) data files:
    #   True  -> overwrite a stale/mismatched/legacy data file with fresh MC.
    #   False -> raise an error naming the mismatching file (no silent overwrite).
    auto_regenerate = False
    # Whether to run step 3b (the topo alpha fit). Turn OFF for the first few
    # trials: beta1_c is only reliable once beta1_list brackets it, and the topo
    # alpha runs at that beta1_c -- so an unconverged beta1 wastes the expensive
    # topo MC. Flip to True once beta1_c from step 2 looks stable.
    do_topo = True
    if json_config_list is None:
        json_config_list = _collect_all_configs()
        print(f"[orchestrator] json_config_list is None -> running all "
              f"{len(json_config_list)} config(s) found under configs/")
    for json_config in json_config_list:
        json_path = _resolve_config_path(json_config)
        print("=-" * 70)
        print(f"[orchestrator] Running full renormalization for {json_path}")
        run_full_renorm(json_path, n_workers, auto_regenerate, do_topo)
        print(f"[orchestrator] Finished full renormalization for {json_path}")
        print("=-" * 70 + "\n")
