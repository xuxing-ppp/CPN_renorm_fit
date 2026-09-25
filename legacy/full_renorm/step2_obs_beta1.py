#!/usr/bin/env python3
"""Step 2 of the new full renorm: the observable fit (halfRefVil) -> beta1_c.

This is the halfRefVil slice of full_renorm_original/step3_observable.py. We use
halfRefVil because at this point alpha/alpha1 are NOT yet split -- only
alpha_eff = alpha + alpha1 is known (from step 1). halfRefVil folds alpha_eff
into its single plaquette coupling (alpha1 = 0 on disk).

The coarse beta1-scan is run at fixed (beta_c, alpha_eff_c) from step 1; only
beta1 is scanned. We keep TWO beta1_c estimates:
  * magsus : sum(conn_PP_corr) match (coarse scaled by L**2)
  * corr   : PP correlation-length match
`avg` is dropped (it is an observable-choice-dependent global mean, considered
artificial here); toposus is unavailable for halfRefVil (no Q_s).

Run as a subprocess with cwd=CPN_observable_fit. Reuses measure_PP_Wil.main and
plot_PP_Wil_beta1.process_case directly (both importable; measure_main is itself
restart-safe, topping up any argzz/Wilson sizes missing from an existing JSON).
"""

import os
import sys

# --- module-level path setup (see step1_2plaq.py for the spawn rationale) ----
_HERE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.dirname(_HERE)
_SUBPROJECT = os.path.join(_BASE, "CPN_observable_fit")
if _SUBPROJECT not in sys.path:
    sys.path.insert(0, _SUBPROJECT)

import step_common  # noqa: E402  (imported after path setup)


def _measure_kwargs(obs):
    return dict(
        heatbath_fraction=obs["heatbath_fraction"],
        epsilon=obs["epsilon"], n_leapfrog=obs["n_leapfrog"],
        mass_a=obs["mass_a"], mass_z=obs["mass_z"],
        s_step=obs.get("s_step", 0.5), s_update_num=obs.get("s_update_num", 1),
        n_therm=obs["n_therm"], n_meas=obs["n_meas"],
        meas_interval=obs["meas_interval"],
        n_total=obs["n_total"], n_workers=obs["n_workers"],
    )


def run_step(cfg):
    from measure_PP_Wil import main as measure_main
    from plot_PP_Wil_beta1 import process_case

    N, mod, renorm_type = cfg["N"], cfg["mod"], cfg["renorm_type"]
    L, L_coarse = cfg["L"], cfg["L_coarse"]
    obs_fit_type = "halfRefVil"  # fixed for this pipeline step
    beta_f, beta1_f = cfg["beta_f"], cfg["beta1_f"]
    beta_c = cfg["beta_c"]
    pp_xy_list_coarse = cfg["pp_xy_list_coarse"]
    argzz_xy_list_coarse = cfg["argzz_xy_list_coarse"]
    wilson_xy_list_coarse = cfg["wilson_xy_list_coarse"]
    beta1_list = [float(b) for b in cfg["beta1_list"]]
    obs = cfg["observable"]
    mkw = _measure_kwargs(obs)

    # halfRefVil folds alpha_eff = alpha + alpha1 (alpha1 = 0 on disk).
    alpha_eff_f, alpha_eff_c = cfg["alpha_eff_f"], cfg["alpha_eff_c"]
    alpha_f, alpha1_f = alpha_eff_f, 0.0
    alpha_c, alpha1_c = alpha_eff_c, 0.0

    L_f = L_coarse * L
    renorm = L
    argzz_xy_list_fine = [[int(x * L), int(y * L)] for x, y in argzz_xy_list_coarse]
    wilson_xy_list_fine = [[int(x * L), int(y * L)] for x, y in wilson_xy_list_coarse]

    print(f"[obs] halfRefVil: fine   L={L_f} beta={beta_f} beta1={beta1_f} "
          f"alpha_eff={alpha_eff_f}")
    print(f"[obs] halfRefVil: coarse L={L_coarse} beta={beta_c} "
          f"alpha_eff={alpha_eff_c} ({len(beta1_list)} beta1 points)")

    # Fine lattice: single point at the given fine model.
    measure_main(N, L_f, [(beta_f, beta1_f, alpha_f, alpha1_f)],
                 argzz_xy_list_fine, wilson_xy_list_fine,
                 mod=mod, obs_fit_type=obs_fit_type, **mkw)

    # Coarse lattice: beta1-scan at fixed (beta_c, alpha_eff_c).
    coarse_list = [(beta_c, b1, alpha_c, alpha1_c) for b1 in beta1_list]
    measure_main(N, L_coarse, coarse_list,
                 argzz_xy_list_coarse, wilson_xy_list_coarse,
                 mod=mod, obs_fit_type=obs_fit_type, **mkw)

    info = dict(
        N=N, L_f=L_f, mod=mod, obs_fit_type=obs_fit_type,
        beta_f=beta_f, beta1_f=beta1_f, alpha_f=alpha_f, alpha1_f=alpha1_f,
        renorm=renorm, L_c=L_coarse, renorm_type=renorm_type,
        beta_c=beta_c, beta1_c_list=beta1_list, alpha_c=alpha_c, alpha1_c=alpha1_c,
        pp_xy_list_coarse=pp_xy_list_coarse,
        argzz_xy_list_coarse=argzz_xy_list_coarse,
        wilson_xy_list_coarse=wilson_xy_list_coarse,
    )
    results_dict = process_case(info, mod=mod)
    if results_dict is None:
        raise RuntimeError(
            "observable matching produced no result (fine/coarse data missing)."
        )

    # Keep only magsus and corr (drop avg as "artificial"; toposus null for halfRefVil).
    corr = results_dict["beta1_c_pp"].get("corr_len")
    magsus = results_dict["beta1_c_pp"].get("magsus")

    return {
        "step": "obs_beta1",
        "obs_fit_type": obs_fit_type,
        "renorm_type": renorm_type, "renorm": renorm, "L_f": L_f, "L_c": L_coarse,
        "N": N, "mod": mod,
        "beta_f": beta_f, "beta1_f": beta1_f,
        "alpha_eff_f": alpha_eff_f, "alpha_eff_c": alpha_eff_c,
        "beta_c": beta_c,
        "beta1_c": {"magsus": magsus, "corr": corr},
        "acc_rate": results_dict.get("acc_rate"),
        "acc_rate_metro": results_dict.get("acc_rate_metro"),
        "per_observable": {
            "pp": results_dict["beta1_c_pp"],
            "argzz": results_dict["beta1_c_argzz"],
            "wilson": results_dict["beta1_c_wilson"],
            "summary": results_dict["summary"],
        },
    }


if __name__ == "__main__":
    cfg = step_common.load_json(sys.argv[1])
    os.chdir(_SUBPROJECT)
    result = run_step(cfg)
    step_common.dump_json(result, cfg["_result_path"])
    b = result['beta1_c']
    print(f"[obs] beta1_c magsus={b['magsus']} corr={b['corr']}")
