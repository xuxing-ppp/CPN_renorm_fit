"""Topo-susceptibility alpha fit for CP^{N-1} renormalization (analysis & plots).

The topo counterpart of `plot_PP_Wil_beta1.py`. At fixed `(beta_c, beta1_c,
alpha_eff_c)` it scans the coarse coupling `alpha` (with `alpha1 = alpha_eff_c -
alpha`) on a RefVil lattice and matches the coarse integer-winding susceptibility
`topo_sus(Q_s)` onto the fine susceptibility scaled by `L**2`:

    topo_sus_coarse(Q_s)(alpha_c)  ==  topo_sus_fine(fine_comp) * L**2

with `fine_comp = "Q_z"` (z-renorm) / `"Q_s"` (U-renorm). Data comes from
`measure_PP_Wil.main` (which already measures topo charge AND conn_PP_corr), so
besides the fit plot we also emit informational magsus / corr_len vs alpha plots.

Run from this subproject's root. Reusable as `from plot_topo_alpha import
process_case_topo` by `full_renorm/step3b_topo.py`.
"""

import numpy as np
import os
import json
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_PP_Wil_beta1 import (
    linear_interpolate, corr_len, match_coupling, _clean_err_list,
    add_fit_curve, add_coarse_points, acc_brief,
)
from measure_PP_Wil import filepath_for


def _round_to_n_sig(x, n):
    """Round x to n significant digits.

    _round_to_n_sig(0.234, 1) -> 0.2,  _round_to_n_sig(0.234, 2) -> 0.23,
    _round_to_n_sig(1.392, 1) -> 1.0,  _round_to_n_sig(5.7, 2)   -> 5.7,
    _round_to_n_sig(0.08, 1)  -> 0.08. Returns 0.0 for x==0.
    """
    if not x:
        return 0.0
    d = math.floor(math.log10(abs(x)))
    return round(x, n - 1 - d)


def default_alpha_list(alpha_1plaq):
    """Default coarse-`alpha` scan bracketing the 1plaq estimate.

    The topo-fit alpha_c is expected to sit near the 1plaq alpha_c, so when the
    caller does not supply an explicit scan we round alpha_1plaq two ways -- to
    one significant digit (-> a1) and to two significant digits (-> a2) -- and
    center the bracket on the finer a2 with additive steps of 0.1*a1:

        [a2 - 0.2*a1, a2 - 0.1*a1, a2, a2 + 0.1*a1, a2 + 0.2*a1]

    Values are 3-decimal rounded to match the .3f path templates. Raises
    ValueError if alpha_1plaq is not a positive number (the bracket is
    meaningless otherwise). Used by full_renorm/step3b_topo.py when the config
    leaves alpha_list empty.
    """
    try:
        a = float(alpha_1plaq)
    except (TypeError, ValueError):
        raise ValueError(
            f"default_alpha_list needs a positive alpha_1plaq, got {alpha_1plaq!r}")
    if not (a > 0):
        raise ValueError(
            f"default_alpha_list needs a positive alpha_1plaq, got {alpha_1plaq!r}")
    a1 = _round_to_n_sig(a, 1)
    a2 = _round_to_n_sig(a, 2)
    return [round(a2 + c * a1, 3) for c in (-0.2, -0.1, 0.0, 0.1, 0.2)]


def _toposus_err(tc_comp, n_total_meas):
    """Approximate standard error of topo_sus = var(Q)/V from stored Q stats.
    For a near-zero-mean series, err(var)/var ~ 2*sqrt(tau/M). Used as the
    fine-target error estimate for the topo-alpha `match_coupling` fit."""
    ts = tc_comp.get("topo_sus")
    tau = tc_comp.get("tau")
    if ts is None or tau is None or n_total_meas is None:
        return None
    return abs(ts) * 2.0 * np.sqrt(max(tau, 0.5) / max(n_total_meas, 1))


def process_case_topo(info_dict, mod=0):
    """Topo-alpha match for one (fine model, beta1_c) case."""
    N = info_dict["N"]
    L_f = info_dict["L_f"]
    L_c = info_dict["L_c"]
    beta_f = info_dict["beta_f"]
    beta1_f = info_dict["beta1_f"]
    alpha_f = info_dict["alpha_f"]
    alpha1_f = info_dict.get("alpha1_f", 0.0)
    beta_c = info_dict["beta_c"]
    beta1_c = info_dict["beta1_c"]
    alpha_eff_c = info_dict["alpha_eff_c"]
    alpha_list = [float(a) for a in info_dict.get("alpha_list", [])]
    if not alpha_list:
        # Empty scan: auto-bracket around the 1plaq alpha if the caller supplied
        # it (the topo alpha_c should sit near the 1plaq alpha_c). The coarse
        # data must already exist for these alphas -- step3b_topo generates it;
        # for a standalone call this just gives a clearer "no data" failure.
        a1p = info_dict.get("alpha_1plaq")
        if a1p is not None:
            try:
                alpha_list = default_alpha_list(a1p)
            except ValueError:
                alpha_list = []
            else:
                print(f"  alpha_list empty -> default from 1plaq alpha={a1p:.4g}: "
                      f"{alpha_list}")
    renorm = info_dict["renorm"]
    renorm_type = info_dict["renorm_type"]
    fine_obs_fit_type = info_dict["fine_obs_fit_type"]
    fine_comp = "Q_z" if renorm_type == "z" else "Q_s"

    title = f"beta_c={beta_c:.3f}, beta1_c={beta1_c:.3f}, alpha_eff_c={alpha_eff_c:.3f}"

    print("=" * 70)
    print(f"Topo-alpha fit: renorm={renorm}, type='{renorm_type}', fine={fine_obs_fit_type}")
    print(f"  Fine:   L={L_f}, beta={beta_f}, beta1={beta1_f}, alpha={alpha_f}, alpha1={alpha1_f}")
    print(f"  Coarse: L={L_c}, beta={beta_c}, beta1_c={beta1_c}, alpha_eff_c={alpha_eff_c}")
    print(f"  alpha scan: {len(alpha_list)} points; fine_comp={fine_comp}")
    print("=" * 70)

    # ── load fine data ──
    fine_path = filepath_for(N, L_f, beta_f, beta1_f, alpha_f, alpha1_f, mod, fine_obs_fit_type)
    if not os.path.exists(fine_path):
        print(f"  ERROR: fine data not found: {fine_path}")
        return None
    with open(fine_path) as f:
        temp = json.load(f)
    conn_PP_corr_f = np.asarray(temp["conn_PP_corr"]["mean"])
    topo_charge_f = temp.get("topo_charge", {})
    fine_M = temp.get("n_total_meas")
    fine_acc = acc_brief(temp.get("acc_rate"))          # HMC acc (null: HB/legacy)
    fine_acc_metro = acc_brief(temp.get("acc_rate_metro"))
    print(f"  Fine data loaded: {fine_path}")

    if fine_comp not in topo_charge_f or "topo_sus" not in topo_charge_f[fine_comp]:
        print(f"  ERROR: fine data has no topo_charge['{fine_comp}']['topo_sus'] "
              f"(fine_obs_fit_type={fine_obs_fit_type}). For U-renorm the fine must "
              f"be RefVil with alpha_f>0; for z-renorm it must carry Q_z.")
        return None
    fine_topo_raw = float(topo_charge_f[fine_comp]["topo_sus"])
    fine_topo_target = fine_topo_raw * renorm ** 2          # fine scaled by L**2
    fine_magsus = float(np.sum(conn_PP_corr_f))
    fine_corr = corr_len(conn_PP_corr_f)

    # ── load coarse data (RefVil alpha-scan) ──
    loaded_alpha = []
    toposus_c = []
    toposus_c_err_raw = []
    magsus_c = []
    corrlen_c = []
    coarse_acc = {}        # "alpha=<val>" -> acc_rate block (brief) per scan point
    coarse_acc_metro = {}  # same, for the s-Metropolis rate
    for a in alpha_list:
        a1 = round(alpha_eff_c - a, 3)
        cpath = filepath_for(N, L_c, beta_c, beta1_c, a, a1, mod, "RefVil")
        if not os.path.exists(cpath):
            print(f"  File not found: alpha={a:.3f}, alpha1={a1:.3f} -> {cpath}")
            continue
        with open(cpath) as f:
            t = json.load(f)
        tc = t.get("topo_charge", {})
        if "Q_s" not in tc:
            print(f"  alpha={a:.3f}: coarse data missing Q_s (not RefVil?); skip")
            continue
        cpc = np.asarray(t["conn_PP_corr"]["mean"])
        loaded_alpha.append(a)
        toposus_c.append(float(tc["Q_s"]["topo_sus"]))
        toposus_c_err_raw.append(tc["Q_s"].get("err"))
        magsus_c.append(float(np.sum(cpc)) * renorm ** 2)
        corrlen_c.append(corr_len(cpc) * renorm)
        coarse_acc[f"alpha={a:.3f}"] = acc_brief(t.get("acc_rate"))
        coarse_acc_metro[f"alpha={a:.3f}"] = acc_brief(t.get("acc_rate_metro"))
        last_M = t.get("n_total_meas")

    if len(loaded_alpha) == 0:
        print("  ERROR: no coarse alpha-scan data found")
        return None

    alpha_arr = np.array(loaded_alpha)
    toposus_c = np.array(toposus_c)
    toposus_c_err = _clean_err_list(toposus_c_err_raw)
    magsus_c = np.array(magsus_c)
    corrlen_c = np.array(corrlen_c)
    print(f"  Coarse data loaded: {len(loaded_alpha)} alpha points")

    # ── output folder (one subfolder per beta1_c) ──
    folder_path = (
        f"results/Topo_alpha_renorm_plot_mod{mod}/"
        f"N{N}_L{renorm}_{renorm_type}/"
        f"beta{beta_f:.3f}_beta1_{beta1_f:.3f}_alpha{alpha_f:.3f}_alpha1_{alpha1_f:.3f}/"
        f"beta1_c_{beta1_c:.3f}"
    )
    os.makedirs(folder_path, exist_ok=True)

    # ================================================================
    #  1. Topological susceptibility vs alpha  (THE FIT)
    # ================================================================
    print("\n--- topo_sus(Q_s) vs alpha (FIT) ---")
    toposus_f_line = np.ones_like(toposus_c) * fine_topo_target
    # Fit topo_sus(alpha) through ALL scan points with an error-weighted
    # smoothing spline and read off where it hits the fine target (replaces the
    # old two-point linear interpolation + post-hoc local-slope error). The MC
    # error in match_coupling folds in both the coarse-point errs and the fine
    # target err.
    fine_target_err = _toposus_err(topo_charge_f[fine_comp], fine_M)
    alpha_c, alpha_c_err = match_coupling(
        alpha_arr, toposus_c, fine_topo_target,
        y_err=toposus_c_err, y_target_err=fine_target_err)

    plt.figure()
    add_coarse_points(alpha_arr, toposus_c, toposus_c_err,
                      label="Coarse topo_sus(Q_s)")
    plt.plot(alpha_arr, toposus_f_line,
             label=f"Fine topo_sus({fine_comp}) * {renorm}**2")
    add_fit_curve(alpha_arr, toposus_c, toposus_c_err)
    plt.xlabel("alpha")
    plt.ylabel("Topological Susceptibility")
    plt.legend()
    plt.title(f"topo_sus  {title}")
    if alpha_c is not None:
        y_min, y_max = plt.ylim()
        plt.axvline(x=alpha_c, color="red", linestyle="--", alpha=0.7)
        plt.text(alpha_c, y_min + 0.95 * (y_max - y_min),
                 f"  alpha = {alpha_c:.4f}", color="red", fontweight="bold", va="top")
        err_str = f" +- {alpha_c_err:.4f}" if alpha_c_err is not None else ""
        print(f"  topo_sus: fine({fine_comp})={fine_topo_raw:.6g}*{renorm}**2="
              f"{fine_topo_target:.6g} -> alpha_c = {alpha_c:.4f}{err_str}")
    else:
        print(f"  topo_sus: fine_target={fine_topo_target:.6g} -> NO MATCH "
              f"(outside coarse range [{toposus_c.min():.6g},{toposus_c.max():.6g}])")
    plt.savefig(os.path.join(folder_path, "topo_sus.png"))
    plt.close()

    # ================================================================
    #  2. Magnetic susceptibility vs alpha  (INFORMATIONAL)
    # ================================================================
    print("\n--- mag_sus vs alpha (informational) ---")
    magsus_f_line = np.ones_like(magsus_c) * fine_magsus
    plt.figure()
    plt.plot(alpha_arr, magsus_c, label=f"Coarse mag_sus * {renorm}**2", marker="o")
    plt.plot(alpha_arr, magsus_f_line, label="Fine mag_sus")
    plt.xlabel("alpha")
    plt.ylabel("Magnetic Susceptibility")
    plt.legend()
    plt.title(f"mag_sus  {title}")
    plt.savefig(os.path.join(folder_path, "mag_sus.png"))
    plt.close()
    print(f"  fine mag_sus={fine_magsus:.6g}; coarse range "
          f"[{magsus_c.min():.6g},{magsus_c.max():.6g}]")

    # ================================================================
    #  3. PP correlation length vs alpha  (INFORMATIONAL)
    # ================================================================
    print("\n--- corr_len vs alpha (informational) ---")
    corrlen_f_line = np.ones_like(corrlen_c) * fine_corr
    plt.figure()
    plt.plot(alpha_arr, corrlen_c, label=f"Coarse corr_len * {renorm}", marker="o")
    plt.plot(alpha_arr, corrlen_f_line, label="Fine corr_len")
    plt.xlabel("alpha")
    plt.ylabel("PP Correlation Length")
    plt.legend()
    plt.title(f"corr_len  {title}")
    plt.savefig(os.path.join(folder_path, "corr_len.png"))
    plt.close()
    print(f"  fine corr_len={fine_corr:.6g}; coarse range "
          f"[{corrlen_c.min():.6g},{corrlen_c.max():.6g}]")

    # ── save results ──
    results = {
        "N": N, "L_f": L_f, "L_c": L_c, "renorm": renorm, "renorm_type": renorm_type,
        "fine_obs_fit_type": fine_obs_fit_type, "fine_comp": fine_comp,
        "beta_f": beta_f, "beta1_f": beta1_f, "alpha_f": alpha_f, "alpha1_f": alpha1_f,
        "beta_c": beta_c, "beta1_c": beta1_c, "alpha_eff_c": alpha_eff_c,
        "alpha_c": alpha_c, "alpha_c_err": alpha_c_err,
        "fine_topo_sus": fine_topo_raw, "fine_target": fine_topo_target,
        "alpha_list": alpha_arr.tolist(),
        "coarse_topo_sus": toposus_c.tolist(),
        "coarse_magsus": magsus_c.tolist(),
        "coarse_corr_len": corrlen_c.tolist(),
        "matched": alpha_c is not None,
        # HMC / s-Metropolis acceptance rates of the fine ensemble and each
        # coarse scan point, as {"mean","std"} (per_chain stays in the data
        # JSONs; null components for heatbath-generated or legacy files).
        "acc_rate": {"fine": fine_acc, "coarse": coarse_acc},
        "acc_rate_metro": {"fine": fine_acc_metro, "coarse": coarse_acc_metro},
    }
    with open(os.path.join(folder_path, "topo_alpha_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {folder_path}/topo_alpha_results.json")
    return results


if __name__ == "__main__":
    # Self-test of the fit math on synthetic coarse data
    # (process_case_topo itself needs on-disk measure_PP_Wil data; see step3b_topo.)
    renorm = 4
    fine_json = {"n_total_meas": 5000,
                 "topo_charge": {"Q_z": {"topo_sus": 0.01, "tau": 2.0}}}
    alpha_list = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    sus = [0.40, 0.22, 0.12, 0.06, 0.025, 0.008]
    toposus_c = np.array(sus)
    fine_target = 0.01 * renorm ** 2
    # error-weighted smoothing-spline fit through all 6 points
    fine_target_err = _toposus_err(fine_json["topo_charge"]["Q_z"],
                                   fine_json["n_total_meas"])
    alpha_c, alpha_c_err = match_coupling(
        alpha_list, toposus_c, fine_target,
        y_err=[0.01] * len(alpha_list), y_target_err=fine_target_err)
    alpha_c_lin = linear_interpolate(alpha_list, toposus_c, fine_target)
    print(f"self-test target={fine_target:.4f}: "
          f"spline alpha_c = {alpha_c:.4f} +- {alpha_c_err:.4f}, "
          f"linear alpha_c = {alpha_c_lin:.4f}")
