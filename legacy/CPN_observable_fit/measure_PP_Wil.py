import numpy as np
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

import sys
# Make the local `func` package importable regardless of the working directory
# (so the script runs from CPN_renorm_fit/ or this subproject alike).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from func.func_CPN_HB import CPNSampler, integrated_autocorr_time
from func.func_CPN_halfRefVil_HMC import CPN_halfRefVil_HMCSampler
from func.func_CPN_RefVil_HMC import CPN_RefVil_HMCSampler


def _worker(params):
    (worker_id, L, N, beta, beta1, alpha, alpha1, mod, obs_fit_type,
     s_step, s_update_num,
     heatbath_fraction, epsilon, n_leapfrog, mass_a, mass_z,
     n_therm, n_meas, meas_interval, argzz_xy_list, wilson_xy_list) = params

    # seed = worker_id * 100000 + abs(hash((L, N, beta, beta1, alpha))) % 99999
    seed = None

    # obs_fit_type selects the action family for the beta1 != 0 model:
    #   "RefVil"    -> full-Villain (integer s field, alpha1) HMC, always used.
    #   "halfRefVil"-> half-Villain (cos(da), no s/alpha1) HMC, or the pure-beta
    #                  heatbath sampler when beta1 ~= 0.
    if obs_fit_type == "RefVil":
        sampler = CPN_RefVil_HMCSampler(
            N, L, L, beta, beta1, alpha, alpha1,
            epsilon=epsilon, n_leapfrog=n_leapfrog,
            mass_a=mass_a, mass_z=mass_z,
            s_step=s_step, s_update_num=s_update_num, seed=seed,
        )
    elif np.abs(beta1) < 1e-8:
        # Pure-beta model: mod is irrelevant for the physics (heatbath sampler),
        # but it is still recorded in the output path/metadata for a uniform layout.
        sampler = CPNSampler(L, L, N, beta, alpha, seed=seed)
    else:
        sampler = CPN_halfRefVil_HMCSampler(
            L, L, N, beta, beta1, alpha,
            epsilon=epsilon, n_leapfrog=n_leapfrog,
            mass_a=mass_a, mass_z=mass_z, seed=seed
        )

    def _sweep():
        if obs_fit_type == "RefVil":
            return sampler.sweep(mod=mod)
        if np.abs(beta1) < 1e-8:
            return sampler.sweep(heatbath_fraction)
        return sampler.sweep(mod=mod)

    for _ in tqdm(range(n_therm), desc="therm", leave=False):
        _sweep()

    P_exp_list = []
    conn_PP_corr_list = []
    argzz_loop_list = []
    wilson_loop_list = []
    topo_list = []  # list of topo_charge() tuples: (Q_U, Q_z[, Q_s])
    for _ in tqdm(range(n_meas // meas_interval), desc="meas", leave=False):
        for _ in range(meas_interval):
            _sweep()
        P_exp_list.append(sampler.P_exp())
        conn_PP_corr_list.append(sampler.conn_PP_corr())
        argzz_loop_list.append(sampler.argzz_loop_xy_list(argzz_xy_list))
        wilson_loop_list.append(sampler.wilson_loop_xy_list(wilson_xy_list))
        topo_list.append(sampler.topo_charge())

    # HMC acceptance rates of this chain, normalized to (acc_hmc, acc_metro).
    # The heatbath sampler has no HMC (and the half-Villain action no s-Metro),
    # so both are None there -> the JSON reports acc_rate=null.
    if obs_fit_type == "RefVil":
        acc_hmc, acc_metro = sampler.accept_rate
    elif np.abs(beta1) < 1e-8:
        acc_hmc, acc_metro = None, None
    else:
        acc_hmc, acc_metro = sampler.accept_rate, None

    return (np.array(P_exp_list), np.array(conn_PP_corr_list),
            np.array(argzz_loop_list), np.array(wilson_loop_list),
            np.array(topo_list), acc_hmc, acc_metro)


def _run_workers(N, L, beta, beta1, alpha, alpha1, mod, obs_fit_type,
                 s_step, s_update_num,
                 heatbath_fraction, epsilon, n_leapfrog, mass_a, mass_z,
                 n_therm, n_meas, meas_interval, argzz_xy_list, wilson_xy_list,
                 n_total, n_workers):
    """Run `n_total` independent chains; return concatenated measurement arrays."""
    worker_args = [
        (i, L, N, beta, beta1, alpha, alpha1, mod, obs_fit_type,
         s_step, s_update_num,
         heatbath_fraction, epsilon, n_leapfrog, mass_a, mass_z,
         n_therm, n_meas, meas_interval, argzz_xy_list, wilson_xy_list)
        for i in range(n_total)
    ]
    all_P_exp, all_conn_PP_corr, all_argzz_loop, all_wilson_loop, all_topo = [], [], [], [], []
    all_acc_hmc, all_acc_metro = [], []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(_worker, args) for args in worker_args]
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc=f"Workers, beta={beta:.3f}, beta1={beta1:.3f}, alpha={alpha:.3f}",
                           position=1, leave=False):
            pe, pc, al, wl, tp, acc_hmc, acc_metro = future.result()
            all_P_exp.append(pe)
            all_conn_PP_corr.append(pc)
            all_argzz_loop.append(al)
            all_wilson_loop.append(wl)
            all_topo.append(tp)
            all_acc_hmc.append(acc_hmc)
            all_acc_metro.append(acc_metro)
    return (np.concatenate(all_P_exp, axis=0),
            np.concatenate(all_conn_PP_corr, axis=0),
            np.concatenate(all_argzz_loop, axis=0),
            np.concatenate(all_wilson_loop, axis=0),
            np.concatenate(all_topo, axis=0),
            all_acc_hmc, all_acc_metro)


def _loop_stats(all_argzz_loop, all_wilson_loop, argzz_xy_list, wilson_xy_list, M):
    """mean/err/tau dicts for the argzz (scalar) and Wilson (2-comp) loop sizes."""
    argzz_loop_dict = {}
    Ka = len(argzz_xy_list)
    if Ka > 0:
        al_flat = all_argzz_loop.reshape(M, -1)                     # (M, Ka)
        tau_int_al = integrated_autocorr_time(al_flat.T).reshape(Ka)
        al_mean = np.mean(all_argzz_loop, axis=0)                   # (Ka,)
        al_err = np.std(all_argzz_loop, axis=0, ddof=1) * np.sqrt(
            2.0 * np.maximum(tau_int_al, 0.5) / M)
        for idx, (x, y) in enumerate(argzz_xy_list):
            argzz_loop_dict[f"{x}x{y}"] = {
                "tau": float(tau_int_al[idx]),
                "mean": float(al_mean[idx]),
                "err": float(al_err[idx]),
            }

    wilson_loop_dict = {}
    Kw = len(wilson_xy_list)
    if Kw > 0:
        wl_flat = all_wilson_loop.reshape(M, -1)                    # (M, Kw*2)
        tau_int_wl = integrated_autocorr_time(wl_flat.T).reshape(Kw, 2)
        wl_mean = np.mean(all_wilson_loop, axis=0)                  # (Kw, 2)
        wl_err = np.std(all_wilson_loop, axis=0, ddof=1) * np.sqrt(
            2.0 * np.maximum(tau_int_wl, 0.5) / M)
        for idx, (x, y) in enumerate(wilson_xy_list):
            wilson_loop_dict[f"{x}x{y}"] = {
                "tau": tau_int_wl[idx].tolist(),
                "mean": wl_mean[idx].tolist(),
                "err": wl_err[idx].tolist(),
            }

    return argzz_loop_dict, wilson_loop_dict


def _topo_stats(all_topo, L, M):
    """Topological charge stats per component.

    `all_topo` has shape (M, n_comp) with n_comp = 2 for the heatbath/half-Villain
    samplers (Q_U, Q_z) and 3 for the full-Villain sampler (Q_U, Q_z, Q_s). For each
    component present, compute mean(Q), mean(Q^2), the topological susceptibility
    topo_sus = (mean(Q^2) - mean(Q)^2) / V (V = L*L), its integrated autocorr time,
    and an error bar on Q.
    """
    names = ["Q_U", "Q_z", "Q_s"]
    V = L * L
    topo_dict = {}
    n_comp = all_topo.shape[1]
    for i in range(n_comp):
        q = all_topo[:, i]
        mean_q = float(np.mean(q))
        mean_q2 = float(np.mean(q ** 2))
        tau = float(integrated_autocorr_time(q))
        err = float(np.std(q, ddof=1) * np.sqrt(2.0 * max(tau, 0.5) / M))
        topo_dict[names[i]] = {
            "mean": mean_q,
            "mean2": mean_q2,
            "topo_sus": (mean_q2 - mean_q ** 2) / V,
            "tau": tau,
            "err": err,
        }
    return topo_dict


def _acc_stats(acc_list):
    """Per-chain acceptance list -> {"per_chain", "mean", "std"}, or None.

    None when every chain lacks an acceptance rate (heatbath sampler, or a
    legacy data file) -- the JSON then reports acc_rate=null.
    """
    vals = [a for a in acc_list if a is not None]
    if not vals:
        return None
    return {
        "per_chain": [None if a is None else float(a) for a in acc_list],
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
    }


def _topup_and_merge(filepath, existing, N, L, beta, beta1, alpha, alpha1, mod,
                     obs_fit_type, s_step, s_update_num,
                     heatbath_fraction, epsilon, n_leapfrog, mass_a, mass_z,
                     n_therm, n_meas, meas_interval, n_total, n_workers,
                     missing_argzz, missing_wilson):
    """Run a fresh simulation measuring only the missing argzz/Wilson sizes and
    merge their stats into the existing data file.

    Raw configurations aren't stored, so a missing size can't be added to the
    original ensemble — it is measured on a NEW independent ensemble at the same
    couplings. Each observable's mean is an independent expectation value, so the
    merge is statistically valid (the matching interpolates observables
    independently). Existing observables (incl. conn_PP_corr, P_exp, topo_charge)
    are kept unchanged.
    """
    (_, _, all_argzz_loop, all_wilson_loop, _,
     topup_acc_hmc, topup_acc_metro) = _run_workers(
        N, L, beta, beta1, alpha, alpha1, mod, obs_fit_type,
        s_step, s_update_num,
        heatbath_fraction, epsilon, n_leapfrog,
        mass_a, mass_z, n_therm, n_meas, meas_interval,
        missing_argzz, missing_wilson, n_total, n_workers)
    M = all_argzz_loop.shape[0]

    argzz_new, wilson_new = _loop_stats(
        all_argzz_loop, all_wilson_loop, missing_argzz, missing_wilson, M)

    existing.setdefault("argzz_loop", {}).update(argzz_new)
    existing.setdefault("wilson_loop", {}).update(wilson_new)
    argzz_list = existing.setdefault("argzz_xy_list", [])
    for xy in missing_argzz:
        if [int(xy[0]), int(xy[1])] not in argzz_list:
            argzz_list.append([int(xy[0]), int(xy[1])])
    wilson_list = existing.setdefault("wilson_xy_list", [])
    for xy in missing_wilson:
        if [int(xy[0]), int(xy[1])] not in wilson_list:
            wilson_list.append([int(xy[0]), int(xy[1])])

    # The top-up ensemble is independent of the original one, so its acceptance
    # rates are recorded separately (the original acc_rate keys stay untouched).
    existing["acc_rate_topup"] = _acc_stats(topup_acc_hmc)
    existing["acc_rate_metro_topup"] = _acc_stats(topup_acc_metro)

    with open(filepath, "w") as f:
        json.dump(existing, f, indent=2)

    tqdm.write(f"  topped up + merged into: {filepath}")
    for key, d in argzz_new.items():
        tqdm.write(f"    argzz {key}: mean={d['mean']:.6f}, err={d['err']:.6g}")
    for key, d in wilson_new.items():
        tqdm.write(f"    W {key}: mean={d['mean']}, err={d['err']}")
    if existing["acc_rate_topup"] is not None:
        tqdm.write(f"    topup HMC acc_rate: mean={existing['acc_rate_topup']['mean']:.4f}")


def main(N, L, beta_beta1_alpha_alpha1_list, argzz_xy_list, wilson_xy_list, mod=0,
         obs_fit_type="halfRefVil",
         heatbath_fraction=0.4,
         epsilon=0.05, n_leapfrog=20, mass_a=1.0, mass_z=1.0,
         s_step=0.5, s_update_num=1,
         n_therm=1000, n_meas=1000, meas_interval=10, n_total=4, n_workers=4):

    if mod not in (0, 1):
        raise ValueError(f"mod must be 0 or 1, got {mod}")
    if obs_fit_type not in ("RefVil", "halfRefVil"):
        raise ValueError(f"obs_fit_type must be 'RefVil' or 'halfRefVil', got {obs_fit_type!r}")

    pbar_params = tqdm(beta_beta1_alpha_alpha1_list, desc="Params", position=0)

    for beta, beta1, alpha, alpha1 in pbar_params:
        pbar_params.set_postfix_str(
            f"beta={beta:.3f}, beta1={beta1:.3f}, alpha={alpha:.3f}, alpha1={alpha1:.3f}, "
            f"mod={mod}, {obs_fit_type}")

        alpha_eff = alpha + alpha1
        data_dir = (f"data_{obs_fit_type}/data_PP_corr_N{N}_L{L}_mod{mod}/"
                    f"N{N}_L{L}_beta{beta:.3f}_alpha_eff{alpha_eff:.3f}")
        os.makedirs(data_dir, exist_ok=True)

        filename = (f"PP_corr_N{N}_L{L}_beta{beta:.3f}_beta1_{beta1:.3f}"
                    f"_alpha{alpha:.3f}_alpha1_{alpha1:.3f}.json")
        filepath = os.path.join(data_dir, filename)

        # If the file already exists, top up any requested sizes not yet measured
        # (a fresh sim for just the missing sizes, merged in). Otherwise the full
        # measurement below is performed.
        if os.path.exists(filepath):
            existing = json.load(open(filepath))
            ex_argzz_keys = set(existing.get("argzz_loop", {}).keys())
            ex_wilson_keys = set(existing.get("wilson_loop", {}).keys())
            missing_argzz = [xy for xy in argzz_xy_list
                             if f"{int(xy[0])}x{int(xy[1])}" not in ex_argzz_keys]
            missing_wilson = [xy for xy in wilson_xy_list
                              if f"{int(xy[0])}x{int(xy[1])}" not in ex_wilson_keys]
            if not missing_argzz and not missing_wilson:
                tqdm.write(f"File exists with all requested sizes, skipping: {filepath}")
                continue
            tqdm.write(f"File exists; topping up missing sizes: {filepath}")
            tqdm.write(f"  missing argzz: {missing_argzz}   missing wilson: {missing_wilson}")
            _topup_and_merge(filepath, existing, N, L, beta, beta1, alpha, alpha1, mod,
                             obs_fit_type, s_step, s_update_num,
                             heatbath_fraction, epsilon, n_leapfrog, mass_a, mass_z,
                             n_therm, n_meas, meas_interval, n_total, n_workers,
                             missing_argzz, missing_wilson)
            continue

        all_P_exp, all_conn_PP_corr, all_argzz_loop, all_wilson_loop, all_topo, \
            all_acc_hmc, all_acc_metro = _run_workers(
            N, L, beta, beta1, alpha, alpha1, mod, obs_fit_type,
            s_step, s_update_num,
            heatbath_fraction, epsilon, n_leapfrog,
            mass_a, mass_z, n_therm, n_meas, meas_interval,
            argzz_xy_list, wilson_xy_list, n_total, n_workers)

        M = all_P_exp.shape[0]
        tqdm.write(f"  Total measurements: {M}")

        P_exp_mean = np.mean(all_P_exp, axis=0)
        conn_PP_corr_mean = np.mean(all_conn_PP_corr, axis=0)

        P_flat = all_P_exp.reshape(M, -1)                     # (M, N*N) complex
        tau_int_P = integrated_autocorr_time(P_flat.T)         # (N*N,) complex → real tau per element
        tau_int_P = tau_int_P.reshape(N, N)

        C_flat = all_conn_PP_corr.reshape(M, -1)               # (M, L*L) real
        tau_int_C = integrated_autocorr_time(C_flat.T).reshape(L, L)

        tau_P_scalar = integrated_autocorr_time(np.trace(all_P_exp, axis1=1, axis2=2).real)
        tau_C_scalar = integrated_autocorr_time(np.sum(all_conn_PP_corr, axis=(1, 2)))

        P_exp_err = np.std(all_P_exp, axis=0, ddof=1) * np.sqrt(2.0 * np.maximum(tau_int_P, 0.5) / M)
        conn_PP_corr_err = np.std(all_conn_PP_corr, axis=0, ddof=1) * np.sqrt(2.0 * np.maximum(tau_int_C, 0.5) / M)

        argzz_loop_dict, wilson_loop_dict = _loop_stats(
            all_argzz_loop, all_wilson_loop, argzz_xy_list, wilson_xy_list, M)

        topo_dict = _topo_stats(all_topo, L, M)

        results = {
            "N": N,
            "L": L,
            "mod": mod,
            "obs_fit_type": obs_fit_type,
            "beta": beta,
            "beta1": beta1,
            "alpha": alpha,
            "alpha1": alpha1,
            "n_therm": n_therm,
            "n_meas": n_meas,
            "meas_interval": meas_interval,
            "n_total_sim": n_total,
            "n_workers": n_workers,
            "n_total_meas": int(M),
            "heatbath_fraction": heatbath_fraction,
            "epsilon": epsilon,
            "n_leapfrog": n_leapfrog,
            "mass_a": mass_a,
            "mass_z": mass_z,
            "s_step": s_step,
            "s_update_num": s_update_num,
            # Per-chain HMC (and s-Metropolis) acceptance rates, or null when
            # the ensemble used the heatbath sampler (no HMC).
            "acc_rate": _acc_stats(all_acc_hmc),
            "acc_rate_metro": _acc_stats(all_acc_metro),
            "argzz_xy_list": [[int(x), int(y)] for x, y in argzz_xy_list],
            "wilson_xy_list": [[int(x), int(y)] for x, y in wilson_xy_list],
            "P_exp": {
                "tau_scalar": float(tau_P_scalar),
                "tau_matrix": tau_int_P.tolist(),
                "mean_real": P_exp_mean.real.tolist(),
                "mean_imag": P_exp_mean.imag.tolist(),
                "err": P_exp_err.tolist(),
            },
            "argzz_loop": argzz_loop_dict,
            "wilson_loop": wilson_loop_dict,
            "conn_PP_corr": {
                "tau_scalar": float(tau_C_scalar),
                "tau_matrix": tau_int_C.tolist(),
                "mean": conn_PP_corr_mean.tolist(),
                "err": conn_PP_corr_err.tolist(),
            },
            "topo_charge": topo_dict,
        }

        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)

        tqdm.write(f"Finished: beta={beta}, beta1={beta1}, alpha={alpha}, alpha1={alpha1}")
        tqdm.write(f"tau_int(P_exp, scalar): {tau_P_scalar:.2f},  "
                   f"tau_int(conn_PP_corr, scalar): {tau_C_scalar:.2f}")
        if results["acc_rate"] is not None:
            tqdm.write(f"  HMC acc_rate: mean={results['acc_rate']['mean']:.4f} "
                       f"+/- {results['acc_rate']['std']:.4f}")
        if results["acc_rate_metro"] is not None:
            tqdm.write(f"  s-Metro acc_rate: mean={results['acc_rate_metro']['mean']:.4f} "
                       f"+/- {results['acc_rate_metro']['std']:.4f}")
        for name, d in topo_dict.items():
            tqdm.write(f"  topo {name}: mean={d['mean']:.4f}, topo_sus={d['topo_sus']:.6g}, "
                       f"tau={d['tau']:.2f}")
        for (x, y) in argzz_xy_list:
            d = argzz_loop_dict[f"{x}x{y}"]
            tqdm.write(f"  argzz({x}x{y}): tau={d['tau']:.3f}, mean={d['mean']:.6f}")
        for (x, y) in wilson_xy_list:
            d = wilson_loop_dict[f"{x}x{y}"]
            tqdm.write(f"  W({x}x{y}): tau={d['tau']}, mean={d['mean']}")


def filepath_for(N, L, beta, beta1, alpha, alpha1, mod, obs_fit_type):
    """The (cwd-relative) data path `main` writes for one coupling point.

    The subfolder is keyed by `alpha_eff = alpha + alpha1` (NOT by alpha/alpha1
    separately), so the obs beta1-scan and the topo alpha-scan at the same
    alpha_eff share a subfolder. Exposed so callers (plot_topo_alpha.py,
    full_renorm/step3b_topo.py) can locate a file without duplicating the template.
    """
    alpha_eff = alpha + alpha1
    data_dir = (f"data_{obs_fit_type}/data_PP_corr_N{N}_L{L}_mod{mod}/"
                f"N{N}_L{L}_beta{beta:.3f}_alpha_eff{alpha_eff:.3f}")
    filename = (f"PP_corr_N{N}_L{L}_beta{beta:.3f}_beta1_{beta1:.3f}"
                f"_alpha{alpha:.3f}_alpha1_{alpha1:.3f}.json")
    return os.path.join(data_dir, filename)


if __name__ == "__main__":
    N = 2
    L = 10
    argzz_xy_list = [[a, a] for a in [1]]
    wilson_xy_list = [[a, a] for a in [1]]
    beta_beta1_alpha_alpha1_list = [
        (1.553, b1, 0.470, 0.0) for b1 in np.arange(-1.5, -0.5, 0.1)
    ]
    # beta_beta1_alpha_alpha1_list = [(1.457, -1.220, 0.535, 0.0)]
    mod = 1
    obs_fit_type = "halfRefVil"  # "RefVil" uses the full-Villain sampler (+ Q_s topo)

    main(
        N=N, L=L, argzz_xy_list=argzz_xy_list, wilson_xy_list=wilson_xy_list,
        beta_beta1_alpha_alpha1_list=beta_beta1_alpha_alpha1_list,
        mod=mod, obs_fit_type=obs_fit_type,
        heatbath_fraction=0.4,
        epsilon=0.05, n_leapfrog=20, mass_a=1.0, mass_z=1.0,
        s_step=0.5, s_update_num=1,
        n_therm=1000, n_meas=5000, meas_interval=1, n_total=20, n_workers=5,
    )
