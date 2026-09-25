#!/usr/bin/env python3
"""Step 3a of the new full renorm: the 1-plaquette fit -> alpha_c (one of two
alpha methods; the other is the topo-sus match in step3b_topo.py).

The 1plaq full-Villain code keeps alpha and alpha1 separate and fits THREE
independent plaquette couplings (U / z / s vortex windings) on the same samples.
We select the one matching renorm_type: "s" for U-renorm, "z" for z-renorm.
Then alpha1_c = alpha_eff_c (from step 1) - alpha_c is computed by the
orchestrator. Independent of beta_c/beta1_c -- it only needs the fine model.

Run as a subprocess with cwd=CPN_1plaq_fit. Reuses fit_v_dist_batch.fit_one_case
and replicates the 1plaq data-generation pool.
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

# --- module-level path setup (see step1_2plaq.py for the spawn rationale) ----
_HERE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.dirname(_HERE)
_SUBPROJECT = os.path.join(_BASE, "CPN_1plaq_fit")
if _SUBPROJECT not in sys.path:
    sys.path.insert(0, _SUBPROJECT)

from func.func_CPN_RefVil_1plaq_fit import run_RefVil_HMC, get_X_freq
from func.func_CPN_Original_1plaq_fit import run_Original
from func.data_provenance import (data_fingerprint_1plaq,
                                  save_npz_with_fingerprint, decide_data_action)
import step_common

# Column of v_array -> boundary flux used by get_X_freq (matches
# CPN_1plaq_data_generate.py: U->da_U, z->da_z, s->da_U).
_DA_FOR_COL = {0: "da_U", 1: "da_z", 2: "da_U"}


def generate_1plaq_case(cfg):
    """Generate (or reuse) the 1plaq train npz for the fine model.

    Mirrors CPN_1plaq_data_generate.py for ONE (beta, beta1, alpha, alpha1)
    case, train mode only. Returns (filepath, use_Original, provenance) where
    provenance records whether the data was reused / generated / regenerated and
    the sampler fingerprint.
    """
    N, L, mod = cfg["N"], cfg["L"], cfg["mod"]
    beta, beta1 = cfg["beta_f"], cfg["beta1_f"]
    alpha, alpha1 = cfg["alpha_f"], cfg["alpha1_f"]
    BC, pad = cfg["BC"], cfg["pad"]
    p = cfg["1plaq"]

    fit_times, zero_pad = p["fit_times"], p["zero_pad"]
    n_therm, meas_interval, n_meas = p["n_therm"], p["meas_interval"], p["n_meas"]
    boundary_n_therm = p["boundary_n_therm"]
    s_step, s_update_num = p["s_step"], p["s_update_num"]
    epsilon, n_leapfrog = p["epsilon"], p["n_leapfrog"]
    mass_a, mass_z = p["mass_a"], p["mass_z"]
    hf_therm, hf_meas = p["hf_therm"], p["hf_meas"]
    n_workers = p["n_workers"]

    file_mod = step_common.mod_for(beta1, mod)
    use_Original = step_common.use_Original_1plaq(beta1, alpha)

    folder = f"data/data_{BC}_pad{pad}_mod{file_mod}/N{N}_L{L}"
    filename = (
        f"data_beta{beta:.3f}_beta1_{beta1:.3f}_alpha{alpha:.3f}_alpha1_{alpha1:.3f}_"
        f"num{fit_times}_zpad{zero_pad}_N{N}_L{L}_{BC}_pad{pad}_mod{file_mod}.npz"
    )
    filepath = os.path.join(folder, filename)

    # The full sampler config that this file's content depends on. The filename
    # encodes only couplings + fit_times/zero_pad, NOT the MC knobs, so existence
    # alone is not a safe reuse signal. We stamp this fingerprint into the .npz
    # at write time and compare it on reuse.
    requested_fp = data_fingerprint_1plaq(
        N=N, L=L, mod=mod, beta_f=beta, beta1_f=beta1,
        alpha_f=alpha, alpha1_f=alpha1, BC=BC, pad=pad,
        fit_times=fit_times, zero_pad=zero_pad,
        n_therm=n_therm, meas_interval=meas_interval, n_meas=n_meas,
        boundary_n_therm=boundary_n_therm,
        s_step=s_step, s_update_num=s_update_num,
        epsilon=epsilon, n_leapfrog=n_leapfrog, mass_a=mass_a, mass_z=mass_z,
        hf_therm=hf_therm, hf_meas=hf_meas)

    auto_regenerate = cfg.get("auto_regenerate", True)
    existed = os.path.exists(filepath)
    action, stored_fp, diff = decide_data_action(
        filepath, requested_fp, auto_regenerate=auto_regenerate)
    if action == "reuse":
        print(f"[1plaq] reusing existing data (fingerprint matches): {filepath}")
        return (filepath, use_Original,
                {"action": "reuse", "requested_fingerprint": requested_fp,
                 "stored_fingerprint": stored_fp, "diff": []})
    if action == "raise":
        raise RuntimeError(
            f"[1plaq] existing data does not match the current sampler config "
            f"(auto_regenerate=False):\n  file: {filepath}\n  "
            + "\n  ".join(diff)
            + "\nDelete the file or set auto_regenerate=True (in the orchestrator) "
            "to overwrite it.")
    label = "regenerate" if existed else "generate"
    if existed:
        print(f"[1plaq] regenerating data ({'; '.join(diff)}): {filepath}")
    else:
        print(f"[1plaq] generating data: {filepath}")

    if use_Original:
        args = (N, L, beta, alpha1, BC, pad, boundary_n_therm, n_therm, n_meas,
                meas_interval, hf_therm, hf_meas)
        run_func = run_Original
    else:
        args = (N, L, beta, beta1, alpha, alpha1, BC, pad, mod, boundary_n_therm,
                s_step, s_update_num, n_therm, n_meas, meas_interval,
                epsilon, n_leapfrog, mass_a, mass_z)
        run_func = run_RefVil_HMC

    acc = {tag: ([], []) for tag in ("U", "z", "s")}  # tag -> (X_list, freq_list)
    acc_hmc_list, acc_metro_list = [], []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(run_func, args) for _ in range(fit_times)]
        for fut in tqdm(as_completed(futures), total=fit_times,
                        desc=f"1plaq b={beta:.3f} b1={beta1:.3f} a={alpha:.3f} a1={alpha1:.3f}"):
            v_array, da_U, da_z, acc_hmc, acc_metro = fut.result()
            acc_hmc_list.append(acc_hmc)
            acc_metro_list.append(acc_metro)
            da_map = {"da_U": da_U, "da_z": da_z}
            for col, tag in zip((0, 1, 2), ("U", "z", "s")):
                x_arr, f_arr = get_X_freq(v_array[:, col], da_map[_DA_FOR_COL[col]],
                                          zero_pad=zero_pad)
                acc[tag][0].extend(x_arr)
                acc[tag][1].extend(f_arr)

    savez_kwargs = {}
    for tag in ("U", "z", "s"):
        x_list, f_list = acc[tag]
        savez_kwargs[f"X_{tag}"] = np.array(x_list)
        savez_kwargs[f"freq_{tag}"] = np.array(f_list)
    # HMC runs carry per-chain HMC / s-Metropolis acceptance rates; heatbath
    # runs (all-None acc) store no acc keys at all (readers report null).
    if any(a is not None for a in acc_hmc_list):
        savez_kwargs["acc_rate"] = np.array(acc_hmc_list, dtype=float)
        print(f"[1plaq] HMC acc_rate: mean = {np.mean(savez_kwargs['acc_rate']):.4f}")
    if any(a is not None for a in acc_metro_list):
        savez_kwargs["acc_rate_metro"] = np.array(acc_metro_list, dtype=float)
        print(f"[1plaq] s-Metro acc_rate: mean = {np.mean(savez_kwargs['acc_rate_metro']):.4f}")
    # save_npz_with_fingerprint creates the folder and stamps the sampler
    # config into the .npz in the same savez call.
    save_npz_with_fingerprint(filepath, savez_kwargs, requested_fp)
    print(f"[1plaq] wrote data: {filepath}")
    return (filepath, use_Original,
            {"action": label, "requested_fingerprint": requested_fp,
             "stored_fingerprint": stored_fp, "diff": diff})


def run_step(cfg):
    from fit_v_dist_batch import fit_one_case as fit_vortex

    N, L, mod = cfg["N"], cfg["L"], cfg["mod"]
    beta, beta1 = cfg["beta_f"], cfg["beta1_f"]
    alpha, alpha1 = cfg["alpha_f"], cfg["alpha1_f"]
    renorm_type = cfg["renorm_type"]
    BC, pad = cfg["BC"], cfg["pad"]
    p = cfg["1plaq"]

    data_file, use_Original, data_provenance = generate_1plaq_case(cfg)
    entry_mod = step_common.mod_for(beta1, mod)
    data_folder = f"data/data_{BC}_pad{pad}_mod{entry_mod}/N{N}_L{L}"
    output_root = f"results/v_dist_fit_{BC}_pad{pad}_mod{entry_mod}"

    out_path, result = fit_vortex(
        n_val=N, l_val=L, beta=beta, beta1=beta1, alpha=alpha, alpha1=alpha1,
        boundary_BC=BC, boundary_pad=pad, mod=entry_mod,
        data_folder=data_folder, output_root=output_root,
        p0=p["p0"], fit_times_train=p["fit_times"], zero_pad=p["zero_pad"],
    )

    tag = "s" if renorm_type == "U" else "z"
    chosen = result["villain_fit"][tag]
    if renorm_type == "U" and use_Original:
        raise RuntimeError(
            "U-renorm selects the s-vortex, but the fine model (beta1_f~=0, "
            "alpha_f~=0) used run_Original where vs≡0, so the s-fit is degenerate. "
            "Use renorm_type='z', or set alpha_f/beta1_f ≠ 0."
        )
    if not chosen.get("success"):
        raise RuntimeError(f"1plaq {tag}-vortex fit failed: {chosen.get('error')}")

    return {
        "step": "1plaq",
        "renorm_type": renorm_type, "selected_tag": tag, "entry_mod": entry_mod,
        "use_Original": use_Original,
        "beta_f": beta, "beta1_f": beta1, "alpha_f": alpha, "alpha1_f": alpha1,
        "N": N, "L": L,
        "alpha_c": chosen["alpha_fit"], "alpha_c_err": chosen["alpha_fit_err"],
        "villain_mse_train": chosen["villain_mse_train"], "success": chosen["success"],
        "all_three": result["villain_fit"],
        "acc_rate": result.get("acc_rate"),
        "acc_rate_metro": result.get("acc_rate_metro"),
        "data_file": os.path.relpath(data_file, _SUBPROJECT),
        "data_provenance": data_provenance,
        "fit_out_path": os.path.relpath(out_path, _SUBPROJECT),
    }


if __name__ == "__main__":
    cfg = step_common.load_json(sys.argv[1])
    os.chdir(_SUBPROJECT)
    result = run_step(cfg)
    step_common.dump_json(result, cfg["_result_path"])
    print(f"[1plaq] tag={result['selected_tag']} alpha_c={result['alpha_c']:.6g} "
          f"mse={result['villain_mse_train']:.3e}")
