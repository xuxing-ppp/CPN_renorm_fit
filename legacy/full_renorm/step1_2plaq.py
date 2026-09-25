#!/usr/bin/env python3
"""Step 1 of the full renorm: the 2-plaquette fit -> beta_c and alpha_eff_c.

The 2plaq code has a single plaquette coupling `alpha` that folds together what
the full RefVil model calls alpha + alpha1. So for the fine model we feed it
`alpha_eff_f = alpha_f + alpha1_f`. Fitting the Berry-connection (a) distribution
extracts beta_c and alpha_eff_c (= alpha_c + alpha1_c).

Run as a subprocess with cwd=CPN_2plaq_fit (set by the orchestrator). Reuses
fit_a_dist_batch.fit_one_case verbatim and replicates the 2plaq data-generation
pool (it is not factored into an importable function in the subproject).
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

# --- module-level path setup ------------------------------------------------
# Computed from __file__ (not argv/cwd) so it also runs when a `spawn` worker
# re-imports this module as __mp_main__ -- without this the worker cannot find
# the colliding top-level `func` package and fails to unpickle run_func.
_HERE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.dirname(_HERE)
_SUBPROJECT = os.path.join(_BASE, "CPN_2plaq_fit")
if _SUBPROJECT not in sys.path:
    sys.path.insert(0, _SUBPROJECT)

from func.func_CPN_Original_2plaq_fit import run_Original
from func.func_CPN_halfRefVil_2plaq_fit import run_halfRefVil_HMC
from func.data_provenance import (data_fingerprint_2plaq,
                                  save_npz_with_fingerprint, decide_data_action)
import step_common


def generate_2plaq_case(cfg):
    """Generate (or reuse) the 2plaq train npz for the fine model.

    Mirrors CPN_2plaq_data_generate.py for ONE (beta, beta1, alpha) case, train
    mode only. Returns (filepath, provenance) where provenance records whether
    the data was reused / generated / regenerated and the sampler fingerprint.
    """
    N, L, mod = cfg["N"], cfg["L"], cfg["mod"]
    beta, beta1, alpha = cfg["beta_f"], cfg["beta1_f"], cfg["alpha_eff_f"]
    BC, pad = cfg["BC"], cfg["pad"]
    p = cfg["2plaq"]

    fit_times, bins = p["fit_times"], p["bins"]
    n_therm, meas_interval = p["n_therm"], p["meas_interval"]
    n_meas = bins * p["n_meas_factor"] * meas_interval
    boundary_n_therm = p["boundary_n_therm"]
    hf_therm, hf_meas = p["hf_therm"], p["hf_meas"]
    epsilon, n_leapfrog = p["epsilon"], p["n_leapfrog"]
    mass_a, mass_z = p["mass_a"], p["mass_z"]
    n_workers = p["n_workers"]

    file_mod = step_common.mod_for(beta1, mod)
    folder = f"data/data_{BC}_pad{pad}_mod{file_mod}/N{N}_L{L}"
    filename = (
        f"data_beta{beta:.3f}_beta1_{beta1:.3f}_alpha{alpha:.3f}_"
        f"num{fit_times}x{bins}_N{N}_L{L}_{BC}_pad{pad}_mod{file_mod}.npz"
    )
    filepath = os.path.join(folder, filename)

    # The full sampler config that this file's content depends on. The filename
    # encodes only couplings + fit_times/bins, NOT the MC knobs, so existence
    # alone is not a safe reuse signal. We stamp this fingerprint into the .npz
    # at write time and compare it on reuse.
    requested_fp = data_fingerprint_2plaq(
        N=N, L=L, mod=mod, beta_f=beta, beta1_f=beta1, alpha_eff_f=alpha,
        BC=BC, pad=pad, fit_times=fit_times, bins=bins,
        n_therm=n_therm, meas_interval=meas_interval, n_meas=n_meas,
        boundary_n_therm=boundary_n_therm, hf_therm=hf_therm, hf_meas=hf_meas,
        epsilon=epsilon, n_leapfrog=n_leapfrog, mass_a=mass_a, mass_z=mass_z)

    auto_regenerate = cfg.get("auto_regenerate", True)
    existed = os.path.exists(filepath)
    action, stored_fp, diff = decide_data_action(
        filepath, requested_fp, auto_regenerate=auto_regenerate)
    if action == "reuse":
        print(f"[2plaq] reusing existing data (fingerprint matches): {filepath}")
        return filepath, {"action": "reuse", "requested_fingerprint": requested_fp,
                          "stored_fingerprint": stored_fp, "diff": []}
    if action == "raise":
        raise RuntimeError(
            f"[2plaq] existing data does not match the current sampler config "
            f"(auto_regenerate=False):\n  file: {filepath}\n  "
            + "\n  ".join(diff)
            + "\nDelete the file or set auto_regenerate=True (in the orchestrator) "
            "to overwrite it.")
    label = "regenerate" if existed else "generate"
    if existed:
        print(f"[2plaq] regenerating data ({'; '.join(diff)}): {filepath}")
    else:
        print(f"[2plaq] generating data: {filepath}")

    if abs(beta1) < 1e-10:
        args = (N, L, beta, alpha, BC, pad, boundary_n_therm, n_therm, n_meas,
                meas_interval, hf_therm, hf_meas, bins)
        run_func = run_Original
    else:
        args = (N, L, beta, beta1, alpha, BC, pad, mod, boundary_n_therm, n_therm,
                n_meas, meas_interval, epsilon, n_leapfrog, mass_a, mass_z, bins)
        run_func = run_halfRefVil_HMC

    freq_U, freq_z, a_list_U, a_list_z, phi_list = [], [], [], [], []
    acc_list = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(run_func, args) for _ in range(fit_times)]
        for fut in tqdm(as_completed(futures), total=fit_times,
                        desc=f"2plaq b={beta:.3f} b1={beta1:.3f} a={alpha:.3f}"):
            fU, fz, aU, az, phi, acc = fut.result()
            freq_U.append(fU)
            freq_z.append(fz)
            a_list_U.append(aU)
            a_list_z.append(az)
            phi_list.append(phi)
            acc_list.append(acc)

    arrays = dict(freq_U=np.array(freq_U), freq_z=np.array(freq_z),
                  a_list_U=np.array(a_list_U), a_list_z=np.array(a_list_z),
                  phi_list=np.array(phi_list))
    # HMC runs carry a per-chain acceptance rate; heatbath runs (all-None acc)
    # store no acc_rate key at all (readers then report acc_rate=null).
    if any(a is not None for a in acc_list):
        arrays["acc_rate"] = np.array(acc_list, dtype=float)
        print(f"[2plaq] HMC acc_rate: mean = {np.mean(arrays['acc_rate']):.4f}")

    save_npz_with_fingerprint(filepath, arrays, requested_fp)
    print(f"[2plaq] wrote data: {filepath}")
    return filepath, {"action": label, "requested_fingerprint": requested_fp,
                      "stored_fingerprint": stored_fp, "diff": diff}


def run_step(cfg):
    from fit_a_dist_batch import fit_one_case

    N, L, mod = cfg["N"], cfg["L"], cfg["mod"]
    beta, beta1, alpha = cfg["beta_f"], cfg["beta1_f"], cfg["alpha_eff_f"]
    conn_type = cfg["renorm_type"]  # "U" or "z" -- identical to 2plaq conn_type
    BC, pad = cfg["BC"], cfg["pad"]
    p = cfg["2plaq"]

    data_file, data_provenance = generate_2plaq_case(cfg)
    entry_mod = step_common.mod_for(beta1, mod)
    data_folder = f"data/data_{BC}_pad{pad}_mod{entry_mod}/N{N}_L{L}"
    output_root = f"results/a_dist_fit_{BC}_pad{pad}_mod{entry_mod}"

    out_path, result = fit_one_case(
        n_val=N, l_val=L, beta=beta, beta1=beta1, alpha=alpha,
        conn_type=conn_type, boundary_BC=BC, boundary_pad=pad, mod=entry_mod,
        data_folder=data_folder, output_root=output_root,
        p0=p["p0"], fit_times_train=p["fit_times"], bins=p["bins"],
    )
    vf = result["villain_fit"]
    return {
        "step": "2plaq",
        "renorm_type": conn_type, "conn_type": conn_type, "entry_mod": entry_mod,
        "beta_f": beta, "beta1_f": beta1, "alpha_eff_f": alpha, "N": N, "L": L,
        "beta_c": vf["beta_fit"], "beta_c_err": vf["beta_fit_err"],
        "alpha_eff_c": vf["alpha_fit"], "alpha_eff_c_err": vf["alpha_fit_err"],
        "villain_mse_train": vf["villain_mse_train"],
        "acc_rate": result.get("acc_rate"),
        "data_file": os.path.relpath(data_file, _SUBPROJECT),
        "data_provenance": data_provenance,
        "fit_out_path": os.path.relpath(out_path, _SUBPROJECT),
    }


if __name__ == "__main__":
    cfg = step_common.load_json(sys.argv[1])
    os.chdir(_SUBPROJECT)
    result = run_step(cfg)
    step_common.dump_json(result, cfg["_result_path"])
    print(f"[2plaq] beta_c={result['beta_c']:.6g} "
          f"alpha_eff_c={result['alpha_eff_c']:.6g} "
          f"mse={result['villain_mse_train']:.3e}")
