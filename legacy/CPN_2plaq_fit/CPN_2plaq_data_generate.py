import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import sys
# Make the local `func` package importable regardless of the working directory
# (so the script runs from CPN_renorm_fit/ or this subproject alike).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from func.func_CPN_Original_2plaq_fit import run_Original
from func.func_CPN_halfRefVil_2plaq_fit import run_halfRefVil_HMC
from func.data_provenance import (data_fingerprint_2plaq,
                                  save_npz_with_fingerprint, decide_data_action)

if __name__ == "__main__":
    N = 2
    L = 1
    boundary_BC = 'OBC'
    boundary_pad = 5
    mod = 1  # halfRefVil simulation mode: 0 or 1 (ignored for beta1=0 Original; naming uses mod=0)

    # List of (beta, beta1, alpha) triples to generate data for
    beta_beta1_alpha_list = [(0.5, 1.0, 0.5), (1.0, 1.0, 1.0)]
    
    # Choice of what data to generate: 'train', 'test', or 'both'
    generate_choice = 'train'  # options: 'train', 'test', or 'both'

    # Reference fit_times for each mode
    fit_times_train = 4000   # Number of fits for training data
    fit_times_test = 1000    # Number of fits for test data

    # boundary thermalization settings
    boundary_n_therm = 200

    bins = 60

    save_data = True
    n_workers = 6

    # Policy when an existing data file's stored sampler config does NOT match
    # the current knobs (a mismatch, or a legacy file with no stored config):
    #   True  -> overwrite the file with fresh MC (default).
    #   False -> raise an error naming the file and the mismatch, so you must
    #            explicitly delete it or align the config before regenerating.
    # (A file that simply does not exist yet is always generated.)
    auto_regenerate = True

    n_therm = 400
    meas_interval = 1
    n_meas = bins * 50 * meas_interval

    hf_therm = 0.4
    hf_meas = 0.4

    epsilon = 0.05
    n_leapfrog = 20
    mass_a = 1
    mass_z = 1

    # Outer loop: iterate over each (beta, beta1, alpha) triple
    for beta, beta1, alpha in beta_beta1_alpha_list:
        
        # mod used for file naming: Original (beta1=0) has no mod concept -> force 0
        file_mod = 0 if np.abs(beta1) < 1e-10 else mod

        # Build args tuple for this (beta, beta1, alpha) triple
        if np.abs(beta1) < 1e-10:
            args = (N, L, beta, alpha, boundary_BC, boundary_pad, boundary_n_therm, n_therm, n_meas, meas_interval, hf_therm, hf_meas, bins)
            run_func = run_Original
        else:
            args = (N, L, beta, beta1, alpha, boundary_BC, boundary_pad, mod, boundary_n_therm, n_therm, n_meas, meas_interval, epsilon, n_leapfrog, mass_a, mass_z, bins)
            run_func = run_halfRefVil_HMC

        # Inner logic: decide what to generate based on generate_choice
        generate_modes = []
        if generate_choice in ['train', 'both']:
            generate_modes.append(('train', fit_times_train, False))
        if generate_choice in ['test', 'both']:
            generate_modes.append(('test', fit_times_test, True))
        
        # Generate data for each mode (train, test, or both)
        for mode_name, fit_times, is_test in generate_modes:
            
            # Build folder and filename
            folder_path = f'data/data_{boundary_BC}_pad{boundary_pad}_mod{file_mod}/N{N}_L{L}'
            if is_test:
                filename = f'data_beta{beta:.3f}_beta1_{beta1:.3f}_alpha{alpha:.3f}_num{fit_times}x{bins}_test_N{N}_L{L}_{boundary_BC}_pad{boundary_pad}_mod{file_mod}.npz'
            else:
                filename = f'data_beta{beta:.3f}_beta1_{beta1:.3f}_alpha{alpha:.3f}_num{fit_times}x{bins}_N{N}_L{L}_{boundary_BC}_pad{boundary_pad}_mod{file_mod}.npz'
            
            filepath = os.path.join(folder_path, filename)

            # Validate provenance: reuse only if the stored sampler config matches
            # the current knobs. The filename alone is NOT enough -- it omits
            # n_therm/epsilon/hf_*/mass_*/..., so a filename-only check would
            # silently reuse stale data when only an MC knob changes. On mismatch
            # or legacy (no stored config) we regenerate the file with fresh MC.
            requested_fp = data_fingerprint_2plaq(
                N=N, L=L, mod=mod, beta_f=beta, beta1_f=beta1, alpha_eff_f=alpha,
                BC=boundary_BC, pad=boundary_pad, fit_times=fit_times, bins=bins,
                n_therm=n_therm, meas_interval=meas_interval, n_meas=n_meas,
                boundary_n_therm=boundary_n_therm, hf_therm=hf_therm, hf_meas=hf_meas,
                epsilon=epsilon, n_leapfrog=n_leapfrog, mass_a=mass_a, mass_z=mass_z)
            action, _stored_fp, diff = decide_data_action(
                filepath, requested_fp, auto_regenerate=auto_regenerate)
            if action == "reuse":
                print(f"Skipping {mode_name} for β={beta:.3f}, β1={beta1:.3f}, α={alpha:.3f} (fingerprint matches: {filename})")
                continue
            if action == "raise":
                raise RuntimeError(
                    f"Existing data file does not match the current sampler config "
                    f"(auto_regenerate=False):\n  file: {filepath}\n  "
                    + "\n  ".join(diff)
                    + "\nDelete the file or set auto_regenerate=True to overwrite it.")
            if os.path.exists(filepath):
                print(f"Regenerating {mode_name} for β={beta:.3f}, β1={beta1:.3f}, α={alpha:.3f} ({'; '.join(diff)}): {filename}")
            else:
                print(f"Generating {mode_name} for β={beta:.3f}, β1={beta1:.3f}, α={alpha:.3f}: {filename}")
            
            freq_U = []
            freq_z = []
            a_list_U = []
            a_list_z = []
            phi_list = []
            acc_list = []
            
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                futures = [pool.submit(run_func, args) for _ in range(fit_times)]
                
                for fut in tqdm(as_completed(futures), total=fit_times, desc=f"Pair(β={beta:.3f}, β1={beta1:.3f}, α={alpha:.3f}) {mode_name}"):
                    fU, fz, aU, az, phi, acc = fut.result()
                    freq_U.append(fU)
                    freq_z.append(fz)
                    a_list_U.append(aU)
                    a_list_z.append(az)
                    phi_list.append(phi)
                    acc_list.append(acc)

            freq_U = np.array(freq_U)
            freq_z = np.array(freq_z)
            a_list_U = np.array(a_list_U)
            a_list_z = np.array(a_list_z)
            phi_list = np.array(phi_list)
            
            print(f"Finished {mode_name} for β={beta:.3f}, β1={beta1:.3f}, α={alpha:.3f}")
            print(f"shape(freq_U) = {freq_U.shape}, shape(freq_z) = {freq_z.shape}, shape(a_list_U) = {a_list_U.shape}, shape(a_list_z) = {a_list_z.shape}, shape(phi_list) = {phi_list.shape}")
            
            if save_data:
                # save_npz_with_fingerprint creates the folder and stamps the
                # sampler config into the .npz in a single savez call.
                arrays = dict(freq_U=freq_U, freq_z=freq_z, a_list_U=a_list_U,
                              a_list_z=a_list_z, phi_list=phi_list)
                # HMC runs carry a per-chain acceptance rate; heatbath runs
                # (all-None acc) store no acc_rate key at all.
                if any(a is not None for a in acc_list):
                    arrays["acc_rate"] = np.array(acc_list, dtype=float)
                    print(f"HMC acc_rate: mean = {np.mean(arrays['acc_rate']):.4f}, "
                          f"std = {np.std(arrays['acc_rate'], ddof=1) if fit_times > 1 else 0.0:.4f}")
                save_npz_with_fingerprint(filepath, arrays, requested_fp)