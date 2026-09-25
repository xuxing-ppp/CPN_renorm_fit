import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import sys
# Make the local `func` package importable regardless of the working directory
# (so the script runs from CPN_renorm_fit/ or this subproject alike).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from func.func_CPN_RefVil_1plaq_fit import run_RefVil_HMC, get_X_freq
from func.func_CPN_Original_1plaq_fit import run_Original
from func.data_provenance import (data_fingerprint_1plaq,
                                  save_npz_with_fingerprint, decide_data_action)

if __name__ == "__main__":
    N = 2
    L = 4
    boundary_BC = 'OBC'
    boundary_pad = 5
    mod = 1  # RefVil simulation mode: 0 or 1 (beta1=0 Original entries always use mod=0)

    # List of (beta, beta1, alpha, alpha1) quadruples to generate data for
    beta_beta1_alpha_alpha1_list = [(b, 0.0, 0.0, 0.0) for b in [1.0, 1.5, 2.0, 3.0]]

    # Choice of what data to generate: 'train', 'test', or 'both'
    generate_choice = 'train'  # options: 'train', 'test', or 'both'

    # Reference fit_times for each mode
    fit_times_train = 1000   # Number of fits for training data
    fit_times_test = 1000    # Number of fits for test data

    n_therm = 400
    meas_interval = 1
    n_meas = 1000  # vortex samples per fit (integer-valued; a few thousand is ample)

    # boundary thermalization settings
    boundary_n_therm = 200

    # zero_pad: extra empty integer bins appended on both sides of each run's
    # observed vortex range (handled inside run_RefVil_HMC). Lets the fit see
    # the tails without each run needing to actually sample them.
    zero_pad = 2

    save_data = True
    n_workers = 6

    # Policy when an existing data file's stored sampler config does NOT match
    # the current knobs (a mismatch, or a legacy file with no stored config):
    #   True  -> overwrite the file with fresh MC (default).
    #   False -> raise an error naming the file and the mismatch, so you must
    #            explicitly delete it or align the config before regenerating.
    # (A file that simply does not exist yet is always generated.)
    auto_regenerate = True

    # Villain integer-field (s) metropolis settings
    s_step = 0.5
    s_update_num = 1

    epsilon = 0.05
    n_leapfrog = 20
    mass_a = 1
    mass_z = 1

    # Original-model (beta1=0, alpha=0) heatbath fractions; used only by run_Original
    hf_therm = 0.4   # heatbath fraction during interior thermalization
    hf_meas = 0.4    # heatbath fraction during measurement

    # Outer loop: iterate over each (beta, beta1, alpha, alpha1) quadruple
    for beta, beta1, alpha, alpha1 in beta_beta1_alpha_alpha1_list:

        # mod used for file naming: Original (beta1=0) has no mod concept -> force 0
        file_mod = 0 if np.abs(beta1) < 1e-10 else mod

        # When beta1 = alpha = 0 the action has no Villain integer field s and
        # reduces to the Original (cos-plaquette) model: use the faster heatbath
        # sampler run_Original. Otherwise run the full-Villain HMC sampler.
        # Both return (v_array(n_meas_kept, 3)=[vU, vz, vs], da_U, da_z), so the
        # histogram-building below is identical. For run_Original, vs == 0.
        use_Original = (np.abs(beta1) < 1e-10) and (np.abs(alpha) < 1e-10)

        if use_Original:
            # run_Original expects:
            #   (N, L, beta, alpha1, boundary_BC, boundary_pad,
            #    boundary_n_therm, n_therm, n_meas, meas_interval, hf_therm, hf_meas)
            args = (N, L, beta, alpha1, boundary_BC, boundary_pad,
                    boundary_n_therm, n_therm, n_meas, meas_interval, hf_therm, hf_meas)
            run_func = run_Original
        else:
            # run_RefVil_HMC expects:
            #   (N, L, beta, beta1, alpha, alpha1, boundary_BC, boundary_pad, mod,
            #    boundary_n_therm, s_step, s_update_num, n_therm, n_meas,
            #    meas_interval, epsilon, n_leapfrog, mass_a, mass_z)
            args = (N, L, beta, beta1, alpha, alpha1, boundary_BC, boundary_pad, mod,
                    boundary_n_therm, s_step, s_update_num, n_therm, n_meas,
                    meas_interval, epsilon, n_leapfrog, mass_a, mass_z)
            run_func = run_RefVil_HMC

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
                filename = f'data_beta{beta:.3f}_beta1_{beta1:.3f}_alpha{alpha:.3f}_alpha1_{alpha1:.3f}_num{fit_times}_zpad{zero_pad}_test_N{N}_L{L}_{boundary_BC}_pad{boundary_pad}_mod{file_mod}.npz'
            else:
                filename = f'data_beta{beta:.3f}_beta1_{beta1:.3f}_alpha{alpha:.3f}_alpha1_{alpha1:.3f}_num{fit_times}_zpad{zero_pad}_N{N}_L{L}_{boundary_BC}_pad{boundary_pad}_mod{file_mod}.npz'

            filepath = os.path.join(folder_path, filename)

            # Validate provenance: reuse only if the stored sampler config matches
            # the current knobs. The filename alone is NOT enough -- it omits
            # n_therm/epsilon/s_step/hf_*/mass_*/..., so a filename-only check
            # would silently reuse stale data when only an MC knob changes. On
            # mismatch or legacy (no stored config) we regenerate with fresh MC.
            requested_fp = data_fingerprint_1plaq(
                N=N, L=L, mod=mod, beta_f=beta, beta1_f=beta1,
                alpha_f=alpha, alpha1_f=alpha1, BC=boundary_BC, pad=boundary_pad,
                fit_times=fit_times, zero_pad=zero_pad,
                n_therm=n_therm, meas_interval=meas_interval, n_meas=n_meas,
                boundary_n_therm=boundary_n_therm,
                s_step=s_step, s_update_num=s_update_num,
                epsilon=epsilon, n_leapfrog=n_leapfrog, mass_a=mass_a, mass_z=mass_z,
                hf_therm=hf_therm, hf_meas=hf_meas)
            action, _stored_fp, diff = decide_data_action(
                filepath, requested_fp, auto_regenerate=auto_regenerate)
            if action == "reuse":
                print(f"Skipping {mode_name} for β={beta:.3f}, β1={beta1:.3f}, α={alpha:.3f}, α1={alpha1:.3f} (fingerprint matches: {filename})")
                continue
            if action == "raise":
                raise RuntimeError(
                    f"Existing data file does not match the current sampler config "
                    f"(auto_regenerate=False):\n  file: {filepath}\n  "
                    + "\n  ".join(diff)
                    + "\nDelete the file or set auto_regenerate=True to overwrite it.")
            if os.path.exists(filepath):
                print(f"Regenerating {mode_name} for β={beta:.3f}, β1={beta1:.3f}, α={alpha:.3f}, α1={alpha1:.3f} ({'; '.join(diff)}): {filename}")
            else:
                print(f"Generating {mode_name} for β={beta:.3f}, β1={beta1:.3f}, α={alpha:.3f}, α1={alpha1:.3f}: {filename}")

            # Each run produces its own ragged histogram (bin count depends on the
            # observed vortex range + zero_pad), so collect the raw X_run / freq_run
            # and store them as object arrays rather than re-binning onto a fixed grid.
            X_U, freq_U = [], []
            X_z, freq_z = [], []
            X_s, freq_s = [], []
            acc_hmc_list, acc_metro_list = [], []

            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                futures = [pool.submit(run_func, args) for _ in range(fit_times)]

                for fut in tqdm(as_completed(futures), total=fit_times,
                                desc=f"Quad(β={beta:.3f}, β1={beta1:.3f}, α={alpha:.3f}, α1={alpha1:.3f}) {mode_name}"):
                    # run_RefVil_HMC / run_Original both return
                    #   (v_array, da_U, da_z, acc_hmc, acc_metro):
                    #   v_array : (n_meas_kept, 3) = [vU, vz, vs] (vs == 0 for Original)
                    #   acc_*   : HMC / s-Metropolis acceptance rates (None for
                    #             the heatbath run_Original)
                    v_array, da_U, da_z, acc_hmc, acc_metro = fut.result()

                    x_U, f_U = get_X_freq(v_array[:, 0], da_U, zero_pad=zero_pad)
                    X_U.extend(x_U)
                    freq_U.extend(f_U)

                    x_z, f_z = get_X_freq(v_array[:, 1], da_z, zero_pad=zero_pad)
                    X_z.extend(x_z)
                    freq_z.extend(f_z)

                    x_s, f_s = get_X_freq(v_array[:, 2], da_U, zero_pad=zero_pad)
                    X_s.extend(x_s)
                    freq_s.extend(f_s)

                    acc_hmc_list.append(acc_hmc)
                    acc_metro_list.append(acc_metro)

            X_U, freq_U = np.array(X_U), np.array(freq_U)
            X_z, freq_z = np.array(X_z), np.array(freq_z)
            X_s, freq_s = np.array(X_s), np.array(freq_s)

            print(f"Finished {mode_name} for β={beta:.3f}, β1={beta1:.3f}, α={alpha:.3f}, α1={alpha1:.3f}")
            print(f"runs = {fit_times}\n"
                  f"U: X shape = {X_U.shape}, freq shape = {freq_U.shape}\n"
                  f"z: X shape = {X_z.shape}, freq shape = {freq_z.shape}\n"
                  f"s: X shape = {X_s.shape}, freq shape = {freq_s.shape}")

            if save_data:
                # save_npz_with_fingerprint creates the folder and stamps the
                # sampler config into the .npz in a single savez call.
                arrays = dict(X_U=X_U, freq_U=freq_U, X_z=X_z, freq_z=freq_z,
                              X_s=X_s, freq_s=freq_s)
                # HMC runs carry per-chain HMC / s-Metropolis acceptance rates;
                # heatbath runs (all-None acc) store no acc keys at all.
                if any(a is not None for a in acc_hmc_list):
                    arrays["acc_rate"] = np.array(acc_hmc_list, dtype=float)
                    print(f"HMC acc_rate: mean = {np.mean(arrays['acc_rate']):.4f}")
                if any(a is not None for a in acc_metro_list):
                    arrays["acc_rate_metro"] = np.array(acc_metro_list, dtype=float)
                    print(f"s-Metro acc_rate: mean = {np.mean(arrays['acc_rate_metro']):.4f}")
                save_npz_with_fingerprint(filepath, arrays, requested_fp)
