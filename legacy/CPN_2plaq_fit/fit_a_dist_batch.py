#!/usr/bin/env python3
"""
Batch Villain fitting for a-dist data.

This script mirrors the Villain fitting workflow:
1) Load train data from data/data_{boundary_BC}_pad{boundary_pad}_mod{mod}/N{N}_L{L}
2) Build X_list_train = [a_bin_center, a_boundary, phi]
3) Fit (beta_fit, alpha_fit) with Villain_fit(N).Villain_dist_norm_fit
4) Save one JSON per (N, L, beta, beta1, alpha, conn_type) to results/a_dist_fit_{boundary_BC}_pad{boundary_pad}_mod{mod}/N{n_val}_L{l_val}_{conn_type}
"""

import json
import os
import re
from datetime import datetime

import numpy as np
from scipy.optimize import curve_fit
import sys
# Make the local `func` package importable regardless of the working directory
# (so the script runs from CPN_renorm_fit/ or this subproject alike).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from func.model import Villain_fit, z1z4_extract


TRAIN_FILE_RE = re.compile(
    r"^data_beta(?P<beta>[-+0-9.eE]+)_beta1_(?P<beta1>[-+0-9.eE]+)_alpha(?P<alpha>[-+0-9.eE]+)_num(?P<fit_times>\d+)x(?P<bins>\d+)_N(?P<N>\d+)_L(?P<L>\d+)_(?P<boundary_BC>[A-Za-z0-9]+)_pad(?P<boundary_pad>\d+)_mod(?P<mod>\d+)\.npz$"
)

def list_train_candidates(data_folder, n_target, l_target, beta_target, beta1_target, alpha_target, mod_target):
    """Find all matching non-test train files for one (N, L, beta, beta1, alpha, mod)."""
    if not os.path.isdir(data_folder):
        return []

    candidates = []
    for name in os.listdir(data_folder):
        match = TRAIN_FILE_RE.match(name)
        if not match:
            continue

        n_val = int(match.group("N"))
        l_val = int(match.group("L"))
        beta_val = float(match.group("beta"))
        beta1_val = float(match.group("beta1"))
        alpha_val = float(match.group("alpha"))
        mod_val = int(match.group("mod"))
        fit_times = int(match.group("fit_times"))
        bins = int(match.group("bins"))

        if n_val != n_target or l_val != l_target:
            continue
        if mod_val != mod_target:
            continue
        atol = 1e-3
        if not np.isclose(beta_val, beta_target, atol=atol):
            continue
        if not np.isclose(beta1_val, beta1_target, atol=atol):
            continue
        if not np.isclose(alpha_val, alpha_target, atol=atol):
            continue

        candidates.append(
            {
                "file_path": os.path.join(data_folder, name),
                "file_name": name,
                "fit_times": fit_times,
                "bins": bins,
                "beta": beta_val,
                "beta1": beta1_val,
                "alpha": alpha_val,
            }
        )

    return candidates


def pick_candidate(candidates, fit_times_train=None, bins=None):
    """Pick the best candidate file based on optional constraints."""
    filtered = candidates
    if fit_times_train is not None:
        filtered = [c for c in filtered if c["fit_times"] == fit_times_train]
    if bins is not None:
        filtered = [c for c in filtered if c["bins"] == bins]

    if not filtered:
        return None

    # Prefer more statistics by default.
    filtered.sort(key=lambda c: (c["fit_times"], c["bins"]), reverse=True)
    return filtered[0]


def build_training_arrays(npz_path, conn_type, bins):
    """Build X_list_train and freq_array_train with (a, a_array, z1z4) layout."""
    key_a = f"a_list_{conn_type}"
    key_freq = f"freq_{conn_type}"

    with np.load(npz_path) as data:
        if key_a not in data:
            raise KeyError(f"Missing key '{key_a}' in {npz_path}")
        if "phi_list" not in data:
            raise KeyError(f"Missing key 'phi_list' in {npz_path}")
        if key_freq not in data:
            raise KeyError(f"Missing key '{key_freq}' in {npz_path}")

        a_array = np.asarray(data[key_a], dtype=float)
        phi = np.asarray(data["phi_list"], dtype=float)
        freq_array = np.asarray(data[key_freq], dtype=float).reshape(-1)

    a_array_rep = np.repeat(a_array, bins, axis=0)
    phi_rep = np.repeat(phi, bins, axis=0)
    z1z4_rep = z1z4_extract(phi_rep)

    temp = np.linspace(-np.pi, np.pi, bins + 1)
    a_centers = (temp[:-1] + temp[1:]) / 2
    a_train = np.tile(a_centers, len(a_array_rep) // bins).reshape(-1, 1)

    x_list_train = np.concatenate([a_train, a_array_rep, z1z4_rep], axis=1)

    if x_list_train.shape[0] != freq_array.shape[0]:
        raise ValueError(
            "Shape mismatch: "
            f"X_list_train has {x_list_train.shape[0]} rows, "
            f"freq array has {freq_array.shape[0]} rows"
        )

    return x_list_train, freq_array


def villain_mse(n_val, beta_fit, alpha_fit, x_arr, y_arr):
    pred = Villain_fit(n_val).Villain_dist_norm_fit(x_arr, beta_fit, alpha_fit)
    return float(np.linalg.norm(pred - y_arr) ** 2 / len(x_arr))


def acc_summary(npz_path):
    """HMC acceptance-rate summary from the data npz, or None.

    Returns {"mean", "std"} over chains when the file carries an `acc_rate`
    key (HMC-generated data); None for heatbath-generated or legacy files,
    which the result JSON then reports as acc_rate=null.
    """
    with np.load(npz_path) as data:
        if "acc_rate" not in data.files:
            return None
        acc = np.asarray(data["acc_rate"], dtype=float).reshape(-1)
    if acc.size == 0:
        return None
    return {
        "mean": float(np.mean(acc)),
        "std": float(np.std(acc, ddof=1)) if acc.size > 1 else 0.0,
    }


def fit_one_case(
    n_val,
    l_val,
    beta,
    beta1,
    alpha,
    conn_type,
    boundary_BC,
    boundary_pad,
    mod,
    data_folder,
    output_root,
    p0,
    fit_times_train=None,
    bins=None,
):
    candidates = list_train_candidates(data_folder, n_val, l_val, beta, beta1, alpha, mod)
    selected = pick_candidate(candidates, fit_times_train=fit_times_train, bins=bins)
    if selected is None:
        raise FileNotFoundError(
            f"No train data file found for N={n_val}, L={l_val}, beta={beta}, beta1={beta1}, alpha={alpha}, "
            f"mod={mod}, fit_times_train={fit_times_train}, bins={bins}"
        )

    x_list_train, freq_array_train = build_training_arrays(
        selected["file_path"], conn_type=conn_type, bins=selected["bins"]
    )

    villain = Villain_fit(n_val)
    popt, pcov = curve_fit(
        villain.Villain_dist_norm_fit,
        xdata=x_list_train,
        ydata=freq_array_train,
        p0=p0,
        maxfev=20000,
    )

    beta_fit, alpha_fit = popt
    beta_fit_err, alpha_fit_err = np.sqrt(np.diag(pcov))
    mse_train = villain_mse(n_val, beta_fit, alpha_fit, x_list_train, freq_array_train)

    out_dir = os.path.join(output_root, f"N{n_val}_L{l_val}_{conn_type}")
    os.makedirs(out_dir, exist_ok=True)

    out_name = (
        f"a_dist_fit_beta{beta:.3f}_beta1_{beta1:.3f}_alpha{alpha:.3f}_"
        f"N{n_val}_L{l_val}_{conn_type}_{boundary_BC}_pad{boundary_pad}_mod{mod}.json"
    )
    out_path = os.path.join(out_dir, out_name)

    result = {
        "N": int(n_val),
        "L": int(l_val),
        "boundary_BC": boundary_BC,
        "boundary_pad": int(boundary_pad),
        "mod": int(mod),
        "beta": float(beta),
        "beta1": float(beta1),
        "alpha": float(alpha),
        "conn_type": conn_type,
        "fit_inputs": {
            "data_file": selected["file_name"],
            "fit_times_train": int(selected["fit_times"]),
            "bins": int(selected["bins"]),
            "boundary_BC": boundary_BC,
            "boundary_pad": int(boundary_pad),
            "p0": [float(p0[0]), float(p0[1])],
            "x_shape": [int(x) for x in x_list_train.shape],
            "freq_shape": [int(x) for x in freq_array_train.shape],
        },
        "villain_fit": {
            "beta_fit": float(beta_fit),
            "beta_fit_err": float(beta_fit_err),
            "alpha_fit": float(alpha_fit),
            "alpha_fit_err": float(alpha_fit_err),
            "villain_mse_train": float(mse_train),
        },
        "acc_rate": acc_summary(selected["file_path"]),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return out_path, result


def main():
    # -----------------------------------------------------------------
    # User inputs: edit these variables directly before running.
    # -----------------------------------------------------------------
    n_value = 2
    l_value = 2
    boundary_BC = "OBC"
    boundary_pad = 5
    mod = 1  # halfRefVil simulation mode: 0 or 1 (beta1=0 Original entries always use mod=0)
    conn_type_list = ["U"]  # "U" or "z"
    beta_beta1_alpha_list = [(5.070, -4.564, 1.087)]

    # Optional filters for selecting train data files.
    fit_times_train = None  # e.g. 4000
    bins = None  # e.g. 60

    # Initial guess for curve_fit.
    p0 = [2, 0]

    print("=" * 108)
    print(f"beta_beta1_alpha_list={beta_beta1_alpha_list}")
    print(f"N={n_value}, L={l_value}, mod={mod}")
    for conn_type in conn_type_list:
        print("=" * 108)
        print(f"Batch a-dist Villain fit, conn_type={conn_type}")
        print("=" * 72)

        success = []
        failed = []

        for beta, beta1, alpha in beta_beta1_alpha_list:
            # Per-entry mod: Original (beta1=0) has no mod concept -> force 0
            entry_mod = 0 if abs(beta1) < 1e-10 else mod

            # Output root folder and data folder (depend on entry_mod).
            output_root = f"results/a_dist_fit_{boundary_BC}_pad{boundary_pad}_mod{entry_mod}"
            data_folder = f"data/data_{boundary_BC}_pad{boundary_pad}_mod{entry_mod}/N{n_value}_L{l_value}"
            os.makedirs(output_root, exist_ok=True)

            out_dir = os.path.join(output_root, f"N{n_value}_L{l_value}_{conn_type}")
            out_name = (
                f"a_dist_fit_beta{beta:.3f}_beta1_{beta1:.3f}_alpha{alpha:.3f}_"
                f"N{n_value}_L{l_value}_{conn_type}_{boundary_BC}_pad{boundary_pad}_mod{entry_mod}.json"
            )
            out_path = os.path.join(out_dir, out_name)
            if os.path.isfile(out_path):
                print(f"[SKIP] beta={beta}, beta1={beta1}, alpha={alpha} -> already exists: {out_path}")
                success.append((beta, beta1, alpha, out_path))
                continue
            try:
                out_path, result = fit_one_case(
                    n_val=n_value,
                    l_val=l_value,
                    beta=beta,
                    beta1=beta1,
                    alpha=alpha,
                    conn_type=conn_type,
                    boundary_BC=boundary_BC,
                    boundary_pad=boundary_pad,
                    mod=entry_mod,
                    data_folder=data_folder,
                    output_root=output_root,
                    p0=p0,
                    fit_times_train=fit_times_train,
                    bins=bins,
                )

                fit_info = result["villain_fit"]
                print(
                    f"[OK] beta={beta}, beta1={beta1}, alpha={alpha} -> "
                    f"beta_fit={fit_info['beta_fit']:.6g}, alpha_fit={fit_info['alpha_fit']:.6g}, "
                    f"mse={fit_info['villain_mse_train']:.3e}"
                )
                print(f"     saved: {out_path}")
                success.append((beta, beta1, alpha, out_path))
            except Exception as exc:
                print(f"[FAIL] beta={beta}, beta1={beta1}, alpha={alpha}: {exc}")
                failed.append((beta, beta1, alpha, str(exc)))

        print("=" * 72)
        print(f"Finished. Success: {len(success)}, Failed: {len(failed)}")
        if failed:
            print("Failed pairs:")
            for beta, beta1, alpha, reason in failed:
                print(f"  ({beta}, {beta1}, {alpha}) -> {reason}")


if __name__ == "__main__":
    main()
