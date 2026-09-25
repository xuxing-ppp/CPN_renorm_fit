#!/usr/bin/env python3
"""
Batch Villain fitting for vortex-dist data (single coarse plaquette, full Villain).

Targets the 1-plaquette full-Villain model (func_CPN_RefVil_1plaq_fit.py /
func/model_1plaq.py). Each data file holds THREE independent integer-vortex
observables measured on the same Monte Carlo samples:

  - U : winding reconstructed from the U(1) gauge links a   (boundary flux da_U)
  - z : winding reconstructed from the CP^{N-1} matter z    (boundary flux da_z)
  - s : the integer Villain field s itself                  (boundary flux da_U)

For every (N, L, beta, beta1, alpha, alpha1) case we run three INDEPENDENT
single-coupling fits (alpha_U, alpha_z, alpha_s) with Villain_dist_norm_fit
and save ALL THREE results in one JSON under
  results/v_dist_fit_{boundary_BC}_pad{boundary_pad}_mod{mod}/N{N}_L{L}/

If an individual fit fails, its entry is filled with nulls (success=False) and
the JSON is still written with whatever fits succeeded. A missing data file is
a hard failure (no JSON written, reported as [FAIL]).

The npz stores six flat arrays (X_U, freq_U, X_z, freq_z, X_s, freq_s); each
X_{tag} is already a (N, 2)=[vortex, da] design matrix with that vortex type's
da baked into column 1, so each fit just consumes its own (X, freq) pair.
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

from func.model_1plaq import Villain_dist_norm_fit

TRAIN_FILE_RE = re.compile(
    r"^data_beta(?P<beta>[-+0-9.eE]+)_beta1_(?P<beta1>[-+0-9.eE]+)_alpha(?P<alpha>[-+0-9.eE]+)_alpha1_(?P<alpha1>[-+0-9.eE]+)_num(?P<fit_times>\d+)_zpad(?P<zero_pad>\d+)_N(?P<N>\d+)_L(?P<L>\d+)_(?P<boundary_BC>[A-Za-z0-9]+)_pad(?P<boundary_pad>\d+)_mod(?P<mod>\d+)\.npz$"
)

# The three vortex observables fit by this script. Each tag corresponds to a
# (X_{tag}, freq_{tag}) pair stored in the data npz.
VORTEX_TAGS = ("U", "z", "s")


def list_train_candidates(data_folder, n_target, l_target, beta_target, beta1_target, alpha_target, alpha1_target, mod_target):
    """Find all matching non-test train files for one (N, L, beta, beta1, alpha, alpha1, mod)."""
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
        alpha1_val = float(match.group("alpha1"))
        mod_val = int(match.group("mod"))
        fit_times = int(match.group("fit_times"))
        zero_pad = int(match.group("zero_pad"))

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
        if not np.isclose(alpha1_val, alpha1_target, atol=atol):
            continue

        candidates.append(
            {
                "file_path": os.path.join(data_folder, name),
                "file_name": name,
                "fit_times": fit_times,
                "zero_pad": zero_pad,
                "beta": beta_val,
                "beta1": beta1_val,
                "alpha": alpha_val,
                "alpha1": alpha1_val,
            }
        )

    return candidates


def pick_candidate(candidates, fit_times_train=None, zero_pad=None):
    """Pick the best candidate file based on optional constraints."""
    filtered = candidates
    if fit_times_train is not None:
        filtered = [c for c in filtered if c["fit_times"] == fit_times_train]
    if zero_pad is not None:
        filtered = [c for c in filtered if c["zero_pad"] == zero_pad]

    if not filtered:
        return None

    # Prefer more statistics by default.
    filtered.sort(key=lambda c: c["fit_times"], reverse=True)
    return filtered[0]


def acc_summary(acc_arr):
    """Per-chain acceptance array -> {"mean", "std"} summary, or None.

    Returns None when the data npz carried no acceptance key (heatbath-generated
    or legacy data), which the result JSON then reports as acc_rate=null.
    """
    if acc_arr is None:
        return None
    acc = np.asarray(acc_arr, dtype=float).reshape(-1)
    if acc.size == 0:
        return None
    return {
        "mean": float(np.mean(acc)),
        "std": float(np.std(acc, ddof=1)) if acc.size > 1 else 0.0,
    }


def load_vortex_arrays(npz_path):
    """Load the three (X, freq) pairs for the U, z, s vortex observables.

    Each pair is a flat design matrix produced by CPN_1plaq_data_generate.py:
      X_{tag}   : (N, 2) -- column 0 = integer vortex bin centers, column 1 = da
                            (the boundary flux used for that vortex type)
      freq_{tag}: (N,)   -- histogram density over column 0

    Also loads the optional per-chain acceptance-rate arrays (`acc_rate` for the
    HMC updates, `acc_rate_metro` for the s-Metropolis); both are None when the
    file does not carry them (heatbath-generated or legacy data).

    :return dict: keys 'X_{tag}' and 'freq_{tag}' for tag in VORTEX_TAGS, plus
                  'acc_rate' and 'acc_rate_metro'.
    """
    out = {}
    with np.load(npz_path, allow_pickle=True) as data:
        for tag in VORTEX_TAGS:
            x_key = f"X_{tag}"
            f_key = f"freq_{tag}"
            if x_key not in data:
                raise KeyError(f"Missing key '{x_key}' in {npz_path}")
            if f_key not in data:
                raise KeyError(f"Missing key '{f_key}' in {npz_path}")
            x_arr = np.asarray(data[x_key], dtype=float)
            f_arr = np.asarray(data[f_key], dtype=float)

            if x_arr.ndim != 2 or x_arr.shape[1] != 2:
                raise ValueError(
                    f"{x_key} in {npz_path} has unexpected shape {x_arr.shape}; expected (N, 2)"
                )
            if x_arr.shape[0] != f_arr.shape[0]:
                raise ValueError(
                    f"Shape mismatch in {npz_path}: {x_key} has {x_arr.shape[0]} rows, "
                    f"{f_key} has {f_arr.shape[0]} rows"
                )

            out[x_key] = x_arr
            out[f_key] = f_arr

        out["acc_rate"] = (np.asarray(data["acc_rate"], dtype=float)
                           if "acc_rate" in data.files else None)
        out["acc_rate_metro"] = (np.asarray(data["acc_rate_metro"], dtype=float)
                                 if "acc_rate_metro" in data.files else None)
    return out


def villain_mse(alpha_fit, x_arr, y_arr):
    pred = Villain_dist_norm_fit(x_arr, alpha_fit)
    return float(np.linalg.norm(pred - y_arr) ** 2 / len(x_arr))


def fit_one_vortex(x_arr, freq_arr, p0):
    """Fit a single alpha to (X=[vortex, da], freq).

    Returns a result dict. On failure alpha_fit / alpha_fit_err / mse are None,
    success=False, and 'error' carries the exception message -- the caller can
    still serialize the (null) result so one failed fit does not abort the case.

    bounds=(0, inf) is required because model_1plaq.norm() uses sqrt(alpha); an
    unbounded optimizer probes negative alpha and NaNs.
    """
    try:
        popt, pcov = curve_fit(
            Villain_dist_norm_fit,
            xdata=x_arr,
            ydata=freq_arr,
            p0=p0,
            bounds=(0, np.inf),
            maxfev=20000,
        )
    except Exception as exc:
        return {
            "success": False,
            "alpha_fit": None,
            "alpha_fit_err": None,
            "villain_mse_train": None,
            "error": str(exc),
        }

    alpha_fit = float(popt[0])
    alpha_fit_err = float(np.sqrt(np.diag(pcov))[0])
    return {
        "success": True,
        "alpha_fit": alpha_fit,
        "alpha_fit_err": alpha_fit_err,
        "villain_mse_train": villain_mse(alpha_fit, x_arr, freq_arr),
        "error": None,
    }


def fit_one_case(
    n_val,
    l_val,
    beta,
    beta1,
    alpha,
    alpha1,
    boundary_BC,
    boundary_pad,
    mod,
    data_folder,
    output_root,
    p0,
    fit_times_train=None,
    zero_pad=None,
):
    candidates = list_train_candidates(data_folder, n_val, l_val, beta, beta1, alpha, alpha1, mod)
    selected = pick_candidate(candidates, fit_times_train=fit_times_train, zero_pad=zero_pad)
    if selected is None:
        raise FileNotFoundError(
            f"No train data file found for N={n_val}, L={l_val}, beta={beta}, beta1={beta1}, "
            f"alpha={alpha}, alpha1={alpha1}, mod={mod}, fit_times_train={fit_times_train}, zero_pad={zero_pad}"
        )

    arrays = load_vortex_arrays(selected["file_path"])

    # Three independent single-alpha fits (alpha_U, alpha_z, alpha_s).
    # Each may fail individually; the result still gets written with nulls.
    fits = {
        tag: fit_one_vortex(arrays[f"X_{tag}"], arrays[f"freq_{tag}"], p0)
        for tag in VORTEX_TAGS
    }

    out_dir = os.path.join(output_root, f"N{n_val}_L{l_val}")
    os.makedirs(out_dir, exist_ok=True)

    out_name = (
        f"v_dist_fit_beta{beta:.3f}_beta1_{beta1:.3f}_alpha{alpha:.3f}_alpha1_{alpha1:.3f}_"
        f"N{n_val}_L{l_val}_{boundary_BC}_pad{boundary_pad}_mod{mod}.json"
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
        "alpha1": float(alpha1),
        "fit_inputs": {
            "data_file": selected["file_name"],
            "fit_times_train": int(selected["fit_times"]),
            "zero_pad": int(selected["zero_pad"]),
            "boundary_BC": boundary_BC,
            "boundary_pad": int(boundary_pad),
            "p0": [float(p0[0])],
            "shapes": {
                tag: {
                    "X": [int(s) for s in arrays[f"X_{tag}"].shape],
                    "freq": [int(s) for s in arrays[f"freq_{tag}"].shape],
                }
                for tag in VORTEX_TAGS
            },
        },
        "villain_fit": fits,
        "acc_rate": acc_summary(arrays["acc_rate"]),
        "acc_rate_metro": acc_summary(arrays["acc_rate_metro"]),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return out_path, result


def _fmt_vortex_fit(tag, fit):
    """One-line summary of a per-vortex fit for logging."""
    if not fit["success"]:
        return f"{tag}: alpha=None ({fit['error']})"
    return f"{tag}: alpha={fit['alpha_fit']:.6g}, mse={fit['villain_mse_train']:.3e}"


def main():
    # -----------------------------------------------------------------
    # User inputs: edit these variables directly before running.
    # -----------------------------------------------------------------
    n_value = 2
    l_value = 2
    boundary_BC = "OBC"
    boundary_pad = 5
    mod = 1  # RefVil simulation mode: 0 or 1 (beta1=0 Original entries always use mod=0)
    beta_beta1_alpha_alpha1_list = [(5.07, -4.26, 0.321, 0.766)]

    # Optional filters for selecting train data files.
    fit_times_train = None  # e.g. 4000
    zero_pad = None  # e.g. 2

    # Initial guess for curve_fit (single parameter: alpha), shared by U/z/s.
    p0 = [1.0]

    print("=" * 108)
    print(f"beta_beta1_alpha_alpha1_list={beta_beta1_alpha_alpha1_list}")
    print(f"N={n_value}, L={l_value}, mod={mod}")
    print("=" * 108)
    print("Batch vortex-dist Villain fit (U, z, s) -- single coarse plaquette")
    print("=" * 72)

    success = []
    failed = []

    for beta, beta1, alpha, alpha1 in beta_beta1_alpha_alpha1_list:
        # Per-entry mod: Original (beta1=0) has no mod concept -> force 0
        entry_mod = 0 if abs(beta1) < 1e-10 else mod

        # Output root folder and data folder (depend on entry_mod).
        output_root = f"results/v_dist_fit_{boundary_BC}_pad{boundary_pad}_mod{entry_mod}"
        data_folder = f"data/data_{boundary_BC}_pad{boundary_pad}_mod{entry_mod}/N{n_value}_L{l_value}"
        os.makedirs(output_root, exist_ok=True)

        out_dir = os.path.join(output_root, f"N{n_value}_L{l_value}")
        out_name = (
            f"v_dist_fit_beta{beta:.3f}_beta1_{beta1:.3f}_alpha{alpha:.3f}_alpha1_{alpha1:.3f}_"
            f"N{n_value}_L{l_value}_{boundary_BC}_pad{boundary_pad}_mod{entry_mod}.json"
        )
        out_path = os.path.join(out_dir, out_name)
        if os.path.isfile(out_path):
            print(f"[SKIP] beta={beta}, beta1={beta1}, alpha={alpha}, alpha1={alpha1} -> already exists: {out_path}")
            success.append((beta, beta1, alpha, alpha1, out_path))
            continue
        try:
            out_path, result = fit_one_case(
                n_val=n_value,
                l_val=l_value,
                beta=beta,
                beta1=beta1,
                alpha=alpha,
                alpha1=alpha1,
                boundary_BC=boundary_BC,
                boundary_pad=boundary_pad,
                mod=entry_mod,
                data_folder=data_folder,
                output_root=output_root,
                p0=p0,
                fit_times_train=fit_times_train,
                zero_pad=zero_pad,
            )

            vf = result["villain_fit"]
            print(
                f"[OK] beta={beta}, beta1={beta1}, alpha={alpha}, alpha1={alpha1} -> "
                + " | ".join(_fmt_vortex_fit(tag, vf[tag]) for tag in VORTEX_TAGS)
            )
            print(f"     saved: {out_path}")
            success.append((beta, beta1, alpha, alpha1, out_path))
        except Exception as exc:
            print(f"[FAIL] beta={beta}, beta1={beta1}, alpha={alpha}, alpha1={alpha1}: {exc}")
            failed.append((beta, beta1, alpha, alpha1, str(exc)))

    print("=" * 72)
    print(f"Finished. Success: {len(success)}, Failed: {len(failed)}")
    if failed:
        print("Failed quads:")
        for beta, beta1, alpha, alpha1, reason in failed:
            print(f"  ({beta}, {beta1}, {alpha}, {alpha1}) -> {reason}")


if __name__ == "__main__":
    main()
