"""
RG matching analysis for CP^{N-1} model.

For each case in the results folder (driven by renorm_info.json files),
loads fine-lattice and coarse-lattice Monte Carlo data, compares three
observables (PP correlation, argzz loop, Wilson loop), finds matched
beta1_c by an error-weighted smoothing-spline fit over the full coarse
scan (`match_coupling`), and produces annotated plots + summary.

Usage:
    python plot_PP_Wil_beta1.py                                 # run all cases (mod=0, halfRefVil)
    python plot_PP_Wil_beta1.py --mod 1                         # run all cases for mod=1
    python plot_PP_Wil_beta1.py --obs_fit_type RefVil           # use the full-Villain data/results trees
    python plot_PP_Wil_beta1.py --case <name>                   # run a specific case
    python plot_PP_Wil_beta1.py --mod 0 --list                  # list available cases
"""

import numpy as np
import os
import json
import glob
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import make_smoothing_spline
from scipy.optimize import brentq


# ── linear interpolation (anchor + fallback for match_coupling) ───

def acc_brief(acc):
    """Data-file acc block -> {"mean", "std"} for result files.

    Result files carry only the mean/std over chains; the per-chain record
    stays in the data JSON the block came from. None / empty -> None.
    """
    if not acc:
        return None
    return {"mean": acc.get("mean"), "std": acc.get("std")}


def linear_interpolate(x_vals, y_vals, y_target):
    """Find x where y(x) = y_target by two-point linear interpolation between
    the adjacent scan points bracketing y_target. Returns None if y_target is
    outside the range of y_vals.

    Kept as the reference location (anchor) for root selection in
    `match_coupling`, and as its graceful fallback. NB: the matched x depends
    ONLY on the two bracketing points -- it cannot use the rest of the scan to
    average out noise -- which is why `match_coupling` is preferred."""
    x_vals = np.asarray(x_vals, dtype=float)
    y_vals = np.asarray(y_vals, dtype=float)
    order = np.argsort(x_vals)
    x_vals = x_vals[order]
    y_vals = y_vals[order]

    if y_target < y_vals.min() or y_target > y_vals.max():
        return None

    for i in range(len(y_vals) - 1):
        if (y_vals[i] - y_target) * (y_vals[i + 1] - y_target) <= 0:
            t = (y_target - y_vals[i]) / (y_vals[i + 1] - y_vals[i])
            return x_vals[i] + t * (x_vals[i + 1] - x_vals[i])

    if np.isclose(y_target, y_vals[0]):
        return x_vals[0]
    if np.isclose(y_target, y_vals[-1]):
        return x_vals[-1]
    return None


# ── observable-to-coupling fit (uses ALL scan points + error bars) ─

def _select_root(roots, x_ref, x_lo, x_hi, tol=1e-7):
    """From `roots` (possibly complex), keep real ones inside [x_lo, x_hi] and
    return the one nearest x_ref (or the interval midpoint if x_ref is None).
    Returns None if none qualify."""
    cand = []
    for r in np.asarray(roots, dtype=complex).ravel():
        if abs(r.imag) > tol:
            continue
        rv = r.real
        if x_lo - 1e-9 <= rv <= x_hi + 1e-9:
            cand.append(float(rv))
    if not cand:
        return None
    ref = x_ref if x_ref is not None else 0.5 * (x_lo + x_hi)
    return min(cand, key=lambda v: abs(v - ref))


def _make_weights(y_err):
    """From an error array (already aligned to the sorted abscissae), return
    (w, sigma): w = 1/sigma^2 -- the weights `make_smoothing_spline` uses, which
    enter its objective LINEARLY -- and sigma the clamped positive per-point
    standard deviation (reused by the Monte-Carlo). (None, None) if no usable
    errors were supplied (-> unweighted fit)."""
    if y_err is None:
        return None, None
    e = np.asarray(y_err, dtype=float)
    good = np.isfinite(e) & (e > 0)
    if not np.any(good):
        return None, None
    med = float(np.median(e[good]))
    e = np.where(good, e, med)
    return 1.0 / e ** 2, e


def _fit_callable(x, y, w, force_poly=False):
    """The smooth curve y(x) that `match_coupling` fits and inverts: a GCV
    smoothing spline (n >= 5, error-weighted), otherwise -- or on request -- a
    weighted low-order polynomial (degree auto-capped so it smooths rather than
    interpolates). Returns a callable f(x) -> y, or None. Also used by
    `fit_curve`/`add_fit_curve` to draw the fit on the plots."""
    n = len(x)
    if n < 2:
        return None
    if n >= 5 and not force_poly:
        try:
            return make_smoothing_spline(x, y, w=w)
        except Exception:
            pass
    deg = min(2, n - 2)          # leave >=1 dof so the fit actually smooths
    if deg < 1:
        deg = 1                  # n == 2 -> exact line (== linear_interpolate)
    try:
        return np.poly1d(np.polyfit(x, y, deg, w=w))
    except Exception:
        return None


def _roots_in_range(f, target, x_lo, x_hi, n_grid=1001):
    """Real crossings of f(x) = target inside [x_lo, x_hi] (dense grid to locate
    sign changes, then Brent refinement). Works for any callable curve f
    (spline BSpline or np.poly1d)."""
    if f is None:
        return []
    grid = np.linspace(x_lo, x_hi, n_grid)
    try:
        g = np.asarray(f(grid), dtype=float) - target
    except Exception:
        return []
    roots = []
    for i in range(len(g) - 1):
        if g[i] == 0.0:
            roots.append(float(grid[i]))
        elif g[i] * g[i + 1] < 0:
            try:
                roots.append(float(brentq(lambda z: float(f(z) - target),
                                          grid[i], grid[i + 1])))
            except Exception:
                pass
    return roots


def match_coupling(x_vals, y_vals, y_target, y_err=None, y_target_err=None,
                   n_mc=256):
    """Match a coupling by fitting y(x) through ALL scan points and reading off
    where the fitted curve equals y_target.

    Replaces two-point `linear_interpolate`, whose answer depended only on the
    two bracketing points: here a GCV smoothing spline (error-weighted, n >= 5)
    -- or a weighted low-order polynomial for shorter scans -- uses every point
    and every error bar, so one noisy point cannot dominate. The crossing is
    anchored to the two-point linear location so a wiggly fit can't jump to a
    spurious root.

    Returns (x_match, x_err):
      - x_match: the coupling where the fitted curve hits y_target, or None.
      - x_err:   Monte-Carlo (parametric bootstrap) standard deviation of
                 x_match from the per-point and target error bars; None when no
                 errors were supplied.
    """
    x = np.asarray(x_vals, dtype=float)
    y = np.asarray(y_vals, dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    n = len(x)
    if n < 2:
        return None, None
    y_target = float(y_target)

    # Weights for make_smoothing_spline: weights enter LINEARLY
    # (sum w_i (y_i - f_i)^2), so w_i = 1/sigma_i^2 gives chi-square weighting.
    w, e_y = _make_weights(np.asarray(y_err, dtype=float)[order]
                           if y_err is not None else None)

    # Two-point location on the raw points: anchor for root selection + fallback.
    x_lin = linear_interpolate(x, y, y_target)

    def solve(yy, tt):
        """Crossing of tt for ordinate yy: fit the curve, invert, anchor to the
        two-point location. Spline first (n >= 5), weighted-polynomial fallback."""
        roots = _roots_in_range(_fit_callable(x, yy, w), tt, x[0], x[-1])
        if not roots and n >= 5:
            roots = _roots_in_range(_fit_callable(x, yy, w, force_poly=True),
                                    tt, x[0], x[-1])
        if not roots:
            return None
        return _select_root(roots, x_lin, x[0], x[-1])

    x_match = solve(y, y_target)
    if x_match is None:
        x_match = x_lin              # never worse than the two-point answer
    if x_match is None:
        return None, None

    # Monte-Carlo error: perturb each y_i and the target by their errors (weights
    # held fixed = known per-point precision), refit, reinvert, take the spread.
    x_err = None
    if e_y is not None or y_target_err is not None:
        rng = np.random.default_rng()
        ey = e_y if e_y is not None else np.zeros(n)
        et = float(y_target_err) if y_target_err is not None else 0.0
        samples = []
        for _ in range(n_mc):
            yp = y + ey * rng.standard_normal(n)
            tp = y_target + et * rng.standard_normal()
            xr = solve(yp, tp)
            if xr is not None:
                samples.append(xr)
        if len(samples) >= 2:
            x_err = float(np.std(samples, ddof=1))

    return float(x_match), x_err


def fit_curve(x_vals, y_vals, y_err=None):
    """Fit the same smooth y(x) curve `match_coupling` inverts (error-weighted
    GCV smoothing spline for n >= 5, else a weighted low-order polynomial) and
    return it as a callable f(x) for plotting, or None if it cannot be fit."""
    x = np.asarray(x_vals, dtype=float)
    y = np.asarray(y_vals, dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    if len(x) < 2:
        return None
    w, _ = _make_weights(np.asarray(y_err, dtype=float)[order]
                         if y_err is not None else None)
    return _fit_callable(x, y, w)


def add_fit_curve(x_vals, y_vals, y_err=None, label="smoothing-spline fit"):
    """Overlay the fitted curve (the one `match_coupling` uses) on the current
    axes, evaluated on a dense grid over the scan range."""
    f = fit_curve(x_vals, y_vals, y_err)
    if f is None:
        return
    xs = np.linspace(float(np.min(x_vals)), float(np.max(x_vals)), 200)
    plt.plot(xs, np.asarray(f(xs), dtype=float), color="#2ca02c",
             linewidth=2.2, label=label, zorder=2)


def add_coarse_points(x_vals, y_vals, y_err=None, label="Coarse", color=None):
    """Plot the coarse scan as markers with per-point error bars on top of the
    fitted curve (zorder 3 > the spline's 2). No connecting line. Pass
    y_err=None (e.g. corr_len, which has no per-point error) for points only."""
    plt.errorbar(np.asarray(x_vals, dtype=float), np.asarray(y_vals, dtype=float),
                 yerr=y_err, fmt="o", color=color, capsize=3,
                 label=label, zorder=3)


def _clean_err_list(vals):
    """All-or-nothing extraction of per-point error bars. Returns a list of
    floats if every entry is a finite number, else None -- so `match_coupling`
    falls back to an unweighted fit when any point is missing its error."""
    out = []
    for v in vals:
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(fv):
            return None
        out.append(fv)
    return out


def _fmt_match(val, err=None):
    """Format a matched coupling (and its MC error) for log lines."""
    if val is None:
        return "NO MATCH"
    if err is None:
        return f"{val:.4f}"
    return f"{val:.4f} +- {err:.4f}"


# ── plot helper ───────────────────────────────────────────────────

def add_beta1_annotation(beta1_val):
    """Draw a dashed red vertical line + label at the matched beta1."""
    y_min, y_max = plt.ylim()
    plt.axvline(x=beta1_val, color="red", linestyle="--", alpha=0.7)
    plt.text(
        beta1_val,
        y_min + 0.95 * (y_max - y_min),
        f"  beta1 = {beta1_val:.4f}",
        color="red",
        fontweight="bold",
        va="top",
    )


# ── correlation length from PP correlator ─────────────────────────

def corr_len(conn_PP_corr):
    """Compute correlation length via FFT of the PP correlator."""
    L = conn_PP_corr.shape[0]
    PP_corr_k = np.fft.fftn(conn_PP_corr).real
    xi_x = 0.5 * np.sqrt(PP_corr_k[0, 0] / PP_corr_k[1, 0] - 1) / np.sin(np.pi / L)
    xi_y = 0.5 * np.sqrt(PP_corr_k[0, 0] / PP_corr_k[0, 1] - 1) / np.sin(np.pi / L)
    return (xi_x + xi_y) / 2


def _coarse_title(obs_fit_type, beta_c, alpha_c, alpha1_c):
    """Coarse-coupling title suffix for plots.

    halfRefVil folds the plaquette into a single `alpha_eff`, so only alpha_eff_c
    is meaningful; RefVil keeps alpha and alpha1 separate."""
    alpha_eff_c = alpha_c + alpha1_c
    if obs_fit_type == "RefVil":
        return f"beta_c={beta_c:.3f}, alpha_c={alpha_c:.3f}, alpha1_c={alpha1_c:.3f}"
    return f"beta_c={beta_c:.3f}, alpha_eff_c={alpha_eff_c:.3f}"


# ── main processing for one case ──────────────────────────────────

def process_case(info_dict, mod=0):
    """Run matching analysis for a single renorm case."""
    # Unpack info
    N = info_dict["N"]
    L_f = info_dict["L_f"]
    L_c = info_dict["L_c"]
    beta_f = info_dict["beta_f"]
    beta1_f = info_dict["beta1_f"]
    alpha_f = info_dict["alpha_f"]
    renorm = info_dict["renorm"]
    renorm_type = info_dict["renorm_type"]
    beta_c = info_dict["beta_c"]
    alpha_c = info_dict["alpha_c"]
    alpha1_f = info_dict.get("alpha1_f", 0.0)
    alpha1_c = info_dict.get("alpha1_c", 0.0)
    obs_fit_type = info_dict.get("obs_fit_type", "halfRefVil")
    beta1_c_list = sorted(info_dict["beta1_c_list"])

    # Per-observable (x,y) separation lists (coarse). Fine separation = (x,y)*renorm.
    # Legacy fallback so old renorm_info.json (standalone batch path) still runs.
    pp_xy = info_dict.get("pp_xy_list_coarse")
    argzz_xy = info_dict.get("argzz_xy_list_coarse")
    wilson_xy = info_dict.get("wilson_xy_list_coarse")
    legacy = info_dict.get("xy_list_coarse") or info_dict.get("xy_list")
    if pp_xy is None:  # old hard-coded (0,l) heuristic
        pp_xy = [[0, l] for l in range(1, 10) if l * 10 <= L_c and l * 10 * renorm <= L_f]
    if argzz_xy is None:
        argzz_xy = legacy
    if wilson_xy is None:
        wilson_xy = legacy

    print("=" * 70)
    print(f"Processing: renorm={renorm}, type='{renorm_type}'")
    print(f"  Fine:   L={L_f}, beta={beta_f}, beta1={beta1_f}, alpha={alpha_f}")
    print(f"  Coarse: L={L_c}, beta={beta_c}, alpha={alpha_c}")
    print(f"  beta1_c scan: {len(beta1_c_list)} points")
    print("=" * 70)

    # ── load fine data ──
    # Build file pattern: handle floating-point formatting.
    # Files live in a per-(beta, alpha_eff) subfolder under the per-mod root
    # (alpha_eff = alpha + alpha1), so the obs beta1-scan and the topo alpha-scan
    # at the same alpha_eff share a subfolder:
    #   data_{obs_fit_type}/data_PP_corr_N{N}_L{L}_mod{mod}/N{N}_L{L}_beta{beta}_alpha_eff{alpha_eff}/
    alpha_eff_f = alpha_f + alpha1_f
    fine_dir = f"data_{obs_fit_type}/data_PP_corr_N{N}_L{L_f}_mod{mod}"
    fine_files = sorted(glob.glob(
        f"{fine_dir}/N{N}_L{L_f}_beta{beta_f:.3f}_alpha_eff{alpha_eff_f:.3f}/"
        f"PP_corr_N{N}_L{L_f}_beta{beta_f:.3f}_beta1_{beta1_f:.3f}"
        f"_alpha{alpha_f:.3f}_alpha1_{alpha1_f:.3f}.json"
    ))
    if not fine_files:
        # Fallback: search all per-(beta, alpha, alpha1) subfolders and match by metadata
        fine_files = sorted(glob.glob(
            f"{fine_dir}/N{N}_L{L_f}_beta*_alpha*_alpha1_*/PP_corr_*.json"
        ))
        matched = []
        for fp in fine_files:
            try:
                with open(fp) as f:
                    d = json.load(f)
                if (abs(d["beta"] - beta_f) < 0.001 and
                    abs(d.get("beta1", 0) - beta1_f) < 0.001 and
                    abs(d.get("alpha", 0) - alpha_f) < 0.001 and
                    abs(d.get("alpha1", 0) - alpha1_f) < 0.001):
                    matched.append(fp)
            except Exception:
                pass
        fine_files = matched

    if not fine_files:
        print("  ERROR: No fine data file found!")
        return None

    with open(fine_files[0]) as f:
        temp = json.load(f)
    argzz_loop_f = temp["argzz_loop"]
    wilson_loop_f = temp["wilson_loop"]
    conn_PP_corr_f = np.asarray(temp["conn_PP_corr"]["mean"])
    # per-element errors (stored on disk, used to weight the observable fits)
    conn_PP_corr_f_err = np.asarray(temp["conn_PP_corr"].get("err", []))
    topo_charge_f = temp.get("topo_charge", {})
    # HMC / s-Metropolis acceptance rates of the fine ensemble, reduced to
    # {"mean","std"} for the result file (per_chain stays in the data JSON).
    # Null for heatbath-generated or legacy data files.
    fine_acc = acc_brief(temp.get("acc_rate"))
    fine_acc_metro = acc_brief(temp.get("acc_rate_metro"))
    print(f"  Fine data loaded: {fine_files[0]}")

    # ── load coarse data ──
    alpha_eff_c = alpha_c + alpha1_c
    coarse_dir = (f"data_{obs_fit_type}/data_PP_corr_N{N}_L{L_c}_mod{mod}/"
                  f"N{N}_L{L_c}_beta{beta_c:.3f}_alpha_eff{alpha_eff_c:.3f}")
    data_beta1_c_list = []
    argzz_loop_c_list = []
    wilson_loop_c_list = []
    conn_PP_corr_c_list = []
    conn_PP_corr_c_err_list = []
    topo_charge_c_list = []
    coarse_acc = {}        # "beta1=<val>" -> acc_rate block (brief) per scan point
    coarse_acc_metro = {}  # same, for the s-Metropolis rate
    for b1c in beta1_c_list:
        cfile = (f"{coarse_dir}/PP_corr_N{N}_L{L_c}_beta{beta_c:.3f}_beta1_{b1c:.3f}"
                 f"_alpha{alpha_c:.3f}_alpha1_{alpha1_c:.3f}.json")
        if not os.path.exists(cfile):
            # try alternate formatting
            alt = sorted(glob.glob(
                f"{coarse_dir}/PP_corr_N{N}_L{L_c}_beta{beta_c:.3f}_beta1_{b1c:.2f}"
                f"_alpha{alpha_c:.3f}_alpha1_{alpha1_c:.3f}.json"
            ))
            if alt:
                cfile = alt[0]
        if os.path.exists(cfile):
            data_beta1_c_list.append(b1c)
            with open(cfile) as f:
                t = json.load(f)
            argzz_loop_c_list.append(t["argzz_loop"])
            wilson_loop_c_list.append(t["wilson_loop"])
            conn_PP_corr_c_list.append(np.asarray(t["conn_PP_corr"]["mean"]))
            conn_PP_corr_c_err_list.append(np.asarray(t["conn_PP_corr"].get("err", [])))
            topo_charge_c_list.append(t.get("topo_charge", {}))
            coarse_acc[f"beta1={b1c:.3f}"] = acc_brief(t.get("acc_rate"))
            coarse_acc_metro[f"beta1={b1c:.3f}"] = acc_brief(t.get("acc_rate_metro"))
        else:
            print(f"  File not found: beta1_c = {b1c:.3f}")

    if len(data_beta1_c_list) == 0:
        print("  ERROR: No coarse data files found!")
        return None

    print(f"  Coarse data loaded: {len(data_beta1_c_list)} files")

    # ── output folder ──
    # Classified by (N, blocking factor L=renorm, renorm_type), then by the fine
    # model couplings -- mirrors the full_renorm grouping. obs_fit_type is a suffix
    # on the mod-level folder (both families live under one top-level results/).
    folder_path = (
        f"results/PP_Wil_beta1_renorm_plot_mod{mod}_{obs_fit_type}/"
        f"N{N}_L{renorm}_{renorm_type}/"
        f"beta{beta_f:.3f}_beta1_{beta1_f:.3f}_alpha{alpha_f:.3f}_alpha1_{alpha1_f:.3f}"
    )
    os.makedirs(folder_path, exist_ok=True)

    # ── result dictionaries ──
    results_pp = {}
    results_pp_err = {}
    results_argzz = {}
    results_argzz_err = {}
    results_wilson = {}
    results_wilson_err = {}

    # ================================================================
    #  1. PP correlation
    # ================================================================
    print("\n--- PP correlation ---")
    for xy in pp_xy:
        x, y = int(xy[0]), int(xy[1])
        if not (0 <= x < L_c and 0 <= y < L_c
                and 0 <= x * renorm < L_f and 0 <= y * renorm < L_f):
            print(f"  ({x},{y}): out of correlator bounds, skip")
            continue
        ppcorr_c = np.array([cpc[x, y] for cpc in conn_PP_corr_c_list])
        fine_target = conn_PP_corr_f[x * renorm, y * renorm]
        ppcorr_f = np.ones_like(ppcorr_c) * fine_target
        ppcorr_c_err = _clean_err_list(
            [em[x, y] if em.size else None for em in conn_PP_corr_c_err_list])
        fine_target_err = (float(conn_PP_corr_f_err[x * renorm, y * renorm])
                           if conn_PP_corr_f_err.size else None)

        beta1_pp, beta1_pp_err = match_coupling(
            data_beta1_c_list, ppcorr_c, fine_target,
            y_err=ppcorr_c_err, y_target_err=fine_target_err)
        key = f"({x},{y})"
        results_pp[key] = beta1_pp
        results_pp_err[key] = beta1_pp_err

        plt.figure()
        add_coarse_points(data_beta1_c_list, ppcorr_c, ppcorr_c_err,
                          label=f"Coarse ({x},{y})")
        plt.plot(data_beta1_c_list, ppcorr_f, label=f"Fine ({x * renorm},{y * renorm})")
        add_fit_curve(data_beta1_c_list, ppcorr_c, ppcorr_c_err)
        plt.xlabel("beta1")
        plt.ylabel("PP Correlation")
        plt.legend()
        plt.title(f"PP corr  {_coarse_title(obs_fit_type, beta_c, alpha_c, alpha1_c)}")

        if beta1_pp is not None:
            add_beta1_annotation(beta1_pp)
            print(f"  ({x},{y}): fine={fine_target:.6f}  ->  "
                  f"beta1_c = {_fmt_match(beta1_pp, beta1_pp_err)}")
        else:
            print(f"  ({x},{y}): fine={fine_target:.6f}  ->  NO MATCH")

        plt.savefig(os.path.join(folder_path, f"PP_corr_({x},{y}).png"))
        plt.close()

    # ================================================================
    #  1b. PP correlation length
    # ================================================================
    print("\n--- PP correlation length ---")
    corrlen_c = np.array(
        [corr_len(cpc) for cpc in conn_PP_corr_c_list]
    ) * renorm
    fine_target = corr_len(conn_PP_corr_f)
    corrlen_f = np.ones_like(corrlen_c) * fine_target
    # corr_len is FFT-derived -> no per-point error bars; fit unweighted.
    beta1_pp, beta1_pp_err = match_coupling(data_beta1_c_list, corrlen_c, fine_target)
    results_pp["corr_len"] = beta1_pp
    results_pp_err["corr_len"] = beta1_pp_err

    plt.figure()
    add_coarse_points(data_beta1_c_list, corrlen_c, None,
                      label=f"Coarse length * {renorm}")
    plt.plot(data_beta1_c_list, corrlen_f, label="Fine length")
    add_fit_curve(data_beta1_c_list, corrlen_c)
    plt.xlabel("beta1")
    plt.ylabel("PP Correlation Length")
    plt.legend()
    plt.title(f"PP corr len  {_coarse_title(obs_fit_type, beta_c, alpha_c, alpha1_c)}")

    if beta1_pp is not None:
        add_beta1_annotation(beta1_pp)
        print(f"  corr_len: fine={fine_target:.6f}  ->  "
              f"beta1_c = {_fmt_match(beta1_pp, beta1_pp_err)}")
    else:
        print(f"  corr_len: fine={fine_target:.6f}  ->  NO MATCH")

    plt.savefig(os.path.join(folder_path, "PP_corr_len.png"))
    plt.close()

    # ================================================================
    #  1c. Magnetic susceptibility  (mag_sus = sum of conn_PP_corr).
    #      Under blocking by factor `renorm` the coarse susceptibility picks up
    #      an extra L**2 factor, so match mag_sus_fine vs mag_sus_coarse*L**2.
    #      Kept as its own output (excluded from the global avg, like corr_len).
    # ================================================================
    print("\n--- Magnetic susceptibility (sum conn_PP_corr) ---")
    magsus_c = np.array([np.sum(cpc) for cpc in conn_PP_corr_c_list]) * renorm ** 2
    fine_target = float(np.sum(conn_PP_corr_f))
    magsus_f = np.ones_like(magsus_c) * fine_target
    # error on sum(conn_PP_corr): sqrt(sum of variances) (independence approx),
    # scaled by renorm**2 just like mag_sus itself.
    magsus_c_err = _clean_err_list(
        [np.sqrt(np.sum(em ** 2)) if em.size else None
         for em in conn_PP_corr_c_err_list])
    if magsus_c_err is not None:
        magsus_c_err = [e * renorm ** 2 for e in magsus_c_err]
    fine_target_err = (float(np.sqrt(np.sum(conn_PP_corr_f_err ** 2)))
                       if conn_PP_corr_f_err.size else None)

    beta1_magsus, beta1_magsus_err = match_coupling(
        data_beta1_c_list, magsus_c, fine_target,
        y_err=magsus_c_err, y_target_err=fine_target_err)
    results_pp["magsus"] = beta1_magsus
    results_pp_err["magsus"] = beta1_magsus_err

    plt.figure()
    add_coarse_points(data_beta1_c_list, magsus_c, magsus_c_err,
                      label=f"Coarse mag_sus * {renorm}**2")
    plt.plot(data_beta1_c_list, magsus_f, label="Fine mag_sus")
    add_fit_curve(data_beta1_c_list, magsus_c, magsus_c_err)
    plt.xlabel("beta1")
    plt.ylabel("Magnetic Susceptibility")
    plt.legend()
    plt.title(f"mag_sus  {_coarse_title(obs_fit_type, beta_c, alpha_c, alpha1_c)}")

    if beta1_magsus is not None:
        add_beta1_annotation(beta1_magsus)
        print(f"  magsus: fine={fine_target:.6f}  ->  "
              f"beta1_c = {_fmt_match(beta1_magsus, beta1_magsus_err)}")
    else:
        print(f"  magsus: fine={fine_target:.6f}  ->  NO MATCH")

    plt.savefig(os.path.join(folder_path, "mag_sus.png"))
    plt.close()

    # ================================================================
    #  1d. Topological susceptibility  (RefVil only).
    #      topo_sus = (<Q^2> - <Q>^2)/V, stored per component in topo_charge.
    #      Coarse is always the integer-winding Q_s susceptibility (unscaled);
    #      the FINE value is scaled by L**2 before matching. The fine component
    #      depends on renorm_type: z -> Q_z ,  U -> Q_s.
    #      Skipped silently when Q_s is unavailable (i.e. halfRefVil data).
    #      Kept as its own output (excluded from the global avg).
    # ================================================================
    print("\n--- Topological susceptibility (topo_sus) ---")
    fine_comp = "Q_z" if renorm_type == "z" else "Q_s"
    coarse_has_s = all("Q_s" in tc for tc in topo_charge_c_list)
    if (coarse_has_s and fine_comp in topo_charge_f
            and "topo_sus" in topo_charge_f[fine_comp]):
        toposus_c = np.array([tc["Q_s"]["topo_sus"] for tc in topo_charge_c_list])
        fine_raw = float(topo_charge_f[fine_comp]["topo_sus"])
        fine_target = fine_raw * renorm ** 2          # fine scaled by L**2
        toposus_f = np.ones_like(toposus_c) * fine_target
        toposus_c_err = _clean_err_list(
            [tc["Q_s"].get("err") for tc in topo_charge_c_list])
        # the fine topo_sus is scaled by renorm**2, so its error scales too
        ferr = topo_charge_f[fine_comp].get("err")
        fine_target_err = float(ferr) * renorm ** 2 if ferr is not None else None

        beta1_toposus, beta1_toposus_err = match_coupling(
            data_beta1_c_list, toposus_c, fine_target,
            y_err=toposus_c_err, y_target_err=fine_target_err)
        results_pp["toposus"] = beta1_toposus
        results_pp_err["toposus"] = beta1_toposus_err

        plt.figure()
        add_coarse_points(data_beta1_c_list, toposus_c, toposus_c_err,
                          label=f"Coarse topo_sus(Q_s)")
        plt.plot(data_beta1_c_list, toposus_f,
                 label=f"Fine topo_sus({fine_comp}) * {renorm}**2")
        add_fit_curve(data_beta1_c_list, toposus_c, toposus_c_err)
        plt.xlabel("beta1")
        plt.ylabel("Topological Susceptibility")
        plt.legend()
        plt.title(f"topo_sus  {_coarse_title(obs_fit_type, beta_c, alpha_c, alpha1_c)}")

        if beta1_toposus is not None:
            add_beta1_annotation(beta1_toposus)
            print(f"  topo_sus: fine({fine_comp})={fine_raw:.6f}*{renorm}**2={fine_target:.6f}  "
                  f"->  beta1_c = {_fmt_match(beta1_toposus, beta1_toposus_err)}")
        else:
            print(f"  topo_sus: fine({fine_comp})={fine_raw:.6f}*{renorm}**2={fine_target:.6f}  "
                  f"->  NO MATCH")

        plt.savefig(os.path.join(folder_path, "topo_sus.png"))
        plt.close()
    else:
        print(f"  topo_sus: Q_s unavailable on coarse/fine (obs_fit_type={obs_fit_type}); skip")

    # ================================================================
    #  2. argzz loop
    # ================================================================
    print("\n--- argzz loop ---")
    for xy in argzz_xy:
        x, y = int(xy[0]), int(xy[1])
        kc, kf = f"{x}x{y}", f"{x * renorm}x{y * renorm}"
        if not all(kc in wl for wl in argzz_loop_c_list) or kf not in argzz_loop_f:
            print(f"  ({x},{y}): argzz_loop not measured for this size, skip")
            continue
        argzz_loop_c = np.array([wl[kc]["mean"] for wl in argzz_loop_c_list])
        fine_target = argzz_loop_f[kf]["mean"]
        argzz_line = np.ones_like(argzz_loop_c) * fine_target
        argzz_c_err = _clean_err_list([wl[kc].get("err") for wl in argzz_loop_c_list])
        fine_target_err = argzz_loop_f[kf].get("err")

        beta1_argzz, beta1_argzz_err = match_coupling(
            data_beta1_c_list, argzz_loop_c, fine_target,
            y_err=argzz_c_err, y_target_err=fine_target_err)
        key = f"({x},{y})"
        results_argzz[key] = beta1_argzz
        results_argzz_err[key] = beta1_argzz_err

        plt.figure()
        add_coarse_points(data_beta1_c_list, argzz_loop_c, argzz_c_err,
                          label=f"coarse ({x},{y})")
        plt.plot(data_beta1_c_list, argzz_line, label=f"fine ({x * renorm},{y * renorm})")
        add_fit_curve(data_beta1_c_list, argzz_loop_c, argzz_c_err)
        plt.xlabel("beta1")
        plt.ylabel("Arg(zz) Loop")
        plt.legend()
        plt.title(f"argzz loop  {_coarse_title(obs_fit_type, beta_c, alpha_c, alpha1_c)}")

        if beta1_argzz is not None:
            add_beta1_annotation(beta1_argzz)
            print(f"  ({x},{y}): fine={fine_target:.6f}  ->  "
                  f"beta1_c = {_fmt_match(beta1_argzz, beta1_argzz_err)}")
        else:
            print(f"  ({x},{y}): fine={fine_target:.6f}  ->  NO MATCH")

        plt.savefig(os.path.join(folder_path, f"argzz_loop_({x},{y}).png"))
        plt.close()

    # ================================================================
    #  3. Wilson loop
    # ================================================================
    print("\n--- Wilson loop ---")
    idx = 0 if renorm_type == "U" else 1
    wl_label = "Wilson(U)" if renorm_type == "U" else "Wilson(z)"

    for xy in wilson_xy:
        x, y = int(xy[0]), int(xy[1])
        kc, kf = f"{x}x{y}", f"{x * renorm}x{y * renorm}"
        if not all(kc in wl for wl in wilson_loop_c_list) or kf not in wilson_loop_f:
            print(f"  ({x},{y}): wilson_loop not measured for this size, skip")
            continue
        wilson_loop_c = np.array([wl[kc]["mean"][0] for wl in wilson_loop_c_list])
        fine_target = wilson_loop_f[kf]["mean"][idx]
        wilson_loop_f_plot = np.ones_like(wilson_loop_c) * fine_target
        wl_c_err = _clean_err_list([wl[kc]["err"][0] for wl in wilson_loop_c_list])
        fine_target_err = wilson_loop_f[kf]["err"][idx]

        beta1_wl, beta1_wl_err = match_coupling(
            data_beta1_c_list, wilson_loop_c, fine_target,
            y_err=wl_c_err, y_target_err=fine_target_err)
        key = f"({x},{y})"
        results_wilson[key] = beta1_wl
        results_wilson_err[key] = beta1_wl_err

        plt.figure()
        add_coarse_points(data_beta1_c_list, wilson_loop_c, wl_c_err,
                          label=f"coarse ({x},{y})")
        plt.plot(data_beta1_c_list, wilson_loop_f_plot, label=f"fine ({x * renorm},{y * renorm})")
        add_fit_curve(data_beta1_c_list, wilson_loop_c, wl_c_err)
        plt.xlabel("beta1")
        plt.ylabel("Wilson Loop")
        plt.legend()
        plt.title(f"{wl_label}  {_coarse_title(obs_fit_type, beta_c, alpha_c, alpha1_c)}")

        if beta1_wl is not None:
            add_beta1_annotation(beta1_wl)
            print(f"  {wl_label} ({x},{y}): fine={fine_target:.6f}  ->  "
                  f"beta1_c = {_fmt_match(beta1_wl, beta1_wl_err)}")
        else:
            print(f"  {wl_label} ({x},{y}): fine={fine_target:.6f}  ->  NO MATCH")

        plt.savefig(os.path.join(folder_path, f"wilson_loop_({x},{y}).png"))
        plt.close()

    # ================================================================
    #  4. Summary  (single global avg over all matched beta1 across
    #     pp + argzz + wilson; corr_len is excluded, kept as its own output)
    # ================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: Matched beta1_c by error-weighted smoothing spline")
    print(f"renorm = {renorm}, type = '{renorm_type}'")
    print(f"Fine:   L={L_f}, beta={beta_f}, beta1={beta1_f}, alpha={alpha_f}")
    print(f"Coarse: L={L_c}, beta={beta_c}, alpha={alpha_c}")
    print("=" * 70)

    collected = []
    for result_dict, err_dict, label in [
        (results_pp, results_pp_err, "PP corr"),
        (results_argzz, results_argzz_err, "argzz"),
        (results_wilson, results_wilson_err, "Wilson"),
    ]:
        for key, v in result_dict.items():
            if key in ("corr_len", "magsus", "toposus") or v is None:
                continue
            collected.append(v)
            print(f"  {label} {key}: beta1_c = {_fmt_match(v, err_dict.get(key))}")

    if collected:
        avg = float(np.mean(collected))
        std = float(np.std(collected)) if len(collected) > 1 else 0.0
        print(f"\n  >>> GLOBAL AVERAGE beta1_c = {avg:.4f} +/- {std:.4f}  (n={len(collected)})")
    else:
        avg, std = None, None
        print("\n  >>> NO MATCH for any observable")
    # corr_len / magsus / toposus are their own beta1_c estimates (excluded from avg).
    summary = {
        "avg": avg, "std": std, "n": len(collected),
        "corr": results_pp.get("corr_len"),
        "corr_err": results_pp_err.get("corr_len"),
        "magsus": results_pp.get("magsus"),
        "magsus_err": results_pp_err.get("magsus"),
        "toposus": results_pp.get("toposus"),
        "toposus_err": results_pp_err.get("toposus"),
    }

    # Save matching results
    results_dict = {
        "N": N,
        "L_f": L_f,
        "L_c": L_c,
        "renorm": renorm,
        "renorm_type": renorm_type,
        "obs_fit_type": obs_fit_type,
        "beta_f": beta_f,
        "beta1_f": beta1_f,
        "alpha_f": alpha_f,
        "alpha1_f": alpha1_f,
        "beta_c": beta_c,
        "alpha_c": alpha_c,
        "alpha1_c": alpha1_c,
        "beta1_c_pp": {k: v for k, v in results_pp.items()},
        "beta1_c_pp_err": {k: v for k, v in results_pp_err.items()},
        "beta1_c_argzz": {k: v for k, v in results_argzz.items()},
        "beta1_c_argzz_err": {k: v for k, v in results_argzz_err.items()},
        "beta1_c_wilson": {k: v for k, v in results_wilson.items()},
        "beta1_c_wilson_err": {k: v for k, v in results_wilson_err.items()},
        "summary": summary,
        # HMC / s-Metropolis acceptance rates of the fine ensemble and each
        # coarse scan point, as {"mean","std"} (per_chain stays in the data
        # JSONs; null components for heatbath-generated or legacy files).
        "acc_rate": {"fine": fine_acc, "coarse": coarse_acc},
        "acc_rate_metro": {"fine": fine_acc_metro, "coarse": coarse_acc_metro},
    }
    results_file = os.path.join(folder_path, "matching_results.json")
    with open(results_file, "w") as f:
        json.dump(results_dict, f, indent=4, default=str)
    print(f"\nResults saved to {results_file}")

    return results_dict


# ── entry point ───────────────────────────────────────────────────

def _case_name(info_path, mod, obs_fit_type="halfRefVil"):
    """Case label = path relative to the per-(obs_fit_type,mod) results root, e.g.
    'N2_L4_z/beta1.500_beta1_0.000_alpha0.000_alpha1_0.000'. Used for --list display
    and --case substring matching (so --case can target a specific L/renorm_type)."""
    mod_root = f"results/PP_Wil_beta1_renorm_plot_mod{mod}_{obs_fit_type}"
    return os.path.relpath(os.path.dirname(info_path), mod_root)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="RG matching analysis")
    parser.add_argument(
        "--mod",
        type=int,
        default=0,
        choices=[0, 1],
        help="Sampler mod (0 or 1); selects the mod-specific data/results folders.",
    )
    parser.add_argument(
        "--obs_fit_type",
        type=str,
        default="halfRefVil",
        choices=["halfRefVil", "RefVil"],
        help="Observable fit type; selects the data_{type}/results_{type} folders.",
    )
    parser.add_argument(
        "--case",
        type=str,
        default=None,
        help="Run a specific case by path or substring (e.g., "
        "'N2_L4_z/beta1.500_beta1_0.000_alpha0.000' or 'beta1.500')",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available cases and exit",
    )
    args = parser.parse_args()

    # Discover cases from renorm_info.json files in the per-(obs_fit_type,mod)
    # results folder. Layout is PP_Wil_...mod{mod}/N{N}_L{renorm}_{renorm_type}/
    # <leaf>/renorm_info.json, so renorm_info.json sits two levels below the root.
    info_files = sorted(
        glob.glob(f"results/PP_Wil_beta1_renorm_plot_mod{args.mod}_{args.obs_fit_type}"
                  f"/*/*/renorm_info.json")
    )

    if args.list:
        print(f"Available cases (mod={args.mod}, obs_fit_type={args.obs_fit_type}):")
        for fpath in info_files:
            with open(fpath) as f:
                d = json.load(f)
            name = _case_name(fpath, args.mod, args.obs_fit_type)
            print(
                f"  renorm={d['renorm']} {d['renorm_type']}  "
                f"bf={d['beta_f']} b1f={d['beta1_f']} af={d['alpha_f']}  ->  "
                f"bc={d['beta_c']} ac={d['alpha_c']}  "
                f"Lf={d['L_f']}->Lc={d['L_c']}  |  {name}"
            )
        return

    all_results = []

    for fpath in info_files:
        name = _case_name(fpath, args.mod, args.obs_fit_type)
        if args.case is not None and args.case not in name:
            continue

        with open(fpath) as f:
            info_dict = json.load(f)

        result = process_case(info_dict, mod=args.mod)
        if result is not None:
            all_results.append(result)

    # ── global summary ──
    if len(all_results) > 1:
        print("\n" + "=" * 70)
        print("GLOBAL SUMMARY")
        print("=" * 70)
        for r in all_results:
            s = r.get("summary", {})
            if s.get("avg") is not None:
                print(
                    f"  renorm={r['renorm']} {r['renorm_type']}  "
                    f"bf={r['beta_f']}  "
                    f"beta1_c = {s['avg']:.4f} +/- {s['std']:.4f}  (n={s['n']})"
                )
            else:
                print(
                    f"  renorm={r['renorm']} {r['renorm_type']}  "
                    f"bf={r['beta_f']}  NO MATCH"
                )


if __name__ == "__main__":
    main()
