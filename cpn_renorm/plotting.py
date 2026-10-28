"""Headless fit diagnostics, loaded only inside the plotting worker."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .exact_fitting import Conditional
from .fitting import evaluate_match_curve, one_plaq_probability, two_plaq_probability


def one_plaq_curve(f: np.ndarray, alpha: float) -> np.ndarray:
    winding = np.rint(f)
    return one_plaq_probability(
        np.column_stack((winding, 2 * np.pi * (f - winding))), alpha)[0]


def select_boundaries(X: np.ndarray, bins: int, seed: int) -> np.ndarray:
    """Select histogram blocks, retaining distinct samples with identical boundaries."""
    if bins < 1 or len(X) % bins:
        raise ValueError("2plaq rows must contain complete boundary histogram blocks")
    count = len(X) // bins
    return np.random.default_rng(seed).choice(count, min(9, count), replace=False)


def two_plaq_curves(N: int, X: np.ndarray, fit: dict) -> tuple[np.ndarray, np.ndarray]:
    full, folded = fit["new"], fit["old"]
    model = Conditional(N, X, nodes=int(full.get("normalization_nodes", 256)))
    return (model.predict([full["beta"], full["alpha"], full["alpha1"]]),
            two_plaq_probability(N, X, folded["beta"], folded["alpha_eff"]))


def _save(fig, destination: str) -> dict:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.tight_layout()
        fig.savefig(path, dpi=150)
    finally:
        plt.close(fig)
    return {"path": str(path)}


def plot_one(request: dict, arrays: dict) -> dict:
    tags = request["tags"]
    fig, axes = plt.subplots(1, len(tags), figsize=(6 * len(tags), 4.8), squeeze=False)
    for ax, tag in zip(axes.flat, tags):
        X, freq = arrays[f"X_{tag}"], arrays[f"freq_{tag}"]
        f = X[:, 0] + X[:, 1] / (2 * np.pi)
        ax.scatter(f, freq, s=14, alpha=.4, label="data (all boundaries)")
        fit = request["fits"][tag]
        alpha = fit.get("alpha")
        if fit.get("success") and alpha is not None and alpha > 0:
            grid = np.linspace(float(f.min()) - .5, float(f.max()) + .5, 2000)
            ax.plot(grid, one_plaq_curve(grid, alpha), color="C3", lw=2,
                    label=rf"fit ($\alpha={alpha:.5g}$)")
            ax.set_title(f"{tag}: 1plaq")
        else:
            ax.set_title(f"{tag}: data only ({fit.get('message', 'fit unavailable')})")
        ax.set(xlabel=r"$f = \mathrm{vortex} + da/(2\pi)$", ylabel="Probability")
        ax.legend()
    return _save(fig, request["destination"])


def plot_two(request: dict, arrays: dict) -> dict:
    X, freq, bins = arrays["X"], arrays["freq"], int(request["bins"])
    selected = select_boundaries(X, bins, int(request["seed"]))
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), squeeze=False)
    fit = request["fit"]
    full, folded = fit["new"], fit["old"]
    fig.suptitle(
        f"{request['tag']}: full (new): beta={full['beta']:.5g}, "
        f"alpha={full['alpha']:.5g}, alpha1={full['alpha1']:.5g}\n"
        f"folded (old): beta={folded['beta']:.5g}, alpha_eff={folded['alpha_eff']:.5g}")
    for ax, index in zip(axes.flat, selected):
        block = slice(int(index) * bins, (int(index) + 1) * bins)
        data = X[block]
        if not np.all(data[:, 1:] == data[0, 1:]):
            raise ValueError("2plaq histogram block mixes different boundaries")
        grid = np.linspace(-np.pi, np.pi, 600)
        features = np.column_stack((grid, np.repeat(data[:1, 1:], len(grid), axis=0)))
        new_y, old_y = two_plaq_curves(int(request["N"]), features, fit)
        ax.plot(data[:, 0], freq[block], ".", color="C0", label="data")
        ax.plot(grid, new_y, color="C3", label="full (new)")
        ax.plot(grid, old_y, "--", color="C2", label="folded (old)")
        ax.set(title=f"Boundary #{index}", xlabel=r"$a$", ylabel="Density")
        ax.legend()
    for ax in list(axes.flat)[len(selected):]:
        ax.set_visible(False)
    result = _save(fig, request["destination"])
    return {**result, "boundary_indices": selected.tolist()}


def plot_scan(request: dict, arrays: dict) -> dict:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = arrays["x"]
    zoom_x, zoom_y = [], []
    for key in request["series"]:
        color = request.get("colors", {}).get(key, "C0")
        error_key = request.get("errors", {}).get(key)
        y = arrays[key]
        err = arrays[error_key] if error_key else np.zeros_like(y)
        match = request.get("matches", {}).get(key, {})
        points = np.asarray(match.get("fit_points", []), float)
        fit_mask = np.zeros(len(x), bool)
        for point in points:
            fit_mask |= np.isclose(x, point, rtol=0, atol=1e-12)
        valid = np.isfinite(x) & np.isfinite(y)
        for ax in axes:
            for mask, marker, alpha, label in (
                    (valid & ~fit_mask, "x", .45, "exploration"),
                    (valid & fit_mask, "o", 1., "fit points")):
                if mask.any():
                    ax.errorbar(x[mask], y[mask], yerr=err[mask], fmt=marker,
                                color=color, alpha=alpha, capsize=3, label=f"{key} {label}")
        local = valid & fit_mask
        zoom_x.extend(x[local].tolist())
        zoom_y.extend((y[local] - np.nan_to_num(err[local])).tolist())
        zoom_y.extend((y[local] + np.nan_to_num(err[local])).tolist())
        if match.get("bracketed") and match.get("fit_coefficients") and len(points) >= 2:
            grid = np.linspace(points.min(), points.max(), 400)
            curve = evaluate_match_curve(match, grid)
            zoom_y.extend(curve.tolist())
            value, error = match.get("value"), match.get("error")
            for ax in axes:
                ax.plot(grid, curve, color=color, lw=2, label=f"{key} local fit")
                if value is not None:
                    crossing = float(evaluate_match_curve(match, np.array([value]))[0])
                    ax.axvline(value, color=color, linestyle=":", alpha=.8)
                    ax.errorbar([value], [crossing], xerr=error, fmt="*", ms=12,
                                color="C3", capsize=4, zorder=5)
            if value is not None:
                label = f"{request['xlabel']} = {value:.6g}"
                if error is not None:
                    label += f" ± {error:.3g}"
                axes[1].annotate(label, (value, crossing), xytext=(8, 12),
                                 textcoords="offset points", fontsize=9)
        else:
            axes[1].text(.03, .95, match.get("reason", "Final fit unavailable"),
                         transform=axes[1].transAxes, va="top", fontsize=9, wrap=True)
    for label, target in request.get("targets", {}).items():
        error = request.get("target_errors", {}).get(label)
        zoom_y.append(float(target))
        if error is not None:
            zoom_y.extend([target - error, target + error])
        for ax in axes:
            color = request.get("colors", {}).get(label, "C1")
            ax.axhline(target, color=color, linestyle="--", label=label)
            if error is not None:
                ax.axhspan(target - error, target + error, color=color, alpha=.12)
    axes[0].set_title("All scan points")
    axes[1].set_title("Final fit region")
    if zoom_x:
        lo, hi = min(zoom_x), max(zoom_x)
        margin = max((hi - lo) * .15, abs(lo) * 1e-6, 1e-9)
        axes[1].set_xlim(lo - margin, hi + margin)
        finite_y = np.asarray(zoom_y)[np.isfinite(zoom_y)]
        if len(finite_y):
            lo, hi = float(finite_y.min()), float(finite_y.max())
            margin = max((hi - lo) * .15, abs(lo) * 1e-6, 1e-9)
            axes[1].set_ylim(lo - margin, hi + margin)
        axes[0].axvspan(min(zoom_x), max(zoom_x), color="grey", alpha=.1)
    for ax in axes:
        ax.set(xlabel=request["xlabel"], ylabel=request["ylabel"])
        ax.legend(fontsize=8)
    return _save(fig, request["destination"])
