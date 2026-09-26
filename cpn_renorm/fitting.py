from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from scipy.interpolate import make_smoothing_spline
from scipy.optimize import brentq, curve_fit
from scipy.special import i0e


def _villain_norm(da: np.ndarray, alpha: float) -> np.ndarray:
    n_terms = max(int(np.ceil(10 / np.sqrt(max(alpha, 1e-12)))), 10)
    n = np.arange(-n_terms, n_terms + 1)[:, None]
    return np.exp(-0.5 * alpha * (2 * np.pi * n + da[None]) ** 2).sum(0)


def fit_one_plaq(X: np.ndarray, freq: np.ndarray, p0: float = 0.5) -> dict:
    X, freq = np.asarray(X, float), np.asarray(freq, float)

    def model(x: np.ndarray, alpha: float) -> np.ndarray:
        s, da = x[:, 0], x[:, 1]
        return np.exp(-0.5 * alpha * (da + 2 * np.pi * s) ** 2) / _villain_norm(da, alpha)

    popt, pcov = curve_fit(model, X, freq, p0=[p0], bounds=(1e-12, np.inf), maxfev=20000)
    pred = model(X, *popt)
    return {"alpha": float(popt[0]), "alpha_err": float(np.sqrt(pcov[0, 0])),
            "mse": float(np.mean((pred - freq) ** 2))}


def fit_two_plaq(N: int, X: np.ndarray, freq: np.ndarray,
                  p0: tuple[float, float] = (0.5, 0.5)) -> dict:
    X, freq = np.asarray(X, float), np.asarray(freq, float)

    def model(x: np.ndarray, beta: float, alpha: float) -> np.ndarray:
        U, boundary = np.exp(1j * x[:, 0]), np.exp(1j * x[:, 1:7])
        overlap = x[:, 7] * np.exp(-1j * x[:, 8])
        field = (2 * N * beta * overlap + alpha * boundary[:, 0] * boundary[:, 4]
                 * boundary[:, 5] + alpha * np.conj(boundary[:, 1]
                 * boundary[:, 2] * boundary[:, 3]))
        # exp(real(UF))/I0(|F|), evaluated without overflow.
        return np.exp(np.real(U * field) - np.abs(field)) / i0e(np.abs(field)) / (2 * np.pi)

    popt, pcov = curve_fit(model, X, freq, p0=p0, bounds=(0, np.inf), maxfev=50000)
    pred = model(X, *popt)
    return {"beta": float(popt[0]), "beta_err": float(np.sqrt(pcov[0, 0])),
            "alpha_eff": float(popt[1]), "alpha_eff_err": float(np.sqrt(pcov[1, 1])),
            "mse": float(np.mean((pred - freq) ** 2))}


def match_coupling(x: np.ndarray, y: np.ndarray, target: float,
                   y_err: np.ndarray | None = None,
                   target_err: float | None = None) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    valid = np.isfinite(x) & np.isfinite(y)
    if y_err is not None:
        raw_err = np.asarray(y_err, float)
        valid &= np.isfinite(raw_err)
        y_err = raw_err[valid]
    x, y = x[valid], y[valid]
    if len(x) < 2:
        return {"value": None, "bracketed": False,
                "reason": "fewer than two finite scan points"}
    order = np.argsort(x)
    x, y = x[order], y[order]
    if len(np.unique(x)) < 2:
        return {"value": None, "bracketed": False, "reason": "fewer than two scan points"}
    err = None if y_err is None else np.asarray(y_err, float)[order]
    raw_crossings = np.flatnonzero((y[:-1] - target) * (y[1:] - target) <= 0)
    if not len(raw_crossings):
        direction = "low" if abs(y[0] - target) < abs(y[-1] - target) else "high"
        return {"value": None, "bracketed": False, "direction": direction,
                "range": [float(np.min(y)), float(np.max(y))],
                "fit_type": None, "raw_bracket": None}
    anchors = []
    for idx in raw_crossings:
        dy = y[idx + 1] - y[idx]
        root = (0.5 * (x[idx] + x[idx + 1]) if abs(dy) < 1e-15 else
                x[idx] + (target - y[idx]) * (x[idx + 1] - x[idx]) / dy)
        anchors.append((float(root), int(idx)))
    anchor, anchor_idx = min(anchors, key=lambda item: abs(item[0] - x[np.argmin(abs(y-target))]))
    raw_bracket = [float(x[anchor_idx]), float(x[anchor_idx + 1])]
    weights = None
    if err is not None and np.all(np.isfinite(err)):
        weights = 1 / np.maximum(err, np.finfo(float).eps)
    try:
        if len(x) < 5:
            degree = min(2, len(x) - 1)
            coeff = np.polyfit(x, y, degree, w=weights)
            fn: Callable[[np.ndarray], np.ndarray] = lambda q: np.polyval(coeff, q)
            fit_type = f"weighted_poly_{degree}"
        else:
            fn = make_smoothing_spline(x, y, w=weights)
            fit_type = "gcv_smoothing_spline"
    except Exception:
        fn = lambda q: np.interp(q, x, y)
        fit_type = "linear_fallback"
    grid = np.linspace(x[0], x[-1], 2001)
    delta = np.asarray(fn(grid)) - target
    crossings = np.flatnonzero(delta[:-1] * delta[1:] <= 0)
    if not len(crossings):
        candidates = [anchor]
    else:
        candidates = []
        for idx in crossings:
            try:
                candidates.append(brentq(lambda q: float(fn(q) - target),
                                          grid[idx], grid[idx + 1]))
            except ValueError:
                pass
    if not candidates:
        return {"value": None, "bracketed": False, "reason": "root solve failed"}
    value = min(candidates, key=lambda q: abs(q - anchor))
    slope_h = max((x[-1] - x[0]) * 1e-4, 1e-8)
    slope = (float(fn(value + slope_h)) - float(fn(value - slope_h))) / (2 * slope_h)
    local_err = (float(np.interp(value, x, err)) if err is not None else 0.0)
    combined = math.hypot(local_err, float(target_err or 0.0))
    xerr = None if combined == 0 or abs(slope) < 1e-14 else combined / abs(slope)
    return {"value": float(value), "error": xerr, "bracketed": True,
            "anchor": anchor, "raw_bracket": raw_bracket, "fit_type": fit_type}


def extend_scan(points: list[float], result: dict, limits: tuple[float, float],
                growth: float) -> list[float]:
    points = sorted(set(float(x) for x in points))
    lo, hi = limits
    if len(points) < 2:
        raise ValueError("scan requires at least two initial points")
    span = points[-1] - points[-2] if result.get("direction") == "high" else points[1] - points[0]
    if result.get("direction") == "high":
        candidate = min(hi, points[-1] + growth * span)
    else:
        candidate = max(lo, points[0] - growth * span)
    if any(math.isclose(candidate, x, abs_tol=1e-12) for x in points):
        return points
    return sorted(points + [candidate])
