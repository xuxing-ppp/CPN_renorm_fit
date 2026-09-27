from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq, curve_fit, minimize
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


def _crossing_brackets(x: np.ndarray, y: np.ndarray, target: float) -> list[int]:
    return np.flatnonzero((y[:-1] - target) * (y[1:] - target) <= 0).tolist()


def _linear_anchor(x: np.ndarray, y: np.ndarray, target: float,
                   indices: list[int]) -> tuple[float, int]:
    anchors = []
    nearest = float(x[np.argmin(abs(y - target))])
    for idx in indices:
        dy = y[idx + 1] - y[idx]
        root = (0.5 * (x[idx] + x[idx + 1]) if abs(dy) < 1e-15 else
                x[idx] + (target - y[idx]) * (x[idx + 1] - x[idx]) / dy)
        anchors.append((float(root), int(idx)))
    return min(anchors, key=lambda item: abs(item[0] - nearest))


def _local_indices(x: np.ndarray, bracket_idx: int, anchor: float,
                   max_points: int) -> np.ndarray:
    chosen = {bracket_idx, bracket_idx + 1}
    candidates = sorted((abs(float(value) - anchor), idx)
                         for idx, value in enumerate(x) if idx not in chosen)
    for _, idx in candidates:
        if len(chosen) >= max(2, max_points):
            break
        chosen.add(idx)
    return np.array(sorted(chosen), dtype=int)


def _monotonic_polynomial(x: np.ndarray, y: np.ndarray, err: np.ndarray | None,
                          increasing: bool, degree: int) -> tuple[np.ndarray, float, float] | None:
    """Weighted polynomial whose derivative keeps one sign on the fit domain."""
    center = 0.5 * float(x[0] + x[-1])
    scale = max(0.5 * float(x[-1] - x[0]), np.finfo(float).eps)
    t = (x - center) / scale
    sigma = np.ones_like(y) if err is None else np.maximum(err, np.finfo(float).eps)
    weights = 1.0 / sigma ** 2
    design = np.column_stack([t ** power for power in range(degree + 1)])
    try:
        initial = np.linalg.lstsq(design * np.sqrt(weights[:, None]),
                                  y * np.sqrt(weights), rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    sign = 1.0 if increasing else -1.0

    def objective(coeff: np.ndarray) -> float:
        residual = design @ coeff - y
        return float(np.sum(weights * residual ** 2))

    constraints = []
    if degree == 1:
        constraints.append({"type": "ineq", "fun": lambda coeff: sign * coeff[1]})
    else:
        constraints.extend([
            {"type": "ineq", "fun": lambda coeff: sign * (coeff[1] - 2 * coeff[2])},
            {"type": "ineq", "fun": lambda coeff: sign * (coeff[1] + 2 * coeff[2])},
        ])
    fitted = minimize(objective, initial, method="SLSQP", constraints=constraints,
                      options={"ftol": 1e-12, "maxiter": 500})
    if not fitted.success or not np.isfinite(fitted.x).all():
        return None
    return np.asarray(fitted.x, float), center, scale


def _polynomial_root(coeff: np.ndarray, center: float, scale: float,
                     target: float, bounds: tuple[float, float],
                     anchor: float) -> float | None:
    lo, hi = bounds

    def fn(value: float) -> float:
        return float(np.polynomial.polynomial.polyval((value - center) / scale, coeff)
                     - target)

    grid = np.linspace(lo, hi, 513)
    delta = np.array([fn(value) for value in grid])
    candidates: list[float] = []
    for idx in np.flatnonzero(delta[:-1] * delta[1:] <= 0):
        try:
            candidates.append(float(brentq(fn, grid[idx], grid[idx + 1])))
        except ValueError:
            pass
    return min(candidates, key=lambda value: abs(value - anchor)) if candidates else None


def evaluate_match_curve(match: dict, x: np.ndarray) -> np.ndarray:
    """Evaluate the serialized local curve returned by :func:`match_coupling`."""
    coeff = np.asarray(match["fit_coefficients"], dtype=float)
    center, scale = float(match["fit_center"]), float(match["fit_scale"])
    return np.polynomial.polynomial.polyval((np.asarray(x) - center) / scale, coeff)


def match_coupling(x: np.ndarray, y: np.ndarray, target: float,
                   y_err: np.ndarray | None = None,
                   target_err: float | None = None, *, increasing: bool = True,
                   max_points: int = 5, bootstrap_seed: int = 1729,
                   bootstrap_samples: int = 256) -> dict:
    """Invert a local error-weighted fit constrained to the expected monotonicity."""
    raw_x, raw_y = np.asarray(x, float), np.asarray(y, float)
    valid = np.isfinite(raw_x) & np.isfinite(raw_y)
    raw_err = None if y_err is None else np.asarray(y_err, float)
    if raw_err is not None:
        valid &= np.isfinite(raw_err) & (raw_err >= 0)
    x, y = raw_x[valid], raw_y[valid]
    err = None if raw_err is None else raw_err[valid]
    if len(x) < 2 or not np.isfinite(target):
        return {"value": None, "bracketed": False,
                "reason": "fewer than two finite scan points"}
    order = np.argsort(x)
    x, y = x[order], y[order]
    err = None if err is None else err[order]
    if len(np.unique(x)) < 2:
        return {"value": None, "bracketed": False,
                "reason": "fewer than two distinct scan points"}
    crossings = _crossing_brackets(x, y, target)
    if not crossings:
        direction = "low" if abs(y[0] - target) < abs(y[-1] - target) else "high"
        return {"value": None, "bracketed": False, "direction": direction,
                "range": [float(np.min(y)), float(np.max(y))],
                "fit_type": None, "raw_bracket": None, "fit_points": []}
    anchor, bracket_idx = _linear_anchor(x, y, target, crossings)
    raw_bracket = [float(x[bracket_idx]), float(x[bracket_idx + 1])]
    local_idx = _local_indices(x, bracket_idx, anchor, max_points)
    local_x, local_y = x[local_idx], y[local_idx]
    local_err = None if err is None else err[local_idx]
    degree = 2 if len(local_x) >= 4 else 1
    fit = _monotonic_polynomial(local_x, local_y, local_err, increasing, degree)
    fit_type = f"monotonic_weighted_poly_{degree}"
    if fit is None and degree == 2:
        degree = 1
        fit = _monotonic_polynomial(local_x, local_y, local_err, increasing, degree)
        fit_type = "monotonic_weighted_linear_fallback"
    if fit is None:
        return {"value": None, "bracketed": False,
                "reason": "monotonic local fit failed", "raw_bracket": raw_bracket,
                "fit_points": local_x.tolist()}
    coeff, center, scale = fit
    value = _polynomial_root(coeff, center, scale, target,
                             (raw_bracket[0], raw_bracket[1]), anchor)
    if value is None:
        # A smoothed curve can miss the raw crossing. Retain a monotonic,
        # error-weighted linear fallback before giving up.
        linear = _monotonic_polynomial(local_x, local_y, local_err, increasing, 1)
        if linear is not None:
            coeff, center, scale = linear
            fit_type = "monotonic_weighted_linear_fallback"
            value = _polynomial_root(coeff, center, scale, target,
                                     (raw_bracket[0], raw_bracket[1]), anchor)
    if value is None:
        value = anchor
        fit_type = "raw_linear_fallback"
        raw_slope = ((y[bracket_idx + 1] - y[bracket_idx]) /
                     (x[bracket_idx + 1] - x[bracket_idx]))
        coeff = np.array([target, raw_slope])
        center, scale = anchor, 1.0

    xerr = None
    successes = 0
    if bootstrap_samples > 1 and (local_err is not None or target_err is not None):
        rng = np.random.default_rng(bootstrap_seed)
        sigma_y = np.zeros_like(local_y) if local_err is None else local_err
        sigma_target = max(float(target_err or 0.0), 0.0)
        samples: list[float] = []
        for _ in range(bootstrap_samples):
            perturbed_y = local_y + sigma_y * rng.standard_normal(len(local_y))
            perturbed_target = target + sigma_target * rng.standard_normal()
            boot = _monotonic_polynomial(local_x, perturbed_y, local_err,
                                         increasing, degree)
            if boot is None:
                continue
            root = _polynomial_root(*boot, perturbed_target,
                                    (float(local_x[0]), float(local_x[-1])), value)
            if root is not None:
                samples.append(root)
        successes = len(samples)
        if successes >= max(32, bootstrap_samples // 4):
            xerr = float(np.std(samples, ddof=1))
    return {"value": float(value), "error": xerr, "bracketed": True,
            "anchor": anchor, "raw_bracket": raw_bracket, "fit_type": fit_type,
            "fit_points": local_x.tolist(), "fit_values": local_y.tolist(),
            "fit_errors": None if local_err is None else local_err.tolist(),
            "fit_coefficients": coeff.tolist(), "fit_center": float(center),
            "fit_scale": float(scale), "monotonic": "increasing" if increasing else "decreasing",
            "bootstrap_successes": successes}


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
