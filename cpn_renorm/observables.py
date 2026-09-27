from __future__ import annotations

import math
from collections.abc import Callable, Iterable

import numpy as np
import torch

from .sampler import BatchedHMCSampler
from .progress import SimulationProgress


def _z_link_phase(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Return the forward Berry-link phase ``arg(<left|right>)``."""
    return torch.angle(torch.sum(torch.conj(left) * right, dim=-1))


def projector_structure(sampler: BatchedHMCSampler) -> torch.Tensor:
    z, N = sampler.z, sampler.N
    proj = z[..., :, None] * torch.conj(z[..., None, :])
    eye = torch.eye(N, device=z.device, dtype=z.dtype)
    proj = proj - eye / N
    pk = torch.fft.fftn(proj, dim=(1, 2))
    return torch.abs(pk).square().sum((-1, -2)) / (sampler.Lx * sampler.Ly)


def connected_pp(sampler: BatchedHMCSampler) -> torch.Tensor:
    return torch.fft.ifftn(projector_structure(sampler), dim=(1, 2)).real


def second_moment_xi_from_modes(modes: np.ndarray, L: int) -> np.ndarray:
    """modes[..., :] = S(0,0), S(kmin,0), S(0,kmin)."""
    modes = np.asarray(modes, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        x = 0.5 * np.sqrt(modes[..., 0] / modes[..., 1] - 1) / np.sin(np.pi / L)
        y = 0.5 * np.sqrt(modes[..., 0] / modes[..., 2] - 1) / np.sin(np.pi / L)
    return 0.5 * (x + y)


def corr_length(corr: np.ndarray) -> float:
    spec = np.fft.fftn(corr).real
    return float(second_moment_xi_from_modes(
        np.array([spec[0, 0], spec[1, 0], spec[0, 1]]), corr.shape[0]))


def topo_charge(sampler: BatchedHMCSampler) -> dict[str, torch.Tensor]:
    da = sampler.plaquette_da()
    q_u = torch.angle(torch.exp(1j * da)).sum((1, 2)) / (2 * math.pi)
    z = sampler.z
    zx = torch.roll(z, -1, 1)
    zy = torch.roll(z, -1, 2)
    zxy = torch.roll(z, (-1, -1), (1, 2))
    vdot = lambda left, right: torch.sum(torch.conj(left) * right, dim=-1)
    tri0 = vdot(z, zx) * vdot(zx, zxy) * vdot(zxy, z)
    tri1 = vdot(z, zxy) * vdot(zxy, zy) * vdot(zy, z)
    q_z = (torch.angle(tri0) + torch.angle(tri1)).sum((1, 2)) / (2 * math.pi)
    out = {"Q_U": q_u, "Q_z": q_z}
    if sampler.kind == "full":
        out["Q_s"] = sampler.s.sum((1, 2)).to(sampler.dtype)
    return out


def _rect_phase(link: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    """Oriented rectangular loop phase for all origins on a PBC lattice."""
    phase = torch.zeros(link.shape[:-1], device=link.device, dtype=link.dtype)
    for i in range(dx):
        phase += torch.roll(link[..., 0], shifts=-i, dims=1)
    for j in range(dy):
        phase += torch.roll(link[..., 1], shifts=(-dx, -j), dims=(1, 2))
    for i in range(dx):
        phase -= torch.roll(link[..., 0], shifts=(-i, -dy), dims=(1, 2))
    for j in range(dy):
        phase -= torch.roll(link[..., 1], shifts=-j, dims=2)
    return phase


def rectangular_loops(sampler: BatchedHMCSampler,
                      sizes: Iterable[tuple[int, int]]) -> dict[str, dict[str, torch.Tensor]]:
    if not sampler.periodic:
        raise ValueError("loop observables require PBC")
    z = sampler.z
    zlink = torch.stack((_z_link_phase(z, torch.roll(z, -1, 1)),
                         _z_link_phase(z, torch.roll(z, -1, 2))), dim=-1)
    out: dict[str, dict[str, torch.Tensor]] = {}
    for dx, dy in sizes:
        if not (0 <= dx < sampler.Lx and 0 <= dy < sampler.Ly):
            raise ValueError(f"loop {dx}x{dy} exceeds lattice")
        key = f"{dx}x{dy}"
        out[key] = {
            "U": torch.cos(_rect_phase(sampler.a, dx, dy)).mean((1, 2)),
            "z": torch.cos(_rect_phase(zlink, dx, dy)).mean((1, 2)),
        }
    return out


def autocorr_time(values: np.ndarray, c: float = 5.0) -> float:
    x = np.asarray(values, dtype=float)
    if x.size < 4 or not np.isfinite(x).all() or np.var(x) == 0:
        return 0.5
    x = x - x.mean()
    n = len(x)
    f = np.fft.rfft(x, n=2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    ac /= ac[0]
    tau = 0.5
    for t in range(1, n):
        if ac[t] <= 0:
            break
        tau += ac[t]
        if t > c * tau:
            break
    return float(max(tau, 0.5))


def gamma_method(values: np.ndarray, c: float = 5.0) -> dict[str, np.ndarray | float | int]:
    """Estimate a multi-chain mean covariance with an automatic Gamma window.

    ``values`` has shape ``[sweeps, chains, observables]`` (a final singleton
    dimension is added for scalar input).  Chains are centered separately and
    are never joined across a chain boundary.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim == 2:
        x = x[..., None]
    if x.ndim != 3 or x.shape[0] < 2 or x.shape[1] < 1:
        raise ValueError("Gamma-method input must have shape [sweeps, chains, observables]")
    if not np.isfinite(x).all():
        raise ValueError("Gamma-method input contains non-finite values")
    n_time, n_chains, n_obs = x.shape
    # Use the common ensemble mean, while every lag product remains confined to
    # its own chain.  This retains legitimate between-chain fluctuation without
    # ever introducing a fictitious transition between chains.
    centered = x - x.mean(axis=(0, 1), keepdims=True)

    def covariance(lag: int) -> np.ndarray:
        left, right = centered[:n_time - lag], centered[lag:]
        return np.einsum("tci,tcj->ij", left, right) / (n_chains * (n_time - lag))

    gamma0 = covariance(0)
    spectral = gamma0.copy()
    tau = np.full(n_obs, 0.5)
    window = 0
    for lag in range(1, n_time):
        gamma_lag = covariance(lag)
        spectral += gamma_lag + gamma_lag.T
        denom = np.maximum(np.diag(gamma0), np.finfo(float).tiny)
        tau = np.maximum(0.5, np.diag(spectral) / (2 * denom))
        window = lag
        if lag >= c * float(np.max(tau)):
            break
    # A noisy tail can make a diagonal slightly negative.  Preserve the full
    # cross covariance while flooring only those invalid diagonal estimates.
    diagonal = np.maximum(np.diag(spectral), np.diag(gamma0))
    spectral[np.diag_indices(n_obs)] = diagonal
    covariance_mean = spectral / (n_time * n_chains)
    tau = np.maximum(0.5, diagonal /
                     (2 * np.maximum(np.diag(gamma0), np.finfo(float).tiny)))
    ess = n_time * n_chains / (2 * np.maximum(tau, 0.5))
    return {"mean": x.mean(axis=(0, 1)), "covariance": covariance_mean,
            "gamma0": gamma0, "tau": tau, "ess": ess, "window": window}


def second_moment_xi_jacobian(mean_modes: np.ndarray, L: int) -> np.ndarray:
    """Analytic gradient of the direction-averaged second-moment xi."""
    s0, sx, sy = np.asarray(mean_modes, dtype=float)
    scale = 0.25 / math.sin(math.pi / L)
    rx, ry = s0 / sx - 1.0, s0 / sy - 1.0
    if min(rx, ry, sx, sy) <= 0 or not np.isfinite([rx, ry, sx, sy]).all():
        return np.full(3, np.nan)
    qx, qy = math.sqrt(rx), math.sqrt(ry)
    return scale * np.array([0.5 / (sx * qx) + 0.5 / (sy * qy),
                             -0.5 * s0 / (sx * sx * qx),
                             -0.5 * s0 / (sy * sy * qy)])


def xi_from_modes_gamma(modes: np.ndarray, L: int) -> tuple[float, float, dict]:
    stats = gamma_method(modes)
    mean = np.asarray(stats["mean"])
    xi = float(second_moment_xi_from_modes(mean, L))
    jac = second_moment_xi_jacobian(mean, L)
    variance = float(jac @ np.asarray(stats["covariance"]) @ jac)
    return xi, math.sqrt(max(variance, 0.0)), stats


def measurement_stop_reason(rounds: int, chains: int, min_meas_total: int,
                            target_ess: float, max_meas_total: int,
                            ess: np.ndarray) -> str | None:
    """Return the production stop reason, respecting batched-chain rounding."""
    if rounds < math.ceil(min_meas_total / chains):
        return None
    if np.all(np.asarray(ess, dtype=float) >= target_ess):
        return "target_ess"
    if rounds >= math.ceil(max_meas_total / chains):
        return "max_meas_total"
    return None


def _mean_err(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = x.reshape((-1,) + x.shape[2:])
    return flat.mean(0), flat.std(0, ddof=1) / math.sqrt(max(len(flat), 1))


def _cheap_probe(sampler: BatchedHMCSampler, kind: str) -> np.ndarray:
    structure = projector_structure(sampler)
    modes = torch.stack((structure[:, 0, 0], structure[:, 1, 0],
                         structure[:, 0, 1]), -1)
    values = [modes[:, 0], modes[:, 1], modes[:, 2]]
    if kind == "topology":
        charge = topo_charge(sampler)
        key = "Q_s" if "Q_s" in charge else "Q_z"
        values.append(charge[key].square())
    return torch.stack(values, -1).detach().cpu().numpy()


@torch.no_grad()
def automatic_warmup(sampler: BatchedHMCSampler, *, adapt_steps: int,
                     target_accept: float, min_warmup: int, max_warmup: int,
                     tau_multiplier: float = 10.0, stability_rtol: float = 0.25,
                     probe_kind: str = "observable", s_step: float = 0.5,
                     s_updates: int = 1,
                     probe: Callable[[BatchedHMCSampler], np.ndarray] | None = None,
                     progress: SimulationProgress | None = None,
                     phase_prefix: str = "") -> dict:
    """Adapt epsilon, freeze the kernel, then establish equilibrium from probes."""
    if progress is not None:
        progress.phase(f"{phase_prefix}adapt", adapt_steps)

    def adaptation_progress(epsilon: float, accept: float) -> None:
        if progress is not None:
            progress.update(epsilon=f"{epsilon:.3g}", accept=f"{accept:.3f}")

    tuned = sampler.adapt(adapt_steps, target_accept, s_step=s_step, s_updates=s_updates,
                          progress=adaptation_progress)
    probes: list[np.ndarray] = []
    previous: np.ndarray | None = None
    stable = False
    chunk = max(16, min(128, max(1, min_warmup // 4)))
    stop_reason = "max_warmup"
    if progress is not None:
        progress.phase(f"{phase_prefix}warmup", max_warmup)
    while len(probes) < max_warmup:
        count = min(chunk, max_warmup - len(probes))
        for _ in range(count):
            sampler.sweep(s_step=s_step, s_updates=s_updates)
            probes.append((probe or (lambda item: _cheap_probe(item, probe_kind)))(sampler))
            if progress is not None:
                progress.update()
        if len(probes) < min_warmup or len(probes) < 20:
            continue
        stats = gamma_method(np.asarray(probes))
        tau = np.asarray(stats["tau"])
        enough = len(probes) >= tau_multiplier * float(np.max(tau))
        stable = (previous is not None and
                  bool(np.all(np.abs(tau - previous) /
                              np.maximum(previous, 0.5) <= stability_rtol)))
        previous = tau
        if progress is not None:
            progress.update(0, tau_max=f"{float(np.max(tau)):.2f}",
                            stable="yes" if stable else "no")
        if enough and stable:
            stop_reason = "tau_stable"
            break
    arr = np.asarray(probes)
    stats = gamma_method(arr) if len(arr) >= 2 else {"tau": np.full(arr.shape[-1], 0.5)}
    sampler.reset_diagnostics()
    return {"epsilon": tuned, "sweeps": len(probes),
            "tau": np.asarray(stats["tau"]).tolist(), "stable": stable,
            "stop_reason": stop_reason}


@torch.no_grad()
def measure_ensemble(sampler: BatchedHMCSampler, *, min_warmup: int,
                     max_warmup: int, min_meas_total: int, target_ess: float,
                     max_meas_total: int, adapt_steps: int, target_accept: float,
                     warmup_tau_multiplier: float = 10.0,
                     tau_stability_rtol: float = 0.25,
                     probe_kind: str = "observable", s_step: float = 0.5,
                     s_updates: int = 1,
                     loop_sizes: Iterable[tuple[int, int]] = (),
                     progress: SimulationProgress | None = None) -> dict:
    warmup = automatic_warmup(
        sampler, adapt_steps=adapt_steps, target_accept=target_accept,
        min_warmup=min_warmup, max_warmup=max_warmup,
        tau_multiplier=warmup_tau_multiplier, stability_rtol=tau_stability_rtol,
        probe_kind=probe_kind, s_step=s_step, s_updates=s_updates,
        progress=progress)
    modes, topologies = [], []
    corr_sum = np.zeros((sampler.chains, sampler.Lx, sampler.Ly), dtype=float)
    corr_sumsq = np.zeros_like(corr_sum)
    loops: dict[str, dict[str, list[np.ndarray]]] = {}
    minimum_rounds = int(math.ceil(min_meas_total / sampler.chains))
    maximum_rounds = int(math.ceil(max_meas_total / sampler.chains))
    block = max(16, min(128, minimum_rounds // 4 or 1))
    stop_reason = "max_meas_total"
    primary_stats: dict = {}
    rounds = 0
    if progress is not None:
        progress.phase("sampling", maximum_rounds)
    while rounds < maximum_rounds:
        count = min(block, maximum_rounds - rounds)
        if rounds < minimum_rounds:
            count = min(count, minimum_rounds - rounds)
        for _ in range(count):
            sampler.sweep(s_step=s_step, s_updates=s_updates)
            structure = projector_structure(sampler)
            corr = torch.fft.ifftn(structure, dim=(1, 2)).real.cpu().numpy()
            corr_sum += corr
            corr_sumsq += corr * corr
            modes.append(torch.stack((structure[:, 0, 0], structure[:, 1, 0],
                                      structure[:, 0, 1]), -1).cpu().numpy())
            tc = topo_charge(sampler)
            topologies.append({k: v.cpu().numpy() for k, v in tc.items()})
            for key, values in rectangular_loops(sampler, loop_sizes).items():
                entry = loops.setdefault(key, {"U": [], "z": []})
                for tag in ("U", "z"):
                    entry[tag].append(values[tag].cpu().numpy())
            if progress is not None:
                progress.update()
        rounds += count
        if rounds < minimum_rounds:
            continue
        mode_arr = np.asarray(modes)
        primary = mode_arr
        if probe_kind == "topology":
            key = "Q_s" if "Q_s" in topologies[0] else "Q_z"
            q2 = np.stack([x[key] for x in topologies]) ** 2
            q2 = q2[..., None]
            primary = np.concatenate((mode_arr, q2), axis=-1)
        primary_stats = gamma_method(primary)
        if progress is not None:
            progress.update(0, samples=rounds * sampler.chains,
                            ess=f"{float(np.min(primary_stats['ess'])):.1f}/{target_ess:g}")
        decision = measurement_stop_reason(rounds, sampler.chains, min_meas_total,
                                           target_ess, max_meas_total,
                                           np.asarray(primary_stats["ess"]))
        if decision is not None:
            stop_reason = decision
            break
    mode_arr = np.asarray(modes)
    mode_stats = gamma_method(mode_arr)
    xi, xi_err, _ = xi_from_modes_gamma(mode_arr, sampler.Lx)
    total = rounds * sampler.chains
    corr_mean = corr_sum.sum(0) / total
    raw_var = np.maximum(corr_sumsq.sum(0) / total - corr_mean ** 2, 0.0)
    corr_tau = float(np.asarray(mode_stats["tau"])[0])
    corr_err = np.sqrt(raw_var * 2 * corr_tau / total)
    topo_out = {}
    for tag in topologies[0]:
        arr = np.stack([x[tag] for x in topologies])
        flat = arr.reshape(-1)
        q2_stats = gamma_method(arr ** 2)
        tau = float(np.asarray(q2_stats["tau"])[0])
        variance = float(np.var(flat, ddof=1))
        topo_out[tag] = {"mean": float(flat.mean()), "var": variance,
                         "topo_sus": variance / (sampler.Lx * sampler.Ly),
                         "tau": tau,
                         "ess": float(np.asarray(q2_stats["ess"])[0]),
                         "err": math.sqrt(float(np.asarray(q2_stats["covariance"])[0, 0]))
                                / (sampler.Lx * sampler.Ly)}
    loop_out = {}
    for key, values in loops.items():
        loop_out[key] = {}
        for tag, series in values.items():
            arr = np.asarray(series)
            mean, err = _mean_err(arr[..., None])
            loop_out[key][tag] = {"mean": float(mean[0]), "err": float(err[0])}
    return {
        "conn_PP_corr": corr_mean, "conn_PP_corr_err": corr_err,
        "modes": mode_arr, "xi": xi, "xi_err": xi_err,
        "magsus": float(corr_mean.sum()),
        "magsus_err": float(math.sqrt(np.asarray(mode_stats["covariance"])[0, 0])),
        "topology": topo_out,
        "corr_tau": corr_tau,
        "corr_ess": float(np.asarray(mode_stats["ess"])[0]),
        "mode_tau": np.asarray(mode_stats["tau"]).tolist(),
        "mode_ess": np.asarray(mode_stats["ess"]).tolist(),
        "loops": loop_out, "epsilon": warmup["epsilon"],
        "accept_rate": sampler.accept_rate.detach().cpu().numpy(),
        "s_accept_rate": (sampler.metro_accept_rate.detach().cpu().numpy()
                          if sampler.kind == "full" else None),
        "warmup": warmup,
        "sampling": {"minimum": int(min_meas_total), "actual": total,
                     "maximum": int(max_meas_total), "per_chain": rounds,
                     "target_ess": float(target_ess),
                     "ess": np.asarray(primary_stats["ess"]).tolist(),
                     "stop_reason": stop_reason},
        "n_samples": total,
    }
