from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .config import fingerprint, section
from .sampler import BatchedHMCSampler, Couplings
from .pilot import resolve_device, torch_dtype
from .observables import autocorr_time, automatic_warmup
from .progress import SimulationProgress
from .storage import atomic_npz


PATCH_CACHE_SCHEMA = 2


@dataclass(frozen=True)
class RectanglePatch:
    """A cropped equilibrium patch and its oriented boundary loop."""

    z: torch.Tensor
    a: torch.Tensor
    s: torch.Tensor
    zloop: torch.Tensor
    aloop: torch.Tensor

    def __iter__(self):
        """Keep the former ``zloop, aloop = _extract_rectangle(...)`` API."""
        yield self.zloop
        yield self.aloop


def _wrap(x: torch.Tensor) -> torch.Tensor:
    return torch.remainder(x + math.pi, 2 * math.pi) - math.pi


def _make_sampler(cfg: dict, settings: dict, *, chains: int, Lx: int, Ly: int,
                  couplings: Couplings, kind: str, periodic: bool,
                  seed_offset: int) -> BatchedHMCSampler:
    runtime = cfg["runtime"]
    return BatchedHMCSampler(
        chains=chains, Lx=Lx, Ly=Ly, N=int(cfg["model"]["N"]),
        couplings=couplings, kind=kind, periodic=periodic,
        device=resolve_device(runtime["device"]), dtype=torch_dtype(runtime["dtype"]),
        seed=int(runtime["seed"]) + seed_offset, epsilon=float(settings["epsilon"]),
        trajectory_length=float(settings["trajectory_length"]), mass_a=float(settings["mass_a"]),
        mass_z=float(settings["mass_z"]))


def _warmup(sampler: BatchedHMCSampler, settings: dict, probe=None, *,
            progress: SimulationProgress | None = None,
            phase_prefix: str = "") -> dict:
    return automatic_warmup(
        sampler, adapt_steps=int(settings["adapt_steps"]),
        target_accept=float(settings["target_accept"]),
        min_warmup=int(settings["min_warmup"]),
        max_warmup=int(settings["max_warmup"]),
        tau_multiplier=float(settings["warmup_tau_multiplier"]),
        stability_rtol=float(settings["tau_stability_rtol"]),
        s_step=float(settings["s_step"]), s_updates=int(settings["s_updates"]),
        probe=probe, progress=progress, phase_prefix=phase_prefix)


def _adaptive_samples(sampler: BatchedHMCSampler, settings: dict, observe,
                      ess_transform=None, *,
                      progress: SimulationProgress | None = None) -> tuple[np.ndarray, dict]:
    minimum = int(settings["min_meas_per_boundary"])
    maximum = int(settings["max_meas_per_boundary"])
    target = float(settings["target_ess_per_boundary"])
    block = max(16, min(128, minimum // 4 or 1))
    samples, reason = [], "max_meas_per_boundary"
    if progress is not None:
        progress.phase("sampling", maximum)
    while len(samples) < maximum:
        count = min(block, maximum - len(samples))
        if len(samples) < minimum:
            count = min(count, minimum - len(samples))
        for _ in range(count):
            sampler.sweep(s_step=float(settings["s_step"]),
                          s_updates=int(settings["s_updates"]))
            samples.append(observe(sampler))
            if progress is not None:
                progress.update()
        if len(samples) >= minimum:
            primary = np.asarray(samples)
            stats = _per_boundary_stats(
                ess_transform(primary) if ess_transform else primary)
            if progress is not None:
                progress.update(0, samples=len(samples),
                                ess=f"{float(np.min(stats['ess'])):.1f}/{target:g}")
            if np.all(stats["ess"] >= target):
                reason = "target_ess_per_boundary"
                break
    data = np.asarray(samples)
    stats = _per_boundary_stats(ess_transform(data) if ess_transform else data)
    return data, {"minimum": minimum, "actual": len(data), "maximum": maximum,
                  "per_chain": len(data), "target_ess": target,
                  "tau": stats["tau"], "ess": stats["ess"],
                  "stop_reason": reason}


def _per_boundary_stats(values: np.ndarray) -> dict[str, np.ndarray]:
    """Return autocorrelation diagnostics without pooling frozen boundaries."""
    x = np.asarray(values, dtype=float)
    if x.ndim == 2:
        x = x[..., None]
    if x.ndim != 3:
        raise ValueError("patch samples must have shape [sweeps, chains, observables]")
    tau = np.empty((x.shape[1], x.shape[2]), dtype=float)
    for chain in range(x.shape[1]):
        for observable in range(x.shape[2]):
            tau[chain, observable] = autocorr_time(x[:, chain, observable])
    return {"tau": tau, "ess": x.shape[0] / (2 * np.maximum(tau, 0.5))}


def patch_batch_count(fit_times: int, chains: int) -> int:
    """Return the number of complete GPU batches needed for a fit ensemble."""
    if fit_times < 1 or chains < 1:
        raise ValueError("fit_times and chains must be positive")
    return int(math.ceil(fit_times / chains))


def _extract_rectangle(sampler: BatchedHMCSampler, width: int, height: int,
                       pad: int) -> RectanglePatch:
    if pad < 0 or pad + width >= sampler.Lx or pad + height >= sampler.Ly:
        raise ValueError("rectangle does not fit inside the sampler lattice")
    z, U, p = sampler.z, sampler.U, pad
    zloop = torch.cat((z[:, p:p + width, p], z[:, p + width, p:p + height],
                       torch.flip(z[:, p + 1:p + width + 1, p + height], (1,)),
                       torch.flip(z[:, p, p + 1:p + height + 1], (1,))), dim=1)
    uloop = torch.cat((U[:, p:p + width, p, 0], U[:, p + width, p:p + height, 1],
                       torch.flip(torch.conj(U[:, p:p + width, p + height, 0]), (1,)),
                       torch.flip(torch.conj(U[:, p, p:p + height, 1]), (1,))), dim=1)
    return RectanglePatch(
        z=z[:, p:p + width + 1, p:p + height + 1].detach().clone(),
        a=sampler.a[:, p:p + width + 1, p:p + height + 1].detach().clone(),
        s=sampler.s[:, p:p + width, p:p + height].detach().clone(),
        zloop=zloop.detach().clone(),
        aloop=torch.angle(uloop).detach().clone())


def _install_rectangle(sampler: BatchedHMCSampler,
                       patch_or_zloop: RectanglePatch | torch.Tensor,
                       aloop: torch.Tensor | None = None,
                       width: int | None = None, height: int | None = None) -> None:
    if isinstance(patch_or_zloop, RectanglePatch):
        patch = patch_or_zloop
        zloop, aloop = patch.zloop, patch.aloop
        width = patch.z.shape[1] - 1
        height = patch.z.shape[2] - 1
    else:
        zloop = patch_or_zloop
    if aloop is None or width is None or height is None:
        raise TypeError("aloop, width, and height are required for boundary-only installation")
    if zloop.shape[1] != 2 * (width + height):
        raise ValueError("boundary loop length mismatch")
    w, h = width, height
    with torch.no_grad():
        sampler.z[:, :w, 0] = zloop[:, :w]
        sampler.z[:, w, :h] = zloop[:, w:w + h]
        sampler.z[:, 1:w + 1, h] = torch.flip(zloop[:, w + h:w + h + w], (1,))
        sampler.z[:, 0, 1:h + 1] = torch.flip(zloop[:, w + h + w:], (1,))
        sampler.a[:, :w, 0, 0] = aloop[:, :w]
        sampler.a[:, w, :h, 1] = aloop[:, w:w + h]
        sampler.a[:, :w, h, 0] = -torch.flip(aloop[:, w + h:w + h + w], (1,))
        sampler.a[:, 0, :h, 1] = -torch.flip(aloop[:, w + h + w:], (1,))
    sampler.frozen_z[:, :, 0] = True
    sampler.frozen_z[:, :, h] = True
    sampler.frozen_z[:, 0, :] = True
    sampler.frozen_z[:, w, :] = True
    sampler.frozen_a[:, :w, 0, 0] = True
    sampler.frozen_a[:, :w, h, 0] = True
    sampler.frozen_a[:, 0, :h, 1] = True
    sampler.frozen_a[:, w, :h, 1] = True


def _restore_patch(sampler: BatchedHMCSampler, patch: RectanglePatch) -> None:
    """Restore the complete crop without changing masks or tuned HMC settings."""
    expected = (sampler.chains, sampler.Lx, sampler.Ly)
    if patch.z.shape[:3] != expected or patch.a.shape[:3] != expected:
        raise ValueError("patch shape does not match inner sampler")
    if patch.s.shape != (sampler.chains, sampler.Lx - 1, sampler.Ly - 1):
        raise ValueError("patch plaquette shape does not match inner sampler")
    with torch.no_grad():
        sampler.z.copy_(patch.z.to(sampler.device, sampler.cdtype))
        sampler.a.copy_(patch.a.to(sampler.device, sampler.dtype))
        sampler.s.zero_()
        sampler.s[:, :-1, :-1].copy_(patch.s.to(sampler.device, torch.int64))


def _adapt_inner(sampler: BatchedHMCSampler, patch: RectanglePatch,
                 settings: dict, *, progress: SimulationProgress | None = None) -> dict:
    """Tune on a random interior, then restore the cropped equilibrium state."""
    _install_rectangle(sampler, patch)
    adapt_steps = int(settings["adapt_steps"])
    if progress is not None:
        progress.phase("inner adapt", adapt_steps)

    def adaptation_progress(epsilon: float, accept: float) -> None:
        if progress is not None:
            progress.update(epsilon=f"{epsilon:.3g}", accept=f"{accept:.3f}")

    tuned = sampler.adapt(
        adapt_steps, float(settings["target_accept"]),
        s_step=float(settings["s_step"]), s_updates=int(settings["s_updates"]),
        progress=adaptation_progress)
    _restore_patch(sampler, patch)
    sampler.reset_diagnostics()
    return {"epsilon": tuned, "sweeps": 0, "tau": [], "stable": True,
            "stop_reason": "copied_equilibrium",
            "initialization": "copied_equilibrium", "adapt_steps": adapt_steps}


def _fine_couplings(cfg: dict, *, fold_alpha: bool) -> tuple[Couplings, str]:
    m = cfg["model"]
    if fold_alpha:
        return Couplings(float(m["beta"]), float(m["beta1"]),
                         float(m["alpha"] + m["alpha1"]), 0.0, int(m["mod"])), "half"
    if abs(float(m["alpha"])) < 1e-12:
        return Couplings(float(m["beta"]), float(m["beta1"]), float(m["alpha1"]),
                         0.0, int(m["mod"])), "half"
    return Couplings(float(m["beta"]), float(m["beta1"]), float(m["alpha"]),
                     float(m["alpha1"]), int(m["mod"])), "full"


def _boundary_data(zloop: torch.Tensor, aloop: torch.Tensor, L: int,
                   segments: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a_u, a_z = [], []
    for i in range(segments):
        sl = slice(i * L, (i + 1) * L)
        a_u.append(_wrap(aloop[:, sl].sum(1)))
        indices = torch.arange(i * L, (i + 1) * L, device=zloop.device)
        nxt = (indices + 1) % (segments * L)
        inner = torch.sum(torch.conj(zloop[:, indices]) * zloop[:, nxt], dim=-1)
        a_z.append(_wrap(torch.angle(inner).sum(1)))
    au, az = torch.stack(a_u, 1), torch.stack(a_z, 1)
    corners = zloop[:, torch.arange(segments, device=zloop.device) * L]
    return au.cpu().numpy(), az.cpu().numpy(), corners.cpu().numpy()


def _outer_boundary(cfg: dict, settings: dict, *, L: int, pad: int,
                    width: int, height: int, couplings: Couplings, kind: str,
                    seed_offset: int, progress: SimulationProgress | None = None
                    ) -> tuple[RectanglePatch, dict]:
    bc = cfg["geometry"]["boundary_bc"]
    periodic = bc == "PBC"
    if periodic and pad < 1:
        raise ValueError("automatic geometry requires padding >= 1 for PBC patches")
    add = 0 if periodic else 1
    outer = _make_sampler(cfg, settings, chains=int(settings["chains"]),
                          Lx=width + 2 * pad + add, Ly=height + 2 * pad + add,
                          couplings=couplings, kind=kind, periodic=periodic,
                          seed_offset=seed_offset)
    warmup = _warmup(outer, settings, progress=progress, phase_prefix="outer ")
    return _extract_rectangle(outer, width, height, pad), warmup


def _generate_two_plaq_batch(cfg: dict, geometry: dict, batch_index: int) -> dict:
    label = f"two_plaq batch {batch_index + 1}"
    with SimulationProgress(label) as progress:
        return _generate_two_plaq_batch_impl(cfg, geometry, batch_index, progress)


def _generate_two_plaq_batch_impl(cfg: dict, geometry: dict, batch_index: int,
                                   progress: SimulationProgress) -> dict:
    settings = section(cfg, "two_plaq")
    L, pad = int(cfg["renormalization"]["factor"]), int(geometry["padding"])
    c, kind = _fine_couplings(cfg, fold_alpha=True)
    patch, outer_warmup = _outer_boundary(
        cfg, settings, L=L, pad=pad, width=2 * L, height=L,
        couplings=c, kind=kind, seed_offset=20_000 + 2 * batch_index,
        progress=progress)
    sampler = _make_sampler(cfg, settings, chains=int(settings["chains"]),
                            Lx=2 * L + 1, Ly=L + 1, couplings=c, kind="half",
                            periodic=False, seed_offset=20_001 + 2 * batch_index)
    zloop, aloop = patch.zloop, patch.aloop
    warmup = _adapt_inner(sampler, patch, settings, progress=progress)
    def observe(item):
        conn_u = _wrap(item.a[:, L, :L, 1].sum(1))
        inner = torch.sum(torch.conj(sampler.z[:, L, 1:]) * sampler.z[:, L, :-1], -1)
        conn_z = _wrap(torch.angle(inner).sum(1))
        return torch.stack((conn_u, conn_z), -1).cpu().numpy()
    samples, sampling = _adaptive_samples(
        sampler, settings, observe,
        lambda values: np.concatenate((np.sin(values), np.cos(values)), axis=-1),
        progress=progress)
    au, az, corners = _boundary_data(zloop, aloop, L, 6)
    bins = int(settings["bins"])
    centers = np.linspace(-np.pi, np.pi, bins, endpoint=False) + np.pi / bins
    rows, freq_u, freq_z = [], [], []
    cu, cz = samples[..., 0], samples[..., 1]
    for chain in range(sampler.chains):
        freq_u.append(np.histogram(cu[:, chain], bins=bins, range=(-np.pi, np.pi), density=True)[0])
        freq_z.append(np.histogram(cz[:, chain], bins=bins, range=(-np.pi, np.pi), density=True)[0])
        overlap = np.sum(np.conj(corners[chain, 1]) * corners[chain, 4])
        base = np.column_stack((centers, au[chain][None].repeat(bins, 0),
                                np.full(bins, abs(overlap)), np.full(bins, np.angle(overlap))))
        rows.append(base)
    return {"X_U": np.concatenate(rows), "freq_U": np.concatenate(freq_u),
            "X_z": np.concatenate([np.column_stack((centers, az[i][None].repeat(bins, 0),
                                      np.full(bins, abs(np.sum(np.conj(corners[i, 1]) * corners[i, 4]))),
                                      np.full(bins, np.angle(np.sum(np.conj(corners[i, 1]) * corners[i, 4])))))
                                     for i in range(sampler.chains)]),
            "freq_z": np.concatenate(freq_z),
            "accept_rate": sampler.accept_rate.cpu().numpy(), "epsilon": sampler.epsilon,
            "outer_warmup_sweeps": outer_warmup["sweeps"],
            "outer_warmup_tau": outer_warmup["tau"],
            "outer_warmup_stop_reason": outer_warmup["stop_reason"],
            "warmup_sweeps": warmup["sweeps"], "warmup_tau": warmup["tau"],
            "warmup_stop_reason": warmup["stop_reason"],
            "inner_initialization": warmup["initialization"],
            "inner_adapt_steps": warmup["adapt_steps"],
            "sampling_tau": sampling["tau"], "sampling_ess": sampling["ess"],
            "sampling_minimum": sampling["minimum"],
            "sampling_actual": sampling["actual"],
            "sampling_maximum": sampling["maximum"],
            "sampling_per_chain": sampling["per_chain"],
            "sampling_target_ess": sampling["target_ess"],
            "sampling_stop_reason": sampling["stop_reason"]}


def _obc_topology(sampler: BatchedHMCSampler) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    da = sampler.plaquette_da()
    q_u = torch.angle(torch.exp(1j * da)).sum((1, 2)) / (2 * math.pi)
    z = sampler.z
    z00, z10, z01, z11 = z[:, :-1, :-1], z[:, 1:, :-1], z[:, :-1, 1:], z[:, 1:, 1:]
    vdot = lambda left, right: torch.sum(torch.conj(left) * right, dim=-1)
    tri0 = vdot(z00, z10) * vdot(z10, z11) * vdot(z11, z00)
    tri1 = vdot(z00, z11) * vdot(z11, z01) * vdot(z01, z00)
    q_z = (torch.angle(tri0) + torch.angle(tri1)).sum((1, 2)) / (2 * math.pi)
    q_s = sampler.s[:, :-1, :-1].sum((1, 2)).to(sampler.dtype)
    return q_u, q_z, q_s


def _generate_one_plaq_batch(cfg: dict, geometry: dict, batch_index: int) -> dict:
    label = f"one_plaq batch {batch_index + 1}"
    with SimulationProgress(label) as progress:
        return _generate_one_plaq_batch_impl(cfg, geometry, batch_index, progress)


def _generate_one_plaq_batch_impl(cfg: dict, geometry: dict, batch_index: int,
                                   progress: SimulationProgress) -> dict:
    settings = section(cfg, "one_plaq")
    L, pad = int(cfg["renormalization"]["factor"]), int(geometry["padding"])
    c, kind = _fine_couplings(cfg, fold_alpha=False)
    if cfg["renormalization"]["type"] == "U" and kind != "full":
        raise RuntimeError("U-renorm with alpha_f=0 has no constrained s sector")
    patch, outer_warmup = _outer_boundary(
        cfg, settings, L=L, pad=pad, width=L, height=L,
        couplings=c, kind=kind, seed_offset=30_000 + 2 * batch_index,
        progress=progress)
    sampler = _make_sampler(cfg, settings, chains=int(settings["chains"]), Lx=L + 1,
                            Ly=L + 1, couplings=c, kind=kind, periodic=False,
                            seed_offset=30_001 + 2 * batch_index)
    zloop, aloop = patch.zloop, patch.aloop
    warmup = _adapt_inner(sampler, patch, settings, progress=progress)
    au, az, _ = _boundary_data(zloop, aloop, L, 4)
    da_u, da_z = au.sum(1), az.sum(1)
    def observe(item):
        qu, qz, qs = _obc_topology(sampler)
        return torch.stack((qu, qz, qs), -1).cpu().numpy()
    q, sampling = _adaptive_samples(sampler, settings, observe, progress=progress)
    offsets = np.stack((da_u, da_z, da_u), -1) / (2 * np.pi)
    vortex = q - offsets[None]
    out: dict[str, np.ndarray | float] = {}
    for col, tag, da in ((0, "U", da_u), (1, "z", da_z), (2, "s", da_u)):
        xs, fs = [], []
        for chain in range(sampler.chains):
            values = vortex[:, chain, col]
            zp = int(settings["zero_pad"])
            edges = np.arange(np.floor(values.min()) - 0.5 - zp,
                              np.ceil(values.max()) + 0.6 + zp, 1.0)
            hist, edge = np.histogram(values, bins=edges, density=True)
            centers = np.round((edge[:-1] + edge[1:]) / 2)
            xs.append(np.column_stack((centers, np.full(len(centers), da[chain]))))
            fs.append(hist)
        out[f"X_{tag}"], out[f"freq_{tag}"] = np.concatenate(xs), np.concatenate(fs)
    out["accept_rate"] = sampler.accept_rate.cpu().numpy()
    out["s_accept_rate"] = (sampler.metro_accept_rate.cpu().numpy()
                            if kind == "full" else np.array([]))
    out["epsilon"] = sampler.epsilon
    out["outer_warmup_sweeps"] = outer_warmup["sweeps"]
    out["outer_warmup_tau"] = np.asarray(outer_warmup["tau"])
    out["outer_warmup_stop_reason"] = outer_warmup["stop_reason"]
    out["warmup_sweeps"] = warmup["sweeps"]
    out["warmup_tau"] = np.asarray(warmup["tau"])
    out["warmup_stop_reason"] = warmup["stop_reason"]
    out["inner_initialization"] = warmup["initialization"]
    out["inner_adapt_steps"] = warmup["adapt_steps"]
    out["sampling_tau"] = sampling["tau"]
    out["sampling_ess"] = sampling["ess"]
    out["sampling_minimum"] = sampling["minimum"]
    out["sampling_actual"] = sampling["actual"]
    out["sampling_maximum"] = sampling["maximum"]
    out["sampling_per_chain"] = sampling["per_chain"]
    out["sampling_target_ess"] = sampling["target_ess"]
    out["sampling_stop_reason"] = sampling["stop_reason"]
    return out


def _batch_key(cfg: dict, geometry: dict, stage: str, batch_index: int) -> str:
    clean_cfg = {key: value for key, value in cfg.items() if not key.startswith("_")}
    return fingerprint({"schema": PATCH_CACHE_SCHEMA, "stage": stage, "batch": batch_index,
                        "config": clean_cfg, "geometry": geometry})


def _batch_summary(stage: str, batch_index: int, result: dict) -> str | None:
    required = {"sampling_ess", "accept_rate", "outer_warmup_sweeps",
                "outer_warmup_stop_reason", "warmup_sweeps", "warmup_stop_reason",
                "sampling_actual", "sampling_target_ess", "sampling_stop_reason",
                "epsilon"}
    if not required.issubset(result):
        return None
    ess_min = float(np.min(np.asarray(result["sampling_ess"], dtype=float)))
    accept = float(np.mean(np.asarray(result["accept_rate"], dtype=float)))
    return (f"finished {stage} batch {batch_index + 1}: "
            f"warmup=outer:{int(result['outer_warmup_sweeps'])}"
            f"({result['outer_warmup_stop_reason']})/"
            f"inner:{int(result['warmup_sweeps'])}({result['warmup_stop_reason']}) "
            f"samples={int(result['sampling_actual'])}/boundary "
            f"ess={ess_min:.1f}/{float(result['sampling_target_ess']):g} "
            f"stop={result['sampling_stop_reason']} accept={accept:.3f} "
            f"epsilon={float(result['epsilon']):.4g}")


def _load_or_generate_batch(cfg: dict, geometry: dict, *, stage: str,
                            batch_index: int, cache_dir: Path | None,
                            generate) -> dict:
    key = _batch_key(cfg, geometry, stage, batch_index)
    path = None if cache_dir is None else cache_dir / f"batch_{batch_index:05d}.npz"
    if path is not None and path.exists() and not cfg.get("_force"):
        try:
            with np.load(path, allow_pickle=False) as stored:
                if str(stored["batch_fingerprint"].item()) == key:
                    print(f"[cpn-renorm] reuse {stage} batch {batch_index + 1}", flush=True)
                    return {name: stored[name] for name in stored.files
                            if name != "batch_fingerprint"}
        except (KeyError, OSError, ValueError):
            pass
    print(f"[cpn-renorm] simulate {stage} batch {batch_index + 1}", flush=True)
    result = generate(cfg, geometry, batch_index)
    summary = _batch_summary(stage, batch_index, result)
    if summary is not None:
        print(f"[cpn-renorm] {summary}", flush=True)
    if path is not None:
        atomic_npz(path, batch_fingerprint=np.asarray(key), **result)
    return result


def _aggregate_batches(batches: list[dict], settings: dict,
                       data_keys: tuple[str, ...]) -> dict:
    out = {key: np.concatenate([np.asarray(batch[key]) for batch in batches])
           for key in data_keys}
    out["accept_rate"] = np.concatenate(
        [np.asarray(batch["accept_rate"]) for batch in batches])
    if "s_accept_rate" in batches[0]:
        values = [np.asarray(batch["s_accept_rate"]) for batch in batches]
        out["s_accept_rate"] = (np.concatenate(values) if any(x.size for x in values)
                                else np.array([]))
    for key in ("epsilon", "outer_warmup_sweeps", "outer_warmup_stop_reason",
                "warmup_sweeps", "warmup_stop_reason", "inner_initialization",
                "inner_adapt_steps",
                "sampling_actual", "sampling_per_chain", "sampling_stop_reason"):
        out[key] = np.asarray([np.asarray(batch[key]).item() for batch in batches])
    for key in ("outer_warmup_tau", "warmup_tau", "sampling_tau", "sampling_ess"):
        out[key] = np.stack([np.asarray(batch[key]) for batch in batches])
    chains = int(settings["chains"])
    requested = int(settings["fit_times"])
    out.update({
        "requested_fit_times": requested,
        "effective_fit_times": len(batches) * chains,
        "batch_count": len(batches),
        "chains": chains,
        "sampling_minimum": int(settings["min_meas_per_boundary"]),
        "sampling_maximum": int(settings["max_meas_per_boundary"]),
        "sampling_target_ess": float(settings["target_ess_per_boundary"]),
        "sampling_total_actual": int(sum(int(np.asarray(x["sampling_actual"]).item())
                                             for x in batches) * chains),
    })
    return out


def generate_two_plaq(cfg: dict, geometry: dict,
                      cache_dir: str | Path | None = None) -> dict:
    settings = section(cfg, "two_plaq")
    count = patch_batch_count(int(settings["fit_times"]), int(settings["chains"]))
    print(f"[cpn-renorm] two_plaq boundaries: requested={settings['fit_times']} "
          f"effective={count * int(settings['chains'])} batches={count}", flush=True)
    root = None if cache_dir is None else Path(cache_dir)
    batches = [_load_or_generate_batch(
        cfg, geometry, stage="two_plaq", batch_index=index, cache_dir=root,
        generate=_generate_two_plaq_batch) for index in range(count)]
    return _aggregate_batches(batches, settings, ("X_U", "freq_U", "X_z", "freq_z"))


def generate_one_plaq(cfg: dict, geometry: dict,
                      cache_dir: str | Path | None = None) -> dict:
    settings = section(cfg, "one_plaq")
    count = patch_batch_count(int(settings["fit_times"]), int(settings["chains"]))
    print(f"[cpn-renorm] one_plaq boundaries: requested={settings['fit_times']} "
          f"effective={count * int(settings['chains'])} batches={count}", flush=True)
    root = None if cache_dir is None else Path(cache_dir)
    batches = [_load_or_generate_batch(
        cfg, geometry, stage="one_plaq", batch_index=index, cache_dir=root,
        generate=_generate_one_plaq_batch) for index in range(count)]
    return _aggregate_batches(
        batches, settings, ("X_U", "freq_U", "X_z", "freq_z", "X_s", "freq_s"))
