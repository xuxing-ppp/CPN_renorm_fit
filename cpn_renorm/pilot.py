from __future__ import annotations

import math
from pathlib import Path

import torch

from .config import fingerprint, resolved_geometry, section
from .observables import measure_ensemble
from .progress import SimulationProgress, ensemble_summary
from .sampler import BatchedHMCSampler, Couplings
from .storage import atomic_json, atomic_npz, split_measurement


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested ({name}) but torch.cuda.is_available() is false")
    return device


def torch_dtype(name: str) -> torch.dtype:
    return torch.float64 if name == "float64" else torch.float32


def fine_sampler(cfg: dict, L: int, settings: dict, *, seed_offset: int = 0,
                 chains: int | None = None) -> BatchedHMCSampler:
    m, rt = cfg["model"], cfg["runtime"]
    # At alpha=0 the integer field is unconstrained. The alpha1 cosine term is
    # exactly represented by halfRefVil, so pilot xi must not create a random s walk.
    kind = "full" if abs(float(m["alpha"])) > 1e-12 else "half"
    alpha = float(m["alpha"]) if kind == "full" else float(m["alpha"] + m["alpha1"])
    alpha1 = float(m["alpha1"]) if kind == "full" else 0.0
    return BatchedHMCSampler(
        chains=int(chains or settings["chains"]), Lx=L, Ly=L, N=int(m["N"]),
        couplings=Couplings(float(m["beta"]), float(m["beta1"]), alpha,
                            alpha1, int(m["mod"])), kind=kind, periodic=True,
        device=resolve_device(rt["device"]), dtype=torch_dtype(rt["dtype"]),
        seed=int(rt["seed"]) + seed_offset, epsilon=float(settings["epsilon"]),
        trajectory_length=float(settings["trajectory_length"]), mass_a=float(settings["mass_a"]),
        mass_z=float(settings["mass_z"]))


def _pilot_fingerprint(cfg: dict, *, schema: int = 5) -> str:
    return fingerprint({"model": cfg["model"],
                        "renormalization": {"factor": cfg["renormalization"]["factor"]},
                        "runtime": cfg["runtime"], "pilot": cfg["pilot"],
                        "hmc": cfg["hmc"], "chains": cfg["chains"]["pilot"],
                        "geometry": cfg["geometry"], "schema": schema})


def _legacy_pilot_fingerprints(cfg: dict) -> set[str]:
    geometry = dict(cfg["geometry"])
    for key in ("padding_min", "padding_max", "coarse_L_min", "coarse_L_max"):
        geometry.pop(key, None)
    return {fingerprint({"model": cfg["model"],
                         "renormalization": {"type": kind,
                                             "factor": cfg["renormalization"]["factor"]},
                         "runtime": cfg["runtime"], "pilot": cfg["pilot"],
                         "hmc": cfg["hmc"], "chains": cfg["chains"]["pilot"],
                         "geometry": geometry, "schema": 4})
            for kind in ("U", "z")}


def run_pilot(cfg: dict, run_dir: str | Path) -> dict:
    p = cfg["pilot"]
    settings = section(cfg, "pilot")
    L, attempts = int(p["initial_L"]), []
    run_dir = Path(run_dir)
    result_path = run_dir / "pilot" / "result.json"
    pilot_key = _pilot_fingerprint(cfg)
    if result_path.exists() and not cfg.get("_force"):
        existing = __import__("json").loads(result_path.read_text(encoding="utf-8"))
        if (existing.get("fingerprint") == pilot_key or
                existing.get("fingerprint") in _legacy_pilot_fingerprints(cfg)):
            existing["fingerprint"] = pilot_key
            existing["cache"] = "reused"
            atomic_json(result_path, existing)
            return existing
    while True:
        failure = None
        for retry in range(2):
            multiplier = 2 ** retry
            sampler = fine_sampler(cfg, L, settings, seed_offset=10_000 + L + retry)
            label = f"pilot L={L} retry={retry + 1}"
            with SimulationProgress(label) as progress:
                measured = measure_ensemble(
                    sampler, min_warmup=int(p["min_warmup"]),
                    max_warmup=int(p["max_warmup"]),
                    min_meas_total=int(p["min_meas_total"]) * multiplier,
                    target_ess=float(p["target_ess"]),
                    max_meas_total=int(p["max_meas_total"]) * multiplier,
                    adapt_steps=int(settings["adapt_steps"]),
                    target_accept=float(settings["target_accept"]),
                    warmup_tau_multiplier=float(settings["warmup_tau_multiplier"]),
                    tau_stability_rtol=float(settings["tau_stability_rtol"]),
                    s_step=float(settings["s_step"]), s_updates=int(settings["s_updates"]),
                    progress=progress)
            print(f"[cpn-renorm] {ensemble_summary(label, measured)}", flush=True)
            xi, xi_err = float(measured["xi"]), float(measured["xi_err"])
            if math.isfinite(xi) and xi > 0 and math.isfinite(xi_err):
                failure = None
                break
            failure = "Gamma-method xi estimate is not finite and positive"
            attempts.append({"L": L, "retry": retry, "valid": False,
                             "reason": failure,
                             "min_meas_total": int(p["min_meas_total"]) * multiplier})
        if failure is not None:
            atomic_json(run_dir / "pilot" / "failed.json", {"attempts": attempts})
            raise RuntimeError(f"pilot xi failed after doubled-statistics retry: {failure}")
        ratio = L / (xi + 2 * xi_err) if xi > 0 and math.isfinite(xi) else 0.0
        metadata, arrays = split_measurement(measured)
        atomic_npz(run_dir / "pilot" / f"L{L}.npz", **arrays)
        attempts.append({"L": L, "retry": retry, "valid": True,
                         "xi": xi, "xi_err": xi_err,
                         "conservative_L_over_xi": ratio, **metadata})
        if ratio >= float(p["min_L_over_xi"]):
            result = {"xi": xi, "xi_err": xi_err, "attempts": attempts,
                      "fingerprint": pilot_key, "cache": "generated",
                      "geometry": resolved_geometry(cfg, xi)}
            atomic_json(run_dir / "pilot" / "result.json", result)
            return result
        new_L = min(int(p["max_L"]), int(math.ceil(L * p["growth_factor"])))
        if new_L <= L:
            atomic_json(run_dir / "pilot" / "failed.json", {"attempts": attempts})
            raise RuntimeError("pilot reached max_L before finite-volume criterion passed")
        L = new_L
