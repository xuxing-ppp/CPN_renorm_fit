from __future__ import annotations

import math
from pathlib import Path

import torch

from .config import fine_model_key, fingerprint, mod_config, resolved_geometry, section
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
    # exactly represented by halfRefVil, so pilot xi_G must not create a random s walk.
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


def _pilot_fingerprint(cfg: dict) -> str:
    return fingerprint({"model": fine_model_key(cfg["model"]),
                        "renormalization": {"factor": cfg["renormalization"]["factor"]},
                        "runtime": cfg["runtime"], "pilot": cfg["pilot"],
                        "hmc": cfg["hmc"], "chains": cfg["chains"]["pilot"],
                        "geometry": cfg["geometry"]})




def run_pilot(cfg: dict, run_dir: str | Path) -> dict:
    if isinstance(cfg["model"]["mod"], list):
        mods = cfg["model"]["mod"]
        if float(cfg["model"]["beta1"]) == 0 or len(mods) == 1:
            scalar = mod_config(cfg, 0 if float(cfg["model"]["beta1"]) == 0 else mods[0])
            scalar["_tuning_records"] = {}
            result = run_pilot(scalar, run_dir)
            if cfg["runtime"]["auto_chains"]:
                atomic_json(Path(run_dir) / "chain_tuning.json", {
                    "stages": scalar["_tuning_records"], "chains": {"pilot": scalar["chains"]["pilot"]}})
            return result
        results, records, chains = {}, {}, {}
        for mod in mods:
            scalar = mod_config(cfg, mod)
            scalar["_tuning_records"] = records
            scalar["_tuning_prefix"] = f"mod{mod}/"
            results[str(mod)] = run_pilot(scalar, Path(run_dir) / "common" / f"mod{mod}")
            chains[f"mod{mod}"] = scalar["chains"]["pilot"]
        if cfg["runtime"]["auto_chains"]:
            atomic_json(Path(run_dir) / "chain_tuning.json", {"stages": records, "chains": chains})
        return {"mods": results}
    p = cfg["pilot"]
    settings = section(cfg, "pilot")
    L, attempts = int(p["initial_L"]), []
    run_dir = Path(run_dir)
    result_path = run_dir / "pilot" / "result.json"
    pilot_key = _pilot_fingerprint(cfg)
    if result_path.exists() and not cfg.get("_force"):
        existing = __import__("json").loads(result_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") == pilot_key:
            existing["fingerprint"] = pilot_key
            existing["cache"] = "reused"
            atomic_json(result_path, existing)
            return existing
    while True:
        if cfg["runtime"]["auto_chains"]:
            from .chain_tuning import tune_pilot
            tune_pilot(cfg, L)
            settings = section(cfg, "pilot")
            pilot_key = _pilot_fingerprint(cfg)
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
            xi_G, xi_G_err = float(measured["xi_G"]), float(measured["xi_G_err"])
            if math.isfinite(xi_G) and xi_G > 0 and math.isfinite(xi_G_err):
                failure = None
                break
            failure = "Gamma-method xi_G estimate is not finite and positive"
            attempts.append({"L": L, "retry": retry, "valid": False,
                             "reason": failure,
                             "min_meas_total": int(p["min_meas_total"]) * multiplier})
        if failure is not None:
            atomic_json(run_dir / "pilot" / "failed.json", {"attempts": attempts})
            raise RuntimeError(f"pilot xi_G failed after doubled-statistics retry: {failure}")
        ratio = L / (xi_G + 2 * xi_G_err) if xi_G > 0 and math.isfinite(xi_G) else 0.0
        metadata, arrays = split_measurement(measured)
        if cfg.get("storage", {}).get("save_npz", False):
            atomic_npz(run_dir / "pilot" / f"L{L}.npz", **arrays)
        attempts.append({"L": L, "retry": retry, "valid": True,
                         "chains": settings["chains"],
                         "xi_G": xi_G, "xi_G_err": xi_G_err,
                         "conservative_L_over_xi_G": ratio, **metadata})
        if ratio >= float(p["min_L_over_xi_G"]):
            result = {"xi_G": xi_G, "xi_G_err": xi_G_err, "attempts": attempts,
                      "chains": settings["chains"],
                      "fingerprint": pilot_key, "cache": "generated",
                      "geometry": resolved_geometry(cfg, xi_G)}
            atomic_json(run_dir / "pilot" / "result.json", result)
            return result
        new_L = min(int(p["max_L"]), int(math.ceil(L * p["growth_factor"])))
        if new_L <= L:
            atomic_json(run_dir / "pilot" / "failed.json", {"attempts": attempts})
            raise RuntimeError("pilot reached max_L before finite-volume criterion passed")
        L = new_L
