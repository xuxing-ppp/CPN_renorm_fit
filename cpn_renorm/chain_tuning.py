from __future__ import annotations

import gc
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from .config import section
from .pilot import resolve_device, run_pilot, torch_dtype
from .sampler import ActionKind, BatchedHMCSampler, Couplings
from .storage import atomic_json


@dataclass(frozen=True)
class Workload:
    """One sampler shape contributing to a pipeline stage."""

    name: str
    Lx: int
    Ly: int
    periodic: bool
    kind: ActionKind
    couplings: Couplings
    projected_sweeps: int
    n_meas: int = 0
    stores_correlations: bool = False
    measurement_total: int = 0
    ensembles: int = 1
    batched_ensembles: bool = False
    settings_name: str | None = None


def chain_candidates(max_chains: int) -> list[int]:
    if max_chains < 1:
        raise ValueError("max_chains must be positive")
    values, value = [], 1
    while value <= max_chains:
        values.append(value)
        value *= 2
    if values[-1] != max_chains:
        values.append(max_chains)
    return values


def recommend_chains(rows: list[dict], throughput_fraction: float = 0.9) -> int | None:
    """Choose the fastest safe count for a common statistical target."""
    safe = [row for row in rows if row.get("safe") and row.get("chain_sweeps_per_s")]
    if not safe:
        return None
    timed = [row for row in safe if row.get("projected_target_seconds") is not None]
    if timed:
        best = min(row["projected_target_seconds"] for row in timed)
        return min(row["chains"] for row in timed
                   if row["projected_target_seconds"] <= best / throughput_fraction)
    target = max(row["chain_sweeps_per_s"] for row in safe) * throughput_fraction
    return min(row["chains"] for row in safe if row["chain_sweeps_per_s"] >= target)


def _fine_action(cfg: dict) -> tuple[Couplings, ActionKind]:
    model = cfg["model"]
    alpha = float(model["alpha"])
    if abs(alpha) > 1e-12:
        return (Couplings(float(model["beta"]), float(model["beta1"]), alpha,
                          float(model["alpha1"]), int(model["mod"])), "full")
    return (Couplings(float(model["beta"]), float(model["beta1"]),
                      float(model["alpha1"]), 0.0, int(model["mod"])), "half")


def _sampling_sweeps(settings: dict) -> int:
    return int(settings["adapt_steps"]) + int(settings["max_warmup"]) + int(math.ceil(
        settings["max_meas_total"] / settings["chains"]))


def _workloads(cfg: dict, pilot: dict) -> dict[str, list[Workload]]:
    model, geometry = cfg["model"], pilot["geometry"]
    factor, pad = int(geometry["factor"]), int(geometry["padding"])
    fine_L, coarse_L = int(geometry["L_fine"]), int(geometry["L_coarse"])
    base, base_kind = _fine_action(cfg)
    folded = Couplings(float(model["beta"]), float(model["beta1"]),
                       float(model["alpha"] + model["alpha1"]), 0.0,
                       int(model["mod"]))

    valid_attempts = [x for x in pilot["attempts"] if x.get("valid")]
    pilot_L = int(valid_attempts[-1]["L"])
    pilot_settings = cfg["runtime"] | cfg["hmc"] | cfg["pilot"]
    stages: dict[str, list[Workload]] = {
        "pilot": [Workload("pbc", pilot_L, pilot_L, True, base_kind, base,
                           _sampling_sweeps(pilot_settings),
                           int(math.ceil(pilot_settings["max_meas_total"] /
                                         pilot_settings["chains"])), True,
                           int(pilot_settings["max_meas_total"]),
                           settings_name="pilot")]
    }

    for stage, width, height in (("two_plaq", 2 * factor, factor),
                                 ("one_plaq", factor, factor)):
        settings = section(cfg, stage)
        fit_times = int(settings["fit_times"])
        periodic = cfg["geometry"]["boundary_bc"] == "PBC"
        add = 0 if periodic else 1
        inner_kind = "half" if stage == "two_plaq" else base_kind
        inner_couplings = folded if stage == "two_plaq" else base
        stages[stage] = [
            Workload("outer_boundary", width + 2 * pad + add,
                     height + 2 * pad + add, periodic, inner_kind,
                     inner_couplings,
                     int(settings["adapt_steps"]) + int(settings["max_warmup"]),
                     ensembles=fit_times, batched_ensembles=True,
                     settings_name=stage),
            Workload("inner_patch", width + 1, height + 1, False, inner_kind,
                     inner_couplings,
                     int(settings["adapt_steps"]) +
                     int(settings["max_meas_per_boundary"]),
                     int(settings["max_meas_per_boundary"]), False, 0,
                     fit_times, True, stage),
        ]

    observable = section(cfg, "observable")
    beta_points = len(cfg["scan"]["beta1"].get("points", [])) or 3
    obs_sweeps = _sampling_sweeps(observable)
    stages["observable"] = [
        Workload("fine", fine_L, fine_L, True, "half", folded, obs_sweeps,
                 int(math.ceil(observable["max_meas_total"] / observable["chains"])), True,
                 int(observable["max_meas_total"]), settings_name="observable"),
        Workload("coarse_scan", coarse_L, coarse_L, True, "half", folded,
                 obs_sweeps * beta_points,
                 int(math.ceil(observable["max_meas_total"] / observable["chains"]))
                 * beta_points, True,
                 int(observable["max_meas_total"]) * beta_points, beta_points,
                 settings_name="observable"),
    ]

    skip_topo = (cfg["renormalization"]["type"] == "U"
                 and abs(float(model["alpha"])) < 1e-12)
    if not skip_topo:
        topo = section(cfg, "topo")
        topo_sweeps = _sampling_sweeps(topo)
        alpha_points = len(cfg["scan"]["alpha"].get("points", [])) or 5
        coarse_alpha = max(float(cfg["scan"]["alpha"]["min"]), 1e-6)
        coarse = Couplings(float(model["beta"]), float(model["beta1"]),
                           coarse_alpha, 0.0, int(model["mod"]))
        stages["topo"] = [
            Workload("fine", fine_L, fine_L, True, base_kind, base, topo_sweeps,
                     int(math.ceil(topo["max_meas_total"] / topo["chains"])), True,
                     int(topo["max_meas_total"]), settings_name="topo"),
            Workload("coarse_scans", coarse_L, coarse_L, True, "full", coarse,
                     topo_sweeps * 2 * alpha_points,
                     int(math.ceil(topo["max_meas_total"] / topo["chains"]))
                     * 2 * alpha_points, True,
                     int(topo["max_meas_total"]) * 2 * alpha_points,
                     2 * alpha_points, settings_name="topo"),
        ]
    return stages


def _host_gib(workload: Workload, chains: int, dtype: torch.dtype) -> float:
    if not workload.stores_correlations:
        return 0.0
    itemsize = 8 if dtype == torch.float64 else 4
    # Correlations are accumulated online (sum and sum of squares).  Only the
    # small mode/probe series grows with the number of measurements.
    series_values = (workload.measurement_total if workload.measurement_total
                     else workload.n_meas * chains)
    byte_count = (2 * chains * workload.Lx * workload.Ly * itemsize
                  + 8 * series_values * itemsize)
    return byte_count / 1024 ** 3


def _make_sampler(cfg: dict, workload: Workload, chains: int) -> BatchedHMCSampler:
    settings = (cfg["runtime"] | cfg["hmc"] | cfg["pilot"]
                if workload.settings_name == "pilot" else
                section(cfg, workload.settings_name) if workload.settings_name else
                cfg["runtime"] | cfg["hmc"])
    return BatchedHMCSampler(
        chains=chains, Lx=workload.Lx, Ly=workload.Ly,
        N=int(cfg["model"]["N"]), couplings=workload.couplings,
        kind=workload.kind, periodic=workload.periodic,
        device=resolve_device(cfg["runtime"]["device"]),
        dtype=torch_dtype(cfg["runtime"]["dtype"]),
        seed=int(cfg["runtime"]["seed"]) + chains,
        epsilon=float(settings["epsilon"]),
        trajectory_length=float(settings["trajectory_length"]),
        mass_a=float(settings["mass_a"]), mass_z=float(settings["mass_z"]))


def _time_workload(cfg: dict, workload: Workload, chains: int,
                   warmup_sweeps: int, timed_sweeps: int) -> dict:
    device = resolve_device(cfg["runtime"]["device"])
    settings = (cfg["runtime"] | cfg["hmc"] | cfg["pilot"]
                if workload.settings_name == "pilot" else
                section(cfg, workload.settings_name) if workload.settings_name else
                cfg["runtime"] | cfg["hmc"])
    sampler = _make_sampler(cfg, workload, chains)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(warmup_sweeps):
        sampler.sweep(s_step=float(settings["s_step"]),
                      s_updates=int(settings["s_updates"]))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(timed_sweeps):
        sampler.sweep(s_step=float(settings["s_step"]),
                      s_updates=int(settings["s_updates"]))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = (time.perf_counter() - started) / timed_sweeps
    peak = (torch.cuda.max_memory_allocated(device) / 1024 ** 3
            if device.type == "cuda" else None)
    del sampler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"name": workload.name, "batch_seconds": seconds,
            "peak_gpu_gib": peak}


def _benchmark_stage(cfg: dict, workloads: list[Workload], candidates: list[int],
                     *, warmup_sweeps: int, timed_sweeps: int,
                     gpu_memory_fraction: float, max_host_memory_gib: float) -> dict:
    device = resolve_device(cfg["runtime"]["device"])
    dtype = torch_dtype(cfg["runtime"]["dtype"])
    gpu_total = (torch.cuda.get_device_properties(device).total_memory / 1024 ** 3
                 if device.type == "cuda" else None)
    rows = []
    for chains in candidates:
        timings, oom = [], False
        try:
            for workload in workloads:
                timings.append(_time_workload(cfg, workload, chains,
                                              warmup_sweeps, timed_sweeps))
        except (torch.OutOfMemoryError, MemoryError):
            oom = True
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if oom:
            rows.append({"chains": chains, "safe": False, "reason": "out_of_memory"})
            break
        target_sweeps = []
        too_short = False
        for workload in workloads:
            if workload.batched_ensembles:
                batches = int(math.ceil(workload.ensembles / chains))
                target_sweeps.append(workload.projected_sweeps * batches)
            elif workload.measurement_total:
                per_ensemble = int(math.ceil(
                    workload.measurement_total / (workload.ensembles * chains)))
                too_short |= per_ensemble < 20
                measurement_sweeps = per_ensemble * workload.ensembles
                target_sweeps.append(workload.projected_sweeps - workload.n_meas
                                     + measurement_sweeps)
            else:
                target_sweeps.append(workload.projected_sweeps)
        projected_seconds = sum(
            timing["batch_seconds"] * sweeps
            for timing, sweeps in zip(timings, target_sweeps))
        total_chain_sweeps = chains * sum(target_sweeps)
        peak_gpu = max((x["peak_gpu_gib"] or 0.0) for x in timings)
        host_gib = max(_host_gib(x, chains, dtype) for x in workloads)
        gpu_ok = gpu_total is None or peak_gpu <= gpu_total * gpu_memory_fraction
        host_ok = host_gib <= max_host_memory_gib
        reasons = []
        if not gpu_ok:
            reasons.append("gpu_memory_limit")
        if not host_ok:
            reasons.append("host_measurement_memory_limit")
        if too_short:
            reasons.append("per_chain_series_too_short")
        rows.append({
            "chains": chains,
            "chain_sweeps_per_s": total_chain_sweeps / projected_seconds,
            "projected_batch_seconds": projected_seconds,
            "projected_target_seconds": projected_seconds,
            "peak_gpu_gib": peak_gpu if gpu_total is not None else None,
            "estimated_host_measurement_gib": host_gib,
            "safe": gpu_ok and host_ok and not too_short,
            "reason": ",".join(reasons) or None,
            "workloads": timings,
        })
        if not gpu_ok:
            break
    return {"recommended_chains": recommend_chains(rows), "candidates": rows}


def run_chain_tuning(cfg: dict, output_dir: str | Path, *, max_chains: int = 256,
                     warmup_sweeps: int = 2, timed_sweeps: int = 5,
                     gpu_memory_fraction: float = 0.8,
                     max_host_memory_gib: float = 4.0) -> dict:
    """Run/reuse the pilot and benchmark safe chain counts for each stage."""
    if warmup_sweeps < 0 or timed_sweeps < 1:
        raise ValueError("benchmark sweep counts are invalid")
    if not 0 < gpu_memory_fraction <= 1 or max_host_memory_gib <= 0:
        raise ValueError("benchmark memory limits are invalid")
    output_dir = Path(output_dir)
    run_dir = output_dir / Path(cfg["_config_path"]).stem
    pilot = run_pilot(cfg, run_dir)
    candidates = chain_candidates(max_chains)
    stages = {}
    for name, workloads in _workloads(cfg, pilot).items():
        print(f"[cpn-renorm] tune chains for {name}", flush=True)
        stages[name] = _benchmark_stage(
            cfg, workloads, candidates, warmup_sweeps=warmup_sweeps,
            timed_sweeps=timed_sweeps,
            gpu_memory_fraction=gpu_memory_fraction,
            max_host_memory_gib=max_host_memory_gib)
    report = {
        "device": str(resolve_device(cfg["runtime"]["device"])),
        "dtype": cfg["runtime"]["dtype"],
        "geometry": pilot["geometry"],
        "limits": {"max_chains": max_chains,
                   "gpu_memory_fraction": gpu_memory_fraction,
                   "max_host_memory_gib": max_host_memory_gib},
        "stages": stages,
    }
    report["toml"] = _toml_suggestion(report)
    atomic_json(run_dir / "chain_tuning.json", report)
    return report


def _toml_suggestion(report: dict) -> str:
    lines = []
    mapping = (("pilot", "pilot"), ("two_plaq", "steps.two_plaq"),
               ("observable", "steps.observable"),
               ("one_plaq", "steps.one_plaq"), ("topo", "steps.topo"))
    for stage, section_name in mapping:
        if stage not in report["stages"]:
            continue
        value = report["stages"][stage]["recommended_chains"]
        if value is not None:
            lines.extend((f"[{section_name}]", f"chains = {value}", ""))
    return "\n".join(lines).rstrip()
