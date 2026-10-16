from __future__ import annotations

import gc
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterator

import torch

from .config import DEFAULTS, mod_config, section
from .pilot import resolve_device, run_pilot, torch_dtype
from .progress import SimulationProgress
from .sampler import ActionKind, BatchedHMCSampler, Couplings
from .storage import atomic_json
from .workspace import workspace_dir


SEARCH_GROWTH_FACTOR = 4
SEARCH_MIN_IMPROVEMENT = 0.05
SEARCH_STALL_LIMIT = 2


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
    if isinstance(cfg["model"]["mod"], list):
        cfg = mod_config(cfg, cfg["model"]["mod"][0])
    model, geometry = cfg["model"], pilot["geometry"]
    factor, pad = int(geometry["factor"]), int(geometry["padding"])
    fine_L, coarse_L = int(geometry["L_fine"]), int(geometry["L_coarse"])
    base, base_kind = _fine_action(cfg)
    folded = Couplings(float(model["beta"]), float(model["beta1"]),
                       float(model["alpha"] + model["alpha1"]), 0.0,
                       int(model["mod"]))

    valid_attempts = [x for x in pilot["attempts"] if x.get("valid")]
    pilot_L = int(valid_attempts[-1]["L"])
    pilot_settings = section(cfg, "pilot")
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
        if stage == "one_plaq" and "1plaq" not in cfg["renormalization"]["alpha_methods"]:
            continue
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
    beta_scan = cfg["scan"]["beta1"]
    beta_samples = beta_scan.get("points") or [
        x for x in (-0.1, 0.0, 0.1) if beta_scan["min"] <= x <= beta_scan["max"]]
    coarse_beta1 = max(beta_samples or [beta_scan["min"], beta_scan["max"]], key=abs)
    coarse_folded = replace(folded, beta1=float(coarse_beta1))
    beta_points = len(cfg["scan"]["beta1"].get("points", [])) or 3
    obs_sweeps = _sampling_sweeps(observable)
    stages["observable"] = [
        Workload("fine", fine_L, fine_L, True, "half", folded, obs_sweeps,
                 int(math.ceil(observable["max_meas_total"] / observable["chains"])), True,
                 int(observable["max_meas_total"]), settings_name="observable"),
        Workload("coarse_scan", coarse_L, coarse_L, True, "half", coarse_folded,
                 obs_sweeps * beta_points,
                 int(math.ceil(observable["max_meas_total"] / observable["chains"]))
                 * beta_points, True,
                 int(observable["max_meas_total"]) * beta_points, beta_points,
                 settings_name="observable"),
    ]

    types = set(cfg["renormalization"]["type"])
    if not cfg["renormalization"]["beta1_methods"]:
        stages.pop("observable", None)
    topo_enabled = ("topo" in cfg["renormalization"]["alpha_methods"] and
                    not (types == {"U"} and abs(float(model["alpha"])) < 1e-12))
    if topo_enabled:
        topo = section(cfg, "topo")
        topo_sweeps = _sampling_sweeps(topo)
        alpha_points = len(cfg["scan"]["alpha"].get("points", [])) or 3
        coarse_alpha = max(float(cfg["scan"]["alpha"]["min"]), 1e-6)
        coarse = Couplings(float(model["beta"]), float(coarse_beta1),
                           coarse_alpha, 0.0, int(model["mod"]))
        stages["topo"] = [
            Workload("fine", fine_L, fine_L, True, base_kind, base, topo_sweeps,
                     int(math.ceil(topo["max_meas_total"] / topo["chains"])), True,
                     int(topo["max_meas_total"]), settings_name="topo"),
            Workload("coarse_scans", coarse_L, coarse_L, True, "full", coarse,
                     topo_sweeps * len(cfg["renormalization"]["beta1_methods"]) * alpha_points,
                     int(math.ceil(topo["max_meas_total"] / topo["chains"]))
                     * len(cfg["renormalization"]["beta1_methods"]) * alpha_points, True,
                     int(topo["max_meas_total"]) * len(cfg["renormalization"]["beta1_methods"]) * alpha_points,
                     len(cfg["renormalization"]["beta1_methods"]) * alpha_points, settings_name="topo"),
        ]
    for stage in ("observable", "topo"):
        if stage in stages:
            fine, coarse = stages.pop(stage)
            fine_settings = section(cfg, stage, "fine")
            fine = replace(fine, projected_sweeps=_sampling_sweeps(fine_settings),
                           n_meas=int(math.ceil(fine_settings["max_meas_total"] /
                                                fine_settings["chains"])))
            stages[f"{stage}_fine"] = [fine]
            stages[f"{stage}_coarse"] = [coarse]
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
    settings = (section(cfg, workload.settings_name) if workload.settings_name else
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
    settings = (section(cfg, workload.settings_name) if workload.settings_name else
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
    peak_allocated = (torch.cuda.max_memory_allocated(device) / 1024 ** 3
                      if device.type == "cuda" else None)
    peak_reserved = (torch.cuda.max_memory_reserved(device) / 1024 ** 3
                     if device.type == "cuda" else None)
    del sampler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"name": workload.name, "batch_seconds": seconds,
            "peak_gpu_gib": peak_reserved,
            "peak_gpu_allocated_gib": peak_allocated}


@contextmanager
def _cuda_allocator_limit(device: torch.device,
                          fraction: float) -> Iterator[None]:
    """Temporarily enforce the tuning memory limit in the CUDA allocator."""
    if device.type != "cuda":
        yield
        return
    cuda_device = torch.device(
        "cuda", device.index if device.index is not None else torch.cuda.current_device())
    gc.collect()
    torch.cuda.empty_cache()
    previous = torch.cuda.get_per_process_memory_fraction(cuda_device)
    torch.cuda.set_per_process_memory_fraction(
        min(previous, fraction), cuda_device)
    try:
        yield
    finally:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(previous, cuda_device)


def _target_sweeps(workloads: list[Workload], chains: int) -> tuple[list[int], bool]:
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
    return target_sweeps, too_short


def _benchmark_candidate(cfg: dict, workloads: list[Workload], chains: int,
                         *, warmup_sweeps: int, timed_sweeps: int,
                         gpu_memory_fraction: float, max_host_memory_gib: float,
                         progress: SimulationProgress | None = None) -> dict:
    device = resolve_device(cfg["runtime"]["device"])
    dtype = torch_dtype(cfg["runtime"]["dtype"])
    gpu_total = (torch.cuda.get_device_properties(device).total_memory / 1024 ** 3
                 if device.type == "cuda" else None)
    target_sweeps, too_short = _target_sweeps(workloads, chains)
    host_gib = max(_host_gib(x, chains, dtype) for x in workloads)
    preflight_reasons = []
    if host_gib > max_host_memory_gib:
        preflight_reasons.append("host_measurement_memory_limit")
    if too_short:
        preflight_reasons.append("per_chain_series_too_short")
    if preflight_reasons:
        return {"chains": chains, "safe": False,
                "reason": ",".join(preflight_reasons),
                "estimated_host_measurement_gib": host_gib,
                "workloads": []}

    timings = []
    try:
        for workload in workloads:
            if progress is not None:
                progress.extend()
                progress.update(0, chains=chains, workload=workload.name)
            try:
                timing = _time_workload(cfg, workload, chains,
                                        warmup_sweeps, timed_sweeps)
                timings.append(timing)
            finally:
                if progress is not None:
                    progress.update()
            peak = timing["peak_gpu_gib"] or 0.0
            if gpu_total is not None and peak > gpu_total * gpu_memory_fraction:
                return {"chains": chains, "safe": False,
                        "reason": "gpu_memory_limit", "peak_gpu_gib": peak,
                        "estimated_host_measurement_gib": host_gib,
                        "workloads": timings}
    except (torch.OutOfMemoryError, MemoryError):
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return {"chains": chains, "safe": False, "reason": "out_of_memory",
                "estimated_host_measurement_gib": host_gib,
                "workloads": timings}

    projected_seconds = sum(
        timing["batch_seconds"] * sweeps
        for timing, sweeps in zip(timings, target_sweeps))
    total_chain_sweeps = chains * sum(target_sweeps)
    peak_gpu = max((x["peak_gpu_gib"] or 0.0) for x in timings)
    return {
        "chains": chains,
        "chain_sweeps_per_s": total_chain_sweeps / projected_seconds,
        "projected_batch_seconds": projected_seconds,
        "projected_target_seconds": projected_seconds,
        "peak_gpu_gib": peak_gpu if gpu_total is not None else None,
        "estimated_host_measurement_gib": host_gib,
        "safe": True,
        "reason": None,
        "workloads": timings,
    }


def _power_two_seed(configured_chains: int, max_chains: int) -> int:
    if configured_chains < 1 or max_chains < 1:
        raise ValueError("chain counts must be positive")
    exponent = math.floor(math.log2(configured_chains) + 0.5)
    seed = 2 ** exponent
    if seed > max_chains:
        seed = 2 ** int(math.floor(math.log2(max_chains)))
    return max(seed, 1)


def _search_stop_reason(row: dict) -> str:
    reason = str(row.get("reason") or "")
    if "per_chain_series_too_short" in reason:
        return "series_too_short"
    return "memory_limit"


def _log_midpoint(low: int, high: int) -> int | None:
    if low < 1 or high <= low or high / low <= 2:
        return None
    exponent = round((math.log2(low) + math.log2(high)) / 2)
    midpoint = 2 ** exponent
    return midpoint if low < midpoint < high else None


def _adaptive_chain_search(
        seed_chains: int, max_chains: int, evaluate: Callable[[int], dict],
        *, growth_factor: int = SEARCH_GROWTH_FACTOR,
        min_improvement: float = SEARCH_MIN_IMPROVEMENT,
        stall_limit: int = SEARCH_STALL_LIMIT) -> dict:
    """Bracket the fastest safe chain count, then refine on a factor-two grid."""
    rows: dict[int, dict] = {}

    def measure(chains: int) -> dict:
        chains = max(1, min(int(chains), max_chains))
        if chains not in rows:
            rows[chains] = evaluate(chains)
        return rows[chains]

    seed_chains = max(1, min(seed_chains, max_chains))
    initial_seed = seed_chains
    seed_row = measure(seed_chains)
    while not seed_row.get("safe") and seed_chains > 1:
        seed_chains = max(1, seed_chains // growth_factor)
        seed_row = measure(seed_chains)
    if not seed_row.get("safe"):
        ordered = [rows[key] for key in sorted(rows)]
        return {"recommended_chains": None, "candidates": ordered,
                "search": {"seed_chains": initial_seed,
                           "growth_factor": growth_factor,
                           "min_improvement_fraction": min_improvement,
                           "evaluated_candidates": len(ordered),
                           "stop_reason": "no_safe_candidate"}}

    lower = max(1, seed_chains // growth_factor)
    upper = min(max_chains, seed_chains * growth_factor)
    measure(lower)
    measure(upper)

    def safe_time(row: dict) -> float:
        if not row.get("safe"):
            return float("inf")
        return float(row["projected_target_seconds"])

    initial = [value for value in {lower, seed_chains, upper}
               if rows[value].get("safe")]
    best_chains = min(initial, key=lambda value: safe_time(rows[value]))
    direction = -1 if best_chains == lower < seed_chains else (
        1 if best_chains == upper > seed_chains else 0)
    stop_reason = "converged"
    if direction == 0 and not rows[upper].get("safe"):
        stop_reason = _search_stop_reason(rows[upper])

    if direction:
        edge = best_chains
        stalled = 0
        while True:
            if direction > 0:
                if edge >= max_chains:
                    stop_reason = "max_chains_reached"
                    break
                candidate = min(max_chains, edge * growth_factor)
            else:
                if edge <= 1:
                    break
                candidate = max(1, edge // growth_factor)
            old_best = min(safe_time(item) for item in rows.values()
                           if item.get("safe"))
            row = measure(candidate)
            if not row.get("safe"):
                stop_reason = _search_stop_reason(row)
                break
            new_time = safe_time(row)
            if new_time <= old_best * (1.0 - min_improvement):
                stalled = 0
            else:
                stalled += 1
            edge = candidate
            if stalled >= stall_limit:
                break

    safe_values = [chains for chains, row in rows.items() if row.get("safe")]
    if safe_values:
        coarse_best = min(safe_values, key=lambda value: safe_time(rows[value]))
        below = max((value for value in rows if value < coarse_best), default=None)
        above = min((value for value in rows if value > coarse_best), default=None)
        for bound in (below, above):
            if bound is None:
                continue
            midpoint = _log_midpoint(min(bound, coarse_best), max(bound, coarse_best))
            if midpoint is not None:
                measure(midpoint)

    ordered = [rows[key] for key in sorted(rows)]
    return {
        "recommended_chains": recommend_chains(ordered),
        "candidates": ordered,
        "search": {"seed_chains": initial_seed,
                   "growth_factor": growth_factor,
                   "min_improvement_fraction": min_improvement,
                   "evaluated_candidates": len(ordered),
                   "stop_reason": stop_reason},
    }


def _benchmark_stage(cfg: dict, workloads: list[Workload], configured_chains: int,
                     *, max_chains: int, warmup_sweeps: int, timed_sweeps: int,
                     gpu_memory_fraction: float, max_host_memory_gib: float,
                     progress: SimulationProgress | None = None) -> dict:
    seed_chains = _power_two_seed(configured_chains, max_chains)
    device = resolve_device(cfg["runtime"]["device"])
    gpu_limit_gib = (
        torch.cuda.get_device_properties(device).total_memory / 1024 ** 3
        * gpu_memory_fraction if device.type == "cuda" else None)
    measured_rows: list[dict] = []

    def evaluate(chains: int) -> dict:
        smaller = [row for row in measured_rows
                   if row.get("safe") and row["chains"] < chains
                   and row.get("peak_gpu_gib") is not None]
        if smaller and gpu_limit_gib is not None:
            prior = max(smaller, key=lambda row: row["chains"])
            predicted = (float(prior["peak_gpu_gib"]) * chains
                         / int(prior["chains"]))
            if predicted > gpu_limit_gib:
                row = {"chains": chains, "safe": False,
                       "reason": "predicted_gpu_memory_limit",
                       "estimated_peak_gpu_gib": predicted,
                       "peak_gpu_gib": None, "workloads": []}
                measured_rows.append(row)
                return row
        row = _benchmark_candidate(
            cfg, workloads, chains, warmup_sweeps=warmup_sweeps,
            timed_sweeps=timed_sweeps,
            gpu_memory_fraction=gpu_memory_fraction,
            max_host_memory_gib=max_host_memory_gib, progress=progress)
        measured_rows.append(row)
        return row

    return _adaptive_chain_search(seed_chains, max_chains, evaluate)


def _tuning_summary(stage: str, result: dict) -> str:
    rows = result["candidates"]
    safe = [row for row in rows if row.get("safe")]
    tested_range = (f"{rows[0]['chains']}..{rows[-1]['chains']}" if rows else "none")
    recommended = result.get("recommended_chains")
    selected = next((row for row in safe if row["chains"] == recommended), None)
    seconds = (f"{selected['projected_target_seconds']:.3g}s"
               if selected is not None else "n/a")
    peaks = [float(row["peak_gpu_gib"]) for row in rows
             if row.get("peak_gpu_gib") is not None]
    peak = f"{max(peaks):.3f}GiB" if peaks else "n/a"
    stop = result.get("search", {}).get("stop_reason", "unknown")
    return (f"[cpn-renorm] tuned {stage}: tested={len(rows)} "
            f"range={tested_range} safe={len(safe)} recommended={recommended} "
            f"projected={seconds} gpu_peak={peak} stop={stop}")




def _toml_suggestion(report: dict) -> str:
    lines = ["[chains]"]
    for stage in DEFAULTS["chains"]:
        if stage not in report["stages"]:
            continue
        value = report["stages"][stage]["recommended_chains"]
        if value is not None:
            lines.append(f"{stage} = {value}")
    return "\n".join(lines)


def tune_stage(cfg: dict, stage: str, workloads: list[Workload], *,
               label: str | None = None, **limits) -> int:
    """Benchmark a stage and record the batch width actually selected."""
    options = {"max_chains": 65536, "warmup_sweeps": 5, "timed_sweeps": 10,
               "gpu_memory_fraction": 0.8, "max_host_memory_gib": 8.0,
               **cfg.get("_tuning_limits", {}), **limits}
    if options["max_chains"] < 1 or options["timed_sweeps"] < 1 or options["warmup_sweeps"] < 0:
        raise ValueError("benchmark chain/sweep limits are invalid")
    if not 0 < options["gpu_memory_fraction"] <= 1 or options["max_host_memory_gib"] <= 0:
        raise ValueError("benchmark memory limits are invalid")
    device = resolve_device(cfg["runtime"]["device"])
    with _cuda_allocator_limit(device, options["gpu_memory_fraction"]):
        with SimulationProgress(f"tune {label or stage}") as progress:
            progress.phase(stage, 0, unit="test")
            result = _benchmark_stage(cfg, workloads, int(cfg["chains"][stage]),
                                      progress=progress, **options)
    name = label or f"{cfg.get('_tuning_prefix', '')}{stage}"
    result["lattices"] = [{"name": w.name, "Lx": w.Lx, "Ly": w.Ly,
                            "kind": w.kind, "mod": w.couplings.mod} for w in workloads]
    result["limits"] = options
    cfg.setdefault("_tuning_records", {})[name] = result
    value = result["recommended_chains"]
    print(f"[cpn-renorm] {name}: chains={value}; lattices={result['lattices']}",
          flush=True)
    if value is None:
        raise RuntimeError(f"no safe chains candidate for {name}")
    cfg["chains"][stage] = int(value)
    return int(value)


def tune_pilot(cfg: dict, L: int) -> int:
    settings = section(cfg, "pilot")
    couplings, kind = _fine_action(cfg)
    workload = Workload("pbc", L, L, True, kind, couplings,
                        _sampling_sweeps(settings),
                        n_meas=int(math.ceil(settings["max_meas_total"] / settings["chains"])),
                        stores_correlations=True,
                        measurement_total=int(settings["max_meas_total"]),
                        settings_name="pilot")
    return tune_stage(cfg, "pilot", [workload],
                      label=f"{cfg.get('_tuning_prefix', '')}pilot_L{L}",
                      **cfg.get("_tuning_limits", {}))


def run_chain_tuning(cfg: dict, output_dir: str | Path, **limits) -> dict:
    """Tune independent lattice widths for every requested mod."""
    run_dir = workspace_dir(cfg, output_dir)
    reports = {}
    shared_pilot = None
    shared_chains = {}
    records = {}
    for mod in cfg["model"]["mod"]:
        branch = mod_config(cfg, mod)
        # The pilot must be tuned before it is sampled, including size growth.
        branch["runtime"]["auto_chains"] = True
        branch["_tuning_prefix"] = f"mod{mod}/"
        branch["_tuning_records"] = records
        branch["_tuning_limits"] = limits
        independent = float(cfg["model"]["beta1"]) != 0
        if independent or shared_pilot is None:
            pilot_cfg = mod_config(branch, mod if independent else 0)
            pilot_cfg["_tuning_records"] = records
            pilot_cfg["_tuning_prefix"] = f"mod{mod}/" if independent else "common/"
            pilot_dir = run_dir / "common" / f"mod{mod}" if independent else run_dir
            pilot = run_pilot(pilot_cfg, pilot_dir)
            branch["chains"]["pilot"] = pilot_cfg["chains"]["pilot"]
            if not independent:
                shared_pilot = pilot
                shared_chains["pilot"] = branch["chains"]["pilot"]
        else:
            pilot = shared_pilot
            branch["chains"].update(shared_chains)
        for stage, workloads in _workloads(branch, pilot).items():
            if stage == "pilot":
                continue
            if stage == "one_plaq" and (branch["renormalization"]["type"] == ["U"]
                                        and abs(float(branch["model"]["alpha"])) < 1e-12):
                continue
            if not independent and stage in shared_chains:
                continue
            prefix = "common/" if not independent and stage in ("two_plaq", "one_plaq") else f"mod{mod}/"
            tune_stage(branch, stage, workloads, label=f"{prefix}{stage}", **limits)
            if not independent and stage in ("two_plaq", "one_plaq"):
                shared_chains[stage] = branch["chains"][stage]
        reports[f"mod{mod}"] = {"chains": branch["chains"],
                                "geometry": pilot["geometry"]}
        reports[f"mod{mod}"]["toml"] = "[chains]\n" + "\n".join(
            f"{key} = {value}" for key, value in branch["chains"].items())
    report = {"device": str(resolve_device(cfg["runtime"]["device"])),
              "dtype": cfg["runtime"]["dtype"], "mods": reports, "stages": records,
              "toml": "\n\n".join(f"# {name}\n{item['toml']}"
                                     for name, item in reports.items())}
    atomic_json(run_dir / "chain_tuning.json", report)
    return report
