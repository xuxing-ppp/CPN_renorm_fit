"""Opt-in 1plaq ESS investigation; does not write pipeline caches or configs.

python tests/diagnose_patch_ess.py --output-dir /tmp/cpn-ess-diagnostic
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from cpn_renorm.config import load_config, mod_config, section
from cpn_renorm.observables import autocorr_time
from cpn_renorm.patches import (_adapt_inner, _fine_couplings, _make_sampler,
                                _obc_topology, _outer_boundary)
from cpn_renorm.progress import SimulationProgress


def describe(values: np.ndarray) -> dict:
    """Compare raw ESS to sector ESS, recording when neither proves mixing."""
    output = {}
    n = len(values)
    for col, name in enumerate(("Q_U", "Q_z", "Q_s")):
        raw = values[..., col]
        # Frozen boundaries fix the fractional charge. Differences between
        # configurations should therefore be integers; verify before rounding.
        delta = raw - raw[0]
        sectors = np.rint(delta)
        residual = float(np.max(np.abs(delta - sectors)))
        if residual > 1e-8:
            raise AssertionError(f"{name}: noninteger charge differences {residual}")
        raw_ess = np.array([n / (2 * autocorr_time(raw[:, c]))
                            for c in range(raw.shape[1])])
        centered_ess = np.array([n / (2 * autocorr_time(delta[:, c]))
                                 for c in range(raw.shape[1])])
        sector_ess = np.array([n / (2 * autocorr_time(sectors[:, c]))
                               for c in range(raw.shape[1])])
        transitions = np.count_nonzero(np.diff(sectors, axis=0), axis=0)
        spans = np.ptp(raw, axis=0)
        worst = int(np.argmin(raw_ess))
        output[name] = {
            "raw_ess_min_median_max": np.quantile(raw_ess, [0, .5, 1]).tolist(),
            "sector_ess_min_median_max": np.quantile(sector_ess, [0, .5, 1]).tolist(),
            "centered_ess_min_median_max": np.quantile(centered_ess, [0, .5, 1]).tolist(),
            "chains_below_target_512_raw": int(np.sum(raw_ess < 512)),
            "chains_below_target_512_sector": int(np.sum(sector_ess < 512)),
            "chains_below_target_512_centered": int(np.sum(centered_ess < 512)),
            "low_raw_ess_but_centered_passes":
                int(np.sum((raw_ess < 512) & (centered_ess >= 512))),
            "no_sector_transitions": int(np.sum(transitions == 0)),
            "fewer_than_10_transitions": int(np.sum(transitions < 10)),
            "false_low_raw_ess_below_5_without_transitions":
                int(np.sum((raw_ess < 5) & (transitions == 0))),
            "max_quantization_residual": residual,
            "worst_raw_chain": worst,
            "worst_raw_chain_ess": float(raw_ess[worst]),
            "worst_raw_chain_span": float(spans[worst]),
            "worst_raw_chain_transitions": int(transitions[worst]),
            "raw_ess": raw_ess.tolist(), "sector_ess": sector_ess.tolist(),
            "centered_ess": centered_ess.tolist(),
            "transitions": transitions.tolist(),
            "mean_sector_first_half": sectors[:n // 2].mean(0).tolist(),
            "mean_sector_second_half": sectors[n // 2:].mean(0).tolist(),
            "caution": "ESS assigned to a constant sector does not establish ergodicity",
        }
    return output


def synthetic() -> dict:
    n = 5504
    rng = np.random.default_rng(937)
    cases = {"constant_0.1": np.full(n, .1),
             "constant_0.3": np.full(n, .3),
             "one_jump": np.r_[np.zeros(n // 2), np.ones(n - n // 2)]}
    cases["roundoff_only"] = .1 + rng.normal(size=n) * 1e-16
    ar = np.zeros(n)
    noise = rng.normal(size=n)
    for i in range(1, n):
        ar[i] = .8 * ar[i - 1] + noise[i]
    cases["ar1_rho_0.8"] = ar
    result = {}
    for name, x in cases.items():
        tau = autocorr_time(x)
        # Subtracting a reference first removes a large common offset before
        # estimating the small variation; it does not alter physical differences.
        stable_tau = autocorr_time(x - x[0])
        result[name] = dict(variance=float(np.var(x)), span=float(np.ptp(x)),
                            raw_ess=n / (2 * tau),
                            reference_centered_ess=n / (2 * stable_tau))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(
        "configs/N2_L2/N2_L2_b1.0-L2_mod0_corrG_topo.toml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chains", type=int, default=128)
    parser.add_argument("--sweeps", type=int, default=5504)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--padding", type=int, default=3)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--variants", nargs="+", choices=("baseline", "s12"),
                        default=["baseline", "s12"])
    args = parser.parse_args()
    torch.set_num_threads(1)
    cfg = mod_config(load_config(args.config), 0)
    cfg["runtime"].update(device=args.device, auto_chains=False)
    if args.seed is not None:
        cfg["runtime"]["seed"] = args.seed
    cfg["chains"]["one_plaq"] = args.chains
    settings = section(cfg, "one_plaq")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = dict(config=str(args.config.resolve()), model=cfg["model"],
                  seed=cfg["runtime"]["seed"], chains=args.chains, sweeps=args.sweeps,
                  dtype=cfg["runtime"]["dtype"], device=args.device,
                  padding=args.padding, synthetic=synthetic(), variants={})
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2))
    print("synthetic", json.dumps(report["synthetic"]), flush=True)
    couplings, kind = _fine_couplings(cfg)
    L = int(cfg["renormalization"]["factor"])
    with SimulationProgress("ESS diagnostic", enabled=False) as progress:
        patch, outer_warmup = _outer_boundary(
            cfg, settings, L=L, pad=args.padding, width=L, height=L,
            couplings=couplings, kind=kind, seed_offset=30000, progress=progress)
        sampler = _make_sampler(cfg, settings, chains=args.chains, Lx=L + 1, Ly=L + 1,
                                couplings=couplings, kind=kind, periodic=False,
                                seed_offset=30001)
        _adapt_inner(sampler, patch, settings, progress=progress)
    initial = deepcopy(sampler.state_dict())
    report.update(outer_warmup=outer_warmup, inner_epsilon=sampler.epsilon,
                  n_leapfrog=sampler.n_leapfrog)
    for variant in args.variants:
        sampler.load_state_dict(deepcopy(initial))
        history = np.empty((args.sweeps, args.chains, 3), dtype=float)
        checkpoints = {}
        started = time.perf_counter()
        for index in range(args.sweeps):
            sampler.sweep(s_step=float(settings["s_step"]),
                          s_updates=12 if variant == "s12" else int(settings["s_updates"]))
            history[index] = torch.stack(_obc_topology(sampler), -1).detach().cpu().numpy()
            actual = index + 1
            if actual in (512, 2048, 5504, args.sweeps):
                checkpoints[str(actual)] = describe(history[:actual])
                compact = {k: {name: value for name, value in v.items()
                               if name in ("raw_ess_min_median_max", "no_sector_transitions",
                                           "worst_raw_chain_transitions",
                                           "false_low_raw_ess_below_5_without_transitions")}
                           for k, v in checkpoints[str(actual)].items()}
                print(variant, actual, json.dumps(compact), flush=True)
        seconds = time.perf_counter() - started
        np.savez_compressed(args.output_dir / f"{variant}_history.npz", history=history)
        report["variants"][variant] = dict(
            seconds=seconds, acceptance=sampler.accept_rate.cpu().tolist(),
            s_acceptance=sampler.metro_accept_rate.cpu().tolist(), checkpoints=checkpoints)
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2))
        print(variant, "finished", seconds, flush=True)


if __name__ == "__main__":
    main()
