import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from cpn_renorm.config import fingerprint
from cpn_renorm.patches import (
    PATCH_CACHE_SCHEMA,
    _adapt_inner,
    _adaptive_samples,
    _aggregate_batches,
    _batch_key,
    _extract_rectangle,
    _install_rectangle,
    _load_or_generate_batch,
    _restore_patch,
    patch_batch_count,
)
from cpn_renorm.sampler import BatchedHMCSampler, Couplings


class PatchTests(unittest.TestCase):
    def test_fit_times_round_up_to_complete_batches(self):
        self.assertEqual(patch_batch_count(5000, 32), 157)
        self.assertEqual(patch_batch_count(2000, 32), 63)
        self.assertEqual(patch_batch_count(3, 8), 1)

    def test_boundary_round_trip(self):
        outer = BatchedHMCSampler(chains=2, Lx=9, Ly=7, N=2,
            couplings=Couplings(1, 0, 0), periodic=False, seed=4)
        zloop, aloop = _extract_rectangle(outer, width=4, height=2, pad=1)
        inner = BatchedHMCSampler(chains=2, Lx=5, Ly=3, N=2,
            couplings=Couplings(1, 0, 0), periodic=False, seed=5)
        _install_rectangle(inner, zloop, aloop, 4, 2)
        got_z, got_a = _extract_rectangle(inner, width=4, height=2, pad=0)
        self.assertTrue(torch.allclose(got_z, zloop))
        self.assertTrue(torch.allclose(torch.exp(1j * got_a), torch.exp(1j * aloop)))

    def test_complete_patch_round_trip_for_patch_geometries_and_boundary_types(self):
        cases = ((4, 2, False, 0), (4, 2, True, 2),
                 (3, 3, False, 1), (3, 3, True, 2))
        for width, height, periodic, pad in cases:
            with self.subTest(width=width, height=height,
                              periodic=periodic, pad=pad):
                add = 0 if periodic else 1
                outer = BatchedHMCSampler(
                    chains=2, Lx=width + 2 * pad + add,
                    Ly=height + 2 * pad + add, N=2,
                    couplings=Couplings(0.7, 0.1, 0.4), kind="full",
                    periodic=periodic, seed=11)
                outer.s.random_(-2, 3, generator=outer.generator)
                patch = _extract_rectangle(outer, width, height, pad)
                inner = BatchedHMCSampler(
                    chains=2, Lx=width + 1, Ly=height + 1, N=2,
                    couplings=outer.c, kind="full", periodic=False, seed=12)
                _install_rectangle(inner, patch)
                _restore_patch(inner, patch)
                self.assertTrue(torch.equal(inner.z, patch.z))
                self.assertTrue(torch.equal(inner.a, patch.a))
                self.assertTrue(torch.equal(inner.s[:, :-1, :-1], patch.s))

    def test_patch_action_change_matches_outer_local_action_change(self):
        for periodic, pad in ((False, 1), (True, 2)):
            with self.subTest(periodic=periodic):
                width, height = 4, 3
                add = 0 if periodic else 1
                couplings = Couplings(0.8, 0.2, 0.5, 0.1)
                outer = BatchedHMCSampler(
                    chains=1, Lx=width + 2 * pad + add,
                    Ly=height + 2 * pad + add, N=2, couplings=couplings,
                    kind="full", periodic=periodic, seed=21,
                    trajectory_length=0.1)
                outer.s.random_(-1, 2, generator=outer.generator)
                patch = _extract_rectangle(outer, width, height, pad)
                inner = BatchedHMCSampler(
                    chains=1, Lx=width + 1, Ly=height + 1, N=2,
                    couplings=couplings, kind="full", periodic=False,
                    seed=22, trajectory_length=0.1)
                _install_rectangle(inner, patch)
                _restore_patch(inner, patch)
                outer_before = outer.action().clone()
                inner_before = inner.action().clone()

                phase = torch.exp(torch.tensor(0.23j, dtype=outer.cdtype))
                outer.z[:, pad + 1, pad + 1] *= phase
                inner.z[:, 1, 1] *= phase
                outer.a[:, pad + 1, pad + 1, 0] += 0.17
                inner.a[:, 1, 1, 0] += 0.17
                outer.s[:, pad + 1, pad + 1] += 1
                inner.s[:, 1, 1] += 1
                outer_delta = outer.action() - outer_before
                inner_delta = inner.action() - inner_before
                self.assertTrue(torch.allclose(outer_delta, inner_delta,
                                               rtol=1e-11, atol=1e-11))

    def test_inner_adaptation_restores_crop_and_resets_diagnostics(self):
        width, height = 3, 2
        couplings = Couplings(0.4, 0.0, 0.2)
        outer = BatchedHMCSampler(
            chains=1, Lx=width + 3, Ly=height + 3, N=2,
            couplings=couplings, kind="full", periodic=False, seed=31,
            epsilon=0.04, trajectory_length=0.08)
        outer.s.random_(-1, 2, generator=outer.generator)
        patch = _extract_rectangle(outer, width, height, pad=1)
        inner = BatchedHMCSampler(
            chains=1, Lx=width + 1, Ly=height + 1, N=2,
            couplings=couplings, kind="full", periodic=False, seed=32,
            epsilon=0.04, trajectory_length=0.08)
        settings = {"adapt_steps": 2, "target_accept": 0.75,
                    "s_step": 0.5, "s_updates": 1}
        result = _adapt_inner(inner, patch, settings)
        self.assertEqual(result["sweeps"], 0)
        self.assertEqual(result["stop_reason"], "copied_equilibrium")
        self.assertEqual(result["initialization"], "copied_equilibrium")
        self.assertEqual(result["adapt_steps"], 2)
        self.assertEqual(result["epsilon"], inner.epsilon)
        self.assertTrue(torch.equal(inner.z, patch.z))
        self.assertTrue(torch.equal(inner.a, patch.a))
        self.assertTrue(torch.equal(inner.s[:, :-1, :-1], patch.s))
        self.assertEqual(int(inner.attempted.sum()), 0)
        self.assertEqual(int(inner.metro_attempted.sum()), 0)
        boundary_z = inner.z[inner.frozen_z].clone()
        boundary_a = inner.a[inner.frozen_a].clone()
        inner.sweep(s_step=0.5, s_updates=1)
        self.assertTrue(torch.allclose(inner.z[inner.frozen_z], boundary_z,
                                       rtol=1e-14, atol=1e-14))
        self.assertTrue(torch.allclose(inner.a[inner.frozen_a], boundary_a,
                                       rtol=0, atol=0))

    def test_adaptive_sampling_uses_per_boundary_counts_and_ess(self):
        class RecordingProgress:
            def __init__(self):
                self.phases = []
                self.updates = []
            def phase(self, name, total):
                self.phases.append((name, total))
            def update(self, count=1, **metrics):
                self.updates.append((count, metrics))

        class FakeSampler:
            chains = 4
            def sweep(self, **_kwargs):
                pass

        settings = {"min_meas_per_boundary": 20, "max_meas_per_boundary": 40,
                    "target_ess_per_boundary": 5, "s_step": 0.5, "s_updates": 1}
        sampler = FakeSampler()
        progress = RecordingProgress()
        observe = lambda _item: np.ones((sampler.chains, 1))
        data, sampling = _adaptive_samples(sampler, settings, observe,
                                           progress=progress)
        self.assertEqual(len(data), 20)
        self.assertEqual(sampling["actual"], 20)
        self.assertEqual(sampling["ess"].shape, (4, 1))
        self.assertEqual(float(sampling["ess"][0, 0]), 20.0)
        self.assertEqual(sampling["stop_reason"], "target_ess_per_boundary")
        self.assertEqual(progress.phases, [("sampling", 40)])
        ess_updates = [metrics["ess"] for _, metrics in progress.updates
                       if "ess" in metrics]
        self.assertEqual(ess_updates, ["20.0/5"])

    def test_batch_cache_reuses_matching_fingerprint(self):
        cfg = {"runtime": {"seed": 1}}
        geometry = {"padding": 5}
        calls = []
        def generate(_cfg, _geometry, index):
            calls.append(index)
            return {"value": np.array([index])}
        with tempfile.TemporaryDirectory(dir=Path(__file__).parents[1]) as folder:
            first = _load_or_generate_batch(cfg, geometry, stage="two_plaq",
                                            batch_index=2, cache_dir=Path(folder),
                                            generate=generate)
            second = _load_or_generate_batch(cfg, geometry, stage="two_plaq",
                                             batch_index=2, cache_dir=Path(folder),
                                             generate=generate)
        self.assertEqual(calls, [2])
        np.testing.assert_array_equal(first["value"], second["value"])

    def test_batch_fingerprint_ignores_runtime_metadata(self):
        base = {"runtime": {"seed": 1}, "_config_path": "first.toml", "_force": False}
        changed = {**base, "_config_path": "elsewhere.toml", "_force": True}
        self.assertEqual(_batch_key(base, {"padding": 5}, "two_plaq", 0),
                         _batch_key(changed, {"padding": 5}, "two_plaq", 0))
        old_key = fingerprint({"schema": PATCH_CACHE_SCHEMA - 1,
                               "stage": "two_plaq", "batch": 0,
                               "config": {"runtime": {"seed": 1}},
                               "geometry": {"padding": 5}})
        self.assertNotEqual(_batch_key(base, {"padding": 5}, "two_plaq", 0), old_key)

    def test_batch_aggregation_reports_requested_and_effective_counts(self):
        batch = {"X_U": np.zeros((2, 2)), "freq_U": np.zeros(2),
                 "accept_rate": np.array([0.8, 0.9]), "epsilon": 0.1,
                 "outer_warmup_sweeps": 10,
                 "outer_warmup_stop_reason": "tau_stable",
                 "outer_warmup_tau": np.array([1.0]),
                 "warmup_sweeps": 10, "warmup_stop_reason": "tau_stable",
                 "inner_initialization": "copied_equilibrium", "inner_adapt_steps": 200,
                 "warmup_tau": np.array([1.0]), "sampling_actual": 20,
                 "sampling_per_chain": 20, "sampling_stop_reason": "target_ess_per_boundary",
                 "sampling_tau": np.array([1.0]), "sampling_ess": np.array([10.0])}
        settings = {"chains": 2, "fit_times": 3, "min_meas_per_boundary": 20,
                    "max_meas_per_boundary": 40, "target_ess_per_boundary": 10}
        out = _aggregate_batches([batch, batch], settings, ("X_U", "freq_U"))
        self.assertEqual(out["requested_fit_times"], 3)
        self.assertEqual(out["effective_fit_times"], 4)
        self.assertEqual(out["sampling_total_actual"], 80)
