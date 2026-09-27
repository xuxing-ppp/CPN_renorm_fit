import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from cpn_renorm.config import fingerprint
from cpn_renorm.patches import (
    PATCH_CACHE_SCHEMA,
    _adapt_inner,
    _adaptive_samples,
    _aggregate_batches,
    _batch_key,
    _boundary_data,
    _extract_rectangle,
    _install_rectangle,
    _load_or_generate_batch,
    _restore_patch,
    _two_plaq_connections,
    patch_result_key,
    patch_batch_count,
)
from cpn_renorm.sampler import BatchedHMCSampler, Couplings


class PatchTests(unittest.TestCase):
    def test_two_plaq_middle_z_connection_uses_forward_legacy_orientation(self):
        sampler = BatchedHMCSampler(
            chains=1, Lx=5, Ly=3, N=2,
            couplings=Couplings(1, 0, 0), periodic=False, seed=2)
        delta = 0.2
        phases = torch.arange(3, dtype=sampler.dtype) * delta
        line = torch.zeros((3, 2), dtype=sampler.cdtype)
        line[:, 0] = torch.exp(1j * phases)
        sampler.z[:, 2, :] = line
        sampler.a[:, 2, :2, 1] = torch.tensor([0.1, 0.3], dtype=sampler.dtype)

        connections = _two_plaq_connections(sampler, L=2)

        self.assertAlmostEqual(float(connections[0, 0]), 0.4, places=12)
        self.assertAlmostEqual(float(connections[0, 1]), 2 * delta, places=12)

    def test_two_plaq_boundary_data_matches_legacy_extract_param(self):
        generator = torch.Generator().manual_seed(7)
        real = torch.randn((2, 12, 2), generator=generator, dtype=torch.float64)
        imag = torch.randn((2, 12, 2), generator=generator, dtype=torch.float64)
        zloop = torch.complex(real, imag)
        zloop /= torch.linalg.vector_norm(zloop, dim=-1, keepdim=True)
        aloop = torch.linspace(-0.7, 0.8, 12, dtype=torch.float64).repeat(2, 1)

        au, az, corners = _boundary_data(zloop, aloop, L=2, segments=6)

        z_np = zloop.numpy()
        expected_az = np.empty((2, 6))
        for chain in range(2):
            for segment in range(6):
                expected_az[chain, segment] = sum(
                    np.angle(np.vdot(z_np[chain, segment * 2 + offset],
                                     z_np[chain, (segment * 2 + offset + 1) % 12]))
                    for offset in range(2))
        expected_az = (expected_az + np.pi) % (2 * np.pi) - np.pi
        expected_au = aloop.reshape(2, 6, 2).sum(-1).numpy()
        expected_au = (expected_au + np.pi) % (2 * np.pi) - np.pi
        np.testing.assert_allclose(au, expected_au, rtol=0, atol=1e-12)
        np.testing.assert_allclose(az, expected_az, rtol=0, atol=1e-12)
        np.testing.assert_allclose(corners, z_np[:, ::2], rtol=0, atol=0)

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

    def test_patch_cache_ignores_scan_but_tracks_stage_settings(self):
        from copy import deepcopy
        from cpn_renorm.config import load_config

        cfg = load_config(Path(__file__).parents[1] / "configs" / "example.toml")
        geometry = {"padding": 5, "L_fine": 28, "L_coarse": 7, "factor": 4}
        original = _batch_key(cfg, geometry, "two_plaq", 0)
        scan_changed = deepcopy(cfg)
        scan_changed["scan"]["beta1"]["min"] = -7.5
        self.assertEqual(original,
                         _batch_key(scan_changed, geometry, "two_plaq", 0))
        settings_changed = deepcopy(cfg)
        settings_changed["steps"]["two_plaq"]["max_warmup"] += 1
        self.assertNotEqual(original,
                            _batch_key(settings_changed, geometry, "two_plaq", 0))

    def test_patch_fit_key_reprocesses_p0_without_invalidating_batches(self):
        from copy import deepcopy
        from cpn_renorm.config import load_config

        cfg = load_config(Path(__file__).parents[1] / "configs" / "example.toml")
        geometry = {"padding": 5, "L_fine": 28, "L_coarse": 7, "factor": 4}
        changed = deepcopy(cfg)
        changed["steps"]["two_plaq"]["p0"] = [1.5, 0.2]
        self.assertEqual(_batch_key(cfg, geometry, "two_plaq", 0),
                         _batch_key(changed, geometry, "two_plaq", 0))
        self.assertNotEqual(patch_result_key(cfg, geometry, "two_plaq"),
                            patch_result_key(changed, geometry, "two_plaq"))

    def test_two_plaq_revision_invalidates_only_two_plaq_cache(self):
        from cpn_renorm.config import load_config

        cfg = load_config(Path(__file__).parents[1] / "configs" / "example.toml")
        geometry = {"padding": 5, "L_fine": 28, "L_coarse": 7, "factor": 4}
        two_before = _batch_key(cfg, geometry, "two_plaq", 0)
        one_before = _batch_key(cfg, geometry, "one_plaq", 0)
        with patch("cpn_renorm.patches.TWO_PLAQ_ALGORITHM_REVISION", 3):
            two_after = _batch_key(cfg, geometry, "two_plaq", 0)
            one_after = _batch_key(cfg, geometry, "one_plaq", 0)
        self.assertNotEqual(two_before, two_after)
        self.assertEqual(one_before, one_after)

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
