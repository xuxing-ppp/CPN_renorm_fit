from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from cpn_renorm.chain_tuning import _benchmark_candidate, _host_gib, _workloads
from cpn_renorm.config import load_config, mod_config
from cpn_renorm.memory import host_budget_gib, patch_peak_gib
from cpn_renorm.observables import autocorr_time
from cpn_renorm.patches import _adaptive_samples, _check_patch_memory


class PatchMemoryTests(unittest.TestCase):
    def config(self):
        cfg = mod_config(load_config(Path(__file__).parents[1] /
                                     "configs/example.toml"), 0)
        cfg["runtime"]["device"] = "cpu"
        return cfg

    def test_sampling_reuses_buffer_and_matches_full_history_statistics(self):
        rng = np.random.default_rng(203)
        history = rng.normal(size=(40, 7, 2))
        history[1:] += .3 * history[:-1]

        class Sampler:
            chains = 7
            step = -1

            def sweep(self, **kwargs):
                self.step += 1

        for target, expected in ((0., 20), (1e9, 40)):
            with self.subTest(target=target):
                shapes = []

                def transform(values):
                    shapes.append(values.shape)
                    return np.concatenate((np.sin(values), np.cos(values)), axis=-1)

                settings = dict(min_meas_per_boundary=20, max_meas_per_boundary=40,
                                target_ess_per_boundary=target, s_step=.5, s_updates=1)
                data, stats = _adaptive_samples(
                    Sampler(), settings, lambda s: history[s.step], transform)
                np.testing.assert_array_equal(data, history[:expected])
                self.assertIsNotNone(data.base)
                self.assertEqual(data.base.shape, history.shape)
                self.assertTrue(all(shape[1] == 1 for shape in shapes))
                reference = np.concatenate((np.sin(history[:expected]),
                                            np.cos(history[:expected])), axis=-1)
                tau = np.array([[autocorr_time(reference[:, c, o])
                                 for o in range(4)] for c in range(7)])
                np.testing.assert_array_equal(stats["tau"], tau)
                np.testing.assert_allclose(stats["ess"], expected / (2 * tau))

    def test_budget_leaves_headroom_and_respects_explicit_limit(self):
        with patch("cpn_renorm.memory.available_host_bytes", return_value=6 * 1024 ** 3):
            self.assertEqual(host_budget_gib(), 3.)
            self.assertEqual(host_budget_gib(1.), 1.)

    def test_patch_estimate_includes_maximum_history_for_both_dtypes(self):
        cfg = self.config()
        geometry = dict(factor=2, padding=3, L_fine=20, L_coarse=10)
        workloads = _workloads(cfg, dict(geometry=geometry,
                                       attempts=[dict(L=32, valid=True)]))
        for stage, observables in (("one_plaq", 3), ("two_plaq", 2)):
            inner = workloads[stage][1]
            raw = inner.n_meas * 4096 * observables * 8 / 1024 ** 3
            estimate = _host_gib(inner, 4096, torch.float64)
            self.assertGreater(estimate, raw)
            self.assertEqual(estimate, _host_gib(inner, 4096, torch.float32))
            with patch("cpn_renorm.chain_tuning.host_budget_gib", return_value=.001), patch(
                    "cpn_renorm.chain_tuning._time_workload") as timing:
                row = _benchmark_candidate(cfg, workloads[stage], 4096,
                    warmup_sweeps=1, timed_sweeps=1, gpu_memory_fraction=.8,
                    max_host_memory_gib=8.)
            self.assertFalse(row["safe"])
            self.assertIn("host_measurement_memory_limit", row["reason"])
            timing.assert_not_called()

    def test_runtime_rejects_restored_or_manual_width_before_allocation(self):
        cfg = self.config()
        geometry = dict(padding=3)
        for automatic in (False, True):
            cfg["runtime"]["auto_chains"] = automatic
            cfg["chains"]["two_plaq"] = 65536
            with patch("cpn_renorm.patches.host_budget_gib", return_value=8.):
                with self.assertRaisesRegex(MemoryError, "restored chain counts"):
                    _check_patch_memory(cfg, geometry, "two_plaq")
                cfg["chains"]["two_plaq"] = 128
                _check_patch_memory(cfg, geometry, "two_plaq")

    def test_large_history_exceeds_budget_without_duplicate_arrays(self):
        self.assertGreater(patch_peak_gib("two_plaq", 65536, 32768, 2, 5000), 32.)


if __name__ == "__main__":
    unittest.main()
