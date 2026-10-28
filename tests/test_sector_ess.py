import unittest

import numpy as np

from cpn_renorm.observables import autocorr_time
from cpn_renorm.patches import (_adaptive_samples, _aggregate_batches,
                                _batch_summary, _per_boundary_stats)


class SectorEssTests(unittest.TestCase):
    def test_constant_and_translation_invariant_autocorrelation(self):
        for value in (.1, .2, .3, 1 / np.pi):
            self.assertEqual(autocorr_time(np.full(5504, value)), .5)
        rng = np.random.default_rng(431)
        x = np.zeros(4096)
        for i, noise in enumerate(rng.normal(size=len(x) - 1), 1):
            x[i] = .8 * x[i - 1] + noise
        tau = autocorr_time(x)
        self.assertGreater(tau, 2.)
        self.assertLess(tau, 8.)
        for offset in (-100., .1, 100.):
            self.assertAlmostEqual(autocorr_time(x + offset), tau, places=11)

    def test_roundoff_only_sectors_and_real_jumps_for_both_dtypes(self):
        for dtype in (np.float32, np.float64):
            with self.subTest(dtype=dtype):
                n = 512
                rng = np.random.default_rng(51)
                sectors = rng.integers(-1, 2, (n, 2, 3)).astype(float)
                sectors[:, 0] = 0.
                values = sectors + np.array([.12, -.37, 0.])
                values += rng.normal(size=values.shape) * np.finfo(dtype).eps
                stats = _per_boundary_stats(values.astype(dtype), topology_area=4,
                                             topology_dtype=dtype)
                np.testing.assert_array_equal(stats["constant_sector"][0], True)
                np.testing.assert_array_equal(stats["tau"][0], .5)
                np.testing.assert_array_equal(stats["ess"][0], n)
                np.testing.assert_array_equal(stats["sector_transitions"][0], 0)
                expected = np.count_nonzero(np.diff(sectors[:, 1], axis=0), axis=0)
                np.testing.assert_array_equal(stats["sector_transitions"][1], expected)
                for col in range(3):
                    self.assertAlmostEqual(stats["tau"][1, col],
                                           autocorr_time(sectors[:, 1, col]))

    def test_single_jump_remains_correlated(self):
        values = np.zeros((5504, 1, 3)) + np.array([.1, .2, 0.])
        values[2752:, 0, 1] += 1.
        stats = _per_boundary_stats(values, topology_area=4)
        self.assertEqual(stats["sector_transitions"][0, 1], 1)
        self.assertFalse(stats["constant_sector"][0, 1])
        self.assertLess(stats["ess"][0, 1], 4.)
        np.testing.assert_array_equal(stats["constant_sector"][0, [0, 2]], True)

    def test_quantization_tolerance_uses_area_dtype_and_charge_scale(self):
        epsilon = np.finfo(np.float64).eps
        for area, charge in ((4, 0.), (16, 0.), (4, 100.),
                             (16384, 0.), (4, 16384.)):
            tolerance = max(1e-10, 64 * epsilon * max(1, area, abs(charge)))
            values = np.full((16, 1, 3), charge)
            values[1, 0, 1] += .5 * tolerance
            self.assertTrue(_per_boundary_stats(values, topology_area=area)
                            ["constant_sector"].all())
            values[1, 0, 1] = charge + 2 * tolerance
            with self.assertRaisesRegex(ValueError, "chain=0 observable=Q_z.*residual=.*tolerance="):
                _per_boundary_stats(values, topology_area=area)
        values = np.zeros((16, 2, 3))
        values[1, 1, 2] = .1
        with self.assertRaisesRegex(ValueError, "chain=1 observable=Q_s"):
            _per_boundary_stats(values, topology_area=4)
        for dtype in (np.float32, np.float64):
            values = np.zeros((16, 1, 3))
            tolerance = 64 * np.finfo(dtype).eps * 4
            if dtype == np.float64:
                tolerance = max(tolerance, 1e-10)
            values[1, 0, 0] = 2 * tolerance
            with self.assertRaisesRegex(ValueError, "observable=Q_U"):
                _per_boundary_stats(values, topology_area=4, topology_dtype=dtype)
        values[1, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            _per_boundary_stats(values, topology_area=4)

    def test_float64_sector_roundoff_floor_preserves_integer_jumps(self):
        values = np.zeros((32, 1, 3))
        values[1:, 0, 1] = 6.17284e-14
        stats = _per_boundary_stats(values, topology_area=4)
        self.assertTrue(stats["constant_sector"].all())
        np.testing.assert_array_equal(stats["ess"], 32.)

        values[16:, 0, 1] += 1.
        stats = _per_boundary_stats(values, topology_area=4)
        self.assertEqual(stats["sector_transitions"][0, 1], 1)
        self.assertFalse(stats["constant_sector"][0, 1])

    @staticmethod
    def sample(history, minimum, target):
        class Sampler:
            chains = history.shape[1]
            step = -1

            def sweep(self, **kwargs):
                self.step += 1

        class Progress:
            metrics = []

            def phase(self, *args):
                pass

            def update(self, count=1, **metrics):
                self.metrics.append(metrics)

        settings = dict(min_meas_per_boundary=minimum,
                        max_meas_per_boundary=len(history),
                        target_ess_per_boundary=target, s_step=1., s_updates=3)
        progress = Progress()
        data, stats = _adaptive_samples(
            Sampler(), settings, lambda s: history[s.step], progress=progress,
            topology_area=4)
        return data, stats, progress.metrics

    def test_constant_sectors_obey_minimum_and_nominal_ess_target(self):
        history = np.zeros((128, 2, 3)) + np.array([.1, .2, 0.])
        for target, actual in ((16, 32), (64, 64)):
            data, stats, metrics = self.sample(history, 32, target)
            self.assertEqual(len(data), actual)
            self.assertEqual(stats["stop_reason"], "target_ess_per_boundary")
            np.testing.assert_array_equal(data, history[:actual])
            np.testing.assert_array_equal(stats["ess"], actual)
            self.assertTrue(any(m.get("constant_sectors") == "Q_U:2,Q_z:2,Q_s:2"
                                for m in metrics))

    def test_each_charge_can_block_stopping_and_raw_history_is_unchanged(self):
        for col in range(3):
            with self.subTest(col=col):
                history = np.zeros((128, 2, 3)) + np.array([.1, .2, 0.])
                history[32:, 1, col] += 1.
                data, stats, _ = self.sample(history, 64, 32)
                np.testing.assert_array_equal(data, history)
                self.assertEqual(stats["stop_reason"], "max_meas_per_boundary")
                self.assertEqual(stats["sector_transitions"][1, col], 1)
                self.assertFalse(stats["constant_sector"][1, col])

    def test_angle_checks_still_include_both_sines_and_cosines(self):
        rng = np.random.default_rng(76)
        angles = rng.uniform(-np.pi, np.pi, (512, 2, 2))
        transform = lambda v: np.concatenate((np.sin(v), np.cos(v)), axis=-1)
        stats = _per_boundary_stats(angles, transform)
        self.assertEqual(stats["ess"].shape, (2, 4))
        reference = transform(angles)
        for chain in range(2):
            for col in range(4):
                self.assertEqual(stats["tau"][chain, col],
                                 autocorr_time(reference[:, chain, col]))
        self.assertNotIn("constant_sector", stats)

    def test_sector_diagnostics_aggregate_and_appear_in_summary(self):
        batch = dict(
            freq_z=np.ones(1), accept_rate=np.array([.8]), epsilon=.05,
            outer_warmup_sweeps=20, outer_warmup_stop_reason="tau_stable",
            outer_warmup_tau=np.ones(3), warmup_sweeps=0,
            warmup_stop_reason="copied_equilibrium", warmup_tau=np.empty(0),
            inner_initialization="copied_equilibrium", inner_adapt_steps=2,
            sampling_actual=2, sampling_per_chain=2,
            sampling_stop_reason="target_ess_per_boundary", sampling_target_ess=1,
            sampling_tau=np.full((1, 3), .5), sampling_ess=np.full((1, 3), 2.),
            sampling_constant_sector=np.array([[True, False, True]]),
            sampling_sector_transitions=np.array([[0, 1, 0]]))
        settings = dict(chains=1, fit_times=2, min_meas_per_boundary=2,
                        max_meas_per_boundary=4, target_ess_per_boundary=1)
        result = _aggregate_batches([batch, batch], settings, ("freq_z",))
        self.assertEqual(result["sampling_constant_sector"].shape, (2, 1, 3))
        self.assertEqual(result["sampling_sector_transitions"].shape, (2, 1, 3))
        self.assertIn("constant_sectors=Q_U:1,Q_z:0,Q_s:1",
                      _batch_summary("one_plaq", 0, batch))


if __name__ == "__main__":
    unittest.main()
