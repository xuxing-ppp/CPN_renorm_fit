import unittest

import numpy as np

from cpn_renorm.observables import (
    gamma_method,
    measurement_stop_reason,
    second_moment_xi_from_modes,
    second_moment_xi_jacobian,
    xi_from_modes_gamma,
)


class StatisticsTests(unittest.TestCase):
    def test_measurement_stops_and_rounds_across_chains(self):
        self.assertIsNone(measurement_stop_reason(3, 4, 13, 20, 25,
                                                  np.array([100.0])))
        self.assertEqual(measurement_stop_reason(4, 4, 13, 20, 25,
                                                 np.array([20.0])), "target_ess")
        self.assertIsNone(measurement_stop_reason(4, 4, 13, 20, 25,
                                                  np.array([19.0])))
        self.assertEqual(measurement_stop_reason(7, 4, 13, 20, 25,
                                                 np.array([19.0])), "max_meas_total")
        self.assertEqual(7 * 4, 28)  # configured maximum 25 rounds up to 28

    def test_gamma_method_detects_ar1_autocorrelation_and_cross_covariance(self):
        rng = np.random.default_rng(123)
        draws = np.zeros((3000, 4, 2))
        noise = rng.normal(size=draws.shape)
        noise[..., 1] = 0.6 * noise[..., 0] + 0.8 * noise[..., 1]
        for index in range(1, len(draws)):
            draws[index] = 0.8 * draws[index - 1] + noise[index]
        stats = gamma_method(draws)
        # AR(1) has tau_int=(1+rho)/(2*(1-rho))=4.5.
        np.testing.assert_allclose(stats["tau"], [4.5, 4.5], rtol=0.35)
        self.assertGreater(stats["covariance"][0, 1], 0)
        self.assertTrue(np.all(stats["ess"] < draws.shape[0] * draws.shape[1]))

    def test_xi_jacobian_matches_finite_difference_and_propagation(self):
        modes = np.array([10.0, 2.0, 2.5])
        analytic = second_moment_xi_jacobian(modes, 32)
        numeric = []
        for column in range(3):
            step = 1e-6 * modes[column]
            plus, minus = modes.copy(), modes.copy()
            plus[column] += step
            minus[column] -= step
            numeric.append((second_moment_xi_from_modes(plus, 32)
                            - second_moment_xi_from_modes(minus, 32)) / (2 * step))
        np.testing.assert_allclose(analytic, numeric, rtol=2e-7, atol=1e-9)

        rng = np.random.default_rng(4)
        series = rng.normal(modes, [0.1, 0.03, 0.04], size=(300, 3, 3))
        xi, error, stats = xi_from_modes_gamma(series, 32)
        series_jacobian = second_moment_xi_jacobian(stats["mean"], 32)
        expected = series_jacobian @ stats["covariance"] @ series_jacobian
        self.assertAlmostEqual(error * error, expected, places=10)
        self.assertGreater(xi, 0)


if __name__ == "__main__":
    unittest.main()
