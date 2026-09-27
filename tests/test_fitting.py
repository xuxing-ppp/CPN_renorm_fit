import numpy as np
import unittest

from cpn_renorm.fit_client import extend_scan, isolated_match


def match_coupling(*args, **kwargs):
    return isolated_match(*args, **kwargs)


def evaluate_match_curve(match, x):
    coeff = np.asarray(match["fit_coefficients"])
    scaled = (np.asarray(x) - match["fit_center"]) / match["fit_scale"]
    return np.polynomial.polynomial.polyval(scaled, coeff)


class FittingTests(unittest.TestCase):
    def test_match_increasing_and_decreasing(self):
        x = np.array([0.0, 1.0, 2.0])
        self.assertLess(abs(match_coupling(x, x, 0.75)["value"] - 0.75), 1e-3)
        self.assertLess(abs(match_coupling(x, 2 - x, 0.75)["value"] - 1.25), 1e-3)

    def test_extension_follows_endpoint_closest_to_target(self):
        result = match_coupling(np.array([0.0, 1.0]), np.array([2.0, 1.0]), 0.0)
        self.assertEqual(result["direction"], "high")
        self.assertEqual(extend_scan([0.0, 1.0], result, (-4.0, 4.0), 2.0),
                         [0.0, 1.0, 3.0])

    def test_default_geometric_extension_reaches_distant_bracket(self):
        points = [-0.1, 0.0, 0.1]
        for expected in (0.3, 0.7, 1.5, 3.1, 6.3):
            points = extend_scan(points, {"direction": "high"}, (-8.0, 8.0), 2.0)
            self.assertAlmostEqual(points[-1], expected)
        points = [-0.1, 0.0, 0.1]
        for expected in (-0.3, -0.7, -1.5, -3.1, -6.3):
            points = extend_scan(points, {"direction": "low"}, (-8.0, 8.0), 2.0)
            self.assertAlmostEqual(points[0], expected)

    def test_distant_legacy_scale_root_is_not_anchored_to_fine_beta1(self):
        # Legacy results include coarse matches near -5.67 even at fine beta1=0.
        x = np.array([-6.3, -5.7, -5.5, -3.1, 0.0])
        y = x + 5.67
        result = isolated_match(x, y, 0.0, np.full(len(x), 0.03))
        self.assertTrue(result["bracketed"])
        self.assertAlmostEqual(result["value"], -5.67, delta=0.08)

    def test_weighted_match_records_raw_bracket_and_target_error(self):
        x = np.linspace(-3, 3, 7)
        result = isolated_match(x, x ** 3 + x, 0.2,
                                np.full(len(x), 0.05), target_err=0.1)
        self.assertTrue(result["bracketed"])
        self.assertEqual(result["fit_type"], "monotonic_weighted_poly_2")
        self.assertEqual(result["raw_bracket"], [0.0, 1.0])
        self.assertGreater(result["error"], 0)
        self.assertLessEqual(len(result["fit_points"]), 5)

    def test_local_fit_is_monotonic_and_ignores_distant_points(self):
        x = np.arange(9, dtype=float)
        y = np.array([-100.0, -50.0, 0.0, 1.0, 2.0, 3.0, 4.0, 60.0, 100.0])
        result = match_coupling(x, y, 2.4, np.full(len(x), 0.1),
                                increasing=True, max_points=5)
        grid = np.linspace(min(result["fit_points"]), max(result["fit_points"]), 101)
        self.assertTrue(np.all(np.diff(evaluate_match_curve(result, grid)) >= -1e-9))
        self.assertEqual(len(result["fit_points"]), 5)
        self.assertNotIn(0.0, result["fit_points"])
        self.assertNotIn(8.0, result["fit_points"])

    def test_decreasing_fit_and_uncertainty(self):
        x = np.arange(5, dtype=float)
        y = np.array([5.0, 4.2, 3.0, 2.1, 1.0])
        small = match_coupling(x, y, 2.5, np.full(5, 0.02), 0.02,
                               increasing=False, bootstrap_seed=7)
        large = match_coupling(x, y, 2.5, np.full(5, 0.2), 0.2,
                               increasing=False, bootstrap_seed=7)
        self.assertEqual(small["monotonic"], "decreasing")
        self.assertGreater(large["error"], small["error"])
