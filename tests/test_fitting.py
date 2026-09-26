import numpy as np
import unittest

from cpn_renorm.fitting import extend_scan, match_coupling
from cpn_renorm.fit_client import isolated_match


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
        self.assertEqual(result["fit_type"], "gcv_smoothing_spline")
        self.assertEqual(result["raw_bracket"], [0.0, 1.0])
        self.assertGreater(result["error"], 0)
