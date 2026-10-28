import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from cpn_renorm.exact_fitting import Conditional
from cpn_renorm.fit_client import isolated_plaq_plot, isolated_plot
from cpn_renorm.fitting import match_coupling, one_plaq_probability, two_plaq_probability
from cpn_renorm.plotting import (one_plaq_curve, plot_scan, select_boundaries,
                                 two_plaq_curves)


def boundary_data(count=12, bins=24):
    rng = np.random.default_rng(83)
    boundary = rng.uniform(-np.pi, np.pi, (count, 8))
    boundary[:, 6] = rng.uniform(.1, .9, count)
    angles = -np.pi + (np.arange(bins) + .5) * 2 * np.pi / bins
    X = np.column_stack((np.tile(angles, count), np.repeat(boundary, bins, axis=0)))
    fit = {"new": {"beta": .6, "alpha": .3, "alpha1": -.1,
                    "normalization_nodes": 512},
           "old": {"beta": .5, "alpha_eff": .2}}
    freq = Conditional(2, X, nodes=512).predict([.6, .3, -.1])
    return X, freq, fit


class PlottingTests(unittest.TestCase):
    def test_continuous_one_curve_matches_original_unwrapped_flux(self):
        X = np.array([[s, d] for s in range(-3, 4) for d in (-9., -np.pi, .2, 8.)])
        f = X[:, 0] + X[:, 1] / (2 * np.pi)
        for alpha in (.001, .3, 2.):
            np.testing.assert_allclose(one_plaq_curve(f, alpha),
                                       one_plaq_probability(X, alpha)[0], atol=1e-13)

    def test_full_and_folded_predictions_and_periodicity(self):
        X, _, fit = boundary_data()
        full, folded = two_plaq_curves(2, X, fit)
        np.testing.assert_allclose(full, Conditional(2, X, nodes=512).predict([.6, .3, -.1]))
        np.testing.assert_allclose(folded, two_plaq_probability(2, X, .5, .2))
        shifted = X.copy()
        shifted[:, 0] += 2 * np.pi
        for actual, expected in zip(two_plaq_curves(2, shifted, fit), (full, folded)):
            np.testing.assert_allclose(actual, expected, atol=1e-13)
        # At alpha=0, the full model reduces to the folded cosine model.
        fit["new"].update(beta=.5, alpha=0., alpha1=.2)
        full, folded = two_plaq_curves(2, X, fit)
        np.testing.assert_allclose(full, folded, atol=1e-13)

    def test_boundary_selection_preserves_blocks_and_random_state(self):
        X, _, _ = boundary_data()
        np.random.seed(3)
        state = np.random.get_state()
        selected = select_boundaries(X, 24, 17)
        np.testing.assert_array_equal(selected, select_boundaries(X, 24, 17))
        self.assertEqual(len(set(selected)), 9)
        np.testing.assert_array_equal(state[1], np.random.get_state()[1])
        self.assertEqual(set(select_boundaries(X[:48], 24, 17)), {0, 1})
        with self.assertRaises(ValueError):
            select_boundaries(X[:-1], 24, 17)

    def test_scan_zoom_uses_final_points_and_keeps_all_errors(self):
        x = np.array([-10., 0., .5, 1., 2., 10.])
        y = 2 * x + 1
        match = match_coupling(x[1:5], y[1:5], 2.5, np.full(4, .02), .03)
        request = {"destination": "unused.png", "series": ["obs"],
                   "matches": {"obs": match}, "errors": {"obs": "err"},
                   "targets": {"fine": 2.5}, "target_errors": {"fine": .03},
                   "xlabel": "beta1", "ylabel": "observable"}
        with patch("cpn_renorm.plotting._save", side_effect=lambda fig, _: fig):
            fig = plot_scan(request, {"x": x, "obs": y, "err": np.full(6, .02)})
        try:
            overview, zoom = fig.axes
            self.assertLess(overview.get_xlim()[0], -10.)
            self.assertGreater(overview.get_xlim()[1], 10.)
            self.assertGreater(zoom.get_xlim()[0], -1.)
            self.assertLess(zoom.get_xlim()[1], 3.)
            for ax in fig.axes:
                # Exploration, actual fit points, and crossing all carry errors.
                self.assertEqual(len(ax.containers), 3)
            self.assertTrue(any("beta1 =" in t.get_text() for t in zoom.texts))
        finally:
            import matplotlib.pyplot as plt
            plt.close(fig)

    def test_worker_png_outputs_and_unavailable_fit(self):
        X, freq, fit = boundary_data(count=2)
        one_X = np.array([[s, d] for s in range(-3, 4) for d in (-1., .5)])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            isolated_plaq_plot(root / "one.png", operation="plot_one",
                request={"tags": ["U"], "fits": {"U": {"alpha": .3, "success": True}}},
                X_U=one_X, freq_U=one_plaq_probability(one_X, .3)[0])
            result = isolated_plaq_plot(root / "two.png", operation="plot_two",
                request={"N": 2, "tag": "U", "fit": fit, "bins": 24, "seed": 17},
                X=X, freq=freq)
            self.assertEqual(set(result["boundary_indices"]), {0, 1})
            isolated_plot(root / "failed.png", x=np.array([0., 1., 2.]),
                series={"obs": np.array([1., np.nan, 2.])}, targets={"fine": 3.},
                matches={"obs": {"bracketed": False, "reason": "no bracket"}},
                xlabel="alpha", ylabel="susceptibility")
            for name in ("one.png", "two.png", "failed.png"):
                self.assertTrue((root / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
                self.assertGreater((root / name).stat().st_size, 10000)


if __name__ == "__main__":
    unittest.main()
