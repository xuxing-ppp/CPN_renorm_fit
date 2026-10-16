from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cpn_renorm.config import load_config
from cpn_renorm.migrate import _replay_search
from cpn_renorm.pipeline import _retain_scan, _search_fit

ROOT = Path(__file__).parents[1]


class RetentionTests(unittest.TestCase):
    def test_replay_separates_coarse_bisection_and_fit_grid(self):
        scan = {"min": -1., "max": 1., "growth_factor": 2., "max_rounds": 7,
                "refine_tolerance": .11, "max_refine_rounds": 10}
        measured = {x: {"value": x} for x in (-.1, 0., .1, .3, .4, .46, .48, .5, .7)}
        matches = {"corrG": {"expansion_rounds": 2, "refinement_rounds": 2}}
        coarse = _replay_search(scan, [-.1, 0., .1], measured,
            {"corrG": (lambda r: r["value"], .5, False)}, matches)
        self.assertEqual(coarse, {-.1, 0., .1, .3, .7})
        matches["corrG"]["refinement_rounds"] = 9
        with self.assertRaisesRegex(ValueError, "refinement_rounds"):
            _replay_search(scan, [-.1, 0., .1], measured,
                {"corrG": (lambda r: r["value"], .5, False)}, matches)

    def test_incomplete_trace_stops_replay(self):
        cfg = load_config(ROOT / "configs/example.toml")
        with self.assertRaisesRegex(ValueError, "missing historical"):
            _replay_search(cfg["scan"]["beta1"], [-.1, 0., .1], {0.: {}}, {}, {})

    def test_failed_fit_grid_is_not_retained_as_final(self):
        cfg = load_config(ROOT / "configs/example.toml")
        scan = {**cfg["scan"]["beta1"], "refine_tolerance": 2.}
        measured, phases = {}, {}
        def measure(value):
            return {"value": value}
        def failed(**kw):
            for value in (.2, .3, .4):
                kw["measured"][value] = kw["measure"](value)
            return {"bracketed": False, "fit_grid": [.2, .3, .4], "fit_points": []}
        with patch("cpn_renorm.pipeline._uniform_grid_match", side_effect=failed):
            _search_fit(cfg, scan, measured, phases, measure, lambda r: (r["value"], .01),
                        .5, .01, initial=[0., .5, 1.], increasing=True, seed=1)
        with tempfile.TemporaryDirectory() as tmp:
            entries, _ = _retain_scan(Path(tmp), measured, phases, tag="obs_coarse")
            self.assertEqual(set(entries), {"0.0", "0.5", "1.0"})
            self.assertFalse(any("final" in labels for labels in phases.values()))
