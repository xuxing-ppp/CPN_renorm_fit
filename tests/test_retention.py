from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cpn_renorm.config import load_config, mod_config
from cpn_renorm.migrate import _replay_search
from cpn_renorm.pipeline import (_prune_scan_ensembles, _retain_scan,
                                 _run_beta1_scan, _search_fit)
from cpn_renorm.cache import read_json
from cpn_renorm.storage import atomic_json

ROOT = Path(__file__).parents[1]


class RetentionTests(unittest.TestCase):
    def test_deselecting_method_drops_its_old_final_points_and_cache(self):
        cfg = mod_config(load_config(ROOT / "configs/example.toml"), 0)
        cfg["renormalization"]["beta1_methods"] = ["magsus", "corrG"]
        cfg["runtime"]["auto_chains"] = False
        geometry = {"factor": 1, "L_fine": 8, "L_coarse": 8}
        def measure(*args, **kwargs):
            return {"magsus": 1., "magsus_err": .01, "xi_G": 1., "xi_G_err": .01}
        def search(*args, **kwargs):
            measured, phases, sample = args[2:5]
            point = .02 if kwargs["seed"] == 201 else .04
            measured[point] = sample(point)
            phases[point] = {"final"}
            return {"value": point, "bracketed": True, "fit_points": [point]}
        with tempfile.TemporaryDirectory() as tmp, patch(
                "cpn_renorm.pipeline._measure_point", side_effect=measure), patch(
                "cpn_renorm.pipeline._search_fit", side_effect=search), patch(
                "cpn_renorm.pipeline._plot_beta1_scan"):
            root = Path(tmp)
            first = _run_beta1_scan(cfg, root, geometry, .6, .2)
            self.assertEqual(len(first["measurements"]), 2)
            cfg["renormalization"]["beta1_methods"] = ["magsus"]
            second = _run_beta1_scan(cfg, root, geometry, .6, .2)
            self.assertEqual(set(second["matches"]), {"magsus"})
            self.assertEqual(len(second["measurements"]), 1)
            self.assertEqual(len(_prune_scan_ensembles(root, second)), 1)
            self.assertEqual(set(read_json(root / "step3_observable/result.json")["matches"]), {"magsus"})

    def test_prune_removes_old_points_and_preserves_shared_current_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("fine", "coarse", "final", "topo_fine", "old_final", "failed_grid"):
                atomic_json(root / "ensembles" / name / "result.json", {})
            def entry(name):
                return {"path": f"ensembles/{name}/result.json"}
            observable = {"fine_path": entry("fine")["path"],
                          "measurements": {"0": entry("coarse"), "1": entry("final")},
                          "matches": {"corrG": {"fit_points": [1]}}}
            topo = {"fine_path": entry("topo_fine")["path"], "matches": {
                "corrG": {"measurements": {"0": entry("final")}},
                "magsus": {"measurements": {"0": entry("coarse")}}}}
            self.assertEqual(_prune_scan_ensembles(root, observable, topo),
                             ["failed_grid", "old_final"])
            self.assertEqual({p.name for p in (root / "ensembles").iterdir()},
                             {"fine", "coarse", "final", "topo_fine"})
            self.assertEqual(_prune_scan_ensembles(root, observable, topo), [])

    def test_prune_validates_all_references_before_deleting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / "ensembles/old/result.json"
            atomic_json(old, {})
            for bad in ("ensembles/missing/result.json", "../outside/result.json"):
                with self.assertRaises(RuntimeError):
                    _prune_scan_ensembles(root, {"fine_path": bad})
                self.assertTrue(old.exists())

    def test_disabled_scans_drop_all_old_ensembles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            atomic_json(root / "ensembles/old/result.json", {})
            self.assertEqual(_prune_scan_ensembles(root, {"matches": {}}), ["old"])

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
