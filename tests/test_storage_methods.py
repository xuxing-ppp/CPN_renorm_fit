from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from cpn_renorm.cache import read_json, scan_measurements
from cpn_renorm.config import load_config, mod_config, validate_config, section
from cpn_renorm.pipeline import (_default_alpha_points, _measure_point,
                                 _remove_scan_ensembles, _run_beta1_scan, run_pipeline)

ROOT = Path(__file__).parents[1]


class StorageMethodsTests(unittest.TestCase):
    def config(self):
        cfg = load_config(ROOT / "configs/example.toml")
        cfg["runtime"].update(device="cpu", auto_chains=False)
        return cfg

    def test_validation(self):
        cfg = self.config()
        cfg["renormalization"].update(beta1_methods=[])
        validate_config(cfg)
        cfg["renormalization"]["alpha_methods"] = ["topo"]
        with self.assertRaisesRegex(ValueError, "obsolete configuration field renormalization.alpha_methods"):
            validate_config(cfg)
        for methods in (["corrG", "corrG"], ["unknown"], "corrG", [False]):
            cfg = self.config()
            cfg["renormalization"]["beta1_methods"] = methods
            with self.assertRaises(ValueError):
                validate_config(cfg)
        cfg = self.config()
        cfg["storage"]["save_npz"] = "false"
        with self.assertRaisesRegex(ValueError, "boolean"):
            validate_config(cfg)

    def test_json_only_point_reuses_and_npz_is_optional(self):
        cfg = mod_config(self.config(), 1)
        measured = {"modes": np.ones((4, 2, 3)), "accept_rate": np.array([0.7, 0.9]),
                    "sampling": {"stop_reason": "target_ess"},
                    "warmup": {"stop_reason": "tau_stable"}}
        for enabled in (False, True):
            cfg["storage"]["save_npz"] = enabled
            with self.subTest(enabled=enabled), tempfile.TemporaryDirectory() as tmp, patch(
                    "cpn_renorm.pipeline._point_sampler"), patch(
                    "cpn_renorm.pipeline.measure_ensemble", return_value=measured) as sample, patch(
                    "cpn_renorm.pipeline.ensemble_summary", return_value="done"):
                args = dict(tag="obs_coarse", L=4, beta=1., beta1=0., alpha=0., alpha1=0., kind="half")
                _measure_point(cfg, Path(tmp), section(cfg, "observable"), **args)
                reused = _measure_point(cfg, Path(tmp), section(cfg, "observable"), **args)
                sample.assert_called_once()
                self.assertAlmostEqual(reused["accept_rate"], 0.8)
                self.assertEqual(bool(list(Path(tmp).rglob("*.npz"))), enabled)

    def test_empty_methods_still_estimate_independent_alpha(self):
        cfg = self.config()
        cfg["renormalization"].update(beta1_methods=[])
        geometry = {"padding": 0, "L_fine": 8, "L_coarse": 2, "factor": 4}
        with tempfile.TemporaryDirectory() as tmp, patch(
                "cpn_renorm.pipeline.run_pilot", return_value={"geometry": geometry}), patch(
                "cpn_renorm.pipeline._run_two_plaq", return_value={"fits": {"z": {"new": {"beta": .6, "alpha1": .2}}}}), patch(
                "cpn_renorm.pipeline._run_beta1_scan") as obs, patch(
                "cpn_renorm.pipeline._run_one_plaq", return_value={"fits": {"z": {"alpha": .1}}}) as one, patch(
                "cpn_renorm.pipeline._run_topo") as topo:
            result = run_pipeline(cfg, tmp)
            branch = result["branches"]["z_mod1"]
            self.assertIsNone(branch["results"]["1plaq"]["couplings"]["beta1"])
            self.assertEqual(branch["results"]["1plaq"]["couplings"]["alpha"], .1)
            obs.assert_not_called()
            one.assert_called_once()
            topo.assert_not_called()

    def test_topo_automatically_runs_after_one_plaquette(self):
        cfg = self.config()
        cfg["renormalization"].update(beta1_methods=["corrG"])
        geometry = {"padding": 0, "L_fine": 8, "L_coarse": 2, "factor": 4}
        with tempfile.TemporaryDirectory() as tmp, patch(
                "cpn_renorm.pipeline.run_pilot", return_value={"geometry": geometry}), patch(
                "cpn_renorm.pipeline._run_two_plaq", return_value={"fits": {"z": {"new": {"beta": .6, "alpha1": .2}}}}), patch(
                "cpn_renorm.pipeline._run_beta1_scan", return_value={"matches": {"corrG": {"value": .3}}}), patch(
                "cpn_renorm.pipeline._run_one_plaq", return_value={"fits": {"z": {"alpha": .1}}}) as one, patch(
                "cpn_renorm.pipeline._run_topo", return_value={"matches": {"corrG": {"value": .1}}}) as topo:
            result = run_pipeline(cfg, tmp)
            one.assert_called_once()
            topo.assert_called_once()
            self.assertEqual(set(result["branches"]["z_mod1"]["results"]), {"corrG_1plaq"})
        with self.assertRaisesRegex(ValueError, "outside scan range"):
            _default_alpha_points(123., cfg["scan"]["alpha"])

    def test_adding_method_reuses_old_fit_and_coarse_points(self):
        cfg = self.config()
        cfg["renormalization"]["beta1_methods"] = ["magsus"]
        geometry = {"padding": 0, "L_fine": 8, "L_coarse": 8, "factor": 1}
        def measure(_cfg, _dir, _settings, *, tag, beta1, **kw):
            value = 1. if tag == "obs_fine" else 1. + beta1
            return {"magsus": value, "magsus_err": .001, "xi_G": value, "xi_G_err": .001}
        def match(**kw):
            # A deterministic final grid checks lifecycle, not the fitter formula.
            for x in (-.02, -.01, 0., .01, .02):
                if x not in kw["measured"]:
                    kw["measured"][x] = kw["measure"](x)
            return {"value": 0., "error": .001, "bracketed": True,
                    "fit_grid": [-.02, -.01, 0., .01, .02],
                    "fit_points": [-.02, -.01, 0., .01, .02]}
        with tempfile.TemporaryDirectory() as tmp, patch(
                "cpn_renorm.pipeline._measure_point", side_effect=measure) as sample, patch(
                "cpn_renorm.pipeline._uniform_grid_match", side_effect=match) as fit, patch(
                "cpn_renorm.pipeline._plot_beta1_scan"):
            directory = Path(tmp)
            first = _run_beta1_scan(cfg, directory, geometry, .6, .2)
            _remove_scan_ensembles(directory, first["_cleanup_folders"])
            sampled = sample.call_count
            cached_fit = deepcopy(first["matches"]["magsus"])
            cfg["renormalization"]["beta1_methods"].append("corrG")
            second = _run_beta1_scan(cfg, directory, geometry, .6, .2)
            self.assertEqual(fit.call_count, 2)
            self.assertEqual(sample.call_count, sampled)
            self.assertEqual(second["matches"]["magsus"], cached_fit)
            self.assertEqual(set(second["matches"]), {"magsus", "corrG"})
            stored = read_json(directory / "step3_observable/result.json")
            values, phases = scan_measurements(directory, stored)
            self.assertTrue({-.1, 0., .1}.issubset(values))
            self.assertIn("coarse", phases[-.1])
            self.assertTrue(all(labels != {"refine"} for labels in phases.values()))
            self.assertEqual(list(directory.rglob("*.npz")), [])
