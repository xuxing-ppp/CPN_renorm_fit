import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cpn_renorm.config import load_config, mod_config
from cpn_renorm.pilot import _pilot_fingerprint, run_pilot


class PilotTests(unittest.TestCase):
    def config(self):
        cfg = mod_config(load_config(Path(__file__).parents[1] / "configs/example.toml"), 1)
        cfg["runtime"].update(device="cpu", auto_chains=False)
        cfg["pilot"].update(initial_L=20, min_L_over_xi_G=2.0)
        return cfg

    def run_mocked(self, cfg, measurements, folder):
        with patch("cpn_renorm.pilot.fine_sampler") as sampler, patch(
                "cpn_renorm.pilot.measure_ensemble", side_effect=measurements), patch(
                "cpn_renorm.pilot.ensemble_summary", return_value="done"):
            result = run_pilot(cfg, folder)
        return result, [call.args[1] for call in sampler.call_args_list]

    def test_growth_uses_conservative_estimate_and_margin(self):
        # The center alone would pass at L=20; its error requires growth.
        for margin, expected in ((1.0, 25), (1.25, 31), (1.5, 37)):
            with self.subTest(margin=margin), tempfile.TemporaryDirectory() as folder:
                cfg = self.config()
                cfg["pilot"]["pilot_growth_margin"] = margin
                result, sizes = self.run_mocked(cfg, [
                    {"xi_G": 10.0, "xi_G_err": 1.1},
                    {"xi_G": 10.0, "xi_G_err": 0.0}], folder)
                self.assertEqual(sizes, [20, expected])
                self.assertEqual([row["L"] for row in result["attempts"]], sizes)
                self.assertAlmostEqual(result["attempts"][0]["conservative_L_over_xi_G"], 20 / 12.2)

    def test_margin_does_not_change_pass_condition(self):
        cfg = self.config()
        cfg["pilot"]["pilot_growth_margin"] = 3.0
        with tempfile.TemporaryDirectory() as folder:
            _, sizes = self.run_mocked(cfg, [{"xi_G": 8.0, "xi_G_err": 1.0}], folder)
        self.assertEqual(sizes, [20])

    def test_growth_caps_at_128_and_can_pass_there(self):
        cfg = self.config()
        with tempfile.TemporaryDirectory() as folder:
            _, sizes = self.run_mocked(cfg, [
                {"xi_G": 80.0, "xi_G_err": 0.0},
                {"xi_G": 60.0, "xi_G_err": 2.0}], folder)
        self.assertEqual(sizes, [20, 128])

    def test_failure_at_128_is_saved_without_sampling_larger_lattice(self):
        cfg = self.config()
        with tempfile.TemporaryDirectory() as folder, patch(
                "cpn_renorm.pilot.fine_sampler") as sampler, patch(
                "cpn_renorm.pilot.measure_ensemble",
                return_value={"xi_G": 80.0, "xi_G_err": 0.0}), patch(
                "cpn_renorm.pilot.ensemble_summary", return_value="done"):
            with self.assertRaisesRegex(RuntimeError, "reached max_L"):
                run_pilot(cfg, folder)
            failed = json.loads((Path(folder) / "pilot/failed.json").read_text())
            self.assertEqual([row["L"] for row in failed["attempts"]], [20, 128])
            self.assertEqual([call.args[1] for call in sampler.call_args_list], [20, 128])
            self.assertFalse((Path(folder) / "pilot/result.json").exists())

    def test_margin_change_invalidates_cached_result(self):
        cfg = self.config()
        measurements = [{"xi_G": 8.0, "xi_G_err": 1.0}]
        with tempfile.TemporaryDirectory() as folder:
            first, sizes = self.run_mocked(cfg, measurements, folder)
            self.assertEqual(sizes, [20])
            cached, sizes = self.run_mocked(cfg, [], folder)
            self.assertEqual(cached["cache"], "reused")
            self.assertEqual(sizes, [])
            cfg["pilot"]["pilot_growth_margin"] = 1.5
            self.assertNotEqual(first["fingerprint"], _pilot_fingerprint(cfg))
            regenerated, sizes = self.run_mocked(cfg, measurements, folder)
            self.assertEqual(regenerated["cache"], "generated")
            self.assertEqual(sizes, [20])
