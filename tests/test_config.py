from pathlib import Path
from copy import deepcopy
import unittest

from cpn_renorm.config import load_config, resolved_geometry, section, validate_config


class ConfigTests(unittest.TestCase):
    def test_example_config_and_geometry(self):
        cfg = load_config(Path(__file__).parents[1] / "configs" / "example.toml")
        geo = resolved_geometry(cfg, 2.01)
        self.assertEqual(geo["padding"], 5)
        self.assertEqual(geo["L_coarse"], 10)
        self.assertEqual(geo["L_fine"] % cfg["renormalization"]["factor"], 0)
        self.assertEqual(geo["L_coarse"] * geo["factor"], geo["L_fine"])

    def test_geometry_minus_one_uses_xi_rules(self):
        cfg = load_config(Path(__file__).parents[1] / "configs" / "example.toml")
        cfg["geometry"]["padding"] = -1
        cfg["geometry"]["coarse_L"] = -1
        geo = resolved_geometry(cfg, 2.01)
        self.assertEqual(geo["padding"], 3)
        self.assertEqual(geo["L_coarse"], 4)

    def test_patch_fit_counts_match_reference(self):
        cfg = load_config(Path(__file__).parents[1] / "configs" / "example.toml")
        self.assertEqual(cfg["steps"]["two_plaq"]["fit_times"], 5000)
        self.assertEqual(cfg["steps"]["two_plaq"]["min_meas_per_boundary"], 3000)
        self.assertEqual(cfg["steps"]["one_plaq"]["fit_times"], 2000)
        self.assertEqual(cfg["steps"]["one_plaq"]["min_meas_per_boundary"], 1000)

    def test_patch_rejects_aggregate_measurement_fields(self):
        cfg = load_config(Path(__file__).parents[1] / "configs" / "example.toml")
        cfg["steps"]["two_plaq"]["min_meas_total"] = 100
        with self.assertRaisesRegex(ValueError, "per_boundary"):
            validate_config(cfg)

    def test_chains_are_stage_specific(self):
        cfg = load_config(Path(__file__).parents[1] / "configs" / "example.toml")
        self.assertNotIn("chains", cfg["runtime"])
        self.assertEqual(cfg["pilot"]["chains"], 16)
        for settings in cfg["steps"].values():
            self.assertGreater(settings["chains"], 0)

    def test_rejects_obsolete_runtime_chains(self):
        cfg = load_config(Path(__file__).parents[1] / "configs" / "example.toml")
        bad = deepcopy(cfg)
        bad["runtime"]["chains"] = 8
        with self.assertRaisesRegex(ValueError, "runtime.chains is not used"):
            validate_config(bad)

    def test_rejects_nonpositive_stage_chains(self):
        cfg = load_config(Path(__file__).parents[1] / "configs" / "example.toml")
        bad = deepcopy(cfg)
        bad["steps"]["topo"]["chains"] = 0
        with self.assertRaisesRegex(ValueError, "steps.topo.chains"):
            validate_config(bad)

    def test_all_steps_use_shared_hmc_settings(self):
        cfg = load_config(Path(__file__).parents[1] / "configs" / "example.toml")
        for name, step_settings in cfg["steps"].items():
            self.assertTrue(set(step_settings).isdisjoint(cfg["hmc"]))
            settings = section(cfg, name)
            for key, value in cfg["hmc"].items():
                self.assertEqual(settings[key], value)

    def test_rejects_step_local_hmc_settings(self):
        cfg = load_config(Path(__file__).parents[1] / "configs" / "example.toml")
        bad = deepcopy(cfg)
        bad["steps"]["topo"]["epsilon"] = 0.02
        with self.assertRaisesRegex(
                ValueError, r"steps\.topo\.epsilon.*hmc\.epsilon"):
            validate_config(bad)

    def test_load_rejects_obsolete_sampling_fields_with_migration_hint(self):
        source = (Path(__file__).parents[1] / "configs" / "example.toml").read_text()
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "old.toml"
            path.write_text(source + "\n[steps.observable.old]\nn_meas = 10\n")
            with self.assertRaisesRegex(ValueError, "min_meas_total"):
                load_config(path)
