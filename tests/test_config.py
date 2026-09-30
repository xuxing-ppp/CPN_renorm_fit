from pathlib import Path
from copy import deepcopy
import unittest

from cpn_renorm.config import load_config, resolved_geometry, section, validate_config


class ConfigTests(unittest.TestCase):
    def test_example_config_and_geometry(self):
        cfg = load_config(Path(__file__).parents[1] / "configs" / "example.toml")
        geo = resolved_geometry(cfg, 2.01)
        self.assertEqual(geo["padding"], 5)
        self.assertEqual(geo["L_coarse"], 6)
        self.assertEqual(geo["L_fine"] % cfg["renormalization"]["factor"], 0)
        self.assertEqual(geo["L_coarse"] * geo["factor"], geo["L_fine"])

    def test_geometry_minus_one_uses_xi_rules(self):
        cfg = load_config(Path(__file__).parents[1] / "configs" / "example.toml")
        cfg["geometry"]["padding"] = -1
        cfg["geometry"]["coarse_L"] = -1
        geo = resolved_geometry(cfg, 2.01)
        self.assertEqual(geo["padding"], 5)
        self.assertEqual(geo["L_coarse"], 6)

    def test_geometry_automatic_values_are_clamped_but_fixed_values_win(self):
        cfg = load_config(Path(__file__).parents[1] / "configs" / "example.toml")
        cfg["geometry"].update({"padding": -1, "padding_min": 7,
                                "padding_max": 9, "coarse_L": -1,
                                "coarse_L_min": 8, "coarse_L_max": 10})
        geo = resolved_geometry(cfg, 2.01)
        self.assertEqual(geo["padding"], 7)
        self.assertEqual(geo["L_coarse"], 8)
        cfg["geometry"].update({"padding": 2, "coarse_L": 20})
        geo = resolved_geometry(cfg, 2.01)
        self.assertEqual(geo["padding"], 2)
        self.assertEqual(geo["L_coarse"], 20)

    def test_renormalization_type_requires_a_unique_list(self):
        cfg = load_config(Path(__file__).parents[1] / "configs" / "example.toml")
        cfg["renormalization"]["type"] = "z"
        with self.assertRaisesRegex(ValueError, "nonempty unique list"):
            validate_config(cfg)
        cfg = load_config(Path(__file__).parents[1] / "configs" / "example.toml")
        cfg["renormalization"]["type"] = ["U", "z"]
        validate_config(cfg)
        self.assertEqual(cfg["renormalization"]["type"], ["z", "U"])

    def test_geometry_bounds_are_strict_and_pbc_allows_zero_padding(self):
        cfg = load_config(Path(__file__).parents[1] / "configs" / "example.toml")
        cfg["geometry"].update({"boundary_bc": "PBC", "padding": 0,
                                "padding_min": 0, "padding_max": 0})
        validate_config(cfg)
        cfg["geometry"].update({"padding_min": 3, "padding_max": 2})
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            validate_config(cfg)

    def test_patch_fit_counts_match_reference(self):
        cfg = load_config(Path(__file__).parents[1] / "configs" / "example.toml")
        self.assertEqual(cfg["steps"]["two_plaq"]["fit_times"], 5000)
        self.assertEqual(cfg["steps"]["two_plaq"]["min_meas_per_boundary"], 3000)
        self.assertEqual(cfg["steps"]["one_plaq"]["fit_times"], 2000)
        self.assertEqual(cfg["steps"]["one_plaq"]["min_meas_per_boundary"], 1000)
        self.assertEqual(cfg["scan"]["beta1"]["refine_tolerance"], 0.02)
        self.assertEqual(cfg["scan"]["alpha"]["refine_tolerance"], 0.002)
        self.assertEqual(cfg["scan"]["beta1"]["fit_points"], 5)
        self.assertEqual(cfg["scan"]["beta1"]["fit_spacing_min"], 0.01)
        self.assertEqual(cfg["scan"]["alpha"]["fit_spacing_max"], 0.006)

    def test_rejects_invalid_scan_refinement(self):
        cfg = load_config(Path(__file__).parents[1] / "configs" / "example.toml")
        bad = deepcopy(cfg)
        bad["scan"]["beta1"]["refine_tolerance"] = 0.0
        with self.assertRaisesRegex(ValueError, "refinement"):
            validate_config(bad)
        bad = deepcopy(cfg)
        bad["scan"]["alpha"]["fit_points"] = 4
        with self.assertRaisesRegex(ValueError, "refinement"):
            validate_config(bad)
        bad = deepcopy(cfg)
        bad["scan"]["alpha"]["fit_spacing_min"] = 0.01
        bad["scan"]["alpha"]["fit_spacing_max"] = 0.001
        with self.assertRaisesRegex(ValueError, "fit-grid"):
            validate_config(bad)

    def test_patch_rejects_aggregate_measurement_fields(self):
        cfg = load_config(Path(__file__).parents[1] / "configs" / "example.toml")
        cfg["steps"]["two_plaq"]["min_meas_total"] = 100
        with self.assertRaisesRegex(ValueError, "per_boundary"):
            validate_config(cfg)

    def test_chains_are_stage_specific(self):
        cfg = load_config(Path(__file__).parents[1] / "configs" / "example.toml")
        self.assertNotIn("chains", cfg["runtime"])
        self.assertEqual(cfg["chains"], {
            "pilot": 32, "two_plaq": 2048, "observable": 128,
            "one_plaq": 2048, "topo": 256,
        })
        self.assertNotIn("chains", cfg["pilot"])
        for name, settings in cfg["steps"].items():
            self.assertNotIn("chains", settings)
            self.assertEqual(section(cfg, name)["chains"], cfg["chains"][name])

    def test_rejects_obsolete_runtime_chains(self):
        cfg = load_config(Path(__file__).parents[1] / "configs" / "example.toml")
        bad = deepcopy(cfg)
        bad["runtime"]["chains"] = 8
        with self.assertRaisesRegex(ValueError, "runtime.chains is not used"):
            validate_config(bad)

    def test_rejects_nonpositive_stage_chains(self):
        cfg = load_config(Path(__file__).parents[1] / "configs" / "example.toml")
        bad = deepcopy(cfg)
        bad["chains"]["topo"] = 0
        with self.assertRaisesRegex(ValueError, "chains.topo"):
            validate_config(bad)

    def test_rejects_chains_in_old_scattered_locations(self):
        cfg = load_config(Path(__file__).parents[1] / "configs" / "example.toml")
        for path in ("pilot", "topo"):
            with self.subTest(path=path):
                bad = deepcopy(cfg)
                target = bad["pilot"] if path == "pilot" else bad["steps"][path]
                target["chains"] = 8
                with self.assertRaisesRegex(ValueError, r"set chains\."):
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
