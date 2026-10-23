from pathlib import Path
from copy import deepcopy
import unittest

from cpn_renorm.config import DEFAULTS, load_config, resolved_geometry, section, validate_config


class ConfigTests(unittest.TestCase):
    def test_pilot_growth_defaults_and_validation(self):
        self.assertEqual(DEFAULTS["pilot"]["max_L"], 128)
        self.assertEqual(DEFAULTS["pilot"]["pilot_growth_margin"], 1.25)
        cfg = load_config(Path(__file__).parents[1] / "configs/example.toml")
        self.assertEqual(cfg["pilot"]["max_L"], 128)
        self.assertEqual(cfg["pilot"]["pilot_growth_margin"], 1.25)
        for margin in (1, 1.5):
            cfg["pilot"]["pilot_growth_margin"] = margin
            validate_config(cfg)
            self.assertEqual(cfg["pilot"]["pilot_growth_margin"], float(margin))
        for margin in (0.9, float("nan"), float("inf"), -float("inf")):
            cfg["pilot"]["pilot_growth_margin"] = margin
            with self.subTest(margin=margin), self.assertRaisesRegex(ValueError, "finite and at least 1"):
                validate_config(cfg)
        for margin in (None, "invalid"):
            cfg["pilot"]["pilot_growth_margin"] = margin
            with self.subTest(margin=margin), self.assertRaisesRegex(TypeError, "must be a number"):
                validate_config(cfg)

    def test_old_pilot_growth_factor_is_rejected_but_scan_growth_is_valid(self):
        cfg = load_config(Path(__file__).parents[1] / "configs/example.toml")
        self.assertIn("growth_factor", cfg["scan"]["beta1"])
        self.assertIn("growth_factor", cfg["scan"]["alpha"])
        cfg["pilot"]["growth_factor"] = 2.0
        with self.assertRaisesRegex(ValueError, "replace it with pilot.pilot_growth_margin"):
            validate_config(cfg)

    def test_example_config_and_geometry(self):
        cfg = load_config(Path(__file__).parents[1] / "configs" / "example.toml")
        self.assertNotIn("alpha_methods", DEFAULTS["renormalization"])
        self.assertNotIn("alpha_methods", cfg["renormalization"])
        geo = resolved_geometry(cfg, 2.01)
        self.assertEqual(geo["padding"], 5)
        self.assertEqual(geo["L_coarse"], 10)
        self.assertEqual(geo["L_fine"] % cfg["renormalization"]["factor"], 0)
        self.assertEqual(geo["L_coarse"] * geo["factor"], geo["L_fine"])

    def test_old_length_config_names_are_rejected(self):
        for section_name, key in (("pilot", "min_L_over_xi"),
                                  ("geometry", "padding_xi_mul"),
                                  ("geometry", "obs_L_xi_mul")):
            cfg = load_config(Path(__file__).parents[1] / "configs/example.toml")
            cfg[section_name][key] = 1.
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "rename it to"):
                validate_config(cfg)

    def test_geometry_minus_one_uses_xi_G_rules(self):
        cfg = load_config(Path(__file__).parents[1] / "configs" / "example.toml")
        cfg["geometry"]["padding"] = -1
        cfg["geometry"]["coarse_L"] = -1
        cfg["geometry"]["coarse_L_min"] = -1
        cfg["geometry"]["obs_L_xi_G_mul"] = 10.0
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
        self.assertEqual(cfg["steps"]["two_plaq"]["min_meas_per_boundary"], 4096)
        self.assertEqual(cfg["steps"]["one_plaq"]["fit_times"], 3000)
        self.assertEqual(cfg["steps"]["one_plaq"]["min_meas_per_boundary"], 2048)
        self.assertEqual(cfg["scan"]["beta1"]["refine_tolerance"], 0.02)
        self.assertEqual(cfg["scan"]["alpha"]["refine_tolerance"], 0.002)
        self.assertEqual(cfg["scan"]["beta1"]["fit_points"], 5)
        self.assertEqual(cfg["scan"]["beta1"]["fit_spacing_min"], 0.01)
        self.assertEqual(DEFAULTS["scan"]["alpha"]["fit_spacing_max"], 0.006)

    def test_fit_spacing_growth_defaults_overrides_and_validation(self):
        cfg = load_config(Path(__file__).parents[1] / "configs/example.toml")
        for kind in ("beta1", "alpha"):
            self.assertEqual(DEFAULTS["scan"][kind]["fit_spacing_growth_factor"], 2.0)
            self.assertEqual(cfg["scan"][kind]["fit_spacing_growth_factor"], 2.0)
            cfg["scan"][kind]["fit_spacing_growth_factor"] = 1.5
        validate_config(cfg)
        for kind in ("beta1", "alpha"):
            for value in (0, 1, -2, float("nan"), float("inf")):
                bad = deepcopy(cfg)
                bad["scan"][kind]["fit_spacing_growth_factor"] = value
                with self.subTest(kind=kind, value=value), self.assertRaisesRegex(ValueError, "fit-grid"):
                    validate_config(bad)

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
            "pilot": 32, "two_plaq": 1024, "observable_fine": 64,
            "observable_coarse": 64, "one_plaq": 2048,
            "topo_fine": 128, "topo_coarse": 128,
        })
        self.assertNotIn("chains", cfg["pilot"])
        for name, settings in cfg["steps"].items():
            self.assertNotIn("chains", settings)
            key = f"{name}_coarse" if name in ("observable", "topo") else name
            self.assertEqual(section(cfg, name)["chains"], cfg["chains"][key])

    def test_rejects_obsolete_runtime_chains(self):
        cfg = load_config(Path(__file__).parents[1] / "configs" / "example.toml")
        bad = deepcopy(cfg)
        bad["runtime"]["chains"] = 8
        with self.assertRaisesRegex(ValueError, "runtime.chains is not used"):
            validate_config(bad)

    def test_rejects_nonpositive_stage_chains(self):
        cfg = load_config(Path(__file__).parents[1] / "configs" / "example.toml")
        bad = deepcopy(cfg)
        bad["chains"]["topo_fine"] = 0
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

    def test_alpha_topo_fit_is_a_boolean(self):
        cfg = load_config(Path(__file__).parents[1] / 'configs/example.toml')
        self.assertIs(DEFAULTS['renormalization']['alpha_topo_fit'], True)
        for enabled in (False, True):
            cfg['renormalization']['alpha_topo_fit'] = enabled
            validate_config(cfg)
        for invalid in (0, 1, 'true', 'false', None, []):
            cfg['renormalization']['alpha_topo_fit'] = invalid
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, 'alpha_topo_fit must be a boolean'):
                validate_config(cfg)
