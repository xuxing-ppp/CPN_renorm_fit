import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cpn_renorm.config import load_config
from cpn_renorm.pipeline import (_censored_xi_bracket, _dynamic_fit_spacing,
                                 _grid_significantly_brackets,
                                 _remove_scan_ensembles, _run_beta1_scan,
                                 _run_one_plaq, _run_topo, _run_two_plaq,
                                 _scan_decision, _uniform_fit_grid, run_pipeline)
import numpy as np
from cpn_renorm.workspace import config_fingerprint


ROOT = Path(__file__).parents[1]


class PipelineWorkspaceTests(unittest.TestCase):
    @staticmethod
    def _patch_raw() -> dict:
        return {
            "X_U": np.array([[1.0]]), "freq_U": np.array([0.4]),
            "X_z": np.array([[2.0]]), "freq_z": np.array([0.5]),
            "X_s": np.array([[3.0]]), "freq_s": np.array([0.6]),
            "accept_rate": np.array([0.8]), "s_accept_rate": np.array([0.7]),
            "requested_fit_times": 1, "effective_fit_times": 1,
            "chains": 1, "batch_count": 1, "epsilon": np.array([0.05]),
            "outer_warmup_sweeps": np.array([2]),
            "outer_warmup_tau": np.array([[1.0]]),
            "outer_warmup_stop_reason": np.array(["tau_stable"]),
            "warmup_sweeps": np.array([0]), "warmup_tau": np.array([[1.0]]),
            "warmup_stop_reason": np.array(["copied_equilibrium"]),
            "inner_initialization": np.array(["copied_equilibrium"]),
            "inner_adapt_steps": np.array([2]),
            "sampling_minimum": 2, "sampling_actual": np.array([2]),
            "sampling_total_actual": 2, "sampling_maximum": 4,
            "sampling_target_ess": 1.0, "sampling_tau": np.array([[[0.5]]]),
            "sampling_ess": np.array([[[2.0]]]),
            "sampling_stop_reason": np.array(["target_ess_per_boundary"]),
        }

    def test_common_plaquette_stages_fit_both_definitions(self):
        cfg = load_config(ROOT / "configs" / "example.toml")
        cfg["runtime"]["device"] = "cpu"
        cfg["model"]["alpha"] = 0.2
        geometry = {"padding": 0, "L_fine": 8, "L_coarse": 2, "factor": 4}
        raw = self._patch_raw()

        def fake_fit(operation, _request, *, X, **_arrays):
            marker = float(X[0, 0])
            return ({"beta": marker, "alpha_eff": marker / 10}
                    if operation == "two_plaq" else {"alpha": marker})

        with tempfile.TemporaryDirectory(dir=ROOT) as folder, patch(
                "cpn_renorm.pipeline.generate_two_plaq", return_value=raw), patch(
                "cpn_renorm.pipeline.generate_one_plaq", return_value=raw), patch(
                "cpn_renorm.pipeline.isolated_fit", side_effect=fake_fit) as fit:
            two = _run_two_plaq(cfg, Path(folder), geometry)
            one = _run_one_plaq(cfg, Path(folder), geometry)
        self.assertEqual(set(two["fits"]), {"U", "z"})
        self.assertEqual(two["fits"]["U"]["beta"], 1.0)
        self.assertEqual(two["fits"]["z"]["beta"], 2.0)
        self.assertEqual(one["fits"]["U"]["alpha"], 3.0)
        self.assertEqual(one["fits"]["z"]["alpha"], 2.0)
        self.assertEqual(fit.call_count, 4)

    def test_pipeline_uses_stable_workspace_and_reports_config_fingerprint(self):
        cfg = load_config(ROOT / "configs" / "example.toml")
        cfg["runtime"]["device"] = "cpu"
        geometry = {"padding": 5, "L_fine": 28, "L_coarse": 7, "factor": 4}
        pilot = {"xi": 2.0, "xi_err": 0.1, "geometry": geometry}
        step1 = {"fits": {"z": {"beta": 0.6, "alpha_eff": 0.2},
                          "U": {"beta": 0.7, "alpha_eff": 0.25}}}
        match = {"value": 0.3, "bracketed": True}
        step2 = {"matches": {"magsus": match, "corr": match}}
        step3 = {"fits": {"z": {"alpha": 0.1}, "U": {"alpha": None}}}

        with tempfile.TemporaryDirectory(dir=ROOT) as folder, patch(
                "cpn_renorm.pipeline.run_pilot", return_value=pilot) as run_pilot, patch(
                "cpn_renorm.pipeline._run_two_plaq", return_value=step1), patch(
                "cpn_renorm.pipeline._run_beta1_scan", return_value=step2), patch(
                "cpn_renorm.pipeline._run_one_plaq", return_value=step3):
            summary = run_pipeline(cfg, Path(folder), skip_topo=True)
            expected = Path(folder).resolve() / "example"
            self.assertEqual(run_pilot.call_args.args[1], expected)
            self.assertEqual(summary["run_id"], "example")
            self.assertEqual(summary["config_fingerprint"], config_fingerprint(cfg))
            branch = summary["branches"]["z"]
            self.assertIn("beta = 0.6", branch["renorm_as_fine"]["magsus_1plaq"])
            self.assertEqual(branch["fit_points"]["beta1"]["magsus"], [])
            self.assertTrue((expected / "input.toml").exists())
            self.assertTrue((expected / "summary.json").exists())
            self.assertTrue((expected / "branches" / "z" / "summary.json").exists())

    def test_dual_type_pipeline_runs_common_stages_once_and_two_branches(self):
        cfg = load_config(ROOT / "configs" / "example.toml")
        cfg["runtime"]["device"] = "cpu"
        cfg["renormalization"]["type"] = ["z", "U"]
        cfg["model"]["alpha"] = 0.2
        geometry = {"padding": 0, "L_fine": 8, "L_coarse": 2, "factor": 4}
        pilot = {"xi": 1.0, "xi_err": 0.1, "geometry": geometry}
        two = {"fits": {"z": {"beta": 0.6, "alpha_eff": 0.2},
                        "U": {"beta": 0.7, "alpha_eff": 0.3}}}
        one = {"fits": {"z": {"alpha": 0.1}, "U": {"alpha": 0.15}}}
        match = {"value": 0.3, "error": 0.01, "bracketed": True}
        scan = {"matches": {"magsus": match, "corr": match}}
        topo = {"matches": {"magsus": match, "corr": match}}
        with tempfile.TemporaryDirectory(dir=ROOT) as folder, patch(
                "cpn_renorm.pipeline.run_pilot", return_value=pilot), patch(
                "cpn_renorm.pipeline._run_two_plaq", return_value=two) as run_two, patch(
                "cpn_renorm.pipeline._run_one_plaq", return_value=one) as run_one, patch(
                "cpn_renorm.pipeline._run_beta1_scan", return_value=scan) as run_beta, patch(
                "cpn_renorm.pipeline._run_topo", return_value=topo) as run_topo:
            summary = run_pipeline(cfg, Path(folder))
        run_two.assert_called_once()
        run_one.assert_called_once()
        self.assertEqual(run_beta.call_count, 2)
        self.assertEqual(run_topo.call_count, 2)
        self.assertEqual(set(summary["branches"]), {"z", "U"})
        branch_dirs = {call.args[1].name for call in run_beta.call_args_list}
        self.assertEqual(branch_dirs, {"z", "U"})

    def test_undefined_xi_creates_censored_refinement_bracket(self):
        xs = np.array([-1.5, -0.7, -0.3])
        ys = np.array([np.nan, 5.1, 11.2])
        bracket = _censored_xi_bracket(xs, ys, 2.3)
        self.assertEqual(bracket, [-1.5, -0.7])
        candidate, reason = _scan_decision(
            {"bracketed": False}, tolerance=0.01, max_refine=10,
            refine_count=0, censored=bracket)
        self.assertAlmostEqual(candidate, -1.1)
        self.assertEqual(reason, "refine")

    def test_dynamic_spacing_and_uniform_grid_respect_bounds(self):
        cfg = load_config(ROOT / "configs" / "example.toml")
        scan = cfg["scan"]["beta1"]
        self.assertEqual(_dynamic_fit_spacing(None, scan), 0.01)
        self.assertAlmostEqual(_dynamic_fit_spacing(0.01, scan), 0.03)
        self.assertEqual(_dynamic_fit_spacing(1.0, scan), 0.05)
        grid = _uniform_fit_grid(-7.99, 0.03, scan)
        self.assertEqual(len(grid), 5)
        self.assertGreaterEqual(grid[0], scan["min"])
        self.assertTrue(np.allclose(np.diff(grid), 0.03))

    def test_grid_requires_error_separated_inner_neighbors(self):
        accepted, _ = _grid_significantly_brackets(
            np.array([0.0, 0.2, 0.5, 0.8, 1.0]), np.full(5, 0.02),
            0.5, 0.02, True, 2.0)
        rejected, reason = _grid_significantly_brackets(
            np.array([0.4, 0.48, 0.5, 0.52, 0.6]), np.full(5, 0.02),
            0.5, 0.02, True, 2.0)
        self.assertTrue(accepted)
        self.assertFalse(rejected)
        self.assertEqual(reason, "insufficient_separation")

    def test_cleanup_only_removes_resolved_ensemble_folders(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            run_dir = Path(folder)
            discard = run_dir / "ensembles" / "discard"
            keep = run_dir / "ensembles" / "keep"
            discard.mkdir(parents=True)
            keep.mkdir(parents=True)
            (discard / "data.npz").write_bytes(b"data")
            removed = _remove_scan_ensembles(run_dir, [str(discard)])
            self.assertEqual(removed, ["discard"])
            self.assertFalse(discard.exists())
            self.assertTrue(keep.exists())
            with self.assertRaisesRegex(RuntimeError, "outside"):
                _remove_scan_ensembles(run_dir, [str(run_dir / "summary.json")])

    def test_beta1_scan_refines_across_undefined_xi(self):
        cfg = load_config(ROOT / "configs" / "example.toml")
        cfg["runtime"]["device"] = "cpu"
        cfg["scan"]["beta1"].update({
            "min": -2.0, "max": 1.0, "max_rounds": 6,
            "refine_tolerance": 0.05, "max_refine_rounds": 8,
        })
        geometry = {"padding": 1, "L_fine": 8, "L_coarse": 8, "factor": 1}

        def fake_measure(_cfg, _run_dir, _settings, *, tag, beta1, **_kwargs):
            if tag == "obs_fine":
                xi, magsus = 1.0, 0.8
            else:
                xi = None if beta1 < -1.0 else 4.0 * (beta1 + 1.0)
                magsus = beta1 + 2.0
            return {"xi": xi, "xi_err": None if xi is None else 0.02,
                    "magsus": magsus, "magsus_err": 0.02,
                    "mode_tau": [1.0] * 3, "mode_ess": [100.0] * 3,
                    "sampling": {"stop_reason": "target_ess"}}

        with tempfile.TemporaryDirectory(dir=ROOT) as folder, patch(
                "cpn_renorm.pipeline._measure_point",
                side_effect=fake_measure) as measure_mock, patch(
                "cpn_renorm.pipeline.isolated_plot") as plot_mock:
            run_dir = Path(folder)
            old_plot = run_dir / "step2_observable" / "beta1_scan.png"
            old_plot.parent.mkdir(parents=True)
            old_plot.write_bytes(b"old combined plot")
            result = _run_beta1_scan(cfg, run_dir, geometry, 1.0, 0.1)
            self.assertEqual(
                [call.args[0].name for call in plot_mock.call_args_list],
                ["beta1_scan_magsus.png", "beta1_scan_corr.png"])
            self.assertEqual(
                [set(call.kwargs["series"]) for call in plot_mock.call_args_list],
                [{"magsus_scaled"}, {"xi_scaled"}])
            self.assertFalse(old_plot.exists())

            measurement_count = measure_mock.call_count
            plot_mock.reset_mock()
            old_plot.write_bytes(b"old combined plot")
            cached = _run_beta1_scan(cfg, run_dir, geometry, 1.0, 0.1)
            self.assertEqual(measure_mock.call_count, measurement_count)
            self.assertEqual(len(plot_mock.call_args_list), 2)
            self.assertFalse(old_plot.exists())
            self.assertEqual(cached["matches"], result["matches"])
        self.assertTrue(result["matches"]["corr"]["bracketed"])
        self.assertAlmostEqual(result["matches"]["corr"]["value"], -0.75,
                               delta=0.08)
        self.assertLessEqual(
            result["matches"]["corr"]["raw_bracket"][1] -
            result["matches"]["corr"]["raw_bracket"][0], 0.05)
        self.assertLessEqual(len(result["matches"]["corr"]["fit_points"]), 5)
        grid = result["matches"]["corr"]["fit_grid"]
        self.assertEqual(len(grid), 5)
        self.assertTrue(np.allclose(np.diff(grid), result["matches"]["corr"]["fit_spacing"]))
        self.assertEqual(result["matches"]["corr"]["fit_points"], grid)

    def test_alpha_scan_uses_uniform_decreasing_fit_grid(self):
        cfg = load_config(ROOT / "configs" / "example.toml")
        cfg["runtime"]["device"] = "cpu"
        geometry = {"padding": 1, "L_fine": 8, "L_coarse": 8, "factor": 1}

        def fake_measure(_cfg, _run_dir, _settings, *, tag, alpha, **_kwargs):
            if tag == "topo_fine":
                return {"topology": {"Q_z": {"topo_sus": 0.5, "err": 0.005,
                                                "tau": 1.0, "ess": 100.0}},
                        "sampling": {"stop_reason": "target_ess"}}
            value = 1.0 - 5.0 * alpha
            return {"topology": {"Q_s": {"topo_sus": value, "err": 0.005,
                                            "tau": 1.0, "ess": 100.0}},
                    "sampling": {"stop_reason": "target_ess"}}

        beta_matches = {"magsus": {"value": -1.0, "bracketed": True},
                        "corr": {"value": -0.8, "bracketed": True}}
        with tempfile.TemporaryDirectory(dir=ROOT) as folder, patch(
                "cpn_renorm.pipeline._measure_point", side_effect=fake_measure), patch(
                "cpn_renorm.pipeline.isolated_plot"):
            result = _run_topo(cfg, Path(folder), geometry, 1.0, 0.2,
                               beta_matches, 0.1)
        for method in ("magsus", "corr"):
            match = result["matches"][method]
            self.assertTrue(match["bracketed"])
            self.assertEqual(match["monotonic"], "decreasing")
            self.assertEqual(match["fit_points"], match["fit_grid"])
            self.assertTrue(np.allclose(np.diff(match["fit_grid"]),
                                        match["fit_spacing"]))
