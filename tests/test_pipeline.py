import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

from cpn_renorm.config import load_config as _load_config


def load_config(path):
    cfg = _load_config(path)
    cfg["runtime"]["auto_chains"] = False
    return cfg
from cpn_renorm.pipeline import (_censored_xi_G_bracket, _dynamic_fit_spacing,
                                 _grid_significantly_brackets,
                                 _remove_scan_ensembles, _run_beta1_scan,
                                 _run_one_plaq, _run_topo, _run_two_plaq,
                                 _scan_decision, _scan_fingerprint, _uniform_fit_grid,
                                 _uniform_grid_match, run_pipeline)
import numpy as np


ROOT = Path(__file__).parents[1]


def raw_placeholder():
    return {f"{key}_{tag}": np.ones((1,9)) if key == "X" else np.ones(1)
            for key in ("X", "freq") for tag in ("U", "z")}


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
        cfg["renormalization"]["type"] = ["z", "U"]
        geometry = {"padding": 0, "L_fine": 8, "L_coarse": 2, "factor": 4}
        raw = self._patch_raw()

        def fake_fit(operation, _request, *, X, **_arrays):
            marker = float(X[0, 0])
            return ({"new": {"beta": marker, "alpha1": marker / 10}}
                    if operation == "two_fixed" else {"alpha": marker})

        with tempfile.TemporaryDirectory(dir=ROOT) as folder, patch(
                "cpn_renorm.pipeline.generate_two_plaq", return_value=raw), patch(
                "cpn_renorm.pipeline.generate_one_plaq", return_value=raw), patch(
                "cpn_renorm.pipeline.isolated_fit", side_effect=fake_fit) as fit:
            one = _run_one_plaq(cfg, Path(folder), geometry)
            two = _run_two_plaq(cfg, Path(folder), geometry, one)
        self.assertEqual(set(two["fits"]), {"U", "z"})
        self.assertEqual(two["fits"]["U"]["new"]["beta"], 1.0)
        self.assertEqual(two["fits"]["z"]["new"]["beta"], 2.0)
        self.assertEqual(one["fits"]["U"]["alpha"], 3.0)
        self.assertEqual(one["fits"]["z"]["alpha"], 2.0)
        self.assertEqual(fit.call_count, 4)

    def test_pipeline_uses_stable_workspace_and_compact_summary(self):
        cfg = load_config(ROOT / "configs" / "example.toml")
        cfg["renormalization"]["alpha_topo_fit"] = False
        cfg["runtime"]["device"] = "cpu"
        geometry = {"padding": 5, "L_fine": 28, "L_coarse": 7, "factor": 4}
        pilot = {"xi_G": 2.0, "xi_G_err": 0.1, "geometry": geometry}
        step1 = {"fits": {"z": {"new": {"beta": 0.6, "alpha1": 0.2, "mse": .01, "normalization_nodes": 512}},
                          "U": {"new": {"beta": 0.7, "alpha1": 0.25}}}}
        match = {"value": 0.3, "bracketed": True}
        step2 = {"matches": {"magsus": match, "corrG": match}}
        step3 = {"fits": {"z": {"alpha": 0.1}, "U": {"alpha": None}}}

        with tempfile.TemporaryDirectory(dir=ROOT) as folder, patch(
                "cpn_renorm.pipeline.run_pilot", return_value=pilot) as run_pilot, patch(
                "cpn_renorm.pipeline._run_two_plaq", return_value=step1), patch(
                "cpn_renorm.pipeline._run_beta1_scan", return_value=step2), patch(
                "cpn_renorm.pipeline._run_one_plaq", return_value=step3), patch(
                "cpn_renorm.pipeline._run_topo", return_value={"matches": {}}) as topo:
            summary = run_pipeline(cfg, Path(folder))
            topo.assert_not_called()
            expected = Path(folder).resolve() / "example"
            self.assertEqual(run_pilot.call_args.args[1], expected)
            self.assertEqual(summary["model"], cfg["model"])
            self.assertNotIn("config_fingerprint", summary)
            branch = summary["branches"]["z_mod1"]
            self.assertEqual(branch["results"]["magsus_1plaq"]["couplings"]["beta"], 0.6)
            self.assertEqual(branch["results"]["magsus_1plaq"]["couplings"]["alpha"], 0.1)
            exported = tomllib.loads(branch["renorm_as_fine"]["magsus_1plaq"])
            self.assertEqual(exported, {"beta": .6, "beta1": .3, "alpha": .1, "alpha1": .2})
            self.assertEqual(set(branch["results"]["magsus_1plaq"]["couplings"]),
                             {"beta", "beta1", "alpha", "alpha1"})
            self.assertNotIn("fit_points", branch)
            self.assertTrue((expected / "input.toml").exists())
            self.assertTrue((expected / "summary.json").exists())
            self.assertFalse((expected / "branches" / "z_mod1" / "summary.json").exists())

    def test_dual_type_pipeline_runs_common_stages_once_and_two_branches(self):
        cfg = load_config(ROOT / "configs" / "example.toml")
        cfg["runtime"]["device"] = "cpu"
        cfg["renormalization"]["type"] = ["z", "U"]
        cfg["model"]["alpha"] = 0.2
        geometry = {"padding": 0, "L_fine": 8, "L_coarse": 2, "factor": 4}
        pilot = {"xi_G": 1.0, "xi_G_err": 0.1, "geometry": geometry}
        two = {"fits": {"z": {"new": {"beta": 0.6, "alpha1": 0.2, "mse": .01, "normalization_nodes": 512}},
                        "U": {"new": {"beta": 0.7, "alpha1": 0.3, "mse": .01, "normalization_nodes": 512}}}}
        one = {"fits": {"z": {"alpha": 0.1}, "U": {"alpha": 0.15}}}
        match = {"value": 0.3, "error": 0.01, "bracketed": True}
        scan = {"matches": {"magsus": match, "corrG": match}}
        topo = {"matches": {"magsus": match, "corrG": match}}
        with tempfile.TemporaryDirectory(dir=ROOT) as folder, patch(
                "cpn_renorm.pipeline.run_pilot", return_value=pilot), patch(
                "cpn_renorm.pipeline._run_two_plaq", return_value=two) as run_two, patch(
                "cpn_renorm.pipeline._run_one_plaq", return_value=one) as run_one, patch(
                "cpn_renorm.pipeline._run_beta1_scan", return_value=scan) as run_beta, patch(
                "cpn_renorm.pipeline._run_topo", return_value=topo) as run_topo, patch(
                "cpn_renorm.pipeline.generate_two_plaq", return_value=raw_placeholder()), patch(
                "cpn_renorm.pipeline.isolated_fit", return_value={"mse": .01}):
            summary = run_pipeline(cfg, Path(folder))
        run_two.assert_called_once()
        run_one.assert_called_once()
        self.assertEqual(run_beta.call_count, 2)
        self.assertEqual(run_topo.call_count, 2)
        self.assertEqual(set(summary["branches"]), {"z_mod1", "U_mod1"})
        branch_dirs = {call.args[1].name for call in run_beta.call_args_list}
        self.assertEqual(branch_dirs, {"z_mod1", "U_mod1"})

    def test_undefined_xi_G_creates_censored_refinement_bracket(self):
        xs = np.array([-1.5, -0.7, -0.3])
        ys = np.array([np.nan, 5.1, 11.2])
        bracket = _censored_xi_G_bracket(xs, ys, 2.3)
        self.assertEqual(bracket, [-1.5, -0.7])
        candidate, reason = _scan_decision(
            {"bracketed": False}, tolerance=0.01, max_refine=10,
            refine_count=0, censored=bracket)
        self.assertAlmostEqual(candidate, -1.1)
        self.assertEqual(reason, "refine")

    def test_fit_spacing_growth_preserves_legacy_cache_keys_at_default(self):
        cfg = load_config(ROOT / "configs/example.toml")
        for kind in ("beta1", "alpha"):
            scan = cfg["scan"][kind]
            scan.pop("fit_spacing_growth_factor")
            legacy = _scan_fingerprint(kind, cfg, {}, renorm_type="z")
            scan["fit_spacing_growth_factor"] = 2.0
            self.assertEqual(_scan_fingerprint(kind, cfg, {}, renorm_type="z"), legacy)
            scan["fit_spacing_growth_factor"] = 3.0
            self.assertNotEqual(_scan_fingerprint(kind, cfg, {}, renorm_type="z"), legacy)

    def test_uniform_grid_retries_use_configured_growth_and_cap(self):
        cfg = load_config(ROOT / "configs/example.toml")
        for kind in ("beta1", "alpha"):
            for factor in (1.5, 2.0, 3.0):
                scan = dict(cfg["scan"][kind])
                scan.update(min=-1., max=1., fit_spacing_min=0.01,
                            fit_spacing_max=0.05, fit_spacing_growth_factor=factor,
                            max_fit_grid_rounds=3)
                grids = []

                def reject(values, *_args):
                    grids.append(values.copy())
                    return False, "insufficient_separation"

                with patch("cpn_renorm.pipeline.isolated_match", return_value={
                        "bracketed": True, "value": 0., "error": 0.}), patch(
                        "cpn_renorm.pipeline._grid_significantly_brackets", side_effect=reject):
                    result = _uniform_grid_match(
                        measured={0.: {"value": 0.}},
                        measure=lambda x: {"value": x},
                        observable=lambda d: (d["value"], 0.01),
                        target=0., target_err=0.01, increasing=True,
                        scan=scan, bootstrap_seed=1)
                with self.subTest(kind=kind, factor=factor):
                    self.assertIsNone(result["value"])
                    self.assertEqual(len(grids), 4)
                    for index, grid in enumerate(grids):
                        self.assertTrue(np.allclose(np.diff(grid), min(0.01 * factor**index, 0.05)))

    def test_dynamic_spacing_and_uniform_grid_respect_bounds(self):
        cfg = load_config(ROOT / "configs" / "example.toml")
        scan = cfg["scan"]["beta1"]
        self.assertEqual(_dynamic_fit_spacing(None, scan), 0.01)
        self.assertAlmostEqual(_dynamic_fit_spacing(0.01, scan), 0.03)
        self.assertEqual(_dynamic_fit_spacing(1.0, scan), scan["fit_spacing_max"])
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

    def test_beta1_scan_refines_across_undefined_xi_G(self):
        cfg = load_config(ROOT / "configs" / "example.toml")
        cfg["runtime"]["device"] = "cpu"
        cfg["scan"]["beta1"].update({
            "min": -2.0, "max": 1.0, "max_rounds": 6,
            "refine_tolerance": 0.05, "max_refine_rounds": 8,
        })
        geometry = {"padding": 1, "L_fine": 8, "L_coarse": 8, "factor": 1}

        def fake_measure(_cfg, _run_dir, _settings, *, tag, beta1, **_kwargs):
            if tag == "obs_fine":
                xi_G, magsus = 1.0, 0.8
            else:
                xi_G = None if beta1 < -1.0 else 4.0 * (beta1 + 1.0)
                magsus = beta1 + 2.0
            return {"xi_G": xi_G, "xi_G_err": None if xi_G is None else 0.02,
                    "magsus": magsus, "magsus_err": 0.02,
                    "mode_tau": [1.0] * 3, "mode_ess": [100.0] * 3,
                    "sampling": {"stop_reason": "target_ess"}}

        with tempfile.TemporaryDirectory(dir=ROOT) as folder, patch(
                "cpn_renorm.pipeline._measure_point",
                side_effect=fake_measure) as measure_mock, patch(
                "cpn_renorm.pipeline.isolated_plot") as plot_mock:
            run_dir = Path(folder)
            old_plot = run_dir / "step3_observable" / "beta1_scan.png"
            old_plot.parent.mkdir(parents=True)
            old_plot.write_bytes(b"old combined plot")
            result = _run_beta1_scan(cfg, run_dir, geometry, 1.0, 0.1)
            self.assertEqual(
                [call.args[0].name for call in plot_mock.call_args_list],
                ["beta1_scan_magsus.png", "beta1_scan_corrG.png"])
            self.assertEqual(
                [set(call.kwargs["series"]) for call in plot_mock.call_args_list],
                [{"magsus_scaled"}, {"xi_G_scaled"}])
            self.assertFalse(old_plot.exists())

            measurement_count = measure_mock.call_count
            plot_mock.reset_mock()
            old_plot.write_bytes(b"old combined plot")
            cached = _run_beta1_scan(cfg, run_dir, geometry, 1.0, 0.1)
            self.assertEqual(measure_mock.call_count, measurement_count)
            self.assertEqual(len(plot_mock.call_args_list), 2)
            self.assertFalse(old_plot.exists())
            self.assertEqual(cached["matches"], result["matches"])
        self.assertTrue(result["matches"]["corrG"]["bracketed"])
        self.assertAlmostEqual(result["matches"]["corrG"]["value"], -0.75,
                               delta=0.08)
        self.assertLessEqual(
            result["matches"]["corrG"]["raw_bracket"][1] -
            result["matches"]["corrG"]["raw_bracket"][0], 0.05)
        self.assertLessEqual(len(result["matches"]["corrG"]["fit_points"]), 5)
        grid = result["matches"]["corrG"]["fit_grid"]
        self.assertEqual(len(grid), 5)
        self.assertTrue(np.allclose(np.diff(grid), result["matches"]["corrG"]["fit_spacing"]))
        self.assertEqual(result["matches"]["corrG"]["fit_points"], grid)

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
                        "corrG": {"value": -0.8, "bracketed": True}}
        with tempfile.TemporaryDirectory(dir=ROOT) as folder, patch(
                "cpn_renorm.pipeline._measure_point", side_effect=fake_measure), patch(
                "cpn_renorm.pipeline.isolated_plot"):
            result = _run_topo(cfg, Path(folder), geometry, 1.0, 0.2,
                               beta_matches, 0.1)
        for method in ("magsus", "corrG"):
            match = result["matches"][method]
            self.assertTrue(match["bracketed"])
            self.assertEqual(match["monotonic"], "decreasing")
            self.assertEqual(match["fit_points"], match["fit_grid"])
            self.assertTrue(np.allclose(np.diff(match["fit_grid"]),
                                        match["fit_spacing"]))
