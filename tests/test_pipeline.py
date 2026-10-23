import tempfile
import tomllib
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from cpn_renorm.config import load_config as _load_config


def load_config(path):
    cfg = _load_config(path)
    cfg["runtime"]["auto_chains"] = False
    return cfg
from cpn_renorm.pipeline import (_censored_xi_G_bracket, _dynamic_fit_spacing,
                                 _finish_topo, _two_raw_key,
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


class TopoRefitTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(ROOT / "configs" / "example.toml")
        self.cfg["runtime"]["device"] = "cpu"
        self.cfg["model"].update(mod=[0, 1], alpha=.1)
        self.cfg["renormalization"]["type"] = ["z", "U"]
        self.geometry = {"padding": 0, "L_fine": 8, "L_coarse": 2, "factor": 4}
        self.two = {"fingerprint": "initial-fit", "fits": {
            tag: {"old": {"alpha_eff": .25}, "new": {
                "beta": .6, "alpha": .1, "alpha1": .2, "mse": .01,
                "beta_err": .02, "alpha1_err": .03}}
            for tag in ("z", "U")}}
        self.one = {"fits": {tag: {"alpha": .1, "alpha_err": .01} for tag in ("z", "U")}}
        self.topo = {"matches": {method: {"value": alpha, "error": .015,
                                        "bracketed": True}
                                 for method, alpha in (("magsus", .2), ("corrG", .4))}}
        self.common = {"two_plaq": self.two, "raw": raw_placeholder(), "raw_source": "current_fit"}

    @staticmethod
    def fake_fit(operation, request, **_arrays):
        if operation == "two_mse":
            return {"mse": .02, "normalization_nodes": 512}
        alpha = request["alpha"]
        return {"new": {"beta": .6 + alpha, "alpha": alpha, "alpha1": .2 - alpha,
                        "mse": .005, "success": True, "beta_err": .004,
                        "alpha1_err": .006}}

    def finish(self, folder, topo=None):
        _finish_topo(self.cfg, Path(folder) / "branch", Path(folder), self.geometry,
                     self.common, self.topo if topo is None else topo, "z", .1)

    def test_each_method_refits_and_resume_needs_no_raw(self):
        from cpn_renorm.cache import read_json
        with tempfile.TemporaryDirectory() as folder, patch(
                "cpn_renorm.pipeline.isolated_fit", side_effect=self.fake_fit) as fit, patch(
                "cpn_renorm.pipeline.generate_two_plaq") as generate:
            self.finish(folder)
            self.assertEqual(fit.call_count, 4)
            self.assertEqual([c.args[1]["alpha"] for c in fit.call_args_list
                              if c.args[0] == "two_fixed"], [.2, .4])
            stored = read_json(Path(folder) / "branch/step4_topo/result.json")
            self.common["raw"] = None
            fit.reset_mock()
            self.finish(folder, stored)
            fit.assert_not_called()
            generate.assert_not_called()
            # Changing only one matched alpha refits only that method.
            self.common["raw"] = raw_placeholder()
            stored["matches"]["corrG"]["value"] = .5
            self.finish(folder, stored)
            self.assertEqual(fit.call_count, 2)

    def test_invalidates_on_raw_settings_type_device_and_initial_fit(self):
        with tempfile.TemporaryDirectory() as folder, patch(
                "cpn_renorm.pipeline.isolated_fit", side_effect=self.fake_fit) as fit:
            self.finish(folder)
            for change in (lambda: self.cfg["steps"]["two_plaq"].update(bins=64),
                           lambda: self.cfg["steps"]["two_plaq"]["p0"].update(z=[1., .1]),
                           lambda: self.two.update(fingerprint="replacement-fit"),
                           lambda: self.cfg["runtime"].update(device="cuda")):
                change()
                fit.reset_mock()
                with patch("cpn_renorm.pipeline.resolve_device", return_value=self.cfg["runtime"]["device"]):
                    self.finish(folder)
                self.assertEqual(fit.call_count, 4)
            fit.reset_mock()
            with patch("cpn_renorm.pipeline.resolve_device", return_value="cuda"):
                _finish_topo(self.cfg, Path(folder) / "branch", Path(folder), self.geometry,
                             self.common, self.topo, "U", .1)
            self.assertEqual(fit.call_count, 4)

    def test_loads_only_compatible_persisted_data_and_regenerates_once(self):
        from cpn_renorm.storage import atomic_npz
        for compatible in (True, False):
            with self.subTest(compatible=compatible), tempfile.TemporaryDirectory() as folder:
                self.common["raw"] = None
                topo = deepcopy(self.topo)
                path = Path(folder) / "step2_2plaq/measurements.npz"
                raw = raw_placeholder()
                atomic_npz(path, **raw, two_plaq_raw_key=np.asarray(
                    _two_raw_key(self.cfg, self.geometry) if compatible else "stale"))
                with patch("cpn_renorm.pipeline.isolated_fit", side_effect=self.fake_fit) as fit, patch(
                        "cpn_renorm.pipeline.generate_two_plaq", return_value=raw) as generate:
                    self.finish(folder, topo)
                self.assertEqual(generate.call_count, 0 if compatible else 1)
                self.assertEqual(fit.call_count, 4 if compatible else 6)
                source = "persisted_measurements" if compatible else "regenerated_measurements"
                for match in topo["matches"].values():
                    self.assertEqual(match["two_plaq_refit"]["data_source"], source)
                    self.assertEqual(match["two_plaq_mse_before"], .01 if compatible else .02)
                self.assertTrue(path.exists())  # Cleanup belongs to the successful full pipeline.

    def test_failed_or_missing_topo_does_not_fit(self):
        with tempfile.TemporaryDirectory() as folder, patch(
                "cpn_renorm.pipeline.isolated_fit") as fit, patch(
                "cpn_renorm.pipeline.generate_two_plaq") as generate:
            self.finish(folder, {"matches": {}})
            self.finish(folder, {"matches": {"magsus": {"bracketed": False, "value": None}}})
            fit.assert_not_called()
            generate.assert_not_called()

    def test_refit_failure_is_not_published_as_success(self):
        def fail(operation, request, **arrays):
            result = self.fake_fit(operation, request, **arrays)
            if operation == "two_fixed":
                result["new"]["success"] = False
            return result
        with tempfile.TemporaryDirectory() as folder, patch(
                "cpn_renorm.pipeline.isolated_fit", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "topo 2plaq refit failed"):
                self.finish(folder)
            self.assertNotIn("two_plaq_refit", self.topo["matches"]["magsus"])
            self.assertFalse((Path(folder) / "branch/step4_topo/result.json").exists())

    def test_initial_fit_keeps_raw_until_refits_and_does_not_delete_existing_file(self):
        from cpn_renorm.storage import atomic_npz
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "step2_2plaq/measurements.npz"
            atomic_npz(path, marker=np.asarray("existing"))
            raw = PipelineWorkspaceTests._patch_raw()
            with patch("cpn_renorm.pipeline.generate_two_plaq", return_value=raw) as generate, patch(
                    "cpn_renorm.pipeline.isolated_fit", side_effect=self.fake_fit):
                two = _run_two_plaq(self.cfg, root, self.geometry, self.one)
                self.assertTrue(path.exists())
                self.common.update(two_plaq=two, raw=self.cfg.pop("_two_raw"))
                self.finish(folder)
                generate.assert_called_once()
            self.assertIs(self.common["raw"], raw)
            with np.load(path) as stored:
                self.assertEqual(str(stored["marker"]), "existing")

    def test_pipeline_shares_data_and_exports_compact_summary(self):
        scan = {"matches": {method: {"value": .3, "error": .01, "bracketed": True}
                             for method in ("magsus", "corrG")}}
        for beta1 in (0., .1):
            for save_npz in (False, True):
                with self.subTest(beta1=beta1, save_npz=save_npz), tempfile.TemporaryDirectory() as folder:
                    self.cfg["model"]["beta1"] = beta1
                    self.cfg["storage"]["save_npz"] = save_npz
                    # An existing stale aggregate survives until successful completion.
                    from cpn_renorm.storage import atomic_npz
                    root = Path(folder) / "example"
                    common_dirs = [root] if beta1 == 0 else [root / "common" / f"mod{m}" for m in (0, 1)]
                    for common_dir in common_dirs:
                        atomic_npz(common_dir / "step2_2plaq/measurements.npz", stale=np.asarray(True))
                    with patch("cpn_renorm.pipeline.run_pilot", return_value={"geometry": self.geometry}), patch(
                            "cpn_renorm.pipeline._run_one_plaq", return_value=self.one), patch(
                            "cpn_renorm.pipeline._run_two_plaq", return_value=self.two), patch(
                            "cpn_renorm.pipeline._run_beta1_scan", return_value=scan) as beta_scan, patch(
                            "cpn_renorm.pipeline._run_topo", side_effect=lambda *a, **kw: deepcopy(self.topo)) as topo_scan, patch(
                            "cpn_renorm.pipeline.generate_two_plaq", side_effect=lambda *a, **kw: raw_placeholder()) as generate, patch(
                            "cpn_renorm.pipeline.isolated_fit", side_effect=self.fake_fit):
                        summary = run_pipeline(self.cfg, folder)
                    self.assertEqual(generate.call_count, 1 if beta1 == 0 else 2)
                    self.assertEqual(beta_scan.call_count, 4)
                    self.assertEqual(topo_scan.call_count, 4)
                    self.assertEqual(len(summary["branches"]), 4)
                    for branch in summary["branches"].values():
                        comparison = branch["two_plaq_comparison"]
                        self.assertEqual(comparison["old_alpha_eff"], .25)
                        self.assertAlmostEqual(comparison["new_alpha_plus_alpha1"], .3)
                        self.assertAlmostEqual(comparison["difference"], .05)
                        for method, alpha in (("magsus", .2), ("corrG", .4)):
                            original = branch["results"][method + "_1plaq"]
                            final = branch["results"][method + "_obs_fit_topo"]
                            self.assertNotIn("topo_correction", original)
                            self.assertEqual(final["errors"]["beta"], .004)
                            self.assertEqual(final["errors"]["alpha1"], .006)
                            self.assertEqual(final["error_scope"], "beta_and_alpha1_conditional_on_topo_alpha")
                            self.assertEqual(set(final["topo_correction"]), {"alpha", "beta", "alpha1"})
                            for name in ("alpha", "beta", "alpha1"):
                                self.assertAlmostEqual(final["topo_correction"][name],
                                    final["couplings"][name] - original["couplings"][name])
                            self.assertAlmostEqual(final["couplings"]["alpha"], alpha)
                            self.assertAlmostEqual(final["couplings"]["beta"], .6 + alpha)
                            self.assertEqual(tomllib.loads(branch["renorm_as_fine"][method + "_obs_fit_topo"]),
                                             final["couplings"])
                    files = list(Path(folder).rglob("measurements.npz"))
                    self.assertEqual(len(files), (1 if beta1 == 0 else 2) if save_npz else 0)


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
                "cpn_renorm.pipeline.isolated_fit", side_effect=lambda op, req, **kw:
                {"new": {"beta": .65, "alpha": req["alpha"], "alpha1": .22,
                         "success": True, "mse": .009}} if op == "two_fixed" else {"mse": .01}):
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
