import json
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

from cpn_renorm.chain_tuning import Workload, _target_sweeps, tune_pilot, tune_stage
from cpn_renorm.config import load_config, mod_config, section, validate_config
from cpn_renorm.patches import _batch_key
from cpn_renorm.pilot import run_pilot
from cpn_renorm.pipeline import _measure_point, run_pipeline
from cpn_renorm.sampler import Couplings
from cpn_renorm.workspace import affected_stages


ROOT = Path(__file__).parents[1]


class AutoChainsTests(unittest.TestCase):
    def config(self):
        cfg = load_config(ROOT / "configs/example.toml")
        cfg["runtime"].update(device="cpu", auto_chains=False)
        return cfg

    def test_new_format_and_rejected_old_fields(self):
        for filename in ("example.toml", "N2_L4_b1.0_mod1.toml"):
            cfg = load_config(ROOT / "configs" / filename)
            self.assertEqual(cfg["model"]["mod"], [1])
            self.assertTrue(cfg["runtime"]["auto_chains"])
            self.assertEqual(len(cfg["chains"]), 7)
        for mods in (1, [], [0, 0], [2], [True], [0.0]):
            cfg = self.config()
            cfg["model"]["mod"] = mods
            with self.assertRaisesRegex(ValueError, "model.mod"):
                validate_config(cfg)
        for old in ("observable", "topo"):
            cfg = self.config()
            cfg["chains"][old] = 8
            with self.assertRaisesRegex(ValueError, "obsolete"):
                validate_config(cfg)
        cfg = self.config()
        cfg["runtime"]["auto_chains"] = "true"
        with self.assertRaisesRegex(ValueError, "boolean"):
            validate_config(cfg)

    def test_independent_manual_measurement_widths(self):
        cfg = mod_config(self.config(), 1)
        cfg["chains"].update(observable_fine=3, observable_coarse=7,
                             topo_fine=5, topo_coarse=9)
        with tempfile.TemporaryDirectory() as folder, patch(
                "cpn_renorm.pipeline._point_sampler") as sampler, patch(
                "cpn_renorm.pipeline.measure_ensemble", return_value={}), patch(
                "cpn_renorm.pipeline._apply_point_diagnostics"), patch(
                "cpn_renorm.pipeline.ensemble_summary", return_value="done"), patch(
                "cpn_renorm.chain_tuning.tune_stage") as tune:
            for tag in ("obs_fine", "obs_coarse", "topo_fine", "topo_coarse_corr"):
                stage = "topo" if tag.startswith("topo") else "observable"
                _measure_point(cfg, Path(folder), section(cfg, stage), tag=tag,
                               L=4, beta=1.0, beta1=0.0, alpha=0.0,
                               alpha1=0.0, kind="half")
            self.assertEqual([call.args[1]["chains"] for call in sampler.call_args_list],
                             [3, 7, 5, 9])
            tune.assert_not_called()

    def test_automatic_measurement_widths_and_once_per_layer(self):
        cfg = mod_config(self.config(), 1)
        cfg["runtime"]["auto_chains"] = True
        cfg["_tuning_prefix"] = "z_mod1/"
        recommendation = {"observable_fine": 3, "observable_coarse": 7,
                          "topo_fine": 5, "topo_coarse": 9}

        def benchmark(_cfg, workloads, configured, **kwargs):
            name = workloads[0].name
            stage = "topo" if name.startswith("topo") else "observable"
            key = f"{stage}_{'fine' if '_fine' in name else 'coarse'}"
            return {"recommended_chains": recommendation[key], "candidates": []}

        with tempfile.TemporaryDirectory() as folder, patch(
                "cpn_renorm.pipeline._point_sampler") as sampler, patch(
                "cpn_renorm.pipeline.measure_ensemble", return_value={}), patch(
                "cpn_renorm.pipeline._apply_point_diagnostics"), patch(
                "cpn_renorm.pipeline.ensemble_summary", return_value="done"), patch(
                "cpn_renorm.chain_tuning._benchmark_stage", side_effect=benchmark) as tune:
            for tag in ("obs_fine", "obs_coarse", "topo_fine", "topo_coarse_magsus",
                        "topo_coarse_corr"):
                stage = "topo" if tag.startswith("topo") else "observable"
                _measure_point(cfg, Path(folder), section(cfg, stage), tag=tag,
                               L=4, beta=1.0, beta1=0.0, alpha=0.0,
                               alpha1=0.0, kind="half")
            self.assertEqual(tune.call_count, 4)
            self.assertEqual([call.args[1]["chains"] for call in sampler.call_args_list],
                             [3, 7, 5, 9, 9])
            self.assertEqual(set(cfg["_tuning_records"]),
                             {f"z_mod1/{key}" for key in recommendation})

    def test_no_safe_candidate_fails(self):
        cfg = mod_config(self.config(), 1)
        workload = Workload("fine", 4, 4, True, "half", Couplings(1.0, 0.0, 0.0), 10)
        with patch("cpn_renorm.chain_tuning._benchmark_stage", return_value={
                "recommended_chains": None, "candidates": []}):
            with self.assertRaisesRegex(RuntimeError, "no safe chains"):
                tune_stage(cfg, "observable_fine", [workload])
        self.assertIsNone(cfg["_tuning_records"]["observable_fine"]["recommended_chains"])

    def test_pilot_projection_replaces_configured_measurement_sweeps(self):
        cfg = mod_config(self.config(), 1)
        with patch("cpn_renorm.chain_tuning.tune_stage", return_value=8) as tune:
            tune_pilot(cfg, 16)
        workloads = tune.call_args.args[2]
        targets, _ = _target_sweeps(workloads, 8)
        settings = section(cfg, "pilot")
        expected = settings["adapt_steps"] + settings["max_warmup"]
        expected += (settings["max_meas_total"] + 7) // 8
        self.assertEqual(targets, [expected])

    def test_cross_mod_cache_keys_follow_physics(self):
        cfg = self.config()
        geometry = {"padding": 0, "factor": 4, "L_fine": 8, "L_coarse": 2}
        for stage in ("one_plaq", "two_plaq"):
            self.assertEqual(_batch_key(mod_config(cfg, 0), geometry, stage, 0),
                             _batch_key(mod_config(cfg, 1), geometry, stage, 0))
            cfg["model"]["beta1"] = 0.1
            self.assertNotEqual(_batch_key(mod_config(cfg, 0), geometry, stage, 0),
                                _batch_key(mod_config(cfg, 1), geometry, stage, 0))
            cfg["model"]["beta1"] = 0.0
        actions = affected_stages([{"path": "model.mod"}], beta1=0.0)
        self.assertEqual(actions["pilot"], "reuse")
        self.assertEqual(actions["two_plaq"], "reuse")

    def test_pilot_retunes_each_growing_size(self):
        cfg = mod_config(self.config(), 1)
        cfg["runtime"]["auto_chains"] = True
        cfg["pilot"].update(initial_L=4, max_L=8, min_L_over_xi=3)
        measurements = [{"xi": 2.0, "xi_err": 0.0}, {"xi": 1.0, "xi_err": 0.0}]

        def tune(config, size):
            config["chains"]["pilot"] = size
            return size

        with tempfile.TemporaryDirectory() as folder, patch(
                "cpn_renorm.chain_tuning.tune_pilot", side_effect=tune) as tuner, patch(
                "cpn_renorm.pilot.fine_sampler") as sampler, patch(
                "cpn_renorm.pilot.measure_ensemble", side_effect=measurements), patch(
                "cpn_renorm.pilot.ensemble_summary", return_value="done"):
            result = run_pilot(cfg, folder)
        self.assertEqual([call.args[1] for call in tuner.call_args_list], [4, 8])
        self.assertEqual([call.args[2]["chains"] for call in sampler.call_args_list], [4, 8])
        self.assertEqual(result["chains"], 8)
        self.assertEqual([attempt["chains"] for attempt in result["attempts"]], [4, 8])

    def test_auto_pipeline_persists_selected_chains_and_skips_topo(self):
        cfg = self.config()
        cfg["runtime"]["auto_chains"] = True
        geometry = {"padding": 0, "factor": 4, "L_fine": 8, "L_coarse": 2}
        pilot = {"xi": 1.0, "xi_err": 0.1, "geometry": geometry,
                 "attempts": [{"L": 8, "valid": True}]}
        two = {"fits": {"z": {"beta": 0.6, "alpha_eff": 0.2}}}
        one = {"fits": {"z": {"alpha": 0.1}}}
        match = {"value": 0.3, "error": 0.01, "bracketed": True}
        scan = {"matches": {"magsus": match, "corr": match}}

        def benchmark(_cfg, workloads, configured, **kwargs):
            value = 11 if workloads[0].name == "outer_boundary" else 13
            return {"recommended_chains": value, "candidates": []}

        def beta_scan(config, *_args, **_kwargs):
            from cpn_renorm.chain_tuning import tune_stage
            for key in ("observable_fine", "observable_coarse"):
                tune_stage(config, key, [Workload(key, 4, 4, True, "half", Couplings(1., 0., 0.), 20)])
            return scan

        with tempfile.TemporaryDirectory() as folder, patch(
                "cpn_renorm.pipeline.run_pilot", return_value=pilot), patch(
                "cpn_renorm.pipeline._run_two_plaq", return_value=two), patch(
                "cpn_renorm.pipeline._run_one_plaq", return_value=one), patch(
                "cpn_renorm.pipeline._run_beta1_scan", side_effect=beta_scan), patch(
                "cpn_renorm.chain_tuning._benchmark_stage", side_effect=benchmark), patch(
                "cpn_renorm.pipeline._run_topo") as topo:
            summary = run_pipeline(cfg, folder, skip_topo=True)
            report = json.loads((Path(folder) / "example/chain_tuning.json").read_text())
            resolved = json.loads((Path(folder) / "example/resolved.json").read_text())
        topo.assert_not_called()
        self.assertEqual(summary["chains"]["z_mod1"]["two_plaq"], 11)
        self.assertEqual(summary["chains"]["z_mod1"]["observable_fine"], 13)
        self.assertEqual(summary["chains"], report["chains"])
        self.assertEqual(summary["chains"], resolved["chains"])
        self.assertFalse(any("topo" in key for key in report["stages"]))

    def test_type_mod_product_reuse_precision_and_errors(self):
        for beta1, expected_common_calls in ((0.0, 1), (0.1, 2)):
            with self.subTest(beta1=beta1):
                cfg = self.config()
                cfg["model"].update(mod=[0, 1], beta1=beta1, alpha=0.2)
                cfg["renormalization"]["type"] = ["z", "U"]
                geometry = {"padding": 0, "factor": 4, "L_fine": 8, "L_coarse": 2}
                pilot = {"xi": 1.0, "xi_err": 0.1, "geometry": geometry}
                fit2 = {"beta": 0.6123456789012345, "alpha_eff": 0.2123456789012345,
                        "beta_err": 0.02, "alpha_eff_err": 0.03}
                two = {"fits": {name: fit2 for name in ("z", "U")}}
                one = {"fits": {name: {"alpha": 0.1, "alpha_err": 0.04}
                                for name in ("z", "U")}}
                match = {"value": 0.3, "error": 0.01, "bracketed": True}
                scan = {"matches": {"magsus": match, "corr": match}}
                with tempfile.TemporaryDirectory() as folder, patch(
                        "cpn_renorm.pipeline.run_pilot", return_value=pilot) as p, patch(
                        "cpn_renorm.pipeline._run_two_plaq", return_value=two) as t, patch(
                        "cpn_renorm.pipeline._run_one_plaq", return_value=one) as o, patch(
                        "cpn_renorm.pipeline._run_beta1_scan", return_value=scan) as b, patch(
                        "cpn_renorm.pipeline._run_topo", return_value=scan) as a:
                    summary = run_pipeline(cfg, folder)
                    self.assertEqual(p.call_count, expected_common_calls)
                    self.assertEqual(t.call_count, expected_common_calls)
                    self.assertEqual(o.call_count, expected_common_calls)
                    self.assertEqual(b.call_count, 4)
                    self.assertEqual(a.call_count, 4)
                    self.assertEqual(set(summary["branches"]),
                                     {"z_mod0", "z_mod1", "U_mod0", "U_mod1"})
                    for call in b.call_args_list:
                        self.assertEqual(call.args[3:5], (fit2["beta"], fit2["alpha_eff"]))
                    for name, result in summary["branches"].items():
                        path = Path(folder) / "example/branches" / name / "summary.json"
                        self.assertEqual(json.loads(path.read_text())["mod"], result["mod"])
                        couplings = result["renormalized_couplings"]
                        self.assertEqual(couplings["beta_c_err"], 0.02)
                        self.assertEqual(couplings["alpha_eff_c_err"], 0.03)
                        self.assertEqual(couplings["alpha_c_1plaq_err"], 0.04)
                        fragment = tomllib.loads(result["renorm_as_fine"]["magsus_1plaq"])
                        self.assertEqual(fragment["beta"], fit2["beta"])
                    resolved = json.loads((Path(folder) / "example/resolved.json").read_text())
                    self.assertEqual(resolved["chains"], summary["chains"])


if __name__ == "__main__":
    unittest.main()
