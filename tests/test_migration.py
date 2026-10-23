from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from cpn_renorm.config import load_config, mod_config, section
from cpn_renorm.migrate import migrate_run, verify_run, _rename_legacy_corr, _rename_legacy_length
from cpn_renorm.pipeline import _run_beta1_scan, _run_topo
from cpn_renorm.storage import atomic_json
from cpn_renorm.cache import read_json

ROOT = Path(__file__).parents[1]


class MigrationTests(unittest.TestCase):
    def fixture(self, root):
        cfg = load_config(ROOT / "configs/example.toml")
        cfg["runtime"].update(auto_chains=False, device="cpu")
        cfg["model"].update(mod=[1], alpha=.2)
        cfg["renormalization"].update(type=["z"], factor=1)
        cfg["scan"]["beta1"].update(points=[-.1, .3], refine_tolerance=1.)
        cfg["scan"]["alpha"].update(points=[.1, .3], refine_tolerance=1.)
        root.mkdir()
        (root / "input.toml").write_text((ROOT / "configs/example.toml").read_text())
        geometry = {"padding": 0, "L_fine": 8, "L_coarse": 8, "factor": 1}
        two = {"beta": .6, "beta_err": .01, "alpha_eff": .2, "alpha_eff_err": .02}
        one = {"alpha": .1, "alpha_err": .01}
        estimates = {name: .3 for name in ("magsus", "corrG")}
        errors = {name: .01 for name in estimates}
        branch = root / "branches/z_mod1"
        original = {"beta_c": .6, "beta_c_err": .01, "alpha_eff_c": .2, "alpha_eff_c_err": .02, "beta1_c": estimates,
                    "beta1_c_err": errors, "alpha_c_1plaq": .1,
                    "alpha_c_1plaq_err": .01, "alpha_c_topo": estimates,
                    "alpha_c_topo_err": errors}
        summary = {"geometries": {"mod1": geometry}, "chains": {"z_mod1": cfg["chains"]},
                   "branches": {"z_mod1": {"renormalized_couplings": original}}}
        atomic_json(root / "summary.json", summary)
        atomic_json(branch / "summary.json", summary["branches"]["z_mod1"])
        cfg.pop("storage")
        cfg["renormalization"].pop("alpha_methods", None)
        cfg["renormalization"].pop("beta1_methods")
        atomic_json(root / "run_state.json", {"schema": 2, "config": cfg,
                    "source": str(ROOT / "configs/example.toml"), "command": "run", "status": "complete"})
        atomic_json(root / "resolved.json", {"config": cfg, "chains": summary["chains"]})
        atomic_json(root / "pilot/result.json", {"geometry": geometry, "xi_G": 1., "xi_G_err": .1})
        atomic_json(root / "step1_2plaq/result.json", {"cache_schema": 5, "fits": {"z": two}})
        atomic_json(root / "step3a_1plaq/result.json", {"cache_schema": 5, "fits": {"z": one}})
        scalar = mod_config(cfg, 1)
        observation = {"xi_G": 1., "xi_G_err": .01, "magsus": 2., "magsus_err": .01,
                       "sampling": {"stop_reason": "target_ess"}}
        topology = {"topology": {"Q_z": {"topo_sus": .5, "err": .01, "tau": 1., "ess": 100.},
                                  "Q_s": {"topo_sus": .5, "err": .01, "tau": 1., "ess": 100.}},
                    "sampling": {"stop_reason": "target_ess"}}
        for tag in ("obs_fine", "obs_coarse", "topo_fine", "topo_coarse_magsus", "topo_coarse_corrG"):
            payload = {"schema": 4, "tag": tag, "L": 8 if "fine" in tag else 8,
                       "beta": 1. if "fine" in tag else .6,
                       "beta1": 0. if "fine" in tag else .3,
                       "alpha": .3, "alpha1": -.1, "kind": "full" if tag.startswith("topo") else "half",
                       "N": scalar["model"]["N"], "mod": 1, "runtime": scalar["runtime"],
                       "settings": section(scalar, "topo" if tag.startswith("topo") else "observable",
                                           "fine" if "fine" in tag else "coarse")}
            if tag == "obs_coarse":
                payload.update(alpha=.2, alpha1=0.)
            if "fine" in tag:
                payload["alpha"] = float(scalar["model"]["alpha"])
                payload["alpha1"] = float(scalar["model"]["alpha1"]) if tag.startswith("topo") else 0.0
            folder = branch / "ensembles" / (tag + "_old")
            atomic_json(folder / "result.json", {"parameters": payload,
                        **(topology if tag.startswith("topo") else observation)})
            np.savez(folder / "data.npz", accept_rate=np.array([.8]))
        matches = {name: {"value": .3, "error": .01, "bracketed": True,
                          "fit_points": [.3], "fit_grid": [.3]} for name in estimates}
        atomic_json(branch / "step2_observable/result.json", {
            "step": "observable_beta1", "fine": observation, "matches": deepcopy(matches),
            "coarse": {str(x): observation for x in (-.1, .2, .3)}})
        for match in matches.values():
            match.update(beta1_c=.3, measurements={str(x): {"value": .5, "error": .01,
                "tau": 1., "ess": 100., "sampling": topology["sampling"]} for x in (.1, .2, .3)})
        atomic_json(branch / "step3b_topo/result.json", {"matches": matches, "fine_component": "Q_z"})
        return cfg, geometry

    def test_migration_preserves_results_and_reuses_without_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "case"
            old, geometry = self.fixture(root)
            report = migrate_run(root)
            self.assertGreater(report["recovered_scalar_points"], 0)
            verify_run(root)
            self.assertFalse(list(root.rglob("*.npz")))
            self.assertFalse((root / "branches/z_mod1/summary.json").exists())
            cfg = read_json(root / "resolved.json")["config"]
            scalar = mod_config(cfg, 1)
            branch = root / "branches/z_mod1"
            # Folded legacy observables cannot serve the full-action pipeline.
            with patch("cpn_renorm.pipeline._measure_point", side_effect=RuntimeError("resampling required")), patch(
                    "cpn_renorm.pipeline._plot_beta1_scan"):
                with self.assertRaisesRegex(RuntimeError, "resampling required"):
                    _run_beta1_scan(scalar, branch, geometry, .6, .2, alpha_c=.1)
            snapshot = {p.relative_to(root): p.read_bytes() for p in root.rglob("*.json")}
            migrate_run(root)
            self.assertEqual(snapshot, {p.relative_to(root): p.read_bytes() for p in root.rglob("*.json")})

    def test_legacy_second_moment_label_migrates_without_changing_correlations(self):
        source = {"corr": {"corr_tau": 1., "conn_PP_corr": [2., 3.]},
                  "corr_1plaq": "alpha_scan_corr.png", "topo_corr": "topo_coarse_corr"}
        renamed = _rename_legacy_corr(source)
        self.assertEqual(renamed["corrG"], {"corr_tau": 1., "conn_PP_corr": [2., 3.]})
        self.assertEqual(renamed["corrG_1plaq"], "alpha_scan_corrG.png")
        self.assertEqual(renamed["topo_corrG"], "topo_coarse_corrG")
        self.assertEqual(_rename_legacy_corr(renamed), renamed)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "case"
            self.fixture(root)
            for path in root.rglob("*.json"):
                path.write_text(path.read_text().replace("corrG", "corr").replace("xi_G", "xi"))
            migrate_run(root)
            self.assertEqual(read_json(root / "resolved.json")["config"]["renormalization"]["beta1_methods"],
                             ["magsus", "corrG"])
            self.assertIn("corrG_1plaq", read_json(root / "summary.json")["branches"]["z_mod1"]["results"])

    def test_second_moment_length_names_are_renamed_once(self):
        source = {"xi": 2., "xi_err": .1, "min_L_over_xi": 8.,
                  "geometry": {"padding_xi_mul": 3., "obs_L_xi_mul": 10.},
                  "existing": True, "maximum": 4, "description": "xi_G estimate"}
        renamed = _rename_legacy_length(source)
        self.assertEqual(renamed["xi_G"], 2.)
        self.assertEqual(renamed["xi_G_err"], .1)
        self.assertEqual(renamed["min_L_over_xi_G"], 8.)
        self.assertEqual(renamed["geometry"], {"padding_xi_G_mul": 3., "obs_L_xi_G_mul": 10.})
        self.assertEqual(renamed["existing"], True)
        self.assertEqual(renamed["maximum"], 4)
        self.assertEqual(_rename_legacy_length(renamed), renamed)

    def test_failed_staging_leaves_source_and_npz_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "case"
            self.fixture(root)
            before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
            with patch("cpn_renorm.migrate._migrate_copy", side_effect=ValueError("bad data")):
                with self.assertRaisesRegex(ValueError, "bad data"):
                    migrate_run(root)
            self.assertEqual(before, {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()})
