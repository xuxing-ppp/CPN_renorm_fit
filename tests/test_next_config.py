import io
import json
from contextlib import chdir, redirect_stdout
from copy import deepcopy
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import patch

from cpn_renorm.config import load_config
from cpn_renorm.next_config import generate_next_config


PROJECT = Path(__file__).resolve().parents[1]


class NextConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / "runs/category/source"
        self.run.mkdir(parents=True)
        self.cfg = load_config(PROJECT / "configs/example.toml")
        self.cfg.pop("_config_path")
        self.cfg["runtime"]["seed"] = 2468
        self.summary = {"model": {"N": 2}, "branches": {}}
        for kind in ("z", "U"):
            for mod in (0, 1):
                results = {}
                for method in ("magsus", "corrG"):
                    for alpha in ("1plaq", "obs_fit_topo"):
                        # Distinct final beta/alpha1 ensure the entire selected
                        # result is carried forward, rather than only alpha.
                        value = (1 + mod + (kind == "U") * 2 +
                                 (method == "corrG") * 4 + (alpha == "obs_fit_topo") * 8)
                        results[f"{method}_{alpha}"] = {"couplings": {
                            "beta": value + .123456789012345,
                            "beta1": -value, "alpha": value / 100,
                            "alpha1": value / 10}}
                self.summary["branches"][f"{kind}_mod{mod}"] = {
                    "results": results, "renorm_as_fine": {"ignored": "not TOML"}}
        self.save()
        (self.run / "input.toml").write_text(
            (PROJECT / "configs/example.toml").read_text())

    def save(self):
        (self.run / "summary.json").write_text(json.dumps(self.summary))
        (self.run / "resolved.json").write_text(json.dumps({
            "config": self.cfg, "geometries": {"do_not_copy": True},
            "chains": {"z_mod0": {"pilot": 999}}}))

    def generate(self, **kwargs):
        args = dict(source="source", mod=0, source_type="z", beta_method="corrG",
                    alpha_method="topo", factor=2, output_dir="nested", filename="next.toml",
                    project_root=self.root)
        args.update(kwargs)
        with redirect_stdout(io.StringIO()):
            return generate_next_config(**args)

    def test_combinations_and_round_trip(self):
        for kind in ("z", "U"):
            for mod in (0, 1):
                for method in ("magsus", "corrG"):
                    for alpha in ("1plaq", "topo"):
                        with self.subTest(kind=kind, mod=mod, method=method, alpha=alpha):
                            path = self.generate(source_type=kind, mod=mod,
                                beta_method=method, alpha_method=alpha, overwrite=True, factor=3)
                            cfg = load_config(path)
                            suffix = "obs_fit_topo" if alpha == "topo" else "1plaq"
                            expected = self.summary["branches"][f"{kind}_mod{mod}"][
                                "results"][f"{method}_{suffix}"]["couplings"]
                            self.assertEqual(cfg["model"], {"N": 2, **expected, "mod": [mod]})
                            self.assertEqual(cfg["renormalization"], {
                                "type": ["U"], "factor": 3, "beta1_methods": [method],
                                "alpha_topo_fit": alpha == "topo"})
                            self.assertEqual(cfg["runtime"]["seed"], 2468)
                            self.assertEqual(cfg["chains"], self.cfg["chains"])
                            self.assertNotIn("geometries", cfg)
                            self.assertIn(f"Source branch: {kind}_mod{mod}", path.read_text())

    def test_overrides_and_path_resolution_from_other_cwd(self):
        with chdir("/tmp"):
            for source in ("source", self.run, "runs/category/source"):
                path = self.generate(source=source, overwrite=True, next_type="z",
                                     overrides={"runtime.device": "cpu", "chains.pilot": 8})
                self.assertEqual(path, self.root / "configs/nested/next.toml")
                cfg = load_config(path)
                self.assertEqual(cfg["renormalization"]["type"], ["z"])
                self.assertEqual(cfg["runtime"]["device"], "cpu")
                self.assertEqual(cfg["chains"]["pilot"], 8)
        absolute = self.root / "outside.toml"
        self.assertEqual(self.generate(output_dir=absolute.parent, filename=absolute.name), absolute)

    def test_default_and_custom_filenames(self):
        for filename in (None, ""):
            path = self.generate(filename=filename, overwrite=True)
            self.assertEqual(path.name, "source-L2_mod0_corrG_topo.toml")
        path = self.generate(filename=None, factor=4, mod=1,
                             beta_method="magsus", alpha_method="1plaq")
        self.assertEqual(path.name, "source-L4_mod1_magsus_1plaq.toml")
        for filename in ("custom", "custom.toml", "N2_L2_b1.0"):
            path = self.generate(filename=filename, overwrite=True)
            self.assertEqual(path.name, filename if filename.endswith(".toml") else filename + ".toml")
        dotted_run = self.run.with_name("N2_L2_b1.0")
        self.run.rename(dotted_run)
        path = self.generate(source=dotted_run, filename=None)
        self.assertEqual(path.name, "N2_L2_b1.0-L2_mod0_corrG_topo.toml")

    def test_input_fallback(self):
        for resolved in (None, {}):
            path = self.run / "resolved.json"
            if resolved is None:
                path.unlink()
            else:
                path.write_text(json.dumps(resolved))
            cfg = load_config(self.generate(overwrite=True))
            self.assertEqual(cfg["runtime"]["seed"],
                             load_config(self.run / "input.toml")["runtime"]["seed"])

    def test_output_protection_and_invalid_override(self):
        path = self.generate()
        before = path.read_bytes()
        with self.assertRaises(FileExistsError):
            self.generate()
        for key, value in (("model", {}), ("model.beta", 1),
                           ("renormalization", {}), ("renormalization.type", ["z"]),
                           ("renormalization.factor", 3),
                           ("renormalization.beta1_methods", ["magsus"]),
                           ("renormalization.alpha_topo_fit", False),
                           ("runtime.missing", 1), ("runtime", {}),
                           ("chains.pilot", 0)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.generate(overwrite=True, overrides={key: value})
        self.assertEqual(path.read_bytes(), before)
        with self.assertRaises(ValueError):
            self.generate(output_dir=self.run, filename="input.toml", overwrite=True)

    def test_ambiguous_missing_and_unavailable_results(self):
        duplicate = self.root / "runs/other/source"
        duplicate.mkdir(parents=True)
        (duplicate / "summary.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            self.generate()
        self.generate(source=self.run)
        with self.assertRaises(FileNotFoundError):
            self.generate(source="absent", overwrite=True)
        del self.summary["branches"]["z_mod0"]["results"]["corrG_obs_fit_topo"]
        self.save()
        with self.assertRaisesRegex(ValueError, "available:.*corrG_1plaq"):
            self.generate(source=self.run, overwrite=True)
        del self.summary["branches"]["z_mod0"]
        self.save()
        with self.assertRaisesRegex(ValueError, "available:.*U_mod0"):
            self.generate(source=self.run, overwrite=True)

    def test_invalid_couplings_and_parameters(self):
        original = deepcopy(self.summary)
        for value in (None, float("nan"), float("inf"), "1", True):
            self.summary = deepcopy(original)
            self.summary["branches"]["z_mod0"]["results"]["corrG_obs_fit_topo"][
                "couplings"]["beta"] = value
            self.save()
            with self.assertRaisesRegex(ValueError, "invalid coupling beta"):
                self.generate()
        self.summary = original
        self.save()
        for args in ({"mod": True}, {"source_type": "x"}, {"next_type": "x"},
                     {"factor": 2.5}, {"factor": 0}, {"beta_method": "x"},
                     {"alpha_method": "x"}, {"filename": "../next.toml"},
                     {"filename": "/tmp/next.toml"}, {"filename": "nested\\next"},
                     {"filename": ".."}, {"filename": " "}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.generate(**args)
        self.assertFalse((self.root / "configs/nested/next.toml").exists())

    def test_script_entrypoint_uses_its_own_root(self):
        with chdir("/tmp"), patch("cpn_renorm.next_config.generate_next_config") as generate:
            runpy.run_path(str(PROJECT / "configs/next_config.py"), run_name="__main__")
        args = generate.call_args.kwargs
        self.assertEqual(args["project_root"], PROJECT)
        self.assertEqual(args["source_type"], "z")
        self.assertEqual(args["next_type"], "U")


if __name__ == "__main__":
    unittest.main()
