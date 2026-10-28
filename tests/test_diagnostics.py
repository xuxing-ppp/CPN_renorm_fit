import tempfile
import unittest
from pathlib import Path

from cpn_renorm.diagnostics import collect_diagnostics, measurement_warnings, refresh_summary
from cpn_renorm.storage import atomic_json


class DiagnosticsTests(unittest.TestCase):
    def test_collects_active_sources_and_preserves_deleted_measurement_warnings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = {"model": {"beta1": 0, "mod": [0]},
                       "renormalization": {"beta1_methods": ["corrG"], "alpha_topo_fit": True},
                       "branches": {"z_mod0": {"results": {}}}}
            cfg = {"diagnostics": {"min_accept": .5, "max_accept": .95}}
            atomic_json(root / "summary.json", summary)
            atomic_json(root / "resolved.json", {"config": cfg})
            atomic_json(root / "pilot/result.json", {"attempts": [{
                "accept_rate": .1, "warmup": {"stop_reason": "max_warmup"}}]})
            branch = root / "branches/z_mod0"
            ensemble = branch / "ensembles/used"
            point = {"_folder": str(ensemble),
                     "diagnostics": {"warnings": ["low ESS"]}}
            saved = measurement_warnings(branch, {0: point})
            self.assertEqual(measurement_warnings(branch, {0: point}, saved), saved)
            atomic_json(ensemble / "result.json", {"diagnostics": point["diagnostics"]})
            atomic_json(branch / "step3_observable/result.json", {
                "measurements": {"0": {"path": "ensembles/used/result.json"}},
                "measurement_warnings": saved,
                "matches": {"corrG": {"bracketed": False, "reason": "scan limit"},
                            "magsus": {"bracketed": False, "reason": "unused"}}})
            atomic_json(branch / "ensembles/stale/result.json", {"diagnostics": {"warnings": ["stale"]}})
            atomic_json(root / "old/pilot/result.json", {"diagnostics": {"warnings": ["old"]}})
            atomic_json(root / "step1_1plaq/result.json", {"fits": {
                "z": {"success": False, "message": "iteration limit"},
                "U": {"success": False, "message": "unused fit"}}})
            first = collect_diagnostics(root, summary, cfg)
            self.assertFalse(first["valid"])
            self.assertEqual(first["warning_count"], 5)
            self.assertEqual(sum(w["message"] == "low ESS" for w in first["warnings"]), 1)
            (ensemble / "result.json").unlink()
            self.assertEqual(refresh_summary(root), first)
            self.assertEqual(refresh_summary(root), first)

    def test_empty_report_is_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = {"model": {"beta1": 1, "mod": [0, 1]},
                       "renormalization": {"beta1_methods": [], "alpha_topo_fit": False},
                       "branches": {}}
            self.assertEqual(collect_diagnostics(Path(tmp), summary, {}),
                             {"valid": True, "warning_count": 0, "warnings": []})

    def test_independent_common_stages_and_unavailable_topo(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = {"model": {"beta1": .1, "alpha": 0, "mod": [0, 1]},
                       "renormalization": {"beta1_methods": ["corrG"], "alpha_topo_fit": True},
                       "branches": {"U_mod0": {}, "U_mod1": {}}}
            for mod in (0, 1):
                atomic_json(root / f"common/mod{mod}/step2_2plaq/result.json",
                            {"diagnostics": {"warnings": ["patch limit"]}})
                atomic_json(root / f"branches/U_mod{mod}/step4_topo/result.json",
                            {"matches": {"corrG": {"bracketed": False, "reason": "stale topo"}}})
            report = collect_diagnostics(root, summary, {})
            self.assertEqual(report["warning_count"], 2)
            self.assertTrue(all(w["source"].startswith("common/") for w in report["warnings"]))
