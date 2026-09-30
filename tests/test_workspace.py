import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from cpn_renorm.cli import _approve_changes, _parser
from cpn_renorm.config import load_config
from cpn_renorm.workspace import (affected_stages, inspect_workspace,
                                  workspace_dir, write_workspace_state)


ROOT = Path(__file__).parents[1]


class WorkspaceTests(unittest.TestCase):
    def test_commands_accept_noninteractive_change_flag(self):
        for command in ("pilot", "run", "tune-chains"):
            with self.subTest(command=command):
                args = _parser().parse_args(
                    [command, "case.toml", "--accept-changes"])
                self.assertTrue(args.accept_changes)
        args = _parser().parse_args(
            ["batch", "configs/*.toml", "--accept-changes"])
        self.assertTrue(args.accept_changes)

    def test_scan_change_reuses_upstream_stages(self):
        changes = [{"path": "scan.beta1.min", "old": -1.0, "new": -2.0}]
        self.assertEqual(affected_stages(changes), {
            "pilot": "reuse", "two_plaq": "reuse",
            "observable": "reprocess", "one_plaq": "reuse",
            "topology": "reprocess",
        })

    def test_adding_renormalization_type_reuses_common_stages(self):
        changes = [{"path": "renormalization.type", "old": ["z"],
                    "new": ["z", "U"]}]
        self.assertEqual(affected_stages(changes), {
            "pilot": "reuse", "two_plaq": "reuse",
            "observable": "resimulate", "one_plaq": "reuse",
            "topology": "resimulate",
        })

    def test_stage_settings_invalidate_only_dependents(self):
        changes = [{"path": "steps.one_plaq.max_warmup", "old": 10, "new": 20}]
        actions = affected_stages(changes)
        self.assertEqual(actions["one_plaq"], "resimulate")
        self.assertEqual(actions["topology"], "resimulate")
        self.assertEqual(actions["two_plaq"], "reuse")
        self.assertEqual(actions["observable"], "reuse")

    def test_workspace_tracks_effective_config_and_source(self):
        cfg = load_config(ROOT / "configs" / "example.toml")
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            output = Path(folder) / "runs"
            first = inspect_workspace(cfg, output)
            self.assertFalse(first.needs_confirmation)
            self.assertEqual(first.run_dir, output.resolve() / "example")
            write_workspace_state(first, command="pilot", status="complete")

            changed_cfg = deepcopy(cfg)
            changed_cfg["scan"]["beta1"]["min"] = -7.5
            changed = inspect_workspace(changed_cfg, output)
            self.assertTrue(changed.needs_confirmation)
            self.assertEqual([item["path"] for item in changed.changes],
                             ["scan.beta1.min"])
            self.assertEqual(changed.actions["pilot"], "reuse")
            self.assertEqual(changed.actions["observable"], "reprocess")

    def test_source_change_requires_confirmation_even_when_values_match(self):
        cfg = load_config(ROOT / "configs" / "example.toml")
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            folder = Path(folder)
            output = folder / "runs"
            first = inspect_workspace(cfg, output)
            write_workspace_state(first, command="run", status="complete")
            alias = folder / "example.toml"
            alias.write_text(Path(cfg["_config_path"]).read_text(encoding="utf-8"),
                             encoding="utf-8")
            same = load_config(alias)
            second = inspect_workspace(same, output)
            self.assertTrue(second.source_changed)
            self.assertFalse(second.changes)
            self.assertTrue(second.needs_confirmation)

    def test_noninteractive_change_requires_explicit_acceptance(self):
        cfg = load_config(ROOT / "configs" / "example.toml")
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            output = Path(folder) / "runs"
            first = inspect_workspace(cfg, output)
            write_workspace_state(first, command="run", status="complete")
            cfg["scan"]["alpha"]["max"] = 9.0
            changed = inspect_workspace(cfg, output)
            with patch("cpn_renorm.cli.sys.stdin.isatty", return_value=False):
                with self.assertRaisesRegex(SystemExit, "--accept-changes"):
                    _approve_changes([changed], accepted=False)
            _approve_changes([changed], accepted=True)

    def test_corrupt_workspace_state_is_not_silently_overwritten(self):
        cfg = load_config(ROOT / "configs" / "example.toml")
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            run_dir = workspace_dir(cfg, Path(folder) / "runs")
            run_dir.mkdir(parents=True)
            (run_dir / "run_state.json").write_text("not json", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "cannot read workspace state"):
                inspect_workspace(cfg, Path(folder) / "runs")


if __name__ == "__main__":
    unittest.main()
