import io
import json
import tempfile
import unittest
from contextlib import chdir, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from cpn_renorm.chain_tuning import run_chain_tuning
from cpn_renorm.cli import main
from cpn_renorm.config import load_config, resolve_config_path
from cpn_renorm.workspace import workspace_dir


EXAMPLE = (Path(__file__).parents[1] / "configs/example.toml").read_text()


class ConfigPathTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.enterContext(chdir(self.root))
        self.source = self.write_config("configs/folder0/folder1/1.toml")

    def write_config(self, path):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(EXAMPLE)
        return target

    def test_unique_filename_and_explicit_paths(self):
        for path in ("1.toml", "folder0/folder1/1.toml",
                     "configs/folder0/folder1/1.toml", self.source):
            with self.subTest(path=path):
                self.assertEqual(resolve_config_path(path), self.source)

    def test_duplicate_names_require_path_even_for_top_level_match(self):
        self.write_config("configs/1.toml")
        with self.assertRaises(ValueError) as caught:
            resolve_config_path("1.toml")
        self.assertIn("folder0/folder1/1.toml", str(caught.exception))
        self.assertIn("ambiguous", str(caught.exception))
        self.assertEqual(resolve_config_path(self.source), self.source)

    def test_missing_filename_and_missing_root(self):
        for path in ("missing.toml", "absent/missing.toml"):
            with self.subTest(path=path), self.assertRaises(FileNotFoundError):
                resolve_config_path(path)
        with chdir(self.source.parent), self.assertRaises(FileNotFoundError):
            resolve_config_path("missing.toml")

    def test_explicit_existing_path_has_priority(self):
        external = self.write_config("folder0/folder1/1.toml")
        self.assertEqual(resolve_config_path("folder0/folder1/1.toml"), external)

    def test_workspace_mapping(self):
        for source, relative in (
            (self.source, "folder0/folder1/1"),
            (self.write_config("configs/1.toml"), "1"),
            (self.write_config("configs/other/1.toml"), "other/1"),
            (self.write_config("external/1.toml"), "1"),
        ):
            for output in ("runs", "custom"):
                with self.subTest(source=source, output=output):
                    self.assertEqual(workspace_dir(load_config(source), output),
                                     self.root / output / relative)

    def test_single_command_entrypoints(self):
        expected = self.root / "custom/folder0/folder1/1"
        with patch("cpn_renorm.cli.run_pipeline", return_value={}) as run, \
                patch("cpn_renorm.cli.run_pilot", return_value={}) as pilot, \
                patch("cpn_renorm.cli.run_chain_tuning", return_value={}) as tune, \
                patch("cpn_renorm.cli._print_tuning") as display:
            for command in ("validate", "run", "pilot", "tune-chains"):
                with self.subTest(command=command), redirect_stdout(io.StringIO()) as out:
                    main([command, "1.toml", "--device", "cpu",
                          "--output-dir", "custom"])
                    if command == "validate":
                        self.assertEqual(json.loads(out.getvalue())["config"],
                                         str(self.source))
            self.assertEqual(run.call_args.args[0]["_config_path"], str(self.source))
            self.assertEqual(pilot.call_args.args[1], expected)
            self.assertEqual(tune.call_args.args[0]["_config_path"], str(self.source))
            self.assertEqual(display.call_args.args[1], expected / "chain_tuning.json")
        state = json.loads((expected / "run_state.json").read_text())
        self.assertEqual(state["status"], "complete")

    def test_cli_search_error_does_not_start_simulation(self):
        self.write_config("configs/other/1.toml")
        with patch("cpn_renorm.cli.run_pipeline") as run:
            with self.assertRaisesRegex(SystemExit, "ambiguous"):
                main(["run", "1.toml"])
            with self.assertRaisesRegex(SystemExit, "not found"):
                main(["run", "missing.toml"])
            run.assert_not_called()
        self.assertFalse((self.root / "runs").exists())

    def test_batch_same_names_have_independent_workspaces(self):
        self.write_config("configs/other/1.toml")
        with patch("cpn_renorm.cli.run_pipeline", return_value={}) as run, \
                redirect_stdout(io.StringIO()):
            main(["batch", "configs/**/*.toml", "--device", "cpu"])
        self.assertEqual(run.call_count, 2)
        for relative in ("folder0/folder1/1", "other/1"):
            directory = self.root / "runs" / relative
            self.assertTrue((directory / "input.toml").is_file())
            state = json.loads((directory / "run_state.json").read_text())
            self.assertEqual(state["status"], "complete")

    def test_tuning_writes_report_to_mapped_workspace(self):
        cfg = load_config(self.source)
        cfg["runtime"]["device"] = "cpu"
        with patch("cpn_renorm.chain_tuning.run_pilot",
                   return_value={"geometry": {}}), \
                patch("cpn_renorm.chain_tuning._workloads", return_value={}):
            run_chain_tuning(cfg, "runs")
        self.assertTrue((self.root / "runs/folder0/folder1/1/chain_tuning.json").is_file())


if __name__ == "__main__":
    unittest.main()
