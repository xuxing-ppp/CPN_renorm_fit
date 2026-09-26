from pathlib import Path
import unittest

import torch

from cpn_renorm.chain_tuning import (
    Workload,
    _host_gib,
    _toml_suggestion,
    _workloads,
    chain_candidates,
    recommend_chains,
)
from cpn_renorm.cli import _parser
from cpn_renorm.config import load_config
from cpn_renorm.sampler import Couplings


class ChainTuningTests(unittest.TestCase):
    def test_candidates_include_non_power_of_two_limit(self):
        self.assertEqual(chain_candidates(20), [1, 2, 4, 8, 16, 20])

    def test_recommendation_uses_smallest_near_peak_safe_value(self):
        rows = [
            {"chains": 8, "safe": True, "chain_sweeps_per_s": 80.0},
            {"chains": 16, "safe": True, "chain_sweeps_per_s": 95.0},
            {"chains": 32, "safe": True, "chain_sweeps_per_s": 100.0},
            {"chains": 64, "safe": False, "chain_sweeps_per_s": 120.0},
        ]
        self.assertEqual(recommend_chains(rows), 16)

    def test_recommendation_handles_no_safe_candidate(self):
        self.assertIsNone(recommend_chains([
            {"chains": 1, "safe": False, "reason": "out_of_memory"}
        ]))

    def test_host_estimate_scales_with_chains(self):
        workload = Workload("fine", 32, 32, True, "half",
                            Couplings(1.0, 0.0, 0.0), 100, 50, True)
        self.assertAlmostEqual(_host_gib(workload, 16, torch.float64),
                               2 * _host_gib(workload, 8, torch.float64))

    def test_cli_exposes_tuning_limits(self):
        args = _parser().parse_args([
            "tune-chains", "case.toml", "--max-chains", "64",
            "--max-host-memory-gb", "2.5",
        ])
        self.assertEqual(args.command, "tune-chains")
        self.assertEqual(args.config, Path("case.toml"))
        self.assertEqual(args.max_chains, 64)
        self.assertEqual(args.max_host_memory_gb, 2.5)

    def test_workloads_use_pilot_geometry_and_stage_shapes(self):
        cfg = load_config(Path(__file__).parents[1] / "configs" / "example.toml")
        pilot = {"attempts": [{"L": 32, "valid": True}],
                 "geometry": {"factor": 4, "padding": 3,
                              "L_fine": 16, "L_coarse": 4}}
        stages = _workloads(cfg, pilot)
        self.assertEqual(stages["pilot"][0].Lx, 32)
        self.assertEqual((stages["two_plaq"][0].Lx,
                          stages["two_plaq"][0].Ly), (15, 11))
        self.assertTrue(stages["two_plaq"][0].batched_ensembles)
        self.assertEqual(stages["two_plaq"][0].ensembles, 5000)
        self.assertEqual(stages["observable"][0].Lx, 16)
        self.assertIn("topo", stages)

    def test_toml_suggestion_contains_only_available_recommendations(self):
        report = {"stages": {
            "pilot": {"recommended_chains": 8},
            "observable": {"recommended_chains": 16},
            "topo": {"recommended_chains": None},
        }}
        suggestion = _toml_suggestion(report)
        self.assertIn("[pilot]\nchains = 8", suggestion)
        self.assertIn("[steps.observable]\nchains = 16", suggestion)
        self.assertNotIn("steps.topo", suggestion)


if __name__ == "__main__":
    unittest.main()
