import math
from pathlib import Path
import unittest
from unittest.mock import call, patch

import torch

from cpn_renorm.chain_tuning import (
    Workload,
    _adaptive_chain_search,
    _benchmark_candidate,
    _benchmark_stage,
    _cuda_allocator_limit,
    _host_gib,
    _power_two_seed,
    _toml_suggestion,
    _tuning_summary,
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

    def test_cli_default_allows_automatic_expansion(self):
        args = _parser().parse_args(["tune-chains", "case.toml"])
        self.assertEqual(args.max_chains, 65536)

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
        self.assertIn("[chains]\npilot = 8", suggestion)
        self.assertIn("observable = 16", suggestion)
        self.assertNotIn("topo =", suggestion)

    def test_benchmark_reports_each_workload_to_progress(self):
        class RecordingProgress:
            def __init__(self):
                self.extensions = []
                self.updates = []

            def extend(self, count=1):
                self.extensions.append(count)

            def update(self, count=1, **metrics):
                self.updates.append((count, metrics))

        cfg = load_config(Path(__file__).parents[1] / "configs" / "example.toml")
        cfg["runtime"]["device"] = "cpu"
        workload = Workload("fine", 4, 4, True, "half",
                            Couplings(1.0, 0.0, 0.0), 10)
        progress = RecordingProgress()
        timing = {"name": "fine", "batch_seconds": 0.1,
                  "peak_gpu_gib": None}
        with patch("cpn_renorm.chain_tuning._time_workload", return_value=timing):
            row = _benchmark_candidate(
                cfg, [workload], 2, warmup_sweeps=1, timed_sweeps=1,
                gpu_memory_fraction=0.8, max_host_memory_gib=4.0,
                progress=progress)

        self.assertTrue(row["safe"])
        self.assertEqual(progress.extensions, [1])
        self.assertEqual(progress.updates, [
            (0, {"chains": 2, "workload": "fine"}), (1, {}),
        ])

    def test_preflight_rejects_short_series_without_timing(self):
        cfg = load_config(Path(__file__).parents[1] / "configs" / "example.toml")
        cfg["runtime"]["device"] = "cpu"
        workload = Workload("fine", 4, 4, True, "half",
                            Couplings(1.0, 0.0, 0.0), 10, 10, True, 10)
        with patch("cpn_renorm.chain_tuning._time_workload") as timing:
            row = _benchmark_candidate(
                cfg, [workload], 1, warmup_sweeps=1, timed_sweeps=1,
                gpu_memory_fraction=0.8, max_host_memory_gib=4.0)
        self.assertFalse(row["safe"])
        self.assertIn("per_chain_series_too_short", row["reason"])
        timing.assert_not_called()

    def test_seed_uses_nearest_power_of_two_within_limit(self):
        self.assertEqual(_power_two_seed(20, 65536), 16)
        self.assertEqual(_power_two_seed(100000, 3000), 2048)

    @staticmethod
    def _curve_evaluator(times, unsafe_at=None):
        def evaluate(chains):
            if unsafe_at is not None and chains >= unsafe_at:
                return {"chains": chains, "safe": False,
                        "reason": "gpu_memory_limit"}
            seconds = times(chains)
            return {"chains": chains, "safe": True,
                    "projected_target_seconds": seconds,
                    "chain_sweeps_per_s": chains / seconds}
        return evaluate

    def test_adaptive_search_refines_factor_four_bracket(self):
        curve = lambda chains: 1.0 + (math.log2(chains) - 6.0) ** 2
        result = _adaptive_chain_search(
            16, 65536, self._curve_evaluator(curve))
        tested = [row["chains"] for row in result["candidates"]]
        self.assertEqual(result["recommended_chains"], 64)
        self.assertIn(32, tested)
        self.assertIn(128, tested)
        self.assertLess(len(tested), len(chain_candidates(65536)))

    def test_adaptive_search_continues_past_2048(self):
        curve = lambda chains: 1.0 + (math.log2(chains) - 13.0) ** 2
        result = _adaptive_chain_search(
            32, 65536, self._curve_evaluator(curve))
        tested = [row["chains"] for row in result["candidates"]]
        self.assertEqual(result["recommended_chains"], 8192)
        self.assertIn(8192, tested)
        self.assertEqual(result["search"]["stop_reason"], "converged")

    def test_adaptive_search_tolerates_one_stalled_point(self):
        times = {8: 100.0, 32: 80.0, 128: 60.0, 512: 65.0,
                 1024: 45.0, 2048: 30.0, 4096: 35.0,
                 8192: 40.0, 32768: 50.0}
        result = _adaptive_chain_search(
            32, 65536, self._curve_evaluator(lambda chains: times[chains]))
        tested = [row["chains"] for row in result["candidates"]]
        self.assertIn(2048, tested)
        self.assertEqual(result["recommended_chains"], 2048)

    def test_adaptive_search_reports_memory_boundary(self):
        result = _adaptive_chain_search(
            32, 65536,
            self._curve_evaluator(lambda chains: 1000.0 / chains,
                                  unsafe_at=1024))
        self.assertEqual(result["search"]["stop_reason"], "memory_limit")
        self.assertLess(result["recommended_chains"], 1024)

    def test_adaptive_search_reports_improving_hard_limit(self):
        result = _adaptive_chain_search(
            32, 4096,
            self._curve_evaluator(lambda chains: 1000.0 / chains))
        self.assertEqual(result["recommended_chains"], 4096)
        self.assertEqual(result["search"]["stop_reason"],
                         "max_chains_reached")

    def test_stage_skips_candidate_predicted_to_exceed_gpu_limit(self):
        cfg = load_config(Path(__file__).parents[1] / "configs" / "example.toml")
        workload = Workload("fine", 4, 4, True, "half",
                            Couplings(1.0, 0.0, 0.0), 10)
        measured = []

        def benchmark(_cfg, _workloads, chains, **_kwargs):
            measured.append(chains)
            return {"chains": chains, "safe": True,
                    "projected_target_seconds": 1000.0 / chains,
                    "chain_sweeps_per_s": float(chains),
                    "peak_gpu_gib": chains / 16.0, "workloads": []}

        properties = type("Properties", (), {"total_memory": 4 * 1024 ** 3})()
        with patch("cpn_renorm.chain_tuning.resolve_device",
                   return_value=torch.device("cuda")), patch(
                       "cpn_renorm.chain_tuning.torch.cuda.get_device_properties",
                       return_value=properties), patch(
                           "cpn_renorm.chain_tuning._benchmark_candidate",
                           side_effect=benchmark):
            result = _benchmark_stage(
                cfg, [workload], 16, max_chains=64, warmup_sweeps=1,
                timed_sweeps=1, gpu_memory_fraction=0.5,
                max_host_memory_gib=4.0)

        row64 = next(row for row in result["candidates"] if row["chains"] == 64)
        self.assertNotIn(64, measured)
        self.assertEqual(row64["reason"], "predicted_gpu_memory_limit")
        self.assertEqual(result["search"]["stop_reason"], "memory_limit")

    def test_cuda_allocator_limit_is_restored(self):
        with patch("cpn_renorm.chain_tuning.torch.cuda.empty_cache") as empty, patch(
                "cpn_renorm.chain_tuning.torch.cuda.current_device",
                return_value=0), patch(
                "cpn_renorm.chain_tuning.torch.cuda.get_per_process_memory_fraction",
                return_value=0.8), patch(
                    "cpn_renorm.chain_tuning.torch.cuda.set_per_process_memory_fraction"
                ) as set_fraction:
            with _cuda_allocator_limit(torch.device("cuda"), 0.6):
                pass
        self.assertEqual(set_fraction.call_args_list, [
            call(0.6, torch.device("cuda:0")),
            call(0.8, torch.device("cuda:0")),
        ])
        self.assertEqual(empty.call_count, 2)

    def test_tuning_summary_keeps_compact_stage_result(self):
        result = {
            "recommended_chains": 64,
            "candidates": [
                {"chains": 16, "safe": True, "peak_gpu_gib": 0.5,
                 "projected_target_seconds": 20.0},
                {"chains": 64, "safe": True, "peak_gpu_gib": 1.5,
                 "projected_target_seconds": 10.0},
            ],
            "search": {"stop_reason": "converged"},
        }
        summary = _tuning_summary("pilot", result)
        self.assertIn("tuned pilot: tested=2 range=16..64 safe=2", summary)
        self.assertIn("recommended=64 projected=10s gpu_peak=1.500GiB", summary)


if __name__ == "__main__":
    unittest.main()
