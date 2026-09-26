import io
import unittest
from unittest.mock import patch

import numpy as np

from cpn_renorm.observables import automatic_warmup
from cpn_renorm.progress import SimulationProgress, ensemble_summary


class RecordingProgress:
    def __init__(self):
        self.phases = []
        self.updates = []

    def phase(self, name, total):
        self.phases.append((name, total))

    def update(self, count=1, **metrics):
        self.updates.append((count, metrics))


class ProgressTests(unittest.TestCase):
    def test_tqdm_is_transient_and_automatically_follows_tty(self):
        class Stream(io.StringIO):
            def __init__(self, tty):
                super().__init__()
                self.tty = tty

            def isatty(self):
                return self.tty

        for tty in (False, True):
            with self.subTest(tty=tty), patch("cpn_renorm.progress.tqdm") as factory:
                bar = factory.return_value
                progress = SimulationProgress("test", stream=Stream(tty))
                progress.phase("sampling", 10)
                progress.update(2, ess="4.0/5")
                progress.close()
                self.assertEqual(factory.call_args.kwargs["leave"], False)
                self.assertEqual(factory.call_args.kwargs["disable"], not tty)
                bar.reset.assert_called_once_with(total=10)
                bar.update.assert_called_once_with(2)
                bar.close.assert_called_once()

    def test_warmup_reports_adaptation_and_tau_diagnostics(self):
        class FakeSampler:
            def adapt(self, steps, _target, **kwargs):
                callback = kwargs["progress"]
                for _ in range(steps):
                    callback(0.1, 0.8)
                return 0.1

            def sweep(self, **_kwargs):
                pass

            def reset_diagnostics(self):
                pass

        progress = RecordingProgress()
        result = automatic_warmup(
            FakeSampler(), adapt_steps=3, target_accept=0.75,
            min_warmup=20, max_warmup=20,
            probe=lambda _sampler: np.ones((2, 1)), progress=progress)
        self.assertEqual(progress.phases, [("adapt", 3), ("warmup", 20)])
        self.assertEqual(sum(count for count, _ in progress.updates), 23)
        self.assertTrue(any("tau_max" in metrics for _, metrics in progress.updates))
        self.assertEqual(result["stop_reason"], "max_warmup")

    def test_ensemble_summary_contains_stopping_diagnostics(self):
        text = ensemble_summary("obs L=8", {
            "warmup": {"sweeps": 24, "stop_reason": "tau_stable"},
            "sampling": {"actual": 160, "target_ess": 20.0,
                         "ess": [24.0, 21.5], "stop_reason": "target_ess"},
            "accept_rate": np.array([0.8, 0.9]), "epsilon": 0.04,
        })
        self.assertIn("warmup=24(tau_stable)", text)
        self.assertIn("samples=160 ess=21.5/20", text)
        self.assertIn("stop=target_ess accept=0.850 epsilon=0.04", text)


if __name__ == "__main__":
    unittest.main()
