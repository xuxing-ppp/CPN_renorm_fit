from __future__ import annotations

import sys
from typing import TextIO

import numpy as np
from tqdm import tqdm


class SimulationProgress:
    """Reuse one transient tqdm bar across the phases of a simulation."""

    def __init__(self, label: str, *, stream: TextIO | None = None,
                 enabled: bool | None = None) -> None:
        self.label = label
        self.stream = stream or sys.stderr
        visible = bool(getattr(self.stream, "isatty", lambda: False)())
        if enabled is not None:
            visible = enabled
        self._bar = tqdm(total=0, desc=label, leave=False, dynamic_ncols=True,
                         disable=not visible, file=self.stream, unit="sweep")

    def phase(self, name: str, total: int, *, unit: str | None = None) -> None:
        """Reset the bar for a new phase without leaving the previous bar behind."""
        self._bar.set_description_str(f"{self.label} {name}", refresh=False)
        if unit is not None:
            self._bar.unit = unit
        self._bar.reset(total=max(int(total), 0))
        self._bar.set_postfix_str("", refresh=False)

    def extend(self, count: int = 1) -> None:
        """Add newly discovered work to an adaptive phase."""
        self._bar.total = max(int(self._bar.total or 0) + int(count), 0)
        self._bar.refresh()

    def update(self, count: int = 1, **metrics: object) -> None:
        if metrics:
            fields = [f"{key}={value}" for key, value in metrics.items()
                      if value is not None]
            self._bar.set_postfix_str(" ".join(fields), refresh=False)
        self._bar.update(int(count))

    def message(self, value: str) -> None:
        """Write a persistent line without corrupting the transient bar."""
        self._bar.write(value, file=self.stream)

    def close(self) -> None:
        self._bar.close()

    def __enter__(self) -> SimulationProgress:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def ensemble_summary(label: str, result: dict) -> str:
    """Format the diagnostics that explain why an ensemble stopped."""
    sampling, warmup = result["sampling"], result["warmup"]
    ess = np.asarray(sampling.get("ess", result.get("mode_ess", [])), dtype=float)
    ess_min = float(np.min(ess)) if ess.size else float("nan")
    accept = float(np.mean(np.asarray(result["accept_rate"], dtype=float)))
    return (f"finished {label}: warmup={warmup['sweeps']}({warmup['stop_reason']}) "
            f"samples={sampling['actual']} ess={ess_min:.1f}/{sampling['target_ess']:g} "
            f"stop={sampling['stop_reason']} accept={accept:.3f} "
            f"epsilon={float(result['epsilon']):.4g}")
