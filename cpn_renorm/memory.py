"""Conservative host-memory budgets for frozen-boundary sampling."""
from __future__ import annotations

import os
from pathlib import Path


def available_host_bytes() -> int:
    """Read available RAM, respecting a Linux cgroup memory limit when present."""
    try:
        fields = dict(line.split(":", 1) for line in
                      Path("/proc/meminfo").read_text().splitlines())
        available = int(fields["MemAvailable"].split()[0]) * 1024
    except (OSError, KeyError, ValueError):
        available = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    # Both a container namespace root and the process's unified cgroup path.
    roots = [Path("/sys/fs/cgroup")]
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            if line.startswith("0::"):
                roots.append(roots[0] / line[3:].lstrip("/"))
    except OSError:
        pass
    for root in roots:
        for directory in (root, *root.parents):
            if not directory.is_relative_to(Path("/sys/fs/cgroup")):
                break
            try:
                limit = (directory / "memory.max").read_text().strip()
                if limit != "max":
                    used = int((directory / "memory.current").read_text())
                    available = min(available, max(0, int(limit) - used))
            except (OSError, ValueError):
                pass
    return available


def host_budget_gib(requested: float = 8.0) -> float:
    """Leave half of currently available RAM for other allocations/processes."""
    return min(requested, available_host_bytes() / 2 / 1024 ** 3)


def patch_peak_gib(stage: str, chains: int, measurements: int, factor: int,
                   fit_times: int, bins: int = 128, zero_pad: int = 1) -> float:
    """Estimate history, one-chain diagnostics, and histogram copies.

    Histories and statistics use float64 even for float32 sampling.  Histogram
    space includes three copies and a patch-area allowance for charge support.
    """
    observables = 3 if stage == "one_plaq" else 2
    history = measurements * chains * observables * 8
    # sin/cos plus FFT autocorrelation work arrays, processed one chain at a time.
    diagnostics = measurements * 256
    boundaries = ((fit_times + chains - 1) // chains) * chains
    rows = 2 * factor * factor + 6 + 2 * zero_pad if stage == "one_plaq" else bins
    columns = 9 if stage == "one_plaq" else 20
    # Per-batch rows, aggregation, and construction temporaries can coexist.
    histograms = boundaries * rows * columns * 8 * 3
    return (history + diagnostics + histograms) / 1024 ** 3


def patch_workspace_gib(chains: int, sites: int, components: int,
                        warmup: int, cpu: bool) -> float:
    """Allow for warmup probes/statistics and CPU sampler working tensors."""
    probes = warmup * chains * 16 * 8
    sampler = chains * sites * (components + 4) * 512 if cpu else 0
    return (probes + sampler) / 1024 ** 3
