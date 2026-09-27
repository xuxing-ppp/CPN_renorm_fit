from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def isolated_fit(operation: str, request: dict, **arrays: np.ndarray) -> dict:
    worker = Path(__file__).with_name("fit_worker.py")
    with tempfile.TemporaryDirectory(prefix="cpn-fit-") as tmp:
        root = Path(tmp)
        req, data, out = root / "request.json", root / "arrays.npz", root / "result.json"
        req.write_text(json.dumps({"operation": operation, **request}), encoding="utf-8")
        np.savez_compressed(data, **arrays)
        completed = subprocess.run([sys.executable, str(worker), str(req), str(data), str(out)],
                                   check=False, text=True, capture_output=True)
        if completed.returncode:
            raise RuntimeError("isolated fit failed:\n" + completed.stdout + completed.stderr)
        return json.loads(out.read_text(encoding="utf-8"))


def isolated_match(x: np.ndarray, y: np.ndarray, target: float,
                   y_err: np.ndarray | None = None,
                   target_err: float | None = None, *, increasing: bool = True,
                   max_points: int = 5, bootstrap_seed: int = 1729,
                   bootstrap_samples: int = 256) -> dict:
    arrays = {"x": np.asarray(x), "y": np.asarray(y)}
    if y_err is not None:
        arrays["y_err"] = np.asarray(y_err)
    return isolated_fit("match", {"target": float(target), "increasing": increasing,
                        "max_points": int(max_points), "bootstrap_seed": int(bootstrap_seed),
                        "bootstrap_samples": int(bootstrap_samples),
                        "target_err": (None if target_err is None else float(target_err))},
                        **arrays)


def isolated_plot(destination: str | Path, *, x: np.ndarray,
                  series: dict[str, np.ndarray], targets: dict[str, float],
                  xlabel: str, ylabel: str,
                  errors: dict[str, np.ndarray] | None = None,
                  matches: dict[str, dict] | None = None,
                  colors: dict[str, str] | None = None) -> None:
    arrays = {"x": np.asarray(x), **series}
    error_keys = {}
    for key, value in (errors or {}).items():
        array_key = f"error__{key}"
        arrays[array_key] = np.asarray(value)
        error_keys[key] = array_key
    isolated_fit("plot", {"destination": str(Path(destination).resolve()),
                 "series": list(series), "targets": targets,
                 "errors": error_keys, "matches": matches or {},
                 "colors": colors or {}, "xlabel": xlabel, "ylabel": ylabel},
                 **arrays)


def extend_scan(points: list[float], result: dict, limits: tuple[float, float],
                growth: float) -> list[float]:
    points = sorted(set(float(x) for x in points))
    lo, hi = limits
    if len(points) < 2:
        raise ValueError("scan requires at least two initial points")
    span = points[-1] - points[-2] if result.get("direction") == "high" else points[1] - points[0]
    candidate = (min(hi, points[-1] + growth * span) if result.get("direction") == "high"
                 else max(lo, points[0] - growth * span))
    if any(math.isclose(candidate, x, abs_tol=1e-12) for x in points):
        return points
    return sorted(points + [candidate])
