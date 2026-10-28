"""Isolated worker: load Torch for CUDA fits, SciPy for CPU fits and scans."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    request_path, arrays_path, output_path = map(Path, sys.argv[1:4])
    request = json.loads(request_path.read_text(encoding="utf-8"))
    with np.load(arrays_path) as data:
        arrays = {key: data[key] for key in data.files}
    operation = request["operation"]
    device = request.get("device", "cpu")
    if operation in ("one_plaq", "two_fixed", "two_mse") and device != "cpu":
        from cpn_renorm.torch_fitting import fit_one, fit_two_fixed, prediction_mse
        if operation == "one_plaq":
            result = fit_one(arrays["X"], arrays["freq"], request["p0"], device)
        elif operation == "two_fixed":
            result = fit_two_fixed(request["N"], arrays["X"], arrays["freq"],
                                   request["alpha"], request["p0"], device)
        else:
            result = prediction_mse(request["N"], arrays["X"], arrays["freq"],
                                    request["parameters"], device=device)
    elif operation == "two_fixed":
        from cpn_renorm.exact_fitting import fit_two_fixed
        result = fit_two_fixed(request["N"], arrays["X"], arrays["freq"], request["alpha"], request["p0"])
    elif operation == "two_mse":
        from cpn_renorm.exact_fitting import prediction_mse
        result = prediction_mse(request["N"], arrays["X"], arrays["freq"], request["parameters"])
    elif operation == "one_plaq":
        from cpn_renorm.fitting import fit_one_plaq
        result = fit_one_plaq(arrays["X"], arrays["freq"], float(request["p0"]))
    elif operation == "two_plaq":
        from cpn_renorm.fitting import fit_two_plaq
        result = fit_two_plaq(int(request["N"]), arrays["X"], arrays["freq"],
                               tuple(request["p0"]))
    elif operation == "match":
        from cpn_renorm.fitting import match_coupling
        err = arrays.get("y_err")
        result = match_coupling(arrays["x"], arrays["y"], float(request["target"]), err,
                                request.get("target_err"),
                                increasing=bool(request.get("increasing", True)),
                                max_points=int(request.get("max_points", 5)),
                                bootstrap_seed=int(request.get("bootstrap_seed", 1729)),
                                bootstrap_samples=int(request.get("bootstrap_samples", 256)))
    elif operation in ("plot", "plot_one", "plot_two"):
        from cpn_renorm.plotting import plot_one, plot_scan, plot_two
        result = {"plot": plot_scan, "plot_one": plot_one,
                  "plot_two": plot_two}[operation](request, arrays)
    else:
        raise ValueError(f"unknown fitting operation {operation!r}")
    if operation in ("one_plaq", "two_fixed", "two_mse") and device.startswith("cuda"):
        import torch
        torch.cuda.synchronize()
        result["peak_cuda_memory_bytes"] = torch.cuda.max_memory_allocated()
    output_path.write_text(json.dumps(result, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        if len(sys.argv) >= 4:
            Path(sys.argv[3]).write_text(json.dumps({
                "worker_error": type(exc).__name__, "message": str(exc)}))
        raise
