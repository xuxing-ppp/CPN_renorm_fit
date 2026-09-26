"""SciPy-only worker used to isolate CPU fitting from PyTorch's OpenMP runtime."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpn_renorm.fitting import fit_one_plaq, fit_two_plaq, match_coupling


def main() -> None:
    request_path, arrays_path, output_path = map(Path, sys.argv[1:4])
    request = json.loads(request_path.read_text(encoding="utf-8"))
    with np.load(arrays_path) as data:
        arrays = {key: data[key] for key in data.files}
    operation = request["operation"]
    if operation == "one_plaq":
        result = fit_one_plaq(arrays["X"], arrays["freq"], float(request["p0"]))
    elif operation == "two_plaq":
        result = fit_two_plaq(int(request["N"]), arrays["X"], arrays["freq"],
                               tuple(request["p0"]))
    elif operation == "match":
        err = arrays.get("y_err")
        result = match_coupling(arrays["x"], arrays["y"], float(request["target"]), err,
                                request.get("target_err"))
    elif operation == "plot":
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for key in request["series"]:
            plt.plot(arrays["x"], arrays[key], marker="o", label=key)
        for label, target in request.get("targets", {}).items():
            plt.axhline(float(target), linestyle="--", label=label)
        plt.xlabel(request["xlabel"])
        plt.ylabel(request["ylabel"])
        plt.legend()
        plt.tight_layout()
        destination = Path(request["destination"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(destination, dpi=150)
        plt.close()
        result = {"path": str(destination)}
    else:
        raise ValueError(f"unknown fitting operation {operation!r}")
    output_path.write_text(json.dumps(result, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
