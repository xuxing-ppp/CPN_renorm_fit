"""Edit the parameters below, then run: python configs/next_config.py."""
from pathlib import Path
import sys


# Source run name (searched under runs/), or a run directory path.
# Relative directory paths are relative to the project root.
SOURCE = "runs/N2_L2/N2_L2_b1.2"
MOD = 0
TYPE = "z"                 # Source branch type, U or z.
BETA_METHOD = "magsus"       # magsus or corrG; also selects the next fit method.
ALPHA_METHOD = "topo"       # 1plaq or topo; also controls the next topo fit.
FACTOR = 2                 # Blocking factor for the NEXT renormalization.

# Relative directories are relative to configs/; absolute directories also work.
OUTPUT_DIR = "N2_L2"
# Leave empty (or None) for <source>-L<factor>_mod<mod>_<beta_method>_<alpha_method>.
# The .toml extension is added automatically if omitted.
OUTPUT_FILENAME = ""
NEXT_TYPE = "U"
OVERWRITE = False

# Optional overrides of inherited settings, using Python values.
# Model and dedicated parameters above cannot be changed through OVERRIDES.
OVERRIDES = {
    "runtime.seed": 1018,
    # "runtime.device": "cuda",
    # "steps.observable.target_ess": 5000,
}


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))
    from cpn_renorm.next_config import generate_next_config

    try:
        generate_next_config(
            SOURCE, mod=MOD, source_type=TYPE, beta_method=BETA_METHOD,
            alpha_method=ALPHA_METHOD, factor=FACTOR,
            output_dir=OUTPUT_DIR, filename=OUTPUT_FILENAME,
            project_root=project_root, next_type=NEXT_TYPE,
            overwrite=OVERWRITE, overrides=OVERRIDES,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
