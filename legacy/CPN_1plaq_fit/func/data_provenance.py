"""Sampler-config provenance for the 1plaq MC data `.npz` files.

The data filename encodes the couplings and the data-shape knobs
(`fit_times`, `zero_pad`) but NOT the MC knobs (`n_therm`, `epsilon`, `s_step`,
...).  A filename-only "skip if exists" would therefore silently reuse stale
data when only an MC knob changes.  This module stamps the full sampler config
that produced a file into the `.npz` itself (as a JSON string under
`PROVENANCE_KEY`), so reuse can be validated against the requested config.

Pure-python (`json` / `os` / `math` / `numpy` only) -> safe to import at module
level, including inside macOS-spawn `ProcessPoolExecutor` workers.

NOTE: the generic helpers below are duplicated (intentionally, and kept in
sync) in `CPN_2plaq_fit/func/data_provenance.py` so each subproject stays
independently runnable.  Only `data_fingerprint_1plaq` is specific to this
subproject.
"""

import json
import math
import os

import numpy as np

PROVENANCE_KEY = "__sampler_config__"


# --------------------------------------------------------------------------- #
# Generic helpers (identical twin lives in CPN_2plaq_fit/func/data_provenance.py)
# --------------------------------------------------------------------------- #
def save_npz_with_fingerprint(filepath, arrays_kwds, fingerprint, key=PROVENANCE_KEY):
    """Write the measurement arrays AND the fingerprint in a single `np.savez`.

    `np.savez` overwrites (there is no append), so the fingerprint must go into
    the same call as the arrays.  Creates the parent directory.
    """
    folder = os.path.dirname(filepath)
    if folder:
        os.makedirs(folder, exist_ok=True)
    kwds = dict(arrays_kwds)
    kwds[key] = np.array(json.dumps(fingerprint, sort_keys=True))
    np.savez(filepath, **kwds)


def load_npz_fingerprint(filepath, key=PROVENANCE_KEY):
    """Return the stored fingerprint dict, or `None` if absent/unparseable.

    Opens with `allow_pickle=False`: the fingerprint is stored as a `<U` string
    array, which is NOT pickled, so this works for files consumed by either fit
    loader (the 2plaq loader uses `allow_pickle=False`, the 1plaq one `True`).
    Never raises -- the caller decides what to do.
    """
    try:
        with np.load(filepath, allow_pickle=False) as data:
            if key not in data.files:
                return None
            raw = data[key].item()
    except (OSError, ValueError) as exc:
        print(f"[data_provenance] could not read fingerprint from {filepath}: {exc}")
        return None
    try:
        return json.loads(raw)
    except (TypeError, ValueError) as exc:
        print(f"[data_provenance] could not parse fingerprint JSON in {filepath}: {exc}")
        return None


def configs_match(requested_fp, stored_fp, float_tol=1e-9):
    """Compare two fingerprints -> ``(ok, diff_lines)``.

    Union of keys; a key present on only one side counts as a diff.  ints and
    strings use ``==``; floats use ``math.isclose(rel_tol=0, abs_tol=float_tol)``
    (NOT ``np.isclose`` defaults -- too loose; they would equate 0.050 vs 0.051).
    A ``None`` stored fingerprint (legacy file with no provenance) is a mismatch.
    """
    if stored_fp is None:
        return False, ["no stored config (legacy file)"]
    diff = []
    for k in sorted(set(requested_fp) | set(stored_fp)):
        if k not in requested_fp:
            diff.append(f"{k}: stored={stored_fp[k]!r} requested=<missing>")
        elif k not in stored_fp:
            diff.append(f"{k}: stored=<missing> requested={requested_fp[k]!r}")
        else:
            a, b = requested_fp[k], stored_fp[k]
            if isinstance(a, bool) or isinstance(b, bool):
                if a != b:
                    diff.append(f"{k}: stored={b!r} requested={a!r}")
            elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
                if not math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=float_tol):
                    diff.append(f"{k}: stored={b!r} requested={a!r}")
            elif a != b:
                diff.append(f"{k}: stored={b!r} requested={a!r}")
    return (len(diff) == 0), diff


def decide_data_action(filepath, requested_fp, auto_regenerate=True):
    """Decide whether to reuse, regenerate, or raise.

    Returns ``(action, stored_fp, diff)`` with ``action`` in ``{"reuse",
    "regenerate", "raise"}``:

    * ``"reuse"``      -- the file exists and its stored fingerprint matches.
    * ``"regenerate"`` -- overwrite with fresh MC. Happens when the file does not
                          yet exist, OR when ``auto_regenerate`` is on and the
                          existing file is stale (mismatch) or legacy (no stored
                          config).
    * ``"raise"``      -- ``auto_regenerate`` is off and the existing file is
                          stale/legacy. The caller should raise, using ``diff``
                          (and the filepath it passed in) to describe the problem.

    A missing file always regenerates regardless of ``auto_regenerate`` -- there
    is no conflict to raise about, just a first-time generation.
    """
    if not os.path.exists(filepath):
        return "regenerate", None, ["file does not exist"]
    stored_fp = load_npz_fingerprint(filepath)
    ok, diff = configs_match(requested_fp, stored_fp)
    if ok:
        return "reuse", stored_fp, []
    if auto_regenerate:
        return "regenerate", stored_fp, diff
    return "raise", stored_fp, diff


# --------------------------------------------------------------------------- #
# 1plaq fingerprint
# --------------------------------------------------------------------------- #
def _use_Original_1plaq(beta1, alpha):
    """Mirrors CPN_1plaq_data_generate.py / step_common.use_Original_1plaq.

    True when BOTH beta1 and alpha are ~0: the action then has no Villain
    integer field `s` and reduces to the Original (cos-plaquette) heatbath
    sampler.  (Independent of alpha1.)
    """
    return abs(float(beta1)) < 1e-10 and abs(float(alpha)) < 1e-10


def data_fingerprint_1plaq(*, N, L, mod, beta_f, beta1_f, alpha_f, alpha1_f, BC, pad,
                           fit_times, zero_pad, n_therm, meas_interval, n_meas,
                           boundary_n_therm, s_step, s_update_num,
                           epsilon, n_leapfrog, mass_a, mass_z,
                           hf_therm, hf_meas):
    """The exact set of parameters that determine a 1plaq data file's content.

    ``n_workers`` is excluded (parallelism does not affect the distribution);
    ``p0`` is fit-only.  The sampler tag is implied by ``beta1_f`` and
    ``alpha_f`` together (see ``_use_Original_1plaq``).
    """
    # Effective mod: when beta1 ~= 0 the beta1-matter term vanishes and mod is
    # physically irrelevant; by convention (matching `file_mod` / the _mod{m}
    # path token) the stored config then carries mod=0, so the fingerprint's mod
    # stays consistent with the folder the file actually lives under.
    eff_mod = 0 if abs(float(beta1_f)) < 1e-10 else int(mod)
    sampler = "Original" if _use_Original_1plaq(beta1_f, alpha_f) else "RefVil_HMC"
    return {
        "N": int(N), "L": int(L), "mod": eff_mod,
        "beta_f": float(beta_f), "beta1_f": float(beta1_f),
        "alpha_f": float(alpha_f), "alpha1_f": float(alpha1_f),
        "BC": str(BC), "pad": int(pad),
        "fit_times": int(fit_times), "zero_pad": int(zero_pad),
        "n_therm": int(n_therm), "meas_interval": int(meas_interval),
        "n_meas": int(n_meas),
        "boundary_n_therm": int(boundary_n_therm),
        "s_step": float(s_step), "s_update_num": int(s_update_num),
        "epsilon": float(epsilon), "n_leapfrog": int(n_leapfrog),
        "mass_a": float(mass_a), "mass_z": float(mass_z),
        "hf_therm": float(hf_therm), "hf_meas": float(hf_meas),
        "sampler": sampler,
    }
