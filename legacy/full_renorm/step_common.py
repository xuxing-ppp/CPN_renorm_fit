"""Shared helpers for the full_renorm pipeline steps (pure-python, no `func` deps).

These are intentionally lightweight and side-effect-free so they can be imported
at module level by the step scripts (which spawn ProcessPoolExecutor workers —
on macOS the `spawn` start method re-imports the step module in every worker).
"""

import json
import os


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(obj, path):
    """Write JSON, creating the parent directory if needed."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def mod_for(beta1, mod):
    """mod token used in data/results paths.

    The "Original" (beta1=0) action has no Villain integer field and therefore
    no mod concept, so the path always carries mod=0 for it. Matches
    CPN_2plaq_data_generate.py / CPN_1plaq_data_generate.py.
    """
    return 0 if abs(beta1) < 1e-10 else int(mod)


def use_Original_1plaq(beta1, alpha):
    """Whether the 1plaq fine model reduces to the Original (cos-plaquette) sampler.

    True only when BOTH beta1 and alpha are ~0 (note: it tests *alpha*, not
    alpha1). In that regime the integer s field is absent (vs≡0), which makes the
    s-vortex fit degenerate -- the orchestrator skips the alpha fits (1plaq/topo)
    and fixes alpha_c=0 for U-renorm in that case.
    """
    return abs(beta1) < 1e-10 and abs(alpha) < 1e-10
