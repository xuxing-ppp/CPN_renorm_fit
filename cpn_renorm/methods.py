"""Observable matching methods; corrG uses the second-moment length xi_G."""

BETA1_METHODS = {
    "magsus": {"field": "magsus", "power": 2, "increasing": True, "seed": 201},
    "corrG": {"field": "xi_G", "power": 1, "increasing": True, "seed": 202},
}
ALPHA_METHODS = ("1plaq", "topo")
