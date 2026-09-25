# full_renorm — new combined renormalization driver (topo-alpha pipeline)

Alternative to `full_renorm_original/`. Takes a **fine model**
`(beta_f, beta1_f, alpha_f, alpha1_f)` and produces renormalized (coarse)
couplings, with `alpha` determined two independent ways.

```
step 1   CPN_2plaq_fit            -> beta_c , alpha_eff_c (= alpha_c + alpha1_c)
step 2   CPN_observable_fit       -> beta1_c  {magsus, corr}        (halfRefVil obs fit)
step 3a  CPN_1plaq_fit            -> alpha_c                         (1plaq vortex)
step 3b  CPN_observable_fit       -> alpha_c  {magsus, corr}        (topo-sus match,
                                  via plot_topo_alpha.process_case_topo)
```

`beta1` comes from step 2 (two estimates). `alpha` comes from EITHER step 3a
(1plaq) OR step 3b (topo). The deliverable is the cross product — **four
renormalization combos**:

| combo | beta1_c from | alpha_c from |
|---|---|---|
| `magsus_obs_fit_topo` | magsus | topo@magsus |
| `corr_obs_fit_topo` | corr | topo@corr |
| `magsus_1plaq` | magsus | 1plaq |
| `corr_1plaq` | corr | 1plaq |

### Why this pipeline differs from `full_renorm_original/`

- **Step 2 uses halfRefVil only**, returning only `magsus` and `corr` (no `avg`,
  no `toposus`). halfRefVil because at this stage `alpha`/`alpha1` are not yet
  split — only `alpha_eff` is known.
- **Step 3b is new**: a topo-susceptibility alpha fit (`CPN_observable_fit/plot_topo_alpha.py`).
  At fixed `(beta_c, beta1_c, alpha_eff_c)` it scans coarse `alpha`
  (`alpha1 = alpha_eff_c - alpha`) on a **RefVil** lattice (data from the same
  `measure_PP_Wil.main` that step 2 uses), measures the integer-winding
  susceptibility `topo_sus(Q_s)`, and matches it onto the fine susceptibility
  scaled by `L**2`. Because `measure_PP_Wil` also records `conn_PP_corr`,
  `process_case_topo` emits informational `mag_sus.png` / `corr_len.png` vs alpha
  alongside the `topo_sus.png` fit plot. The coarse scan is run **per beta1**
  (magsus and corr; one `beta1_c_{...}/` results subfolder each); the fine sim is
  shared. This gives a second, independent estimate of `alpha_c` alongside the
  1plaq vortex fit. (The standalone `CPN_observable_fit_topo/` subproject is
  merged away — `measure_PP_Wil` already measured topo charge.)

The action family is fixed by the pipeline, so (unlike `full_renorm_original/`)
there is **no `results_{obs_fit_type}/` split** — output lands in `results/`.

## Run

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate myenv
cd /path/to/CPN_renorm_fit_mod
python full_renorm/orchestrator.py full_renorm/configs/N2_L4_z_mod0/N2_4_z_b1.5_mod0.json
```

Or edit `json_config_list` / `n_workers` in `orchestrator.py::__main__` and run
`python full_renorm/orchestrator.py` (set the list to `None` to run every config
under `configs/`).

Output lands in `full_renorm/results/N{N}_L{L}_{renorm_type}_mod{mod}/<config-stem>/`:

| file | contents |
|---|---|
| `config.json` | the master config (with `n_workers` stamped on each block) |
| `step{1,2,3a,3b}_config.json` | the slice each step received |
| `step{1,2,3a,3b}_result.json` | that step's outputs |
| `full_renorm_summary.json` | `renormalized_couplings`, `four_combos`, `renorm_as_fine` |

Each step also writes its native artifacts into its subproject's own `data/` and
`results/` trees (so the per-subproject CLAUDE conventions are unaffected). MC
data is reused automatically — every generator/fit skips existing outputs, and
`measure_PP_Wil.main` tops up missing argzz/Wilson sizes (so step 3b's fine call
with empty argzz/wilson reuses step 2's fine data when the sampler family matches).

## `renorm_type` (one choice, consistent across all steps)

| `renorm_type` | 2plaq `conn_type` | 1plaq vortex | topo fine component |
|---|---|---|---|
| `"U"` | `"U"` | `s`-vortex | fine `Q_s` (RefVil fine, needs `alpha_f > 0`) |
| `"z"` (default) | `"z"` | `z`-vortex | fine `Q_z` (halfRefVil/HB fine) |

U-renorm determines `alpha` through the fine integer field `s` (1plaq s-vortex,
topo `Q_s`), which needs `alpha_f > 0`: `s` is constrained only by the
`alpha_f*(da + 2*pi*s)**2` term (the `alpha1*cos(da + 2*pi*s)` term is
`s`-invariant), so with `alpha_f ≈ 0` the fine `s` field is unconstrained
(`vs ≡ 0` in `run_Original`; a random walk in RefVil) and BOTH alpha methods are
degenerate, regardless of `beta1_f`. In that case the orchestrator does **not**
error: it runs steps 1–2 only (steps 3a/3b are skipped) and the summary fixes
**`alpha_c = 0.0`** (`alpha1_c = alpha_eff_c`) in every combo and in
`renorm_as_fine`, carrying `beta_c`, `beta1_c {magsus, corr}` and `alpha_eff_c`
as fitted. The skip is recorded in `inputs.skip_alpha_fits`, `provenance`
(`SKIPPED (U-renorm alpha_f~=0 ...)`) and `warnings` — `alpha_c=0.0` there is a
fixed convention, not a fit result.

**Topo fine sampler dispatch (step 3b)**: the coarse α-scan is always RefVil
(needs `Q_s`); the fine lattice is **RefVil if `alpha_f ≠ 0` OR `renorm_type=="U"`**,
otherwise **halfRefVil** (pure-β → heatbath inside). So z-renorm with a pure-β
fine uses halfRefVil (matches `Q_z`); z-renorm with `alpha_f ≠ 0` and all U-renorm
use RefVil (matching `Q_z` / `Q_s` respectively).

## Config schema

Same top-level fields as `full_renorm_original` (`N, mod, renorm_type, L,
L_coarse, fine{...}, boundary{BC,pad}, pp_xy_list_coarse, argzz_xy_list_coarse,
wilson_xy_list_coarse, beta1_list, 2plaq, 1plaq, observable`) plus two NEW keys:

- **`alpha_list`** — list of coarse `alpha` scan values for step 3b (the topo
  fit). Case-specific; tune it to bracket the matched `alpha_c`, just as
  `beta1_list` brackets `beta1_c`. A reasonable starting range is
  `np.linspace(0, ~alpha_eff_c, 7-9)` — `alpha_c` should sit inside it. **May be
  left empty (`[]`, or omitted)**: then step 3b auto-builds a 5-point bracket
  from the **1plaq** alpha_c — round it to one significant digit (`a1`) and to
  two significant digits (`a2`) and use
  `[a2-0.2a1, a2-0.1a1, a2, a2+0.1a1, a2+0.2a1]` (centered on the finer `a2`,
  ±20% of the coarser `a1`) — exploiting that the topo alpha_c should sit near
  the 1plaq alpha_c. This needs step 3a to have produced a positive alpha_c;
  otherwise supply an explicit list. (The orchestrator passes the 1plaq alpha_c
  to step 3b as `alpha_1plaq`, and records the actual list used in
  `full_renorm_summary.json`.)
- **`topo`** — MC knobs for step 3b's `measure_PP_Wil.main` calls (`s_step,
  s_update_num, heatbath_fraction, epsilon, n_leapfrog, mass_a, mass_z, n_therm,
  n_meas, meas_interval, n_total`). Topological charge has a long autocorrelation
  time, so prefer generous `n_meas`/`n_total`. `s_step`/`s_update_num` are needed
  (the coarse scan is RefVil).

Note on `s_step`/`s_update_num`: these are the integer-`s` Metropolis knobs and
only matter for the **RefVil** sampler. The `observable` block omits them — step 2
is always halfRefVil/HB (no integer `s`). The `topo` block (coarse = RefVil) and
the `1plaq` block (RefVil vortex fit) keep them; in `1plaq` they're unused only
for the pure-β z-mod cases that route to `run_Original`, but are required for the
U-renorm / non-pure-β cases.

The representative configs under `configs/` (one or two per
`N/L/renorm_type/mod` category) carry placeholder `alpha_list` ranges — **tune
them per case before trusting the topo alpha**. Port more cases from
`full_renorm_original/configs/` by adding these two keys.

`n_workers` is a caller variable in `orchestrator.py::__main__` (shared across
all four steps), not a config key. So is `do_topo` (default `True`): set it to
`False` to skip step 3b entirely. Do this during the first few trials — `beta1_c`
(step 2) is only reliable once `beta1_list` brackets it, and the topo alpha runs
*at* that `beta1_c`, so an unconverged `beta1_c` would waste the expensive topo
MC. With `do_topo=False` the two `*_obs_fit_topo` combos get `alpha_c=None`
(recorded as skipped, not a failed fit), the two `*_1plaq` combos are unaffected,
and no `step3b_*` files are written; flip it back to `True` once step 2's
`beta1_c` is stable.

## Notes / pitfalls

- 2plaq takes a **single** plaquette coupling = `alpha_eff_f` (pre-sum
  `alpha_f + alpha1_f`).
- `file_mod = 0 if |beta1| < 1e-10 else mod` in every path token.
- Step 3b lives in the same `CPN_observable_fit/` subproject as step 2 and reuses
  its fine data when the sampler family matches: for z-renorm with a pure-β fine,
  step 3b's fine is halfRefVil (→ heatbath) — identical to step 2's — so
  `measure_PP_Wil`'s top-up just reuses the existing fine file. For z-renorm with
  `alpha_f ≠ 0` and for all U-renorm, step 3b needs a RefVil fine (different
  `data_RefVil/` file), so it runs its own fine sim.
- `alpha_c` from 1plaq and from the topo match should be roughly consistent (both
  estimate the renormalized Villain-Gaussian plaquette coupling); large
  disagreement usually means `alpha_list` does not bracket the true value, or the
  topo autocorrelation is undersampled. An empty `alpha_list` auto-brackets
  around the 1plaq value precisely because of this expected consistency.
