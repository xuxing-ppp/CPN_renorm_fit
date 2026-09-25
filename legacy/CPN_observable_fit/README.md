# CPN\_observable\_fit — observable matching for non-perturbative renormalization

Lattice Monte Carlo analysis for the **observable-based renormalization step** of
the 2D CP^{N−1} nonlinear sigma model coupled to a U(1) gauge field. A fine
lattice is blocked (coarse-grained) by a factor `R = L_f / L_c`, and the matched
coarse coupling is found by requiring that coarse-lattice observables — scanned
over that coupling — reproduce the fine-lattice values.

This directory implements two such matches:

| Fit | Coarse coupling scanned | Matched coupling | Driver |
|-----|------------------------|------------------|--------|
| **β1 obs fit** | `beta1_c` at fixed `(beta_c, alpha_c, alpha1_c)` | **`beta1_c`** | `plot_PP_Wil_beta1.{py,ipynb}` |
| **topo-α fit** | `alpha_c` at fixed `(beta_c, beta1_c, alpha_eff_c)` | **`alpha_c`** | `plot_topo_alpha.py` |

The action ansatz is `S = -N Σ_links (2β Re(z̄_j e^{ia} z_i) + β1·|z̄_j z_i|²) − α Σ_plaq cos(δa)`
(half-refined Villain), or the full-Villain form with an integer winding field `s`
and a separate `alpha1` for `obs_fit_type="RefVil"`. See [`results.typ`](results.typ)
for the full methodology and how this step composes with the 1plaq / 2plaq schemes.

---

## 1. Fitting pipeline (rough)

### 1a. Data generation — `measure_PP_Wil.py`

`measure_PP_Wil.main(...)` runs the MC (parallel `ProcessPoolExecutor`): thermalize,
then measure every `meas_interval` sweeps. Each coupling point is written as one
JSON under

```
data_{obs_fit_type}/data_PP_corr_N{N}_L{L}_mod{mod}/
    N{N}_L{L}_beta{beta:.3f}_alpha_eff{alpha+alpha1:.3f}/
        PP_corr_N{N}_L{L}_beta{beta:.3f}_beta1_{beta1:.3f}
            _alpha{alpha:.3f}_alpha1_{alpha1:.3f}.json
```

(subfolder keyed by `alpha_eff = alpha + alpha1`, so a β1-scan and an α-scan at the
same `alpha_eff` share it). Each JSON stores, **per observable, a mean, an error
`err`, and an integrated autocorrelation time `tau`** (error ≈ `std·√(2τ/M)`):

- `conn_PP_corr.mean / .err` — full `L×L` connected Polyakov–Polyakov correlator.
- `argzz_loop["{x}x{y}"]` — scalar `cos` of the summed `arg(z̄z)` around a rectangle.
- `wilson_loop["{x}x{y}"]` — 2-component `[U-plaquette, Uz-plaquette]`.
- `topo_charge[Q_U | Q_z | Q_s]` — topological charge with `topo_sus = (⟨Q²⟩−⟨Q⟩²)/V` (`Q_s` RefVil only).

### 1b. The β1 observable fit — `plot_PP_Wil_beta1.{py,ipynb}`

Fine lattice: one point at `(β_f, β1_f=0, α_f=0)`. Coarse lattice: a scan over
`beta1_c` at fixed `(β_c, α_c, α1_c)` for the same `mod`.

For each observable the matched `beta1_c` is read off where the coarse observable
curve (vs `beta1_c`) equals the fine target — by the smoothing-spline fit of
§2. The fine separation is `(x,y)·R` matched to the coarse `(x,y)`.

| Observable | coarse y | fine target | in the headline `avg`? |
|------------|----------|-------------|------------------------|
| PP correlator entry `conn_PP_corr[x,y]` | `conn_PP_corr_c[x,y]` | `conn_PP_corr_f[x·R, y·R]` | **yes** |
| `argzz` loop | `argzz_loop_c[k]["mean"]` | `argzz_loop_f[k·R]["mean"]` | **yes** |
| Wilson loop (coarse comp 0) | `wilson_loop_c[k]["mean"][0]` | `wilson_loop_f[k·R]["mean"][idx]` (`idx`: U→0, z→1) | **yes** |
| PP correlation length | `corr_len·R` | `corr_len` | no (own output `corr`) |
| magnetic susceptibility `Σ conn_PP_corr` | `Σ·R²` | `Σ` | no (own output `magsus`) |
| topological susceptibility `topo_sus(Q_s)` | `topo_sus(Q_s)` | `topo_sus(fine_comp)·R²` | no (own output `toposus`, RefVil only) |

The **headline `beta1_c = avg`** is the simple (unweighted) mean of every matched
`beta1_c` across PP-corr + argzz + Wilson; `corr`/`magsus`/`toposus` are reported
separately and excluded from the mean. Each entry also carries an MC error bar
(§2.4), stored in `matching_results.json` as `beta1_c_*_err`.

### 1c. The topo-α fit — `plot_topo_alpha.py`

At fixed `(β_c, β1_c, α_eff_c)` it scans coarse `alpha_c` on a **RefVil** lattice
(with `alpha1 = α_eff_c − alpha`), and matches the integer-winding susceptibility
`topo_sus(Q_s)` onto the fine susceptibility scaled by `R²`
(`fine_comp = Q_z` for z-renorm, `Q_s` for U-renorm). The same smoothing-spline
fit gives the matched `alpha_c` (and its MC error). `mag_sus`/`corr_len` vs `alpha`
are plotted for information but not fit.

### 1d. Outputs

Per case, under `results/PP_Wil_beta1_renorm_plot_mod{mod}_{obs_fit_type}/...`
(β1 fit) or `results/Topo_alpha_renorm_plot_mod{mod}/.../beta1_c_{...}/` (topo fit):

- one PNG per observable (PP-corr entries, `PP_corr_len`, `mag_sus`, `topo_sus`,
  argzz entries, Wilson entries) showing **coarse points with error bars**, the
  **fine-target line**, the **green smoothing-spline fit**, and a **red dashed
  vertical line** at the matched coupling labelled with its MC error;
- `matching_results.json` / `topo_alpha_results.json` (numbers incl. `*_err`);
- `renorm_info.json` (the case spec, written by the notebook, replayed by the `.py`).

---

## 2. The fitting method — error-weighted smoothing spline (in detail)

### 2.1 Why not two-point linear interpolation

The old helper `linear_interpolate(x, y, target)` found the two adjacent scan
points bracketing `target` and drew a straight line between them. The matched
coupling therefore depended on **exactly those two points**: if either was noisy,
the answer was biased, and **no other scan point could mitigate it**. It also
ignored the per-point error bars entirely.

### 2.2 The replacement — `match_coupling` (`plot_PP_Wil_beta1.py`)

Fit a smooth curve `y(x)` through **all** scan points (error-weighted), then read
off where it crosses the fine target. Every point and every error bar contributes.

```
beta1_c, beta1_c_err = match_coupling(x_vals, y_vals, y_target,
                                       y_err=..., y_target_err=...)
```

Returns `(x_match, x_err)`, or `(None, None)` if no in-range crossing exists.

### 2.3 The curve — GCV smoothing spline, polynomial fallback

The curve is built by `_fit_callable(x, y, w)`:

- **`n ≥ 5` points → error-weighted GCV smoothing spline**
  (`scipy.interpolate.make_smoothing_spline`). A cubic smoothing spline minimizes

  ```
  Σᵢ wᵢ (yᵢ − f(xᵢ))²  +  λ ∫ (f ″(x))² dx ,
  ```

  with the smoothing parameter `λ` chosen automatically by **generalized
  cross-validation (GCV)**. The first term pulls `f` toward the data (weighted by
  `wᵢ`); the second penalizes curvature. GCV picks the `λ` that best predicts
  held-out data, so the curve follows the systematic trend while averaging out
  point-to-point noise — exactly using every point rather than a bracketing pair.

- **`n < 5` points (too few for the spline) → weighted low-order polynomial**,
  `np.polyfit` of degree `min(2, n−2)` (≥1 dof so it smooths; `n=2` → a line,
  equivalent to the old linear interpolation).

**Weight convention.** `make_smoothing_spline` enters weights **linearly**
(`Σ wᵢ (·)²`, *not* squared as in `splrep`). For chi-square weighting we therefore
use `wᵢ = 1/σᵢ²`. Weights are built by `_make_weights`, which clamps any
non-positive/missing `σᵢ` to the median of the good ones; if **no** usable errors
are supplied, the fit is unweighted.

### 2.4 Inverting the curve and propagating the error

1. **Anchor.** Compute the two-point linear location `x_lin` on the raw points.
2. **Find crossings.** Evaluate `f(x) − target` on a dense 1001-point grid over
   `[x_min, x_max]`, locate sign changes, refine each with Brent's method
   (`_roots_in_range`). Among all real in-range roots, `_select_root` keeps the
   one **nearest `x_lin`** — so a wiggly spline can't jump to a spurious crossing.
3. **Fallback chain.** Spline → weighted polynomial → `x_lin` (the old answer).
   The fit is never worse than two-point linear interpolation.
4. **Monte-Carlo error (`x_err`).** Parametric bootstrap, `n_mc = 256` draws: in
   each draw perturb every `yᵢ → yᵢ + σᵢ·N(0,1)` and the target
   `→ target + σ_target·N(0,1)` (weights held fixed = the known per-point
   precision), refit + reinvert, and collect the resulting `x`. The standard
   deviation of those samples is `x_err`. This folds in **both** the coarse-point
   errors and the fine-target error.

When errors are unavailable the fit is unweighted and `x_err = None`. `corr_len`
(FFT-derived) has no per-point error, so it is always fit unweighted with
`x_err = None`.

### 2.5 What gets plotted (`add_coarse_points`, `add_fit_curve`)

Each observable plot shows:

- the **coarse scan as markers joined by a line, with per-point error bars**
  (`plt.errorbar`, cap size 3) — for every observable **except `corr_len`**, which
  has no per-point error and so is drawn without bars;
- the **fine-target horizontal line**;
- the **green smoothing-spline fit** (`fit_curve` evaluated on a 200-point grid),
  i.e. the actual curve `match_coupling` inverts;
- a **red dashed vertical line** at the matched coupling, annotated with the MC
  error (e.g. `beta1 = −0.8903 +- 0.0024`).

### 2.6 Aggregation

The headline `beta1_c` is the **simple unweighted mean** of the per-observable
matched values (PP-corr + argzz + Wilson). The per-observable MC errors are
reported individually (and stored in `matching_results.json`) but do **not**
weight the mean.

---

## 3. Running

Activate the env first:
```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate myenv
```

```bash
# 1. generate data (edit __main__ params: beta1/alpha scans, mod, obs_fit_type, …)
python measure_PP_Wil.py

# 2a. β1 fit — interactive (sets fine params, auto-discovers a coarse β1-scan,
#     writes renorm_info.json + matching_results.json + plots)
jupyter notebook plot_PP_Wil_beta1.ipynb

# 2b. β1 fit — batch (replays every renorm_info.json two levels under the results root)
python plot_PP_Wil_beta1.py --mod 0 --list                 # list cases
python plot_PP_Wil_beta1.py --mod 0                        # run all for that mod
python plot_PP_Wil_beta1.py --mod 0 --case <name-substr>   # run one case

# 3. topo-α fit (self-test of the fit math on synthetic data)
python plot_topo_alpha.py
```

`process_case(info, mod)` / `process_case_topo(info, mod)` are also imported by
the external orchestrator `full_renorm/step3_observable.py` / `step3b_topo.py`.

Runs are restart-safe: `measure_PP_Wil.py` skips coupling points whose JSON
already exists (topping up missing loop sizes only).

---

## 4. Files

| File | Role |
|------|------|
| `measure_PP_Wil.py` | MC data generation; `filepath_for(...)` for callers |
| `plot_PP_Wil_beta1.py` | β1 fit (batch): `process_case`, `match_coupling`, `fit_curve`, `add_fit_curve`, `add_coarse_points`, `linear_interpolate` |
| `plot_PP_Wil_beta1.ipynb` | β1 fit (interactive); same pipeline as the `.py` |
| `plot_topo_alpha.py` | topo-α fit: `process_case_topo`, `default_alpha_list` |
| `func/func_CPN_HB.py` | pure-β heatbath sampler (`β1=0`) |
| `func/func_CPN_halfRefVil_HMC.py` | β+β1 half-Villain HMC sampler |
| `func/func_CPN_RefVil_HMC.py` | β+β1+α+α1 full-Villain HMC sampler (+ integer `s`) |
| `data_halfRefVil/`, `data_RefVil/` | measured JSONs (per `obs_fit_type`) |
| `results/` | per-case plots + `matching_results.json` / `topo_alpha_results.json` |
| `results.typ` | full methodology write-up |
