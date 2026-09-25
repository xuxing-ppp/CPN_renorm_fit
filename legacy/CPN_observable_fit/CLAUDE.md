# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Lattice field theory Monte Carlo code for the CP^{N-1} nonlinear sigma model coupled to a U(1) gauge field on a 2D periodic lattice. The model uses a "half-Villainized" formulation (cos(da) plaquette term but no integer gauge field). A deformation coupling β1 adds a `|z† z|²`-type term. The goal is non-perturbative renormalization — matching bare couplings across lattice sizes so that physical observables agree.

The action ansatz (see `results.typ` for the full methodology) is
`S = -N Σ_links (2β Re(z̄_j e^{ia} z_i) + β1·|z̄_j z_i|²) - α Σ_plaq cos(δa)`,
characterized by three couplings (β, β1, α). Renormalization blocks a fine lattice `(β_f, β1_f=0, α_f=0)` by factor `R = L_f/L_c` to a coarse lattice and finds the matched coarse `β1_c` by fitting the coarse-lattice observable (scanned over β1) with an error-weighted smoothing spline that uses **all** scan points and reads off where it equals the fine-lattice value (see `match_coupling` in `plot_PP_Wil_beta1.py`).

## Running

All commands need the conda env `myenv` (numpy/scipy/tqdm/matplotlib). For test/verification runs, always activate it first:
```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate myenv
```

```bash
# Monte Carlo measurement sweep (edit __main__ block for parameters, incl. mod)
python measure_PP_Wil.py

# RG analysis & plotting — interactive
jupyter notebook plot_PP_Wil_beta1.ipynb

# RG analysis & plotting — batch script, driven by renorm_info.json files
python plot_PP_Wil_beta1.py --mod 0 --list                 # list cases for a mod
python plot_PP_Wil_beta1.py --mod 0                        # run all cases for that mod
python plot_PP_Wil_beta1.py --mod 0 --case <name-substr>   # run one case
```

No build system, no requirements.txt.

## Architecture

### The `mod` parameter (action variant of the β1 term)

`mod ∈ {0,1}` selects how the HMC sampler treats the deformation term (only matters when β1 ≠ 0):
- **mod=0**: `β1·|z†z|²` quadratic form (the original).
- **mod=1**: `log I0(2N·β1·|z†z|)` Bessel form (see `action_density`/`_z_force` in both `func_CPN_halfRefVil_HMC.py` and `func_CPN_RefVil_HMC.py`).

`mod` is threaded through `measure_PP_Wil.py` and is part of **every** data and results path, so mod=0 and mod=1 runs are fully separated. When β1≈0 the pure-β heatbath sampler is used instead and `mod` is physically irrelevant — but the path still carries it, so the fine-lattice (β1_f=0) data is identical across mods in physics and can be copied between mod folders rather than re-simulated.

### `obs_fit_type` (action family for the β1≠0 model)

`obs_fit_type ∈ {"halfRefVil", "RefVil"}` (default `"halfRefVil"`) selects which sampler simulates the β+β1 model, and routes data into `data_{obs_fit_type}/`. Results go under a **unified `results/`** tree: `results/PP_Wil_beta1_renorm_plot_mod{mod}_{obs_fit_type}/` for the β1 obs fit, and `results/Topo_alpha_renorm_plot_mod{mod}/` for the topo-α fit (always RefVil coarse). It is a parameter of `measure_PP_Wil.main(...)` and `plot_PP_Wil_beta1.process_case(...)` (read from the case/info dict), and an orchestrator-level variable in `full_renorm`. See "Three samplers" below for the action each selects.

### Three samplers (in `func/`)

- **`func_CPN_HB.py` — `CPNSampler`**: Pure β model (β1 = 0). Local **heatbath + overrelaxation** on individual z-spins and U(1) links. `sweep(heatbath_fraction)`. `topo_charge()` → `(Q_U, Q_z)`.
- **`func_CPN_halfRefVil_HMC.py` — `CPN_halfRefVil_HMCSampler`**: β + β1, **half-Villain** action (plaquette `−α·(cos(da)−1)`, no integer `s`, no `α1`). Global **HMC** with leapfrog + Metropolis on conjugate momenta of `z` and the gauge phase `a` (`U = e^{ia}`). `sweep(mod=mod)`. `topo_charge()` → `(Q_U, Q_z)`. Selected for `obs_fit_type="halfRefVil"` when β1≠0.
- **`func_CPN_RefVil_HMC.py` — `CPN_RefVil_HMCSampler`**: β + β1 + α + α1, **full-Villain** action (integer winding field `s`, plaquette `α/2·(da+2πs)² − α1·(cos(da+2πs)−1)`). Global HMC for `z`/`a` plus a per-plaquette Metropolis update of `s` (`s_step`/`s_update_num`). `__init__(N, Lx, Ly, β, β1, α, α1, ...)` — note the `(N,Lx,Ly)` order. `sweep(mod=mod)`. `topo_charge()` → `(Q_U, Q_z, Q_s)` with `Q_s = Σs`. Selected for `obs_fit_type="RefVil"` (always, even when β1≈0, since the Villain plaquette needs it).

All three expose the same measurement API: `P_exp()`, `conn_PP_corr()` (full `Lx×Ly` connected correlator), `argzz_loop_xy_list(xy)` → `(len,)` cos(argzz) loop, `wilson_loop_xy_list(xy)` → `(len,2)` `[Re(U-plaquette), Re(Uz-plaquette)]`, `poly_loop()`, `topo_charge()`. `measure_PP_Wil.py::_worker()` picks the sampler by `obs_fit_type` first (RefVil), then `|beta1| < 1e-8` (heatbath vs halfRefVil). `topo_charge()` is now **measured every run** and stored per component (see below).

### Data layout & measurement flow

1. `measure_PP_Wil.py::main()` loops over (β, β1, α, α1) combos (4-tuples; halfRefVil folds `α1=0`), launches parallel workers via `ProcessPoolExecutor`, thermalizes, then measures every `meas_interval` sweeps.
2. Each measurement file is written to
   `data_{obs_fit_type}/data_PP_corr_N{N}_L{L}_mod{mod}/N{N}_L{L}_beta{beta:.3f}_alpha_eff{alpha_eff:.3f}/PP_corr_N{N}_L{L}_beta{beta:.3f}_beta1_{beta1:.3f}_alpha{alpha:.3f}_alpha1_{alpha1:.3f}.json`
   where `alpha_eff = alpha + alpha1`. The subfolder is keyed by `(β, alpha_eff, mod)` — NOT by α/α1 separately — so the obs β1-scan and the topo α-scan at the same `alpha_eff` share a subfolder. Within it, files differ by β1 (a β1-scan at fixed α,α1) and/or by α,α1 (a topo α-scan at fixed β1, with `alpha1 = alpha_eff − alpha`). `measure_PP_Wil.filepath_for(N,L,beta,beta1,alpha,alpha1,mod,obs_fit_type)` returns this path for callers (`plot_topo_alpha.py`, `full_renorm/step3b_topo.py`).
3. Each JSON stores `N`, `mod`, `obs_fit_type`, couplings (incl. `alpha1`), sweep params (incl. `s_step`/`s_update_num`), the two measurement lists `argzz_xy_list` / `wilson_xy_list`, and per-observable mean/err/τ:
   - `argzz_loop[f"{x}x{y}"] = {tau, mean, err}` (scalar — cos(argzz)),
   - `wilson_loop[f"{x}x{y}"] = {tau:[2], mean:[U, Uz], err:[2]}`,
   - `conn_PP_corr.mean` (full `Lx×Ly` matrix), `P_exp`,
   - `topo_charge[name] = {mean, mean2, topo_sus=(mean2−mean²)/V, tau, err}` for `name ∈ {Q_U, Q_z}` (halfRefVil/HB) or `{Q_U, Q_z, Q_s}` (RefVil).
   (Old files used a single `xy_list` and a 3-column `wilson_loop` `[U, Uz, argzz]`; that format is no longer produced.)
4. **Restart safety / top-up**: if a (β,β1,α) JSON already exists, `main()` checks the `argzz_loop`/`wilson_loop` keys against the requested sizes — if all are present it skips; otherwise it runs a fresh sim measuring **only** the missing sizes and merges them into the file (existing observables, `P_exp`, `conn_PP_corr` kept as-is). The missing sizes come from a new independent ensemble at the same couplings — valid because each observable's mean is an independent expectation value and the matching interpolates them separately.
5. **Seed handling**: `seed=None` in `_worker()` — each worker gets a different RNG state since they run in separate processes. The deterministic-seeding line is commented out.

### Renormalization analysis (`plot_PP_Wil_beta1.{ipynb,py}`)

Fine lattice: fixed `(β_f, β1_f=0, α_f=0)` — one data file. Coarse lattice: a β1-scan at `(β_c, α_c)` for the same `mod`. For each observable the matched `β1_c` is found by `match_coupling`: an error-weighted GCV smoothing spline (weighted low-order polynomial when the scan has <5 points) is fit through **all** coarse points and inverted to find where it crosses the fine target — so every scan point and every error bar contributes, unlike two-point linear interpolation. The matched `β1_c` also carries a Monte-Carlo error bar from the per-point / per-target errors; `linear_interpolate` is retained as its anchor and fallback.

The three observables are **fully separated**, each driven by its own list of coarse `(x,y)` separations (fine separation = `(x,y)·L`), passed in the case/info dict:
- **PP correlation** (`pp_xy_list_coarse`): indexes the full `conn_PP_corr` matrix — coarse `conn_PP_corr[x,y]`, fine `conn_PP_corr[x·L, y·L]`.
- **argzz loop** (`argzz_xy_list_coarse`): reads the scalar `argzz_loop[key]["mean"]` store.
- **Wilson loop** (`wilson_xy_list_coarse`): reads the 2-component `wilson_loop[key]["mean"]` store.

Result keys are `"(x,y)"` (not `"1x1"`). The reported `avg` is a **single global mean** of every matched `β1_c` across all three observables; three further matches are kept separate and excluded from `avg`: the PP **correlation-length** match (`corr`), the **magnetic-susceptibility** match (`magsus`, `mag_sus = sum(conn_PP_corr)`, coarse scaled by `L**2`), and the **topological-susceptibility** match (`toposus`, RefVil only). For `toposus` the coarse `topo_sus(Q_s)` is matched against the **fine** susceptibility scaled by `L**2`, with the fine component chosen by `renorm_type` — **z → `Q_z`**, **U → `Q_s`**; it is skipped when `Q_s` is unavailable (halfRefVil). (`process_case` falls back to a legacy `xy_list_coarse`/`xy_list` for the selection lists if the three keys are absent.)

- **renorm_type `'U'` vs `'z'`**: controls which Wilson-loop component the *fine* lattice is matched against — `U` (component 0) vs `Uz` (component 1, with `U_z = z†z/|z†z|`). The coarse Wilson loop is always the standard U(1) loop (component 0), so z-type is a mixed comparison testing whether z-blocked and gauge-blocked Wilson loops respond similarly to β1_c.
- argzz and Wilson used to come from one 3-column `wilson_loop_xy_list`; they are now separate methods/stores (argzz scalar in `argzz_loop`, Wilson 2-column `[U, Uz]` in `wilson_loop`).

The `full_renorm/step3_observable.py` driver is the main consumer of this schema: it calls `measure_PP_Wil.main(..., argzz_xy_list, wilson_xy_list, obs_fit_type=..., ...)` (PP needs no list — full correlator), then `process_case` with the three `*_xy_list_coarse` lists, and returns `beta1_c = {avg, corr, magsus, toposus}` (`toposus` is null for halfRefVil).

The **notebook** writes `renorm_info.json` (the case spec: fine/coarse params, β1_c scan, renorm, type, mod, obs_fit_type, alpha1_f/alpha1_c) plus `matching_results.json` and plots into
`results/PP_Wil_beta1_renorm_plot_mod{mod}_{obs_fit_type}/N{N}_L{renorm}_{renorm_type}/beta{beta_f}_beta1_{beta1_f}_alpha{alpha_f}_alpha1_{alpha1_f}/` — classified by `(obs_fit_type, N, L=renorm, renorm_type)`, then by the fine model couplings. Plot titles carry `beta_c, alpha_eff_c` (halfRefVil) or `beta_c, alpha_c, alpha1_c` (RefVil).

The **`.py` script** is the batch form: it discovers cases by globbing those `renorm_info.json` files two levels under `--obs_fit_type`/`--mod` (the `results/PP_Wil_…_mod{mod}_{type}/N{N}_L{L}_{renorm_type}/<leaf>/` tree), then re-runs the matching for each. `--case` matches the path relative to the per-(type,mod) root. Hand-off pattern: define a case interactively in the notebook (which writes `renorm_info.json`), then reproduce/refresh it headless with the `.py`.

### Topo-α analysis (`plot_topo_alpha.py`)

The topo-susceptibility **alpha** fit — the α-determination companion to the 1plaq vortex fit. At fixed `(beta_c, beta1_c, alpha_eff_c)` it scans coarse `alpha` (with `alpha1 = alpha_eff_c − alpha`) on a **RefVil** lattice and matches the coarse integer-winding susceptibility `topo_sus(Q_s)` onto the fine susceptibility scaled by `L**2` (fine component `Q_z` for z-renorm, `Q_s` for U-renorm). Data is produced by the same `measure_PP_Wil.main` (which also records `conn_PP_corr`), so `process_case_topo(info, mod)` emits three plots per β1_c under `results/Topo_alpha_renorm_plot_mod{mod}/N{N}_L{renorm}_{renorm_type}/beta{beta_f}_beta1_{beta1_f}_alpha{alpha_f}_alpha1_{alpha1_f}/beta1_c_{beta1_c:.3f}/`: `topo_sus.png` (the fit, annotated with α_c), plus **informational** `mag_sus.png` and `corr_len.png` (coarse vs α, fine target line — not used for fitting). Titles carry `beta_c, beta1_c, alpha_eff_c`. `full_renorm/step3b_topo.py` is the consumer; it runs the coarse scan per β1_c (magsus, corr) and shares one fine sim. (The old standalone `CPN_observable_fit_topo/` subproject is merged away — `measure_PP_Wil` already measured topo charge, so no separate `measure_topo.py` is needed.)

### `integrated_autocorr_time`

Defined identically in **all three** `func_` modules (FFT autocorrelation, automatic window cutoff c=5.0; works on real or complex arrays, reducing multidimensional input along the last axis). Imported from `func_CPN_HB` in `measure_PP_Wil.py`. Error bars on observables scale as `sqrt(2τ_int / M)`; it is also used for the topological-charge series in `_topo_stats`.
