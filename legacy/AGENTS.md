# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## What this repository is

Lattice Monte Carlo research code for the **non-perturbative renormalization** of the 2D CP^{N-1} nonlinear sigma model coupled to a U(1) gauge field. The unifying goal: block (coarse-grain) a fine lattice by a factor `R = L_f/L_c` and match distributions or observables between fine and coarse lattices to extract **renormalized couplings**.

The results of the whole effort are tabulated in [renorm_results.typ](renorm_results.typ) (compiled to [renorm_results.pdf](renorm_results.pdf)) — read this first when you need to know what a "renormalized coupling" means here or what schemes exist.

This is a **monorepo of sibling subprojects**, not one program. Each subproject implements one renormalization scheme and is independently runnable. **Most subdirectories already have their own AGENTS.md — read the one for the subproject you're working in for the real detail.** This top-level file covers only the cross-cutting physics, conventions, and navigation.

## Subprojects → renormalization schemes

| Directory | Scheme | Fits | Docs |
|-----------|--------|------|------|
| [CPN_1plaq_fit/](CPN_1plaq_fit/) | 1plaq-Renorm (s- or z-) | renormalized `alpha` | AGENTS.md |
| [CPN_2plaq_fit/](CPN_2plaq_fit/) | 2plaq-Renorm (a- or z-) | renormalized `beta`, `alpha+alpha1` | *(none — see below)* |
| [CPN_4plaq_fit/](CPN_4plaq_fit/) | 4plaq raw-MC + sampler comparison | — (diagnostic / data gen) | *(none)* |
| [CPN_observable_fit/](CPN_observable_fit/) | Obs-Renorm (avg / corr) | renormalized `beta1` | AGENTS.md |

The 1plaq / 2plaq / Obs schemes each target different couplings and **compose**: 1plaq gives `alpha`, 2plaq gives `beta` and `alpha+alpha1`, Obs gives `beta1`.

## Running (cross-cutting)

No build system, no `requirements.txt`, **no test suite**, and no CLI argument parsers anywhere. Every driver is configured by editing parameter variables in its `if __name__ == "__main__"` block, then run as `python <script>.py` from that subproject's root (drivers import `from func.X import Y`, and the `func/*.py` modules `sys.path.append` their parent so their `__main__` smoke-test blocks also run standalone).

The conda env for all MC/scipy work is `myenv` (numpy / scipy / tqdm / matplotlib). Activate before any run:
```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate myenv
```
For per-subproject exact commands, see that subproject's AGENTS.md.

Restart safety is universal: data-generation and fit drivers skip any case whose output file already exists, so runs are safely resumable. (Exception: [CPN_observable_fit/](CPN_observable_fit/)'s `measure_PP_Wil.py` *tops up* — if a JSON exists but is missing a requested argzz/Wilson loop size, it runs a fresh sim for just those sizes and merges them in.) Parallelism is universal: data generation fans independent chains over `concurrent.futures.ProcessPoolExecutor` (`n_workers`), with `seed=None` giving independent streams because each worker is a fresh process.

## Shared physics conventions

These appear identically in every subproject — learn them once.

**Action ansatz**, couplings `(beta, beta1, alpha, alpha1)`:
- CP matter link term: `-2N·beta·Re(z̄ e^{ia} z)` per link.
- `beta1` deformation: `-N·beta1·|z†z|²` (`mod=0`) or `-sign(beta1)·log I0(2N·beta1·|z†z|)` (`mod=1`).
- Plaquette term: either `cos(da)` (half-Villain, used in 2/4plaq & Obs) or the full Villain `alpha/2·(da+2πs)² − alpha1·cos(da)` carrying an **integer** winding field `s` (used in 1plaq).

**`mod ∈ {0,1}`** selects the form of the `beta1` matter term, is threaded through `sweep`/`action`/force functions, and is embedded in **every** data and results path — so `mod=0` and `mod=1` runs are fully separated on disk. When `beta1 ≈ 0` the term vanishes, `mod` is physically irrelevant, and the pure-`beta` heatbath sampler is used instead (the driver keys off `|beta1| < ~1e-8`).

**Two sampler families**, present in every subproject's `func/`:
- **Original** (`func_CPN_Original*.py`): `beta1 = 0`. Local **heatbath + overrelaxation** on individual spins/links.
- **Refined Villain / "RefVil"** (`func_CPN_RefVil*.py`): `beta1 ≠ 0`. Global **HMC** with leapfrog + Metropolis on conjugate momenta of `z` and the gauge phase `a` (`U = e^{ia}`).

**Fields**: `z` shape `(Lx, Ly, N)` complex, per-site `|z|² = 1`; `a`/`U` link phase shape `(Lx, Ly, 2)`; `s` integer Villain field shape `(Lx, Ly)` (full-Villain only). `phi` is the real rep of a complex `z`, shape `(2N,)` alternating real/imag parts.

**Coupling / renormalization parameters**: `N` (CP^{N-1} model; N=2 ≅ O(3)); `L` (scale factor — different subprojects use different physical lattice sizes for the same `L`); `beta` (z–U coupling), `beta1` (RefVil `|z†z|²` coupling), `alpha`/`alpha1` (plaquette). `conn_type "U"` = Berry connection from link variables, `"z"` = from `z` inner products.

## Cross-cutting architecture patterns

**Two-stage Monte Carlo with a frozen boundary** is the core pattern in the 1plaq/2plaq/4plaq fit subprojects: (1) thermalize a *larger padded* patch to decorrelate, extract its boundary loop of `z`/`U`; (2) freeze that boundary into the small target sampler and measure the interior. `boundary_pad` (default 5) is the spatial padding; it is a *different* token from any `zpad`/histogram zero-padding in filenames.

**`a` is never wrapped during HMC integration** — the Villain plaquette force needs the continuous field. It is wrapped into `[-π,π]` once at trajectory end, simultaneously shifting integer `s` so the physical plaquette variable `da + 2π·s` stays invariant. `s` is updated by a separate Metropolis step on top of the HMC trajectory.

**Frozen-boundary NaN guard**: frozen sites carry zero momentum, which makes the geodesic `z`-update divide by zero. Samplers that freeze boundaries override `hmc_step` with a `safe_mask = p_z_norm > 1e-15` guard — keep it if you reuse the pattern.

**`curve_fit` bounds**: fit scripts pass `bounds=(0, np.inf)` because the Villain model uses `sqrt(alpha)`/`sqrt(beta)` and an unbounded optimizer probes negative values → NaN.

**`integrated_autocorr_time`** (FFT autocorrelation, window cutoff c=5.0) is defined near-identically in several `func_` modules; error bars scale as `sqrt(2τ_int/M)`.

## ⚠️ Naming hazard: "RefVil" is inconsistent across directories

Files/classes named `RefVil` mean **different actions** in different subprojects:
- In [CPN_1plaq_fit/](CPN_1plaq_fit/), `RefVil` = the **full Villain** action (integer `s`, `alpha1`).
- In [CPN_2plaq_fit/](CPN_2plaq_fit/) and [CPN_4plaq_fit/](CPN_4plaq_fit/), `RefVil` = the **half-refined** action (no `s`, no `alpha1`, just the `beta1·|z†z|²` deformation).
- In [CPN_observable_fit/](CPN_observable_fit/), **both** actions now coexist with distinct filenames: `func_CPN_halfRefVil_HMC.py` (`CPN_halfRefVil_HMCSampler`, half-refined) and `func_CPN_RefVil_HMC.py` (`CPN_RefVil_HMCSampler`, full Villain with `s`/`alpha1`); `measure_PP_Wil.py` selects between them via `obs_fit_type`.

Do **not** copy a `func_CPN_RefVil*.py` from one subproject as a physics reference for another without checking which action it implements. Trustworthy full-Villain references live in `CPN_1plaq_fit/` and `CPN_observable_fit/func_CPN_RefVil_HMC.py`.

## Subprojects without their own AGENTS.md

- **[CPN_2plaq_fit/](CPN_2plaq_fit/)** — the 2-plaquette fit: `CPN_2plaq_data_generate.py` (two-stage MC with frozen boundary) → `fit_a_dist_batch.py` (scipy `curve_fit` of `Villain_dist_norm_fit` in `func/model.py`) → `fit_visualize.ipynb`. Uses the half-refined Villain action.
- **[CPN_4plaq_fit/](CPN_4plaq_fit/)** — 4-plaquette raw-MC data generation (`CPN_2plaq_data_generate_raw.py`) and a sampler cross-check ([compare_samplers.py](CPN_4plaq_fit/compare_samplers.py), which benchmarks HMC vs heatbath energy density and Wilson loops over parallel chains).
