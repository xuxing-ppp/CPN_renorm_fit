# Torch/CUDA CPN renormalization pipeline

This repository contains a new batched PyTorch implementation of the refined
four-coupling CP(N-1) renormalization pipeline. `legacy/` remains the numerical
reference and is not imported by the new package.

The pipeline runs **pilot → 1plaq alpha → 2plaq beta/alpha1 → obs beta1 →
an optional topo alpha correction → 2plaq beta/alpha1 refit at the corrected alpha**.
The refit runs only after a successful topo match and reuses the original 2plaq
data. Both plaquette stages sample the complete fine
action. At fine alpha=0 the equivalent cosine representation avoids sampling
unconstrained integers. The 2plaq result contains `old` (alpha=0 cosine model,
beta≥0 and alpha_eff≥0) and `new` (exact Villain conditional density with
independently fixed 1plaq alpha, beta≥0 and signed alpha1), fitted to identical
bin-center data. `steps.two_plaq.p0` initializes the old control. The new fit uses
the old beta with alpha1=0 and alpha1=old alpha_eff as two starts; the latter is
only an initial value. Only `new` feeds later stages. Its MSE may exceed the old
control when the independent matching conditions disagree; this never triggers
a fallback or iteration. `alpha_eff` belongs only to the old control.

Obs uses the full fine action and independent coarse beta, alpha, alpha1, scanning
only beta1. Topo then fixes beta, alpha1 and each method's matched beta1 and changes
only alpha. A final 2plaq fit then adjusts beta and alpha1 at that fixed alpha,
without further beta1 or alpha scans. Observable/topo matching is not rechecked
after this final adjustment; the result is a single sequential correction.

## Running

Activate a Python environment containing PyTorch, NumPy, SciPy and Matplotlib,
then run the package directly from the repository root. Installing this project
itself is not required.

Check the configuration and detected device:

```bash
python -m cpn_renorm validate configs/example.toml
```

Run only the correlation-length pilot or the full pipeline:

```bash
python -m cpn_renorm pilot configs/example.toml --output-dir runs
python -m cpn_renorm run configs/example.toml --output-dir runs
```

For `run`, `pilot`, `validate`, and `tune-chains`, a bare filename is searched
recursively under the current working directory's `configs/`:

```bash
python -m cpn_renorm run 1.toml
python -m cpn_renorm run folder0/folder1/1.toml
```

If several files have the same name, the command lists the candidates and
requires a path. Paths containing directories are checked as given first,
then relative to `configs/`; absolute paths are also supported.
To select a bare filename outside `configs/`, pass its absolute path.

`run` accepts a single TOML file. To run all TOML files in a folder, use
`batch` with a quoted glob pattern:

```bash
# Only TOML files directly inside this folder
python -m cpn_renorm batch "configs/my_folder/*.toml" --output-dir runs

# Include TOML files in all subfolders as well
python -m cpn_renorm batch "configs/my_folder/**/*.toml" --output-dir runs
```

Keep the pattern in quotes so the CLI expands it. Batch patterns are relative
to the current working directory (or absolute); include `configs/` explicitly
when needed. Matched files run sequentially in sorted path order. An error
stops the batch; an unmatched pattern also reports an error. The batch command
also supports `--device`, `--seed`, `--force`, and `--accept-changes`.

Outputs mirror the categories under `configs/`, dropping the `.toml` suffix:
`configs/folder0/folder1/1.toml` writes to `runs/folder0/folder1/1/`.
`--output-dir` replaces only the output root. External configurations still
use `<output-root>/<filename-without-suffix>/`. This applies to batch runs
and tuning reports as well.
Existing flat output directories are retained but are not automatically
migrated or reused for configurations now stored in categories.

Set `[runtime] auto_chains = true` to tune batch widths automatically during
`run` (including the pilot before sampling each lattice size). With `false`,
the pipeline uses the seven `[chains]` values: `pilot`, `two_plaq`, `one_plaq`,
`observable_fine`, `observable_coarse`, `topo_fine`, and `topo_coarse`.
Fine and coarse widths are selected independently. The terminal prints each
selection; `chain_tuning.json` stores the search and `resolved.json` records
the widths actually used by each branch.

To benchmark independently:

```bash
python -m cpn_renorm tune-chains configs/example.toml --output-dir runs
```

This runs or reuses the pilot, benchmarks the pilot, two-plaquette, observable,
one-plaquette and topology workloads separately, and writes
`runs/example/chain_tuning.json`. It prints a TOML snippet but never modifies
the input configuration. The recommendation is the smallest safe chain count
within 90% of the best measured throughput; by default a candidate may use at
most 80% of CUDA memory. The host-memory budget is the smaller of 8 GiB and
half of currently available RAM (including Linux cgroup limits).
Plaquette estimates use the maximum measurements per boundary and include
history, diagnostic temporaries, accumulated histograms, warmup probes, and
CPU sampler workspace when sampling on CPU.
Adjust those limits with `--gpu-memory-fraction`, `--max-host-memory-gb`, and
`--max-chains`.
Before plaquette generation, the same host-memory check is applied to manual,
tuned, and restored chain counts. An unsafe width raises an error before
allocating the history; lower the stage width or retune. These are conservative
estimates with headroom, rather than a reservation against other processes.

All commands for a configuration named `example.toml` share the
`runs/example/` workspace. Re-running reuses every compatible pilot, completed
fit batch, and content-addressed ensemble. Pass `--force` to regenerate the
work used by the current command.

`renormalization.type` is a nonempty list such as `["z"]` or `["z", "U"]`.
The two- and one-plaquette Monte Carlo batches measure all available connection
or vortex definitions together and fit U/z separately, with type-specific
`p0` values. The list controls which observable/topology branches are run under
`runs/example/branches/<type>_mod<mod>/`. `model.mod` is a unique, nonempty
array such as `[1]` or `[0, 1]`; scalar mod and the old unsplit chains fields
are rejected. At fine `beta1 = 0`, all mods share the pilot and plaquette
results. At nonzero `beta1`, each mod has its own pilot geometry and common
stages under `common/mod<mod>/`. Adding `"U"` later reuses the pilot and common
plaquette batches and preserves the completed z branch. For `model.alpha = 0`,
the U branch adopts `alpha_zero_subspace_convention`: alpha=0, no U 1plaq
identification and no topo correction, while 2plaq and obs continue.

Fitted independent couplings retain full floating-point precision in
subsequent scans and exported TOML. The final summary includes direct fit
errors; these do not include propagation of upstream uncertainties.

Each `summary.json` branch (`<type>_mod<mod>`) contains one
`two_plaq_comparison`: `old_alpha_eff`, `new_alpha_plus_alpha1`, and their signed
`difference` (new sum minus old alpha_eff), all from the initial 2plaq stage.
The `*_1plaq` results retain the initial couplings; `*_obs_fit_topo` results and
generated next-step TOML use the final refitted beta/alpha1. Only topo results include
`topo_correction`, containing signed `alpha`, `beta`, and `alpha1` changes relative
to the corresponding 1plaq result. Final beta/alpha1 errors are conditional on the
fixed topo alpha. Detailed fit and MSE diagnostics live in `step4_topo/result.json`.

The workspace records the last accepted effective configuration, including
defaults and command-line device/seed overrides. If the configuration changes,
the CLI prints every changed value and marks each pipeline stage as `reuse`,
`reprocess`, or `resimulate`. An interactive terminal asks for confirmation;
scheduled/non-interactive jobs must pass `--accept-changes`. The `batch`
command performs this check for all matched files before starting simulations.

After changing `scan.beta1.points`, `min`, or `max`, compatible retained beta1
ensembles are reused and missing points are simulated. Every full run estimates
1plaq alpha before 2plaq, then performs obs and a single topo correction for each
selected beta1 method when `alpha_topo_fit=true`. After a successful pipeline run,
initial/coarse-expansion points and points actually used in the final fits are retained as JSON
measurement caches. Points used only for bisection are removed. Retained points
are shared by observable methods, and completed fits are cached per method.
Adding a method runs only its missing fit and measurement work.

Select beta1 matching methods in `[renormalization]`:

```toml
beta1_methods = ["magsus", "corrG"]
alpha_topo_fit = true  # false skips topo; 1plaq always runs
```

`corrG` matches the second-moment correlation length `xi_G`, computed from the
zero and lowest nonzero Fourier modes. The method name distinguishes it from
a future exponential correlation-length fit.

1plaq is a required dependency. Both the baseline 1plaq combination and every
successful topo combination are exported. `alpha_topo_fit` defaults to true.
When true, topo runs once per selected beta1 method, except for the U alpha=0
subspace convention. When false, topo sampling, fitting and chain tuning are
skipped; the 1plaq alpha is used in the final couplings. Empty beta1 methods
skip obs and topo and leave beta1 null. Explicit `scan.alpha.points` take priority;
otherwise the initial grid is the 1plaq center and its two neighbors at spacing
`max(refine_tolerance, fit_spacing_min)`, clipped to min/max. An out-of-range
center raises an error. Existing expansion and refinement then apply.
A failed topo correction retains the 1plaq combination and records the reason.

```toml
[storage]
save_npz = false
```

This defaults to false. Scalar measurement and completed-fit JSON caches are
always saved and can be reused without NPZ files. Setting true additionally
saves measurement arrays and patch batches for future raw-data reanalysis.
NPZ files contain observables/features, not complete lattice configurations.
Without them, interrupted patch batches or changed plaquette fit initial values
may require resampling. The switch affects future writes, not existing files;
turning it on does not regenerate arrays for already cached measurements.
Temporary files used by the isolated fitting worker are removed after use.


HMC uses a fixed `trajectory_length`; whenever dual averaging changes the
candidate epsilon, the leapfrog count is recomputed and the effective epsilon
is adjusted so the trajectory length remains exact. After adaptation the
kernel is frozen. Post-adaptation warmup continues until the relevant probe
autocorrelation times are stable (or `max_warmup` is reached), and its samples
are discarded before acceptance counters and production statistics are reset.

Production measures every sweep. Observable, topology, and pilot stages specify
`min_meas_total`, `target_ess`, and `max_meas_total`. The 1plaq/2plaq fits instead
specify `fit_times` plus `min_meas_per_boundary`, `target_ess_per_boundary`, and
`max_meas_per_boundary`. `chains` is only the GPU batch width: the requested
boundary count is rounded upward to a complete batch, and every chain receives
its own independently thermalized frozen boundary. Completed patch batches are
saved separately and reused after an interrupted run when `storage.save_npz=true`. Correlation functions are
accumulated online, while only compact mode and topology time series are
retained. The pilot xi_G error is
obtained by propagating the full multi-chain Gamma-method covariance of its
three Fourier modes, including both cross-mode and temporal correlations.
Plaquette observables use one float64 history buffer sized for the maximum
measurement count. ESS checks transform and process one boundary at a time,
and the returned history shares the buffer rather than copying it. The full
configuration history is never retained.

The 1plaq stopping rule checks all three charges (`Q_U`, `Q_z`, `Q_s`). For
each frozen boundary, ESS is computed from integer charge differences relative
to the first sample, after validating quantization within a dtype- and
patch-area-dependent roundoff tolerance. Original charges and fit histograms
are unchanged. A sector with no observed changes uses the constant-series
convention `tau=0.5`, nominal `ESS=number of measurements`; this does not prove
that other sectors have been explored. Per-observable constant counts appear
in progress summaries. The 1plaq JSON sampling section records
`constant_sector` and `sector_transitions`, with axes `[batch, chain, Q_U/Q_z/Q_s]`.
These flags alone do not fail strict diagnostics. The 2plaq rule still checks
both sine and cosine for both U/z angles; observable and topology criteria are
unchanged.

The reference example requests 5000 two-plaquette and 2000 one-plaquette
boundaries. With `chains = 32`, these become 157 batches / 5024 boundaries and
63 batches / 2016 boundaries respectively. Reducing `chains` changes batch
width and run time, not the requested statistical coverage.

The default beta1 search starts at `-0.1, 0, 0.1`, expands geometrically in the
direction required by each matching observable within `[-8, 8]`, and then
bisects each raw bracket to the preliminary `refine_tolerance`. Undefined
correlation lengths on the low-beta1 side are treated as censored values for
bracket refinement, never as numerical zero. A preliminary bootstrap match sets
the final uniform-grid spacing to `fit_sigma_multiplier * coupling_error`,
clamped by `fit_spacing_min/max`. If the grid fails the statistical separation
check, `fit_spacing_growth_factor` (default `2.0`, finite and greater than 1)
sets the minimum spacing growth for each retry, capped by `fit_spacing_max`.
Configure it separately in `[scan.beta1]` and `[scan.alpha]`. Existing cached
results remain reusable at the default; new fits use the new growth factor.
Only that odd-sized, error-separated grid is
used by the final inverse-variance-weighted monotonic polynomial. Explicit scan
points select the initial samples; automatic expansion, localization, and final
grid construction still follow.

The top-level `summary.json` contains the input physical parameters, requested
methods, actual geometry and final branch estimates. Each branch lists complete
method combinations under `results`: `couplings` groups `beta`, `beta1`, `alpha`,
independent `alpha1`; `errors` groups the directly fitted uncertainties.
Beta/alpha1 errors are conditional on the fixed alpha from the corresponding
1plaq or topo stage. Alpha errors belong
to their own 1plaq or topo stage; there is no whole-pipeline bootstrap.
`results` always retains the baseline 1plaq combination for each beta1 method,
and adds successful topo combinations. A failed topo keeps the baseline result.
Only topo combinations include
`topo_correction`, with the signed alpha/beta/alpha1 changes; stage files own the
MSE diagnostics and full refit records. Next-step configurations are generated from
`results` using the script below; `renorm_as_fine` is no longer emitted.
`summary.json` is the sole final result file; branches do
not duplicate it. `resolved.json` owns the effective configuration, actual chain
widths and shared simulation contexts. `run_state.json` owns command/status and
configuration identity. Each ensemble JSON owns its scalar measurements;
stage JSON files reference those ensembles and keep fit/search diagnostics,
including cached methods that are no longer selected. Only selected methods
appear in the final summary.

### Generate the next renormalization configuration

Edit the parameter block at the top of `configs/next_config.py`, then run:

```bash
python configs/next_config.py
```

Install the project dependencies first (`python -m pip install -e ".[test]"`);
the generator uses `tomli-w` to write TOML. The script runs directly from the
source checkout and resolves paths independently of the current working directory.
`SOURCE` is a unique run directory name searched recursively under the project's
`runs/`, an absolute run directory, or a directory path relative to the project root.
Ambiguous names require an explicit path. `OUTPUT_DIR` is relative to `configs/`
unless absolute; parent directories are created. `OUTPUT_FILENAME` is optional:
leave it empty or set it to `None` to use
`<source-run-name>-L<factor>_mod<mod>_<beta_method>_<alpha_method>.toml`.
For the supplied parameters this is
`N2_L2_b1.0-L2_mod0_corrG_topo.toml`. Custom names must not contain a
directory path; the `.toml` extension is added if omitted. Existing files require
`OVERWRITE = True`.

Set `MOD`, `TYPE`, `BETA_METHOD`, `ALPHA_METHOD`, and `FACTOR` explicitly.
For example, the supplied parameters select `z_mod0 / corrG_obs_fit_topo` from
`N2_L2_b1.0`. N and all four final couplings come from `summary.json`, including
the refitted beta/alpha1 for topo. The next configuration uses `model.mod = [MOD]`,
`beta1_methods = [BETA_METHOD]`, and the specified blocking `FACTOR`.
`ALPHA_METHOD = "topo"` enables the next topo fit; `"1plaq"` disables it.
`TYPE` selects only the source branch. `NEXT_TYPE` controls the next run and
defaults to `"U"`.

Other settings inherit `resolved.json.config`, falling back to `input.toml` and
configuration defaults when the saved config is absent. Measured geometries and
branch tuning records are not copied. `OVERRIDES` accepts existing dotted field
names and Python values, for example `{"runtime.device": "cpu"}`; model fields
and fields controlled by dedicated parameters cannot be overridden there.
The generated configuration is validated before writing. Missing results or
invalid couplings raise errors without falling back to other methods.
Generation does not start a simulation; run the resulting TOML normally, and
use that run as `SOURCE` for another step. Older summaries containing
`renorm_as_fine` remain readable; the generator ignores that field.

To migrate existing active runs, inspect then apply:

```bash
python -m cpn_renorm.migrate runs
python -m cpn_renorm.migrate runs --apply
```

Migration validates a temporary copy before publishing, preserves final values
and uncertainties, recovers discarded scalar points from old stage JSON, and
then removes NPZ files and duplicate branch summaries. Historical search phases
are reconstructed by replaying the old search against stored scalar values and
checking the recorded expansion/refinement counts. Only coarse search points and
points actually used in the final fits are retained; bisection and discarded fit
grid points are removed. Incomplete traces stop migration before any source data
is deleted. It never recreates lost raw arrays
or reruns simulations, and excludes `runs/old`. Repeating migration is safe.


For a server job, bind one process to one GPU and let the process batch its
independent chains:

```bash
CUDA_VISIBLE_DEVICES=0 python -m cpn_renorm run configs/case.toml --device cuda:0
```

The pilot increases its PBC lattice until
`L / (xi_G + 2*xi_G_err) >= min_L_over_xi_G`. After a failed check, it chooses
`L = min(max_L, ceil(pilot_growth_margin * min_L_over_xi_G * (xi_G + 2*xi_G_err)))`.
`pilot.pilot_growth_margin` defaults to `1.25` and must be finite and at least 1;
setting it to 1 removes growth headroom without changing the pass condition.
`pilot.max_L` defaults to 128; reaching it without passing saves a failure record
and stops the run. Replace the obsolete `pilot.growth_factor` setting with
`pilot.pilot_growth_margin` when updating old configurations.
After passing, it sets
`padding=ceil(padding_xi_G_mul*xi_G)` and rounds `L_fine` upward to a multiple of the
renormalization factor, with `L_coarse >= 2`. Setting `geometry.padding` or
`geometry.coarse_L` to a valid value overrides the corresponding derived size;
`-1` selects automatic derivation. Automatic values can be clamped with
`padding_min/max` and `coarse_L_min/max`; a missing bound or `-1` means
unbounded. Bounds do not constrain an explicit fixed value. PBC intentionally
supports `padding = 0`: opposite sides are periodically identified, leaving two
independent frozen boundaries. Scan ensembles are content-addressed and reused
when an initial beta1/alpha interval must be expanded or the configured scan is
edited.

Recovery is stage-granular: a completed fit batch or scan point is reusable,
but an interrupted in-progress HMC batch/point restarts from the beginning.
Legacy `runs/<name>-<fingerprint>/` directories are left untouched and are not
automatically migrated into the unified workspace.

CPU SciPy fitting and plotting run in a short isolated subprocess. This avoids
OpenMP-runtime collisions with CUDA PyTorch on Windows and has negligible cost
relative to Monte Carlo generation.

The default is float64/complex128. Float32 is available for experiments, but a
production physics run should validate acceptance, Hamiltonian drift and fitted
couplings before using it.

## Package layout

The implementation is directly under `cpn_renorm/`, so running
`python -m cpn_renorm` from the repository root does not require installation
or a path shim. `pyproject.toml` still supports normal packaging if needed.


## Fitting devices and cache compatibility

`runtime.device` controls sampling and plaquette fitting together. CPU uses an
independent SciPy implementation; CUDA uses a local Torch float64 model,
analytic Jacobian, bounded damped Gauss–Newton optimizer and covariance algebra.
No neighboring test project is imported at runtime. Explicit unavailable CUDA or
out-of-memory errors stop the run, without falling back or reducing budgets.
Fixed Villain weights are reused; large normalization grids stream bounded blocks,
with host-backed cached weights when needed. Normalization starts at 256 nodes,
doubles until relative prediction change <1e-8, and fails at 16384 if unresolved.
Torch and SciPy run in separate isolated workers; scans, intersections and plots
remain CPU work. Idle parent CUDA memory is released before a CUDA fit worker.
Temporary NPZ inputs are deleted when the worker exits. Fit JSON records backend,
device, convergence, local errors, elapsed worker time (including startup and
transfer) and peak allocated CUDA memory; tqdm shows stages/iterations.
Each plaquette stage also writes `fit.log`, including Torch per-iteration MSE,
damping and projected-gradient diagnostics. After topo matching, the pipeline
checks the MSE with only alpha changed, then refits beta/alpha1 at the new alpha
with the same normalization convergence checks. Each successful method's refit
and its cache identity are saved in `step4_topo/result.json` independently of the
observable/topo scan cache. If raw data must be regenerated on resume, topo records
the data source and evaluates the initial, alpha-only and refitted MSE on that data.
Compatible persisted 2plaq data carry a sampling identity; untagged or stale files
are regenerated when needed. With `save_npz=false`, fresh 2plaq data stay in memory
until all related branches finish, and an existing aggregate NPZ is deleted only
after the run succeeds. No new persistent raw cache is created. A resumed or
expanded run can therefore require sampling again if a refit is missing and its
raw data were deleted. With `save_npz=true`, raw NPZ files remain available.

Stage directories are `step1_1plaq`, `step2_2plaq`, `step3_observable`,
`step4_topo`. Root `summary.json` and `resolved.json` are updated directly,
without automatic archive copies. Existing runs are not batch-migrated. The legacy JSON
migration utility only serializes historical results and cannot make folded
histograms compatible with the new full-action pipeline. New action/fitting/scan
identities prevent old folded 2plaq and obs/topo reuse; compatible pilot and raw
1plaq data may be reused. Missing raw data are resampled.

No bin counts were changed for this pipeline. Fits still target bin centers,
without bin integration or an automatic second bin-resolution stage. Run the
opt-in verification with `OMP_NUM_THREADS=1 python tests/cuda_smoke.py
/tmp/cpn-cuda-report.json` in an environment with visible CUDA. It exercises real
plaquette sampling, isolated CPU/GPU fits and a paired B/2B comparison on the
same synthetic angle samples. See [verification](docs/verification.md) for the
measured limitations; production bins are not claimed converged.
