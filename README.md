# Torch/CUDA CPN renormalization pipeline

This repository contains a new batched PyTorch implementation of the refined
four-coupling CP(N-1) renormalization pipeline. `legacy/` remains the numerical
reference and is not imported by the new package.

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
most 80% of CUDA memory and an estimated 8 GiB of host measurement memory.
Adjust those limits with `--gpu-memory-fraction`, `--max-host-memory-gb`, and
`--max-chains`.

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
the U one-plaquette/topology result is explicitly unavailable because the fine
integer `s` sector is unconstrained; the U two-plaquette and observable results
remain available.

Fitted beta and effective alpha retain full floating-point precision in
subsequent scans and exported TOML. The final summary includes direct fit
errors; these do not include propagation of upstream uncertainties.

The workspace records the last accepted effective configuration, including
defaults and command-line device/seed overrides. If the configuration changes,
the CLI prints every changed value and marks each pipeline stage as `reuse`,
`reprocess`, or `resimulate`. An interactive terminal asks for confirmation;
scheduled/non-interactive jobs must pass `--accept-changes`. The `batch`
command performs this check for all matched files before starting simulations.

It is therefore safe to run first with `alpha_methods = ["1plaq"]`, adjust
`scan.beta1.points`, `min`, or `max`, and run again. Existing retained beta1
ensembles are reused, only missing points are simulated, and the latest scan
result and summary replace the old ones. Adding `"topo"` to `alpha_methods`
then fills in the topology stage for the accepted beta1 fit. After a successful pipeline run,
initial/coarse-expansion points and points actually used in the final fits are retained as JSON
measurement caches. Points used only for bisection are removed. Retained points
are shared by observable methods, and completed fits are cached per method.
Adding a method runs only its missing fit and measurement work.

Select methods independently in `[renormalization]`:

```toml
beta1_methods = ["magsus", "corrG"]
alpha_methods = ["1plaq", "topo"]
```

`corrG` matches the second-moment correlation length `xi_G`, computed from the
zero and lowest nonzero Fourier modes. The method name distinguishes it from
a future exponential correlation-length fit.

Empty lists skip those fits and produce `null` for their couplings in the final summary. `topo`
requires at least one beta1 method and runs once for each selected beta1 method.
It does not require `1plaq`: without explicit `scan.alpha.points`, its initial
points are `min`, `min+h`, `min+2h`, where `h=min(0.1, (max-min)/2)`.
Failed/unavailable selected fits report a null value; reasons remain in the stage JSON.

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
`alpha1=alpha_eff-alpha` and `alpha_eff`; `errors` groups the directly fitted
uncertainties. `renorm_as_fine` exports each complete combination as four TOML
assignments for the next renormalization step. It is the sole final result file; branches do
not duplicate it. `resolved.json` owns the effective configuration, actual chain
widths and shared simulation contexts. `run_state.json` owns command/status and
configuration identity. Each ensemble JSON owns its scalar measurements;
stage JSON files reference those ensembles and keep fit/search diagnostics,
including cached methods that are no longer selected. Only selected methods
appear in the final summary.

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
