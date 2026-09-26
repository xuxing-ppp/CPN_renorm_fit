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
python -m cpn_renorm run configs/example.toml --output-dir runs --skip-topo
```

Tune the number of batched chains for the actual pilot-derived geometry before
a production run:

```bash
python -m cpn_renorm tune-chains configs/example.toml --output-dir runs
```

This runs or reuses the pilot, benchmarks the pilot, two-plaquette, observable,
one-plaquette and topology workloads separately, and writes
`runs/example/chain_tuning.json`. It prints a TOML snippet but never modifies
the input configuration. The recommendation is the smallest safe chain count
within 90% of the best measured throughput; by default a candidate may use at
most 80% of CUDA memory and an estimated 4 GiB of host measurement memory.
Adjust those limits with `--gpu-memory-fraction`, `--max-host-memory-gb`, and
`--max-chains`.

Re-running the same resolved configuration reuses the pilot, completed fit
steps, and content-addressed ensembles. Pass `--force` to regenerate them.

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
saved separately and reused after an interrupted run. Correlation functions are
accumulated online, while only compact mode and topology time series are
retained. The pilot xi error is
obtained by propagating the full multi-chain Gamma-method covariance of its
three Fourier modes, including both cross-mode and temporal correlations.

The reference example requests 5000 two-plaquette and 2000 one-plaquette
boundaries. With `chains = 32`, these become 157 batches / 5024 boundaries and
63 batches / 2016 boundaries respectively. Reducing `chains` changes batch
width and run time, not the requested statistical coverage.

The default beta1 search starts at `-0.1, 0, 0.1`, expands geometrically in the
direction required by each matching observable within `[-8, 8]`, and adds a
midpoint to each raw bracket before the final error-weighted fit. Explicit scan
points and bounds in TOML override those defaults.

For a server job, bind one process to one GPU and let the process batch its
independent chains:

```bash
CUDA_VISIBLE_DEVICES=0 python -m cpn_renorm run configs/case.toml --device cuda:0
```

The pilot doubles its PBC lattice until
`L / (xi + 2*xi_err) >= min_L_over_xi`. It then sets
`padding=ceil(padding_xi_mul*xi)` and rounds `L_fine` upward to a multiple of the
renormalization factor, with `L_coarse >= 2`. Setting `geometry.padding` or
`geometry.coarse_L` to a valid value overrides the corresponding derived size;
`-1` selects automatic derivation. Scan ensembles are content-addressed and reused when an
initial beta1/alpha interval must be expanded.

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
