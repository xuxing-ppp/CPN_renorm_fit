# Repository Guidelines

## Project Structure & Module Organization

The active implementation lives in `cpn_renorm/`. Core modules separate configuration, Torch samplers, observables, fitting, pilot runs, storage, and pipeline orchestration. `configs/example.toml` is the reference command-line configuration. Tests are in `tests/` and mirror the package by concern. The `legacy/` tree contains the original implementation and serves as a numerical and physics reference; avoid changing it unless a task explicitly targets legacy behavior. Generated run data, checkpoints, and large Monte Carlo arrays should remain outside version control.

## Build, Test, and Development Commands

Use Python 3.11 or newer, preferably in the project's PyTorch environment.

```powershell
python -m pip install -e ".[test]"
cpn-renorm validate configs/example.toml
cpn-renorm pilot configs/example.toml
cpn-renorm run configs/example.toml
python -m unittest discover -s tests -v
python -m compileall -q cpn_renorm tests
```

`validate` checks configuration without starting a simulation. `pilot` estimates the fine-lattice correlation length and derived padding/observable sizes. `run` executes the complete fitting pipeline; use `--skip-topo` when appropriate. Run CUDA work in an environment where `torch.cuda.is_available()` is true.

## Coding Style & Naming Conventions

Use four-space indentation, type hints for public interfaces, and concise docstrings for non-obvious numerical logic. Name modules, functions, and variables with `snake_case`, classes with `PascalCase`, and constants with `UPPER_CASE`. Prefer batched Torch tensor operations over Python loops, preserve explicit device and dtype handling, and document tensor shapes where ambiguity is likely. No formatter or linter is currently enforced; keep changes consistent with surrounding code.

## Testing Guidelines

Tests use standard `unittest` discovery and files named `test_*.py`. Add focused tests for action formulas, analytic forces versus autograd, boundary conditions, topology, configuration parsing, and fitting behavior. CPU tests should be deterministic through fixed seeds. Changes affecting GPU kernels or sampling should also include a CUDA smoke run and report acceptance and numerical diagnostics. There is no fixed coverage threshold; prioritize regression-sensitive physics calculations.

## Commit & Pull Request Guidelines

The repository history is minimal (`initialize: legacy`), so no established convention exists. Use short imperative subjects, optionally scoped, such as `sampler: fix phase wrapping`. Pull requests should explain physics or numerical changes, identify the TOML configuration used, include test output, and state device, dtype, acceptance, and performance results when relevant. Link related issues; attach plots only when they materially demonstrate behavior.
