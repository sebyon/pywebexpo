# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**pyWebExpo** is a Python library for Bayesian analysis of occupational/industrial hygiene exposure data. It is a modernized derivative of the WebExpo project, using PyMC for probabilistic modeling and JAX/Numpyro/Blackjax for efficient NUTS sampling.

**Platform note**: JAX has limited Windows support. Linux/macOS or WSL is recommended for full functionality.

## Platform note

JAX's conda-forge packages are Linux/macOS only (`jaxlib` has no win-64 builds). On Windows, use the provided devcontainer (`.devcontainer/`) which runs a Linux environment where the full pixi stack resolves correctly.

## Package Manager

This project uses **pixi** (not pip/conda/poetry). All dependency and environment management goes through `pixi.toml`.

```sh
pixi install          # install dependencies
pixi run python ...   # run Python in the pixi environment
```

## Running Tests

```sh
pixi run pytest tests/                    # run all tests
pixi run pytest tests/informedvar_test.py # run a single test file
pixi run pytest tests/ -k "test_name"     # run a specific test by name
```

Tests are unittest-based and live in `tests/`. The only current test file is `informedvar_test.py`.

## Architecture

The model lifecycle follows a build → sample → analyse pipeline:

1. **`src/pywebexpo/model.py`** — `WebExpoModel` abstract base class. Defines the interface all models implement: `build_model()`, `sample_model()`, `fit()` (calls both), and `analyse_chains()`. Configuration is externalized through `get_default_model_config()`, `get_default_sampler_config()`, and `get_default_analysis_config()` static methods.

2. **`src/pywebexpo/datapreperation.py`** — `WebExpoData` dataclass and `define_observed_lower_upper()`. Parses WebExpo-format strings with censoring (`'<value'` = left-censored, `'>value'` = right-censored, `'lower-upper'` = interval-censored) and normalizes values by OEL.

3. **`src/pywebexpo/seg_models/informedvar.py`** — `SEGInformedVarLognormal`, the primary concrete model. Implements a lognormal SEG (Simple Exposure Group) model with optional measurement error (`None`, `"CV"`, or `"SD"` mode). This is the main user-facing class.

4. **`src/pywebexpo/interpretation.py`** — `PointEstimates`. Takes an ArviZ `InferenceData` object from a fitted model and computes posterior statistics: geometric mean/SD, exceedance fraction, arithmetic mean, percentiles, and risk band distributions. Uses `@cached_property` for expensive computations.

5. **`src/pywebexpo/utils.py`** — Validation helpers (`check_lognormal_data`, `check_normal_data`) and `inspect_divergent_traces()` for MCMC diagnostics.

### Key design patterns

- **Config dicts** decouple hyperparameters (priors, sampler settings, analysis thresholds) from model logic. Override them by passing dicts to the constructor.
- **`fit()`** is the main entry point: it calls `build_model()` then `sample_model()`, storing results in `self.idata` (ArviZ `InferenceData`).
- Worker models (`src/pywebexpo/worker_models/`) are stubs — not yet implemented.

### Typical usage

```python
from pywebexpo.seg_models import SEGInformedVarLognormal

model = SEGInformedVarLognormal(data=['24.7', '64.1', '<13.8', '>43.7', '19.9-25'], oel=100)
model.fit()
results = model.analyse_chains()  # returns dict: gm, gsd, frac, perc, am, risks, ...
idata = model.idata               # ArviZ InferenceData for further diagnostics
```
