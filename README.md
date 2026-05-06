# pyWebExpo

**pyWebExpo** is a Python library for Bayesian analysis of occupational and industrial hygiene exposure data. It is a modernised derivative of the [WebExpo](https://webexpo.net/) project, built on [PyMC](https://www.pymc.io/) for probabilistic modelling and [JAX](https://jax.readthedocs.io/) (via Numpyro / Blackjax) for fast NUTS sampling.

---

## Contents

- [pyWebExpo](#pywebexpo)
  - [Contents](#contents)
  - [Platform Support](#platform-support)
    - [Windows Devcontainer](#windows-devcontainer)
  - [Installation](#installation)
  - [Data Format](#data-format)
  - [Quick Start](#quick-start)
  - [Usage](#usage)
    - [Error Modes](#error-modes)
    - [Custom Configuration](#custom-configuration)
      - [Model Priors](#model-priors)
      - [Sampler Settings](#sampler-settings)
      - [Analysis Settings](#analysis-settings)
    - [Divergence Diagnostics](#divergence-diagnostics)
    - [ArviZ Integration](#arviz-integration)
    - [Custom Models](#custom-models)
  - [Analysis Output Reference](#analysis-output-reference)

---

## Platform Support

| Platform | Supported | Notes |
|---|---|---|
| Linux | ✅ | Full support via conda-forge |
| macOS | ✅ | Full support via conda-forge |
| Windows | ⚠️ | Use the provided [devcontainer](#windows-devcontainer) |

JAX's conda-forge packages do not include Windows binaries (`jaxlib`). Native Windows support is not available at this time.

### Windows Devcontainer

A devcontainer configuration is included for Windows users. It runs a Linux environment inside Docker where the full conda-forge stack resolves correctly.

**Requirements:** [Docker Desktop](https://www.docker.com/products/docker-desktop/) and the [VS Code Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers).

Open the repository in VS Code and select **Reopen in Container** when prompted (or run `Dev Containers: Reopen in Container` from the command palette).

---

## Installation

pyWebExpo uses [pixi](https://pixi.sh/) for environment and dependency management.

```sh
# Install pixi (if not already installed)
curl -fsSL https://pixi.sh/install.sh | bash

# Install the environment
pixi install
```

---

## Data Format

pyWebExpo uses the WebExpo data format, where each measurement is a string that encodes its censoring status.

| Format | Type | Example |
|---|---|---|
| `"42.3"` | Uncensored | A standard measurement |
| `"<13.8"` | Left-censored | Value is below the detection limit |
| `">200"` | Right-censored | Value exceeds the upper measurement range |
| `"19.9-25.0"` | Interval-censored | Value falls within a known range |

All values are normalised internally by the occupational exposure limit (OEL) before modelling.

---

## Quick Start

```python
from pywebexpo import SEGInformedVarLognormal

data = ['24.7', '64.1', '<13.8', '>43.7', '19.9-25.0', '133', '32.1', '15', '53.7']

model = SEGInformedVarLognormal(data=data, oel=100)
model.fit()

results = model.analyse_chains()
print(results)
```

**Example output:**

```python
{
    'gm':          {'est': 38.4,  'lcl': 24.1,  'ucl': 58.7},
    'gsd':         {'est': 2.81,  'lcl': 1.94,  'ucl': 4.23},
    'frac':        {'est': 12.3,  'lcl': 3.1,   'ucl': 31.6},
    'perc':        {'est': 148.2, 'lcl': 87.4,  'ucl': 289.5},
    'am':          {'est': 56.1,  'lcl': 33.8,  'ucl': 96.4},
    'frac.risk':   62.4,
    'perc.risk':   58.9,
    'am.risk':     18.3,
    'am.riskbands': {
        '<0.01*OEL':     0.0,
        '[0.01-0.1]*OEL': 4.2,
        '[0.1-0.5]*OEL': 41.3,
        '[0.5-1]*OEL':   36.2,
        '>OEL':          18.3
    }
}
```

---

## Usage

### Error Modes

`SEGInformedVarLognormal` supports three measurement error modes:

| Mode | Argument | Description |
|---|---|---|
| None (default) | `error_mode=None` | No measurement error modelled |
| Coefficient of Variation | `error_mode="CV"` | Error modelled as a fraction of the true value |
| Standard Deviation | `error_mode="SD"` | Error modelled as a fixed standard deviation |

```python
model = SEGInformedVarLognormal(data=data, oel=100, error_mode="CV")
model.fit()
```

---

### Custom Configuration

Each aspect of the model is controlled by a configuration dictionary. Pass any of these to override the defaults.

#### Model Priors

Controls the prior distributions on `mu` (log-scale mean) and `log_sigma` (log-scale standard deviation).

```python
model_config = {
    "mu_lower":       -20.0,
    "mu_upper":        20.0,
    "log_sigma_mu":    -0.1744,
    "log_sigma_prec":   2.5523,
    "error_lower":      0.15,   # lower bound for CV or SD error prior
    "error_upper":      0.45,   # upper bound for CV or SD error prior
}

model = SEGInformedVarLognormal(data=data, oel=100, model_config=model_config)
```

#### Sampler Settings

Controls the NUTS sampler behaviour.

```python
sampler_config = {
    "draws":         5000,
    "tune":          4000,
    "chains":           6,
    "target_accept":  0.85,
    "nuts_sampler": "numpyro",
}

model = SEGInformedVarLognormal(data=data, oel=100, sample_config=sampler_config)
```

#### Analysis Settings

Controls credible interval width, the exceedance fraction threshold, and the target percentile.

```python
analysis_config = {
    "probacred":      90,   # credible interval width (%)
    "frac_threshold": 10,   # exceedance fraction threshold (%)
    "target_perc":    95,   # percentile of interest
}

model = SEGInformedVarLognormal(data=data, oel=100, analysis_config=analysis_config)
model.fit()
results = model.analyse_chains()
```

---

### Divergence Diagnostics

After fitting, use `inspect_divergent_traces` to check for NUTS divergences, which can indicate a poorly specified model.

```python
from pywebexpo.utils import inspect_divergent_traces

model.fit()
report = inspect_divergent_traces(model)
print(report)
# {'message': 'No divergent transitions detected.', 'divergent_transitions': 0}
```

The function returns a `message` at one of three levels — acceptable, warning, or critical — along with the raw divergence count.

---

### ArviZ Integration

The full posterior is stored in `model.idata` as an [ArviZ](https://python.arviz.org/) `InferenceData` object, giving access to the complete PyMC diagnostic and visualisation toolkit.

```python
import arviz as az

model.fit()

# Summary statistics for all parameters
print(az.summary(model.idata))

# Trace plots
az.plot_trace(model.idata)

# Posterior predictive checks, pair plots, etc.
az.plot_pair(model.idata, var_names=["mu", "sigma"])
```

---

### Custom Models

All models inherit from `WebExpoModel`, an abstract base class that defines the build → sample → analyse lifecycle. You can subclass it to implement your own PyMC models while reusing the sampler infrastructure.

```python
from pywebexpo.model import WebExpoModel

class CustomModel(WebExpoModel):
    _model_type = "Custom"
    version = "0.1"

    def build_model(self): ...
    def sample_model(self, **kwargs): ...
    def fit(self, **kwargs): ...
    def analyse_chains(self): ...
    def _extract_chains(self): ...
    def _generate_and_preprocess_model_data(self): ...

    @staticmethod
    def get_default_error_mode(): return None

    @staticmethod
    def get_default_model_config(): return {}

    @staticmethod
    def get_default_sampler_config(): return {}

    @staticmethod
    def get_default_analysis_config(): return {}
```

---

## Analysis Output Reference

`analyse_chains()` returns a dictionary with the following keys. Credible limits (`lcl`, `ucl`) are derived from the `probacred` setting (default 90%).

| Key | Type | Description |
|---|---|---|
| `gm` | `{est, lcl, ucl}` | Geometric mean |
| `gsd` | `{est, lcl, ucl}` | Geometric standard deviation |
| `frac` | `{est, lcl, ucl}` | Exceedance fraction — % of the distribution above the OEL |
| `perc` | `{est, lcl, ucl}` | Target percentile (default: 95th) |
| `am` | `{est, lcl, ucl}` | Arithmetic mean |
| `frac.risk` | `float` | % of posterior samples where exceedance fraction > threshold |
| `perc.risk` | `float` | % of posterior samples where the target percentile exceeds the OEL |
| `am.risk` | `float` | % of posterior samples where the arithmetic mean exceeds the OEL |
| `am.riskbands` | `dict` | Distribution of arithmetic mean across OEL-relative risk bands |
