# pyWebExpo 

## Introduction 

pyWebExpo is a derivative of the WebExpo project for the analysis of Occupational/Industrial Hygiene sampling using bayseian analysis.

The project utilises PyMC for the modelling, and JAX (Numpyro / Blackjax) for computation. It utilises a NUTS sampler, allowing for speedy calculations for more complicated modelling.

The project aims to simplify and speed up Bayseian analysis for occupational hygiene use compared to the base Webexpo project.


## How to Use

For most users, the simplest way to get started is by using the generic SEGInformedVarLognormal model. This model requires only the data (in WebExpo format) and the occupational exposure limit (OEL). The library’s defaults will handle the rest.

> #### Limitations
> Currently, pyWebExpo is only supported on Linux and MacOS. This is due to the limited support for [JAX](http://jax.readthedocs.io/) on  Windows. If you are a Windows user, it is currently recommended to use [Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/) as a proxy.

## Quick Start Example

```python
from pywebexpo.seg_models import SEGInformedVarLognormal

# Example data (values in WebExpo format, e.g., censoring symbols like '<', '>', or ranges)
default_data = ['24.7', '64.1', '13.8', '43.7', '19.9', '133', '32.1', '15', '53.7']

# Initialize the model with the default configuration
model = SEGInformedVarLognormal(data=default_data, oel=100)

# Build the model, run the sampler, and analyze the chains
model.fit()
analysis_results = model.analyse_chains()
divergence_check = 

print("Analysis Results:")
print(analysis_results)
```

This example builds and fits the model using default settings, then performs a posterior chain analysis to summarize the results.

## Advanced Use

For users who need more control, pyWebExpo provides an abstract base class, WebExpoModel, which underpins all models in the library. This design allows you to fine-tune various aspects of the modeling process, including custom priors, sampler configurations, and post-sampling analysis.

### Custom Settings

#### Custom Analysis

#### Custom Priors

You can override the default prior distributions by supplying a custom model configuration. For example:
```python
custom_config = {
    "mu_lower": -20.0,
    "mu_upper": 20,
    "log_sigma_mu": -0.1744,
    "log_sigma_prec": 2.5523,
    "error_lower": 0.15,
    "error_upper": 0.45
}

model = SEGInformedVarLognormal(data=default_data, oel=100, model_config=custom_config)
model.fit()
analysis_results = model.analyse_chains()

print("Custom Prior Analysis Results:")
print(analysis_results)
```

This allows you to tailor the prior distributions and other model parameters to better reflect your knowledge or assumptions about the exposure scenario.

#### Custom Sampler Settings

### InferenceData Analysis

After sampling, pyWebExpo stores the results in an ArviZ InferenceData object. This object contains detailed traces of all model parameters, which you can analyze or visualize using ArviZ’s suite of tools. For example:

```python
import arviz as az

# Fit the model
model.fit()

# Access the InferenceData object
idata = model.idata

# Generate summary statistics and trace plots
print(az.summary(idata))
az.plot_trace(idata)
```

This integration makes it easy to perform comprehensive posterior analyses, generate diagnostic plots, and assess model convergence.

### Model ID

Every model instance in pyWebExpo is assigned a unique identifier (id). This identifier is based on the model's version and type and hyperparameter settings. 

This can ensure that results are comparable across revisions. 

```python
print("Model ID:", model.id)
```

### Experimental

#### Custom Modelling

The pyWebExpo utilises an abstract base class for the bayseian modelling, utilising PyMC. This provides you with the tooling required for building and running your own modelling. 

```python

from pywebexpo.model import WebExpoModel

class CustomModel(WebExpoModel): ...
    """
    Whatever custom model you require
    """
...

model = CustomModel()

```

### To Do:
