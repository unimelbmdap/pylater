---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.2
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

+++ {"editable": true, "slideshow": {"slide_type": ""}}

# Overview

```{code-cell} ipython3
%matplotlib inline
import matplotlib.pyplot as plt
```

```{code-cell} ipython3
import numpy as np

import pymc as pm
import arviz as az

import pylater
import pylater.data
```

## Assemble the data

```{code-cell} ipython3
dataset = pylater.data.cw1995["b_p50"]
```

```{code-cell} ipython3
dataset
```

## Visualise the raw data

```{code-cell} ipython3
plot = pylater.plot.ReciprobitPlot()
plot.plot_data(data=dataset);
```

## Estimate a LATER model

```{code-cell} ipython3
model = pylater.build_default_model(datasets=[dataset])
```

```{code-cell} ipython3
model.to_graphviz()
```

### Priors

```{code-cell} ipython3
with model:
    prior_predictive = pm.sample_prior_predictive()
```

```{code-cell} ipython3
plot = pylater.plot.ReciprobitPlot()
plot.plot_predictive(
    idata=prior_predictive,
    predictive_type="prior",
    observed_variable_name="obs_b_p50",
);
```

### Sampling

```{code-cell} ipython3
with model:
    idata = pm.sample()
```

```{code-cell} ipython3
az.stats.summary(idata)
```

```{code-cell} ipython3
with model:
    idata = pm.sample_posterior_predictive(trace=idata, extend_inferencedata=True)
```

```{code-cell} ipython3
plot = pylater.plot.ReciprobitPlot()
plot.plot_predictive(
    idata=idata,
    predictive_type="posterior",
    observed_variable_name="obs_b_p50",
);
plot.plot_data(data=dataset, color="k");
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
%load_ext watermark
%watermark -n -u -v -iv -p pytensor,xarray,pymc
```

```{code-cell} ipython3

```
