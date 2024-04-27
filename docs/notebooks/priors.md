---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell} ipython3
%matplotlib inline
import matplotlib.pyplot as plt
```

```{code-cell} ipython3
import numpy as np

import pymc as pm
import arviz as az

import pylater
import pylater.utils
import pylater.data
```

```{code-cell} ipython3
n_trials = 10_000
```

```{code-cell} ipython3
obs = pylater.data.cw1995["b_p95"]
```

```{code-cell} ipython3
with pm.Model() as model:

    mu = pm.Normal("mu", mu=4.5, sigma=2 / 2)
    
    log_sigma = pm.Normal("log_sigma", mu=np.log(1.5), sigma=np.log(2) / 2)

    sigma = pm.Deterministic("sigma", np.exp(log_sigma))

    sigma_e_scale_factor = pm.Normal("sigma_e_scale_factor", mu=3.0, sigma=np.log(2) / 2)

    sigma_e = pm.Deterministic("sigma_e", pm.math.exp(log_sigma + pm.math.log(sigma_e_scale_factor)))
    
    pylater.LATER("obs", mu=mu, sigma=sigma, sigma_e=sigma_e, observed=obs.rt_s)
```

```{code-cell} ipython3
with model:
    prior_predictive = pm.sample_prior_predictive() #samples=5_000)
```

```{code-cell} ipython3
with model:
    trace = pm.sample()
```

```{code-cell} ipython3
az.summary(trace)
```

```{code-cell} ipython3
with model:
    p = pm.sample_posterior_predictive(trace=trace)
```

```{code-cell} ipython3
p.add_groups({"prior": p.posterior_predictive})
```

```{code-cell} ipython3
(f, a) = pylater.utils.plot_prior_predictive(idata=p);
a.scatter(obs.ecdf_x, obs.ecdf_p)
```

```{code-cell} ipython3
az.plot_pair(trace)
```

# TEST

```{code-cell} ipython3
%load_ext watermark
%watermark -n -u -v -iv -w -p pytensor,xarray
```

```{code-cell} ipython3

```
