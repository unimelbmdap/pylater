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
```

```{code-cell} ipython3
import numpy as np

import pymc as pm
import arviz as az

import pylater
```

```{code-cell} ipython3
n_trials = 10_000
```

```{code-cell} ipython3
with pm.Model() as model:

    mu = pm.Lognormal("mu", mu=np.log(1 / 0.2), sigma=np.log(2) / 2)
    sigma = 1
    sigma_e = 1

    obs = pylater.LATER("obs", mu, sigma=sigma, sigma_e=sigma_e, size=n_trials)    
```

```{code-cell} ipython3
with model:
    prior_predictive = pm.sample_prior_predictive()
```

```{code-cell} ipython3
pm.CustomDist?
```

```{code-cell} ipython3
pm.Normal?
```

# TEST

```{code-cell} ipython3
%load_ext watermark
%watermark -n -u -v -iv -w -p pytensor,xarray
```

```{code-cell} ipython3

```
