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

# Model comparison

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Here, we will look at how we can fit multiple datasets simultaneously with some shared parameters and how we can compare models with different sharing arrangements.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [remove-cell]
---
%matplotlib inline
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

First, we will import the necessary packages:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
import matplotlib.pyplot as plt

import pymc as pm
import arviz as az

import pylater
import pylater.data
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## Datasets

+++ {"editable": true, "slideshow": {"slide_type": ""}}

We will use all of the conditions from participant 'b' in the example data provided in `pylater`:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
datasets = [
    dataset
    for dataset in pylater.data.cw1995.values()
    if dataset.name.startswith("b")
]
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## A 'shift' model

+++ {"editable": true, "slideshow": {"slide_type": ""}}

In our first model that we will fit to the data, we will use a 'shift' sharing arrangement: the datasets will have a common standard deviation ($\sigma$) parameter.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [remove-output]
---
shift_model = pylater.build_default_model(datasets=datasets, share_type="shift")
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

We then fit the model:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [remove-output]
---
with shift_model:
    shift_idata = pm.sample()
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Note that there is a lot of data here, so sampling can take a little while - we are using fewer samples than typical in this example to allow for faster execution.

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## A 'swivel' model

+++ {"editable": true, "slideshow": {"slide_type": ""}}

The second model will use a 'swivel' sharing arrangment: the datasets will have a common intercept ($k$) parameter.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [remove-output]
---
swivel_model = pylater.build_default_model(datasets=datasets, share_type="swivel")
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

We then also fit this model:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [remove-output]
---
with swivel_model:
    swivel_idata = pm.sample()
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## Calculating log-likelihoods

+++ {"editable": true, "slideshow": {"slide_type": ""}}

In order to compare the models, we first need to compute their log-likelihoods.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [remove-output]
---
with shift_model:
    shift_idata = pm.compute_log_likelihood(idata=shift_idata)
with swivel_model:
    swivel_idata = pm.compute_log_likelihood(idata=swivel_idata)
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

However, this has the log-likelihoods calculated separately for each dataset.
We can use the helper function `pylater.combine_multiple_likelihoods` to gather them together:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
(shift_idata, swivel_idata) = (
    pylater.combine_multiple_likelihoods(idata=idata)
    for idata in (shift_idata, swivel_idata)
)
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## Comparing models

+++ {"editable": true, "slideshow": {"slide_type": ""}}

We can then use the ArviZ function `az.compare` to do the model comparison:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [remove-output]
---
comparison = az.compare(
    compare_dict={"shift": shift_idata, "swivel": swivel_idata},
    var_name="obs",
)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
comparison
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

See the [documentation for `az.compare`](https://python.arviz.org/en/stable/api/generated/arviz.compare.html) for details on interpreting this output.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [remove-input]
---
%load_ext watermark
%watermark -n -u -v -iv -p matplotlib
```
