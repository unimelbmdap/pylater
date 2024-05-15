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

# Prior distributions

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Performing Bayesian analysis requires careful consideration of the prior distributions for the model parameters.
Here, we will consider how priors are treated in `pylater` and look at a method for visualising the influence of the priors.

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
import pymc as pm

import pylater
import pylater.data
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## Default priors

+++ {"editable": true, "slideshow": {"slide_type": ""}}

`pylater` uses the PyMC package for describing LATER models and is intended for use by practitioners familiar with PyMC (or willing to become familiar with PyMC).
However, `pylater` also provides the function `build_default_model` that assembles a PyMC model for given data using a default set of priors.
This function is intended to provide an example of how LATER model priors can be specified, and should not be used without assessing the priors that it uses.

+++ {"editable": true, "slideshow": {"slide_type": ""}}

We will examine the usage of `build_default_model` using some example data:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
dataset = pylater.data.cw1995["b_p50"]
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

We can then build the model, using the default priors, via:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
model = pylater.build_default_model(datasets=[dataset])
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## Plotting the effect of priors

+++ {"editable": true, "slideshow": {"slide_type": ""}}

We can visualise the prior distributions for parameters using the [ArviZ](https://python.arviz.org/en/stable/) package.
However, it is particularly useful to look at the distribution of reaction times that is produced by the priors - the prior predictive distribution.

+++ {"editable": true, "slideshow": {"slide_type": ""}}

First, we need to sample the prior predictives using PyMC:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
with model:
    idata = pm.sample_prior_predictive()
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

We then use the `plot_predictive` method of a reciprobit plot variable:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
plot = pylater.ReciprobitPlot()
plot.plot_predictive(idata=idata, predictive_type="prior");
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

This shows the 95% credible range and the median of the ECDFs that would be expected to be encountered given our (default) priors.

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
