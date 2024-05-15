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

# Model fitting

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Here, we will look at an example of fitting a model to an example dataset.

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
tags: [remove-output]
---
import pymc as pm

import pylater
import pylater.data
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

We will use the 50% condition from participant 'a' in the example data provided in `pylater`:

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
tags: [remove-output]
---
model = pylater.build_default_model(datasets=[dataset])
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

We can then fit the model by using the `sample` function from PyMC:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [remove-output]
---
with model:
    idata = pm.sample(chains=4)
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

We can then use the posterior summary methods from PyMC to evaluate the posteriors:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
pm.stats.summary(data=idata)
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

We can futher evaluate the model fit by examining the distribution of draws from the fitted model - the posterior predictive distribution (better described as the posterior *retrodictive* distribution; see ["Towards a principled Bayesian workflow" by Michael Betancourt](https://betanalpha.github.io/assets/case_studies/principled_bayesian_workflow.html#143_Posterior_Retrodiction_Checks)):

+++ {"editable": true, "slideshow": {"slide_type": ""}}

First, we need to use PyMC to sample the posterior predictives:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [remove-output]
---
with model:
    idata = pm.sample_posterior_predictive(trace=idata, extend_inferencedata=True)
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

We can then use the `plot_predictive` method of a `ReciprobitPlot` instance to visualise the posterior predictives:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
plot = pylater.ReciprobitPlot()
plot.plot_predictive(idata=idata, predictive_type="posterior");
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

In the above, we can see the 95% credible interval and median of the distribution of posterior predictive samples.

+++ {"editable": true, "slideshow": {"slide_type": ""}}

This becomes particularly useful when compared against the ECDF of the observed data:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
plot = pylater.ReciprobitPlot()
plot.plot_predictive(idata=idata, predictive_type="posterior");
plot.plot_data(data=dataset);
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

We can see a reasonably good correspondance between the observations and the draws from the fitted model.

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
