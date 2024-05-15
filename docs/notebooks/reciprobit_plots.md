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

# Reciprobit plots

+++

Here, we will examine the functionality in `pylater` for creating figures in 'reciprobit space'.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [remove-cell]
---
%matplotlib inline
```

First, we will import the necessary packages:

```{code-cell} ipython3
import pylater
import pylater.data
```

## Reciprobit plot space

+++

Reciprobit plots are initialised by creating an instance of `pylater.ReciprobitPlot`:

```{code-cell} ipython3
plot = pylater.ReciprobitPlot()
```

As shown above, this style of plot has several noteable features:
* The vertical axis shows probability as a percentage but with a non-linear scale. Between 0.1% and 99.9% (by default), the probability is on a *probit* scale. Outside of these bounds, the probability is on a linear scale (since the probit transform becomes undefined at 0 and 1).
* The lower horizontal axis shows the reaction time ('latency'), again with a non-linear scale. Here, the time is shown with a *negative reciprocal* scale.
* The upper horizontal axis shows the reciprocal of reaction time ('promptness').

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Note: If provided without any arguments, it creates a new figure - you can also use the `fig_ax` parameter to use an already-created figure.

+++

## Plotting observations

+++

To show how reaction time data looks in a reciprobit plot, we can load some example data:

```{code-cell} ipython3
dataset = pylater.data.cw1995["b_p50"]
```

We then use the `plot_data` method of our reciprobit plot variable to add the data to the plot:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
plot = pylater.ReciprobitPlot()
plot.plot_data(data=dataset);
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

This shows an *empirical cumulative distribution function* (ECDF) representation of the observed data using a step plot.

+++ {"editable": true, "slideshow": {"slide_type": ""}}

We can also show the ECDF using a scatter plot:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
plot = pylater.ReciprobitPlot()
plot.plot_data(data=dataset, plot_type="scatter");
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## Customising the plot

+++ {"editable": true, "slideshow": {"slide_type": ""}}

The overall figure and axis properties can be manually adjusted by access the `fig` and `ax` attributes of the `ReciprobitPlot` instance.

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Additionally, customisations can be applied by providing additional keyword arguments to `plot_data`; these are passed directly to the underlying Matplotlib plotting function.
For example, to change the colour of the data plot and give it a different name:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
plot = pylater.ReciprobitPlot()
plot.plot_data(data=dataset, label="Participant B, 50%", c="blue");
```

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
